#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tac3D tactile sensor implementation for LeRobot.

This module provides the Tac3DTactileSensor class for interfacing with Tac3D
high-resolution tactile sensors through the Tac3D-SDK.
"""

import logging
import threading
import time
from queue import Empty, Queue
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.tactile.configs import Tac3DSensorConfig, TactileDataType
from lerobot.tactile.tactile_sensor import TactileSensor

# Try to import Tac3D SDK
try:
    from PyTac3D import Sensor as Tac3DSensor
    TAC3D_AVAILABLE = True
except ImportError as e:
    TAC3D_AVAILABLE = False
    TAC3D_IMPORT_ERROR = e


class Tac3DTactileSensor(TactileSensor):
    """Tac3D tactile sensor implementation.

    Tac3D sensors provide high-resolution tactile feedback with a 20x20 sensing array,
    measuring both 3D displacement (in mm) and 3D force distribution (in N).

    The sensor communicates with Tac3D-Desktop software via UDP, receiving data
    frames at 30Hz. Each frame contains data for 400 sensing points.

    Data format:
        - 3D_Displacements: (400, 3) array [dx, dy, dz] in mm
        - 3D_Forces: (400, 3) array [Fx, Fy, Fz] in N
        - Combined: (400, 6) array [dx, dy, dz, Fx, Fy, Fz]

    Attributes:
        config: Tac3DSensorConfig with sensor-specific parameters.
        is_connected: Whether the sensor is currently connected.
        is_tared: Whether tare calibration has been applied.

    Example:
        ```python
        config = Tac3DSensorConfig(udp_port=9988, sensor_sn="TAC3D001")
        sensor = Tac3DTactileSensor(config)

        with sensor:
            sensor.tare()  # Perform zeroing calibration
            data = sensor.read()  # Get tactile data (400, 6)
        ```
    """

    def __init__(self, config: Tac3DSensorConfig):
        """Initialize the Tac3D sensor.

        Args:
            config: Tac3DSensorConfig containing sensor parameters.

        Raises:
            ImportError: If Tac3D SDK is not available.
            ValueError: If configuration is invalid.
        """
        if not TAC3D_AVAILABLE:
            raise ImportError(
                "Tac3D SDK not available. Please install Tac3D-SDK. "
                f"Import error: {TAC3D_IMPORT_ERROR}"
            )

        super().__init__(config)
        self.config: Tac3DSensorConfig = config

        # Tac3D SDK sensor instance
        self._sensor: Tac3DSensor | None = None

        # Frame buffer for async reading
        self._frame_queue: Queue[NDArray[np.float64]] = Queue(maxsize=5)
        self._latest_frame: NDArray[np.float64] | None = None
        self._frame_lock = threading.Lock()

        # Background thread for continuous reading
        self._reader_thread: threading.Thread | None = None
        self._running = False

        # Statistics
        self._frame_count = 0
        self._dropped_frames = 0
        self._start_time: float | None = None

        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def is_connected(self) -> bool:
        """Check if the sensor is currently connected.

        Returns:
            bool: True if sensor is connected and receiving data.
        """
        return self._sensor is not None and self._running

    @staticmethod
    def find_sensors() -> list[dict[str, Any]]:
        """Detect available Tac3D sensors.

        Note: Tac3D sensors require Tac3D-Desktop to be running. This method
        returns an empty list as sensors are managed by Tac3D-Desktop.

        Returns:
            list[dict]: Empty list (sensors detected via Tac3D-Desktop).
        """
        if not TAC3D_AVAILABLE:
            return []

        # Tac3D sensors are managed by Tac3D-Desktop
        # Users need to check Tac3D-Desktop for available sensors
        return []

    def connect(self, warmup: bool = True) -> None:
        """Establish connection to the Tac3D sensor.

        Creates a UDP server to receive data from Tac3D-Desktop.

        Args:
            warmup: If True, waits for the first frame before returning.

        Raises:
            ConnectionError: If connection to sensor fails.
            TimeoutError: If warmup frame is not received in time.
        """
        if self.is_connected:
            self._logger.warning("Sensor already connected")
            return

        try:
            self._logger.info(
                f"Connecting to Tac3D sensor on UDP port {self.config.udp_port}..."
            )

            # Create Tac3D sensor instance
            self._sensor = Tac3DSensor(
                recvCallback=self._on_frame_received,
                port=self.config.udp_port,
                maxQSize=10,
            )

            self._running = True
            self._start_time = time.time()

            # Start background reader thread
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()

            self._logger.info("Tac3D sensor connected successfully")

            if warmup:
                self._logger.info("Waiting for warmup frame...")
                self._wait_for_frame(timeout_ms=self.config.timeout_ms)

            # Apply tare if configured
            if self.config.tare_on_startup:
                self.tare()

        except Exception as e:
            self.disconnect()
            raise ConnectionError(f"Failed to connect to Tac3D sensor: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from the sensor and release resources."""
        self._logger.info("Disconnecting from Tac3D sensor...")

        self._running = False

        # Stop reader thread
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)

        # Close sensor connection
        if self._sensor:
            try:
                self._sensor.close()
            except Exception as e:
                self._logger.warning(f"Error closing sensor: {e}")
            self._sensor = None

        # Clear buffers
        with self._frame_lock:
            self._latest_frame = None
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break

        self._logger.info("Tac3D sensor disconnected")

    def read(self) -> NDArray[np.float64]:
        """Read a single tactile frame synchronously.

        Blocks until a new frame is available.

        Returns:
            np.ndarray: Tactile data with shape (400, data_dim).

        Raises:
            ConnectionError: If sensor is not connected.
            TimeoutError: If frame is not received within timeout.
        """
        if not self.is_connected:
            raise ConnectionError("Sensor is not connected")

        return self._get_frame_blocking(timeout_ms=self.config.timeout_ms)

    def async_read(self, timeout_ms: float = 1000.0) -> NDArray[np.float64]:
        """Read the most recent tactile frame.

        Blocks up to timeout_ms if no new frame is available.

        Args:
            timeout_ms: Maximum wait time in milliseconds.

        Returns:
            np.ndarray: Tactile data with shape (400, data_dim).

        Raises:
            TimeoutError: If no new frame arrives within timeout.
            ConnectionError: If sensor is not connected.
        """
        if not self.is_connected:
            raise ConnectionError("Sensor is not connected")

        return self._get_frame_blocking(timeout_ms=timeout_ms)

    def read_latest(self, max_age_ms: int = 500) -> NDArray[np.float64]:
        """Get the latest frame immediately (non-blocking).

        Args:
            max_age_ms: Maximum acceptable frame age in milliseconds.

        Returns:
            np.ndarray: The latest tactile data frame.

        Raises:
            TimeoutError: If frame is older than max_age_ms.
            RuntimeError: If no frames have been received.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                raise RuntimeError("No frames received yet")

            if self._start_time:
                frame_age_ms = (time.time() - self._start_time) * 1000
                if frame_age_ms > max_age_ms:
                    raise TimeoutError(
                        f"Latest frame is {frame_age_ms:.1f}ms old (max: {max_age_ms}ms)"
                    )

            return self._latest_frame.copy()

    def tare(self, num_samples: int | None = None) -> None:
        """Perform tare calibration.

        Sends a calibration signal to Tac3D-Desktop to zero the current readings.

        Args:
            num_samples: Number of samples for averaging (not used for Tac3D).

        Raises:
            ConnectionError: If sensor is not connected.
        """
        if not self.is_connected:
            raise ConnectionError("Cannot tare: sensor is not connected")

        # For Tac3D, we use the SDK's calibrate function
        if self.config.sensor_sn:
            self._logger.info(f"Sending tare command to sensor {self.config.sensor_sn}")
            self._sensor.calibrate(self.config.sensor_sn)
        else:
            # If no SN specified, collect samples and compute offset locally
            super().tare(num_samples)

    def get_statistics(self) -> dict[str, Any]:
        """Get sensor statistics.

        Returns:
            dict: Dictionary containing frame count, dropped frames, and FPS.
        """
        elapsed = time.time() - self._start_time if self._start_time else 0
        fps = self._frame_count / elapsed if elapsed > 0 else 0

        return {
            "frame_count": self._frame_count,
            "dropped_frames": self._dropped_frames,
            "elapsed_time": elapsed,
            "average_fps": fps,
            "is_connected": self.is_connected,
            "is_tared": self.is_tared,
        }

    def _on_frame_received(self, frame: dict, callback_param: Any) -> None:
        """Callback for Tac3D SDK when a new frame is received.

        Args:
            frame: Raw frame data from Tac3D.
            callback_param: Additional callback parameters.
        """
        try:
            processed = self._process_frame(frame)

            with self._frame_lock:
                self._latest_frame = processed

            # Add to queue for async reading
            if self._frame_queue.full():
                self._dropped_frames += 1
                try:
                    self._frame_queue.get_nowait()
                except Empty:
                    pass

            self._frame_queue.put(processed)
            self._frame_count += 1

        except Exception as e:
            self._logger.error(f"Error processing frame: {e}")

    def _process_frame(self, frame: dict) -> NDArray[np.float64]:
        """Process raw Tac3D frame into numpy array.

        Args:
            frame: Raw frame dictionary from Tac3D SDK.

        Returns:
            np.ndarray: Processed tactile data.

        Raises:
            ValueError: If required data fields are missing.
        """
        # Extract displacement data
        displacements = frame.get("3D_Displacements")
        forces = frame.get("3D_Forces")

        if displacements is None:
            raise ValueError("Frame missing '3D_Displacements' field")

        # Ensure correct shape
        displacements = np.asarray(displacements, dtype=np.float64)
        if displacements.shape != (400, 3):
            raise ValueError(f"Unexpected displacement shape: {displacements.shape}, expected (400, 3)")

        # Build output based on data_type config
        if self.config.data_type == TactileDataType.DISPLACEMENT:
            data = displacements
        elif self.config.data_type == TactileDataType.FORCE:
            if forces is None:
                raise ValueError("Frame missing '3D_Forces' field")
            forces = np.asarray(forces, dtype=np.float64)
            data = forces
        else:  # TactileDataType.FULL
            if forces is None:
                raise ValueError("Frame missing '3D_Forces' field")
            forces = np.asarray(forces, dtype=np.float64)
            data = np.concatenate([displacements, forces], axis=1)

        # Apply tare if active
        data = self._apply_tare(data)

        return data

    def _get_frame_blocking(self, timeout_ms: float) -> NDArray[np.float64]:
        """Get a frame from the queue with timeout.

        Args:
            timeout_ms: Maximum wait time in milliseconds.

        Returns:
            np.ndarray: Tactile data frame.

        Raises:
            TimeoutError: If no frame arrives within timeout.
        """
        try:
            return self._frame_queue.get(timeout=timeout_ms / 1000.0)
        except Empty:
            raise TimeoutError(f"No frame received within {timeout_ms}ms")

    def _wait_for_frame(self, timeout_ms: float) -> None:
        """Wait for the first frame to arrive.

        Args:
            timeout_ms: Maximum wait time in milliseconds.

        Raises:
            TimeoutError: If no frame arrives within timeout.
        """
        start = time.time()
        while self._latest_frame is None:
            if (time.time() - start) * 1000 > timeout_ms:
                raise TimeoutError(f"No frame received within {timeout_ms}ms")
            time.sleep(0.01)

    def _reader_loop(self) -> None:
        """Background thread for continuous reading."""
        while self._running:
            time.sleep(0.001)  # Small sleep to prevent busy-waiting
