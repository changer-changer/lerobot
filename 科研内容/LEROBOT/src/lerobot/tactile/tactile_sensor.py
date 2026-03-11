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

"""Base tactile sensor interface for LeRobot.

This module provides the abstract base class for tactile sensors, following the same
design pattern as cameras in LeRobot. It defines a standard interface for tactile
sensor operations across different backends.
"""

import abc
import logging
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.tactile.configs import TactileSensorConfig


class TactileSensor(abc.ABC):
    """Base class for tactile sensor implementations.

    Defines a standard interface for tactile sensor operations across different backends.
    Subclasses must implement all abstract methods.

    Tactile sensors provide high-resolution contact information as point clouds,
    typically including 3D position, displacement, and force data for each sensing
    element in the sensor array.

    Attributes:
        config: TactileSensorConfig containing sensor parameters.
        fps: Configured frames per second.
        num_points: Number of sensing points in the sensor array.
        data_dim: Dimensionality of data per point (3 for displacement/force, 6 for both).
        is_tared: Whether tare/zeroing has been applied.
    """

    def __init__(self, config: TactileSensorConfig):
        """Initialize the tactile sensor with the given configuration.

        Args:
            config: TactileSensorConfig containing sensor parameters.
        """
        self.config = config
        self.fps: int = config.fps
        self.num_points: int = config.num_points
        self.data_dim: int = config.data_dim
        self.is_tared: bool = False
        self._tare_offset: NDArray[np.float64] | None = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def __enter__(self):
        """Context manager entry. Automatically connects to the sensor."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Context manager exit. Automatically disconnects and releases resources."""
        self.disconnect()

    def __del__(self) -> None:
        """Destructor safety net. Attempts disconnect if not properly cleaned up."""
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:
            pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Check if the sensor is currently connected.

        Returns:
            bool: True if sensor is connected and ready, False otherwise.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def find_sensors() -> list[dict[str, Any]]:
        """Detect available tactile sensors connected to the system.

        Returns:
            list[dict]: List of dictionaries containing information about each
                detected sensor. Each dict should include at least 'id' and 'type'.
        """
        pass

    @abc.abstractmethod
    def connect(self, warmup: bool = True) -> None:
        """Establish connection to the tactile sensor.

        Args:
            warmup: If True (default), captures a warmup frame before returning.
                Useful for sensors that require time to stabilize.
        """
        pass

    @abc.abstractmethod
    def read(self) -> NDArray[np.float64]:
        """Capture and return a single tactile frame synchronously.

        This is a blocking call that waits for the hardware and returns a complete
        tactile data frame. The returned array contains point cloud data with shape
        (num_points, data_dim).

        Returns:
            np.ndarray: Tactile data array with shape (num_points, data_dim).
                For Tac3D sensors with full data type, this is (400, 6) containing
                [dx, dy, dz, Fx, Fy, Fz] for each of the 400 sensing points.

        Raises:
            ConnectionError: If the sensor is not connected.
            TimeoutError: If data cannot be read within the timeout period.
        """
        pass

    @abc.abstractmethod
    def async_read(self, timeout_ms: float = 1000.0) -> NDArray[np.float64]:
        """Return the most recent new tactile frame.

        This method retrieves the latest frame captured by a background thread.
        If a new frame is already available, it returns immediately. Otherwise,
        it blocks up to timeout_ms waiting for a new frame.

        Usage:
            Ideal for control loops where you want fresh data synchronized to
            the sensor's FPS.

        Args:
            timeout_ms: Maximum time to wait for a new frame in milliseconds.
                Defaults to 1000ms (1s).

        Returns:
            np.ndarray: Tactile data array with shape (num_points, data_dim).

        Raises:
            TimeoutError: If no new frame arrives within timeout_ms.
            ConnectionError: If the sensor is not connected.
        """
        pass

    def read_latest(self, max_age_ms: int = 500) -> NDArray[np.float64]:
        """Return the most recent frame immediately (non-blocking).

        This method returns whatever is currently in the buffer without waiting.
        The frame may be stale (captured some time ago).

        Usage:
            Ideal for scenarios requiring zero latency or decoupled frequencies,
            such as visualization or non-critical monitoring.

        Args:
            max_age_ms: Maximum acceptable age of the frame in milliseconds.
                Raises TimeoutError if the latest frame is older than this.

        Returns:
            np.ndarray: The latest tactile data frame.

        Raises:
            TimeoutError: If the latest frame is older than max_age_ms.
            NotConnectedError: If the sensor is not connected.
            RuntimeError: If no frames have been captured yet.
        """
        warnings.warn(
            f"{self.__class__.__name__}.read_latest() is not implemented. "
            "Please override read_latest(); it will be required in future releases.",
            FutureWarning,
            stacklevel=2,
        )
        return self.async_read()

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the sensor and release all resources."""
        pass

    def tare(self, num_samples: int | None = None) -> None:
        """Perform tare/zeroing calibration.

        Captures the current sensor readings as the zero reference and subtracts
        these values from subsequent readings. This compensates for temperature
        drift and other baseline offsets.

        Args:
            num_samples: Number of samples to average for tare calculation.
                If None, uses config.tare_samples.

        Raises:
            ConnectionError: If the sensor is not connected.
            RuntimeError: If tare calculation fails.
        """
        if not self.is_connected:
            raise ConnectionError("Cannot tare: sensor is not connected")

        num_samples = num_samples or self.config.tare_samples
        self._logger.info(f"Performing tare with {num_samples} samples...")

        samples = []
        for i in range(num_samples):
            try:
                frame = self.read()
                samples.append(frame)
            except Exception as e:
                raise RuntimeError(f"Failed to capture tare sample {i}: {e}") from e

        self._tare_offset = np.mean(samples, axis=0)
        self.is_tared = True
        self._logger.info("Tare calibration completed successfully")

    def clear_tare(self) -> None:
        """Clear the tare calibration and return to raw readings."""
        self._tare_offset = None
        self.is_tared = False
        self._logger.info("Tare calibration cleared")

    def normalize(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize tactile data to the configured range.

        Args:
            data: Raw tactile data array.

        Returns:
            np.ndarray: Normalized data in the range [min, max] as configured.
        """
        # Get normalization ranges from config
        min_val, max_val = self.config.normalization_range

        # For Tac3D-like sensors, we need to handle displacement and force separately
        # This is a generic implementation - subclasses may override for specific sensors
        if hasattr(self.config, 'displacement_range') and hasattr(self.config, 'force_range'):
            disp_min, disp_max = self.config.displacement_range
            force_min, force_max = self.config.force_range

            normalized = np.zeros_like(data)

            # Normalize displacement (first 3 columns)
            if data.shape[1] >= 3:
                normalized[:, :3] = self._normalize_range(
                    data[:, :3], disp_min, disp_max, min_val, max_val
                )

            # Normalize force (next 3 columns)
            if data.shape[1] >= 6:
                normalized[:, 3:6] = self._normalize_range(
                    data[:, 3:6], force_min, force_max, min_val, max_val
                )

            return normalized
        else:
            # Generic normalization using data statistics
            data_min = np.min(data, axis=0, keepdims=True)
            data_max = np.max(data, axis=0, keepdims=True)
            data_range = data_max - data_min
            data_range[data_range == 0] = 1.0  # Avoid division by zero

            normalized = (data - data_min) / data_range
            return normalized * (max_val - min_val) + min_val

    def denormalize(self, normalized_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert normalized data back to physical units.

        Args:
            normalized_data: Normalized tactile data array.

        Returns:
            np.ndarray: Data in physical units.
        """
        min_val, max_val = self.config.normalization_range

        if hasattr(self.config, 'displacement_range') and hasattr(self.config, 'force_range'):
            disp_min, disp_max = self.config.displacement_range
            force_min, force_max = self.config.force_range

            denormalized = np.zeros_like(normalized_data)

            # Denormalize displacement
            if normalized_data.shape[1] >= 3:
                denormalized[:, :3] = self._denormalize_range(
                    normalized_data[:, :3], min_val, max_val, disp_min, disp_max
                )

            # Denormalize force
            if normalized_data.shape[1] >= 6:
                denormalized[:, 3:6] = self._denormalize_range(
                    normalized_data[:, 3:6], min_val, max_val, force_min, force_max
                )

            return denormalized
        else:
            # Generic denormalization
            data_min = np.min(normalized_data, axis=0, keepdims=True)
            data_max = np.max(normalized_data, axis=0, keepdims=True)

            return (normalized_data - min_val) / (max_val - min_val) * (data_max - data_min) + data_min

    def _apply_tare(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply tare offset to data if tare is active.

        Args:
            data: Raw sensor data.

        Returns:
            np.ndarray: Data with tare offset applied.
        """
        if self.is_tared and self._tare_offset is not None:
            return data - self._tare_offset
        return data

    @staticmethod
    def _normalize_range(
        data: NDArray[np.float64],
        data_min: float,
        data_max: float,
        target_min: float,
        target_max: float,
    ) -> NDArray[np.float64]:
        """Normalize data from one range to another."""
        data_range = data_max - data_min
        if data_range == 0:
            return np.full_like(data, (target_min + target_max) / 2)
        return (data - data_min) / data_range * (target_max - target_min) + target_min

    @staticmethod
    def _denormalize_range(
        data: NDArray[np.float64],
        norm_min: float,
        norm_max: float,
        target_min: float,
        target_max: float,
    ) -> NDArray[np.float64]:
        """Denormalize data from normalized range to target range."""
        norm_range = norm_max - norm_min
        if norm_range == 0:
            return np.full_like(data, (target_min + target_max) / 2)
        return (data - norm_min) / norm_range * (target_max - target_min) + target_min

    def get_observation_features(self) -> dict[str, tuple[int, ...]]:
        """Get the observation features specification for this sensor.

        Returns:
            dict: Dictionary mapping feature names to their shapes.
                Example: {"tactile": (400, 6)} for a Tac3D sensor.
        """
        return {"tactile": self.config.expected_shape}
