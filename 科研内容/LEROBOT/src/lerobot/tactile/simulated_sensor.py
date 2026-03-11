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

"""Simulated tactile sensor implementation for LeRobot.

This module provides the SimulatedTactileSensor class for testing and development
without physical hardware.
"""

import logging
import threading
import time
from queue import Empty, Queue
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.tactile.configs import SimulatedTactileSensorConfig, TactileDataType
from lerobot.tactile.tactile_sensor import TactileSensor


class SimulatedTactileSensor(TactileSensor):
    """Simulated tactile sensor for testing and development.

    Generates synthetic tactile data that mimics real sensor behavior including:
    - Realistic noise patterns
    - Contact simulation when "touching"
    - Configurable response characteristics

    Attributes:
        config: SimulatedTactileSensorConfig with simulation parameters.
        is_connected: Whether the simulation is active.

    Example:
        ```python
        config = SimulatedTactileSensorConfig(fps=30, num_points=400)
        sensor = SimulatedTactileSensor(config)

        with sensor:
            # Simulate contact
            sensor.set_contact_center([0.5, 0.5])  # Normalized coordinates
            sensor.set_contact_force(1.0)  # Newtons
            data = sensor.read()
        ```
    """

    def __init__(self, config: SimulatedTactileSensorConfig):
        """Initialize the simulated sensor.

        Args:
            config: SimulatedTactileSensorConfig with simulation parameters.
        """
        super().__init__(config)
        self.config: SimulatedTactileSensorConfig = config

        # Simulation state
        self._running = False
        self._rng = np.random.default_rng(config.seed)

        # Contact simulation parameters
        self._contact_center = np.array([0.5, 0.5])  # Normalized [0,1] coordinates
        self._contact_force = 0.0  # Newtons
        self._contact_radius = 0.1  # Normalized radius
        self._contact_lock = threading.Lock()

        # Sensor grid positions (normalized 0-1)
        self._grid_positions = self._create_grid_positions()

        # Frame buffer
        self._frame_queue: Queue[NDArray[np.float64]] = Queue(maxsize=10)
        self._latest_frame: NDArray[np.float64] | None = None
        self._frame_lock = threading.Lock()

        # Background thread
        self._simulation_thread: threading.Thread | None = None

        # Statistics
        self._frame_count = 0
        self._start_time: float | None = None

        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def is_connected(self) -> bool:
        """Check if simulation is running.

        Returns:
            bool: True if simulation is active.
        """
        return self._running

    @staticmethod
    def find_sensors() -> list[dict[str, Any]]:
        """Return list of simulated sensors.

        Returns:
            list[dict]: Single simulated sensor entry.
        """
        return [{
            'id': 'simulated_tactile_0',
            'type': 'simulated',
            'name': 'Simulated Tactile Sensor',
        }]

    def connect(self, warmup: bool = True) -> None:
        """Start the simulation.

        Args:
            warmup: If True, generates initial frames before returning.
        """
        if self.is_connected:
            self._logger.warning("Simulation already running")
            return

        self._logger.info("Starting simulated tactile sensor...")
        self._running = True
        self._start_time = time.time()

        # Start simulation thread
        self._simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._simulation_thread.start()

        if warmup:
            # Generate a few frames to populate buffer
            time.sleep(0.1)

        if self.config.apply_tare:
            self.tare()

        self._logger.info("Simulated sensor started")

    def disconnect(self) -> None:
        """Stop the simulation."""
        self._logger.info("Stopping simulated sensor...")
        self._running = False

        if self._simulation_thread and self._simulation_thread.is_alive():
            self._simulation_thread.join(timeout=1.0)

        # Clear buffers
        with self._frame_lock:
            self._latest_frame = None
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break

        self._logger.info("Simulated sensor stopped")

    def read(self) -> NDArray[np.float64]:
        """Read a simulated tactile frame.

        Returns:
            np.ndarray: Simulated tactile data.

        Raises:
            ConnectionError: If simulation is not running.
        """
        if not self.is_connected:
            raise ConnectionError("Simulation is not running")

        return self._generate_frame()

    def async_read(self, timeout_ms: float = 1000.0) -> NDArray[np.float64]:
        """Read the most recent simulated frame.

        Args:
            timeout_ms: Not used in simulation (always returns immediately).

        Returns:
            np.ndarray: Simulated tactile data.
        """
        return self.read()

    def read_latest(self, max_age_ms: int = 500) -> NDArray[np.float64]:
        """Get the latest frame.

        Args:
            max_age_ms: Not used in simulation.

        Returns:
            np.ndarray: Simulated tactile data.
        """
        return self.read()

    def set_contact_center(self, center: list[float] | NDArray[np.float64]) -> None:
        """Set the contact center position.

        Args:
            center: Normalized [x, y] coordinates in range [0, 1].
        """
        with self._contact_lock:
            self._contact_center = np.array(center, dtype=np.float64)

    def set_contact_force(self, force: float) -> None:
        """Set the contact force magnitude.

        Args:
            force: Force in Newtons.
        """
        with self._contact_lock:
            self._contact_force = force

    def set_contact_radius(self, radius: float) -> None:
        """Set the contact radius.

        Args:
            radius: Normalized radius in range [0, 1].
        """
        with self._contact_lock:
            self._contact_radius = radius

    def get_statistics(self) -> dict[str, Any]:
        """Get simulation statistics.

        Returns:
            dict: Dictionary with frame count and simulation status.
        """
        elapsed = time.time() - self._start_time if self._start_time else 0
        fps = self._frame_count / elapsed if elapsed > 0 else 0

        return {
            "frame_count": self._frame_count,
            "elapsed_time": elapsed,
            "average_fps": fps,
            "is_connected": self.is_connected,
            "is_tared": self.is_tared,
            "contact_force": self._contact_force,
        }

    def _create_grid_positions(self) -> NDArray[np.float64]:
        """Create normalized grid positions for sensing points.

        Returns:
            np.ndarray: Array of shape (num_points, 2) with [x, y] positions.
        """
        # Assume square grid
        grid_size = int(np.sqrt(self.num_points))
        if grid_size * grid_size != self.num_points:
            # Not a perfect square, use rectangular grid
            grid_size = int(np.ceil(np.sqrt(self.num_points)))

        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        xx, yy = np.meshgrid(x, y)
        positions = np.stack([xx.flatten(), yy.flatten()], axis=1)

        # Trim to exact num_points
        return positions[:self.num_points]

    def _generate_frame(self) -> NDArray[np.float64]:
        """Generate a simulated tactile frame.

        Returns:
            np.ndarray: Simulated tactile data with shape (num_points, data_dim).
        """
        with self._contact_lock:
            center = self._contact_center.copy()
            force = self._contact_force
            radius = self._contact_radius

        # Calculate distances from contact center
        distances = np.linalg.norm(self._grid_positions - center, axis=1)

        # Generate displacement based on contact
        if self.config.data_type == TactileDataType.DISPLACEMENT:
            data = self._generate_displacement(distances, force, radius)
        elif self.config.data_type == TactileDataType.FORCE:
            data = self._generate_force(distances, force, radius)
        else:  # TactileDataType.FULL
            displacement = self._generate_displacement(distances, force, radius)
            force_data = self._generate_force(distances, force, radius)
            data = np.concatenate([displacement, force_data], axis=1)

        # Add noise
        noise = self._rng.normal(0, self.config.noise_std, data.shape)
        data = data + noise

        # Apply tare
        data = self._apply_tare(data)

        return data.astype(np.float64)

    def _generate_displacement(
        self, distances: NDArray[np.float64], force: float, radius: float
    ) -> NDArray[np.float64]:
        """Generate displacement field.

        Args:
            distances: Distance of each point from contact center.
            force: Contact force magnitude.
            radius: Contact radius.

        Returns:
            np.ndarray: Displacement vectors (num_points, 3).
        """
        # Gaussian deformation profile
        displacement_magnitude = force * np.exp(-(distances**2) / (2 * radius**2 + 1e-6))

        # Direction is toward contact center (negative Z)
        displacement = np.zeros((self.num_points, 3), dtype=np.float64)
        displacement[:, 2] = -displacement_magnitude  # Z displacement

        # Add some X/Y displacement based on position relative to center
        directions = self._grid_positions - self._contact_center
        directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-6)
        displacement[:, 0] = directions[:, 0] * displacement_magnitude * 0.1
        displacement[:, 1] = directions[:, 1] * displacement_magnitude * 0.1

        return displacement

    def _generate_force(
        self, distances: NDArray[np.float64], force: float, radius: float
    ) -> NDArray[np.float64]:
        """Generate force distribution.

        Args:
            distances: Distance of each point from contact center.
            force: Contact force magnitude.
            radius: Contact radius.

        Returns:
            np.ndarray: Force vectors (num_points, 3).
        """
        # Gaussian force distribution
        force_magnitude = force * np.exp(-(distances**2) / (2 * radius**2 + 1e-6))

        # Force is primarily in Z direction (normal to surface)
        forces = np.zeros((self.num_points, 3), dtype=np.float64)
        forces[:, 2] = force_magnitude  # Normal force

        # Add tangential components
        forces[:, 0] = force_magnitude * 0.1 * self._rng.normal(0, 1, self.num_points)
        forces[:, 1] = force_magnitude * 0.1 * self._rng.normal(0, 1, self.num_points)

        return forces

    def _simulation_loop(self) -> None:
        """Background thread for continuous frame generation."""
        interval = 1.0 / self.fps

        while self._running:
            start_time = time.time()

            frame = self._generate_frame()

            with self._frame_lock:
                self._latest_frame = frame

            if not self._frame_queue.full():
                self._frame_queue.put(frame)

            self._frame_count += 1

            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
