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

"""Tactile sensor configuration classes for LeRobot.

This module provides configuration classes for tactile sensors, following the same
pattern as camera configurations in LeRobot.
"""

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import draccus
import numpy as np


class TactileDataType(str, Enum):
    """Supported tactile data types."""

    DISPLACEMENT = "displacement"  # 3D displacement field (dx, dy, dz)
    FORCE = "force"  # 3D force distribution (Fx, Fy, Fz)
    FULL = "full"  # Both displacement and force (6D)

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise ValueError(f"`data_type` is expected to be in {list(cls)}, but {value} is provided.")


class PointCloudFormat(str, Enum):
    """Supported point cloud data formats."""

    XYZ = "xyz"  # Only positions
    XYZI = "xyzi"  # Positions + intensity
    XYZRGB = "xyzrgb"  # Positions + colors
    XYZ_DISPLACEMENT_FORCE = "xyz_displacement_force"  # Full Tac3D format

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise ValueError(f"`pointcloud_format` is expected to be in {list(cls)}, but {value} is provided.")


@dataclass(kw_only=True)
class TactileSensorConfig(draccus.ChoiceRegistry, abc.ABC):
    """Base configuration class for tactile sensors.

    This abstract base class defines the common configuration parameters for all
    tactile sensor implementations. Subclasses must implement sensor-specific
    configurations.

    Attributes:
        fps: Frames per second for data acquisition.
        num_points: Number of tactile sensing points (e.g., 400 for 20x20 array).
        data_type: Type of tactile data to capture (displacement, force, or full).
        pointcloud_format: Format for point cloud data representation.
        normalization_range: Tuple of (min, max) for data normalization.
        apply_tare: Whether to apply tare/zeroing on startup.
        tare_samples: Number of samples to collect for tare calculation.
    """

    fps: int = 30
    num_points: int = 400
    data_type: TactileDataType = TactileDataType.FULL
    pointcloud_format: PointCloudFormat = PointCloudFormat.XYZ_DISPLACEMENT_FORCE
    normalization_range: tuple[float, float] = field(default_factory=lambda: (-1.0, 1.0))
    apply_tare: bool = True
    tare_samples: int = 10

    @property
    def type(self) -> str:
        """Return the sensor type identifier."""
        return str(self.get_choice_name(self.__class__))

    @property
    def data_dim(self) -> int:
        """Return the dimensionality of tactile data per point.

        Returns:
            int: Number of data dimensions (3 for displacement/force only, 6 for full).
        """
        dim_map = {
            TactileDataType.DISPLACEMENT: 3,
            TactileDataType.FORCE: 3,
            TactileDataType.FULL: 6,
        }
        return dim_map[self.data_type]

    @property
    def expected_shape(self) -> tuple[int, ...]:
        """Return the expected shape of tactile data.

        Returns:
            tuple: Shape as (num_points, data_dim).
        """
        return (self.num_points, self.data_dim)


@dataclass(kw_only=True)
class Tac3DSensorConfig(TactileSensorConfig):
    """Configuration for Tac3D tactile sensors.

    Tac3D sensors provide high-resolution tactile feedback with 20x20 sensing arrays,
    measuring both 3D displacement and 3D force distributions.

    Attributes:
        udp_port: UDP port for receiving sensor data from Tac3D-Desktop.
        sensor_sn: Serial number of the sensor to connect to.
        tare_on_startup: Whether to perform tare calibration on startup.
        displacement_range: Range for displacement normalization in mm.
        force_range: Range for force normalization in N.
        timeout_ms: Timeout for frame reception in milliseconds.
    """

    udp_port: int = 9988
    sensor_sn: str | None = None
    tare_on_startup: bool = True
    displacement_range: tuple[float, float] = field(default_factory=lambda: (-2.0, 3.0))
    force_range: tuple[float, float] = field(default_factory=lambda: (-0.8, 0.8))
    timeout_ms: float = 1000.0

    def __post_init__(self):
        """Validate Tac3D-specific configuration."""
        if self.num_points != 400:
            raise ValueError(f"Tac3D sensors have exactly 400 points (20x20), got {self.num_points}")


@dataclass(kw_only=True)
class PointCloud2SensorConfig(TactileSensorConfig):
    """Configuration for ROS2 PointCloud2 tactile sensors.

    This configuration is used for generic ROS2-based tactile sensors that publish
    data as PointCloud2 messages.

    Attributes:
        topic_name: ROS2 topic name for PointCloud2 messages.
        node_name: Name of the ROS2 node for this sensor.
        frame_id: TF frame ID for the sensor.
        qos_profile: QoS profile for ROS2 subscription (reliable/best_effort).
        queue_size: Size of the message queue for incoming point clouds.
        timeout_ms: Timeout for message reception in milliseconds.
        use_sim_time: Whether to use simulation time in ROS2.
    """

    topic_name: str = "/tactile/pointcloud"
    node_name: str = "tactile_sensor_node"
    frame_id: str = "tactile_sensor_link"
    qos_profile: str = "reliable"
    queue_size: int = 10
    timeout_ms: float = 1000.0
    use_sim_time: bool = False


@dataclass(kw_only=True)
class SimulatedTactileSensorConfig(TactileSensorConfig):
    """Configuration for simulated tactile sensors.

    Used for testing and development without physical hardware.

    Attributes:
        noise_std: Standard deviation of simulated noise.
        seed: Random seed for reproducible simulation.
        simulate_delay: Whether to simulate realistic sensor delays.
    """

    noise_std: float = 0.01
    seed: int = 42
    simulate_delay: bool = False
