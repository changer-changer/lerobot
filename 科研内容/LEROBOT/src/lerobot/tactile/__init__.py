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

"""Tactile sensor module for LeRobot.

This module provides tactile sensor support for the LeRobot framework, enabling
high-resolution tactile data collection and integration with robot learning pipelines.

Supported sensors:
    - Tac3D: High-resolution tactile sensors with 20x20 sensing arrays
    - ROS2 PointCloud2: Generic ROS2-based tactile sensors
    - Simulated: For testing and development without hardware

Example:
    ```python
    from lerobot.tactile import Tac3DTactileSensor, Tac3DSensorConfig

    config = Tac3DSensorConfig(udp_port=9988)
    sensor = Tac3DTactileSensor(config)

    with sensor:
        sensor.tare()
        data = sensor.read()  # (400, 6) array
    ```
"""

# Configuration classes
from lerobot.tactile.configs import (
    PointCloud2SensorConfig,
    PointCloudFormat,
    SimulatedTactileSensorConfig,
    Tac3DSensorConfig,
    TactileDataType,
    TactileSensorConfig,
)

# Sensor implementations
from lerobot.tactile.ros2_bridge.pointcloud2_sensor import PointCloud2TactileSensor
from lerobot.tactile.simulated_sensor import SimulatedTactileSensor
from lerobot.tactile.direct_connection.tac3d_sensor import Tac3DTactileSensor
from lerobot.tactile.tactile_sensor import TactileSensor

# Dataset integration
from lerobot.tactile.dataset_integration import (
    compute_tactile_stats,
    create_tactile_features,
    get_tactile_feature_key,
    merge_tactile_features,
    normalize_tactile_data,
    tactile_to_frame,
    validate_tactile_data,
)

__all__ = [
    # Base classes
    "TactileSensor",
    "TactileSensorConfig",
    # Enums
    "TactileDataType",
    "PointCloudFormat",
    # Configurations
    "Tac3DSensorConfig",
    "PointCloud2SensorConfig",
    "SimulatedTactileSensorConfig",
    # Sensor implementations
    "Tac3DTactileSensor",
    "PointCloud2TactileSensor",
    "SimulatedTactileSensor",
    # Dataset integration
    "create_tactile_features",
    "get_tactile_feature_key",
    "validate_tactile_data",
    "tactile_to_frame",
    "compute_tactile_stats",
    "normalize_tactile_data",
    "merge_tactile_features",
]
