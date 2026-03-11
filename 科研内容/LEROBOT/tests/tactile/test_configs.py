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

"""Unit tests for tactile sensor configurations."""

import pytest

from lerobot.tactile.configs import (
    PointCloud2SensorConfig,
    PointCloudFormat,
    SimulatedTactileSensorConfig,
    Tac3DSensorConfig,
    TactileDataType,
    TactileSensorConfig,
)


class TestTactileDataType:
    """Tests for TactileDataType enum."""

    def test_valid_types(self):
        """Test that valid data types can be accessed."""
        assert TactileDataType.DISPLACEMENT.value == "displacement"
        assert TactileDataType.FORCE.value == "force"
        assert TactileDataType.FULL.value == "full"

    def test_invalid_type_raises_error(self):
        """Test that invalid data type raises ValueError."""
        with pytest.raises(ValueError):
            TactileDataType("invalid_type")


class TestPointCloudFormat:
    """Tests for PointCloudFormat enum."""

    def test_valid_formats(self):
        """Test that valid formats can be accessed."""
        assert PointCloudFormat.XYZ.value == "xyz"
        assert PointCloudFormat.XYZ_DISPLACEMENT_FORCE.value == "xyz_displacement_force"

    def test_invalid_format_raises_error(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError):
            PointCloudFormat("invalid_format")


class TestTactileSensorConfig:
    """Tests for TactileSensorConfig base class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TactileSensorConfig()

        assert config.fps == 30
        assert config.num_points == 400
        assert config.data_type == TactileDataType.FULL
        assert config.pointcloud_format == PointCloudFormat.XYZ_DISPLACEMENT_FORCE
        assert config.normalization_range == (-1.0, 1.0)
        assert config.apply_tare is True
        assert config.tare_samples == 10

    def test_data_dim(self):
        """Test data dimension calculation."""
        config_disp = TactileSensorConfig(data_type=TactileDataType.DISPLACEMENT)
        assert config_disp.data_dim == 3

        config_force = TactileSensorConfig(data_type=TactileDataType.FORCE)
        assert config_force.data_dim == 3

        config_full = TactileSensorConfig(data_type=TactileDataType.FULL)
        assert config_full.data_dim == 6

    def test_expected_shape(self):
        """Test expected shape calculation."""
        config = TactileSensorConfig(num_points=400, data_type=TactileDataType.FULL)
        assert config.expected_shape == (400, 6)

        config = TactileSensorConfig(num_points=100, data_type=TactileDataType.DISPLACEMENT)
        assert config.expected_shape == (100, 3)


class TestTac3DSensorConfig:
    """Tests for Tac3DSensorConfig."""

    def test_default_values(self):
        """Test default Tac3D configuration values."""
        config = Tac3DSensorConfig()

        assert config.udp_port == 9988
        assert config.sensor_sn is None
        assert config.tare_on_startup is True
        assert config.displacement_range == (-2.0, 3.0)
        assert config.force_range == (-0.8, 0.8)
        assert config.timeout_ms == 1000.0

    def test_num_points_validation(self):
        """Test that Tac3D requires exactly 400 points."""
        # Valid configuration
        config = Tac3DSensorConfig(num_points=400)
        assert config.num_points == 400

        # Invalid configuration should raise ValueError
        with pytest.raises(ValueError, match="400"):
            Tac3DSensorConfig(num_points=100)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = Tac3DSensorConfig(
            udp_port=9999,
            sensor_sn="TAC3D001",
            tare_on_startup=False,
            displacement_range=(-1.0, 2.0),
            force_range=(-0.5, 0.5),
            timeout_ms=500.0,
        )

        assert config.udp_port == 9999
        assert config.sensor_sn == "TAC3D001"
        assert config.tare_on_startup is False
        assert config.displacement_range == (-1.0, 2.0)
        assert config.force_range == (-0.5, 0.5)
        assert config.timeout_ms == 500.0


class TestPointCloud2SensorConfig:
    """Tests for PointCloud2SensorConfig."""

    def test_default_values(self):
        """Test default ROS2 configuration values."""
        config = PointCloud2SensorConfig()

        assert config.topic_name == "/tactile/pointcloud"
        assert config.node_name == "tactile_sensor_node"
        assert config.frame_id == "tactile_sensor_link"
        assert config.qos_profile == "reliable"
        assert config.queue_size == 10
        assert config.timeout_ms == 1000.0
        assert config.use_sim_time is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PointCloud2SensorConfig(
            topic_name="/custom/tactile",
            node_name="custom_node",
            frame_id="custom_frame",
            qos_profile="best_effort",
            queue_size=20,
            timeout_ms=2000.0,
            use_sim_time=True,
        )

        assert config.topic_name == "/custom/tactile"
        assert config.node_name == "custom_node"
        assert config.frame_id == "custom_frame"
        assert config.qos_profile == "best_effort"
        assert config.queue_size == 20
        assert config.timeout_ms == 2000.0
        assert config.use_sim_time is True


class TestSimulatedTactileSensorConfig:
    """Tests for SimulatedTactileSensorConfig."""

    def test_default_values(self):
        """Test default simulated sensor configuration values."""
        config = SimulatedTactileSensorConfig()

        assert config.noise_std == 0.01
        assert config.seed == 42
        assert config.simulate_delay is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SimulatedTactileSensorConfig(
            noise_std=0.05,
            seed=123,
            simulate_delay=True,
        )

        assert config.noise_std == 0.05
        assert config.seed == 123
        assert config.simulate_delay is True
