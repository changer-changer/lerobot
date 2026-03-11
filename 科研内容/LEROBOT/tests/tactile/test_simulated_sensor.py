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

"""Unit tests for simulated tactile sensor."""

import time

import numpy as np
import pytest

from lerobot.tactile.configs import SimulatedTactileSensorConfig, TactileDataType
from lerobot.tactile.simulated_sensor import SimulatedTactileSensor


class TestSimulatedTactileSensor:
    """Tests for SimulatedTactileSensor."""

    @pytest.fixture
    def sensor_config(self):
        """Create a test sensor configuration."""
        return SimulatedTactileSensorConfig(
            fps=30,
            num_points=400,
            data_type=TactileDataType.FULL,
            noise_std=0.01,
            seed=42,
        )

    @pytest.fixture
    def sensor(self, sensor_config):
        """Create and connect a test sensor."""
        sensor = SimulatedTactileSensor(sensor_config)
        sensor.connect(warmup=False)
        yield sensor
        sensor.disconnect()

    def test_initialization(self, sensor_config):
        """Test sensor initialization."""
        sensor = SimulatedTactileSensor(sensor_config)

        assert sensor.config == sensor_config
        assert sensor.fps == 30
        assert sensor.num_points == 400
        assert sensor.data_dim == 6
        assert sensor.is_tared is False
        assert sensor.is_connected is False

    def test_connect_disconnect(self, sensor_config):
        """Test sensor connection and disconnection."""
        sensor = SimulatedTactileSensor(sensor_config)

        assert not sensor.is_connected

        sensor.connect(warmup=False)
        assert sensor.is_connected

        sensor.disconnect()
        assert not sensor.is_connected

    def test_read(self, sensor):
        """Test reading data from sensor."""
        data = sensor.read()

        assert isinstance(data, np.ndarray)
        assert data.shape == (400, 6)
        assert data.dtype == np.float64

    def test_async_read(self, sensor):
        """Test async reading from sensor."""
        data = sensor.async_read(timeout_ms=100)

        assert isinstance(data, np.ndarray)
        assert data.shape == (400, 6)

    def test_read_latest(self, sensor):
        """Test reading latest frame."""
        data = sensor.read_latest(max_age_ms=1000)

        assert isinstance(data, np.ndarray)
        assert data.shape == (400, 6)

    def test_tare(self, sensor):
        """Test tare calibration."""
        # Read raw data
        raw_data = sensor.read()

        # Apply tare
        sensor.tare(num_samples=5)

        assert sensor.is_tared is True
        assert sensor._tare_offset is not None

        # Read tared data (should be near zero)
        tared_data = sensor.read()

        # Tared data should be close to zero (just noise)
        assert np.abs(tared_data.mean()) < 0.1

    def test_clear_tare(self, sensor):
        """Test clearing tare calibration."""
        sensor.tare(num_samples=5)
        assert sensor.is_tared is True

        sensor.clear_tare()
        assert sensor.is_tared is False
        assert sensor._tare_offset is None

    def test_context_manager(self, sensor_config):
        """Test using sensor as context manager."""
        with SimulatedTactileSensor(sensor_config) as sensor:
            assert sensor.is_connected
            data = sensor.read()
            assert data.shape == (400, 6)

        assert not sensor.is_connected

    def test_set_contact_parameters(self, sensor):
        """Test setting contact simulation parameters."""
        sensor.set_contact_center([0.3, 0.7])
        sensor.set_contact_force(1.5)
        sensor.set_contact_radius(0.2)

        assert np.allclose(sensor._contact_center, [0.3, 0.7])
        assert sensor._contact_force == 1.5
        assert sensor._contact_radius == 0.2

    def test_contact_simulation(self, sensor):
        """Test that contact simulation produces expected data."""
        # No contact
        sensor.set_contact_force(0.0)
        no_contact_data = sensor.read()

        # With contact
        sensor.set_contact_force(1.0)
        sensor.set_contact_center([0.5, 0.5])
        sensor.set_contact_radius(0.1)
        contact_data = sensor.read()

        # Contact data should have larger magnitude
        assert np.abs(contact_data).max() > np.abs(no_contact_data).max()

    def test_statistics(self, sensor):
        """Test getting sensor statistics."""
        # Read a few frames
        for _ in range(5):
            sensor.read()

        stats = sensor.get_statistics()

        assert "frame_count" in stats
        assert "elapsed_time" in stats
        assert "average_fps" in stats
        assert "is_connected" in stats
        assert "is_tared" in stats

        assert stats["frame_count"] >= 5
        assert stats["is_connected"] is True

    def test_find_sensors(self):
        """Test find_sensors static method."""
        sensors = SimulatedTactileSensor.find_sensors()

        assert isinstance(sensors, list)
        assert len(sensors) == 1
        assert sensors[0]["type"] == "simulated"

    def test_displacement_only_config(self):
        """Test sensor with displacement-only configuration."""
        config = SimulatedTactileSensorConfig(
            data_type=TactileDataType.DISPLACEMENT,
        )
        sensor = SimulatedTactileSensor(config)
        sensor.connect(warmup=False)

        data = sensor.read()
        assert data.shape == (400, 3)

        sensor.disconnect()

    def test_force_only_config(self):
        """Test sensor with force-only configuration."""
        config = SimulatedTactileSensorConfig(
            data_type=TactileDataType.FORCE,
        )
        sensor = SimulatedTactileSensor(config)
        sensor.connect(warmup=False)

        data = sensor.read()
        assert data.shape == (400, 3)

        sensor.disconnect()

    def test_different_num_points(self):
        """Test sensor with different number of points."""
        config = SimulatedTactileSensorConfig(
            num_points=100,
        )
        sensor = SimulatedTactileSensor(config)
        sensor.connect(warmup=False)

        data = sensor.read()
        assert data.shape == (100, 6)

        sensor.disconnect()

    def test_read_not_connected(self, sensor_config):
        """Test that reading from disconnected sensor raises error."""
        sensor = SimulatedTactileSensor(sensor_config)
        # Don't connect

        with pytest.raises(Exception):
            sensor.read()


class TestSimulatedTactileSensorNormalization:
    """Tests for sensor normalization functionality."""

    @pytest.fixture
    def sensor(self):
        """Create a connected test sensor."""
        config = SimulatedTactileSensorConfig(
            normalization_range=(-1.0, 1.0),
        )
        sensor = SimulatedTactileSensor(config)
        sensor.connect(warmup=False)
        yield sensor
        sensor.disconnect()

    def test_normalize(self, sensor):
        """Test data normalization."""
        data = sensor.read()
        normalized = sensor.normalize(data)

        # Normalized data should be in range [-1, 1]
        assert normalized.min() >= -1.5  # Allow some margin for noise
        assert normalized.max() <= 1.5

    def test_denormalize(self, sensor):
        """Test data denormalization."""
        data = sensor.read()
        normalized = sensor.normalize(data)
        denormalized = sensor.denormalize(normalized)

        # Denormalized data should be close to original
        np.testing.assert_allclose(denormalized, data, rtol=0.1)
