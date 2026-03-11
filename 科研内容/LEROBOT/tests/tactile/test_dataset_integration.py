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

"""Unit tests for dataset integration utilities."""

import numpy as np
import pytest

from lerobot.tactile.configs import SimulatedTactileSensorConfig, TactileDataType
from lerobot.tactile.dataset_integration import (
    compute_tactile_stats,
    create_tactile_features,
    get_tactile_data_for_policy,
    get_tactile_feature_key,
    merge_tactile_features,
    normalize_tactile_data,
    tactile_to_frame,
    validate_tactile_data,
)


class TestTactileFeatureKey:
    """Tests for get_tactile_feature_key."""

    def test_default_sensor_name(self):
        """Test default sensor name."""
        key = get_tactile_feature_key()
        assert key == "observation.tactile"

    def test_custom_sensor_name(self):
        """Test custom sensor name."""
        key = get_tactile_feature_key("finger1")
        assert key == "observation.finger1"


class TestCreateTactileFeatures:
    """Tests for create_tactile_features."""

    def test_full_data_type(self):
        """Test feature creation for full data type."""
        config = SimulatedTactileSensorConfig(
            num_points=400,
            data_type=TactileDataType.FULL,
        )
        features = create_tactile_features(config)

        assert "observation.tactile" in features
        assert features["observation.tactile"]["dtype"] == "float32"
        assert features["observation.tactile"]["shape"] == (400, 6)
        assert features["observation.tactile"]["names"] == ["dx", "dy", "dz", "Fx", "Fy", "Fz"]

    def test_displacement_only(self):
        """Test feature creation for displacement-only data."""
        config = SimulatedTactileSensorConfig(
            num_points=100,
            data_type=TactileDataType.DISPLACEMENT,
        )
        features = create_tactile_features(config)

        assert features["observation.tactile"]["shape"] == (100, 3)
        assert features["observation.tactile"]["names"] == ["dx", "dy", "dz"]

    def test_force_only(self):
        """Test feature creation for force-only data."""
        config = SimulatedTactileSensorConfig(
            num_points=100,
            data_type=TactileDataType.FORCE,
        )
        features = create_tactile_features(config)

        assert features["observation.tactile"]["shape"] == (100, 3)
        assert features["observation.tactile"]["names"] == ["Fx", "Fy", "Fz"]

    def test_custom_sensor_name(self):
        """Test feature creation with custom sensor name."""
        config = SimulatedTactileSensorConfig()
        features = create_tactile_features(config, sensor_name="finger_tip")

        assert "observation.finger_tip" in features

    def test_not_as_observation(self):
        """Test feature creation without observation prefix."""
        config = SimulatedTactileSensorConfig()
        features = create_tactile_features(config, as_observation=False)

        assert "tactile" in features
        assert "observation.tactile" not in features


class TestValidateTactileData:
    """Tests for validate_tactile_data."""

    def test_valid_data(self):
        """Test validation of valid data."""
        config = SimulatedTactileSensorConfig(
            num_points=400,
            data_type=TactileDataType.FULL,
        )
        data = np.random.randn(400, 6).astype(np.float64)

        # Should not raise
        validate_tactile_data(data, config)

    def test_invalid_shape(self):
        """Test validation with wrong shape."""
        config = SimulatedTactileSensorConfig(num_points=400)
        data = np.random.randn(100, 6).astype(np.float64)

        with pytest.raises(ValueError, match="shape"):
            validate_tactile_data(data, config)

    def test_invalid_type(self):
        """Test validation with wrong type."""
        config = SimulatedTactileSensorConfig()
        data = [1, 2, 3]  # Not a numpy array

        with pytest.raises(TypeError):
            validate_tactile_data(data, config)


class TestTactileToFrame:
    """Tests for tactile_to_frame."""

    def test_frame_creation(self):
        """Test frame dictionary creation."""
        config = SimulatedTactileSensorConfig()
        data = np.random.randn(400, 6).astype(np.float64)

        frame = tactile_to_frame(data, config)

        assert "observation.tactile" in frame
        assert frame["observation.tactile"].shape == (400, 6)
        assert frame["observation.tactile"].dtype == np.float32


class TestComputeTactileStats:
    """Tests for compute_tactile_stats."""

    def test_stats_from_list(self):
        """Test computing stats from list of arrays."""
        config = SimulatedTactileSensorConfig()
        frames = [np.random.randn(400, 6).astype(np.float64) for _ in range(10)]

        stats = compute_tactile_stats(frames, config)

        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "count" in stats
        assert "q50" in stats  # median

        assert stats["min"].shape == (6,)
        assert stats["max"].shape == (6,)
        assert stats["mean"].shape == (6,)
        assert stats["count"][0] == 400 * 10  # num_points * num_frames

    def test_stats_from_array(self):
        """Test computing stats from stacked array."""
        config = SimulatedTactileSensorConfig()
        data = np.random.randn(10, 400, 6).astype(np.float64)

        stats = compute_tactile_stats(data, config)

        assert stats["count"][0] == 400 * 10


class TestNormalizeTactileData:
    """Tests for normalize_tactile_data."""

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        stats = {
            "min": np.array([1.0, 2.0]),
            "max": np.array([5.0, 6.0]),
        }

        normalized = normalize_tactile_data(data, stats, method="minmax")

        # First row should be 0, last row should be 1
        np.testing.assert_allclose(normalized[0], [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(normalized[2], [1.0, 1.0], atol=1e-6)

    def test_standardize_normalization(self):
        """Test standardization (z-score)."""
        data = np.random.randn(100, 6).astype(np.float64)
        stats = {
            "mean": np.mean(data, axis=0),
            "std": np.std(data, axis=0),
        }

        normalized = normalize_tactile_data(data, stats, method="standardize")

        # Normalized data should have mean ~0 and std ~1
        np.testing.assert_allclose(np.mean(normalized, axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.std(normalized, axis=0), 1.0, atol=1e-6)

    def test_invalid_method(self):
        """Test with invalid normalization method."""
        data = np.random.randn(10, 6)
        stats = {"min": np.zeros(6), "max": np.ones(6)}

        with pytest.raises(ValueError, match="Unknown"):
            normalize_tactile_data(data, stats, method="invalid")


class TestMergeTactileFeatures:
    """Tests for merge_tactile_features."""

    def test_merge_features(self):
        """Test merging tactile features with base features."""
        base_features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["x", "y", "z", "roll", "pitch", "yaw"],
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw"],
            },
        }

        config = SimulatedTactileSensorConfig()
        merged = merge_tactile_features(base_features, config)

        assert "observation.state" in merged
        assert "action" in merged
        assert "observation.tactile" in merged


class TestGetTactileDataForPolicy:
    """Tests for get_tactile_data_for_policy."""

    def test_flatten_true(self):
        """Test getting flattened data for policy."""
        config = SimulatedTactileSensorConfig()
        data = np.random.randn(400, 6).astype(np.float64)

        flattened = get_tactile_data_for_policy(data, config, flatten=True)

        assert flattened.shape == (400 * 6,)

    def test_flatten_false(self):
        """Test getting 2D data for policy."""
        config = SimulatedTactileSensorConfig()
        data = np.random.randn(400, 6).astype(np.float64)

        result = get_tactile_data_for_policy(data, config, flatten=False)

        assert result.shape == (400, 6)
