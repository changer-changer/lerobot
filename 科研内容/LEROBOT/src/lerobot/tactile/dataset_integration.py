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

"""Tactile dataset integration for LeRobot.

This module provides utilities for integrating tactile data with LeRobot datasets,
including feature specification, data validation, and statistics computation.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.tactile.configs import TactileSensorConfig, TactileDataType


def get_tactile_feature_key(sensor_name: str = "tactile") -> str:
    """Get the standard feature key for tactile data.

    Args:
        sensor_name: Name identifier for the sensor.

    Returns:
        str: Standard feature key (e.g., "observation.tactile").
    """
    return f"observation.{sensor_name}"


def create_tactile_features(
    config: TactileSensorConfig,
    sensor_name: str = "tactile",
    as_observation: bool = True,
) -> dict[str, dict]:
    """Create LeRobot feature specification for tactile data.

    Args:
        config: TactileSensorConfig with sensor parameters.
        sensor_name: Name identifier for the sensor.
        as_observation: If True, prefix with "observation.", else use as-is.

    Returns:
        dict: LeRobot feature specification dictionary.

    Example:
        ```python
        config = Tac3DSensorConfig(data_type=TactileDataType.FULL)
        features = create_tactile_features(config)
        # Returns:
        # {
        #     "observation.tactile": {
        #         "dtype": "float32",
        #         "shape": (400, 6),
        #         "names": ["dx", "dy", "dz", "Fx", "Fy", "Fz"],
        #     }
        # }
        ```
    """
    key = get_tactile_feature_key(sensor_name) if as_observation else sensor_name

    # Create dimension names based on data type
    if config.data_type == TactileDataType.DISPLACEMENT:
        names = ["dx", "dy", "dz"]
    elif config.data_type == TactileDataType.FORCE:
        names = ["Fx", "Fy", "Fz"]
    else:  # TactileDataType.FULL
        names = ["dx", "dy", "dz", "Fx", "Fy", "Fz"]

    return {
        key: {
            "dtype": "float32",
            "shape": config.expected_shape,
            "names": names,
        }
    }


def validate_tactile_data(
    data: NDArray[np.float64],
    config: TactileSensorConfig,
) -> None:
    """Validate tactile data against sensor configuration.

    Args:
        data: Tactile data array to validate.
        config: Expected sensor configuration.

    Raises:
        ValueError: If data shape or dtype is invalid.
        TypeError: If data is not a numpy array.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Tactile data must be numpy array, got {type(data)}")

    expected_shape = config.expected_shape
    if data.shape != expected_shape:
        raise ValueError(
            f"Tactile data shape mismatch: expected {expected_shape}, got {data.shape}"
        )

    if data.dtype not in [np.float32, np.float64]:
        raise ValueError(f"Tactile data must be float32 or float64, got {data.dtype}")


def tactile_to_frame(
    data: NDArray[np.float64],
    config: TactileSensorConfig,
    sensor_name: str = "tactile",
) -> dict[str, NDArray[np.float64]]:
    """Convert tactile data to a dataset frame.

    Args:
        data: Tactile data array.
        config: Sensor configuration.
        sensor_name: Name identifier for the sensor.

    Returns:
        dict: Frame dictionary ready for dataset.add_frame().

    Example:
        ```python
        data = sensor.read()  # (400, 6) array
        frame = tactile_to_frame(data, config)
        dataset.add_frame({**observation_frame, **frame, "task": task})
        ```
    """
    validate_tactile_data(data, config)

    key = get_tactile_feature_key(sensor_name)
    return {key: data.astype(np.float32)}


def compute_tactile_stats(
    data: list[NDArray[np.float64]] | NDArray[np.float64],
    config: TactileSensorConfig,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute statistics for tactile data.

    Computes min, max, mean, std, and quantiles for tactile data.
    Statistics are computed per-dimension across all points.

    Args:
        data: Tactile data as list of arrays or stacked array.
        config: Sensor configuration.

    Returns:
        dict: Statistics dictionary compatible with LeRobot dataset stats.

    Example:
        ```python
        # Collect data from multiple frames
        frames = [sensor.read() for _ in range(100)]
        stats = compute_tactile_stats(frames, config)
        ```
    """
    if isinstance(data, list):
        data = np.stack(data, axis=0)  # (N, num_points, data_dim)

    # Reshape to (N * num_points, data_dim) for per-dimension stats
    N, num_points, data_dim = data.shape
    flat_data = data.reshape(-1, data_dim)

    # Compute statistics per dimension
    stats = {
        "min": np.min(flat_data, axis=0),
        "max": np.max(flat_data, axis=0),
        "mean": np.mean(flat_data, axis=0),
        "std": np.std(flat_data, axis=0),
        "count": np.array([N * num_points]),
    }

    # Add quantiles
    quantiles = [0.01, 0.10, 0.50, 0.90, 0.99]
    for q in quantiles:
        key = f"q{int(q * 100):02d}"
        stats[key] = np.quantile(flat_data, q, axis=0)

    return stats


def normalize_tactile_data(
    data: NDArray[np.float64],
    stats: dict[str, np.ndarray],
    method: str = "minmax",
) -> NDArray[np.float64]:
    """Normalize tactile data using statistics.

    Args:
        data: Tactile data array.
        stats: Statistics dictionary from compute_tactile_stats.
        method: Normalization method ("minmax" or "standardize").

    Returns:
        np.ndarray: Normalized data.
    """
    if method == "minmax":
        data_min = stats["min"]
        data_max = stats["max"]
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0  # Avoid division by zero
        return (data - data_min) / data_range
    elif method == "standardize":
        mean = stats["mean"]
        std = stats["std"]
        std[std == 0] = 1.0  # Avoid division by zero
        return (data - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def denormalize_tactile_data(
    normalized_data: NDArray[np.float64],
    stats: dict[str, np.ndarray],
    method: str = "minmax",
) -> NDArray[np.float64]:
    """Denormalize tactile data using statistics.

    Args:
        normalized_data: Normalized tactile data array.
        stats: Statistics dictionary from compute_tactile_stats.
        method: Normalization method ("minmax" or "standardize").

    Returns:
        np.ndarray: Denormalized data in original units.
    """
    if method == "minmax":
        data_min = stats["min"]
        data_max = stats["max"]
        data_range = data_max - data_min
        return normalized_data * data_range + data_min
    elif method == "standardize":
        mean = stats["mean"]
        std = stats["std"]
        return normalized_data * std + mean
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def merge_tactile_features(
    base_features: dict[str, dict],
    tactile_config: TactileSensorConfig,
    sensor_name: str = "tactile",
) -> dict[str, dict]:
    """Merge tactile features into existing feature specification.

    Args:
        base_features: Base LeRobot feature specification.
        tactile_config: Tactile sensor configuration.
        sensor_name: Name identifier for the sensor.

    Returns:
        dict: Merged feature specification.
    """
    tactile_features = create_tactile_features(tactile_config, sensor_name)
    return {**base_features, **tactile_features}


def get_tactile_data_for_policy(
    data: NDArray[np.float64],
    config: TactileSensorConfig,
    flatten: bool = True,
) -> NDArray[np.float64]:
    """Prepare tactile data for policy input.

    Args:
        data: Tactile data array.
        config: Sensor configuration.
        flatten: If True, flatten to 1D array; else return as 2D.

    Returns:
        np.ndarray: Data formatted for policy input.
    """
    validate_tactile_data(data, config)

    if flatten:
        return data.flatten()
    return data
