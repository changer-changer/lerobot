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

"""Example: Record tactile data with LeRobot dataset.

This example demonstrates how to record tactile sensor data alongside other
observations and actions in a LeRobot dataset.

Usage:
    python record_tactile_data.py --sensor-type tac3d --output my_dataset

    # With simulated sensor for testing
    python record_tactile_data.py --sensor-type simulated --output test_dataset
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.tactile import (
    Tac3DSensorConfig,
    Tac3DTactileSensor,
    SimulatedTactileSensorConfig,
    SimulatedTactileSensor,
    create_tactile_features,
    merge_tactile_features,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Record tactile data with LeRobot")
    parser.add_argument(
        "--sensor-type",
        type=str,
        choices=["tac3d", "simulated"],
        default="simulated",
        help="Type of tactile sensor to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output dataset repository ID (e.g., 'username/dataset_name')",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording frames per second",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=100,
        help="Number of frames per episode",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=9988,
        help="UDP port for Tac3D sensor (only for tac3d type)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory for dataset storage",
    )
    return parser.parse_args()


def create_sensor(sensor_type: str, udp_port: int = 9988):
    """Create and configure tactile sensor."""
    if sensor_type == "tac3d":
        config = Tac3DSensorConfig(
            fps=30,
            udp_port=udp_port,
            data_type="full",
            tare_on_startup=True,
        )
        sensor = Tac3DTactileSensor(config)
    elif sensor_type == "simulated":
        config = SimulatedTactileSensorConfig(
            fps=30,
            num_points=400,
            data_type="full",
            apply_tare=True,
            noise_std=0.01,
        )
        sensor = SimulatedTactileSensor(config)
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")

    return sensor, config


def create_dataset_features(tactile_config, fps: int):
    """Create dataset features including tactile data."""
    # Base features (similar to what a robot would provide)
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

    # Add tactile features
    tactile_features = create_tactile_features(tactile_config, sensor_name="tactile")
    all_features = merge_tactile_features(base_features, tactile_config)

    return all_features


def record_episode(sensor, dataset, episode_length: int, task: str):
    """Record a single episode."""
    logging.info(f"Recording episode with {episode_length} frames...")

    for frame_idx in range(episode_length):
        # Read tactile data
        tactile_data = sensor.read()

        # Simulate other observations (in real use, these come from robot)
        state = np.random.randn(6).astype(np.float32)
        action = np.random.randn(6).astype(np.float32)

        # Build frame
        frame = {
            "observation.state": state,
            "observation.tactile": tactile_data.astype(np.float32),
            "action": action,
            "task": task,
        }

        # Add to dataset
        dataset.add_frame(frame)

        if (frame_idx + 1) % 10 == 0:
            logging.info(f"  Recorded {frame_idx + 1}/{episode_length} frames")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Create sensor
    logging.info(f"Creating {args.sensor_type} sensor...")
    sensor, tactile_config = create_sensor(args.sensor_type, args.udp_port)

    # Create dataset features
    features = create_dataset_features(tactile_config, args.fps)

    # Create dataset
    logging.info(f"Creating dataset: {args.output}")
    dataset = LeRobotDataset.create(
        repo_id=args.output,
        fps=args.fps,
        features=features,
        root=args.root,
        robot_type="tactile_test",
        use_videos=False,
    )

    try:
        # Connect to sensor
        sensor.connect(warmup=True)
        logging.info("Sensor connected and ready")

        # Record episodes
        for episode_idx in range(args.num_episodes):
            task = f"Tactile manipulation episode {episode_idx + 1}"
            logging.info(f"\n=== Recording Episode {episode_idx + 1}/{args.num_episodes} ===")

            record_episode(sensor, dataset, args.episode_length, task)

            # Save episode
            dataset.save_episode()
            logging.info(f"Episode {episode_idx + 1} saved")

    except KeyboardInterrupt:
        logging.info("\nRecording interrupted by user")
    finally:
        # Cleanup
        sensor.disconnect()
        dataset.finalize()
        logging.info("Dataset finalized")

    # Print statistics
    stats = sensor.get_statistics()
    logging.info(f"\nSensor Statistics:")
    for key, value in stats.items():
        logging.info(f"  {key}: {value}")

    logging.info(f"\nDataset saved to: {dataset.root}")
    logging.info(f"Total episodes: {dataset.num_episodes}")
    logging.info(f"Total frames: {dataset.num_frames}")


if __name__ == "__main__":
    main()
