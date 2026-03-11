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

"""ROS2 PointCloud2 Publisher Example for Tactile Sensors.

This example demonstrates how to publish tactile data as ROS2 PointCloud2 messages,
which can be consumed by the PointCloud2TactileSensor class.

Usage:
    # Start the publisher
    python ros2_tactile_publisher.py

    # With custom topic and rate
    python ros2_tactile_publisher.py --topic /my_tactile/sensor --fps 60

The published PointCloud2 message contains the following fields:
    - x, y, z: 3D position of each sensing point (in mm)
    - dx, dy, dz: 3D displacement (in mm)
    - fx, fy, fz: 3D force (in N)

This format is compatible with the Tac3D sensor specification:
    - 400 points (20x20 grid)
    - 30 Hz publishing rate
    - Position: tactile sensor surface coordinates
    - Displacement: ±2mm (edge) / +3mm (center) range
    - Force: ±0.8N range
"""

import argparse
import logging
import math
import time
from typing import Any

import numpy as np

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2, PointField
    import struct
    ROS2_AVAILABLE = True
except ImportError as e:
    ROS2_AVAILABLE = False
    ROS2_IMPORT_ERROR = e


class TactilePointCloudPublisher(Node):
    """ROS2 node for publishing tactile data as PointCloud2 messages.

    This node simulates a tactile sensor (like Tac3D) and publishes
    high-resolution tactile data at a configurable rate.

    Attributes:
        publisher: ROS2 publisher for PointCloud2 messages.
        timer: ROS2 timer for periodic publishing.
        num_points: Number of sensing points (default 400 for 20x20 grid).
        fps: Publishing frequency in Hz.
    """

    def __init__(
        self,
        topic_name: str = "/tactile/pointcloud",
        fps: int = 30,
        num_points: int = 400,
        frame_id: str = "tactile_sensor_link",
    ):
        """Initialize the tactile publisher node.

        Args:
            topic_name: ROS2 topic name for PointCloud2 messages.
            fps: Publishing frequency in Hz.
            num_points: Number of sensing points (400 for Tac3D).
            frame_id: TF frame ID for the sensor.
        """
        super().__init__("tactile_publisher")

        self.num_points = num_points
        self.fps = fps
        self.frame_id = frame_id

        # Create publisher with reliable QoS
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.publisher = self.create_publisher(PointCloud2, topic_name, qos)

        # Create timer for periodic publishing
        timer_period = 1.0 / fps
        self.timer = self.create_timer(timer_period, self.publish_callback)

        # Simulation state
        self.frame_count = 0
        self.start_time = time.time()
        self.rng = np.random.default_rng(42)

        # Contact simulation parameters
        self.contact_center = np.array([0.5, 0.5])  # Normalized [0,1] coordinates
        self.contact_force = 0.5  # Newtons
        self.contact_radius = 0.15  # Normalized radius

        # Create grid positions (20x20 for Tac3D)
        self.grid_positions = self._create_grid_positions()

        self.get_logger().info(
            f"Tactile publisher started on topic '{topic_name}' "
            f"with {num_points} points at {fps} Hz"
        )

    def _create_grid_positions(self) -> np.ndarray:
        """Create grid positions for sensing points.

        Returns:
            np.ndarray: Array of shape (num_points, 2) with [x, y] positions.
        """
        grid_size = int(np.sqrt(self.num_points))
        if grid_size * grid_size != self.num_points:
            grid_size = int(np.ceil(np.sqrt(self.num_points)))

        x = np.linspace(-10, 10, grid_size)  # -10mm to +10mm
        y = np.linspace(-10, 10, grid_size)
        xx, yy = np.meshgrid(x, y)
        positions = np.stack([xx.flatten(), yy.flatten()], axis=1)

        return positions[:self.num_points]

    def publish_callback(self) -> None:
        """Publish a PointCloud2 message."""
        # Generate tactile data
        points = self._generate_tactile_data()

        # Create and publish message
        msg = self._create_pointcloud2(points)
        self.publisher.publish(msg)

        self.frame_count += 1

        # Log statistics every 30 frames
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            actual_fps = self.frame_count / elapsed
            self.get_logger().debug(f"Published {self.frame_count} frames, FPS: {actual_fps:.1f}")

    def _generate_tactile_data(self) -> np.ndarray:
        """Generate simulated tactile data.

        Returns:
            np.ndarray: Array of shape (num_points, 9) with columns:
                [x, y, z, dx, dy, dz, fx, fy, fz]
        """
        # Calculate distances from contact center
        normalized_positions = (self.grid_positions + 10) / 20  # Normalize to [0, 1]
        distances = np.linalg.norm(normalized_positions - self.contact_center, axis=1)

        # Generate displacement (Gaussian profile)
        displacement_mag = self.contact_force * np.exp(-(distances**2) / (2 * self.contact_radius**2 + 1e-6))

        displacement = np.zeros((self.num_points, 3), dtype=np.float32)
        displacement[:, 2] = -displacement_mag  # Z displacement (into surface)

        # Add some X/Y displacement
        directions = normalized_positions - self.contact_center
        directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-6)
        displacement[:, 0] = directions[:, 0] * displacement_mag * 0.1
        displacement[:, 1] = directions[:, 1] * displacement_mag * 0.1

        # Generate force (Gaussian distribution)
        force_mag = self.contact_force * np.exp(-(distances**2) / (2 * self.contact_radius**2 + 1e-6))

        force = np.zeros((self.num_points, 3), dtype=np.float32)
        force[:, 2] = force_mag  # Normal force
        force[:, 0] = force_mag * 0.1 * self.rng.normal(0, 1, self.num_points)  # Tangential X
        force[:, 1] = force_mag * 0.1 * self.rng.normal(0, 1, self.num_points)  # Tangential Y

        # Add noise
        displacement += self.rng.normal(0, 0.001, displacement.shape).astype(np.float32)
        force += self.rng.normal(0, 0.0001, force.shape).astype(np.float32)

        # Z positions (flat surface at z=0)
        z = np.zeros((self.num_points, 1), dtype=np.float32)

        # Combine all data
        points = np.concatenate([
            self.grid_positions.astype(np.float32),  # x, y
            z,  # z
            displacement,  # dx, dy, dz
            force,  # fx, fy, fz
        ], axis=1)

        return points

    def _create_pointcloud2(self, points: np.ndarray) -> PointCloud2:
        """Create a PointCloud2 message from tactile data.

        Args:
            points: Array of shape (num_points, 9) with [x, y, z, dx, dy, dz, fx, fy, fz].

        Returns:
            PointCloud2: ROS2 PointCloud2 message.
        """
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.height = 1
        msg.width = len(points)

        # Define fields
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='dx', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='dy', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='dz', offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name='fx', offset=24, datatype=PointField.FLOAT32, count=1),
            PointField(name='fy', offset=28, datatype=PointField.FLOAT32, count=1),
            PointField(name='fz', offset=32, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.point_step = 36  # 9 floats * 4 bytes
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = points.tobytes()

        return msg


def parse_args():
    parser = argparse.ArgumentParser(description="ROS2 Tactile PointCloud2 Publisher")
    parser.add_argument(
        "--topic",
        type=str,
        default="/tactile/pointcloud",
        help="ROS2 topic name for PointCloud2 messages",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Publishing frequency in Hz",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=400,
        help="Number of sensing points (400 for Tac3D)",
    )
    parser.add_argument(
        "--frame-id",
        type=str,
        default="tactile_sensor_link",
        help="TF frame ID for the sensor",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if not ROS2_AVAILABLE:
        logging.error(f"ROS2 not available. Please install rclpy and sensor_msgs: {ROS2_IMPORT_ERROR}")
        return

    # Initialize ROS2
    rclpy.init()

    # Create node
    node = TactilePointCloudPublisher(
        topic_name=args.topic,
        fps=args.fps,
        num_points=args.num_points,
        frame_id=args.frame_id,
    )

    try:
        # Spin the node
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
