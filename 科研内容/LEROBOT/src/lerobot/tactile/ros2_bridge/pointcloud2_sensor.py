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

"""ROS2 PointCloud2 tactile sensor implementation for LeRobot.

This module provides the PointCloud2TactileSensor class for interfacing with
generic ROS2-based tactile sensors that publish PointCloud2 messages.

Example ROS2 Publisher Node:
    ```python
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    import numpy as np
    import struct

    class TactilePublisher(Node):
        def __init__(self):
            super().__init__('tactile_publisher')
            self.publisher = self.create_publisher(PointCloud2, '/tactile/pointcloud', 10)
            self.timer = self.create_timer(1.0/30.0, self.publish_callback)  # 30 Hz
            self.num_points = 400

        def publish_callback(self):
            # Create point cloud with 400 points
            # Fields: x, y, z, displacement_x, displacement_y, displacement_z, force_x, force_y, force_z
            points = np.random.randn(self.num_points, 9).astype(np.float32)
            msg = self.create_pointcloud2(points)
            self.publisher.publish(msg)

        def create_pointcloud2(self, points):
            msg = PointCloud2()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'tactile_sensor_link'
            msg.height = 1
            msg.width = len(points)
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

    def main():
        rclpy.init()
        node = TactilePublisher()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()

    if __name__ == '__main__':
        main()
    ```
"""

import logging
import struct
import threading
import time
from queue import Empty, Queue
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.tactile.configs import PointCloud2SensorConfig, TactileDataType
from lerobot.tactile.tactile_sensor import TactileSensor

# Try to import ROS2
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2
    ROS2_AVAILABLE = True
except ImportError as e:
    ROS2_AVAILABLE = False
    ROS2_IMPORT_ERROR = e


class PointCloud2TactileSensor(TactileSensor):
    """ROS2 PointCloud2 tactile sensor implementation.

    This sensor subscribes to ROS2 PointCloud2 messages containing tactile data.
    It supports various point cloud formats and can extract displacement and force
    information from the point fields.

    Expected PointCloud2 fields:
        - Basic: x, y, z (position)
        - With displacement: x, y, z, dx, dy, dz
        - With force: x, y, z, fx, fy, fz
        - Full: x, y, z, dx, dy, dz, fx, fy, fz

    Attributes:
        config: PointCloud2SensorConfig with ROS2-specific parameters.
        is_connected: Whether the ROS2 node is active and receiving data.

    Example:
        ```python
        config = PointCloud2SensorConfig(
            topic_name="/tactile/pointcloud",
            node_name="tactile_sensor",
            fps=30
        )
        sensor = PointCloud2TactileSensor(config)

        with sensor:
            data = sensor.read()  # Get tactile data
        ```
    """

    # Standard field name mappings
    FIELD_MAPPINGS = {
        # Displacement fields
        'dx': ['dx', 'displacement_x', 'd_x', 'delta_x'],
        'dy': ['dy', 'displacement_y', 'd_y', 'delta_y'],
        'dz': ['dz', 'displacement_z', 'd_z', 'delta_z'],
        # Force fields
        'fx': ['fx', 'force_x', 'f_x'],
        'fy': ['fy', 'force_y', 'f_y'],
        'fz': ['fz', 'force_z', 'f_z'],
        # Position fields
        'x': ['x', 'X', 'position_x'],
        'y': ['y', 'Y', 'position_y'],
        'z': ['z', 'Z', 'position_z'],
    }

    def __init__(self, config: PointCloud2SensorConfig):
        """Initialize the PointCloud2 tactile sensor.

        Args:
            config: PointCloud2SensorConfig containing ROS2 parameters.

        Raises:
            ImportError: If ROS2 is not available.
        """
        if not ROS2_AVAILABLE:
            raise ImportError(
                "ROS2 not available. Please install rclpy and sensor_msgs. "
                f"Import error: {ROS2_IMPORT_ERROR}"
            )

        super().__init__(config)
        self.config: PointCloud2SensorConfig = config

        # ROS2 node and subscription
        self._node: Node | None = None
        self._subscription: Any | None = None

        # Frame buffer
        self._frame_queue: Queue[NDArray[np.float64]] = Queue(maxsize=10)
        self._latest_frame: NDArray[np.float64] | None = None
        self._frame_lock = threading.Lock()

        # Field mapping from received messages
        self._field_mapping: dict[str, int] = {}
        self._point_step: int = 0
        self._is_bigendian: bool = False

        # Statistics
        self._frame_count = 0
        self._dropped_frames = 0
        self._start_time: float | None = None

        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def is_connected(self) -> bool:
        """Check if the ROS2 node is active.

        Returns:
            bool: True if node is initialized and spinning.
        """
        return self._node is not None and rclpy.ok()

    @staticmethod
    def find_sensors() -> list[dict[str, Any]]:
        """Detect available ROS2 tactile sensors.

        Returns:
            list[dict]: List of available ROS2 topics with PointCloud2 type.
        """
        if not ROS2_AVAILABLE:
            return []

        sensors = []
        try:
            # Check if ROS2 is running
            if not rclpy.ok():
                rclpy.init()

            node = rclpy.create_node('_sensor_discovery')
            topic_names_and_types = node.get_topic_names_and_types()

            for topic_name, topic_types in topic_names_and_types:
                if 'sensor_msgs/msg/PointCloud2' in topic_types:
                    sensors.append({
                        'id': topic_name,
                        'type': 'ros2_pointcloud2',
                        'topic': topic_name,
                    })

            node.destroy_node()
        except Exception as e:
            logging.getLogger('PointCloud2TactileSensor').warning(
                f"Could not discover ROS2 topics: {e}"
            )

        return sensors

    def connect(self, warmup: bool = True) -> None:
        """Initialize ROS2 node and subscribe to topic.

        Args:
            warmup: If True, waits for the first message before returning.

        Raises:
            ConnectionError: If ROS2 initialization fails.
            TimeoutError: If warmup message is not received in time.
        """
        if self.is_connected:
            self._logger.warning("Sensor already connected")
            return

        try:
            self._logger.info(
                f"Connecting to ROS2 topic {self.config.topic_name}..."
            )

            # Initialize ROS2 if needed
            if not rclpy.ok():
                rclpy.init()

            # Create node
            self._node = rclpy.create_node(self.config.node_name)

            # Set up QoS profile
            reliability = (
                ReliabilityPolicy.RELIABLE
                if self.config.qos_profile == 'reliable'
                else ReliabilityPolicy.BEST_EFFORT
            )
            qos = QoSProfile(
                depth=self.config.queue_size,
                reliability=reliability,
            )

            # Create subscription
            self._subscription = self._node.create_subscription(
                PointCloud2,
                self.config.topic_name,
                self._on_message_received,
                qos,
            )

            # Start spinning in background thread
            self._spin_thread = threading.Thread(target=self._spin, daemon=True)
            self._spin_thread.start()

            self._start_time = time.time()
            self._logger.info("ROS2 node started successfully")

            if warmup:
                self._logger.info("Waiting for warmup message...")
                self._wait_for_frame(timeout_ms=self.config.timeout_ms)

            if self.config.apply_tare:
                self.tare()

        except Exception as e:
            self.disconnect()
            raise ConnectionError(f"Failed to connect to ROS2: {e}") from e

    def disconnect(self) -> None:
        """Shutdown ROS2 node and release resources."""
        self._logger.info("Disconnecting from ROS2...")

        if self._node:
            try:
                self._node.destroy_node()
            except Exception as e:
                self._logger.warning(f"Error destroying node: {e}")
            self._node = None

        # Clear buffers
        with self._frame_lock:
            self._latest_frame = None
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break

        self._logger.info("ROS2 node disconnected")

    def read(self) -> NDArray[np.float64]:
        """Read a single tactile frame synchronously.

        Returns:
            np.ndarray: Tactile data with shape (num_points, data_dim).

        Raises:
            ConnectionError: If sensor is not connected.
            TimeoutError: If frame is not received within timeout.
        """
        if not self.is_connected:
            raise ConnectionError("Sensor is not connected")

        return self._get_frame_blocking(timeout_ms=self.config.timeout_ms)

    def async_read(self, timeout_ms: float = 1000.0) -> NDArray[np.float64]:
        """Read the most recent tactile frame.

        Args:
            timeout_ms: Maximum wait time in milliseconds.

        Returns:
            np.ndarray: Tactile data with shape (num_points, data_dim).

        Raises:
            TimeoutError: If no new frame arrives within timeout.
        """
        if not self.is_connected:
            raise ConnectionError("Sensor is not connected")

        return self._get_frame_blocking(timeout_ms=timeout_ms)

    def read_latest(self, max_age_ms: int = 500) -> NDArray[np.float64]:
        """Get the latest frame immediately (non-blocking).

        Args:
            max_age_ms: Maximum acceptable frame age in milliseconds.

        Returns:
            np.ndarray: The latest tactile data frame.

        Raises:
            TimeoutError: If frame is older than max_age_ms.
            RuntimeError: If no frames have been received.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                raise RuntimeError("No frames received yet")

            if self._start_time:
                frame_age_ms = (time.time() - self._start_time) * 1000
                if frame_age_ms > max_age_ms:
                    raise TimeoutError(
                        f"Latest frame is {frame_age_ms:.1f}ms old (max: {max_age_ms}ms)"
                    )

            return self._latest_frame.copy()

    def get_statistics(self) -> dict[str, Any]:
        """Get sensor statistics.

        Returns:
            dict: Dictionary containing frame count and connection status.
        """
        elapsed = time.time() - self._start_time if self._start_time else 0
        fps = self._frame_count / elapsed if elapsed > 0 else 0

        return {
            "frame_count": self._frame_count,
            "dropped_frames": self._dropped_frames,
            "elapsed_time": elapsed,
            "average_fps": fps,
            "is_connected": self.is_connected,
            "is_tared": self.is_tared,
            "topic": self.config.topic_name,
        }

    def _spin(self) -> None:
        """ROS2 spin loop running in background thread."""
        while rclpy.ok() and self._node:
            try:
                rclpy.spin_once(self._node, timeout_sec=0.01)
            except Exception as e:
                self._logger.error(f"Error in ROS2 spin: {e}")
                break

    def _on_message_received(self, msg: "PointCloud2") -> None:
        """Callback for ROS2 PointCloud2 messages.

        Args:
            msg: ROS2 PointCloud2 message.
        """
        try:
            # Parse field mapping on first message
            if not self._field_mapping:
                self._parse_field_mapping(msg)

            # Extract data from message
            data = self._extract_data(msg)

            # Apply tare if active
            data = self._apply_tare(data)

            # Update buffers
            with self._frame_lock:
                self._latest_frame = data

            if self._frame_queue.full():
                self._dropped_frames += 1
                try:
                    self._frame_queue.get_nowait()
                except Empty:
                    pass

            self._frame_queue.put(data)
            self._frame_count += 1

        except Exception as e:
            self._logger.error(f"Error processing message: {e}")

    def _parse_field_mapping(self, msg: "PointCloud2") -> None:
        """Parse field names and offsets from PointCloud2 message.

        Args:
            msg: PointCloud2 message to parse.
        """
        self._field_mapping = {}
        self._point_step = msg.point_step
        self._is_bigendian = msg.is_bigendian

        for field in msg.fields:
            field_name = field.name.lower()
            self._field_mapping[field_name] = {
                'offset': field.offset,
                'datatype': field.datatype,
                'count': field.count,
            }

        self._logger.debug(f"Field mapping: {self._field_mapping}")

    def _extract_data(self, msg: "PointCloud2") -> NDArray[np.float64]:
        """Extract tactile data from PointCloud2 message.

        Args:
            msg: PointCloud2 message.

        Returns:
            np.ndarray: Extracted tactile data.

        Raises:
            ValueError: If required fields are missing.
        """
        num_points = msg.width * msg.height
        data = np.zeros((num_points, self.data_dim), dtype=np.float64)

        # Determine which fields to extract based on data_type
        if self.config.data_type == TactileDataType.DISPLACEMENT:
            field_names = ['dx', 'dy', 'dz']
        elif self.config.data_type == TactileDataType.FORCE:
            field_names = ['fx', 'fy', 'fz']
        else:  # TactileDataType.FULL
            field_names = ['dx', 'dy', 'dz', 'fx', 'fy', 'fz']

        # Extract each field
        for i, field_name in enumerate(field_names):
            field_info = self._find_field(field_name)
            if field_info is None:
                raise ValueError(f"Required field '{field_name}' not found in PointCloud2")

            values = self._extract_field(msg, field_info, num_points)
            data[:, i] = values

        return data

    def _find_field(self, field_name: str) -> dict | None:
        """Find field info by name using field mappings.

        Args:
            field_name: Name of the field to find.

        Returns:
            dict | None: Field information or None if not found.
        """
        # Get possible names for this field
        possible_names = self.FIELD_MAPPINGS.get(field_name, [field_name])

        # Search in field mapping
        for name in possible_names:
            if name in self._field_mapping:
                return self._field_mapping[name]
            # Try lowercase
            if name.lower() in self._field_mapping:
                return self._field_mapping[name.lower()]

        return None

    def _extract_field(self, msg: "PointCloud2", field_info: dict, num_points: int) -> NDArray[np.float64]:
        """Extract a single field from PointCloud2 message.

        Args:
            msg: PointCloud2 message.
            field_info: Field mapping information.
            num_points: Number of points in the cloud.

        Returns:
            np.ndarray: Extracted field values.
        """
        offset = field_info['offset']
        datatype = field_info['datatype']

        # Datatype sizes (in bytes)
        datatype_sizes = {
            1: 1,   # INT8
            2: 1,   # UINT8
            3: 2,   # INT16
            4: 2,   # UINT16
            5: 4,   # INT32
            6: 4,   # UINT32
            7: 4,   # FLOAT32
            8: 8,   # FLOAT64
        }

        size = datatype_sizes.get(datatype, 4)
        fmt = self._get_struct_format(datatype, self._is_bigendian)

        values = np.zeros(num_points, dtype=np.float64)

        for i in range(num_points):
            point_offset = i * self._point_step + offset
            point_bytes = msg.data[point_offset:point_offset + size]

            if len(point_bytes) < size:
                values[i] = 0.0
                continue

            if fmt:
                value = struct.unpack(fmt, point_bytes)[0]
                values[i] = float(value)
            else:
                values[i] = 0.0

        return values

    def _get_struct_format(self, datatype: int, is_bigendian: bool) -> str | None:
        """Get struct format string for datatype.

        Args:
            datatype: ROS2 PointField datatype.
            is_bigendian: Whether data is big-endian.

        Returns:
            str | None: Struct format string or None if unsupported.
        """
        endian = '>' if is_bigendian else '<'

        formats = {
            1: 'b',   # INT8
            2: 'B',   # UINT8
            3: 'h',   # INT16
            4: 'H',   # UINT16
            5: 'i',   # INT32
            6: 'I',   # UINT32
            7: 'f',   # FLOAT32
            8: 'd',   # FLOAT64
        }

        fmt = formats.get(datatype)
        return f"{endian}{fmt}" if fmt else None

    def _get_frame_blocking(self, timeout_ms: float) -> NDArray[np.float64]:
        """Get a frame from the queue with timeout.

        Args:
            timeout_ms: Maximum wait time in milliseconds.

        Returns:
            np.ndarray: Tactile data frame.

        Raises:
            TimeoutError: If no frame arrives within timeout.
        """
        try:
            return self._frame_queue.get(timeout=timeout_ms / 1000.0)
        except Empty:
            raise TimeoutError(f"No frame received within {timeout_ms}ms")

    def _wait_for_frame(self, timeout_ms: float) -> None:
        """Wait for the first frame to arrive.

        Args:
            timeout_ms: Maximum wait time in milliseconds.

        Raises:
            TimeoutError: If no frame arrives within timeout.
        """
        start = time.time()
        while self._latest_frame is None:
            if (time.time() - start) * 1000 > timeout_ms:
                raise TimeoutError(f"No frame received within {timeout_ms}ms")
            time.sleep(0.01)
