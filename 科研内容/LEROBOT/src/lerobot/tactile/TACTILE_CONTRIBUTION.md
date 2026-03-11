# Tactile Sensor Contribution for LeRobot

This contribution adds comprehensive tactile sensor support to the LeRobot framework, enabling high-resolution tactile data collection, storage, and integration with robot learning pipelines.

## Overview

The tactile sensor module provides:

- **Tactile Sensor Interface**: Abstract base class following LeRobot's camera design pattern
- **Tac3D Sensor Support**: Native integration with Tac3D high-resolution tactile sensors
- **ROS2 PointCloud2 Interface**: Generic support for ROS2-based tactile sensors
- **Simulated Sensor**: For testing and development without hardware
- **Dataset Integration**: Full integration with LeRobot's dataset format
- **Data Storage**: Efficient storage in Parquet format with compression
- **Statistics Computation**: Automatic calculation of dataset statistics

## Hardware Support

### Tac3D Sensor Specifications

- **Array Size**: 20×20 = 400 sensing points
- **Data Format**: 400×6 matrix
  - Columns 0-2: 3D displacement field (dx, dy, dz) in mm
    - Range: ±2mm (edge), +3mm (center)
    - Noise: 1.5μm (XY), 6μm (Z)
  - Columns 3-5: 3D force distribution (Fx, Fy, Fz) in N
    - Range: ±0.8N
    - Noise: 0.4mN (XY), 1mN (Z)
- **Frequency**: 30 Hz
- **Interface**: UDP communication with Tac3D-Desktop software

## Installation

The tactile module is included in the base LeRobot installation. No additional dependencies are required for basic functionality.

### Optional Dependencies

For ROS2 PointCloud2 support:
```bash
pip install rclpy sensor_msgs
```

For Tac3D support:
```bash
# Copy Tac3D-SDK to your workspace and add to PYTHONPATH
export PYTHONPATH="/path/to/Tac3D-SDK/Tac3D-API/python:$PYTHONPATH"
```

## Quick Start

### Using Simulated Sensor

```python
from lerobot.tactile import SimulatedTactileSensor, SimulatedTactileSensorConfig

# Create sensor
config = SimulatedTactileSensorConfig(fps=30, num_points=400)
sensor = SimulatedTactileSensor(config)

# Use sensor
with sensor:
    sensor.tare()  # Perform zeroing calibration
    data = sensor.read()  # Get tactile data (400, 6)
    print(f"Data shape: {data.shape}")
    print(f"Displacement range: [{data[:, :3].min():.3f}, {data[:, :3].max():.3f}] mm")
    print(f"Force range: [{data[:, 3:].min():.3f}, {data[:, 3:].max():.3f}] N")
```

### Using Tac3D Sensor

```python
from lerobot.tactile import Tac3DTactileSensor, Tac3DSensorConfig

# Configure sensor
config = Tac3DSensorConfig(
    udp_port=9988,
    data_type="full",
    tare_on_startup=True,
)

# Create and use sensor
sensor = Tac3DTactileSensor(config)
with sensor:
    data = sensor.read()  # (400, 6) array
```

### Using ROS2 PointCloud2 Sensor

```python
from lerobot.tactile import PointCloud2TactileSensor, PointCloud2SensorConfig

# Configure sensor
config = PointCloud2SensorConfig(
    topic_name="/tactile/pointcloud",
    fps=30,
)

# Create and use sensor
sensor = PointCloud2TactileSensor(config)
with sensor:
    data = sensor.read()
```

## Recording Data with LeRobot

```python
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.tactile import (
    Tac3DTactileSensor,
    Tac3DSensorConfig,
    create_tactile_features,
    merge_tactile_features,
)

# Create sensor
sensor_config = Tac3DSensorConfig()
sensor = Tac3DTactileSensor(sensor_config)

# Create dataset features
base_features = robot.observation_features  # From your robot
tactile_features = create_tactile_features(sensor_config)
all_features = merge_tactile_features(base_features, sensor_config)

# Create dataset
dataset = LeRobotDataset.create(
    repo_id="username/dataset_name",
    fps=30,
    features=all_features,
    robot_type=robot.robot_type,
)

# Record data
with sensor:
    sensor.tare()
    
    for episode in range(num_episodes):
        for frame in range(frames_per_episode):
            # Get robot observation
            obs = robot.get_observation()
            action = robot.get_action()
            
            # Get tactile data
            tactile_data = sensor.read()
            
            # Build frame
            frame_data = {
                **obs,
                "observation.tactile": tactile_data.astype(np.float32),
                "action": action,
                "task": task,
            }
            
            dataset.add_frame(frame_data)
        
        dataset.save_episode()

dataset.finalize()
```

## Examples

See the `examples/` directory for complete working examples:

- `record_tactile_data.py`: Record tactile data with LeRobot dataset
- `replay_tactile_data.py`: Replay and visualize stored data
- `visualize_tactile_live.py`: Real-time visualization of tactile data
- `ros2_tactile_publisher.py`: ROS2 PointCloud2 publisher example

## Architecture

### Module Structure

```
lerobot/tactile/
├── __init__.py              # Module exports
├── configs.py               # Configuration classes
├── tactile_sensor.py        # Abstract base class
├── tac3d_sensor.py          # Tac3D implementation
├── pointcloud2_sensor.py    # ROS2 PointCloud2 implementation
├── simulated_sensor.py      # Simulated sensor
└── dataset_integration.py   # Dataset utilities
```

### Design Principles

1. **Consistency with LeRobot**: Follows the same design patterns as cameras
2. **Configurable**: Supports different sensor specifications (points, dimensions)
3. **Extensible**: Easy to add new sensor types
4. **Type-Safe**: Full type annotations throughout
5. **Well-Documented**: Comprehensive docstrings and examples

### Data Flow

```
[Tactile Sensor] → [read()] → [Tactile Data (N, D)]
                                    ↓
[Dataset Integration] → [tactile_to_frame()] → [Dataset Frame]
                                    ↓
[LeRobotDataset] → [add_frame()] → [Parquet Storage]
                                    ↓
[Statistics] → [compute_tactile_stats()] → [stats.json]
```

## Testing

Run the test suite:

```bash
pytest tests/tactile/ -v
```

## License

This contribution follows the same license as LeRobot (Apache 2.0).

## Citation

If you use this tactile sensor module in your research, please cite:

```bibtex
@software{lerobot,
  title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
  author = {Cadène, Rémi and Alibert, Simon and Soare, Alexander and others},
  year = {2024},
  url = {https://github.com/huggingface/lerobot}
}
```

## Acknowledgments

- Tac3D sensor integration based on Tac3D-SDK v3.3.0
- ROS2 PointCloud2 interface follows ROS2 Humble standards
- Design inspired by LeRobot's camera module architecture
