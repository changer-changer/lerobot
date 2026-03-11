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

"""Example: Real-time visualization of tactile sensor data.

This example demonstrates how to visualize tactile data in real-time from
a connected sensor.

Usage:
    # With Tac3D sensor
    python visualize_tactile_live.py --sensor-type tac3d --udp-port 9988

    # With simulated sensor
    python visualize_tactile_live.py --sensor-type simulated

    # Save visualization to video
    python visualize_tactile_live.py --sensor-type simulated --save-video output.mp4
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from lerobot.tactile import (
    Tac3DSensorConfig,
    Tac3DTactileSensor,
    SimulatedTactileSensorConfig,
    SimulatedTactileSensor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize tactile data in real-time")
    parser.add_argument(
        "--sensor-type",
        type=str,
        choices=["tac3d", "simulated"],
        default="simulated",
        help="Type of tactile sensor to use",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=9988,
        help="UDP port for Tac3D sensor",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Visualization duration in seconds",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Save visualization to video file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Visualization frame rate",
    )
    return parser.parse_args()


def create_visualizer():
    """Create matplotlib visualization setup."""
    try:
        import matplotlib
        matplotlib.use('Qt5Agg' if not args.save_video else 'Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        return plt, GridSpec, FuncAnimation, FFMpegWriter
    except ImportError as e:
        logging.error(f"Visualization requires matplotlib: {e}")
        raise


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


def visualize_live(sensor, duration: float, save_video: str | None = None, fps: int = 30):
    """Visualize tactile data in real-time.

    Args:
        sensor: Connected tactile sensor.
        duration: Visualization duration in seconds.
        save_video: Optional video file path to save.
        fps: Frame rate for visualization.
    """
    plt, GridSpec, FuncAnimation, FFMpegWriter = create_visualizer()

    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)

    # Create subplots
    axes = []
    images = []
    grid_size = 20

    titles = [
        'Displacement X (mm)',
        'Displacement Y (mm)',
        'Displacement Z (mm)',
        'Force X (N)',
        'Force Y (N)',
        'Force Z (N)',
    ]
    cmaps = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'hot', 'hot', 'hot']

    for i, (title, cmap) in enumerate(zip(titles, cmaps)):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        im = ax.imshow(np.zeros((grid_size, grid_size)), cmap=cmap, animated=True)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        axes.append(ax)
        images.append(im)

    # Add statistics text
    stats_text = fig.text(0.5, 0.02, "", ha='center', fontsize=10)

    fig.suptitle(f"Tactile Sensor Visualization ({sensor.__class__.__name__})")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Data buffer for statistics
    frame_times = []
    start_time = time.time()

    def update(frame):
        nonlocal start_time

        # Read sensor data
        try:
            data = sensor.read_latest(max_age_ms=100)
        except Exception:
            data = sensor.read()

        frame_start = time.time()

        # Update displacement plots (first 3 channels)
        for i in range(3):
            grid_data = data[:, i].reshape(grid_size, grid_size)
            images[i].set_array(grid_data)
            images[i].set_clim(vmin=grid_data.min(), vmax=grid_data.max())

        # Update force plots (last 3 channels)
        for i in range(3):
            grid_data = data[:, i + 3].reshape(grid_size, grid_size)
            images[i + 3].set_array(grid_data)
            images[i + 3].set_clim(vmin=grid_data.min(), vmax=grid_data.max())

        # Update statistics
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        if len(frame_times) > 30:
            frame_times.pop(0)

        avg_frame_time = np.mean(frame_times)
        actual_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        stats = sensor.get_statistics()
        stats_str = (f"FPS: {actual_fps:.1f} | "
                    f"Frames: {stats.get('frame_count', 0)} | "
                    f"Disp: [{data[:, :3].min():.3f}, {data[:, :3].max():.3f}] | "
                    f"Force: [{data[:, 3:].min():.3f}, {data[:, 3:].max():.3f}]")
        stats_text.set_text(stats_str)

        # Check duration
        elapsed = time.time() - start_time
        if elapsed >= duration:
            plt.close(fig)
            return images + [stats_text]

        return images + [stats_text]

    # Create animation
    interval = int(1000 / fps)  # milliseconds
    anim = FuncAnimation(fig, update, interval=interval, blit=True, cache_frame_data=False)

    if save_video:
        logging.info(f"Saving video to {save_video}...")
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='LeRobot Tactile'))
        anim.save(save_video, writer=writer)
        logging.info("Video saved")
    else:
        plt.show()


def main():
    global args
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Create sensor
    logging.info(f"Creating {args.sensor_type} sensor...")
    sensor, config = create_sensor(args.sensor_type, args.udp_port)

    try:
        # Connect to sensor
        sensor.connect(warmup=True)
        logging.info("Sensor connected")

        # For simulated sensor, set up some contact
        if args.sensor_type == "simulated":
            sensor.set_contact_center([0.5, 0.5])
            sensor.set_contact_force(0.5)
            sensor.set_contact_radius(0.15)

        # Run visualization
        logging.info(f"Starting visualization for {args.duration} seconds...")
        visualize_live(sensor, args.duration, args.save_video, args.fps)

    except KeyboardInterrupt:
        logging.info("\nVisualization interrupted by user")
    finally:
        sensor.disconnect()
        logging.info("Sensor disconnected")

    # Print final statistics
    stats = sensor.get_statistics()
    logging.info(f"\nFinal Statistics:")
    for key, value in stats.items():
        logging.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
