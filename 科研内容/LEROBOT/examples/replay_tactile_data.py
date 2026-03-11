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

"""Example: Replay and visualize tactile data from LeRobot dataset.

This example demonstrates how to load and visualize tactile data from a
previously recorded LeRobot dataset.

Usage:
    python replay_tactile_data.py --dataset my_dataset --episode 0

    # Visualize all episodes
    python replay_tactile_data.py --dataset my_dataset --all-episodes
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.tactile.dataset_integration import normalize_tactile_data


def parse_args():
    parser = argparse.ArgumentParser(description="Replay tactile data from LeRobot dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset repository ID or path",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory for dataset",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to replay",
    )
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Replay all episodes",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization (requires matplotlib)",
    )
    parser.add_argument(
        "--export-video",
        type=str,
        default=None,
        help="Export visualization to video file",
    )
    return parser.parse_args()


def visualize_tactile_frame(data: np.ndarray, title: str = "Tactile Frame"):
    """Visualize a single tactile frame.

    Args:
        data: Tactile data array (400, 6) with [dx, dy, dz, Fx, Fy, Fz].
        title: Plot title.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        logging.error("Visualization requires matplotlib: pip install matplotlib")
        return

    # Reshape to 20x20 grid
    grid_size = 20

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)

    # Displacement components
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(data[:, 0].reshape(grid_size, grid_size), cmap='RdBu_r')
    ax1.set_title('Displacement X (mm)')
    plt.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(data[:, 1].reshape(grid_size, grid_size), cmap='RdBu_r')
    ax2.set_title('Displacement Y (mm)')
    plt.colorbar(im2, ax=ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(data[:, 2].reshape(grid_size, grid_size), cmap='RdBu_r')
    ax3.set_title('Displacement Z (mm)')
    plt.colorbar(im3, ax=ax3)

    # Force components
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(data[:, 3].reshape(grid_size, grid_size), cmap='hot')
    ax4.set_title('Force X (N)')
    plt.colorbar(im4, ax=ax4)

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(data[:, 4].reshape(grid_size, grid_size), cmap='hot')
    ax5.set_title('Force Y (N)')
    plt.colorbar(im5, ax=ax5)

    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(data[:, 5].reshape(grid_size, grid_size), cmap='hot')
    ax6.set_title('Force Z (N)')
    plt.colorbar(im6, ax=ax6)

    fig.suptitle(title)
    plt.tight_layout()

    return fig


def replay_episode(dataset: LeRobotDataset, episode_idx: int, visualize: bool = False):
    """Replay a single episode from the dataset.

    Args:
        dataset: LeRobotDataset instance.
        episode_idx: Episode index to replay.
        visualize: Whether to show visualization.
    """
    logging.info(f"\n=== Replaying Episode {episode_idx} ===")

    # Get episode data
    episode_data = dataset.hf_dataset

    # Find frames for this episode
    episode_mask = episode_data['episode_index'] == episode_idx
    episode_frames = episode_data[episode_mask]

    num_frames = len(episode_frames['index'])
    logging.info(f"Episode has {num_frames} frames")

    # Get tactile key
    tactile_key = None
    for key in dataset.features.keys():
        if 'tactile' in key:
            tactile_key = key
            break

    if tactile_key is None:
        logging.error("No tactile data found in dataset")
        return

    logging.info(f"Tactile data key: {tactile_key}")
    logging.info(f"Tactile data shape: {dataset.features[tactile_key]['shape']}")

    # Replay frames
    for frame_idx in range(num_frames):
        # Get tactile data for this frame
        tactile_data = episode_frames[tactile_key][frame_idx]

        if isinstance(tactile_data, np.ndarray):
            logging.info(f"Frame {frame_idx}: tactile shape={tactile_data.shape}, "
                        f"disp_range=[{tactile_data[:, :3].min():.3f}, {tactile_data[:, :3].max():.3f}], "
                        f"force_range=[{tactile_data[:, 3:].min():.3f}, {tactile_data[:, 3:].max():.3f}]")

            if visualize and frame_idx % 10 == 0:  # Visualize every 10th frame
                fig = visualize_tactile_frame(tactile_data, f"Episode {episode_idx}, Frame {frame_idx}")
                plt.show()
        else:
            logging.info(f"Frame {frame_idx}: tactile data type={type(tactile_data)}")

    # Compute episode statistics
    if isinstance(episode_frames[tactile_key][0], np.ndarray):
        all_tactile = np.stack([episode_frames[tactile_key][i] for i in range(num_frames)])
        logging.info(f"\nEpisode {episode_idx} Statistics:")
        logging.info(f"  Displacement - Mean: {all_tactile[:, :, :3].mean():.4f}, "
                    f"Std: {all_tactile[:, :, :3].std():.4f}")
        logging.info(f"  Force - Mean: {all_tactile[:, :, 3:].mean():.4f}, "
                    f"Std: {all_tactile[:, :, 3:].std():.4f}")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Load dataset
    logging.info(f"Loading dataset: {args.dataset}")
    dataset = LeRobotDataset(
        repo_id=args.dataset,
        root=args.root,
    )

    logging.info(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
    logging.info(f"Features: {list(dataset.features.keys())}")

    # Replay episodes
    if args.all_episodes:
        for episode_idx in range(dataset.num_episodes):
            replay_episode(dataset, episode_idx, args.visualize)
    else:
        if args.episode >= dataset.num_episodes:
            logging.error(f"Episode {args.episode} not found. Dataset has {dataset.num_episodes} episodes.")
            return
        replay_episode(dataset, args.episode, args.visualize)


if __name__ == "__main__":
    main()
