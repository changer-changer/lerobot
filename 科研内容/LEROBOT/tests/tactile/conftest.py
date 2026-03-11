"""Pytest configuration for tactile sensor tests."""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

# Add lerobot to path (for development)
lerobot_dir = Path(__file__).parent.parent.parent / "lerobot" / "src"
if lerobot_dir.exists():
    sys.path.insert(0, str(lerobot_dir))
