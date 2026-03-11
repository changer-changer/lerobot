# LEROBOT Multi-Branch Development

**Repository**: https://github.com/changer-changer/LEROBOT  
**Upstream**: https://github.com/huggingface/lerobot

---

## Branch Structure

| Branch | Description | Status |
|--------|-------------|--------|
| `main` | Sync with HuggingFace LeRobot | ✅ Active |
| `feature/pointcloud` | LeRobot + Tac3D tactile sensor | ✅ Verified |
| `feature/tron2` | LeRobot + Tron2 robot support | ✅ Verified |
| `feature/integrate` | Tron2 + Tac3D combined | ✅ Active |
| `feature/visuotactile` | Research innovations | 🔄 Dev |
| `feature/baseline` | Baseline models | 🔄 Dev |

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/changer-changer/LEROBOT.git
cd LEROBOT

# Choose branch
git checkout feature/integrate  # For full functionality

# Install dependencies
pip install -e .
```

---

## Branch Details

### main
- Clean HuggingFace LeRobot
- Only for syncing upstream updates

### feature/pointcloud
- LeRobot with Tac3D tactile sensor support
- Contains: `src/lerobot/tactile/`

### feature/tron2
- LeRobot with Tron2 robot support
- Contains: `src/lerobot/robots/tron2/`

### feature/integrate
- Combines Tron2 robot + Tac3D sensor
- Ready for data collection

### feature/visuotactile
- Research branch for our innovations
- Will contain: PSTE, Phase-Gating, VT-CAF

### feature/baseline
- Baseline model implementations
- For comparison experiments

---

## Git Workflow

```bash
# Sync upstream
git checkout main
git fetch upstream
git merge upstream/main
git push origin main

# Develop new feature
git checkout -b feature/your-feature
# ... develop ...
git push origin feature/your-feature
```

---

## Notes

- Each branch is a complete LeRobot installation
- feature/pointcloud and feature/tron2 are verified working versions
- feature/integrate combines both functionalities
- Target alignment: Essential for collaboration

---

*Last Updated: 2026-03-11*
