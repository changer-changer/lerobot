# Visuotactile Diffusion Policy

视触觉融合扩散策略 - 基于LeRobot的多模态机器人操作框架

**Repository**: https://github.com/changer-changer/lerobot

**创新点**: Tac3D 6D点云 + RGB视觉 + Diffusion Policy 最优融合

---

## 🎯 项目概述

本项目实现了视触觉融合的机器人操作策略，针对精密装配任务，创新性地融合了：

- **Tac3D触觉传感器** (400点 6D特征: 位移+力)
- **RGB视觉** (单视角/多视角)
- **Diffusion Policy** (扩散策略)

### 核心创新

1. **Tac3D-PSTE** (创新点A): 物理感知双流编码器
2. **Phase-Aware Gating** (创新点B): 阶段感知模态门控
3. **Multi-Rate Fusion** (创新点C): 多速率数据融合
4. **VT-CAF** (创新点D): 视触觉Cross-Attention融合

---

## 📁 文件结构

```
lerobot/
├── baselines/
│   ├── 3d_diffusion_policy/      # DP3基线
│   └── diffusion_policy/         # 标准Diffusion Policy
│
└── visuotactile_diffusion_policy/  # 本项目 ⭐
    ├── README.md                    # 本文件
    ├── visuotactile_policy.py       # 完整模型
    │
    ├── innovation_A_tac3d_encoder.py    # 创新点A: Tac3D编码
    ├── innovation_B_phase_gating.py     # 创新点B: 阶段门控
    ├── innovation_C_multi_rate.py       # 创新点C: 多速率
    ├── innovation_D_cross_attention.py  # 创新点D: Cross-Attention
    │
    ├── baseline_models.py           # 消融实验基线模型
    └── ablation_study.py            # 消融实验脚本
```

---

## 🚀 快速开始

### 安装依赖

```bash
# 基础依赖
pip install torch torchvision

# LeRobot依赖 (如果需要)
cd ~/lerobot
pip install -e .
```

### 测试模型

```bash
cd /home/cuizhixing/.openclaw/workspace-scientist/lerobot/visuotactile_diffusion_policy

# 测试完整模型
python3 visuotactile_policy.py

# 测试消融实验
python3 ablation_study.py
```

### 使用模型

```python
from visuotactile_policy import VisuotactileDiffusionPolicy, VisuotactileDiffusionPolicyConfig

# 方式1: 使用配置类创建模型
model = VisuotactileDiffusionPolicyConfig.full_system(action_dim=7)

# 方式2: 手动配置
model = VisuotactileDiffusionPolicy(
    action_dim=7,
    use_tac3d_pste=True,      # 使用Tac3D双流编码
    use_phase_gating=True,    # 使用阶段门控
    use_vt_caf=True,          # 使用Cross-Attention融合
)

# 前向传播
import torch
rgb = torch.randn(4, 3, 224, 224)      # [B, C, H, W]
tac3d = torch.randn(4, 400, 6)          # [B, 400, 6] (dx,dy,dz,fx,fy,fz)

action, features = model(rgb, tac3d, return_features=True)
# action: [4, 7]
```

---

## 🔬 消融实验

### 运行消融实验

```bash
python3 ablation_study.py
```

### 消融配置

| 配置 | 描述 | 用途 |
|-----|-----|------|
| **Vision-Only** | 仅RGB视觉 | 单模态下界 |
| **Tac3D-Only (Simple)** | 简化Tac3D编码 | 基线对比 |
| **Tac3D-Only (PSTE)** | 我们的Tac3D编码 | 验证PSTE有效性 |
| **Simple Concat** | 简单拼接融合 | 简单融合基线 |
| **Weighted Sum** | 固定权重融合 | 静态权重基线 |
| **Ours (w/o PSTE)** | 去掉双流编码 | 验证PSTE必要性 |
| **Ours (w/o Phase)** | 去掉阶段门控 | 验证阶段感知必要性 |
| **Ours (w/o VT-CAF)** | 去掉Cross-Attention | 验证CAF必要性 |
| **Ours (Full)** | 完整系统 | 最终方案 |

---

## 📝 创新点详情

### 创新点A: Tac3D物理感知双流编码器 (Tac3D-PSTE)

**文件**: `innovation_A_tac3d_encoder.py`

**核心**:
- 将Tac3D的400×6点云分离为位移场和力场
- 保留20×20空间网格结构，使用2D Conv
- Cross-Attention融合

**输入**: `[B, 400, 6]` (dx, dy, dz, fx, fy, fz)  
**输出**: `[B, 512]`

```python
from innovation_A_tac3d_encoder import Tac3DPSTEncoder

encoder = Tac3DPSTEncoder(output_dim=512)
tac3d_feat = encoder(tac3d_data)  # [B, 512]
```

---

### 创新点B: 阶段感知模态门控 (Phase-Aware Gating)

**文件**: `innovation_B_phase_gating.py`

**核心**:
- 根据任务阶段动态调整视觉vs触觉权重
- 支持硬编码规则和可学习模式

**阶段权重**:
| 阶段 | 视觉权重 | 触觉权重 |
|-----|---------|---------|
| Approach | 0.9 | 0.1 |
| Contact | 0.3 | 0.7 |
| Manipulate | 0.5 | 0.5 |
| Retract | 0.8 | 0.2 |

```python
from innovation_B_phase_gating import PhaseAwareModalityFusion, TaskPhase

gating = PhaseAwareModalityFusion(mode='learnable')
fused = gating(vision_feat, tactile_feat)
```

---

### 创新点C: 多速率扩散策略 (Multi-Rate)

**文件**: `innovation_C_multi_rate.py`

**核心**:
- RGB (30Hz) + Tac3D (100Hz) + Policy (10Hz)
- 多速率缓冲区管理
- 时序聚合

```python
from innovation_C_multi_rate import MultiRateBuffer, MultiRateEncoder

buffer = MultiRateBuffer(vision_freq=30, tactile_freq=100, policy_freq=10)
encoder = MultiRateEncoder(...)
```

---

### 创新点D: 视触觉Cross-Attention融合 (VT-CAF)

**文件**: `innovation_D_cross_attention.py`

**核心**:
- 双向Cross-Attention: Visual↔Tactile
- 支持注意力可视化

```python
from innovation_D_cross_attention import VTCAFModule

vt_caf = VTCAFModule(vision_dim=512, tactile_dim=512, output_dim=512)
fused = vt_caf(vision_feat, tactile_feat)
```

---

## 🔧 集成到LeRobot

### 步骤1: 复制到LeRobot

```bash
# 项目已在正确的位置
# /home/cuizhixing/.openclaw/workspace-scientist/lerobot/visuotactile_diffusion_policy/
```

### 步骤2: 修改训练脚本

参考 `lerobot/lerobot/scripts/train.py`，添加：

```python
from visuotactile_diffusion_policy.visuotactile_policy import VisuotactileDiffusionPolicy

# 替换模型创建部分
policy = VisuotactileDiffusionPolicy(
    action_dim=env.action_space.shape[0],
    use_tac3d_pste=True,
    use_phase_gating=True,
    use_vt_caf=True,
)
```

### 步骤3: 数据加载

需要修改数据加载器以支持Tac3D数据：

```python
# 在dataset中添加Tac3D观测
observations = {
    'rgb': rgb_image,          # [3, H, W]
    'tac3d': tac3d_data,       # [400, 6]
    'proprioception': state,   # [...]
}
```

---

## 📊 实验计划

### 任务

1. **Peg-in-Hole插孔** - 高精度装配
2. **Object Grasping** - 不同材质抓取
3. **Surface Following** - 曲面跟踪

### 指标

- Success Rate (成功率)
- Contact Force Profile (接触力平滑性)
- Phase Transition Accuracy (阶段转换准确率)
- Inference Time (推理时间)

### 目标会议

- CoRL 2026
- RSS 2026
- ICRA 2026
- IROS 2026

---

## 🔗 相关资源

### 基线代码

- `baselines/3d_diffusion_policy/` - DP3 (CoRL 2024)
- `baselines/diffusion_policy/` - Diffusion Policy

### 传感器资料

参考 `../resource/` 文件夹：
- Tac3D传感器文档
- RealMan机器人SDK
- 相机配置

---

## 📚 参考文献

核心基线:
- 3D Diffusion Policy (Ze et al., CoRL 2024)
- Reactive Diffusion Policy (Chen et al., RSS 2025)
- 3D-ViTac (Zhang et al., CoRL 2024)
- ForceVLA (2025)
- BFA (2025)

---

## 👥 团队

**Researcher**: Walker Jesse  
**Advisor**: Dr. Sigma  
**Institution**: [Your Institution]

---

## 📄 更新日志

### 2026-03-11
- 创建项目结构
- 整合所有创新点代码
- 添加消融实验脚本
- 创建完整模型类

### 2026-03-10
- 完成4个创新点代码
- 完成文献调研
- 完成Baseline模型

---

## 📝 注意事项

1. **Tac3D数据格式**: 输入应为 `[B, 400, 6]`，前3维位移，后3维力
2. **Git同步**: 每次修改后提交到GitHub
3. **实验记录**: 在 `experiments/` 文件夹记录实验结果

---

*Last Updated: 2026-03-11*  
*Dr. Sigma ⚡*
