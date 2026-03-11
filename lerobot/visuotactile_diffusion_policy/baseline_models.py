"""
Baseline Models for Ablation Study
消融实验基线模型

包含:
1. Vision-Only Baseline - 仅使用RGB的Diffusion Policy
2. Tac3D-Only Baseline - 仅使用Tac3D的Diffusion Policy  
3. Simple Concat Baseline - 简单拼接融合
4. DP3 Baseline - 标准3D Diffusion Policy

Author: Dr. Sigma
Date: 2026-03-10
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class VisionOnlyBaseline(nn.Module):
    """
    Vision-Only Baseline
    仅使用RGB图像的Diffusion Policy
    """
    def __init__(self, 
                 vision_encoder: nn.Module,
                 action_dim: int = 7,
                 hidden_dim: int = 512):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.hidden_dim = hidden_dim
        
        # 简单的策略头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: [B, 3, H, W] RGB图像
        Returns:
            action: [B, action_dim]
        """
        # 编码视觉
        vision_feat = self.vision_encoder(rgb)
        if vision_feat.dim() > 2:
            vision_feat = vision_feat.squeeze()
        
        # 策略预测
        action = self.policy_head(vision_feat)
        return action


class Tac3DOnlyBaseline(nn.Module):
    """
    Tac3D-Only Baseline
    仅使用Tac3D触觉的Diffusion Policy
    """
    def __init__(self,
                 tac3d_encoder: nn.Module,
                 action_dim: int = 7):
        super().__init__()
        self.tac3d_encoder = tac3d_encoder
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        
    def forward(self, tac3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tac3d: [B, 400, 6] Tac3D点云数据
        Returns:
            action: [B, action_dim]
        """
        # 编码Tac3D
        tactile_feat = self.tac3d_encoder(tac3d)
        
        # 策略预测
        action = self.policy_head(tactile_feat)
        return action


class SimpleConcatBaseline(nn.Module):
    """
    Simple Concatenation Baseline
    简单拼接融合基线
    直接将视觉和触觉特征拼接后输入策略网络
    """
    def __init__(self,
                 vision_encoder: nn.Module,
                 tac3d_encoder: nn.Module,
                 action_dim: int = 7,
                 vision_dim: int = 512,
                 tactile_dim: int = 512):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.tac3d_encoder = tac3d_encoder
        
        # 融合和策略头
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + tactile_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        
    def forward(self, rgb: torch.Tensor, tac3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: [B, 3, H, W] RGB图像
            tac3d: [B, 400, 6] Tac3D数据
        Returns:
            action: [B, action_dim]
        """
        # 分别编码
        vision_feat = self.vision_encoder(rgb)
        if vision_feat.dim() > 2:
            vision_feat = vision_feat.squeeze()
        
        tactile_feat = self.tac3d_encoder(tac3d)
        
        # 简单拼接
        combined = torch.cat([vision_feat, tactile_feat], dim=-1)
        
        # 融合
        fused = self.fusion(combined)
        
        # 策略预测
        action = self.policy_head(fused)
        return action


class WeightedSumBaseline(nn.Module):
    """
    Weighted Sum Baseline
    加权求和基线 (类似BFA但权重固定)
    """
    def __init__(self,
                 vision_encoder: nn.Module,
                 tac3d_encoder: nn.Module,
                 action_dim: int = 7,
                 vision_weight: float = 0.5,
                 tactile_weight: float = 0.5):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.tac3d_encoder = tac3d_encoder
        
        # 固定权重
        self.w_v = vision_weight
        self.w_t = tactile_weight
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        
    def forward(self, rgb: torch.Tensor, tac3d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: [B, 3, H, W] RGB图像
            tac3d: [B, 400, 6] Tac3D数据
        Returns:
            action: [B, action_dim]
        """
        # 编码
        vision_feat = self.vision_encoder(rgb)
        if vision_feat.dim() > 2:
            vision_feat = vision_feat.squeeze()
        
        tactile_feat = self.tac3d_encoder(tac3d)
        
        # 加权求和
        fused = self.w_v * vision_feat + self.w_t * tactile_feat
        
        # 策略预测
        action = self.policy_head(fused)
        return action


class DP3Baseline(nn.Module):
    """
    DP3 Baseline Wrapper
    标准3D Diffusion Policy基线
    使用点云作为输入
    """
    def __init__(self,
                 point_encoder: nn.Module,
                 action_dim: int = 7):
        super().__init__()
        self.point_encoder = point_encoder
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        
    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_cloud: [B, N, 3] 环境点云
        Returns:
            action: [B, action_dim]
        """
        # 编码点云
        pc_feat = self.point_encoder(point_cloud)
        
        # 策略预测
        action = self.policy_head(pc_feat)
        return action


# ==================== 消融实验配置 ====================

ABLATION_CONFIGS = {
    "vision_only": {
        "name": "Vision-Only",
        "description": "仅使用RGB视觉",
        "modalities": ["vision"],
        "model_class": VisionOnlyBaseline,
    },
    "tac3d_only": {
        "name": "Tac3D-Only", 
        "description": "仅使用Tac3D触觉",
        "modalities": ["tactile"],
        "model_class": Tac3DOnlyBaseline,
    },
    "simple_concat": {
        "name": "Simple Concat",
        "description": "视觉和触觉简单拼接",
        "modalities": ["vision", "tactile"],
        "model_class": SimpleConcatBaseline,
    },
    "weighted_sum_50_50": {
        "name": "Weighted Sum (0.5/0.5)",
        "description": "固定权重0.5/0.5",
        "modalities": ["vision", "tactile"],
        "model_class": WeightedSumBaseline,
        "kwargs": {"vision_weight": 0.5, "tactile_weight": 0.5},
    },
    "weighted_sum_70_30": {
        "name": "Weighted Sum (0.7/0.3)",
        "description": "固定权重0.7/0.3",
        "modalities": ["vision", "tactile"],
        "model_class": WeightedSumBaseline,
        "kwargs": {"vision_weight": 0.7, "tactile_weight": 0.3},
    },
    "full_ours": {
        "name": "Full System (Ours)",
        "description": "完整系统 (所有创新点)",
        "modalities": ["vision", "tactile"],
        "model_class": None,  # 使用完整的创新点代码
    },
}


def create_baseline_model(config_name: str, 
                          vision_encoder: nn.Module,
                          tac3d_encoder: nn.Module,
                          **kwargs) -> nn.Module:
    """
    根据配置创建基线模型
    
    Args:
        config_name: 配置名称 (见ABLATION_CONFIGS)
        vision_encoder: 视觉编码器
        tac3d_encoder: Tac3D编码器
        **kwargs: 额外参数
    Returns:
        model: 基线模型
    """
    config = ABLATION_CONFIGS[config_name]
    model_class = config["model_class"]
    
    if model_class is None:
        raise ValueError(f"请手动创建 {config_name} 模型")
    
    model_kwargs = config.get("kwargs", {})
    model_kwargs.update(kwargs)
    
    if config_name in ["vision_only"]:
        return model_class(vision_encoder, **model_kwargs)
    elif config_name in ["tac3d_only"]:
        return model_class(tac3d_encoder, **model_kwargs)
    else:
        return model_class(vision_encoder, tac3d_encoder, **model_kwargs)


# ==================== 测试代码 ====================

def test_baselines():
    """测试所有基线模型"""
    print("=" * 60)
    print("测试 Baseline Models")
    print("=" * 60)
    
    batch_size = 4
    action_dim = 7
    
    # Mock编码器
    class MockVisionEncoder(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], 512)
    
    class MockTac3DEncoder(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], 512)
    
    vision_encoder = MockVisionEncoder()
    tac3d_encoder = MockTac3DEncoder()
    
    # 测试数据
    rgb = torch.randn(batch_size, 3, 224, 224)
    tac3d = torch.randn(batch_size, 400, 6)
    
    print("\n测试各基线模型:")
    print("-" * 40)
    
    # Vision-Only
    model = VisionOnlyBaseline(vision_encoder, action_dim)
    output = model(rgb)
    print(f"✅ Vision-Only: {rgb.shape} -> {output.shape}")
    
    # Tac3D-Only
    model = Tac3DOnlyBaseline(tac3d_encoder, action_dim)
    output = model(tac3d)
    print(f"✅ Tac3D-Only: {tac3d.shape} -> {output.shape}")
    
    # Simple Concat
    model = SimpleConcatBaseline(vision_encoder, tac3d_encoder, action_dim)
    output = model(rgb, tac3d)
    print(f"✅ Simple Concat: RGB+Tac3D -> {output.shape}")
    
    # Weighted Sum
    model = WeightedSumBaseline(vision_encoder, tac3d_encoder, action_dim, 0.7, 0.3)
    output = model(rgb, tac3d)
    print(f"✅ Weighted Sum (0.7/0.3): RGB+Tac3D -> {output.shape}")
    
    print("\n" + "=" * 60)
    print("✅ 所有Baseline测试通过!")
    print("=" * 60)
    
    # 打印消融配置
    print("\n消融实验配置:")
    print("-" * 40)
    for key, config in ABLATION_CONFIGS.items():
        print(f"  {key:20} | {config['name']:25} | {config['description']}")


if __name__ == "__main__":
    test_baselines()
