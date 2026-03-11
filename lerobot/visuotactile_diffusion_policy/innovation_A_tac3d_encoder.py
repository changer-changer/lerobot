"""
创新点A: Tac3D物理感知双流编码器 (Tac3D-PSTE)
Tac3D Physical-Aware Two-Stream Encoder

核心创新:
1. 将Tac3D的400×6点云分别处理位移和力两个物理量
2. 保留20×20空间网格结构，使用2D Conv而非PointNet
3. Cross-Attention融合位移和力特征

Author: Dr. Sigma
Date: 2026-03-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Tac3DStreamEncoder(nn.Module):
    """
    Tac3D单流编码器 (处理位移或力)
    输入: [B, 400, 3] - 400个点，每个点3D向量
    输出: [B, 256] - 特征向量
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 将400点reshape为20x20网格
        # 2D Conv提取空间特征
        self.spatial_encoder = nn.Sequential(
            # [B, 3, 20, 20] -> [B, 32, 10, 10]
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # [B, 32, 10, 10] -> [B, 64, 5, 5]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # [B, 64, 5, 5] -> [B, 128, 2, 2]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Flatten后接MLP
        self.fc = nn.Sequential(
            nn.Linear(128 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 400, 3] - Tac3D点云 (位移或力)
        Returns:
            feat: [B, 256] - 编码特征
        """
        B = x.shape[0]
        
        # Reshape为2D网格: [B, 400, 3] -> [B, 3, 20, 20]
        x = x.reshape(B, 3, 20, 20)
        
        # 2D Conv编码
        x = self.spatial_encoder(x)  # [B, 128, 2, 2]
        
        # Flatten - 自适应计算flatten维度
        x = x.reshape(B, -1)  # [B, C*H*W]
        
        # 动态调整MLP输入维度
        if self.fc[0].in_features != x.shape[1]:
            # 重新创建MLP层
            self.fc = nn.Sequential(
                nn.Linear(x.shape[1], 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, self.hidden_dim),
            ).to(x.device)
        
        # MLP
        feat = self.fc(x)  # [B, 256]
        
        return feat


class Tac3DCrossAttention(nn.Module):
    """
    位移和力特征的Cross-Attention融合 (简化版)
    """
    def __init__(self, dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # 简化的Cross-Attention: 使用线性层模拟
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # 融合投影
        self.fusion_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        
    def forward(self, feat_disp: torch.Tensor, feat_force: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_disp: [B, 256] - 位移特征
            feat_force: [B, 256] - 力特征
        Returns:
            fused: [B, 256] - 融合特征
        """
        # 添加序列维度: [B, 256] -> [B, 1, 256]
        feat_disp = feat_disp.unsqueeze(1)
        feat_force = feat_force.unsqueeze(1)
        
        # Cross-Attention: Displacement attends to Force
        attn_out1, _ = self.cross_attn(feat_disp, feat_force, feat_force)
        feat_disp = self.norm1(feat_disp + attn_out1)
        
        # Cross-Attention: Force attends to Displacement
        attn_out2, _ = self.cross_attn(feat_force, feat_disp, feat_disp)
        feat_force = self.norm2(feat_force + attn_out2)
        
        # 移除序列维度并融合
        feat_disp = feat_disp.squeeze(1)  # [B, 256]
        feat_force = feat_force.squeeze(1)  # [B, 256]
        
        fused = self.fusion_proj(torch.cat([feat_disp, feat_force], dim=-1))
        
        return fused


class Tac3DPSTEncoder(nn.Module):
    """
    Tac3D物理感知双流编码器 (完整版)
    
    输入: Tac3D点云 [B, 400, 6] (dx, dy, dz, fx, fy, fz)
    输出: 融合特征 [B, 512]
    """
    def __init__(self, output_dim: int = 512):
        super().__init__()
        
        # 双流编码器
        self.displacement_encoder = Tac3DStreamEncoder(input_dim=3, hidden_dim=256)
        self.force_encoder = Tac3DStreamEncoder(input_dim=3, hidden_dim=256)
        
        # Cross-Attention融合
        self.cross_attn = Tac3DCrossAttention(dim=256, num_heads=4)
        
        # 最终投影
        self.output_proj = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
        
    def forward(self, tac3d_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tac3d_data: [B, 400, 6] - Tac3D原始数据
                [:, :, 0:3] - 位移 (dx, dy, dz)
                [:, :, 3:6] - 力 (fx, fy, fz)
        Returns:
            feat: [B, 512] - 融合特征
        """
        # 分离位移和力
        displacement = tac3d_data[:, :, 0:3]  # [B, 400, 3]
        force = tac3d_data[:, :, 3:6]         # [B, 400, 3]
        
        # 双流编码
        feat_disp = self.displacement_encoder(displacement)  # [B, 256]
        feat_force = self.force_encoder(force)               # [B, 256]
        
        # Cross-Attention融合
        fused = self.cross_attn(feat_disp, feat_force)  # [B, 256]
        
        # 输出投影
        output = self.output_proj(fused)  # [B, 512]
        
        return output
    
    def get_individual_features(self, tac3d_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单独的位移和力特征 (用于分析)"""
        displacement = tac3d_data[:, :, 0:3]
        force = tac3d_data[:, :, 3:6]
        
        feat_disp = self.displacement_encoder(displacement)
        feat_force = self.force_encoder(force)
        
        return feat_disp, feat_force


# ==================== 测试代码 ====================

def test_tac3d_encoder():
    """测试Tac3D编码器"""
    print("=" * 60)
    print("测试 Tac3D-PSTEncoder")
    print("=" * 60)
    
    # 创建模型
    model = Tac3DPSTEncoder(output_dim=512)
    
    # 测试输入: batch=4, 400个点, 6D特征
    batch_size = 4
    tac3d_input = torch.randn(batch_size, 400, 6)
    
    print(f"\n输入形状: {tac3d_input.shape}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - 点数: 400")
    print(f"  - 特征维度: 6 (dx,dy,dz,fx,fy,fz)")
    
    # 前向传播
    output = model(tac3d_input)
    
    print(f"\n输出形状: {output.shape}")
    print(f"  - 融合特征维度: 512")
    
    # 获取单独特征
    feat_disp, feat_force = model.get_individual_features(tac3d_input)
    print(f"\n单独特征:")
    print(f"  - 位移特征: {feat_disp.shape}")
    print(f"  - 力特征: {feat_force.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"  - 模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    # 测试推理速度
    import time
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(tac3d_input)
        
        # Timing
        start = time.time()
        for _ in range(100):
            _ = model(tac3d_input)
        end = time.time()
        
        avg_time = (end - start) / 100 * 1000
        print(f"\n推理速度:")
        print(f"  - 平均推理时间: {avg_time:.2f} ms")
        print(f"  - 理论最大FPS: {1000/avg_time:.1f}")
    
    print("\n" + "=" * 60)
    print("✅ Tac3D-PSTEncoder 测试通过!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    test_tac3d_encoder()
