"""
Visuotactile Diffusion Policy - 视触觉融合扩散策略
整合所有创新点的完整模型

Author: Dr. Sigma
Date: 2026-03-11
Repository: https://github.com/changer-changer/lerobot
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

# 导入创新点模块
from innovation_A_tac3d_encoder import Tac3DPSTEncoder
from innovation_B_phase_gating import PhaseAwareModalityFusion, TaskPhase
from innovation_C_multi_rate import MultiRateEncoder, MultiRateBuffer
from innovation_D_cross_attention import VTCAFModule


class VisuotactileDiffusionPolicy(nn.Module):
    """
    视触觉融合扩散策略 - 完整版
    
    融合创新点:
    1. Tac3D-PSTE: 物理感知双流编码
    2. Phase-Aware Gating: 阶段感知模态门控
    3. Multi-Rate: 多速率数据融合 (可选)
    4. VT-CAF: 视触觉Cross-Attention融合
    
    输入:
        - RGB图像: [B, 3, H, W] 或 [B, T, 3, H, W] (多帧)
        - Tac3D数据: [B, 400, 6] (dx, dy, dz, fx, fy, fz)
        - (可选) 任务阶段: TaskPhase
    
    输出:
        - 动作: [B, action_dim]
    """
    
    def __init__(self,
                 action_dim: int = 7,
                 vision_encoder: Optional[nn.Module] = None,
                 use_tac3d_pste: bool = True,
                 use_phase_gating: bool = True,
                 use_multi_rate: bool = False,
                 use_vt_caf: bool = True,
                 vision_feature_dim: int = 512,
                 tactile_feature_dim: int = 512,
                 fusion_output_dim: int = 512):
        super().__init__()
        
        self.action_dim = action_dim
        self.use_tac3d_pste = use_tac3d_pste
        self.use_phase_gating = use_phase_gating
        self.use_multi_rate = use_multi_rate
        self.use_vt_caf = use_vt_caf
        
        # 1. 视觉编码器
        if vision_encoder is not None:
            self.vision_encoder = vision_encoder
        else:
            # 默认使用ResNet18风格编码器
            self.vision_encoder = self._default_vision_encoder(vision_feature_dim)
        
        # 2. Tac3D编码器 (创新点A)
        if use_tac3d_pste:
            self.tac3d_encoder = Tac3DPSTEncoder(output_dim=tactile_feature_dim)
        else:
            # 简化版Tac3D编码器
            self.tac3d_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(400 * 6, 512),
                nn.ReLU(),
                nn.Linear(512, tactile_feature_dim),
            )
        
        # 3. 多速率编码器 (创新点C, 可选)
        if use_multi_rate:
            self.multi_rate_encoder = MultiRateEncoder(
                vision_feature_dim=vision_feature_dim,
                tactile_feature_dim=tactile_feature_dim,
                output_dim=fusion_output_dim,
                vision_frames=3,  # 30Hz / 10Hz
                tactile_frames=10,  # 100Hz / 10Hz
            )
        
        # 4. 阶段感知门控 (创新点B)
        if use_phase_gating:
            self.phase_gating = PhaseAwareModalityFusion(
                feature_dim=fusion_output_dim,
                mode='learnable'
            )
        
        # 5. Cross-Attention融合 (创新点D)
        if use_vt_caf:
            self.vt_caf = VTCAFModule(
                vision_dim=vision_feature_dim,
                tactile_dim=tactile_feature_dim,
                output_dim=fusion_output_dim,
                num_ca_layers=2,
            )
        
        # 6. 扩散策略头部 (占位符，实际应替换为完整Diffusion Policy)
        self.policy_head = nn.Sequential(
            nn.Linear(fusion_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_dim),
        )
        
        # 多速率缓冲区 (用于实时推理)
        if use_multi_rate:
            self.register_buffer('use_buffer', torch.tensor(True))
        else:
            self.register_buffer('use_buffer', torch.tensor(False))
    
    def _default_vision_encoder(self, output_dim: int) -> nn.Module:
        """默认视觉编码器 (简化版ResNet)"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, output_dim),
        )
    
    def forward(self, 
                rgb: torch.Tensor, 
                tac3d: torch.Tensor,
                phase: Optional[TaskPhase] = None,
                return_features: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        前向传播
        
        Args:
            rgb: RGB图像 [B, 3, H, W] 或 [B, T, 3, H, W] (多帧模式)
            tac3d: Tac3D数据 [B, 400, 6]
            phase: 任务阶段 (可选，用于硬编码门控)
            return_features: 是否返回中间特征
        
        Returns:
            action: 预测动作 [B, action_dim]
            features: (可选) 中间特征字典
        """
        features = {}
        B = rgb.shape[0]
        
        # 处理单帧或多帧输入
        if rgb.dim() == 5:  # [B, T, C, H, W]
            # 多帧模式: 取最后一帧或使用时序编码
            rgb_input = rgb[:, -1, :, :, :]  # 简化: 取最后一帧
        else:
            rgb_input = rgb
        
        # 1. 编码视觉
        vision_feat = self.vision_encoder(rgb_input)
        if vision_feat.dim() > 2:
            vision_feat = vision_feat.squeeze()
        features['vision'] = vision_feat
        
        # 2. 编码Tac3D
        if self.use_tac3d_pste:
            tactile_feat = self.tac3d_encoder(tac3d)
        else:
            tactile_feat = self.tac3d_encoder(tac3d)
        features['tactile'] = tactile_feat
        
        # 3. 多速率融合 (创新点C)
        if self.use_multi_rate and rgb.dim() == 5:
            # 多帧模式
            vision_seq = rgb  # [B, T, C, H, W]
            # 需要处理vision_seq为特征序列
            # 简化: 使用单帧特征重复
            vision_seq_feat = vision_feat.unsqueeze(1).repeat(1, 3, 1)  # [B, 3, D]
            tactile_seq_feat = tactile_feat.unsqueeze(1).repeat(1, 10, 1)  # [B, 10, D]
            fused = self.multi_rate_encoder(vision_seq_feat, tactile_seq_feat)
        else:
            # 4. 融合模块
            if self.use_vt_caf and self.use_phase_gating:
                # 先用CAF，再用门控
                fused_caf = self.vt_caf(vision_feat, tactile_feat)
                fused, phase_logits = self.phase_gating(
                    fused_caf, tactile_feat, return_phase=True
                )
                features['phase_logits'] = phase_logits
            elif self.use_vt_caf:
                fused = self.vt_caf(vision_feat, tactile_feat)
            elif self.use_phase_gating:
                fused, _ = self.phase_gating(vision_feat, tactile_feat, phase)
            else:
                # 简单拼接
                fused = torch.cat([vision_feat, tactile_feat], dim=-1)
                fused = nn.Linear(fused.shape[-1], 512).to(fused.device)(fused)
        
        features['fused'] = fused
        
        # 5. 策略预测
        action = self.policy_head(fused)
        
        if return_features:
            return action, features
        return action, None
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        
        info = {
            'total_params': total_params,
            'total_params_M': total_params / 1e6,
            'action_dim': self.action_dim,
            'use_tac3d_pste': self.use_tac3d_pste,
            'use_phase_gating': self.use_phase_gating,
            'use_multi_rate': self.use_multi_rate,
            'use_vt_caf': self.use_vt_caf,
        }
        
        # 各模块参数量
        if hasattr(self, 'vision_encoder'):
            info['vision_encoder_params'] = sum(p.numel() for p in self.vision_encoder.parameters())
        if hasattr(self, 'tac3d_encoder'):
            info['tac3d_encoder_params'] = sum(p.numel() for p in self.tac3d_encoder.parameters())
        if hasattr(self, 'vt_caf'):
            info['vt_caf_params'] = sum(p.numel() for p in self.vt_caf.parameters())
        if hasattr(self, 'phase_gating'):
            info['phase_gating_params'] = sum(p.numel() for p in self.phase_gating.parameters())
        
        return info


class VisuotactileDiffusionPolicyConfig:
    """
    模型配置类
    用于方便地创建不同配置的模型
    """
    
    @staticmethod
    def full_system(action_dim: int = 7) -> VisuotactileDiffusionPolicy:
        """完整系统 (所有创新点)"""
        return VisuotactileDiffusionPolicy(
            action_dim=action_dim,
            use_tac3d_pste=True,
            use_phase_gating=True,
            use_multi_rate=False,  # 多速率需要特殊数据格式
            use_vt_caf=True,
        )
    
    @staticmethod
    def without_pste(action_dim: int = 7) -> VisuotactileDiffusionPolicy:
        """消融: 不使用PSTE"""
        return VisuotactileDiffusionPolicy(
            action_dim=action_dim,
            use_tac3d_pste=False,
            use_phase_gating=True,
            use_vt_caf=True,
        )
    
    @staticmethod
    def without_phase_gating(action_dim: int = 7) -> VisuotactileDiffusionPolicy:
        """消融: 不使用阶段门控"""
        return VisuotactileDiffusionPolicy(
            action_dim=action_dim,
            use_tac3d_pste=True,
            use_phase_gating=False,
            use_vt_caf=True,
        )
    
    @staticmethod
    def without_vt_caf(action_dim: int = 7) -> VisuotactileDiffusionPolicy:
        """消融: 不使用VT-CAF"""
        return VisuotactileDiffusionPolicy(
            action_dim=action_dim,
            use_tac3d_pste=True,
            use_phase_gating=True,
            use_vt_caf=False,
        )
    
    @staticmethod
    def vision_only(action_dim: int = 7) -> VisuotactileDiffusionPolicy:
        """仅视觉基线"""
        from baseline_models import VisionOnlyBaseline
        return VisionOnlyBaseline(
            vision_encoder=None,  # 使用默认
            action_dim=action_dim,
        )
    
    @staticmethod
    def tac3d_only(action_dim: int = 7, use_pste: bool = True) -> VisuotactileDiffusionPolicy:
        """仅Tac3D基线"""
        from baseline_models import Tac3DOnlyBaseline
        tac3d_encoder = Tac3DPSTEncoder(512) if use_pste else None
        return Tac3DOnlyBaseline(
            tac3d_encoder=tac3d_encoder,
            action_dim=action_dim,
        )


# ==================== 测试代码 ====================

def test_visuotactile_policy():
    """测试视触觉融合策略"""
    print("=" * 70)
    print("测试 Visuotactile Diffusion Policy")
    print("=" * 70)
    
    batch_size = 4
    action_dim = 7
    
    # 测试数据
    rgb = torch.randn(batch_size, 3, 224, 224)
    tac3d = torch.randn(batch_size, 400, 6)
    
    print(f"\n输入:")
    print(f"  - RGB: {rgb.shape}")
    print(f"  - Tac3D: {tac3d.shape}")
    
    # 测试不同配置
    configs = [
        ("Full System", VisuotactileDiffusionPolicyConfig.full_system(action_dim)),
        ("w/o PSTE", VisuotactileDiffusionPolicyConfig.without_pste(action_dim)),
        ("w/o Phase Gating", VisuotactileDiffusionPolicyConfig.without_phase_gating(action_dim)),
        ("w/o VT-CAF", VisuotactileDiffusionPolicyConfig.without_vt_caf(action_dim)),
    ]
    
    print("\n测试各配置:")
    print("-" * 70)
    
    for name, model in configs:
        model.eval()
        with torch.no_grad():
            # 测试前向传播
            action, features = model(rgb, tac3d, return_features=True)
            
            # 获取模型信息
            info = model.get_model_info()
            
            print(f"\n{name}:")
            print(f"  输出: {action.shape}")
            print(f"  参数量: {info['total_params']:,} ({info['total_params_M']:.2f}M)")
            print(f"  特征: {list(features.keys())}")
    
    # 推理速度测试
    print("\n" + "-" * 70)
    print("推理速度测试:")
    print("-" * 70)
    
    import time
    
    for name, model in configs:
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(rgb, tac3d)
            
            # Timing
            start = time.time()
            for _ in range(100):
                _ = model(rgb, tac3d)
            end = time.time()
            
            avg_time = (end - start) / 100 * 1000
            print(f"  {name:20}: {avg_time:.2f} ms")
    
    print("\n" + "=" * 70)
    print("✅ Visuotactile Diffusion Policy 测试通过!")
    print("=" * 70)


if __name__ == "__main__":
    test_visuotactile_policy()
