"""
消融实验脚本 - Ablation Study Script
明天可直接运行验证所有创新点

实验配置:
1. Vision-Only Baseline
2. Tac3D-Only Baseline  
3. Simple Concat Baseline
4. Weighted Sum Baselines
5. Full System (所有创新点)

Author: Dr. Sigma
Date: 2026-03-10
"""

import torch
import torch.nn as nn
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

# 导入所有模块
from innovation_A_tac3d_encoder import Tac3DPSTEncoder
from innovation_B_phase_gating import PhaseAwareModalityFusion, TaskPhase
from innovation_C_multi_rate import MultiRateEncoder
from innovation_D_cross_attention import VTCAFModule
from baseline_models import (VisionOnlyBaseline, Tac3DOnlyBaseline, 
                             SimpleConcatBaseline, WeightedSumBaseline)


class MockVisionEncoder(nn.Module):
    """模拟视觉编码器 (实际应替换为ResNet18/DINOv3)"""
    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.output_dim = output_dim
        # 简化的CNN
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, output_dim),
        )
    
    def forward(self, x):
        return self.encoder(x)


class VisuotactileFusionSystem(nn.Module):
    """
    完整视触觉融合系统 (所有创新点)
    """
    def __init__(self, 
                 action_dim: int = 7,
                 use_tac3d_pste: bool = True,
                 use_phase_gating: bool = True,
                 use_multi_rate: bool = False,  # 需要序列输入
                 use_vt_caf: bool = True):
        super().__init__()
        
        self.action_dim = action_dim
        self.use_tac3d_pste = use_tac3d_pste
        self.use_phase_gating = use_phase_gating
        self.use_multi_rate = use_multi_rate
        self.use_vt_caf = use_vt_caf
        
        # 视觉编码器
        self.vision_encoder = MockVisionEncoder(output_dim=512)
        
        # Tac3D编码器 (创新点A)
        if use_tac3d_pste:
            self.tac3d_encoder = Tac3DPSTEncoder(output_dim=512)
        else:
            # 简化版Tac3D编码器
            self.tac3d_encoder = nn.Sequential(
                nn.Linear(400 * 6, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
            )
        
        # 融合模块
        fusion_dim = 512
        
        # 创新点B: 阶段感知门控
        if use_phase_gating:
            self.phase_gating = PhaseAwareModalityFusion(
                feature_dim=512, mode='learnable'
            )
        
        # 创新点D: Cross-Attention融合
        if use_vt_caf:
            self.vt_caf = VTCAFModule(512, 512, 512)
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
    
    def forward(self, rgb: torch.Tensor, tac3d: torch.Tensor, 
                return_features: bool = False):
        """
        Args:
            rgb: [B, 3, H, W]
            tac3d: [B, 400, 6]
            return_features: 是否返回中间特征
        Returns:
            action: [B, action_dim]
            features: (optional) 中间特征字典
        """
        features = {}
        
        # 编码视觉
        vision_feat = self.vision_encoder(rgb)
        features['vision'] = vision_feat
        
        # 编码Tac3D
        if self.use_tac3d_pste:
            tactile_feat = self.tac3d_encoder(tac3d)
        else:
            B = tac3d.shape[0]
            tactile_feat = self.tac3d_encoder(tac3d.reshape(B, -1))
        features['tactile'] = tactile_feat
        
        # 融合
        if self.use_phase_gating and self.use_vt_caf:
            # 两者都用: 先CAF再门控
            fused_caf = self.vt_caf(vision_feat, tactile_feat)
            fused, phase_logits = self.phase_gating(
                fused_caf, tactile_feat, return_phase=True
            )
            features['phase_logits'] = phase_logits
        elif self.use_vt_caf:
            fused = self.vt_caf(vision_feat, tactile_feat)
        elif self.use_phase_gating:
            fused, _ = self.phase_gating(vision_feat, tactile_feat)
        else:
            # 简单拼接
            fused = torch.cat([vision_feat, tactile_feat], dim=-1)
            fused = nn.Linear(fused.shape[-1], 512).to(fused.device)(fused)
        
        features['fused'] = fused
        
        # 策略预测
        action = self.policy_head(fused)
        
        if return_features:
            return action, features
        return action
    
    def count_parameters(self) -> Dict[str, int]:
        """统计各模块参数量"""
        return {
            'vision_encoder': sum(p.numel() for p in self.vision_encoder.parameters()),
            'tac3d_encoder': sum(p.numel() for p in self.tac3d_encoder.parameters()),
            'fusion': sum(p.numel() for p in self.parameters()) - 
                     sum(p.numel() for p in self.vision_encoder.parameters()) -
                     sum(p.numel() for p in self.tac3d_encoder.parameters()) -
                     sum(p.numel() for p in self.policy_head.parameters()),
            'policy_head': sum(p.numel() for p in self.policy_head.parameters()),
            'total': sum(p.numel() for p in self.parameters()),
        }


def run_ablation_study(save_dir: str = "./ablation_results"):
    """
    运行完整消融实验
    
    测试配置:
    1. Vision-Only
    2. Tac3D-Only
    3. Simple Concat
    4. Weighted Sum (0.5/0.5)
    5. Weighted Sum (0.7/0.3)
    6. Ours (w/o Phase Gating)
    7. Ours (w/o VT-CAF)
    8. Ours (Full System)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("消融实验 - Ablation Study")
    print("=" * 70)
    
    # 测试配置
    batch_size = 4
    action_dim = 7
    
    # 测试数据
    rgb = torch.randn(batch_size, 3, 224, 224)
    tac3d = torch.randn(batch_size, 400, 6)
    
    # 创建编码器
    vision_encoder = MockVisionEncoder(512)
    tac3d_encoder_pste = Tac3DPSTEncoder(512)
    tac3d_encoder_simple = nn.Sequential(
        nn.Flatten(),
        nn.Linear(400 * 6, 512), 
        nn.ReLU(), 
        nn.Linear(512, 512)
    )
    
    results = []
    
    # ========== Baselines ==========
    
    configs = [
        {
            "name": "Vision-Only",
            "model": VisionOnlyBaseline(vision_encoder, action_dim),
            "inputs": (rgb,),
        },
        {
            "name": "Tac3D-Only (Simple)",
            "model": Tac3DOnlyBaseline(tac3d_encoder_simple, action_dim),
            "inputs": (tac3d,),
        },
        {
            "name": "Tac3D-Only (PSTE)",
            "model": Tac3DOnlyBaseline(tac3d_encoder_pste, action_dim),
            "inputs": (tac3d,),
        },
        {
            "name": "Simple Concat",
            "model": SimpleConcatBaseline(vision_encoder, tac3d_encoder_simple, action_dim),
            "inputs": (rgb, tac3d),
        },
        {
            "name": "Weighted Sum (0.5/0.5)",
            "model": WeightedSumBaseline(vision_encoder, tac3d_encoder_simple, action_dim, 0.5, 0.5),
            "inputs": (rgb, tac3d),
        },
        {
            "name": "Weighted Sum (0.7/0.3)",
            "model": WeightedSumBaseline(vision_encoder, tac3d_encoder_simple, action_dim, 0.7, 0.3),
            "inputs": (rgb, tac3d),
        },
    ]
    
    # ========== Our Methods ==========
    
    our_configs = [
        {
            "name": "Ours (w/o PSTE)",
            "model": VisuotactileFusionSystem(action_dim, use_tac3d_pste=False, 
                                             use_phase_gating=True, use_vt_caf=True),
            "inputs": (rgb, tac3d),
        },
        {
            "name": "Ours (w/o Phase Gating)",
            "model": VisuotactileFusionSystem(action_dim, use_tac3d_pste=True,
                                             use_phase_gating=False, use_vt_caf=True),
            "inputs": (rgb, tac3d),
        },
        {
            "name": "Ours (w/o VT-CAF)",
            "model": VisuotactileFusionSystem(action_dim, use_tac3d_pste=True,
                                             use_phase_gating=True, use_vt_caf=False),
            "inputs": (rgb, tac3d),
        },
        {
            "name": "Ours (Full System)",
            "model": VisuotactileFusionSystem(action_dim, use_tac3d_pste=True,
                                             use_phase_gating=True, use_vt_caf=True),
            "inputs": (rgb, tac3d),
        },
    ]
    
    all_configs = configs + our_configs
    
    print(f"\n测试 {len(all_configs)} 个配置:\n")
    
    for i, config in enumerate(all_configs):
        print(f"[{i+1}/{len(all_configs)}] Testing: {config['name']}")
        
        model = config['model']
        model.eval()
        
        # 参数量统计
        total_params = sum(p.numel() for p in model.parameters())
        
        # 推理速度测试
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(*config['inputs'])
            
            # Timing
            import time
            start = time.time()
            for _ in range(100):
                output = model(*config['inputs'])
            end = time.time()
            
            avg_time = (end - start) / 100 * 1000  # ms
        
        result = {
            "name": config['name'],
            "params": total_params,
            "inference_time_ms": avg_time,
            "output_shape": list(output.shape),
        }
        
        # 如果有参数量详情
        if hasattr(model, 'count_parameters'):
            result['param_details'] = model.count_parameters()
        
        results.append(result)
        
        print(f"    ✅ Params: {total_params:,} | Time: {avg_time:.2f}ms | Output: {list(output.shape)}")
    
    # 保存结果
    result_file = os.path.join(save_dir, f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("消融实验完成!")
    print(f"结果保存至: {result_file}")
    print("=" * 70)
    
    # 打印对比表
    print("\n对比表:")
    print("-" * 70)
    print(f"{'Model':<30} {'Params':<12} {'Time(ms)':<12} {'Output'}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<30} {r['params']:<12,} {r['inference_time_ms']:<12.2f} {r['output_shape']}")
    print("-" * 70)
    
    return results


if __name__ == "__main__":
    results = run_ablation_study()
