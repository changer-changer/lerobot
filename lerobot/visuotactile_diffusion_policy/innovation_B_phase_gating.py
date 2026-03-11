"""
创新点B: 阶段感知模态门控 (Phase-Aware Modality Gating)
Phase-Aware Modality Gating

核心创新:
1. 根据任务阶段(Approach/Contact/Manipulate/Retract)动态调整视觉vs触觉权重
2. 可学习的门控网络或硬编码规则
3. 防止模态坍塌，确保触觉信号不被视觉淹没

Author: Dr. Sigma
Date: 2026-03-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from enum import Enum


class TaskPhase(Enum):
    """任务阶段枚举"""
    APPROACH = 0    # 接近阶段: 视觉主导
    CONTACT = 1     # 接触阶段: 触觉主导
    MANIPULATE = 2  # 操作阶段: 平衡
    RETRACT = 3     # 撤回阶段: 视觉主导


class HardcodedPhaseGating:
    """
    硬编码阶段门控 (快速实现版本)
    基于启发式规则
    """
    def __init__(self):
        # 预定义的权重配置
        self.phase_weights = {
            TaskPhase.APPROACH:   {'vision': 0.9, 'tactile': 0.1},
            TaskPhase.CONTACT:    {'vision': 0.3, 'tactile': 0.7},
            TaskPhase.MANIPULATE: {'vision': 0.5, 'tactile': 0.5},
            TaskPhase.RETRACT:    {'vision': 0.8, 'tactile': 0.2},
        }
    
    def get_weights(self, phase: TaskPhase) -> Dict[str, float]:
        """获取指定阶段的权重"""
        return self.phase_weights[phase]
    
    def fuse(self, vision_feat: torch.Tensor, tactile_feat: torch.Tensor, 
             phase: TaskPhase) -> torch.Tensor:
        """
        根据阶段融合特征
        
        Args:
            vision_feat: [B, D] 视觉特征
            tactile_feat: [B, D] 触觉特征
            phase: 当前任务阶段
        Returns:
            fused: [B, D] 融合特征
        """
        weights = self.get_weights(phase)
        w_v = weights['vision']
        w_t = weights['tactile']
        
        fused = w_v * vision_feat + w_t * tactile_feat
        return fused


class LearnablePhaseGating(nn.Module):
    """
    可学习的阶段门控
    使用轻量级网络预测阶段和权重
    """
    def __init__(self, feature_dim: int = 512, num_phases: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_phases = num_phases
        
        # 阶段分类器: 根据视觉+触觉特征预测当前阶段
        self.phase_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_phases),
        )
        
        # 权重预测器: 为每个阶段预测视觉/触觉权重
        # 输出: [w_vision, w_tactile] for each phase
        self.weight_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_phases * 2),  # 2 weights per phase
        )
        
    def forward(self, vision_feat: torch.Tensor, tactile_feat: torch.Tensor,
                return_phase: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            vision_feat: [B, D] 视觉特征
            tactile_feat: [B, D] 触觉特征  
            return_phase: 是否返回阶段预测
        Returns:
            fused: [B, D] 融合特征
            phase_logits: [B, num_phases] 阶段预测 (可选)
        """
        B = vision_feat.shape[0]
        
        # 拼接特征
        combined = torch.cat([vision_feat, tactile_feat], dim=-1)  # [B, 2D]
        
        # 预测阶段
        phase_logits = self.phase_classifier(combined)  # [B, num_phases]
        phase_probs = F.softmax(phase_logits, dim=-1)  # [B, num_phases]
        
        # 预测权重: [B, num_phases * 2]
        weights_logits = self.weight_predictor(combined)
        weights_logits = weights_logits.reshape(B, self.num_phases, 2)  # [B, 4, 2]
        weights = F.softmax(weights_logits, dim=-1)  # [B, 4, 2], 每行之和=1
        
        # 根据阶段概率加权融合权重
        # phase_probs: [B, 4], weights: [B, 4, 2]
        w_v = (phase_probs * weights[:, :, 0]).sum(dim=1, keepdim=True)  # [B, 1]
        w_t = (phase_probs * weights[:, :, 1]).sum(dim=1, keepdim=True)  # [B, 1]
        
        # 融合
        fused = w_v * vision_feat + w_t * tactile_feat  # [B, D]
        
        if return_phase:
            return fused, phase_logits
        return fused, None
    
    def get_phase_weights(self, vision_feat: torch.Tensor, 
                          tactile_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取预测的权重和阶段 (用于分析)"""
        B = vision_feat.shape[0]
        combined = torch.cat([vision_feat, tactile_feat], dim=-1)
        
        phase_logits = self.phase_classifier(combined)
        phase_probs = F.softmax(phase_logits, dim=-1)
        
        weights_logits = self.weight_predictor(combined).reshape(B, self.num_phases, 2)
        weights = F.softmax(weights_logits, dim=-1)
        
        w_v = (phase_probs * weights[:, :, 0]).sum(dim=1)
        w_t = (phase_probs * weights[:, :, 1]).sum(dim=1)
        
        return {
            'vision_weight': w_v,
            'tactile_weight': w_t,
            'phase_probs': phase_probs,
            'predicted_phase': phase_probs.argmax(dim=1),
        }


class PhaseAwareModalityFusion(nn.Module):
    """
    阶段感知模态融合 (完整版)
    支持硬编码和可学习两种模式
    """
    def __init__(self, feature_dim: int = 512, mode: str = 'learnable'):
        super().__init__()
        self.feature_dim = feature_dim
        self.mode = mode
        
        if mode == 'hardcoded':
            self.gating = HardcodedPhaseGating()
            self.learnable_gating = None
        else:  # learnable
            self.gating = None
            self.learnable_gating = LearnablePhaseGating(feature_dim)
    
    def forward(self, vision_feat: torch.Tensor, tactile_feat: torch.Tensor,
                phase: Optional[TaskPhase] = None, return_phase: bool = False):
        """
        前向传播
        
        Args:
            vision_feat: [B, D] 视觉特征
            tactile_feat: [B, D] 触觉特征
            phase: 当前阶段 (硬编码模式必需)
            return_phase: 是否返回阶段信息
        Returns:
            fused: [B, D] 融合特征
            phase_info: 阶段信息 (可选)
        """
        if self.mode == 'hardcoded':
            assert phase is not None, "硬编码模式需要提供phase参数"
            fused = self.gating.fuse(vision_feat, tactile_feat, phase)
            if return_phase:
                return fused, {'phase': phase, 'weights': self.gating.get_weights(phase)}
            return fused, None
        else:
            return self.learnable_gating(vision_feat, tactile_feat, return_phase)
    
    def get_current_weights(self, vision_feat: torch.Tensor, 
                           tactile_feat: torch.Tensor) -> Dict:
        """获取当前权重配置"""
        if self.mode == 'learnable':
            return self.learnable_gating.get_phase_weights(vision_feat, tactile_feat)
        else:
            return {'mode': 'hardcoded', 'weights': '取决于提供的phase参数'}


# ==================== 辅助函数 ====================

def detect_phase_from_proprioception(proprioception: torch.Tensor, 
                                     target_pos: Optional[torch.Tensor] = None,
                                     gripper_force: Optional[torch.Tensor] = None) -> TaskPhase:
    """
    基于本体感觉信息检测当前阶段
    启发式规则版本
    
    Args:
        proprioception: [B, D] 机器人状态 (包含末端位置、速度等)
        target_pos: [B, 3] 目标位置 (可选)
        gripper_force: [B] 夹爪力 (可选)
    Returns:
        phase: TaskPhase
    """
    B = proprioception.shape[0]
    
    # 简化的启发式规则
    # 实际使用时需要根据proprioception的具体定义调整
    
    # 假设proprioception包含: [x, y, z, vx, vy, vz, gripper_width, ...]
    gripper_width = proprioception[:, 6] if proprioception.shape[1] > 6 else torch.ones(B)
    
    if gripper_force is not None:
        # 有力反馈说明在接触
        force_threshold = 0.5
        is_contact = gripper_force > force_threshold
        
        if is_contact.any():
            # 进一步区分Contact和Manipulate
            if gripper_width.mean() > 0.05:  # 夹爪较开: 接触阶段
                return TaskPhase.CONTACT
            else:  # 夹爪闭合: 操作阶段
                return TaskPhase.MANIPULATE
    
    # 无接触时，根据距离判断
    if target_pos is not None:
        current_pos = proprioception[:, :3]
        distance = torch.norm(current_pos - target_pos, dim=1).mean()
        
        if distance > 0.1:  # > 10cm
            return TaskPhase.APPROACH
        elif distance < 0.05:  # < 5cm 且无力
            return TaskPhase.RETRACT
    
    # 默认返回Approach
    return TaskPhase.APPROACH


# ==================== 测试代码 ====================

def test_phase_gating():
    """测试阶段感知门控"""
    print("=" * 60)
    print("测试 Phase-Aware Modality Gating")
    print("=" * 60)
    
    batch_size = 4
    feature_dim = 512
    
    # 创建测试特征
    vision_feat = torch.randn(batch_size, feature_dim)
    tactile_feat = torch.randn(batch_size, feature_dim)
    
    print(f"\n输入特征:")
    print(f"  - 视觉特征: {vision_feat.shape}")
    print(f"  - 触觉特征: {tactile_feat.shape}")
    
    # 测试1: 硬编码模式
    print("\n" + "-" * 40)
    print("测试1: 硬编码模式")
    print("-" * 40)
    
    hard_gating = PhaseAwareModalityFusion(feature_dim, mode='hardcoded')
    
    for phase in TaskPhase:
        fused, info = hard_gating(vision_feat, tactile_feat, phase=phase, return_phase=True)
        weights = info['weights']
        print(f"  {phase.name:12} | Vision: {weights['vision']:.1f} | "
              f"Tactile: {weights['tactile']:.1f} | Output: {fused.shape}")
    
    # 测试2: 可学习模式
    print("\n" + "-" * 40)
    print("测试2: 可学习模式")
    print("-" * 40)
    
    learnable_gating = PhaseAwareModalityFusion(feature_dim, mode='learnable')
    
    fused, phase_logits = learnable_gating(vision_feat, tactile_feat, return_phase=True)
    print(f"  融合输出: {fused.shape}")
    print(f"  阶段预测: {phase_logits.shape}")
    
    # 获取权重分析
    weights_info = learnable_gating.get_current_weights(vision_feat, tactile_feat)
    print(f"\n  预测权重:")
    print(f"    - 视觉权重: {weights_info['vision_weight'].mean().item():.3f} ± "
          f"{weights_info['vision_weight'].std().item():.3f}")
    print(f"    - 触觉权重: {weights_info['tactile_weight'].mean().item():.3f} ± "
          f"{weights_info['tactile_weight'].std().item():.3f}")
    print(f"    - 预测阶段: {weights_info['predicted_phase'].tolist()}")
    
    # 参数量统计
    total_params = sum(p.numel() for p in learnable_gating.parameters())
    print(f"\n  可学习模式参数量: {total_params:,}")
    
    # 推理速度测试
    import time
    learnable_gating.eval()
    with torch.no_grad():
        for _ in range(10):  # Warmup
            _ = learnable_gating(vision_feat, tactile_feat)
        
        start = time.time()
        for _ in range(100):
            _ = learnable_gating(vision_feat, tactile_feat)
        end = time.time()
        
        avg_time = (end - start) / 100 * 1000
        print(f"\n  推理速度: {avg_time:.2f} ms")
    
    print("\n" + "=" * 60)
    print("✅ Phase-Aware Modality Gating 测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    test_phase_gating()
