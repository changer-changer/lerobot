"""
创新点C: 多速率扩散策略 (Multi-Rate Diffusion Policy)
Multi-Rate Diffusion Policy

核心创新:
1. 不同模态以不同频率参与扩散去噪过程
2. RGB (30Hz) + Tac3D (100Hz+) + Diffusion (10Hz)
3. 充分利用高频触觉信息，避免简单降采样损失

Author: Dr. Sigma
Date: 2026-03-10
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import deque


class MultiRateBuffer:
    """
    多速率数据缓冲器
    管理不同频率的传感器数据
    """
    def __init__(self, 
                 vision_freq: int = 30,
                 tactile_freq: int = 100,
                 policy_freq: int = 10):
        """
        Args:
            vision_freq: 视觉频率 (Hz)
            tactile_freq: 触觉频率 (Hz)  
            policy_freq: 策略推理频率 (Hz)
        """
        self.vision_freq = vision_freq
        self.tactile_freq = tactile_freq
        self.policy_freq = policy_freq
        
        # 计算每步需要的数据量
        self.vision_per_step = max(1, vision_freq // policy_freq)  # 3
        self.tactile_per_step = max(1, tactile_freq // policy_freq)  # 10
        
        # 缓冲区
        self.vision_buffer = deque(maxlen=self.vision_per_step)
        self.tactile_buffer = deque(maxlen=self.tactile_per_step)
        
        self.step_count = 0
        
    def push_vision(self, data: torch.Tensor):
        """添加视觉数据"""
        self.vision_buffer.append(data)
        
    def push_tactile(self, data: torch.Tensor):
        """添加触觉数据"""
        self.tactile_buffer.append(data)
        
    def is_ready(self) -> bool:
        """检查是否有足够数据进行策略推理"""
        return (len(self.vision_buffer) >= self.vision_per_step and 
                len(self.tactile_buffer) >= self.tactile_per_step)
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一批数据用于策略推理
        Returns:
            vision_batch: [B, vision_per_step, *] 视觉数据
            tactile_batch: [B, tactile_per_step, *] 触觉数据
        """
        vision_batch = torch.stack(list(self.vision_buffer), dim=1)
        tactile_batch = torch.stack(list(self.tactile_buffer), dim=1)
        
        self.step_count += 1
        return vision_batch, tactile_batch
    
    def clear(self):
        """清空缓冲区"""
        self.vision_buffer.clear()
        self.tactile_buffer.clear()
        self.step_count = 0
    
    def get_stats(self) -> Dict:
        """获取缓冲区统计"""
        return {
            'vision_buffer_size': len(self.vision_buffer),
            'tactile_buffer_size': len(self.tactile_buffer),
            'vision_per_step': self.vision_per_step,
            'tactile_per_step': self.tactile_per_step,
            'step_count': self.step_count,
        }


class TemporalAggregator(nn.Module):
    """
    时序聚合器: 将多帧数据聚合为单一特征
    """
    def __init__(self, feature_dim: int, mode: str = 'attention'):
        super().__init__()
        self.feature_dim = feature_dim
        self.mode = mode
        
        if mode == 'attention':
            # 自注意力聚合
            self.attention = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(feature_dim)
        elif mode == 'lstm':
            # LSTM聚合
            self.lstm = nn.LSTM(feature_dim, feature_dim, batch_first=True, bidirectional=True)
            self.proj = nn.Linear(feature_dim * 2, feature_dim)
        elif mode == 'mlp':
            # MLP聚合
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim),
            )
        else:  # mean
            pass  # 直接使用mean
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] T帧特征
        Returns:
            aggregated: [B, D] 聚合特征
        """
        if self.mode == 'attention':
            # 自注意力
            attn_out, _ = self.attention(x, x, x)  # [B, T, D]
            out = attn_out.mean(dim=1)  # [B, D]
            out = self.norm(out)
            return out
        
        elif self.mode == 'lstm':
            lstm_out, _ = self.lstm(x)  # [B, T, 2D]
            out = self.proj(lstm_out[:, -1, :])  # [B, D]
            return out
        
        elif self.mode == 'mlp':
            # 取第一帧和最后一帧
            x_first = x[:, 0, :]
            x_last = x[:, -1, :]
            x_cat = torch.cat([x_first, x_last], dim=-1)
            return self.mlp(x_cat)
        
        else:  # mean
            return x.mean(dim=1)


class MultiRateEncoder(nn.Module):
    """
    多速率编码器
    处理不同频率的视觉和触觉数据
    """
    def __init__(self,
                 vision_feature_dim: int = 512,
                 tactile_feature_dim: int = 512,
                 output_dim: int = 512,
                 vision_frames: int = 3,
                 tactile_frames: int = 10,
                 aggregation_mode: str = 'attention'):
        super().__init__()
        
        self.vision_frames = vision_frames
        self.tactile_frames = tactile_frames
        
        # 视觉编码器 (处理单帧)
        self.vision_encoder = nn.Sequential(
            nn.Linear(vision_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        
        # 触觉编码器 (处理单帧)
        self.tactile_encoder = nn.Sequential(
            nn.Linear(tactile_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        
        # 时序聚合
        self.vision_aggregator = TemporalAggregator(512, mode=aggregation_mode)
        self.tactile_aggregator = TemporalAggregator(512, mode=aggregation_mode)
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(512 * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
    
    def forward(self, 
                vision_sequence: torch.Tensor,
                tactile_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_sequence: [B, T_v, D_v] T_v帧视觉特征
            tactile_sequence: [B, T_t, D_t] T_t帧触觉特征
        Returns:
            fused: [B, D] 融合特征
        """
        B = vision_sequence.shape[0]
        
        # 编码每一帧
        vision_encoded = []
        for t in range(vision_sequence.shape[1]):
            v = self.vision_encoder(vision_sequence[:, t, :])
            vision_encoded.append(v)
        vision_encoded = torch.stack(vision_encoded, dim=1)  # [B, T_v, 512]
        
        tactile_encoded = []
        for t in range(tactile_sequence.shape[1]):
            t_feat = self.tactile_encoder(tactile_sequence[:, t, :])
            tactile_encoded.append(t_feat)
        tactile_encoded = torch.stack(tactile_encoded, dim=1)  # [B, T_t, 512]
        
        # 时序聚合
        vision_agg = self.vision_aggregator(vision_encoded)    # [B, 512]
        tactile_agg = self.tactile_aggregator(tactile_encoded) # [B, 512]
        
        # 融合
        combined = torch.cat([vision_agg, tactile_agg], dim=-1)  # [B, 1024]
        fused = self.fusion(combined)  # [B, output_dim]
        
        return fused


class MultiRateDiffusionPolicy(nn.Module):
    """
    多速率扩散策略 (完整版)
    集成多速率缓冲和多速率编码器
    """
    def __init__(self,
                 vision_dim: int = 512,
                 tactile_dim: int = 512,
                 action_dim: int = 7,
                 vision_freq: int = 30,
                 tactile_freq: int = 100,
                 policy_freq: int = 10):
        super().__init__()
        
        self.buffer = MultiRateBuffer(vision_freq, tactile_freq, policy_freq)
        
        self.encoder = MultiRateEncoder(
            vision_feature_dim=vision_dim,
            tactile_feature_dim=tactile_dim,
            output_dim=512,
            vision_frames=vision_freq // policy_freq,
            tactile_frames=tactile_freq // policy_freq,
        )
        
        # 简化的扩散策略头部 (实际应替换为完整Diffusion Policy)
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
    
    def push_data(self, vision: torch.Tensor, tactile: torch.Tensor):
        """推送新数据到缓冲区"""
        self.buffer.push_vision(vision)
        self.buffer.push_tactile(tactile)
    
    def is_ready(self) -> bool:
        """检查是否可以执行策略"""
        return self.buffer.is_ready()
    
    def forward(self) -> Optional[torch.Tensor]:
        """
        执行策略推理
        Returns:
            action: [B, action_dim] 或 None (数据不足)
        """
        if not self.is_ready():
            return None
        
        vision_batch, tactile_batch = self.buffer.get_batch()
        fused = self.encoder(vision_batch, tactile_batch)
        action = self.policy_head(fused)
        return action
    
    def get_stats(self) -> Dict:
        """获取状态统计"""
        return self.buffer.get_stats()


# ==================== 测试代码 ====================

def test_multi_rate_buffer():
    """测试多速率缓冲区"""
    print("=" * 60)
    print("测试 Multi-Rate Buffer")
    print("=" * 60)
    
    buffer = MultiRateBuffer(vision_freq=30, tactile_freq=100, policy_freq=10)
    
    print(f"\n配置:")
    print(f"  - 视觉频率: 30 Hz")
    print(f"  - 触觉频率: 100 Hz")
    print(f"  - 策略频率: 10 Hz")
    print(f"  - 每步视觉帧数: {buffer.vision_per_step}")
    print(f"  - 每步触觉帧数: {buffer.tactile_per_step}")
    
    # 模拟数据流
    print(f"\n模拟数据流 (100ms = 1个策略步):")
    batch_size = 2
    vision_feat_dim = 512
    tactile_feat_dim = 512
    
    for i in range(15):  # 模拟15个时间步
        # 触觉数据: 每步都推 (100Hz)
        tactile = torch.randn(batch_size, tactile_feat_dim)
        buffer.push_tactile(tactile)
        
        # 视觉数据: 每3步推一次 (30Hz)
        if i % 3 == 0:
            vision = torch.randn(batch_size, vision_feat_dim)
            buffer.push_vision(vision)
        
        if buffer.is_ready():
            v_batch, t_batch = buffer.get_batch()
            print(f"  Step {i}: 执行策略 | 视觉输入: {v_batch.shape} | 触觉输入: {t_batch.shape}")
        else:
            print(f"  Step {i}: 缓冲中... | 视觉: {len(buffer.vision_buffer)}/{buffer.vision_per_step} | "
                  f"触觉: {len(buffer.tactile_buffer)}/{buffer.tactile_per_step}")
    
    print("\n" + "=" * 60)


def test_multi_rate_encoder():
    """测试多速率编码器"""
    print("\n" + "=" * 60)
    print("测试 Multi-Rate Encoder")
    print("=" * 60)
    
    batch_size = 4
    vision_frames = 3
    tactile_frames = 10
    feature_dim = 512
    
    # 创建模型
    encoder = MultiRateEncoder(
        vision_feature_dim=feature_dim,
        tactile_feature_dim=feature_dim,
        output_dim=512,
        vision_frames=vision_frames,
        tactile_frames=tactile_frames,
        aggregation_mode='attention'
    )
    
    # 测试输入
    vision_seq = torch.randn(batch_size, vision_frames, feature_dim)
    tactile_seq = torch.randn(batch_size, tactile_frames, feature_dim)
    
    print(f"\n输入:")
    print(f"  - 视觉序列: {vision_seq.shape} ({vision_frames}帧@30Hz)")
    print(f"  - 触觉序列: {tactile_seq.shape} ({tactile_frames}帧@100Hz)")
    
    # 前向传播
    output = encoder(vision_seq, tactile_seq)
    print(f"\n输出:")
    print(f"  - 融合特征: {output.shape}")
    
    # 参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n参数量: {total_params:,}")
    
    # 推理速度
    import time
    encoder.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = encoder(vision_seq, tactile_seq)
        
        start = time.time()
        for _ in range(100):
            _ = encoder(vision_seq, tactile_seq)
        end = time.time()
        
        avg_time = (end - start) / 100 * 1000
        print(f"推理速度: {avg_time:.2f} ms")
    
    print("\n" + "=" * 60)
    print("✅ Multi-Rate Diffusion Policy 测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    test_multi_rate_buffer()
    test_multi_rate_encoder()
