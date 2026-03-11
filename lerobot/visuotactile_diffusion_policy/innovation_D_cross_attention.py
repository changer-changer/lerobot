"""
创新点D: 视触觉Cross-Attention融合 (VT-CAF)
Vision-Tactile Cross-Attention Fusion

核心创新:
1. 使用Cross-Attention让视觉和触觉相互查询
2. 双向注意力: Visual->Tactile 和 Tactile->Visual
3. 参考3D-ViTac + ForceVLA的后融合思想

Author: Dr. Sigma
Date: 2026-03-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention融合模块
    支持双向注意力
    """
    def __init__(self, 
                 dim: int = 512,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV投影
        self.q_proj_vision = nn.Linear(dim, dim)
        self.k_proj_vision = nn.Linear(dim, dim)
        self.v_proj_vision = nn.Linear(dim, dim)
        
        self.q_proj_tactile = nn.Linear(dim, dim)
        self.k_proj_tactile = nn.Linear(dim, dim)
        self.v_proj_tactile = nn.Linear(dim, dim)
        
        # 输出投影
        self.out_proj_vision = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.out_proj_tactile = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        
        # LayerNorm
        self.norm_vision = nn.LayerNorm(dim)
        self.norm_tactile = nn.LayerNorm(dim)
        
    def forward(self, 
                vision_feat: torch.Tensor, 
                tactile_feat: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        Args:
            vision_feat: [B, N_v, D] 视觉特征 (N_v可以是1或序列)
            tactile_feat: [B, N_t, D] 触觉特征 (N_t可以是1或序列)
            return_attention: 是否返回注意力权重
        Returns:
            vision_out: [B, N_v, D] 融合后的视觉特征
            tactile_out: [B, N_t, D] 融合后的触觉特征
            attention_weights: (optional) 注意力权重
        """
        B, N_v, D = vision_feat.shape
        N_t = tactile_feat.shape[1]
        
        # Vision attends to Tactile
        q_v = self.q_proj_vision(vision_feat).reshape(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)
        k_t = self.k_proj_tactile(tactile_feat).reshape(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_t = self.v_proj_tactile(tactile_feat).reshape(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_v2t = (q_v @ k_t.transpose(-2, -1)) * self.scale
        attn_v2t = attn_v2t.softmax(dim=-1)
        out_v = (attn_v2t @ v_t).transpose(1, 2).reshape(B, N_v, D)
        out_v = self.out_proj_vision(out_v)
        vision_out = self.norm_vision(vision_feat + out_v)
        
        # Tactile attends to Vision
        q_t = self.q_proj_tactile(tactile_feat).reshape(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)
        k_v = self.k_proj_vision(vision_feat).reshape(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)
        v_v = self.v_proj_vision(vision_feat).reshape(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_t2v = (q_t @ k_v.transpose(-2, -1)) * self.scale
        attn_t2v = attn_t2v.softmax(dim=-1)
        out_t = (attn_t2v @ v_v).transpose(1, 2).reshape(B, N_t, D)
        out_t = self.out_proj_tactile(out_t)
        tactile_out = self.norm_tactile(tactile_feat + out_t)
        
        if return_attention:
            return vision_out, tactile_out, (attn_v2t, attn_t2v)
        return vision_out, tactile_out, None


class VTCAFModule(nn.Module):
    """
    视触觉Cross-Attention融合模块 (完整版)
    Vision-Tactile Cross-Attention Fusion
    """
    def __init__(self,
                 vision_dim: int = 512,
                 tactile_dim: int = 512,
                 output_dim: int = 512,
                 num_ca_layers: int = 2,
                 num_heads: int = 8):
        super().__init__()
        
        # 输入投影 (统一维度)
        self.vision_proj = nn.Linear(vision_dim, 512)
        self.tactile_proj = nn.Linear(tactile_dim, 512)
        
        # Cross-Attention层堆叠
        self.ca_layers = nn.ModuleList([
            CrossAttentionFusion(512, num_heads) 
            for _ in range(num_ca_layers)
        ])
        
        # 最终融合
        self.fusion = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        
    def forward(self, 
                vision_feat: torch.Tensor,
                tactile_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_feat: [B, D_v] 或 [B, N_v, D_v] 视觉特征
            tactile_feat: [B, D_t] 或 [B, N_t, D_t] 触觉特征
        Returns:
            fused: [B, output_dim] 融合特征
        """
        # 添加序列维度 (如果需要)
        if vision_feat.dim() == 2:
            vision_feat = vision_feat.unsqueeze(1)  # [B, 1, D]
        if tactile_feat.dim() == 2:
            tactile_feat = tactile_feat.unsqueeze(1)  # [B, 1, D]
        
        # 投影到统一维度
        vision = self.vision_proj(vision_feat)  # [B, N_v, 512]
        tactile = self.tactile_proj(tactile_feat)  # [B, N_t, 512]
        
        # 多层Cross-Attention
        for ca_layer in self.ca_layers:
            vision, tactile, _ = ca_layer(vision, tactile)
        
        # 池化 (平均池化)
        vision_pooled = vision.mean(dim=1)  # [B, 512]
        tactile_pooled = tactile.mean(dim=1)  # [B, 512]
        
        # 最终融合
        combined = torch.cat([vision_pooled, tactile_pooled], dim=-1)  # [B, 1024]
        fused = self.fusion(combined)  # [B, output_dim]
        
        return fused
    
    def forward_with_attention(self, 
                               vision_feat: torch.Tensor,
                               tactile_feat: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """前向传播并返回注意力权重 (用于可视化)"""
        if vision_feat.dim() == 2:
            vision_feat = vision_feat.unsqueeze(1)
        if tactile_feat.dim() == 2:
            tactile_feat = tactile_feat.unsqueeze(1)
        
        vision = self.vision_proj(vision_feat)
        tactile = self.tactile_proj(tactile_feat)
        
        attention_weights = []
        for ca_layer in self.ca_layers:
            vision, tactile, attn = ca_layer(vision, tactile, return_attention=True)
            attention_weights.append(attn)
        
        vision_pooled = vision.mean(dim=1)
        tactile_pooled = tactile.mean(dim=1)
        combined = torch.cat([vision_pooled, tactile_pooled], dim=-1)
        fused = self.fusion(combined)
        
        return fused, attention_weights


class VTCAFWithGating(nn.Module):
    """
    VT-CAF + 门控机制 (结合创新点B)
    """
    def __init__(self,
                 vision_dim: int = 512,
                 tactile_dim: int = 512,
                 output_dim: int = 512,
                 use_gating: bool = True):
        super().__init__()
        self.use_gating = use_gating
        
        # Cross-Attention融合
        self.caf = VTCAFModule(vision_dim, tactile_dim, 512)
        
        # 门控网络 (可选)
        if use_gating:
            self.gate_network = nn.Sequential(
                nn.Linear(vision_dim + tactile_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 2),  # [w_v, w_t]
                nn.Softmax(dim=-1),
            )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, vision_feat: torch.Tensor, tactile_feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            vision_feat: [B, D_v]
            tactile_feat: [B, D_t]
        Returns:
            fused: [B, output_dim]
        """
        # Cross-Attention融合
        caf_out = self.caf(vision_feat, tactile_feat)  # [B, 512]
        
        if self.use_gating:
            # 预测门控权重
            combined = torch.cat([vision_feat, tactile_feat], dim=-1)
            gates = self.gate_network(combined)  # [B, 2]
            
            # 应用门控 (这里简化处理，实际可更复杂)
            # 可以设计为调整CAF内部特征的权重
            # 这里仅作为示例
            pass
        
        # 输出
        output = self.output_proj(caf_out)
        return output


# ==================== 测试代码 ====================

def test_cross_attention_fusion():
    """测试Cross-Attention融合"""
    print("=" * 60)
    print("测试 Cross-Attention Fusion")
    print("=" * 60)
    
    batch_size = 4
    vision_dim = 512
    tactile_dim = 512
    
    # 创建测试数据
    vision_feat = torch.randn(batch_size, 1, vision_dim)  # [B, 1, D]
    tactile_feat = torch.randn(batch_size, 10, tactile_dim)  # [B, 10, D] (Tac3D序列)
    
    print(f"\n输入:")
    print(f"  - 视觉特征: {vision_feat.shape}")
    print(f"  - 触觉特征: {tactile_feat.shape}")
    
    # 测试基础CA模块
    print("\n" + "-" * 40)
    print("测试基础Cross-Attention")
    print("-" * 40)
    
    ca = CrossAttentionFusion(dim=512, num_heads=8)
    vision_out, tactile_out, attn = ca(vision_feat, tactile_feat, return_attention=True)
    
    print(f"  视觉输出: {vision_out.shape}")
    print(f"  触觉输出: {tactile_out.shape}")
    if attn:
        attn_v2t, attn_t2v = attn
        print(f"  注意力权重 (V->T): {attn_v2t.shape}")
        print(f"  注意力权重 (T->V): {attn_t2v.shape}")
    
    # 测试完整VT-CAF
    print("\n" + "-" * 40)
    print("测试完整VT-CAF模块")
    print("-" * 40)
    
    vt_caf = VTCAFModule(vision_dim, tactile_dim, output_dim=512, num_ca_layers=2)
    
    # 简化输入 (无序列维度)
    vision_simple = torch.randn(batch_size, vision_dim)
    tactile_simple = torch.randn(batch_size, tactile_dim)
    
    output = vt_caf(vision_simple, tactile_simple)
    print(f"  输入: 视觉{vision_simple.shape}, 触觉{tactile_simple.shape}")
    print(f"  输出: {output.shape}")
    
    # 带注意力的前向
    output_attn, attention_weights = vt_caf.forward_with_attention(vision_simple, tactile_simple)
    print(f"\n  带注意力的输出: {output_attn.shape}")
    print(f"  注意力层数: {len(attention_weights)}")
    
    # 参数量
    total_params = sum(p.numel() for p in vt_caf.parameters())
    print(f"\n  参数量: {total_params:,}")
    
    # 推理速度
    import time
    vt_caf.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = vt_caf(vision_simple, tactile_simple)
        
        start = time.time()
        for _ in range(100):
            _ = vt_caf(vision_simple, tactile_simple)
        end = time.time()
        
        avg_time = (end - start) / 100 * 1000
        print(f"  推理速度: {avg_time:.2f} ms")
    
    # 测试带门控的版本
    print("\n" + "-" * 40)
    print("测试VT-CAF + 门控")
    print("-" * 40)
    
    vt_caf_gated = VTCAFWithGating(vision_dim, tactile_dim, output_dim=512, use_gating=True)
    output_gated = vt_caf_gated(vision_simple, tactile_simple)
    print(f"  输出: {output_gated.shape}")
    
    total_params_gated = sum(p.numel() for p in vt_caf_gated.parameters())
    print(f"  参数量: {total_params_gated:,}")
    
    print("\n" + "=" * 60)
    print("✅ Vision-Tactile Cross-Attention Fusion 测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    test_cross_attention_fusion()
