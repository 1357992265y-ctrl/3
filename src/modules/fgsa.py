#!/usr/bin/env python3
"""
Fine-Grained Style Aggregation (FGSA)
=====================================

c实现论文中描述的FGSA模块。

处理步骤
----------------
1. **输入**
   * `content_feats` `(B, C, H, W)` – 内容特征图（查询）
   * `style_feats` `(B, C, H, W)` – 风格特征图（键值对）
     若存在多个风格参考字形，可预先将其聚合（如取平均值）至相同形状。
2. 沿通道维度交错排列两个张量 →
   `F_cs ∈ ℝ^{B×2C×H×W}`。
3. 应用**1×1分组卷积**（`groups=C`，每组2→1）得到
   `F_a ∈ ℝ^{B×C×H×W}`。
4. 通过**7×7深度可分离卷积**（DW + PW）处理，随后应用
   `sigmoid`函数得到注意力图`A ∈ ℝ^{B×C×H×W}`。
5. 元素级乘法：`V' = A ⊗ style_feats`。
6. 残差相加：`F_r = content_feats + V'`，最终返回结果。



"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch


class FGSA(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1,\
            "kernel_size 必须为奇数才能实现对称填充"
        padding = kernel_size // 2

        # 1×1 分组卷积：输入 2C，输出 C，groups=C（每组2个通道→1）
        self.group_conv = nn.Conv2d(
            in_channels=2 * channels,
            out_channels=channels,
            kernel_size=1,
            groups=channels,
            bias=True,
        )

        # 深度可分离卷积
        self.dw_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,  # depthwise
            bias=True,
        )
        self.pw_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            bias=True,
        )

        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _interleave_channels(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Interleave two tensors of shape `(B,C,H,W)` into one `(B,2C,H,W)` along the channel dimension."""
        B, C, H, W = a.shape
        # stack -> (B, C, 2, H, W)
        stack = torch.stack([a, b], dim=2)
        # reshape to (B, 2C, H, W) with interleaved channels
        interleaved = stack.view(B, 2 * C, H, W)
        return interleaved

    def forward(self, content_feats: torch.Tensor, style_feats: torch.Tensor) -> torch.Tensor:
        """c前向传播。

        参数：
            content_feats: (B, C, H, W) 内容特征图（查询）
            style_feats: (B, C, H, W) 风格特征图（键值对）

        返回值：
            out_feats: (B, C, H, W) 聚合特征 F_r
        """
        # 1. interleave content & style features
        f_cs = self._interleave_channels(content_feats, style_feats)

        # 2. grouped 1×1 convolution
        f_a = self.group_conv(f_cs)

        # 3. depth-wise separable convolution (DW + PW)
        x = self.dw_conv(f_a)
        x = self.pw_conv(x)

        # 4. sigmoid → attention map A
        attn = self.sigmoid(x)

        # 5. weight style features
        fused = attn * style_feats

        # 6. residual addition
        out = fused + content_feats
        return out

class MultiRegionFGSA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fgsa = FGSA(channels)

    def forward(self, content, style, masks):
        """
        content: (B,C,H,W)
        style:   (B,C,H,W)
        masks:   list of region masks [(1,1,H,W), ...]
        """
        B, C, H, W = content.shape
        fused = torch.zeros_like(content)

        for m in masks:
            m = m.to(content.device)
            region_c = content * m
            region_s = style * m
            fused_region = self.fgsa(region_c, region_s)
            fused += fused_region * m  # 仅覆盖区域融合

        return fused

if __name__ == "__main__":
    # quick sanity test
    B, C, H, W = 2, 64, 32, 32
    fgsa = FGSA(C)
    q = torch.randn(B, C, H, W)
    k = torch.randn(B, C, H, W)
    out = fgsa(q, k)
    print(out.shape)  # Expected (2, 64, 32, 32)