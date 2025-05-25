"""
models/timesnet.py
------------------
A **very** thin PyTorch implementation of TimesNet (from "TimesNet:
Temporal 2D‑Variation Modeling for General Time Series Analysis", Wu et al.,
ICLR 2023) adapted for small‑feature, minute‑bar futures data.

This module intentionally leaves the training loop minimal so you can plug it
into Lightning, accelerate() or your own batch‑generator.  It provides:
    • TimesBlock – a single multi‑period filter block.
    • TimesNet    – stack of blocks + MLP head for classification.
    • timesnet_forward – helper to forward‑pass a (B, T, F) tensor.

Dependencies: pytorch>=2.1.0, einops (for rearrange).  Install with:
    pip install torch einops
"""

from __future__ import annotations
import torch
import torch.nn as nn
from einops import rearrange

# ------------------------------------------------------------------ #
# TimesBlock                                                         #
# ------------------------------------------------------------------ #
class TimesBlock(nn.Module):
    """One TimesNet block consisting of multi‑period 2D convolutions."""

    def __init__(self, in_channels: int, periods: list[int] = [24, 48, 96]):
        super().__init__()
        self.periods = periods
        self.convs   = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=(1, p), padding=(0, p // 2))
            for p in periods
        ])
        self.proj = nn.Conv1d(in_channels*len(periods), in_channels, kernel_size=1)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        outs = []
        for conv, p in zip(self.convs, self.periods):
            # reshape to (B*F, 1, T, 1) then conv along time as width
            y = rearrange(x, 'b t f -> (b f) 1 1 t')
            y = conv(y)                        # (B*F, 1, 1, T)
            y = rearrange(y, '(b f) 1 1 t -> b t f', f=x.size(-1))
            outs.append(y)
        y = torch.cat(outs, dim=-1)            # concat on feature dim
        y = rearrange(y, 'b t f -> b f t')     # (B, F_concat, T)
        y = self.proj(y)
        y = rearrange(y, 'b f t -> b t f')
        return self.act(y + x)                 # residual


# ------------------------------------------------------------------ #
# TimesNet                                                           #
# ------------------------------------------------------------------ #
class TimesNet(nn.Module):
    def __init__(self,
                 in_features: int,
                 num_classes: int = 3,
                 depth: int = 4,
                 hidden_dim: int = 64,
                 periods: list[int] = [24, 48, 96]):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.blocks = nn.ModuleList([
            TimesBlock(hidden_dim, periods) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        # global average over time dimension
        x = self.norm(x).mean(dim=1)
        return self.cls_head(x)


# ------------------------------------------------------------------ #
# Lightweight test stub                                              #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    B, T, F = 8, 120, 10   # batch, time‑steps, features
    model = TimesNet(in_features=F)
    dummy = torch.randn(B, T, F)
    out = model(dummy)
    print("Output logits:", out.shape)  # (B, 3)
