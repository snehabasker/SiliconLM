"""Small class-conditional U-Net for 32x32 images. Predicts the flow-matching velocity v(x_t, t, y).

Time and class enter every residual block through a FiLM (scale, shift) on the group-normed activations. Class index
`num_classes` is the "null" label used for classifier-free guidance (label dropout during training).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UNetConfig:
    in_ch: int = 3
    base: int = 96
    mults: tuple = (1, 2, 2)
    blocks: int = 2
    attn_levels: tuple = (1, 2)       # indices into mults at which self-attention is used (16x16 and 8x8)
    heads: int = 4
    num_classes: int = 10
    dropout: float = 0.1


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t in [0, 1] -> sinusoidal features of size dim."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = (t[:, None] * 1000.0) * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb: int, dropout: float):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(16, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.film = nn.Linear(emb, 2 * cout)
        self.n2, self.c2 = nn.GroupNorm(16, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(e)[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return self.skip(x) + h


class Attn(nn.Module):
    def __init__(self, ch: int, heads: int):
        super().__init__()
        self.heads, self.norm = heads, nn.GroupNorm(16, ch)
        self.qkv, self.proj = nn.Conv2d(ch, 3 * ch, 1), nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W).transpose(-1, -2).unbind(1)
        o = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(o.transpose(-1, -2).reshape(B, C, H, W))


class UNet(nn.Module):
    def __init__(self, cfg: UNetConfig = UNetConfig()):
        super().__init__()
        self.cfg, c = cfg, cfg.base
        emb = 4 * c
        self.t_mlp = nn.Sequential(nn.Linear(c, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.y_emb = nn.Embedding(cfg.num_classes + 1, emb)
        self.inp = nn.Conv2d(cfg.in_ch, c, 3, padding=1)
        self.down, chans, cur = nn.ModuleList(), [c], c
        for lvl, m in enumerate(cfg.mults):
            for _ in range(cfg.blocks):
                blk = nn.ModuleList([ResBlock(cur, c * m, emb, cfg.dropout)])
                cur = c * m
                if lvl in cfg.attn_levels:
                    blk.append(Attn(cur, cfg.heads))
                self.down.append(blk); chans.append(cur)
            if lvl < len(cfg.mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(cur, cur, 3, stride=2, padding=1)])); chans.append(cur)
        self.mid = nn.ModuleList([ResBlock(cur, cur, emb, cfg.dropout), Attn(cur, cfg.heads), ResBlock(cur, cur, emb, cfg.dropout)])
        self.up = nn.ModuleList()
        for lvl, m in reversed(list(enumerate(cfg.mults))):
            for i in range(cfg.blocks + 1):
                blk = nn.ModuleList([ResBlock(cur + chans.pop(), c * m, emb, cfg.dropout)])
                cur = c * m
                if lvl in cfg.attn_levels:
                    blk.append(Attn(cur, cfg.heads))
                if lvl > 0 and i == cfg.blocks:
                    blk.append(nn.Upsample(scale_factor=2, mode="nearest")); blk.append(nn.Conv2d(cur, cur, 3, padding=1))
                self.up.append(blk)
        self.out = nn.Sequential(nn.GroupNorm(16, cur), nn.SiLU(), nn.Conv2d(cur, cfg.in_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, y):
        e = self.t_mlp(timestep_embedding(t, self.cfg.base)) + self.y_emb(y)
        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            for layer in blk:
                h = layer(h, e) if isinstance(layer, ResBlock) else layer(h)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, e) if isinstance(layer, ResBlock) else layer(h)
        for blk in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in blk:
                h = layer(h, e) if isinstance(layer, ResBlock) else layer(h)
        return self.out(h)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
