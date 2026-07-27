"""Small decoder-only transformer, plain PyTorch: RoPE, causal attention, SwiGLU, RMSNorm, tied embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 384
    block_size: int = 512
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def rotary_freqs(head_dim: int, max_len: int, base: float = 10000.0) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv)                      # (T, head_dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D) -> pair last dim as complex, rotate, unpair
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    xr = xc * freqs[: x.shape[-2]].unsqueeze(0).unsqueeze(0)
    return torch.view_as_real(xr).flatten(-2).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rotary(q, freqs), apply_rotary(k, freqs)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(8 / 3 * cfg.n_embd / 64) * 64  # round to multiple of 64
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs)
        return x + self.mlp(self.norm2(x))


class VerilogGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        freqs = rotary_freqs(cfg.n_embd // cfg.n_head, cfg.block_size)
        self.register_buffer("freqs", freqs, persistent=False)
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * self.cfg.n_layer))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.freqs)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
