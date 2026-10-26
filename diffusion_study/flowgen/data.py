"""CIFAR-10 as GPU tensors in [-1, 1]. Downloads via torchvision into `root` on first use."""

from __future__ import annotations

import numpy as np
import torch


def load_cifar10(root: str, train: bool = True, device: str = "cuda"):
    """-> (images uint8 [N,3,32,32] on device, labels long [N] on device)."""
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root=root, train=train, download=True)
    x = torch.from_numpy(np.asarray(ds.data)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x.to(device), y.to(device)


def to_float(x_uint8: torch.Tensor) -> torch.Tensor:
    return x_uint8.float() / 127.5 - 1.0


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
