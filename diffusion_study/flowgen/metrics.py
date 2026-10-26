"""FID via the pytorch-fid Inception weights (pt_inception-2015-12-05), and a small CIFAR-10 classifier.

FID is computed between generated images and the 50,000 CIFAR-10 TRAIN images (statistics cached in `cache_dir`).
With 10,000 generated samples FID is biased upward relative to 50,000-sample FID and is only comparable within this
repo; it is NOT comparable with published CIFAR-10 numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg


class InceptionFeatures:
    def __init__(self, device="cuda"):
        from pytorch_fid.inception import InceptionV3
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, x_uint8: torch.Tensor, bs: int = 500) -> np.ndarray:
        feats = []
        for i in range(0, len(x_uint8), bs):
            b = x_uint8[i:i + bs].to(self.device).float() / 255.0           # pytorch-fid expects [0, 1]
            feats.append(self.net(b)[0].squeeze(-1).squeeze(-1).cpu().numpy())   # it resizes to 299 internally
        return np.concatenate(feats)


def stats(feats: np.ndarray):
    return feats.mean(0), np.cov(feats, rowvar=False)


def frechet(m1, s1, m2, s2) -> float:
    diff = m1 - m2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def reference_stats(inc: InceptionFeatures, images_uint8: torch.Tensor, cache: Path):
    if cache.exists():
        d = np.load(cache)
        return d["mu"], d["sigma"]
    mu, sigma = stats(inc(images_uint8))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, mu=mu, sigma=sigma)
    return mu, sigma


def fid(inc: InceptionFeatures, gen_uint8: torch.Tensor, ref) -> float:
    mu, sigma = stats(inc(gen_uint8))
    return frechet(mu, sigma, *ref)


class SmallResNet(nn.Module):
    """~1.2M-parameter classifier for CIFAR-10 (used only to measure class agreement of conditional samples)."""

    def __init__(self):
        super().__init__()
        def blk(i, o, pool):
            layers = [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU()]
            return nn.Sequential(*(layers + ([nn.MaxPool2d(2)] if pool else [])))
        self.f = nn.Sequential(blk(3, 64, False), blk(64, 128, True), blk(128, 128, False), blk(128, 256, True),
                               blk(256, 256, False), blk(256, 512, True), nn.AdaptiveMaxPool2d(1), nn.Flatten())
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.fc(self.f(x))
