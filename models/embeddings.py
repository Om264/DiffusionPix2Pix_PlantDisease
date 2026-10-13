"""Embedding modules for the *discrete* (disease) and *continuous* (severity) conditions.

(The diffusion timestep embedding is produced inside diffusers' ``UNet2DConditionModel``: ``time_proj`` +
``time_embedding``; our condition vector is added to it through the UNet's ``class_embedding`` (projection).)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class DiseaseEmbedding(nn.Module):
    """class id -> vector. Index ``num_classes`` is the learned *null* class used for classifier-free guidance."""

    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.emb = nn.Embedding(num_classes + 1, dim)

    @property
    def null_id(self) -> int:
        return self.num_classes

    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(class_ids)


class FourierFeatures(nn.Module):
    """Scalar in [0,1] -> [s, sin(2*pi*2^k*s), cos(2*pi*2^k*s)]_k  (resolves small severity differences)."""

    def __init__(self, num_freqs: int = 8):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(num_freqs)) * 2 * math.pi, persistent=False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = s.float().unsqueeze(-1)
        ang = x * self.freqs
        return torch.cat([x, ang.sin(), ang.cos()], dim=-1)


class SeverityEmbedding(nn.Module):
    """Regression (graded) condition. Unknown / dropped severities map to a learned null vector."""

    def __init__(self, dim: int, num_freqs: int = 8):
        super().__init__()
        self.ff = FourierFeatures(num_freqs)
        self.mlp = nn.Sequential(nn.Linear(1 + 2 * num_freqs, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.null = nn.Parameter(torch.zeros(dim))

    def forward(self, severity: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        e = self.mlp(self.ff(severity.clamp(0, 1)))
        return torch.where(known.bool().unsqueeze(-1), e, self.null.expand_as(e).to(e.dtype))
