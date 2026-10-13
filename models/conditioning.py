"""Conditioning encoders.

ImageEncoder      source image (+ binary mask) -> a few global tokens (the spatial detail reaches the UNet through
                  channel concatenation; the tokens give the cross-attention layers a summary of the plant prior)
ConditionEncoder  (source, mask, disease id, severity) -> cross-attention tokens + a global vector that is added to
                  the diffusion-timestep embedding.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.embeddings import DiseaseEmbedding, SeverityEmbedding


class ImageEncoder(nn.Module):
    def __init__(self, in_ch: int, dim: int, grid: int = 4, width: int = 32):
        super().__init__()
        g = 8 if width % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, 2, 1), nn.GroupNorm(g, width), nn.SiLU(),
            nn.Conv2d(width, width * 2, 3, 2, 1), nn.GroupNorm(g, width * 2), nn.SiLU(),
            nn.Conv2d(width * 2, width * 2, 3, 2, 1), nn.GroupNorm(g, width * 2), nn.SiLU(),
            nn.AdaptiveAvgPool2d(grid))
        self.proj = nn.Linear(width * 2, dim)
        self.pos = nn.Parameter(torch.randn(1, grid * grid, dim) * 0.02)
        self.n_tokens = grid * grid

    def forward(self, source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        f = self.net(torch.cat([source, mask], 1))               # (B, C, g, g)
        return self.proj(f.flatten(2).transpose(1, 2)) + self.pos  # (B, g*g, dim)


class ConditionEncoder(nn.Module):
    def __init__(self, num_classes: int, token_dim: int, global_dim: Optional[int], img_channels: int,
                 grid: int = 4, severity_freqs: int = 8):
        super().__init__()
        self.disease = DiseaseEmbedding(num_classes, token_dim)
        self.severity = SeverityEmbedding(token_dim, severity_freqs)
        self.image = ImageEncoder(img_channels + 1, token_dim, grid)
        self.global_mlp = (nn.Sequential(nn.Linear(2 * token_dim, global_dim), nn.SiLU(), nn.Linear(global_dim, global_dim))
                           if global_dim else None)

    @property
    def null_class_id(self) -> int:
        return self.disease.null_id

    def forward(self, source_feat: torch.Tensor, mask_feat: torch.Tensor, class_ids: torch.Tensor,
                severity: torch.Tensor, sev_known: torch.Tensor, drop_class: Optional[torch.Tensor] = None,
                drop_sev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``drop_class`` (bool, B): replace disease with the null class AND forget severity (classifier-free dropout).
        ``drop_sev`` (bool, B): forget severity only."""
        if drop_class is not None:
            class_ids = torch.where(drop_class, torch.full_like(class_ids, self.null_class_id), class_ids)
            sev_known = sev_known.bool() & ~drop_class
        if drop_sev is not None:
            sev_known = sev_known.bool() & ~drop_sev
        d = self.disease(class_ids)
        s = self.severity(severity, sev_known)
        tokens = torch.cat([d.unsqueeze(1), s.unsqueeze(1), self.image(source_feat, mask_feat)], 1)
        glob = self.global_mlp(torch.cat([d, s], -1)) if self.global_mlp is not None else None
        return tokens, glob
