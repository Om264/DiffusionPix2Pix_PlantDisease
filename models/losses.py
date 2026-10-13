"""Diffusion training loss with mask-aware spatial weighting and optional Min-SNR weighting."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from models.scheduler import NoiseScheduler


def diffusion_loss(pred: torch.Tensor, target: torch.Tensor, edit_mask: torch.Tensor, t: torch.Tensor,
                   scheduler: NoiseScheduler, edit_weight: float = 1.0, bg_weight: float = 0.5,
                   min_snr_gamma: Optional[float] = None) -> torch.Tensor:
    """MSE between model output and target, weighted ``edit_weight`` inside the edit mask and ``bg_weight`` outside.

    ``edit_mask`` is (B,1,H,W) at *any* resolution (resized to the prediction). Outside the mask the source equals
    the target, so that region is easy; a lower ``bg_weight`` focuses capacity on where disease is synthesised."""
    m = F.interpolate(edit_mask.float(), size=pred.shape[-2:], mode="nearest")
    w = bg_weight + (edit_weight - bg_weight) * m
    se = (pred.float() - target.float()) ** 2
    per = (se * w).sum((1, 2, 3)) / (w.sum((1, 2, 3)) * pred.shape[1]).clamp(min=1e-6)
    if min_snr_gamma:
        snr = scheduler.snr(t).float()
        wt = torch.minimum(snr, torch.full_like(snr, min_snr_gamma))
        wt = wt / snr if scheduler.prediction_type == "epsilon" else wt / (snr + 1)
        per = per * wt
    return per.mean()
