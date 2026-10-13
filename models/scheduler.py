"""Noise scheduler: forward process q(x_t | x_0), training targets, and DDIM sampling steps.

Supports 'epsilon' and 'v_prediction' parameterisations and linear / scaled_linear / cosine beta schedules.
Own implementation (a few lines each) so that the maths is explicit and unit-testable.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def make_betas(schedule: str, n: int, beta_start: float, beta_end: float) -> torch.Tensor:
    if schedule == "linear":
        b = np.linspace(beta_start, beta_end, n, dtype=np.float64)
    elif schedule == "scaled_linear":
        b = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n, dtype=np.float64) ** 2
    elif schedule == "cosine":
        s = 0.008
        t = np.arange(n + 1, dtype=np.float64) / n
        f = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        b = np.clip(1 - f[1:] / f[:-1], 0, 0.999)
    else:
        raise ValueError(f"Unknown beta_schedule '{schedule}'")
    return torch.from_numpy(b)


class NoiseScheduler(nn.Module):
    def __init__(self, num_train_timesteps: int = 1000, beta_schedule: str = "cosine", beta_start: float = 1e-4,
                 beta_end: float = 0.02, prediction_type: str = "epsilon"):
        super().__init__()
        if prediction_type not in ("epsilon", "v_prediction"):
            raise ValueError("prediction_type must be 'epsilon' or 'v_prediction'")
        self.T, self.prediction_type = num_train_timesteps, prediction_type
        betas = make_betas(beta_schedule, num_train_timesteps, beta_start, beta_end)
        ac = torch.cumprod(1.0 - betas, 0)
        self.register_buffer("alphas_cumprod", ac.float(), persistent=False)

    def _ac(self, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        a = self.alphas_cumprod.to(like.device)[t.long().clamp(0, self.T - 1)]
        return a.view(-1, *([1] * (like.ndim - 1))).to(like.dtype)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = self._ac(t, x0)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def get_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.prediction_type == "epsilon":
            return noise
        a = self._ac(t, x0)
        return a.sqrt() * noise - (1 - a).sqrt() * x0  # v = sqrt(a) eps - sqrt(1-a) x0

    def predict_x0_eps(self, out: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        a = self._ac(t, x_t)
        if self.prediction_type == "epsilon":
            eps = out
            x0 = (x_t - (1 - a).sqrt() * eps) / a.sqrt()
        else:
            x0 = a.sqrt() * x_t - (1 - a).sqrt() * out
            eps = (1 - a).sqrt() * x_t + a.sqrt() * out
        return x0, eps

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        a = self.alphas_cumprod.to(t.device)[t.long()]
        return a / (1 - a)

    def ddim_timesteps(self, steps: int, strength: float = 1.0) -> torch.Tensor:
        """Descending integer timesteps. ``strength`` < 1 starts from a partially-noised input (SDEdit-style)."""
        ts = np.linspace(self.T - 1, 0, steps).round().astype(np.int64)
        if strength < 1.0:
            ts = ts[ts <= int(round(strength * (self.T - 1)))]
        return torch.from_numpy(ts)

    def ddim_step(self, out: torch.Tensor, t: int, t_prev: int, x_t: torch.Tensor, eta: float = 0.0,
                  x0_clip: Optional[float] = 1.0, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        tt = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        x0, eps = self.predict_x0_eps(out, x_t, tt)
        if x0_clip is not None:
            x0 = x0.clamp(-x0_clip, x0_clip)
            eps = (x_t - self._ac(tt, x_t).sqrt() * x0) / (1 - self._ac(tt, x_t)).sqrt()
        a_t = self.alphas_cumprod[t].to(x_t.device)
        a_p = self.alphas_cumprod[t_prev].to(x_t.device) if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
        sigma = eta * ((1 - a_p) / (1 - a_t)).sqrt() * (1 - a_t / a_p).clamp(min=0).sqrt()
        x_prev = a_p.sqrt() * x0 + (1 - a_p - sigma ** 2).clamp(min=0).sqrt() * eps
        if eta > 0 and t_prev >= 0:
            noise = torch.randn(x_t.shape, device="cpu", generator=generator).to(x_t.device, x_t.dtype)
            x_prev = x_prev + sigma * noise
        return x_prev
