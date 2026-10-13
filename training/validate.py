"""Validation: deterministic diffusion loss (fixed noise & stratified timesteps) on the REAL validation split."""
from __future__ import annotations

import contextlib
from typing import Optional

import torch

from training.checkpoint import EMA


@torch.no_grad()
def validate(model, loader, device: torch.device, amp_dtype: Optional[torch.dtype], ema: Optional[EMA] = None,
             max_batches: int = 20, seed: int = 1234, **loss_kwargs) -> float:
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    ctx = ema.applied(model) if ema is not None else contextlib.nullcontext()
    with ctx:
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            B, T = batch["target"].shape[0], model.scheduler.T
            t = ((torch.arange(B) * T) // B + (bi * 37) % max(T // B, 1)).clamp(max=T - 1)
            g = torch.Generator(device="cpu").manual_seed(seed + bi)
            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    loss = model.training_loss(batch, cond_dropout=0, sev_dropout=0, generator=g, fixed_t=t, **loss_kwargs)
            else:
                loss = model.training_loss(batch, cond_dropout=0, sev_dropout=0, generator=g, fixed_t=t, **loss_kwargs)
            total += float(loss) * B
            n += B
    model.train(was_training)
    return total / max(n, 1)
