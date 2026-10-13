"""EMA, atomic checkpoint save/load, checkpoint path resolution."""
from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from common.errors import UserError
from models.diffusion_pix2pix import DiffusionPix2Pix, ModelConfig

log = logging.getLogger(__name__)


class EMA:
    """Exponential moving average of the trainable parameters (the VAE, if any, is frozen and excluded)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    @contextlib.contextmanager
    def applied(self, model: torch.nn.Module):
        """Temporarily load EMA weights into ``model``."""
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])
        try:
            yield model
        finally:
            for n, p in model.named_parameters():
                if n in backup:
                    p.data.copy_(backup[n])

    def copy_to(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


def save_checkpoint(path: Union[str, Path], state: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: an interrupted save never corrupts the previous checkpoint


def resolve_checkpoint(spec: Union[str, Path]) -> Path:
    """Accepts a .pt file, a directory (best.pt > latest.pt), or a path stem such as ``.../checkpoints/best``."""
    p = Path(spec)
    cands = [p, p.with_suffix(".pt"), Path(str(p) + ".pt")]
    if p.is_dir():
        cands = [p / "best.pt", p / "latest.pt"]
    for c in cands:
        if c.is_file():
            return c
    raise UserError(f"Checkpoint not found: {spec}\nTrain a model first (python -m training.train --config "
                    "configs/training.yaml) or pass --checkpoint path/to/best.pt")


def load_checkpoint(spec: Union[str, Path], map_location: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    return torch.load(resolve_checkpoint(spec), map_location=map_location, weights_only=False)


def load_model_from_checkpoint(spec: Union[str, Path], device: torch.device, use_ema: bool = True
                               ) -> Tuple[DiffusionPix2Pix, Dict[str, Any]]:
    ck = load_checkpoint(spec, "cpu")
    cfg = ModelConfig.from_dict(ck["model_cfg"])
    model = DiffusionPix2Pix(cfg, len(ck["class_to_idx"]))
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    bad = [k for k in missing if not k.startswith("vae.")] + list(unexpected)
    if bad:
        raise UserError(f"Checkpoint does not match the model definition (keys: {bad[:5]} ...). "
                        "Was it trained with a different model.preset / latent setting?")
    if use_ema and ck.get("ema"):
        sd = model.state_dict()
        for k, v in ck["ema"].items():
            if k in sd:
                sd[k].copy_(v)
    return model.to(device).eval(), ck
