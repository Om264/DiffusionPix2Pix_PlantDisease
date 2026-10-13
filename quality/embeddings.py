"""Image embedders used for source-similarity, near-duplicate detection and evaluation.

    dinov2       transformers Dinov2Model (CLS token) - recommended (e.g. facebook/dinov2-small / -base);
                 ``model_name`` may be a hub id or a LOCAL directory.
    clip         transformers CLIPModel image features
    handcrafted  training-free colour/gradient/layout descriptor; weak, only for offline tests and sanity checks.

All embedders return L2-normalised float32 arrays of shape (N, D).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import cv2
import numpy as np
import torch

from common.errors import UserError

log = logging.getLogger(__name__)


def l2n(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)).astype(np.float32)


class Embedder(ABC):
    name = "base"

    @abstractmethod
    def embed(self, imgs: Sequence[np.ndarray], batch_size: int = 32, normalize: bool = True) -> np.ndarray:  # pragma: no cover
        ...


class HandcraftedEmbedder(Embedder):
    name = "handcrafted"

    def embed(self, imgs: Sequence[np.ndarray], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        out = []
        for im in imgs:
            im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
            h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
            g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
            gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
            mag, ang = np.hypot(gx, gy), (np.arctan2(gy, gx) % np.pi)
            ho, _ = np.histogram(ang, bins=12, range=(0, np.pi), weights=mag)
            layout = cv2.resize(im, (6, 6), interpolation=cv2.INTER_AREA).astype(np.float32).flatten() / 255
            out.append(np.concatenate([h / max(h.sum(), 1), ho / max(ho.sum(), 1e-6), layout]))
        return l2n(np.stack(out))  # always normalised


class _HFEmbedder(Embedder):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    size = 224

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            self.model = self._load(model_name).to(self.device).eval()
        except Exception as e:
            raise UserError(
                f"Could not load the {self.name} model '{model_name}' ({type(e).__name__}: {e}).\n"
                "Check your internet connection / Hugging Face access, pass a LOCAL model directory, or use "
                "--embedder handcrafted (weak, for offline sanity checks only).") from e

    def _prep(self, im: np.ndarray) -> torch.Tensor:
        x = cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (x - np.array(self.mean, np.float32)) / np.array(self.std, np.float32)
        return torch.from_numpy(x).permute(2, 0, 1)

    @torch.no_grad()
    def embed(self, imgs: Sequence[np.ndarray], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        out = []
        for s in range(0, len(imgs), batch_size):
            x = torch.stack([self._prep(i) for i in imgs[s:s + batch_size]]).to(self.device)
            out.append(self._forward(x).float().cpu().numpy())
        if not out:
            return np.zeros((0, 1), np.float32)
        f = np.concatenate(out)
        return l2n(f) if normalize else f.astype(np.float32)


class DinoEmbedder(_HFEmbedder):
    name = "dinov2"

    def _load(self, name):
        from transformers import AutoModel
        return AutoModel.from_pretrained(name)

    def _forward(self, x):
        return self.model(pixel_values=x).last_hidden_state[:, 0]


class ClipEmbedder(_HFEmbedder):
    name = "clip"
    mean = (0.4815, 0.4578, 0.4082)
    std = (0.2686, 0.2613, 0.2758)

    def _load(self, name):
        from transformers import CLIPModel
        return CLIPModel.from_pretrained(name)

    def _forward(self, x):
        return self.model.get_image_features(pixel_values=x)


def build_embedder(name: str = "dinov2", model_name: Optional[str] = None, device: Optional[str] = None) -> Embedder:
    if name == "handcrafted":
        return HandcraftedEmbedder()
    if name == "dinov2":
        return DinoEmbedder(model_name or "facebook/dinov2-small", device)
    if name == "clip":
        return ClipEmbedder(model_name or "openai/clip-vit-base-patch32", device)
    raise UserError(f"Unknown embedder '{name}' (dinov2 | clip | handcrafted)")
