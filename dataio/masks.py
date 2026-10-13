"""Leaf/plant segmentation back-ends. All segmenters share one tiny interface so you can plug in your own.

A segmenter maps an RGB uint8 image (H, W, 3) to a uint8 mask (H, W) with 1 = leaf/plant (disease may be
synthesised here) and 0 = background (must not be modified).
"""
from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Type

import cv2
import numpy as np

from common.errors import UserError
from common.imageio_utils import load_mask

log = logging.getLogger(__name__)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    pad = np.pad(m, 1)
    flood = pad.copy() * 255
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 128)
    holes = (flood[1:-1, 1:-1] == 0)
    return np.where(holes, 1, m).astype(np.uint8)


def postprocess_mask(mask: np.ndarray, min_area_frac: float = 0.01, k_frac: float = 0.01) -> np.ndarray:
    """Open/close, drop tiny components, fill holes."""
    m = (mask > 0).astype(np.uint8)
    k = max(3, int(round(min(m.shape) * k_frac)) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_frac * m.size:
            keep[lab == i] = 1
    return fill_holes(keep)


class LeafSegmenter(ABC):
    """Interface. Subclass and register (or pass ``--segmenter_module pkg.mod:Class``) to use your own model."""
    name = "base"

    @abstractmethod
    def segment(self, img_rgb: np.ndarray, **kwargs) -> np.ndarray:  # pragma: no cover - interface
        ...


class HSVSegmenter(LeafSegmenter):
    """Classical, training-free segmentation: colour distance to the border colour + vegetation colours.

    Works for images with a fairly uniform background (e.g. lab/PlantVillage-style). For cluttered field
    images use a learned model (``torchscript``) or provide masks (``existing``).
    """
    name = "hsv"

    def __init__(self, border_frac: float = 0.04, min_area_frac: float = 0.02):
        self.border_frac, self.min_area_frac = border_frac, min_area_frac

    def _foreground_prob(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        b = max(2, int(min(img.shape[:2]) * self.border_frac))
        border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                                 lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
        bg = np.median(border, axis=0)
        dist = np.linalg.norm(lab - bg, axis=-1)
        return np.clip(dist / 255.0 * 4.0, 0, 1)

    def segment(self, img_rgb: np.ndarray, **kwargs) -> np.ndarray:
        d = (self._foreground_prob(img_rgb) * 255).astype(np.uint8)
        d = cv2.GaussianBlur(d, (0, 0), 1.5)
        thr, _ = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = (d > max(thr, 20)).astype(np.uint8)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        green = ((hsv[..., 0] >= 25) & (hsv[..., 0] <= 95) & (hsv[..., 1] > 40) & (hsv[..., 2] > 30)).astype(np.uint8)
        m = np.maximum(m, green) if m.mean() > 0.02 else green  # fall back to vegetation colours
        return postprocess_mask(m, self.min_area_frac)


class GrabCutSegmenter(LeafSegmenter):
    """HSV segmentation refined by OpenCV GrabCut (slower, usually tighter boundaries)."""
    name = "grabcut"

    def __init__(self, iters: int = 3, work_size: int = 256):
        self.iters, self.work_size, self._init = iters, work_size, HSVSegmenter()

    def segment(self, img_rgb: np.ndarray, **kwargs) -> np.ndarray:
        h, w = img_rgb.shape[:2]
        s = self.work_size / max(h, w)
        small = cv2.resize(img_rgb, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        init = self._init.segment(small)
        if init.sum() < 16 or init.mean() > 0.97:
            return cv2.resize(init, (w, h), interpolation=cv2.INTER_NEAREST)
        gc = np.full(init.shape, cv2.GC_PR_BGD, np.uint8)
        gc[init > 0] = cv2.GC_PR_FGD
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        gc[cv2.erode(init, k) > 0] = cv2.GC_FGD
        gc[cv2.dilate(init, k, iterations=2) == 0] = cv2.GC_BGD
        try:
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(small, gc, None, bgd, fgd, self.iters, cv2.GC_INIT_WITH_MASK)
            m = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
        except cv2.error:
            m = init
        m = postprocess_mask(m)
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


class TorchScriptSegmenter(LeafSegmenter):
    """Plug-in for a learned model exported with ``torch.jit.save``.

    Contract: input float32 (1,3,H,W) in [0,1] (ImageNet-normalised if ``imagenet_norm``), output logits/probabilities
    (1,1,H,W) or (1,H,W). Foreground = sigmoid(output) > 0.5 (if the output is already in [0,1] it is used as is).
    """
    name = "torchscript"

    def __init__(self, model_path: str, size: int = 512, device: str = "cpu", imagenet_norm: bool = True):
        import torch
        if not model_path or not Path(model_path).exists():
            raise UserError(f"TorchScript segmentation model not found: {model_path}")
        self.torch, self.size, self.device = torch, size, device
        self.model = torch.jit.load(model_path, map_location=device).eval()
        self.imagenet_norm = imagenet_norm

    def segment(self, img_rgb: np.ndarray, **kwargs) -> np.ndarray:
        torch = self.torch
        h, w = img_rgb.shape[:2]
        x = cv2.resize(img_rgb, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if self.imagenet_norm:
            x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        t = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device)
        with torch.no_grad():
            y = self.model(t)
        y = y.float().reshape(1, 1, *y.shape[-2:])
        p = y if (float(y.min()) >= 0 and float(y.max()) <= 1) else torch.sigmoid(y)
        m = (p[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        return postprocess_mask(cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST))


class ExistingMaskSegmenter(LeafSegmenter):
    """Use masks the user already has: ``masks_dir/<class>/<stem>.png`` (or ``masks_dir/<stem>.png``,
    or ``<stem>_mask.png`` / ``<stem>.png`` in a parallel tree). Optional manual masks use the same layout."""
    name = "existing"

    def __init__(self, masks_dir: str):
        if not masks_dir or not Path(masks_dir).exists():
            raise UserError(f"Masks directory not found: {masks_dir}\nPass --masks_dir with your mask PNGs.")
        self.root = Path(masks_dir)

    def find(self, class_name: str, stem: str, src_path: Optional[str] = None) -> Optional[Path]:
        cands = []
        for base in (self.root / class_name, self.root):
            for suf in ("", "_mask", "_seg", "_label"):
                for ext in (".png", ".jpg", ".bmp", ".tif"):
                    cands.append(base / f"{stem}{suf}{ext}")
        for c in cands:
            if c.exists():
                return c
        return None

    def segment(self, img_rgb: np.ndarray, class_name: str = "", stem: str = "", **kwargs) -> np.ndarray:
        p = self.find(class_name, stem)
        if p is None:
            raise FileNotFoundError(f"no mask for {class_name}/{stem}")
        m = load_mask(p)
        if m.shape != img_rgb.shape[:2]:
            m = cv2.resize(m, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        return m


class FullImageSegmenter(LeafSegmenter):
    """Mask = whole image. Allowed disease region = everywhere (use only if the images are already cropped leaves)."""
    name = "full"

    def segment(self, img_rgb: np.ndarray, **kwargs) -> np.ndarray:
        return np.ones(img_rgb.shape[:2], np.uint8)


SEGMENTERS: Dict[str, Type[LeafSegmenter]] = {
    "hsv": HSVSegmenter, "grabcut": GrabCutSegmenter, "torchscript": TorchScriptSegmenter,
    "existing": ExistingMaskSegmenter, "full": FullImageSegmenter,
}


def build_segmenter(method: str, module: Optional[str] = None, **kwargs) -> LeafSegmenter:
    """``module='pkg.mod:ClassName'`` loads a user-defined :class:`LeafSegmenter` subclass."""
    if module:
        mod_name, cls_name = module.split(":")
        cls = getattr(importlib.import_module(mod_name), cls_name)
        return cls(**kwargs)
    if method not in SEGMENTERS:
        raise UserError(f"Unknown segmentation method '{method}'. Choose from {sorted(SEGMENTERS)}.")
    import inspect
    sig = inspect.signature(SEGMENTERS[method].__init__).parameters
    return SEGMENTERS[method](**{k: v for k, v in kwargs.items() if k in sig})
