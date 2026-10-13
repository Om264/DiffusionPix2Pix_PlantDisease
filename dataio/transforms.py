"""Joint (image, mask) geometry + construction of the *source image prior* used for training pairs.

Training pair construction ("pseudo_prior" strategy, see README section 'Training data requirements'):

    target  T : a real (possibly diseased) leaf image                          -> what the model must produce
    leaf    M : binary leaf mask
    edit    E : region in which the model may change pixels (E subset of M)     -> the conditioning binary mask
    source  S : T outside E; a disease-suppressed *prior* of T inside E         -> the conditioning image

The prior keeps leaf shape, lighting and low-frequency colour but removes lesion detail
(masked Gaussian blur + partial desaturation + blend toward the mean leaf colour).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
import torch


# ----------------------------------------------------------------------------- geometry
@dataclass
class CropParams:
    scale: float = 1.0      # side of the crop relative to the shorter image side
    ratio: float = 1.0      # width/height distortion before resizing (1 = square)
    cx: float = 0.5         # relative crop centre in [0, 1]
    cy: float = 0.5
    flip_h: bool = False
    flip_v: bool = False
    rot90: int = 0

    def to_dict(self) -> Dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def sample_crop_params(rng: np.random.Generator, enabled: bool = True, scale=(0.6, 1.0), ratio=(0.9, 1.1),
                       flip: bool = True, rot90: bool = True) -> CropParams:
    if not enabled:
        return CropParams()
    return CropParams(scale=float(rng.uniform(*scale)), ratio=float(rng.uniform(*ratio)),
                      cx=float(rng.uniform(0, 1)), cy=float(rng.uniform(0, 1)),
                      flip_h=bool(flip and rng.random() < 0.5), flip_v=bool(flip and rng.random() < 0.5),
                      rot90=int(rng.integers(0, 4)) if rot90 else 0)


def apply_geometry(img: np.ndarray, mask: np.ndarray, p: CropParams, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Crop (square-ish) -> resize to ``size`` -> flips / rot90. Deterministic given ``p``."""
    H, W = img.shape[:2]
    side = max(8.0, p.scale * min(H, W))
    cw, ch = min(W, side * np.sqrt(p.ratio)), min(H, side / np.sqrt(p.ratio))
    x0 = int(round((W - cw) * p.cx)); y0 = int(round((H - ch) * p.cy))
    x1, y1 = min(W, x0 + int(round(cw))), min(H, y0 + int(round(ch)))
    ci, cm = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    interp = cv2.INTER_AREA if min(ci.shape[:2]) > size else cv2.INTER_LINEAR
    ci = cv2.resize(ci, (size, size), interpolation=interp)
    cm = cv2.resize(cm, (size, size), interpolation=cv2.INTER_NEAREST)
    if p.flip_h:
        ci, cm = ci[:, ::-1], cm[:, ::-1]
    if p.flip_v:
        ci, cm = ci[::-1], cm[::-1]
    if p.rot90:
        ci, cm = np.rot90(ci, p.rot90), np.rot90(cm, p.rot90)
    return np.ascontiguousarray(ci), np.ascontiguousarray(cm)


# ----------------------------------------------------------------------------- source prior
@dataclass
class PriorConfig:
    sigma_frac: Tuple[float, float] = (0.02, 0.08)   # blur sigma as a fraction of the image side
    mix_mean: Tuple[float, float] = (0.0, 0.5)       # blend toward the mean leaf colour
    desat: Tuple[float, float] = (0.3, 1.0)          # saturation multiplier inside the edit region
    region_random_prob: float = 0.5                  # prob. of using a random sub-region of the leaf as edit mask
    region_min_frac: float = 0.15                    # min. fraction of the leaf covered by a random sub-region

    @classmethod
    def from_dict(cls, d: Dict) -> "PriorConfig":
        d = dict(d or {})
        for k in ("sigma_frac", "mix_mean", "desat"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def random_edit_region(leaf: np.ndarray, rng: np.random.Generator, cfg: PriorConfig) -> np.ndarray:
    """Union of 1-3 random ellipses intersected with the leaf; falls back to the full leaf if too small."""
    if leaf.sum() == 0 or rng.random() >= cfg.region_random_prob:
        return leaf.copy()
    ys, xs = np.nonzero(leaf)
    H, W = leaf.shape
    reg = np.zeros_like(leaf)
    for _ in range(int(rng.integers(1, 4))):
        j = int(rng.integers(0, len(ys)))
        ax = (int(rng.uniform(0.12, 0.4) * W), int(rng.uniform(0.12, 0.4) * H))
        cv2.ellipse(reg, (int(xs[j]), int(ys[j])), (max(ax[0], 2), max(ax[1], 2)),
                    float(rng.uniform(0, 180)), 0, 360, 1, -1)
    reg = reg * leaf
    return reg if reg.sum() >= cfg.region_min_frac * leaf.sum() else leaf.copy()


def build_source_prior(target: np.ndarray, leaf: np.ndarray, rng: np.random.Generator,
                       cfg: PriorConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return (source uint8 HxWx3, edit mask uint8 HxW). Outside the edit mask source == target exactly."""
    H, W = leaf.shape
    edit = random_edit_region(leaf, rng, cfg)
    if edit.sum() == 0:
        return target.copy(), edit
    sigma = float(rng.uniform(*cfg.sigma_frac)) * W
    m3 = leaf[..., None].astype(np.float32)
    t = target.astype(np.float32)
    num = cv2.GaussianBlur(t * m3, (0, 0), sigma)
    den = cv2.GaussianBlur(m3[..., 0], (0, 0), sigma)[..., None]
    prior = np.where(den > 1e-3, num / np.maximum(den, 1e-3), t)
    mean_col = (t * m3).sum((0, 1)) / max(m3.sum(), 1.0)
    a = float(rng.uniform(*cfg.mix_mean))
    prior = (1 - a) * prior + a * mean_col
    hsv = cv2.cvtColor(np.clip(prior, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= float(rng.uniform(*cfg.desat))
    prior = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    src = np.where(edit[..., None] > 0, prior, target).astype(np.uint8)
    return src, edit


def lesion_fraction(img: np.ndarray, leaf: np.ndarray) -> float:
    """Heuristic fraction of leaf pixels that are NOT healthy green (brown/yellow/necrotic/pale).

    Colour heuristic only (HSV). It is *not* an expert severity grade; used for estimated severity and QC."""
    if leaf.sum() == 0:
        return 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(np.int32), hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
    healthy = (h >= 33) & (h <= 90) & (s >= 50) & (v >= 40)
    return float(((~healthy) & (leaf > 0)).sum() / leaf.sum())


# ----------------------------------------------------------------------------- tensors
def to_tensor_m11(img_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_u8.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).contiguous()


def from_tensor_m11(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) or (B,3,H,W) tensor in [-1,1] -> uint8 (H,W,3) / (B,H,W,3)."""
    a = ((t.detach().float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
    return a.transpose(1, 2, 0) if a.ndim == 3 else a.transpose(0, 2, 3, 1)


def color_jitter(img: np.ndarray, rng: np.random.Generator, strength: float = 0.08) -> np.ndarray:
    if strength <= 0:
        return img
    b = 1 + rng.uniform(-strength, strength)
    c = 1 + rng.uniform(-strength, strength)
    f = img.astype(np.float32)
    f = (f - f.mean()) * c + f.mean()
    return np.clip(f * b, 0, 255).astype(np.uint8)
