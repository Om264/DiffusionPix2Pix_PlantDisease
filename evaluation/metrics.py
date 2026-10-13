"""Generative-quality metrics. Read the README section "Metrics: what they mean and when they are NOT valid".

FD / KID    distribution distances between sets of feature vectors (real vs synthetic).
SSIM / PSNR pixel/structure similarity between a synthetic image and its SOURCE image, computed on the background
            (should be ~perfect) and inside the leaf (SHOULD differ: that is where disease is synthesised).
LPIPS       learned perceptual distance (needs the `lpips` package + AlexNet weights); source-distance and
            within-class diversity.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg

from common.errors import UserError

log = logging.getLogger(__name__)


def frechet_distance(f1: np.ndarray, f2: np.ndarray, eps: float = 1e-6) -> float:
    """Squared Frechet distance between Gaussians fitted to two feature sets (the 'FID' formula).

    Unreliable when n < feature dimension (singular covariance, strong positive bias)."""
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    s1, s2 = np.atleast_2d(s1), np.atleast_2d(s2)
    covmean, _ = scipy.linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def kernel_inception_distance(f1: np.ndarray, f2: np.ndarray, n_subsets: int = 50, subset_size: int = 100,
                              seed: int = 0) -> Tuple[float, float]:
    """Unbiased MMD^2 with the cubic polynomial kernel (Binkowski et al.), averaged over random subsets.
    Returns (mean, std). Preferred over FD for small sample sizes."""
    rng = np.random.default_rng(seed)
    m = min(subset_size, len(f1), len(f2))
    if m < 2:
        return float("nan"), float("nan")
    d = f1.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = f1[rng.choice(len(f1), m, replace=False)]
        y = f2[rng.choice(len(f2), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def masked_psnr(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float:
    """PSNR over pixels where ``region`` is True. Returns inf if identical."""
    if not region.any():
        return float("nan")
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2)[region].mean())
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def masked_ssim(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    if not region.any():
        return float("nan")
    win = min(7, min(a.shape[:2]) // 2 * 2 - 1)
    _, smap = structural_similarity(a, b, channel_axis=2, full=True, win_size=win, data_range=255)
    return float(smap.mean(-1)[region].mean())


class LPIPSMetric:
    """LPIPS (AlexNet). Needs ``pip install lpips`` and internet access once (torchvision AlexNet weights)."""

    def __init__(self, device: str = "cpu", pretrained_backbone: bool = True):
        try:
            import lpips
            import torch
            self.torch = torch
            self.net = lpips.LPIPS(net="alex", pnet_rand=not pretrained_backbone, verbose=False).to(device).eval()
        except Exception as e:
            raise UserError(f"LPIPS is unavailable ({type(e).__name__}: {e}). Install with `pip install lpips` and make sure "
                            "torchvision can download AlexNet weights, or run with --no_lpips.") from e
        self.device = device

    def _t(self, x: np.ndarray):
        return self.torch.from_numpy(x.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(self.device)

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        with self.torch.no_grad():
            return float(self.net(self._t(a), self._t(b)).item())


def cosine_stats(a: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"mean": float("nan"), "std": float("nan"), "p05": float("nan"), "median": float("nan")}
    return {"mean": float(a.mean()), "std": float(a.std()), "p05": float(np.percentile(a, 5)), "median": float(np.median(a))}


def get_feature_extractor(name: str, model_name: Optional[str] = None, device: str = "cpu"):
    """Returns an object with ``features(list_of_uint8_rgb) -> (N, D) float array`` (NOT L2-normalised)."""
    if name == "inception":
        try:
            import torch
            from torchmetrics.image.fid import NoTrainInceptionV3
            net = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
        except Exception as e:
            raise UserError(f"Inception features unavailable ({type(e).__name__}: {e}).\nInstall `torchmetrics` and "
                            "`torch-fidelity` (and allow one-time weight download from GitHub), or use "
                            "--feature_extractor dinov2.") from e

        class _Inc:
            dim, label = 2048, "FID (Inception-v3 pool3)"

            @torch.no_grad()
            def features(self, imgs: Sequence[np.ndarray], bs: int = 32) -> np.ndarray:
                import cv2
                out = []
                for s in range(0, len(imgs), bs):
                    x = torch.from_numpy(np.stack([cv2.resize(i, (299, 299), interpolation=cv2.INTER_LINEAR) for i in imgs[s:s + bs]]))
                    out.append(net(x.permute(0, 3, 1, 2).to(device)).float().cpu().numpy())
                return np.concatenate(out)
        return _Inc()
    from quality.embeddings import build_embedder
    emb = build_embedder(name, model_name, device)

    class _Emb:
        label = f"Frechet distance on {name} features (NOT comparable to published FID)"

        def features(self, imgs: Sequence[np.ndarray], bs: int = 32) -> np.ndarray:
            return emb.embed(imgs, bs, normalize=False)
    e = _Emb()
    return e
