"""Per-image quality checks for synthetic images. All thresholds live in configs/evaluation.yaml (``quality:``) and
are HEURISTICS that you should calibrate by inspecting accepted/rejected samples of YOUR data."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from common.imageio_utils import load_mask, load_rgb
from dataio.metadata import is_healthy_class
from dataio.transforms import lesion_fraction

log = logging.getLogger(__name__)


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian of the grayscale image (higher = sharper). Resolution dependent."""
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def unnatural_hue_fraction(img: np.ndarray, leaf: np.ndarray) -> float:
    """Fraction of saturated leaf pixels with blue/purple/cyan hue (OpenCV hue 100-170), implausible for leaves."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    odd = (hsv[..., 0] >= 100) & (hsv[..., 0] <= 170) & (hsv[..., 1] > 60) & (hsv[..., 2] > 40)
    n = max(int((leaf > 0).sum()), 1)
    return float((odd & (leaf > 0)).sum() / n)


def check_one(row: Dict[str, Any], root: Path, q: Dict[str, Any], healthy_pattern: str = "healthy") -> Dict[str, Any]:
    """Return metrics + list of rejection reasons for one generated image."""
    m: Dict[str, Any] = {"image_id": row["image_id"]}
    reasons: List[str] = []
    try:
        syn = load_rgb(root / row["path"])
        src = load_rgb(root / row["source_crop"])
        mask = load_mask(root / row["mask"])
    except Exception as e:  # corrupt / missing
        m.update(reasons="invalid_image", passed=False, error=str(e)[:120])
        return m
    h, w = syn.shape[:2]
    m.update(height=h, width=w)
    if min(h, w) < q["min_size"]:
        reasons.append("low_resolution")
    if src.shape != syn.shape or mask.shape != syn.shape[:2]:
        m.update(reasons="shape_mismatch", passed=False)
        return m

    # --- blur / exposure -------------------------------------------------------------------------------------
    bs, bsrc = sharpness(syn), sharpness(src)
    m["sharpness"], m["sharpness_ratio"] = bs, bs / max(bsrc, 1e-6)
    if bs < q["blur_min"] or m["sharpness_ratio"] < q["blur_ratio_to_source"]:
        reasons.append("blurry")
    if m["sharpness_ratio"] > q.get("sharpness_ratio_max", 8.0):
        reasons.append("noisy_or_artifacts")  # far sharper than the source: typically residual noise / high-freq artifacts
    gray = cv2.cvtColor(syn, cv2.COLOR_RGB2GRAY)
    m["brightness"] = float(gray.mean())
    m["saturated_frac"] = float(((gray <= 3) | (gray >= 252)).mean())
    if not (q["brightness"][0] <= m["brightness"] <= q["brightness"][1]) or m["saturated_frac"] > q["saturated_frac_max"]:
        reasons.append("bad_exposure")

    # --- mask consistency ------------------------------------------------------------------------------------
    diff = np.abs(syn.astype(np.int16) - src.astype(np.int16)).max(-1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bg = cv2.dilate(mask, k) == 0                     # background with a small safety margin around the mask
    inside = mask > 0
    m["bg_l1"] = float(np.abs(syn.astype(np.float32) - src.astype(np.float32))[bg].mean()) if bg.any() else 0.0
    m["outside_change_frac"] = float((diff[bg] > 20).mean()) if bg.any() else 0.0
    m["inside_change_frac"] = float((diff[inside] > 25).mean()) if inside.any() else 0.0
    if m["bg_l1"] > q["bg_l1_max"] or m["outside_change_frac"] > q["outside_change_max"]:
        reasons.append("background_modified")
    if inside.any() and m["inside_change_frac"] < q["min_change_frac"]:
        reasons.append("no_change_in_mask")
    if inside.any() and m["inside_change_frac"] > q["max_change_frac"]:
        reasons.append("over_modified")

    # --- disease-region consistency (colour heuristic) --------------------------------------------------------
    ls, lg = lesion_fraction(src, mask), lesion_fraction(syn, mask)
    m["lesion_src"], m["lesion_syn"], m["lesion_delta"] = ls, lg, lg - ls
    if is_healthy_class(row["class"], healthy_pattern):
        if lg > q["healthy_lesion_max"]:
            reasons.append("healthy_has_lesions")
    elif m["lesion_delta"] < q["lesion_delta_min"]:
        reasons.append("no_disease_signal")
    m["unnatural_hue_frac"] = unnatural_hue_fraction(syn, mask)
    if m["unnatural_hue_frac"] > q["unnatural_hue_frac_max"]:
        reasons.append("unnatural_colors")

    m["reasons"] = ";".join(reasons)
    m["passed"] = not reasons
    return m


def quality_score(df: pd.DataFrame) -> pd.Series:
    """Higher = better. Similarity to the source plus a mild sharpness term; used to rank images (dedup, trimming)."""
    sim = df.get("source_sim", pd.Series(0.0, index=df.index)).fillna(0.0)
    return sim + 0.2 * df["sharpness_ratio"].clip(upper=1.0).fillna(0.0)


def select_per_class(df: pd.DataFrame, quotas: Dict[str, int]) -> pd.DataFrame:
    """Keep the best ``quotas[class]`` accepted images per class (by quality_score)."""
    keep = []
    for c, g in df.groupby("class"):
        keep.append(g.sort_values("quality_score", ascending=False).head(quotas.get(c, len(g))))
    return pd.concat(keep) if keep else df.iloc[0:0]
