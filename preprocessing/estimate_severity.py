"""Estimate a *severity condition* for images without expert severity labels.

IMPORTANT: estimated severity != ground truth. It is a colour heuristic: the fraction of leaf pixels that are not
healthy green, optionally converted to the within-class percentile (so that 0.2 ... 1.0 span the class's observed
range). Rows with real labels (severity_source == 'real') are never overwritten. Provenance is stored in the
``severity_source`` column: real | estimated | none.

    python -m preprocessing.estimate_severity --processed_dir ./data/processed
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from tqdm import tqdm

from common.imageio_utils import load_mask, load_rgb
from common.logging_utils import setup_logging
from common.errors import run_cli
from dataio.metadata import load_metadata, save_metadata
from dataio.transforms import lesion_fraction

log = logging.getLogger("estimate_severity")


def estimate_severity(processed_dir: str, normalize: str = "class_quantile", healthy_zero: bool = True,
                      work_size: int = 256) -> None:
    import cv2
    root = Path(processed_dir)
    meta = load_metadata(root)
    raw = np.full(len(meta), np.nan)
    for i, r in tqdm(list(meta.iterrows()), desc="estimating severity"):
        img = load_rgb(root / r["path"])
        m = load_mask(root / r["mask_path"]) if isinstance(r["mask_path"], str) else np.ones(img.shape[:2], np.uint8)
        s = work_size / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        raw[i] = lesion_fraction(img, m)
    meta["severity_raw"] = raw
    sev = raw.copy()
    if normalize == "class_quantile":  # percentile within the class, using TRAIN images only as reference
        for c, g in meta.groupby("class"):
            ref = np.sort(raw[g[g["split"] == "train"].index.values])
            if len(ref) == 0:
                continue
            sev[g.index.values] = np.searchsorted(ref, raw[g.index.values], side="right") / len(ref)
    elif normalize == "minmax":
        tr = raw[meta["split"] == "train"]
        sev = np.clip((raw - tr.min()) / max(tr.max() - tr.min(), 1e-6), 0, 1)
    elif normalize != "none":
        raise ValueError(normalize)
    if healthy_zero:
        sev[meta["is_healthy"].astype(bool).values] = 0.0
    mask_est = (meta["severity_source"] != "real").values
    meta.loc[mask_est, "severity"] = np.clip(sev[mask_est], 0, 1)
    meta.loc[mask_est, "severity_source"] = "estimated"
    save_metadata(meta, root)
    log.warning("Severity for %d images is ESTIMATED by a colour heuristic (%s), not ground truth. "
                "%d images keep real labels.", int(mask_est.sum()), normalize, int((~mask_est).sum()))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--normalize", choices=["class_quantile", "minmax", "none"], default="class_quantile")
    ap.add_argument("--no_healthy_zero", action="store_true")
    a = ap.parse_args(argv)
    setup_logging(Path(a.processed_dir) / "logs" / "estimate_severity.log")
    estimate_severity(a.processed_dir, a.normalize, not a.no_healthy_zero)


if __name__ == "__main__":
    run_cli(main)
