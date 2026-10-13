"""Create binary leaf masks (1 = disease may be synthesised, 0 = background / keep).

    python -m preprocessing.prepare_masks --processed_dir ./data/processed --method hsv
    python -m preprocessing.prepare_masks --processed_dir ./data/processed --method existing --masks_dir ./my_masks
    python -m preprocessing.prepare_masks --processed_dir ./data/processed --method torchscript --model_path seg.pt
    python -m preprocessing.prepare_masks --processed_dir ./data/processed --segmenter_module mypkg.seg:MySegmenter

Methods: hsv | grabcut | existing (your masks / manual masks) | torchscript | full | custom class via --segmenter_module.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm

from common.errors import UserError, run_cli
from common.imageio_utils import load_mask, load_rgb, save_mask, save_rgb
from common.logging_utils import setup_logging
from dataio.masks import build_segmenter
from dataio.metadata import load_metadata, save_metadata

log = logging.getLogger("prepare_masks")


def prepare_masks(processed_dir: str, method: str = "hsv", masks_dir: Optional[str] = None,
                  model_path: Optional[str] = None, segmenter_module: Optional[str] = None, overwrite: bool = False,
                  min_coverage: float = 0.05, max_coverage: float = 0.98, fallback: str = "full",
                  preview: int = 12, device: str = "cpu") -> None:
    root = Path(processed_dir)
    meta = load_metadata(root)
    seg = build_segmenter(method, segmenter_module, masks_dir=masks_dir, model_path=model_path, device=device)
    fb = build_segmenter("hsv" if fallback == "hsv" else "full")
    prev = []
    n_fallback = 0
    for i, r in tqdm(list(meta.iterrows()), desc=f"masks ({seg.name})"):
        if isinstance(r["mask_path"], str) and (root / r["mask_path"]).exists() and not overwrite:
            continue
        img = load_rgb(root / r["path"])
        src = seg.name
        try:
            m = seg.segment(img, class_name=r["class"], stem=Path(r["path"]).stem.split("__", 1)[-1])
        except FileNotFoundError:
            m = None
        cov = float(m.mean()) if m is not None else 0.0
        if m is None or cov < min_coverage or cov > max_coverage:
            if fallback == "skip":
                continue
            m, src, n_fallback = fb.segment(img), f"fallback_{fb.name}", n_fallback + 1
            cov = float(m.mean())
        rel = Path("masks") / r["split"] / r["class"] / f"{r['image_id']}.png"
        save_mask(m, root / rel)
        meta.at[i, "mask_path"] = str(rel).replace("\\", "/")
        meta.at[i, "mask_source"] = src
        meta.at[i, "mask_coverage"] = cov
        if len(prev) < preview:
            prev.append((img, m))
    save_metadata(meta, root)
    got = meta["mask_path"].notna().sum()
    log.info("Masks written for %d/%d images (%d used fallback '%s').", got, len(meta), n_fallback, fallback)
    if n_fallback > 0.2 * len(meta):
        log.warning("%d%% of masks fell back to '%s'. Check the segmentation method (masks_preview.png) or supply "
                    "your own masks/model: background pixels may be modified by the generator otherwise.",
                    int(100 * n_fallback / len(meta)), fallback)
    if prev:
        tiles = []
        for img, m in prev:
            im = cv2.resize(img, (128, 128)); mm = cv2.resize(m * 255, (128, 128), interpolation=cv2.INTER_NEAREST)
            ov = (0.5 * im + 0.5 * np.stack([mm, np.zeros_like(mm), np.zeros_like(mm)], -1)).astype(np.uint8)
            tiles.append(np.concatenate([im, ov], 0))
        save_rgb(np.concatenate(tiles, 1), root / "masks_preview.png")
        log.info("Wrote %s (top: image, bottom: mask overlay in red)", root / "masks_preview.png")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--method", default="hsv", choices=["hsv", "grabcut", "existing", "torchscript", "full"])
    ap.add_argument("--masks_dir", default=None)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--segmenter_module", default=None, help="'pkg.module:ClassName' (subclass of LeafSegmenter)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--min_coverage", type=float, default=0.05)
    ap.add_argument("--max_coverage", type=float, default=0.98)
    ap.add_argument("--fallback", choices=["full", "hsv", "skip"], default="full")
    ap.add_argument("--preview", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    setup_logging(Path(a.processed_dir) / "logs" / "prepare_masks.log")
    prepare_masks(**vars(a))


if __name__ == "__main__":
    run_cli(main)
