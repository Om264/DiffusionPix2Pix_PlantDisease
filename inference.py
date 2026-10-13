"""Translate ONE real image (or a folder) with a trained checkpoint: source image + mask + disease + severity.

    python inference.py --checkpoint outputs/checkpoints/best.pt --image leaf.jpg --class Tomato___Late_blight --severity 0.6
    python inference.py --checkpoint ... --image leaf.jpg --mask leaf_mask.png --class X --num_samples 4 --out_dir results/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from common.errors import UserError, run_cli
from common.gpu import amp_dtype, get_device
from common.imageio_utils import list_images, load_mask, load_rgb, save_rgb, tile_images
from common.logging_utils import setup_logging
from dataio.masks import build_segmenter
from dataio.transforms import CropParams, apply_geometry, from_tensor_m11, to_tensor_m11
from generation.sampling import ddim_sample
from training.checkpoint import load_model_from_checkpoint


def translate(model, ck, device, img: np.ndarray, mask: np.ndarray, class_name: str, severity: Optional[float], seeds,
              steps: int = 50, guidance: float = 2.0, strength: float = 1.0, feather: float = 1.5, precision: str = "fp32"):
    res = int(ck["config"]["data"]["resolution"])
    c2i = ck["class_to_idx"]
    if class_name not in c2i:
        raise UserError(f"Unknown class '{class_name}'. Available: {sorted(c2i)}")
    src, m = apply_geometry(img, mask, CropParams(), res)
    n = len(seeds)
    st = torch.stack([to_tensor_m11(src)] * n).to(device)
    mt = torch.stack([torch.from_numpy(m.astype(np.float32))[None]] * n).to(device)
    ok = severity is not None and bool(ck.get("severity_supported", True))
    sev = torch.full((n,), float(severity) if ok else 0.0, device=device)
    known = torch.full((n,), 1.0 if ok else 0.0, device=device)
    out = ddim_sample(model, st, mt, torch.full((n,), c2i[class_name], device=device), sev, known, seeds, steps, guidance,
                      strength, 0.0, True, feather, amp_dtype(precision, device))
    return src, m, from_tensor_m11(out)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True, help="image file or folder")
    ap.add_argument("--mask", default=None, help="binary mask PNG (default: automatic HSV segmentation)")
    ap.add_argument("--class", dest="class_name", required=True)
    ap.add_argument("--severity", type=float, default=None)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=2.0)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--out_dir", default="./inference_out")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    setup_logging()
    dev = get_device(a.device)
    model, ck = load_model_from_checkpoint(a.checkpoint, dev)
    paths = list_images(a.image) if Path(a.image).is_dir() else [Path(a.image)]
    seg = build_segmenter("hsv")
    for p in paths:
        img = load_rgb(p)
        mask = load_mask(a.mask) if a.mask else seg.segment(img)
        src, m, outs = translate(model, ck, dev, img, mask, a.class_name, a.severity, [a.seed + i for i in range(a.num_samples)],
                                 a.steps, a.guidance_scale, a.strength)
        row = [src, (m * 255).astype(np.uint8)] + list(outs)
        save_rgb(tile_images([row]), Path(a.out_dir) / f"{p.stem}_{a.class_name}.png")
        print("wrote", Path(a.out_dir) / f"{p.stem}_{a.class_name}.png")


if __name__ == "__main__":
    run_cli(main)
