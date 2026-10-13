"""Create a tiny *procedural* leaf dataset for smoke tests. NOT real data; results on it mean nothing.

    python scripts/make_dummy_dataset.py --out_dir ./tmp/raw --n_per_class 12
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

DEFAULT_CLASSES = ["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight",
                   "Apple___healthy", "Apple___Apple_scab"]


def make_leaf(rng: np.random.Generator, size: int, kind: str, severity: float) -> np.ndarray:
    bg = int(rng.integers(100, 160))
    img = np.full((size, size, 3), bg, np.uint8) + rng.integers(0, 6, (size, size, 3), dtype=np.uint8)
    cx, cy = size / 2 + rng.uniform(-size * .05, size * .05), size / 2 + rng.uniform(-size * .05, size * .05)
    ax, ay = size * rng.uniform(.28, .4), size * rng.uniform(.18, .28)
    ang = rng.uniform(0, 180)
    green = np.array([rng.integers(40, 70), rng.integers(120, 170), rng.integers(30, 60)], np.uint8)
    leaf = np.zeros((size, size), np.uint8)
    cv2.ellipse(leaf, (int(cx), int(cy)), (int(ax), int(ay)), ang, 0, 360, 1, -1)
    tex = cv2.GaussianBlur(rng.normal(0, 14, (size, size)).astype(np.float32), (0, 0), 2)
    leafc = np.clip(green[None, None].astype(np.float32) + tex[..., None], 0, 255)
    vein = np.zeros((size, size), np.uint8)
    dx, dy = math.cos(math.radians(ang)) * ax, math.sin(math.radians(ang)) * ax
    cv2.line(vein, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), 1, max(1, size // 64))
    leafc[vein > 0] *= 1.25
    img = np.where(leaf[..., None] > 0, leafc, img).astype(np.uint8)
    if kind != "healthy":
        n = int(2 + severity * 18)
        ys, xs = np.nonzero(leaf)
        for _ in range(n):
            j = int(rng.integers(0, len(ys)))
            r = max(2, int(size * (0.02 + 0.03 * severity) * rng.uniform(.6, 1.6)))
            col = {"Early_blight": (110, 70, 30), "Late_blight": (50, 40, 30), "Apple_scab": (70, 80, 40)}.get(kind, (100, 60, 30))
            spot = np.zeros((size, size), np.uint8)
            cv2.circle(spot, (int(xs[j]), int(ys[j])), r, 1, -1)
            spot = spot * leaf
            img[spot > 0] = np.clip(np.array(col) + rng.integers(-10, 10, 3), 0, 255)
    return img


def make_dummy_dataset(out_dir: str, n_per_class: int = 12, size: int = 96, seed: int = 0,
                       classes: Optional[Sequence[str]] = None, severity_csv: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    rows = []
    for c in classes or DEFAULT_CLASSES:
        kind = c.split("___")[-1]
        (out / c).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            sev = 0.0 if "healthy" in c.lower() else float(rng.uniform(0.1, 1.0))
            img = make_leaf(rng, size, "healthy" if "healthy" in c.lower() else kind, sev)
            fn = f"{c}_{i:03d}.png"
            cv2.imwrite(str(out / c / fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            rows.append((f"{c}/{fn}", sev))
    if severity_csv:
        with open(out.parent / "severity_labels.csv", "w") as f:
            f.write("image,severity\n" + "\n".join(f"{p},{s:.3f}" for p, s in rows) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_class", type=int, default=12)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--severity_csv", action="store_true")
    a = ap.parse_args()
    print("created", make_dummy_dataset(a.out_dir, a.n_per_class, a.size, a.seed, severity_csv=a.severity_csv))
