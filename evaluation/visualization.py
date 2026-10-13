"""Plots and image grids.

    python -m evaluation.visualization grid --synthetic_dir outputs/filtered --out outputs/visualizations/grid.png
    python -m evaluation.visualization severity --checkpoint outputs/checkpoints/best.pt --class Tomato___Late_blight \
           --severities 0.2 0.4 0.6 0.8 1.0
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.errors import UserError, run_cli
from common.imageio_utils import load_mask, load_rgb, save_rgb, tile_images
from common.logging_utils import setup_logging

log = logging.getLogger("visualization")


def bar_by_class(df: pd.DataFrame, col: str, path: Path, title: str, err_col: Optional[str] = None,
                 unreliable_col: Optional[str] = None) -> None:
    d = df.dropna(subset=[col])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(d)), 4.5))
    colors = ["tab:gray" if (unreliable_col and not r[unreliable_col]) else "tab:blue" for _, r in d.iterrows()]
    ax.bar(range(len(d)), d[col], yerr=d[err_col] if err_col else None, color=colors, capsize=2)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["class"], rotation=75, ha="right", fontsize=8)
    ax.set_title(title + ("\n(grey = n < feature dim: unreliable)" if unreliable_col else ""), fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def hist_similarity(sims: Dict[str, np.ndarray], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in sims.items():
        v = np.asarray(v)[np.isfinite(v)]
        if len(v):
            ax.hist(v, bins=40, alpha=0.55, label=f"{k} (n={len(v)})", density=True)
    ax.set_xlabel("cosine similarity"); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=150); plt.close(fig)


def scatter_real_vs_synth(f_real: np.ndarray, f_syn: np.ndarray, labels_real: Sequence[str], labels_syn: Sequence[str],
                          path: Path, title: str) -> None:
    from sklearn.decomposition import PCA
    X = np.concatenate([f_real, f_syn])
    Z = PCA(n_components=2, random_state=0).fit_transform(X)
    zr, zs = Z[:len(f_real)], Z[len(f_real):]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(zr[:, 0], zr[:, 1], s=10, c="tab:blue", alpha=0.5, label=f"real (n={len(zr)})")
    ax.scatter(zs[:, 0], zs[:, 1], s=10, c="tab:red", alpha=0.5, marker="x", label=f"synthetic (n={len(zs)})")
    ax.set_title(title + " (PCA of features)"); ax.legend()
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=150); plt.close(fig)


def confusion_plot(cm: np.ndarray, classes: Sequence[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(classes) + 3),) * 2)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=75, ha="right", fontsize=7); ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=9)
    if len(classes) <= 15:
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6, color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=150); plt.close(fig)


def source_mask_synthetic_grid(root: Path, meta: pd.DataFrame, n: int, out: Path, seed: int = 0, size: int = 160) -> None:
    """Rows: real source (crop) | mask | synthetic."""
    meta = meta.sample(min(n, len(meta)), random_state=seed).sort_values("class")
    rows = [[load_rgb(root / r["source_crop"]), (load_mask(root / r["mask"]) * 255).astype(np.uint8), load_rgb(root / r["path"])]
            for r in meta.to_dict("records")]
    save_rgb(tile_images(rows, size=size), out)
    log.info("wrote %s  (columns: source | mask | synthetic; classes: %s)", out, ", ".join(meta["class"].tolist()))


def severity_grid(checkpoint: str, class_name: str, severities: Sequence[float], out: Path, image: Optional[str] = None,
                  processed_dir: Optional[str] = None, n_sources: int = 3, seed: int = 0, device: str = "auto",
                  steps: int = 50, guidance: float = 2.0) -> None:
    """Rows: real source | mask | synthetic at each severity (same noise seed => only severity changes)."""
    from common.gpu import get_device
    from dataio.masks import build_segmenter
    from dataio.metadata import load_metadata
    from inference import translate
    from training.checkpoint import load_model_from_checkpoint
    dev = get_device(device)
    model, ck = load_model_from_checkpoint(checkpoint, dev)
    if not ck.get("severity_supported", True):
        log.warning("This checkpoint was trained without severity conditioning: the columns will look alike.")
    if image:
        paths = [Path(image)]
    else:
        pdir = Path(processed_dir or ck["config"]["paths"]["processed_dir"])
        meta = load_metadata(pdir)
        tr = meta[(meta["split"] == "train") & meta["is_healthy"]]
        sp = class_name.split(ck["config"]["data"].get("species_sep", "___"))[0] if ck["config"]["data"].get("species_sep", "___") in class_name else None
        if sp is not None:
            tr = tr[tr["species"] == sp]
        if tr.empty:
            raise UserError("No healthy training images found to use as sources; pass --image.")
        sel = tr.sample(min(n_sources, len(tr)), random_state=seed)
        paths = [pdir / p for p in sel["path"]]
    seg, rows = build_segmenter("hsv"), []
    for p in paths:
        img = load_rgb(p)
        mp = (p.parents[3] / "masks" / p.parents[1].name / p.parent.name / (p.stem + ".png")) if not image else None
        mask = load_mask(mp) if (mp is not None and mp.exists()) else seg.segment(img)
        row, src_ref = [], None
        for s in severities:
            src, m, o = translate(model, ck, dev, img, mask, class_name, s, [seed], steps, guidance)
            if src_ref is None:
                row = [src, (m * 255).astype(np.uint8)]
                src_ref = src
            row.append(o[0])
        rows.append(row)
    save_rgb(tile_images(rows), out)
    log.info("wrote %s | columns: source, mask, severity=%s | class=%s", out, list(severities), class_name)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--synthetic_dir", required=True); g.add_argument("--n", type=int, default=12)
    g.add_argument("--out", default="outputs/visualizations/source_mask_synthetic.png"); g.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("severity")
    s.add_argument("--checkpoint", required=True); s.add_argument("--class", dest="class_name", required=True)
    s.add_argument("--severities", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    s.add_argument("--image", default=None); s.add_argument("--processed_dir", default=None)
    s.add_argument("--n_sources", type=int, default=3); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--steps", type=int, default=50); s.add_argument("--guidance_scale", type=float, default=2.0)
    s.add_argument("--out", default="outputs/visualizations/severity_sweep.png"); s.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    setup_logging()
    if a.cmd == "grid":
        root = Path(a.synthetic_dir)
        meta_p = root / ("filtered_metadata.csv" if (root / "filtered_metadata.csv").exists() else "generation_metadata.csv")
        source_mask_synthetic_grid(root, pd.read_csv(meta_p), a.n, Path(a.out), a.seed)
    else:
        severity_grid(a.checkpoint, a.class_name, a.severities, Path(a.out), a.image, a.processed_dir, a.n_sources, a.seed,
                      a.device, a.steps, a.guidance_scale)


if __name__ == "__main__":
    run_cli(main)
