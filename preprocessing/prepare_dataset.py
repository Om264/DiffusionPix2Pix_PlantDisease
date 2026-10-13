"""Validate -> split -> leakage check -> write ``processed_dir`` (+ metadata.csv, classes.json, report).

    python -m preprocessing.prepare_dataset --data_dir ./data/raw --output_dir ./data/processed

Produces  processed_dir/{train,val,test}/<class>/<image>   (REAL images only)
          processed_dir/metadata.csv, classes.json, dataset_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common.errors import UserError, run_cli
from common.hashing import file_md5, phash
from common.imageio_utils import load_rgb, save_rgb
from common.logging_utils import setup_logging
from dataio.metadata import (check_min_images, discover_images, is_healthy_class, make_class_mapping, save_class_mapping,
                             save_metadata, species_of)
from dataio.splits import assert_no_overlap, assign_splits, find_leaks

log = logging.getLogger("prepare_dataset")


def index_images(df: pd.DataFrame, min_side: int) -> pd.DataFrame:
    """Open every image once: drop corrupt/too-small files, record size, md5 and dHash."""
    keep, recs = [], []
    for i, p in enumerate(tqdm(df["src_path"], desc="validating images")):
        try:
            img = load_rgb(p)
        except Exception as e:  # corrupt / unreadable
            log.warning("Skipping unreadable image %s (%s)", p, e)
            continue
        h, w = img.shape[:2]
        if min(h, w) < min_side:
            log.warning("Skipping %s: %dx%d is smaller than min_side=%d", p, w, h, min_side)
            continue
        keep.append(i)
        recs.append({"width": w, "height": h, "md5": file_md5(p), "phash": phash(img)})
    out = df.iloc[keep].reset_index(drop=True)
    return pd.concat([out, pd.DataFrame(recs)], axis=1)


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def load_real_severity(csv_path: str, scale: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not {"image", "severity"} <= set(df.columns):
        raise UserError("severity_csv needs the columns: image,severity  (image = path relative to data_dir "
                        "or just the file name; severity in [0,1], or use --severity_scale 100 for percent)")
    df["severity"] = df["severity"].astype(float) / scale
    if ((df["severity"] < 0) | (df["severity"] > 1)).any():
        raise UserError("Severity values must lie in [0,1] after dividing by --severity_scale.")
    return df


def prepare_dataset(data_dir: str, output_dir: str, val_frac: float = 0.1, test_frac: float = 0.2, seed: int = 42,
                    min_images_per_class: int = 5, min_side: int = 32, max_side: int = 512, mode: str = "copy",
                    severity_csv: Optional[str] = None, severity_scale: float = 1.0, healthy_pattern: str = "healthy",
                    species_sep: str = "___", max_per_class: Optional[int] = None, leak_hamming: int = 2) -> pd.DataFrame:
    out = Path(output_dir)
    df, fmt = discover_images(data_dir)
    if max_per_class:
        df = df.groupby("class", group_keys=False).apply(lambda g: g.sample(min(len(g), max_per_class), random_state=seed))
        df = df.reset_index(drop=True)
    cls_counts = df.groupby("class").size()
    tiny = cls_counts[cls_counts < max(3, min_images_per_class)]
    if len(tiny):
        raise UserError("\n".join(f"Only {n} images were found for class '{c}'." for c, n in tiny.items())
                        + f"\nEach class needs at least {max(3, min_images_per_class)} images to build train/val/test splits.")
    df = index_images(df, min_side)
    df = assign_splits(df, fmt, val_frac, test_frac, seed)
    drop = find_leaks(df, leak_hamming)
    df = df.drop(index=drop).reset_index(drop=True)
    check_min_images(df, min_images_per_class)

    c2i = make_class_mapping(df["class"])
    sev_map = {}
    if severity_csv:
        sdf = load_real_severity(severity_csv, severity_scale)
        sev_map = dict(zip(sdf["image"].astype(str).str.replace("\\", "/"), sdf["severity"]))
        base_map = {Path(k).name: v for k, v in sev_map.items()}
    else:
        log.warning("No disease severity labels were found. Synthetic severity values will be estimated "
                    "(run preprocessing.estimate_severity); they are NOT ground truth.")

    rows, used = [], set()
    root = Path(data_dir)
    for r in tqdm(df.to_dict("records"), desc="writing processed dataset"):
        cls = r["class"]
        src = Path(r["src_path"])
        base_id = _sanitize(f"{cls}__{src.stem}")
        image_id = base_id if base_id not in used else f"{base_id}_{r['md5'][:6]}"
        used.add(image_id)
        rel = Path(r["split"]) / cls / f"{image_id}{src.suffix.lower()}"
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if max(r["width"], r["height"]) > max_side:
            s = max_side / max(r["width"], r["height"])
            im = Image.open(src).convert("RGB").resize((round(r["width"] * s), round(r["height"] * s)), Image.LANCZOS)
            save_rgb(np.asarray(im), dst)
            w, h = im.size
        else:
            (shutil.copy2 if mode == "copy" else lambda a, b: (b.exists() or b.symlink_to(a.resolve())))(src, dst)
            w, h = r["width"], r["height"]
        try:
            key = str(src.relative_to(root)).replace("\\", "/")
        except ValueError:
            key = src.name
        if r["split"] != "" and fmt == "B":  # also allow keys without the split folder
            key2 = "/".join(Path(key).parts[1:])
        else:
            key2 = key
        sev = sev_map.get(key, sev_map.get(key2, base_map.get(src.name) if sev_map else None))
        rows.append({
            "image_id": image_id, "path": str(rel).replace("\\", "/"), "class": cls, "class_id": c2i[cls],
            "species": species_of(cls, species_sep), "is_healthy": is_healthy_class(cls, healthy_pattern),
            "split": r["split"], "mask_path": None, "mask_source": "none", "mask_coverage": np.nan,
            "severity": sev if sev is not None else np.nan, "severity_source": "real" if sev is not None else "none",
            "severity_raw": np.nan, "width": w, "height": h, "md5": r["md5"], "phash": r["phash"]})
    meta = pd.DataFrame(rows)
    assert_no_overlap(meta)
    out.mkdir(parents=True, exist_ok=True)
    save_metadata(meta, out)
    save_class_mapping(c2i, out)
    report = {
        "format": fmt, "n_images": int(len(meta)), "n_classes": len(c2i), "dropped_for_leakage": len(drop),
        "per_split": meta["split"].value_counts().to_dict(),
        "per_class_split": meta.groupby(["class", "split"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "real_severity_labels": int((meta["severity_source"] == "real").sum()), "seed": seed,
        "note": "val/test contain REAL images only; synthetic images are never written here."}
    with open(out / "dataset_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info("Prepared %d images / %d classes -> %s | splits: %s", len(meta), len(c2i), out, report["per_split"])
    return meta


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", default="./data/processed")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_images_per_class", type=int, default=5)
    ap.add_argument("--max_side", type=int, default=512, help="downscale larger images (speeds up training)")
    ap.add_argument("--mode", choices=["copy", "link"], default="copy")
    ap.add_argument("--severity_csv", default=None, help="optional REAL severity labels: columns image,severity")
    ap.add_argument("--severity_scale", type=float, default=1.0, help="divide labels by this (100 for percent)")
    ap.add_argument("--healthy_pattern", default="healthy")
    ap.add_argument("--species_sep", default="___")
    ap.add_argument("--max_per_class", type=int, default=None, help="cap images per class (quick tests)")
    ap.add_argument("--leak_hamming", type=int, default=2)
    a = ap.parse_args(argv)
    setup_logging(Path(a.output_dir) / "logs" / "prepare_dataset.log")
    prepare_dataset(**{k: v for k, v in vars(a).items()})


if __name__ == "__main__":
    run_cli(main)
