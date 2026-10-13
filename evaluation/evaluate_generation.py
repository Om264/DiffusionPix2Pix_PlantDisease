"""Generative-quality evaluation (FD/FID, KID, SSIM, PSNR, LPIPS, embedding similarity) overall and per class.

    python -m evaluation.evaluate_generation --real_dir data/processed/test --synthetic_dir outputs/filtered

``--real_dir``       folder with <class>/*.jpg of REAL images (use the held-out test split, or train split)
``--synthetic_dir``  folder produced by quality_control (filtered_metadata.csv + <class>/ + _aux/)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common.config import load_config
from common.errors import UserError, run_cli
from common.imageio_utils import list_images, load_mask, load_rgb
from common.logging_utils import setup_logging
from evaluation import metrics as M
from evaluation import visualization as V
from quality.embeddings import build_embedder

log = logging.getLogger("evaluate_generation")


def _load_real(real_dir: Path, classes: Sequence[str], cap: int, rng) -> Dict[str, List[np.ndarray]]:
    out = {}
    for c in classes:
        ps = list_images(real_dir / c)
        if len(ps) > cap:
            ps = [ps[i] for i in rng.choice(len(ps), cap, replace=False)]
        out[c] = [load_rgb(p) for p in ps]
    return out


def evaluate_generation(real_dir: str, synthetic_dir: str, output_dir: str, ecfg: Dict[str, Any], device: str = "cpu",
                        use_lpips: bool = True, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    sdir, odir = Path(synthetic_dir), Path(output_dir)
    meta_p = sdir / "filtered_metadata.csv"
    if not meta_p.exists():
        raise UserError(f"{meta_p} not found. Run `python -m quality.quality_control` first.")
    syn = pd.read_csv(meta_p)
    classes = sorted(c for c in syn["class"].unique() if (Path(real_dir) / c).is_dir())
    missing = sorted(set(syn["class"].unique()) - set(classes))
    if missing:
        log.warning("No real images for class(es) %s in %s: they are skipped.", missing, real_dir)
    if not classes:
        raise UserError(f"No class folders of {real_dir} match the synthetic classes.")
    cap = int(ecfg["max_images_per_class"])
    real = _load_real(Path(real_dir), classes, cap, rng)
    fe = M.get_feature_extractor(ecfg["feature_extractor"], ecfg.get("extractor_name"), device)
    emb = build_embedder(ecfg.get("embedder", "dinov2"), ecfg.get("extractor_name"), device)  # for cosine similarities
    lp = None
    if use_lpips and ecfg.get("lpips", True):
        try:
            lp = M.LPIPSMetric(device)
        except UserError as e:
            log.warning("%s", e)

    per_class, all_fr, all_fs, all_labels_r, all_labels_s = [], [], [], [], []
    sim_src, sim_real = [], []
    for c in tqdm(classes, desc="classes"):
        s = syn[syn["class"] == c]
        if len(s) > cap:
            s = s.sample(cap, random_state=seed)
        simgs = [load_rgb(sdir / p) for p in s["path"]]
        srcs = [load_rgb(sdir / p) for p in s["source_crop"]]
        masks = [load_mask(sdir / p).astype(bool) for p in s["mask"]]
        fr, fs = fe.features(real[c]), fe.features(simgs)
        all_fr.append(fr); all_fs.append(fs)
        all_labels_r += [c] * len(fr); all_labels_s += [c] * len(fs)
        d = fr.shape[1]
        n_min = min(len(fr), len(fs))
        kid_m, kid_s = M.kernel_inception_distance(fr, fs, ecfg["kid_subsets"], ecfg["kid_subset_size"], seed)
        row = {"class": c, "n_real": len(fr), "n_syn": len(fs), "fd": M.frechet_distance(fr, fs) if n_min >= 3 else np.nan,
               "fd_reliable": bool(n_min >= d), "kid_mean": kid_m, "kid_std": kid_s}
        # --- structure preservation vs the source (SSIM/PSNR/LPIPS) ------------------------------------------
        bg_ps, bg_ss, in_ps, in_ss, lps = [], [], [], [], []
        for a, b, m in zip(simgs, srcs, masks):
            bg = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0   # background with a safety margin
            bg_ps.append(M.masked_psnr(a, b, bg)); bg_ss.append(M.masked_ssim(a, b, bg))
            in_ps.append(M.masked_psnr(a, b, m)); in_ss.append(M.masked_ssim(a, b, m))
            if lp is not None:
                lps.append(lp.distance(a, b))
        fin = lambda v: float(np.nanmean(np.where(np.isfinite(v), v, np.nan))) if len(v) and np.isfinite(v).any() else float("inf") if len(v) else np.nan
        row.update(bg_psnr=fin(bg_ps), bg_ssim=float(np.nanmean(bg_ss)), leaf_psnr=fin(in_ps), leaf_ssim=float(np.nanmean(in_ss)),
                   lpips_to_source=float(np.mean(lps)) if lps else np.nan)
        if lp is not None and len(simgs) >= 2:
            pairs = rng.choice(len(simgs), (min(100, len(simgs)), 2))
            row["lpips_diversity"] = float(np.mean([lp.distance(simgs[i], simgs[j]) for i, j in pairs if i != j] or [np.nan]))
        # --- embedding similarity (DINOv2 / handcrafted) --------------------------------------------------------
        es, esrc, er = emb.embed(simgs), emb.embed(srcs), emb.embed(real[c])
        ss = (es * esrc).sum(1)
        cs = es @ er.mean(0) / max(np.linalg.norm(er.mean(0)), 1e-8)
        nn_ = (es @ er.T).max(1)
        row.update(sim_to_source=float(ss.mean()), sim_to_real_centroid=float(cs.mean()), sim_to_nearest_real=float(nn_.mean()))
        sim_src += list(ss); sim_real += list(nn_)
        per_class.append(row)

    fr_all, fs_all = np.concatenate(all_fr), np.concatenate(all_fs)
    kid_m, kid_s = M.kernel_inception_distance(fr_all, fs_all, ecfg["kid_subsets"], ecfg["kid_subset_size"], seed)
    overall = {"feature_extractor": fe.label, "n_real": int(len(fr_all)), "n_syn": int(len(fs_all)), "classes": len(classes),
               "fd": M.frechet_distance(fr_all, fs_all), "fd_reliable": bool(min(len(fr_all), len(fs_all)) >= fr_all.shape[1]),
               "kid_mean": kid_m, "kid_std": kid_s, "sim_to_source": M.cosine_stats(np.array(sim_src)),
               "sim_to_nearest_real": M.cosine_stats(np.array(sim_real)), "lpips_available": lp is not None,
               "real_reference": str(real_dir),
               "caveats": [
                   "Synthetic images are generated from TRAIN-split sources; the real reference here is a different image set, so "
                   "FD/KID also reflect content differences (different leaves/backgrounds), not only generation quality.",
                   "FD needs n >= feature dimension for a reliable covariance estimate; per-class FD is flagged fd_reliable=False "
                   "otherwise. Prefer KID for small samples.",
                   "SSIM/PSNR/LPIPS are computed against the SOURCE image: background values measure background preservation "
                   "(should be high); leaf values are EXPECTED to be lower because disease is synthesised there. They are not fidelity scores."]}
    odir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_class)
    df.to_csv(odir / "generation_metrics_per_class.csv", index=False)
    with open(odir / "generation_metrics_overall.json", "w") as f:
        json.dump(overall, f, indent=2, default=float)
    vis = odir.parent / "visualizations" if odir.name == "metrics" else odir
    V.bar_by_class(df, "fd", vis / "fd_by_class.png", f"{fe.label} by class", unreliable_col="fd_reliable")
    V.bar_by_class(df, "kid_mean", vis / "kid_by_class.png", "KID by class", err_col="kid_std")
    V.bar_by_class(df, "sim_to_source", vis / "embedding_similarity_by_class.png", "cosine(synthetic, its source) by class")
    V.hist_similarity({"synthetic vs its source": np.array(sim_src), "synthetic vs nearest real": np.array(sim_real)},
                      vis / "embedding_similarity_hist.png", "Embedding similarity")
    sub = min(len(fr_all), 1500)
    ir, js = rng.choice(len(fr_all), sub, replace=False), rng.choice(len(fs_all), min(len(fs_all), 1500), replace=False)
    V.scatter_real_vs_synth(fr_all[ir], fs_all[js], [], [], vis / "real_vs_synthetic_distribution.png", "Real vs synthetic")
    log.info("overall: %s=%.3f (reliable=%s) KID=%.5f±%.5f | plots in %s", fe.label, overall["fd"], overall["fd_reliable"], kid_m, kid_s, vis)
    return overall


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--synthetic_dir", required=True)
    ap.add_argument("--output_dir", default="./outputs/metrics")
    ap.add_argument("--config", default="configs/evaluation.yaml")
    ap.add_argument("--feature_extractor", choices=["inception", "dinov2", "handcrafted"], default=None)
    ap.add_argument("--extractor_name", default=None)
    ap.add_argument("--embedder", choices=["dinov2", "clip", "handcrafted"], default=None, help="for cosine similarities")
    ap.add_argument("--max_images_per_class", type=int, default=None)
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    setup_logging(Path(a.output_dir) / "evaluate_generation.log")
    ecfg = load_config(a.config, a.set)["evaluation"]
    for k in ("feature_extractor", "extractor_name", "max_images_per_class", "embedder"):
        if getattr(a, k) is not None:
            ecfg[k] = getattr(a, k)
    evaluate_generation(a.real_dir, a.synthetic_dir, a.output_dir, ecfg, a.device, not a.no_lpips)


if __name__ == "__main__":
    run_cli(main)
