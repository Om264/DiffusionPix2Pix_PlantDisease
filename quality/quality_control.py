"""Quality control + deduplication of generated images.

    python -m quality.quality_control --input_dir outputs/generated --output_dir outputs/filtered

Writes  output_dir/<class>/*.png  (accepted images, self-contained with _aux/), filtered_metadata.csv,
        quality_report.csv (every generated image, all metrics + rejection reasons), quality_summary.json,
        quality_control.log
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.config import load_config
from common.errors import UserError, run_cli
from common.imageio_utils import load_rgb
from common.logging_utils import log_to_file, setup_logging
from quality.deduplicate import find_duplicates
from quality.embeddings import build_embedder
from quality.filtering import check_one, quality_score

log = logging.getLogger("quality_control")


def _place(src: Path, dst: Path, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _run_quality_control(input_dir: str, output_dir: str, q: Dict[str, Any], healthy_pattern: str = "healthy",
                        dedup: bool = True, link: bool = False, device: Optional[str] = None,
                        embeddings_cache: bool = True) -> pd.DataFrame:
    root, out = Path(input_dir), Path(output_dir)
    meta_p = root / "generation_metadata.csv"
    if not meta_p.exists():
        raise UserError(f"{meta_p} not found. Run `python -m generation.generate` first.")
    gen = pd.read_csv(meta_p)
    log.info("Quality control on %d generated images", len(gen))
    rows = [check_one(r, root, q, healthy_pattern) for r in tqdm(gen.to_dict("records"), desc="per-image checks")]
    rep = gen[["image_id", "class", "severity", "seed", "source_image"]].merge(pd.DataFrame(rows), on="image_id")
    rep["reasons"] = rep["reasons"].fillna("")
    valid = rep["reasons"] != "invalid_image"

    # --- embedding similarity to the SOURCE plant ------------------------------------------------------------
    emb = None
    try:
        embedder = build_embedder(q["embedder"], q.get("embedder_name"), device)
        vi = rep.index[valid & (rep["reasons"] != "shape_mismatch")]
        e_syn = embedder.embed([load_rgb(root / gen.at[i, "path"]) for i in tqdm(vi, desc=f"embed synthetic ({embedder.name})")])
        e_src = embedder.embed([load_rgb(root / gen.at[i, "source_crop"]) for i in tqdm(vi, desc="embed sources")])
        emb = np.zeros((len(rep), e_syn.shape[1]), np.float32)
        emb[vi] = e_syn
        rep.loc[vi, "source_sim"] = (e_syn * e_src).sum(1)
        low = rep["source_sim"] < q["source_sim_min"]
        rep.loc[low, "reasons"] = rep.loc[low, "reasons"].map(lambda s: (s + ";" if s else "") + "unrelated_to_source")
        if embeddings_cache:
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "synthetic_embeddings.npy", emb)
            rep[["image_id"]].to_csv(out / "synthetic_embeddings_ids.csv", index=False)
    except UserError as e:
        log.warning("Embedding-based checks skipped: %s", e)
        rep["source_sim"] = np.nan
    rep["passed"] = rep["reasons"] == ""
    rep["quality_score"] = quality_score(rep)

    # --- deduplication among images that passed ---------------------------------------------------------------
    rep["is_duplicate"], rep["duplicate_of"] = False, ""
    n_dup = 0
    if dedup:
        ok = rep[rep["passed"]]
        if len(ok):
            d = find_duplicates(gen.loc[ok.index], root, rep.loc[ok.index, "quality_score"],
                                emb[ok.index.values] if emb is not None else None,
                                int(q["dedup"]["hamming"]), q["dedup"]["embed_sim"] if emb is not None else None)
            d.index = ok.index
            rep.loc[ok.index, "is_duplicate"] = d["is_duplicate"].values
            rep.loc[ok.index, "duplicate_of"] = d["duplicate_of"].values
            n_dup = int(d["is_duplicate"].sum())
            dr = rep.index[rep["is_duplicate"]]
            rep.loc[dr, "reasons"] = rep.loc[dr, "reasons"].map(lambda s: (s + ";" if s else "") + "duplicate")
    rep["accepted"] = rep["passed"] & ~rep["is_duplicate"]

    # --- outputs --------------------------------------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out / "quality_report.csv", index=False)
    acc = gen[gen["image_id"].isin(rep.loc[rep["accepted"], "image_id"])].merge(
        rep[["image_id", "quality_score", "source_sim"]], on="image_id")
    for r in tqdm(acc.to_dict("records"), desc="copying accepted"):
        for key in ("path", "mask", "source_crop"):
            _place(root / r[key], out / r[key], link)
    acc.to_csv(out / "filtered_metadata.csv", index=False)
    reasons = pd.Series([x for s in rep["reasons"] for x in s.split(";") if x]).value_counts().to_dict()
    summary = {"generated": int(len(gen)), "rejected_quality": int((~rep["passed"]).sum()), "duplicates": n_dup,
               "accepted": int(rep["accepted"].sum()), "reasons": reasons,
               "accepted_per_class": acc["class"].value_counts().to_dict(),
               "note": "Thresholds are heuristics (configs/evaluation.yaml); inspect rejected/accepted samples."}
    with open(out / "quality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("generated=%d rejected=%d duplicates=%d accepted=%d | reasons=%s", summary["generated"], summary["rejected_quality"],
             n_dup, summary["accepted"], reasons)
    return rep


def run_quality_control(input_dir: str, output_dir: str, *args, **kwargs) -> pd.DataFrame:
    """Quality control + dedup (see ``_run_quality_control``); also writes ``output_dir/quality_control.log``."""
    with log_to_file(Path(output_dir) / "quality_control.log"):
        return _run_quality_control(input_dir, output_dir, *args, **kwargs)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default="configs/evaluation.yaml")
    ap.add_argument("--embedder", default=None, help="dinov2 | clip | handcrafted")
    ap.add_argument("--embedder_name", default=None)
    ap.add_argument("--source_sim_min", type=float, default=None)
    ap.add_argument("--no_dedup", action="store_true")
    ap.add_argument("--link", action="store_true", help="hard-link instead of copy accepted files")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(a.config, a.set)
    q = cfg["quality"]
    for k in ("embedder", "embedder_name", "source_sim_min"):
        if getattr(a, k) is not None:
            q[k] = getattr(a, k)
    run_quality_control(a.input_dir, a.output_dir, q, cfg["data"]["healthy_pattern"], not a.no_dedup, a.link)


if __name__ == "__main__":
    run_cli(main)
