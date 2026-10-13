"""One command for the whole pipeline.

    python run_pipeline.py --data_dir ./data/raw --output_dir ./outputs --total_images 10000
    python run_pipeline.py --data_dir ./data/raw --output_dir ./outputs_test --total_images 100     # start small!
    python run_pipeline.py --data_dir ./data/raw --output_dir ./outputs --total_images 10000 --skip_train --checkpoint outputs/checkpoints/best.pt

Stages: 1 dataset validation  2 split + metadata  3 masks  4 severity  5 training (+validation, best checkpoint)
        6 generation  7 quality filtering  8 deduplication  9 class balancing  10 dataset construction
        11 generation metrics  12 final report   (optional: --run_downstream)
Everything is resumable: finished stages are skipped, training resumes from latest.pt, generation resumes from
generation_metadata.csv.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from common.config import load_config, save_config
from common.errors import UserError, run_cli
from common.logging_utils import setup_logging

log = logging.getLogger("pipeline")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", default="./outputs")
    ap.add_argument("--processed_dir", default=None, help="default: <output_dir>/processed")
    ap.add_argument("--total_images", type=int, default=10000, help="ACCEPTED synthetic images wanted after filtering")
    ap.add_argument("--images_per_class", type=int, default=None)
    ap.add_argument("--balance_mode", choices=["uniform", "inverse_frequency", "proportional"], default="uniform")
    ap.add_argument("--preset", choices=["default", "smoke"], default="default", help="smoke = tiny CPU model for testing")
    ap.add_argument("--train_config", default=None)
    ap.add_argument("--generation_config", default="configs/generation.yaml")
    ap.add_argument("--evaluation_config", default="configs/evaluation.yaml")
    ap.add_argument("--mask_method", default="hsv", choices=["hsv", "grabcut", "existing", "torchscript", "full"])
    ap.add_argument("--masks_dir", default=None)
    ap.add_argument("--model_path", default=None, help="TorchScript segmentation model (mask_method=torchscript)")
    ap.add_argument("--severity_csv", default=None, help="optional REAL severity labels (image,severity)")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--checkpoint", default=None, help="use this checkpoint instead of training")
    ap.add_argument("--embedder", default=None, help="dinov2 | clip | handcrafted (quality control + evaluation similarity)")
    ap.add_argument("--feature_extractor", default=None, help="inception | dinov2 | handcrafted (FD/KID)")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--run_downstream", action="store_true", help="train classifiers real-only vs real+synthetic (slow)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides applied to all configs, e.g. data.resolution=128")
    a = ap.parse_args(argv)

    out = Path(a.output_dir)
    proc = Path(a.processed_dir) if a.processed_dir else out / "processed"
    setup_logging(out / "logs" / "pipeline.log")
    t0 = time.time()
    train_cfg_path = a.train_config or ("configs/smoke.yaml" if a.preset == "smoke" else "configs/training.yaml")
    ov = [f"paths.data_dir={a.data_dir}", f"paths.processed_dir={proc}", f"paths.output_dir={out}", f"seed={a.seed}"] + list(a.set)
    tcfg = load_config(train_cfg_path, ov)
    gcfg_all = load_config(a.generation_config, ov + (["generation.precision=fp32", "generation.ddim_steps=8", "generation.batch_size=8"]
                                                      if a.preset == "smoke" else []))
    ecfg_all = load_config(a.evaluation_config, ov)
    save_config({"args": vars(a), "train": tcfg, "generation": gcfg_all, "evaluation": ecfg_all}, out / "config_used_pipeline.yaml")
    d = tcfg["data"]
    report: Dict[str, Any] = {"args": vars(a), "stages": {}}

    # ------------------------------------------------------------------ 1-2: validation, split, metadata
    from preprocessing.prepare_dataset import prepare_dataset
    if not (proc / "metadata.csv").exists():
        log.info("=== Stage 1-2: dataset validation, split, metadata ===")
        prepare_dataset(a.data_dir, str(proc), d["val_frac"], d["test_frac"], a.seed, d["min_images_per_class"], 32, d["max_side"],
                        "copy", a.severity_csv, 1.0, d["healthy_pattern"], d["species_sep"])
    else:
        log.info("Stage 1-2 skipped: %s already exists (delete it to rebuild).", proc / "metadata.csv")
    # ------------------------------------------------------------------ 3-4: masks, severity
    from dataio.metadata import load_metadata
    meta = load_metadata(proc)
    if meta["mask_path"].isna().any():
        log.info("=== Stage 3: masks (%s) ===", a.mask_method)
        from preprocessing.prepare_masks import prepare_masks
        prepare_masks(str(proc), a.mask_method, a.masks_dir, a.model_path)
    meta = load_metadata(proc)
    if d.get("severity_mode", "estimated") != "none" and (meta["severity_source"] == "none").any():
        log.info("=== Stage 4: severity estimation (estimated, not ground truth) ===")
        from preprocessing.estimate_severity import estimate_severity
        estimate_severity(str(proc))
    meta = load_metadata(proc)
    report["stages"]["dataset"] = {"per_split": meta["split"].value_counts().to_dict(), "classes": int(meta["class"].nunique()),
                                   "severity_sources": meta["severity_source"].value_counts().to_dict(),
                                   "mask_sources": meta["mask_source"].value_counts().to_dict()}
    # ------------------------------------------------------------------ 5: training
    ckpt = a.checkpoint or str(out / "checkpoints" / "best.pt")
    if not a.skip_train and not a.checkpoint:
        log.info("=== Stage 5-7: training + validation + best checkpoint ===")
        from training.train import main as train_main
        train_main(["--config", train_cfg_path, "--resume", "--set", *ov])
    if not Path(ckpt).exists():
        raise UserError(f"Checkpoint not found: {ckpt}. Train first or pass --checkpoint.")
    # ------------------------------------------------------------------ 8-11: generation + QC + dedup + balancing
    from generation.batch_generate import compute_class_quotas, read_existing_metadata
    from generation.generate import run_generation
    from quality.filtering import select_per_class
    from quality.quality_control import run_quality_control
    g = gcfg_all["generation"]
    g["balance_mode"] = a.balance_mode
    q = ecfg_all["quality"]
    if a.embedder:
        q["embedder"] = a.embedder
    classes = sorted(meta["class"].unique())
    real_counts = meta[meta["split"] == "train"].groupby("class").size().reindex(classes).fillna(0).astype(int).to_dict()
    total = a.images_per_class * len(classes) if a.images_per_class else a.total_images
    target = compute_class_quotas(real_counts, total, g["balance_mode"], g.get("inverse_frequency_power", 1.0), int(g.get("min_per_class", 0)))
    log.info("Target ACCEPTED images: %d over %d classes (%s): %s", total, len(classes), g["balance_mode"], target)
    gen_dir, filt_dir = out / "generated", out / "filtered"
    over = float(gcfg_all["pipeline"]["oversample"])
    accepted = pd.Series(0, index=classes)
    rounds_log = []
    for r in range(int(gcfg_all["pipeline"]["max_rounds"])):
        deficit = {c: max(target[c] - int(accepted[c]), 0) for c in classes}
        if sum(deficit.values()) == 0:
            break
        quotas = {c: int(-(-deficit[c] * (over if r == 0 else max(over, 1.5)) // 1)) for c in classes}
        log.info("=== Generation round %d: need %s more accepted; generating %d ===", r + 1, sum(deficit.values()), sum(quotas.values()))
        n_prev = len(read_existing_metadata(gen_dir))
        run_generation(ckpt, str(gen_dir), g, str(proc), tcfg["device"], a.seed + r, id_offset=n_prev, quotas=quotas, resume=True,
                       run_tag=f"round{r + 1}")
        rep = run_quality_control(str(gen_dir), str(filt_dir), q, d["healthy_pattern"], dedup=True, device=None)
        acc = rep[rep["accepted"]]
        accepted = acc.groupby("class").size().reindex(classes).fillna(0).astype(int)
        rounds_log.append({"round": r + 1, "generated_total": int(len(rep)), "accepted_total": int(len(acc))})
    # trim to the balanced targets (keeps the best-scoring images per class) and make the folder consistent
    fm = pd.read_csv(filt_dir / "filtered_metadata.csv")
    fm.to_csv(filt_dir / "filtered_metadata_all_accepted.csv", index=False)
    final = select_per_class(fm, target)
    for rr in fm[~fm["image_id"].isin(final["image_id"])].to_dict("records"):
        for k in ("path", "mask", "source_crop"):
            (filt_dir / rr[k]).unlink(missing_ok=True)
    final.to_csv(filt_dir / "filtered_metadata.csv", index=False)
    got = final.groupby("class").size().reindex(classes).fillna(0).astype(int).to_dict()
    short = {c: target[c] - got[c] for c in classes if got[c] < target[c]}
    if short:
        log.warning("Target not reached for %d class(es) (shortfall: %s). Quality thresholds may be too strict, the model may "
                    "need more training, or raise pipeline.max_rounds.", len(short), short)
    qs = json.load(open(filt_dir / "quality_summary.json"))
    report["stages"]["generation"] = {"target_total": int(total), "target_per_class": target, "final_per_class": got, "final_total": int(len(final)),
                                      "shortfall": short, "rounds": rounds_log, "quality_summary": qs}
    # ------------------------------------------------------------------ 12: dataset construction
    log.info("=== Stage 12: building real + synthetic datasets (test stays REAL only) ===")
    from preprocessing.build_dataset import build_dataset
    build_dataset(str(proc), str(filt_dir), str(out / "combined"), a.ratios, "per_class", a.seed)
    # ------------------------------------------------------------------ 13: generation metrics
    log.info("=== Stage 13: generation metrics ===")
    from evaluation.evaluate_generation import evaluate_generation
    ev = ecfg_all["evaluation"]
    if a.embedder:
        ev["embedder"] = a.embedder
    if a.feature_extractor:
        ev["feature_extractor"] = a.feature_extractor
    try:
        real_ref = proc / "test" if any((proc / "test").glob("*/*")) else proc / "train"
        report["stages"]["generation_metrics"] = evaluate_generation(str(real_ref), str(filt_dir), str(out / "metrics"), ev, "cpu")
    except UserError as e:
        log.warning("Generation metrics skipped: %s", e)
        report["stages"]["generation_metrics"] = f"Not executed: {e}"
    if a.run_downstream:
        from evaluation.evaluate_downstream import evaluate_downstream
        dd = ecfg_all["downstream"]; dd["ratios"] = a.ratios
        evaluate_downstream(str(proc), str(filt_dir), str(out / "metrics" / "downstream"), dd, tcfg["device"])
        report["stages"]["downstream"] = "see metrics/downstream/downstream_summary.md"
    else:
        report["stages"]["downstream"] = "Not executed (pass --run_downstream)."
    # ------------------------------------------------------------------ 14: report
    report["elapsed_minutes"] = round((time.time() - t0) / 60, 2)
    with open(out / "final_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(out / "final_report.md", "w") as f:
        s = report["stages"]
        f.write("# Pipeline report (numbers below were measured by this run)\n\n")
        f.write(f"- classes: {s['dataset']['classes']}; real images per split: {s['dataset']['per_split']}\n")
        f.write(f"- severity label sources: {s['dataset']['severity_sources']} (estimated != ground truth)\n")
        f.write(f"- mask sources: {s['dataset']['mask_sources']}\n")
        gs = s["generation"]
        f.write(f"- target accepted images: {gs['target_total']}; final accepted: {gs['final_total']}; shortfall: {gs['shortfall'] or 'none'}\n")
        f.write(f"- quality control: generated={qs['generated']} rejected={qs['rejected_quality']} duplicates={qs['duplicates']} accepted={qs['accepted']}\n")
        f.write(f"- rejection reasons: {qs['reasons']}\n- per-class final counts: {got}\n")
        f.write(f"- generation metrics: {'see metrics/' if isinstance(s['generation_metrics'], dict) else s['generation_metrics']}\n")
        f.write(f"- downstream evaluation: {s['downstream']}\n- elapsed: {report['elapsed_minutes']} min\n")
    log.info("Done in %.1f min. Report: %s", report["elapsed_minutes"], out / "final_report.md")


if __name__ == "__main__":
    run_cli(main)
