"""Generate class-balanced synthetic plant-disease images with a trained DiffusionPix2Pix checkpoint.

    python -m generation.generate --checkpoint outputs/checkpoints/best.pt --total_images 10000 --output_dir outputs/generated
    python -m generation.generate --checkpoint outputs/checkpoints/best.pt --images_per_class 500 --balance_mode uniform
    python -m generation.generate --checkpoint ... --total_images 5000 --balance_mode inverse_frequency
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from common.config import apply_overrides, load_config, save_config
from common.errors import UserError, run_cli
from common.gpu import get_device, log_environment
from common.logging_utils import log_to_file, setup_logging
from common.seed import seed_everything
from dataio.metadata import load_metadata
from generation.batch_generate import (build_source_pools, compute_class_quotas, make_jobs, read_existing_metadata,
                                       run_jobs)
from training.checkpoint import load_model_from_checkpoint, resolve_checkpoint

log = logging.getLogger("generate")


def _run_generation(checkpoint: str, output_dir: str, gcfg: Dict[str, Any], processed_dir: Optional[str] = None,
                   device: str = "auto", seed: int = 42, id_offset: int = 0, quotas: Optional[Dict[str, int]] = None,
                   classes: Optional[List[str]] = None, resume: bool = True, run_tag: str = "",
                   base_cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Plan + run generation. If ``quotas`` is None they are derived from total_images/images_per_class/balance_mode."""
    dev = get_device(device)
    ck_path = resolve_checkpoint(checkpoint)
    model, ck = load_model_from_checkpoint(ck_path, dev, use_ema=gcfg.get("use_ema", True))
    c2i = ck["class_to_idx"]
    pdir = Path(processed_dir or ck["config"]["paths"]["processed_dir"])
    meta = load_metadata(pdir)
    dcfg = ck["config"]["data"]
    sev_ok = bool(ck.get("severity_supported", True))
    if not sev_ok:
        log.warning("This checkpoint was trained WITHOUT severity conditioning: severity will not be used.")
    elif ck.get("severity_mode") == "estimated":
        log.warning("Severity conditions come from ESTIMATED (colour-heuristic) labels, not ground truth.")
    use = [c for c in (classes or sorted(c2i)) if c in c2i]
    if not use:
        raise UserError("None of the requested --classes exist in the checkpoint.")
    real_counts = meta[meta["split"] == "train"].groupby("class").size().reindex(use).fillna(0).astype(int).to_dict()
    if quotas is None:
        total = gcfg["images_per_class"] * len(use) if gcfg.get("images_per_class") else int(gcfg["total_images"])
        quotas = compute_class_quotas(real_counts, total, gcfg["balance_mode"], gcfg.get("inverse_frequency_power", 1.0),
                                      int(gcfg.get("min_per_class", 0)))
        log.info("Target %d images over %d classes (balance_mode=%s). Quotas: %s", total, len(use), gcfg["balance_mode"], quotas)
    pools = build_source_pools(meta, [c for c in quotas if quotas[c] > 0], gcfg["source_pool"], dcfg.get("species_sep", "___"),
                               dcfg.get("healthy_pattern", "healthy"))
    for c, (p, k) in pools.items():
        log.info("source pool for %-35s: %4d images [%s]", c, len(p), k)
    existing = read_existing_metadata(Path(output_dir)) if resume else pd.DataFrame()
    forbid = set(existing["seed"].astype(int)) if len(existing) else set()
    jobs = make_jobs(quotas, pools, c2i, gcfg, seed, id_offset, sev_ok, dcfg.get("healthy_pattern", "healthy"), forbid)
    seeds = [j.seed for j in jobs]
    assert len(seeds) == len(set(seeds)), "seed collision"  # every generated image has a unique seed
    n = run_jobs(model, ck, jobs, Path(output_dir), pdir, gcfg, dev, dcfg.get("species_sep", "___"), str(ck_path), run_tag, resume)
    md = read_existing_metadata(Path(output_dir))
    log.info("Generated %d new image(s); %d total in %s", n, len(md), output_dir)
    return md


def run_generation(checkpoint: str, output_dir: str, *args, **kwargs) -> pd.DataFrame:
    """Plan + run generation (see ``_run_generation``); also writes ``output_dir/generation.log``."""
    with log_to_file(Path(output_dir) / "generation.log"):
        return _run_generation(checkpoint, output_dir, *args, **kwargs)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", default="./outputs/generated")
    ap.add_argument("--config", default="configs/generation.yaml")
    ap.add_argument("--processed_dir", default=None, help="default: taken from the checkpoint's config")
    ap.add_argument("--total_images", type=int, default=None)
    ap.add_argument("--images_per_class", type=int, default=None)
    ap.add_argument("--balance_classes", action="store_true", help="shorthand for --balance_mode uniform")
    ap.add_argument("--balance_mode", choices=["uniform", "inverse_frequency", "proportional"], default=None)
    ap.add_argument("--severity_min", type=float, default=None)
    ap.add_argument("--severity_max", type=float, default=None)
    ap.add_argument("--severity_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--ddim_steps", type=int, default=None)
    ap.add_argument("--guidance_scale", type=float, default=None)
    ap.add_argument("--strength", type=float, default=None)
    ap.add_argument("--source_pool", choices=["auto", "healthy", "same_class", "all"], default=None)
    ap.add_argument("--format", choices=["png", "jpg"], default=None)
    ap.add_argument("--classes", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--oversample", type=float, default=1.0, help="multiply the requested number (to absorb QC rejections)")
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip images already listed in generation_metadata.csv")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.set)
    out = Path(a.output_dir)
    setup_logging()
    g = cfg["generation"]
    for k in ("total_images", "images_per_class", "balance_mode", "severity_min", "severity_max", "severity_steps", "batch_size",
              "ddim_steps", "guidance_scale", "strength", "source_pool", "format"):
        if getattr(a, k) is not None:
            g[k] = getattr(a, k)
    if a.balance_classes and a.balance_mode is None:
        g["balance_mode"] = "uniform"
    if a.no_ema:
        g["use_ema"] = False
    if a.oversample != 1.0:
        g["total_images"] = int(round(g["total_images"] * a.oversample))
        if g.get("images_per_class"):
            g["images_per_class"] = int(round(g["images_per_class"] * a.oversample))
    seed = a.seed if a.seed is not None else cfg["seed"]
    seed_everything(seed)
    save_config({"generation": g, "seed": seed, "checkpoint": a.checkpoint}, out / "config_used_generation.yaml")
    log_environment(get_device(cfg["device"]))
    run_generation(a.checkpoint, str(out), g, a.processed_dir, cfg["device"], seed, resume=a.resume, classes=a.classes)


if __name__ == "__main__":
    run_cli(main)
