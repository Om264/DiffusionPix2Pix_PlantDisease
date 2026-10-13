"""Generation planning (class quotas, source pools, unique seeds) and the resumable batch generation loop."""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from common.errors import UserError
from common.gpu import amp_dtype as get_amp_dtype
from common.imageio_utils import save_mask, save_rgb
from common.seed import derive_seed
from dataio.dataset import load_source_sample
from dataio.masks import fill_holes  # noqa: F401  (re-exported for convenience)
from dataio.metadata import disease_of, is_healthy_class, species_of
from dataio.transforms import CropParams, PriorConfig, from_tensor_m11, random_edit_region, sample_crop_params, to_tensor_m11
from generation.sampling import ddim_sample

log = logging.getLogger("generate")

METADATA_COLUMNS = ["image_id", "path", "source_image", "source_id", "source_split", "source_pool_kind", "class",
                    "class_id", "disease", "species", "severity", "severity_kind", "seed", "mask", "source_crop",
                    "checkpoint", "ckpt_step", "timestamp", "generation_parameters", "run_tag"]


# ------------------------------------------------------------------------------------------------ quotas
def compute_class_quotas(real_counts: Dict[str, int], total: int, mode: str = "uniform", power: float = 1.0,
                         min_per_class: int = 0) -> Dict[str, int]:
    """Distribute ``total`` synthetic images over classes.

    uniform            equal share; remainder goes to the classes with the fewest real images
    inverse_frequency  share ~ (1/real_count)^power  -> minority classes get more
    proportional       share ~ real_count
    Always sums exactly to ``total`` (if total >= min_per_class * n_classes)."""
    classes = sorted(real_counts)
    C = len(classes)
    if C == 0 or total <= 0:
        return {c: 0 for c in classes}
    counts = np.array([max(real_counts[c], 1) for c in classes], dtype=np.float64)
    if mode == "uniform":
        w = np.ones(C)
    elif mode == "inverse_frequency":
        w = (1.0 / counts) ** power
    elif mode == "proportional":
        w = counts
    else:
        raise UserError(f"Unknown balance_mode '{mode}' (uniform | inverse_frequency | proportional).")
    ideal = total * w / w.sum()
    q = np.floor(ideal).astype(int)
    rem = total - int(q.sum())
    frac = ideal - q
    # largest remainder; ties -> fewer real images first, then name (deterministic)
    order = sorted(range(C), key=lambda i: (-round(frac[i], 9), counts[i], classes[i]))
    for i in order[:rem]:
        q[i] += 1
    if min_per_class > 0 and total >= min_per_class * C:
        while (q < min_per_class).any():
            lo, hi = int(np.argmin(q)), int(np.argmax(q))
            q[hi] -= 1; q[lo] += 1
    return {c: int(q[i]) for i, c in enumerate(classes)}


# ------------------------------------------------------------------------------------------------ source pools
def build_source_pools(meta: pd.DataFrame, classes: Sequence[str], source_pool: str = "auto",
                       species_sep: str = "___", healthy_pattern: str = "healthy") -> Dict[str, Tuple[pd.DataFrame, str]]:
    """For every target class pick the real images that will serve as *source image priors*.

    Sources are taken from the TRAIN split only (val/test images are never used -> no contamination)."""
    train = meta[meta["split"] == "train"]
    healthy = train[train["class"].map(lambda c: is_healthy_class(c, healthy_pattern))]
    pools: Dict[str, Tuple[pd.DataFrame, str]] = {}
    for c in classes:
        sp = species_of(c, species_sep)
        same = train[train["class"] == c]
        if source_pool == "same_class":
            pool, kind = same, "same_class"
        elif source_pool == "all":
            pool, kind = train, "all"
        else:
            h = healthy[healthy["species"] == sp] if sp is not None else healthy
            if source_pool == "healthy" and h.empty:
                raise UserError(f"source_pool=healthy but no healthy TRAIN images exist for '{c}' "
                                f"(species={sp}). Check data.healthy_pattern / species_sep.")
            if not h.empty:
                pool, kind = h, "healthy_same_species" if sp is not None else "healthy"
            else:
                pool, kind = same, "same_class_fallback"
                log.warning("No healthy source images for class '%s' (species=%s): falling back to its own training "
                            "images as sources. The model then edits already-diseased leaves; consider adding "
                            "healthy images of this species.", c, sp)
        if pool.empty:
            raise UserError(f"No source images available for class '{c}'.")
        pools[c] = (pool.reset_index(drop=True), kind)
    return pools


# ------------------------------------------------------------------------------------------------ jobs
@dataclass
class Job:
    image_id: str
    class_name: str
    class_id: int
    source_row: Dict[str, Any]
    pool_kind: str
    severity: float          # nan => no severity condition
    seed: int
    crop: CropParams
    guidance: float
    region_seed: int


def severity_grid(smin: float, smax: float, steps: int) -> np.ndarray:
    if steps <= 1:
        return np.array([(smin + smax) / 2])
    return np.linspace(smin, smax, steps)


def make_jobs(quotas: Dict[str, int], pools: Dict[str, Tuple[pd.DataFrame, str]], class_to_idx: Dict[str, int],
              gcfg: Dict[str, Any], base_seed: int, id_offset: int = 0, severity_supported: bool = True,
              healthy_pattern: str = "healthy", forbid_seeds: Optional[set] = None) -> List[Job]:
    grid = severity_grid(gcfg["severity_min"], gcfg["severity_max"], gcfg["severity_steps"])
    jobs: List[Job] = []
    seen = set(forbid_seeds or ())
    counter = id_offset
    for c in sorted(quotas):
        n = quotas[c]
        if n <= 0:
            continue
        pool, kind = pools[c]
        rng_c = np.random.default_rng(derive_seed(base_seed, 10_000_000 + class_to_idx[c]))
        order = np.concatenate([rng_c.permutation(len(pool)) for _ in range(int(np.ceil(n / len(pool))))])[:n]
        for k in range(n):
            gidx = counter
            counter += 1
            seed = derive_seed(base_seed, gidx)
            while seed in seen:  # hard uniqueness guarantee
                seed = (seed + 1) % (2**31 - 1)
            seen.add(seed)
            r = np.random.default_rng(seed)
            crop = sample_crop_params(r, **{k_: tuple(v) if isinstance(v, list) else v for k_, v in gcfg["crop"].items()})
            if not severity_supported:
                sev = float("nan")
            elif is_healthy_class(c, healthy_pattern):
                sev = float(gcfg["healthy_severity"])
            else:
                sev = float(r.choice(grid))
            g = max(1.0, float(gcfg["guidance_scale"]) + float(r.uniform(-1, 1)) * float(gcfg.get("guidance_jitter", 0.0))) \
                if gcfg["guidance_scale"] != 1.0 else 1.0
            jobs.append(Job(f"syn_{class_to_idx[c]:03d}_{gidx:07d}", c, class_to_idx[c], pool.iloc[int(order[k])].to_dict(),
                            kind, sev, seed, crop, g, int(r.integers(0, 2**31 - 1))))
    return jobs


def read_existing_metadata(out_dir: Path) -> pd.DataFrame:
    p = out_dir / "generation_metadata.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=METADATA_COLUMNS)


# ------------------------------------------------------------------------------------------------ main loop
@torch.no_grad()
def run_jobs(model, ckpt: Dict[str, Any], jobs: List[Job], out_dir: Path, processed_dir: Path, gcfg: Dict[str, Any],
             device: torch.device, species_sep: str, ckpt_path: str, run_tag: str = "", resume: bool = True) -> int:
    """Generate all jobs (skipping those already present when ``resume``). Returns number newly generated."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "generation_metadata.csv"
    done = set(read_existing_metadata(out_dir)["image_id"]) if resume else set()
    todo = [j for j in jobs if j.image_id not in done]
    log.info("%d image(s) planned, %d already done, %d to generate", len(jobs), len(jobs) - len(todo), len(todo))
    if not todo:
        return 0
    res = int(ckpt["config"]["data"]["resolution"])
    amp = get_amp_dtype(gcfg.get("precision", "fp32"), device)
    ext = "jpg" if gcfg["format"].lower() in ("jpg", "jpeg") else "png"
    new_file = not meta_path.exists() or not resume
    fh = open(meta_path, "w" if new_file else "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=METADATA_COLUMNS)
    if new_file:
        writer.writeheader()
    n_done, bs = 0, int(gcfg["batch_size"])
    try:
        for s in tqdm(range(0, len(todo), bs), desc="generating"):
            batch = todo[s:s + bs]
            srcs, masks = [], []
            for j in batch:
                row = j.source_row
                mp = processed_dir / row["mask_path"] if isinstance(row.get("mask_path"), str) else None
                img, m = load_source_sample(processed_dir / row["path"], mp, res, j.crop)
                p_region = float(gcfg.get("region_random_prob", 0.0))
                if p_region > 0:
                    m = random_edit_region(m, np.random.default_rng(j.region_seed), PriorConfig(region_random_prob=p_region))
                srcs.append(img); masks.append(m)
            src_t = torch.stack([to_tensor_m11(x) for x in srcs]).to(device)
            msk_t = torch.stack([torch.from_numpy(m.astype(np.float32))[None] for m in masks]).to(device)
            cls_t = torch.tensor([j.class_id for j in batch], device=device)
            sev = torch.tensor([0.0 if np.isnan(j.severity) else j.severity for j in batch], device=device)
            known = torch.tensor([0.0 if np.isnan(j.severity) else 1.0 for j in batch], device=device)
            gs = torch.tensor([j.guidance for j in batch], device=device)
            imgs = ddim_sample(model, src_t, msk_t, cls_t, sev, known, [j.seed for j in batch], gcfg["ddim_steps"], gs,
                               gcfg["strength"], gcfg["eta"], gcfg["composite"], gcfg["feather"], amp)
            arr = from_tensor_m11(imgs)
            for i, j in enumerate(batch):
                rel = Path(j.class_name) / f"{j.image_id}.{ext}"
                save_rgb(arr[i], out_dir / rel)
                save_rgb(srcs[i], out_dir / "_aux" / "sources" / f"{j.image_id}.png")
                save_mask(masks[i], out_dir / "_aux" / "masks" / f"{j.image_id}.png")
                params = {"ddim_steps": gcfg["ddim_steps"], "guidance_scale": round(j.guidance, 4), "strength": gcfg["strength"],
                          "eta": gcfg["eta"], "composite": gcfg["composite"], "feather": gcfg["feather"], "crop": j.crop.to_dict(),
                          "use_ema": gcfg.get("use_ema", True), "resolution": res, "region_random_prob": gcfg.get("region_random_prob", 0.0)}
                writer.writerow({
                    "image_id": j.image_id, "path": str(rel).replace("\\", "/"), "source_image": j.source_row["path"],
                    "source_id": j.source_row["image_id"], "source_split": j.source_row["split"], "source_pool_kind": j.pool_kind,
                    "class": j.class_name, "class_id": j.class_id, "disease": disease_of(j.class_name, species_sep),
                    "species": species_of(j.class_name, species_sep) or "",
                    "severity": "" if np.isnan(j.severity) else round(j.severity, 4),
                    "severity_kind": "none" if np.isnan(j.severity) else "synthetic_condition (NOT a measured severity)",
                    "seed": j.seed, "mask": f"_aux/masks/{j.image_id}.png", "source_crop": f"_aux/sources/{j.image_id}.png",
                    "checkpoint": ckpt_path, "ckpt_step": ckpt.get("step", ""), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "generation_parameters": json.dumps(params), "run_tag": run_tag})
            fh.flush()
            n_done += len(batch)
    finally:
        fh.close()
    return n_done
