"""Build combined real + synthetic datasets. The val and test splits stay REAL ONLY.

    python -m preprocessing.build_dataset --processed_dir data/processed --synthetic_dir outputs/filtered \
        --output_dir outputs/combined --ratios 0 0.1 0.25 0.5 1.0

Creates  output_dir/real_only/  and  output_dir/real_plus_syn_<ratio>/  each with {train,val,test}/<class>/ and
dataset_manifest.csv (origin = real | synthetic). Synthetic count per class:
    per_class        n_syn(c) = round(ratio * n_real_train(c))                       (capped by what is available)
    balance_to_max   n_syn(c) = round(ratio * (max_c' n_real_train(c') - n_real_train(c)))   (ratio 1.0 = fully balanced)
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common.errors import UserError, run_cli
from common.logging_utils import setup_logging
from dataio.metadata import load_metadata
from dataio.splits import assert_no_overlap

log = logging.getLogger("build_dataset")


def select_synthetic(real_train_counts: Dict[str, int], syn: pd.DataFrame, ratio: float, mode: str = "per_class",
                     seed: int = 0) -> pd.DataFrame:
    """Deterministically choose synthetic rows (best quality first if ``quality_score`` exists, else random)."""
    mx = max(real_train_counts.values()) if real_train_counts else 0
    parts = []
    rng = np.random.default_rng(seed)
    for c, g in syn.groupby("class"):
        n_real = real_train_counts.get(c, 0)
        want = round(ratio * n_real) if mode == "per_class" else round(ratio * max(mx - n_real, 0))
        if want <= 0:
            continue
        if want > len(g):
            log.warning("Class '%s': %d synthetic images requested but only %d accepted are available.", c, want, len(g))
        g = g.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))  # random subset, not always the top scores
        parts.append(g.head(want))
    return pd.concat(parts) if parts else syn.iloc[0:0]


def check_synthetic_provenance(syn: pd.DataFrame) -> None:
    if "source_split" in syn.columns and (syn["source_split"] != "train").any():
        raise UserError("Some synthetic images were generated from val/test source images. That would contaminate the "
                        "evaluation splits. Regenerate with sources from the train split only.")


def _place(src: Path, dst: Path, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        try:
            os.link(src, dst); return
        except OSError:
            pass
    shutil.copy2(src, dst)


def build_dataset(processed_dir: str, synthetic_dir: Optional[str], output_dir: str, ratios: Sequence[float],
                  ratio_mode: str = "per_class", seed: int = 0, link: bool = False) -> List[Path]:
    pdir, out = Path(processed_dir), Path(output_dir)
    meta = load_metadata(pdir)
    assert_no_overlap(meta)
    syn = None
    if synthetic_dir:
        sdir = Path(synthetic_dir)
        if not (sdir / "filtered_metadata.csv").exists():
            raise UserError(f"{sdir / 'filtered_metadata.csv'} not found. Run quality_control first.")
        syn = pd.read_csv(sdir / "filtered_metadata.csv")
        check_synthetic_provenance(syn)
    counts = meta[meta["split"] == "train"].groupby("class").size().to_dict()
    made = []
    for ratio in ratios:
        name = "real_only" if ratio == 0 else f"real_plus_syn_{ratio:g}"
        if ratio > 0 and syn is None:
            raise UserError("--synthetic_dir is required for ratios > 0.")
        root = out / name
        manifest = []
        for r in meta.to_dict("records"):
            rel = Path(r["split"]) / r["class"] / Path(r["path"]).name
            _place(pdir / r["path"], root / rel, link)
            manifest.append({"path": str(rel).replace("\\", "/"), "class": r["class"], "split": r["split"], "origin": "real",
                             "image_id": r["image_id"]})
        if ratio > 0:
            sel = select_synthetic(counts, syn, ratio, ratio_mode, seed)
            if len(sel) == 0:
                log.warning("No synthetic images were selected for ratio %g (none accepted by quality control?). "
                            "'%s' will be identical to real_only.", ratio, name)
            for r in sel.to_dict("records"):
                rel = Path("train") / r["class"] / Path(r["path"]).name
                _place(Path(synthetic_dir) / r["path"], root / rel, link)
                manifest.append({"path": str(rel).replace("\\", "/"), "class": r["class"], "split": "train", "origin": "synthetic",
                                 "image_id": r["image_id"]})
        mf = pd.DataFrame(manifest)
        # hard guarantees ------------------------------------------------------------------------------------
        if (mf[mf["split"].isin(["val", "test"])]["origin"] != "real").any():
            raise RuntimeError("Synthetic image found in val/test: aborting.")
        mf.to_csv(root / "dataset_manifest.csv", index=False)
        log.info("%s: train real=%d synthetic=%d | val=%d test=%d (real only)", name, int(((mf.split == "train") & (mf.origin == "real")).sum()),
                 int(((mf.split == "train") & (mf.origin == "synthetic")).sum()), int((mf.split == "val").sum()), int((mf.split == "test").sum()))
        made.append(root)
    return made


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--synthetic_dir", default=None)
    ap.add_argument("--output_dir", default="./outputs/combined")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--ratio_mode", choices=["per_class", "balance_to_max"], default="per_class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--link", action="store_true")
    a = ap.parse_args(argv)
    setup_logging(Path(a.output_dir) / "build_dataset.log")
    build_dataset(a.processed_dir, a.synthetic_dir, a.output_dir, a.ratios, a.ratio_mode, a.seed, a.link)


if __name__ == "__main__":
    run_cli(main)
