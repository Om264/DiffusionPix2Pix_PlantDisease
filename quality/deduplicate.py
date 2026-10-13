"""Near-duplicate removal for synthetic images (perceptual hash + optional embedding cosine similarity).

    python -m quality.deduplicate --input_dir outputs/generated --output_csv outputs/generated/dedup_report.csv

Greedy per-class: images are visited best-quality first; an image is a duplicate if it is within ``hamming`` bits of
(or has cosine >= ``embed_sim`` to) an already-kept image of the same class.
Limitation: only duplicates inside the same class are detected.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.errors import run_cli
from common.hashing import hamming_matrix, phash
from common.imageio_utils import load_rgb
from common.logging_utils import setup_logging

log = logging.getLogger("deduplicate")


def find_duplicates(df: pd.DataFrame, root: Path, scores: Optional[pd.Series] = None, embeddings: Optional[np.ndarray] = None,
                    hamming: int = 3, embed_sim: Optional[float] = 0.985) -> pd.DataFrame:
    """Returns DataFrame[image_id, is_duplicate, duplicate_of, dup_reason]. ``embeddings`` is row-aligned with ``df``."""
    df = df.reset_index(drop=True)
    hashes = np.array([phash(load_rgb(root / p)) for p in tqdm(df["path"], desc="perceptual hashes")], dtype=np.uint64)
    score = (scores.reset_index(drop=True) if scores is not None else pd.Series(0.0, index=df.index)).values
    res = pd.DataFrame({"image_id": df["image_id"], "is_duplicate": False, "duplicate_of": "", "dup_reason": ""})
    for c, idx in df.groupby("class").groups.items():
        order = sorted(idx, key=lambda i: -score[i])
        kept = []
        for i in order:
            if kept:
                hm = hamming_matrix([hashes[i]], hashes[kept])[0]
                j = int(np.argmin(hm))
                if hm[j] <= hamming:
                    res.loc[i, ["is_duplicate", "duplicate_of", "dup_reason"]] = [True, df.at[kept[j], "image_id"], f"phash({int(hm[j])})"]
                    continue
                if embeddings is not None and embed_sim is not None:
                    sims = embeddings[kept] @ embeddings[i]
                    j = int(np.argmax(sims))
                    if sims[j] >= embed_sim:
                        res.loc[i, ["is_duplicate", "duplicate_of", "dup_reason"]] = [True, df.at[kept[j], "image_id"], f"embed({sims[j]:.3f})"]
                        continue
            kept.append(i)
    return res


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True, help="folder with generation_metadata.csv")
    ap.add_argument("--output_csv", default=None)
    ap.add_argument("--hamming", type=int, default=3)
    ap.add_argument("--embedder", default=None, help="dinov2 | clip | handcrafted (default: phash only)")
    ap.add_argument("--embedder_name", default=None)
    ap.add_argument("--embed_sim", type=float, default=0.985)
    a = ap.parse_args(argv)
    root = Path(a.input_dir)
    setup_logging(root / "deduplicate.log")
    df = pd.read_csv(root / "generation_metadata.csv")
    emb = None
    if a.embedder:
        from quality.embeddings import build_embedder
        e = build_embedder(a.embedder, a.embedder_name)
        emb = e.embed([load_rgb(root / p) for p in tqdm(df["path"], desc="embedding")])
    r = find_duplicates(df, root, None, emb, a.hamming, a.embed_sim)
    out = Path(a.output_csv) if a.output_csv else root / "dedup_report.csv"
    r.to_csv(out, index=False)
    log.info("generated=%d duplicates=%d unique=%d -> %s", len(r), int(r.is_duplicate.sum()), int((~r.is_duplicate).sum()), out)


if __name__ == "__main__":
    run_cli(main)
