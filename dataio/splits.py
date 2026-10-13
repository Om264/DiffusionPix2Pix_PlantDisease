"""Stratified train/val/test splitting and cross-split leakage detection."""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from common.hashing import hamming_matrix

log = logging.getLogger(__name__)
_PRIORITY = {"test": 0, "val": 1, "train": 2}  # when a duplicate spans splits, keep the higher-priority copy


def _split_counts(n: int, val_frac: float, test_frac: float) -> Tuple[int, int]:
    n_test = int(round(n * test_frac)) if test_frac > 0 else 0
    n_val = int(round(n * val_frac)) if val_frac > 0 else 0
    if n >= 3:
        n_test = max(n_test, 1) if test_frac > 0 else 0
        n_val = max(n_val, 1) if val_frac > 0 else 0
    while n - n_test - n_val < 1 and (n_test + n_val) > 0:  # always keep >=1 training image
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        else:
            n_val -= 1
    return n_val, n_test


def stratified_split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> pd.Series:
    """Per-class random split. Returns a Series of 'train'/'val'/'test' aligned with ``df.index``."""
    rng = np.random.default_rng(seed)
    out = pd.Series("train", index=df.index, dtype=object)
    for _, idx in df.groupby("class").groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n_val, n_test = _split_counts(len(idx), val_frac, test_frac)
        out.loc[idx[:n_test]] = "test"
        out.loc[idx[n_test:n_test + n_val]] = "val"
    return out


def assign_splits(df: pd.DataFrame, fmt: str, val_frac: float, test_frac: float, seed: int) -> pd.DataFrame:
    """Format A: random stratified split. Format B: keep given splits, carve missing val/test from train."""
    df = df.copy()
    if fmt == "A":
        df["split"] = stratified_split(df, val_frac, test_frac, seed)
        return df
    present = set(df["split"].unique())
    train = df[df["split"] == "train"]
    need_val, need_test = "val" not in present, "test" not in present
    if need_val or need_test:
        vf = val_frac if need_val else 0.0
        tf = test_frac if need_test else 0.0
        log.warning("Dataset has no %s split; carving %s from the training folder (stratified, seed=%d).",
                    "/".join(s for s, n in (("val", need_val), ("test", need_test)) if n),
                    f"val={vf:.0%} test={tf:.0%}", seed)
        sub = stratified_split(train, vf, tf, seed)
        df.loc[sub.index, "split"] = sub
    return df


def find_leaks(df: pd.DataFrame, leak_hamming: int = 2, chunk: int = 1500) -> List[int]:
    """Return row indices to DROP so that no exact / near-identical image appears in two splits.

    Exact duplicates (md5) and near duplicates (pHash Hamming <= ``leak_hamming``) across different splits are
    resolved by keeping the copy in the higher-priority split (test > val > train). Limitation: pHash does not
    catch rotated/flipped copies of the same physical leaf; supply your own grouping if your data has them.
    """
    drop: set = set()
    d = df.copy()
    d["_prio"] = d["split"].map(_PRIORITY)
    for _, g in d.groupby("md5"):
        if g["split"].nunique() > 1:
            keep = g.sort_values("_prio").index[0]
            drop.update(i for i in g.index if i != keep and d.loc[i, "split"] != d.loc[keep, "split"])
    if leak_hamming >= 0:
        hashes = d["phash"].astype("uint64").values
        for hi_split, lo_split in (("test", "train"), ("test", "val"), ("val", "train")):
            hi = d[d["split"] == hi_split]
            lo = d[(d["split"] == lo_split) & (~d.index.isin(drop))]
            if hi.empty or lo.empty:
                continue
            hi_h = hashes[[d.index.get_loc(i) for i in hi.index]]
            lo_idx = list(lo.index)
            lo_h = hashes[[d.index.get_loc(i) for i in lo_idx]]
            for s in range(0, len(lo_h), chunk):
                hm = hamming_matrix(lo_h[s:s + chunk], hi_h)
                hit = np.where((hm <= leak_hamming).any(axis=1))[0]
                drop.update(lo_idx[s + int(j)] for j in hit)
    if drop:
        log.warning("Leakage check: dropping %d image(s) duplicated across splits (kept in the higher-priority split).",
                    len(drop))
    return sorted(drop)


def assert_no_overlap(df: pd.DataFrame) -> None:
    """Hard guarantee used by downstream code: an image_id may live in exactly one split."""
    dup = df.groupby("image_id")["split"].nunique()
    if (dup > 1).any():
        raise RuntimeError(f"Data leakage: image_ids in several splits: {list(dup[dup > 1].index[:5])}")
