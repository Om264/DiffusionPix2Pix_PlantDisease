"""Stand-alone split planner: writes ``splits.csv`` (src_path,class,split) without copying images.

    python -m preprocessing.split_dataset --data_dir ./data/raw --output_csv ./data/splits.csv

(``prepare_dataset`` performs the same logic and also materialises the processed dataset.)
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from common.errors import run_cli
from common.logging_utils import setup_logging
from dataio.metadata import discover_images
from dataio.splits import assign_splits, find_leaks
from preprocessing.prepare_dataset import index_images


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_csv", default="splits.csv")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--leak_hamming", type=int, default=2)
    a = ap.parse_args(argv)
    setup_logging()
    df, fmt = discover_images(a.data_dir)
    df = assign_splits(index_images(df, 8), fmt, a.val_frac, a.test_frac, a.seed)
    df = df.drop(index=find_leaks(df, a.leak_hamming))
    df[["src_path", "class", "split"]].to_csv(a.output_csv, index=False)
    print(df.groupby(["class", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    run_cli(main)
