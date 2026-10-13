"""Dataset discovery (Format A / Format B), class mapping and the processed-dataset metadata table."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from common.errors import UserError
from common.imageio_utils import list_images

log = logging.getLogger(__name__)

SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test", "testing": "test"}

META_COLUMNS = ["image_id", "path", "class", "class_id", "species", "is_healthy", "split", "mask_path",
                "mask_source", "mask_coverage", "severity", "severity_source", "width", "height", "md5", "phash"]


def species_of(class_name: str, sep: Optional[str] = "___") -> Optional[str]:
    """'Tomato___Late_blight' -> 'Tomato'. Returns None if the class name has no species prefix."""
    if sep and sep in class_name:
        return class_name.split(sep, 1)[0]
    return None


def disease_of(class_name: str, sep: Optional[str] = "___") -> str:
    """'Tomato___Late_blight' -> 'Late_blight' (whole name when there is no separator)."""
    return class_name.split(sep, 1)[1] if sep and sep in class_name else class_name


def is_healthy_class(class_name: str, pattern: str = "healthy") -> bool:
    return re.search(pattern, class_name, flags=re.IGNORECASE) is not None


def _class_dirs(base: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")):
        imgs = list_images(d)
        if imgs:
            out[d.name] = imgs
        else:
            log.warning("Folder '%s' contains no images and is ignored.", d)
    return out


def discover_images(data_dir: Union[str, Path]) -> Tuple[pd.DataFrame, str]:
    """Scan ``data_dir`` and return (DataFrame[src_path, class, split], format).

    Format A: ``data_dir/<class>/*.jpg``  (split == "" -> assigned later)
    Format B: ``data_dir/{train,val,test}/<class>/*.jpg`` (split taken from the folder)
    """
    base = Path(data_dir)
    if not base.exists() or not base.is_dir():
        raise UserError(f"The data directory does not exist: {base}\n"
                        "Pass --data_dir pointing to a folder that contains one sub-folder per class.")
    subdirs = [p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")]
    split_dirs = {SPLIT_ALIASES[p.name.lower()]: p for p in subdirs if p.name.lower() in SPLIT_ALIASES}
    rows = []
    if split_dirs and len(split_dirs) == len(subdirs) and "train" in split_dirs:
        fmt = "B"
        for split, d in split_dirs.items():
            for cls, imgs in _class_dirs(d).items():
                rows += [(str(p), cls, split) for p in imgs]
    else:
        fmt = "A"
        for cls, imgs in _class_dirs(base).items():
            rows += [(str(p), cls, "") for p in imgs]
    if not rows:
        raise UserError(
            f"No class folders were found in {base}.\n"
            "Expected either\n  data_dir/<class_name>/*.jpg   (Format A)\nor\n"
            "  data_dir/{train,val,test}/<class_name>/*.jpg   (Format B)")
    df = pd.DataFrame(rows, columns=["src_path", "class", "split"])
    log.info("Detected dataset Format %s: %d images, %d classes", fmt, len(df), df["class"].nunique())
    return df, fmt


def make_class_mapping(classes) -> Dict[str, int]:
    """Deterministic class -> id mapping (alphabetical). Never hard-codes disease names."""
    return {c: i for i, c in enumerate(sorted(set(classes)))}


def save_class_mapping(mapping: Dict[str, int], processed_dir: Union[str, Path]) -> None:
    with open(Path(processed_dir) / "classes.json", "w") as f:
        json.dump(mapping, f, indent=2)


def load_class_mapping(processed_dir: Union[str, Path]) -> Dict[str, int]:
    p = Path(processed_dir) / "classes.json"
    if not p.exists():
        raise UserError(f"{p} not found. Run `python -m preprocessing.prepare_dataset` first.")
    with open(p) as f:
        return json.load(f)


def load_metadata(processed_dir: Union[str, Path]) -> pd.DataFrame:
    p = Path(processed_dir) / "metadata.csv"
    if not p.exists():
        raise UserError(f"{p} not found. Run `python -m preprocessing.prepare_dataset` first.")
    df = pd.read_csv(p, keep_default_na=False, na_values=[""])
    for c in ("mask_path", "mask_source", "severity_source", "species"):  # all-empty columns are read as float
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    if "is_healthy" in df.columns:
        df["is_healthy"] = df["is_healthy"].astype(bool)
    return df


def save_metadata(df: pd.DataFrame, processed_dir: Union[str, Path]) -> Path:
    p = Path(processed_dir) / "metadata.csv"
    df.to_csv(p, index=False)
    return p


def check_min_images(df: pd.DataFrame, min_images: int, split_col: str = "split") -> None:
    """Raise a friendly error if any class has fewer than ``min_images`` training images."""
    train = df[df[split_col] == "train"] if (df[split_col] == "train").any() else df
    counts = train.groupby("class").size()
    bad = counts[counts < min_images]
    if len(bad):
        msg = "\n".join(f"Only {n} training images were found for class '{c}'." for c, n in bad.items())
        raise UserError(f"{msg}\nEach class needs at least {min_images} training images "
                        f"(see data.min_images_per_class). Add images or remove the class folder.")
