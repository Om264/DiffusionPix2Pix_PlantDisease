"""PyTorch datasets for training (target/source/mask/class/severity) and helpers to load generation sources."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info

from common.errors import UserError
from common.imageio_utils import load_mask, load_rgb
from dataio.transforms import (CropParams, PriorConfig, apply_geometry, build_source_prior, color_jitter,
                               sample_crop_params, to_tensor_m11)

log = logging.getLogger(__name__)


def severity_allowed(source: str, mode: str) -> bool:
    """Which severity provenance may be used as a *training condition*.

    mode 'real'      -> only expert/real labels;  mode 'estimated' -> real + colour-heuristic estimates;
    mode 'none'      -> never."""
    if mode == "none":
        return False
    if mode == "real":
        return source == "real"
    return source in ("real", "estimated")


class PlantDiseaseDataset(Dataset):
    """One item = one (source prior, mask, class, severity) -> target training example.

    pairing='pseudo_prior' : the source is derived from the target leaf itself (see dataio.transforms).
    pairing='explicit'     : ``pairs_csv`` lists real (source, target) pairs: columns
                             source_path,target_path,class[,severity,mask_path] (paths relative to processed_dir
                             or absolute).
    """

    def __init__(self, meta: pd.DataFrame, processed_dir: Union[str, Path], class_to_idx: Dict[str, int],
                 resolution: int, train: bool, data_cfg: Dict[str, Any], seed: int = 0):
        self.root = Path(processed_dir)
        self.meta = meta.reset_index(drop=True)
        self.c2i, self.size, self.train, self.seed = class_to_idx, resolution, train, seed
        self.aug = data_cfg.get("augment", {}) if train else {"enabled": False}
        self.prior_cfg = PriorConfig.from_dict(data_cfg.get("prior", {}))
        self.pairing = data_cfg.get("pairing", "pseudo_prior")
        self.sev_mode = data_cfg.get("severity_mode", "estimated")
        self.jitter = float(self.aug.get("color_jitter", 0.0)) if train else 0.0
        self._ctr = 0
        if self.pairing == "explicit":
            pc = data_cfg.get("pairs_csv")
            if not pc or not Path(pc).exists():
                raise UserError("pairing='explicit' requires data.pairs_csv (columns: source_path,target_path,class).")
            self.meta = pd.read_csv(pc)
            miss = {"source_path", "target_path", "class"} - set(self.meta.columns)
            if miss:
                raise UserError(f"pairs_csv is missing column(s): {sorted(miss)}")
            if "split" in self.meta.columns:
                self.meta = self.meta[self.meta["split"] == ("train" if train else "val")].reset_index(drop=True)
        elif self.pairing != "pseudo_prior":
            raise UserError(f"Unknown data.pairing '{self.pairing}' (use 'pseudo_prior' or 'explicit').")

    def __len__(self) -> int:
        return len(self.meta)

    def _rng(self, idx: int) -> np.random.Generator:
        if not self.train:
            return np.random.default_rng(self.seed * 100003 + idx)
        self._ctr += 1
        wi = get_worker_info()
        base = torch.initial_seed() if wi is None else wi.seed
        return np.random.default_rng((int(base) + idx * 7919 + self._ctr * 104729) % (2**32))

    def _path(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else self.root / q

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.meta.iloc[idx]
        rng = self._rng(idx)
        params = sample_crop_params(rng, enabled=bool(self.aug.get("enabled", True)),
                                    scale=tuple(self.aug.get("scale", (0.6, 1.0))),
                                    ratio=tuple(self.aug.get("ratio", (0.9, 1.1))),
                                    flip=bool(self.aug.get("flip", True)), rot90=bool(self.aug.get("rot90", True)))
        if self.pairing == "explicit":
            tgt = load_rgb(self._path(row["target_path"]))
            src = load_rgb(self._path(row["source_path"]))
            mp = row.get("mask_path") if "mask_path" in row.index else None
            m = load_mask(self._path(mp)) if isinstance(mp, str) and mp else np.ones(tgt.shape[:2], np.uint8)
            if src.shape != tgt.shape:
                raise UserError(f"Source/target size mismatch for pair {row['source_path']} / {row['target_path']}")
            tgt_c, edit = apply_geometry(tgt, m, params, self.size)
            src_c, _ = apply_geometry(src, m, params, self.size)
            leaf = edit
        else:
            img = load_rgb(self._path(row["path"]))
            mp = row.get("mask_path")
            m = load_mask(self._path(mp)) if isinstance(mp, str) and mp else np.ones(img.shape[:2], np.uint8)
            tgt_c, leaf = apply_geometry(img, m, params, self.size)
            tgt_c = color_jitter(tgt_c, rng, self.jitter)
            src_c, edit = build_source_prior(tgt_c, leaf, rng, self.prior_cfg)
        sev, known = 0.0, 0.0
        if "severity" in row.index and pd.notna(row["severity"]) and severity_allowed(
                str(row.get("severity_source", "none")), self.sev_mode):
            sev, known = float(np.clip(row["severity"], 0, 1)), 1.0
        return {
            "target": to_tensor_m11(tgt_c), "source": to_tensor_m11(src_c),
            "mask": torch.from_numpy(edit.astype(np.float32))[None],
            "leaf": torch.from_numpy(leaf.astype(np.float32))[None],
            "class_id": torch.tensor(self.c2i[row["class"]], dtype=torch.long),
            "severity": torch.tensor(sev, dtype=torch.float32), "sev_known": torch.tensor(known),
        }


def load_source_sample(img_path: Path, mask_path: Optional[Path], size: int, params: CropParams):
    """Load a real healthy source image + its leaf mask, apply the (seeded) crop params. Returns (uint8, uint8)."""
    img = load_rgb(img_path)
    m = load_mask(mask_path) if mask_path is not None and Path(mask_path).exists() else np.ones(img.shape[:2], np.uint8)
    return apply_geometry(img, m, params, size)
