import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import load_config  # noqa: E402
from dataio.metadata import load_class_mapping  # noqa: E402
from models.diffusion_pix2pix import DiffusionPix2Pix, ModelConfig  # noqa: E402
from preprocessing.estimate_severity import estimate_severity  # noqa: E402
from preprocessing.prepare_dataset import prepare_dataset  # noqa: E402
from preprocessing.prepare_masks import prepare_masks  # noqa: E402
from scripts.make_dummy_dataset import make_dummy_dataset  # noqa: E402


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw") / "raw"
    make_dummy_dataset(str(d), n_per_class=12, size=64, seed=0)
    return d


@pytest.fixture(scope="session")
def processed(tmp_path_factory, raw_dir):
    out = tmp_path_factory.mktemp("proc") / "processed"
    prepare_dataset(str(raw_dir), str(out), seed=0, min_images_per_class=5)
    prepare_masks(str(out), "hsv", preview=0)
    estimate_severity(str(out))
    return out


@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory, processed):
    """A randomly initialised (UNTRAINED) tiny checkpoint: only used to test plumbing, never quality."""
    c2i = load_class_mapping(processed)
    cfg = load_config(str(ROOT / "configs/smoke.yaml"), [f"paths.processed_dir={processed}", "data.resolution=32"])
    mc = ModelConfig.from_dict({**cfg["model"], "resolution": 32})
    torch.manual_seed(0)
    m = DiffusionPix2Pix(mc, len(c2i))
    p = tmp_path_factory.mktemp("ck") / "best.pt"
    torch.save({"model": m.trainable_state_dict(), "ema": {}, "step": 0, "best_val": 0.0, "config": cfg, "model_cfg": mc.to_dict(),
                "class_to_idx": c2i, "severity_supported": True, "severity_mode": "estimated"}, p)
    return p
