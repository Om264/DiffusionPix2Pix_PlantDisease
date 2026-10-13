import numpy as np
import pandas as pd
import pytest
import shutil

from common.errors import UserError
from dataio.dataset import PlantDiseaseDataset, severity_allowed
from dataio.metadata import discover_images, load_class_mapping, load_metadata, species_of, is_healthy_class
from dataio.splits import assert_no_overlap, find_leaks, stratified_split
from dataio.transforms import PriorConfig, apply_geometry, build_source_prior, lesion_fraction, sample_crop_params
from preprocessing.prepare_dataset import prepare_dataset


def test_format_a_discovery(raw_dir):
    df, fmt = discover_images(raw_dir)
    assert fmt == "A" and df["class"].nunique() == 5 and len(df) == 60


def test_format_b_discovery(tmp_path, raw_dir):
    for split, k in (("train", 8), ("val", 2), ("test", 2)):
        for c in ("Tomato___healthy", "Tomato___Late_blight"):
            (tmp_path / split / c).mkdir(parents=True)
            for f in sorted((raw_dir / c).glob("*.png"))[:k]:
                shutil.copy(f, tmp_path / split / c / f"{split}_{f.name}")
    df, fmt = discover_images(tmp_path)
    assert fmt == "B" and set(df["split"]) == {"train", "val", "test"}


def test_no_class_folders_message(tmp_path):
    with pytest.raises(UserError, match="No class folders were found"):
        discover_images(tmp_path)


def test_tiny_class_error(tmp_path, raw_dir):
    (tmp_path / "A").mkdir(); (tmp_path / "B").mkdir()
    for i, f in enumerate(sorted((raw_dir / "Tomato___healthy").glob("*.png"))[:2]):
        shutil.copy(f, tmp_path / "A" / f.name)
    for f in sorted((raw_dir / "Apple___healthy").glob("*.png"))[:10]:
        shutil.copy(f, tmp_path / "B" / f.name)
    with pytest.raises(UserError, match="Only 2 images were found for class 'A'"):
        prepare_dataset(str(tmp_path), str(tmp_path / "out"))


def test_split_is_stratified_and_disjoint(processed):
    m = load_metadata(processed)
    assert_no_overlap(m)
    assert set(m["split"]) == {"train", "val", "test"}
    for c, g in m.groupby("class"):
        assert {"train", "val", "test"} <= set(g["split"]), c


def test_leak_detection_drops_lower_priority_copy():
    df = pd.DataFrame({"image_id": list("abc"), "split": ["train", "test", "train"], "md5": ["x", "x", "y"], "phash": [1, 1, 2 ** 40 + 2 ** 41 + 2 ** 42 + 2 ** 43]})
    assert find_leaks(df) == [0]


def test_metadata_flags(processed):
    m = load_metadata(processed)
    assert m["mask_path"].notna().all() and m["mask_coverage"].between(0.05, 0.98).all()
    assert (m[m["is_healthy"]]["severity"] == 0).all()
    assert set(m["severity_source"]) == {"estimated"}
    assert species_of("Tomato___Late_blight") == "Tomato" and is_healthy_class("Apple___healthy")


def test_source_prior_preserves_background():
    rng = np.random.default_rng(0)
    img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    leaf = np.zeros((64, 64), np.uint8); leaf[10:50, 10:50] = 1
    src, edit = build_source_prior(img, leaf, rng, PriorConfig(region_random_prob=1.0))
    assert (edit <= leaf).all()
    assert (src[edit == 0] == img[edit == 0]).all()
    assert (src[edit > 0] != img[edit > 0]).any()


def test_geometry_deterministic():
    rng = np.random.default_rng(1)
    img = (rng.random((50, 70, 3)) * 255).astype(np.uint8); m = np.ones((50, 70), np.uint8)
    p = sample_crop_params(rng)
    a1, m1 = apply_geometry(img, m, p, 32); a2, m2 = apply_geometry(img, m, p, 32)
    assert a1.shape == (32, 32, 3) and (a1 == a2).all() and (m1 == m2).all()


def test_dataset_item_and_severity_gating(processed):
    c2i = load_class_mapping(processed)
    meta = load_metadata(processed)
    tr = meta[meta["split"] == "train"]
    cfg = {"pairing": "pseudo_prior", "severity_mode": "estimated", "augment": {"enabled": True}, "prior": {}}
    it = PlantDiseaseDataset(tr, processed, c2i, 32, True, cfg)[0]
    assert it["target"].shape == (3, 32, 32) and it["mask"].shape == (1, 32, 32) and it["mask"].max() <= 1
    assert float(it["sev_known"]) == 1.0
    it2 = PlantDiseaseDataset(tr, processed, c2i, 32, False, {**cfg, "severity_mode": "real"})[0]
    assert float(it2["sev_known"]) == 0.0           # estimated severity must NOT be used when mode == real
    assert severity_allowed("real", "real") and not severity_allowed("estimated", "real") and not severity_allowed("real", "none")


def test_lesion_fraction_healthy_vs_diseased():
    leaf = np.ones((32, 32), np.uint8)
    green = np.zeros((32, 32, 3), np.uint8); green[...] = (50, 150, 50)
    brown = green.copy(); brown[:16] = (110, 70, 30)
    assert lesion_fraction(green, leaf) < 0.01 and 0.45 < lesion_fraction(brown, leaf) < 0.55
