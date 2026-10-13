import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from common.config import load_config
from common.errors import UserError
from common.imageio_utils import save_mask, save_rgb
from dataio.metadata import load_metadata
from evaluation import metrics as M
from preprocessing.build_dataset import build_dataset, check_synthetic_provenance, select_synthetic
from quality.deduplicate import find_duplicates
from quality.embeddings import DinoEmbedder, build_embedder
from quality.filtering import check_one
from quality.quality_control import run_quality_control
from scripts.make_dummy_dataset import make_leaf

Q = load_config("configs/evaluation.yaml")["quality"]


def _make_case(tmp_path, kind="good", cls="Tomato___Late_blight", seed=0):
    rng = np.random.default_rng(seed)
    src = make_leaf(rng, 64, "healthy", 0.0)
    mask = (cv2.cvtColor(src, cv2.COLOR_RGB2HSV)[..., 1] > 60).astype(np.uint8)
    syn = src.copy()
    if kind == "good":      # brown lesions inside the leaf only
        ys, xs = np.nonzero(mask)
        for j in rng.integers(0, len(ys), 12):
            cv2.circle(syn, (int(xs[j]), int(ys[j])), 4, (90, 60, 30), -1)
        syn[mask == 0] = src[mask == 0]
    elif kind == "noise":
        syn = np.clip(src + rng.normal(0, 60, src.shape), 0, 255).astype(np.uint8)
    elif kind == "blur":
        syn = cv2.GaussianBlur(src, (0, 0), 6)
    elif kind == "bg":
        syn[mask == 0] = (200, 40, 200); syn[mask > 0] = (90, 60, 30)
    elif kind == "unchanged":
        pass
    iid = f"{kind}_{seed}"
    save_rgb(syn, tmp_path / cls / f"{iid}.png"); save_rgb(src, tmp_path / "_aux/sources" / f"{iid}.png"); save_mask(mask, tmp_path / "_aux/masks" / f"{iid}.png")
    return {"image_id": iid, "path": f"{cls}/{iid}.png", "source_crop": f"_aux/sources/{iid}.png", "mask": f"_aux/masks/{iid}.png", "class": cls}


def test_good_image_passes_and_bad_ones_fail(tmp_path):
    ok = check_one(_make_case(tmp_path, "good"), tmp_path, Q)
    assert ok["passed"], ok["reasons"]
    expected = {"blur": "blurry", "bg": "background_modified", "unchanged": "no_change_in_mask", "noise": "noisy_or_artifacts"}
    for kind, reason in expected.items():
        r = check_one(_make_case(tmp_path, kind), tmp_path, Q)
        assert not r["passed"] and reason in r["reasons"], (kind, r["reasons"])
    h = check_one(_make_case(tmp_path, "good", cls="Tomato___healthy", seed=1), tmp_path, Q)       # healthy class must not gain lesions
    assert "healthy_has_lesions" in h["reasons"] or h["lesion_syn"] <= Q["healthy_lesion_max"]
    bad = check_one({"image_id": "x", "path": "nope.png", "source_crop": "n", "mask": "m", "class": "c"}, tmp_path, Q)
    assert bad["reasons"] == "invalid_image"


def test_dedup_finds_near_duplicates(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(4):
        img = make_leaf(rng, 64, "Early_blight", 0.5)
        for suffix, noise in (("a", 0), ("b", 1)):               # b is a near-copy of a
            arr = np.clip(img.astype(int) + noise * rng.integers(-1, 2, img.shape), 0, 255).astype(np.uint8)
            save_rgb(arr, tmp_path / f"c/{i}{suffix}.png")
            rows.append({"image_id": f"{i}{suffix}", "path": f"c/{i}{suffix}.png", "class": "c"})
    df = pd.DataFrame(rows)
    r = find_duplicates(df, tmp_path, pd.Series(np.arange(len(df), 0, -1.0)), None, hamming=3)
    assert r["is_duplicate"].sum() == 4 and r[r.is_duplicate]["image_id"].str.endswith("b").all()
    # unique images are all kept
    assert find_duplicates(df[df.image_id.str.endswith("a")], tmp_path, None, None, hamming=3)["is_duplicate"].sum() == 0


def test_frechet_and_kid_sanity():
    rng = np.random.default_rng(0)
    a, b, c = rng.normal(0, 1, (400, 8)), rng.normal(0, 1, (400, 8)), rng.normal(3, 1, (400, 8))
    assert M.frechet_distance(a, b) < 0.5 < M.frechet_distance(a, c)
    k_same, _ = M.kernel_inception_distance(a, b, 20, 100); k_diff, _ = M.kernel_inception_distance(a, c, 20, 100)
    assert k_diff > k_same
    assert M.kernel_inception_distance(a[:1], b[:1])[0] != M.kernel_inception_distance(a[:1], b[:1])[0]  # NaN for n < 2


def test_masked_psnr_ssim():
    a = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8)
    reg = np.zeros((32, 32), bool); reg[:16] = True
    assert M.masked_psnr(a, a, reg) == float("inf") and M.masked_ssim(a, a, reg) == pytest.approx(1.0)
    b = a.copy(); b[16:] = 0
    assert M.masked_psnr(a, b, reg) == float("inf") and M.masked_psnr(a, b, ~reg) < 20


def test_embedders(tmp_path):
    imgs = [make_leaf(np.random.default_rng(i), 64, "Early_blight", 0.5) for i in range(3)]
    e = build_embedder("handcrafted").embed(imgs)
    assert e.shape[0] == 3 and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-5)
    from transformers import Dinov2Config, Dinov2Model
    Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=224, patch_size=14, intermediate_size=64)
                ).save_pretrained(tmp_path / "dino")
    d = DinoEmbedder(str(tmp_path / "dino"), "cpu").embed(imgs)
    assert d.shape == (3, 32) and np.allclose(np.linalg.norm(d, axis=1), 1, atol=1e-5)
    with pytest.raises(UserError, match="Could not load"):
        DinoEmbedder(str(tmp_path / "missing_model"), "cpu")


def test_lpips_wrapper_error_or_runs():
    try:
        l = M.LPIPSMetric("cpu", pretrained_backbone=False)      # random backbone: exercises the code path only
    except UserError:
        pytest.skip("lpips not installed")
    a = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    assert l.distance(a, a) == pytest.approx(0.0, abs=1e-5)


def test_qc_end_to_end_and_build_dataset(tmp_path, processed):
    gen = tmp_path / "gen"
    rows = []
    for i, kind in enumerate(["good"] * 6 + ["noise", "blur"]):
        r = _make_case(gen, kind, seed=i)
        r.update(source_image="train/x.png", source_id="x", source_split="train", severity=0.5, seed=i, class_id=0)
        rows.append(r)
    pd.DataFrame(rows).to_csv(gen / "generation_metadata.csv", index=False)
    q = dict(Q); q["embedder"] = "handcrafted"; q["source_sim_min"] = 0.3
    rep = run_quality_control(str(gen), str(tmp_path / "filt"), q, dedup=True)
    assert rep["accepted"].sum() >= 1 and not rep[rep.image_id.str.startswith(("noise", "blur"))]["accepted"].any()
    fm = pd.read_csv(tmp_path / "filt" / "filtered_metadata.csv")
    assert all((tmp_path / "filt" / p).exists() for p in fm["path"])
    # synthetic images from a non-train source must be refused
    with pytest.raises(UserError, match="contaminate"):
        check_synthetic_provenance(fm.assign(source_split="test"))
    # combined dataset: test/val real only, ratio semantics
    fm2 = pd.concat([fm.assign(**{"class": "Tomato___Late_blight"})] * 6, ignore_index=True)
    sel = select_synthetic({"Tomato___Late_blight": 8}, fm2, 0.5, "per_class")
    assert len(sel) == 4 and len(select_synthetic({"Tomato___Late_blight": 8}, fm2, 0.0)) == 0
    assert len(select_synthetic({"a": 10, "b": 4}, pd.DataFrame({"class": ["a"] * 9 + ["b"] * 9, "path": "p"}), 1.0, "balance_to_max")) == 6
    made = build_dataset(str(processed), str(tmp_path / "filt"), str(tmp_path / "comb"), [0.0, 0.5], "per_class")
    for root in made:
        mf = pd.read_csv(root / "dataset_manifest.csv")
        assert (mf[mf.split.isin(["val", "test"])]["origin"] == "real").all()
    base = pd.read_csv(made[0] / "dataset_manifest.csv"); plus = pd.read_csv(made[1] / "dataset_manifest.csv")
    assert (base["origin"] == "real").all() and (plus["origin"] == "synthetic").sum() >= 1
    assert set(base[base.split == "test"]["path"]) == set(plus[plus.split == "test"]["path"])
