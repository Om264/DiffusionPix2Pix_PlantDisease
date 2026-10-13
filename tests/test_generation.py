import numpy as np
import pandas as pd
import pytest
import torch

from dataio.metadata import load_metadata
from generation.batch_generate import build_source_pools, compute_class_quotas, make_jobs, severity_grid
from generation.generate import run_generation
from generation.sampling import ddim_sample
from common.config import load_config
from common.errors import UserError
from training.checkpoint import load_model_from_checkpoint


def test_quotas_sum_and_modes():
    counts = {"a": 300, "b": 80, "c": 20}
    for mode in ("uniform", "inverse_frequency", "proportional"):
        assert sum(compute_class_quotas(counts, 10000, mode).values()) == 10000
    u = compute_class_quotas(counts, 10000, "uniform")
    assert max(u.values()) - min(u.values()) <= 1
    inv = compute_class_quotas(counts, 10000, "inverse_frequency")
    assert inv["c"] > inv["b"] > inv["a"]                         # minority gets more
    assert compute_class_quotas({f"c{i}": 5 for i in range(40)}, 20000, "uniform") == {f"c{i}": 500 for i in range(40)}
    assert min(compute_class_quotas(counts, 100, "proportional", min_per_class=10).values()) >= 10
    with pytest.raises(UserError):
        compute_class_quotas(counts, 10, "nope")


def test_source_pools_are_train_only_and_species_matched(processed):
    meta = load_metadata(processed)
    pools = build_source_pools(meta, ["Tomato___Late_blight", "Apple___Apple_scab"])
    for c, (p, kind) in pools.items():
        assert (p["split"] == "train").all() and p["is_healthy"].all() and kind == "healthy_same_species"
        assert (p["species"] == c.split("___")[0]).all()          # tomato lesions only on tomato sources
    assert build_source_pools(meta, ["Tomato___Late_blight"], "same_class")["Tomato___Late_blight"][1] == "same_class"


def test_jobs_unique_seeds_ids_and_severity(processed):
    meta = load_metadata(processed)
    g = load_config("configs/generation.yaml")["generation"]
    pools = build_source_pools(meta, sorted(meta["class"].unique()))
    c2i = {c: i for i, c in enumerate(sorted(meta["class"].unique()))}
    q = {c: 40 for c in c2i}
    jobs = make_jobs(q, pools, c2i, g, 42)
    assert len({j.seed for j in jobs}) == len(jobs) == 200 and len({j.image_id for j in jobs}) == 200
    grid = set(np.round(severity_grid(g["severity_min"], g["severity_max"], g["severity_steps"]), 6))
    assert all(round(j.severity, 6) in grid for j in jobs if "healthy" not in j.class_name)
    assert all(j.severity == 0.0 for j in jobs if "healthy" in j.class_name)
    assert len({round(j.severity, 6) for j in jobs if "Late_blight" in j.class_name}) > 3          # severity diversity
    jobs2 = make_jobs(q, pools, c2i, g, 42)
    assert [j.seed for j in jobs] == [j.seed for j in jobs2]                                  # reproducible plan
    assert not ({j.seed for j in jobs} & {j.seed for j in make_jobs(q, pools, c2i, g, 42, id_offset=10_000)})


def test_sampling_seed_determinism_and_background(tiny_ckpt):
    m, ck = load_model_from_checkpoint(tiny_ckpt, torch.device("cpu"))
    torch.manual_seed(0)
    src = torch.rand(4, 3, 32, 32) * 2 - 1
    mask = torch.zeros(4, 1, 32, 32); mask[:, :, 8:24, 8:24] = 1
    cls, sev, kn = torch.tensor([0, 1, 2, 3]), torch.rand(4), torch.ones(4)
    full = ddim_sample(m, src, mask, cls, sev, kn, [11, 22, 33, 44], steps=4, guidance_scale=2.0)
    solo = ddim_sample(m, src[1:2], mask[1:2], cls[1:2], sev[1:2], kn[1:2], [22], steps=4, guidance_scale=2.0)
    assert torch.allclose(full[1:2], solo, atol=1e-4)                       # output depends on its own seed, not on batching
    other = ddim_sample(m, src[1:2], mask[1:2], cls[1:2], sev[1:2], kn[1:2], [23], steps=4, guidance_scale=2.0)
    assert (other - solo).abs().max() > 1e-4                                  # different seed => different image
    assert torch.equal(full[:, :, :4, :4], src[:, :, :4, :4].float())          # composite keeps the background pixel-exact
    half = ddim_sample(m, src, mask, cls, sev, kn, [1, 2, 3, 4], steps=6, strength=0.5)
    assert half.shape == src.shape


def test_end_to_end_generation_and_resume(tiny_ckpt, processed, tmp_path):
    g = load_config("configs/generation.yaml")["generation"]
    g.update(total_images=10, ddim_steps=3, batch_size=4, precision="fp32", balance_mode="inverse_frequency")
    md = run_generation(str(tiny_ckpt), str(tmp_path / "gen"), g, str(processed), "cpu", 7)
    assert len(md) == 10 and md["seed"].is_unique and md["image_id"].is_unique and (md["source_split"] == "train").all()
    assert set(md.columns) >= {"image_id", "source_image", "class", "disease", "severity", "seed", "mask", "checkpoint", "timestamp", "generation_parameters"}
    assert all((tmp_path / "gen" / p).exists() for p in md["path"])
    assert md["severity_kind"].str.contains("NOT a measured").any()
    md2 = run_generation(str(tiny_ckpt), str(tmp_path / "gen"), g, str(processed), "cpu", 7, resume=True)
    assert len(md2) == 10                                                      # resume: nothing regenerated
