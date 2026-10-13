import torch
import pytest

from common.errors import UserError
from models.diffusion_pix2pix import DiffusionPix2Pix, ModelConfig
from models.scheduler import NoiseScheduler
from training.checkpoint import EMA, load_model_from_checkpoint, resolve_checkpoint


def make_batch(B=3, res=32):
    return dict(target=torch.rand(B, 3, res, res) * 2 - 1, source=torch.rand(B, 3, res, res) * 2 - 1,
                mask=(torch.rand(B, 1, res, res) > .5).float(), class_id=torch.tensor([0, 1, 2][:B]),
                severity=torch.rand(B), sev_known=torch.tensor([1., 0., 1.][:B]))


@pytest.mark.parametrize("pt", ["epsilon", "v_prediction"])
def test_scheduler_roundtrip(pt):
    s = NoiseScheduler(1000, "cosine", prediction_type=pt)
    x0 = torch.rand(4, 3, 8, 8) * 2 - 1; n = torch.randn_like(x0); t = torch.tensor([0, 10, 500, 900])
    xt = s.add_noise(x0, n, t)
    x0r, er = s.predict_x0_eps(s.get_target(x0, n, t), xt, t)
    assert torch.allclose(x0r, x0, atol=1e-3) and torch.allclose(er, n, atol=1e-3)


def test_ddim_strength_shortens_schedule():
    s = NoiseScheduler(1000)
    assert len(s.ddim_timesteps(20, 0.5)) < len(s.ddim_timesteps(20, 1.0)) and s.ddim_timesteps(20)[0] == 999


@pytest.mark.parametrize("latent", [False, True])
def test_forward_backward_and_cfg(latent):
    m = DiffusionPix2Pix(ModelConfig(resolution=32, preset="tiny", latent=latent, vae_random_init=latent), 3)
    b = make_batch()
    m.train(); loss = m.training_loss(b, cond_dropout=0.5, sev_dropout=0.5, min_snr_gamma=5.0)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.vae is None or all(not p.requires_grad for p in m.vae.parameters())
    m.eval()
    with torch.no_grad():
        z = m.encode(b["source"]); mf = m.mask_features(b["mask"], z)
        t = torch.tensor([5, 5, 5])
        o1 = m.predict(torch.randn_like(z), t, z, mf, b["class_id"], b["severity"], b["sev_known"], 1.0)
        o2 = m.predict(torch.randn_like(z), t, z, mf, b["class_id"], b["severity"], b["sev_known"], torch.tensor([1., 2., 3.]))
    assert o1.shape == z.shape == o2.shape


def test_conditioning_matters():
    """Different disease ids must change the prediction (conditioning is wired up)."""
    torch.manual_seed(0)
    m = DiffusionPix2Pix(ModelConfig(resolution=32, preset="tiny"), 3).eval()
    b = make_batch(); x = torch.randn(3, 3, 32, 32); t = torch.tensor([50] * 3)
    with torch.no_grad():
        # perturb the embedding table so that the effect is visible even for a random init
        m.cond.disease.emb.weight.normal_(0, 1)
        a = m.predict(x, t, b["source"], b["mask"], torch.tensor([0, 0, 0]), b["severity"], b["sev_known"])
        c = m.predict(x, t, b["source"], b["mask"], torch.tensor([1, 1, 1]), b["severity"], b["sev_known"])
    assert (a - c).abs().max() > 1e-6


def test_pretrained_unet_path(tmp_path):
    from diffusers import UNet2DConditionModel
    u = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                             up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), block_out_channels=(16, 32), layers_per_block=1,
                             cross_attention_dim=32, attention_head_dim=4, norm_num_groups=8)
    u.save_pretrained(tmp_path / "unet")
    m = DiffusionPix2Pix(ModelConfig(resolution=32, preset="tiny", latent=True, vae_random_init=True, unet_pretrained=str(tmp_path / "unet")), 3)
    assert m.unet.conv_in.in_channels == 9 and not m.use_global
    w = m.unet.conv_in.weight
    assert torch.allclose(w[:, 4:], torch.zeros_like(w[:, 4:]))  # new input channels start at zero (InstructPix2Pix-style)
    m.train(); m.training_loss(make_batch()).backward()


def test_latent_requires_vae():
    with pytest.raises(UserError, match="vae_path"):
        DiffusionPix2Pix(ModelConfig(resolution=32, preset="tiny", latent=True), 3)


def test_unknown_config_key():
    with pytest.raises(UserError, match="Unknown model config key"):
        ModelConfig.from_dict({"presett": "tiny"})


def test_ema_and_checkpoint_loading(tiny_ckpt):
    assert resolve_checkpoint(tiny_ckpt.with_suffix("")) == tiny_ckpt and resolve_checkpoint(tiny_ckpt.parent) == tiny_ckpt
    m, ck = load_model_from_checkpoint(tiny_ckpt, torch.device("cpu"))
    ema = EMA(m, 0.9); before = {k: v.clone() for k, v in ema.shadow.items()}
    for p in m.parameters():
        if p.requires_grad:
            p.data.add_(1.0)
    ema.update(m, 100)
    k = next(iter(ema.shadow)); assert not torch.equal(before[k], ema.shadow[k])
    with ema.applied(m):
        assert torch.allclose(dict(m.named_parameters())[k], ema.shadow[k])
    assert not torch.allclose(dict(m.named_parameters())[k], ema.shadow[k])
    with pytest.raises(UserError, match="Checkpoint not found"):
        resolve_checkpoint("/nonexistent/best")


def test_train_and_resume(processed, tmp_path):
    from training.train import main as train_main
    out = tmp_path / "run"
    base = ["--config", "configs/smoke.yaml", "--set", f"paths.processed_dir={processed}", f"paths.output_dir={out}",
            "data.resolution=32", "train.batch_size=4", "train.val_every=2", "train.save_every=2", "train.log_every=1", "train.val_batches=1"]
    train_main(base + ["train.max_steps=2"])
    assert (out / "checkpoints" / "latest.pt").exists() and (out / "checkpoints" / "best.pt").exists()
    train_main(base[:2] + ["--resume"] + base[2:] + ["train.max_steps=4"])
    import torch as T
    assert T.load(out / "checkpoints" / "latest.pt", weights_only=False)["step"] == 4
    hist = (out / "metrics" / "training_history.csv").read_text().strip().splitlines()
    assert hist[0].startswith("step") and len(hist) > 4
