"""DiffusionPix2Pix-style conditional diffusion model (research-oriented implementation).

Research-oriented implementation inspired by the published DiffusionPix2Pix methodology
(Computers and Electronics in Agriculture 229, 109690). NOT the authors' code.

Data flow (pixel space; in latent mode every image tensor is first mapped through a frozen VAE):

   x_t (noisy target) ┐
   source image prior ├─ channel-concat ─► UNet2DConditionModel ─► eps / v
   binary edit mask   ┘            ▲  ▲
                                   │  └── cross-attention tokens: [disease, severity, image-summary tokens]
                                   └───── timestep embedding (+ global disease/severity vector)
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.errors import UserError
from models.conditioning import ConditionEncoder
from models.losses import diffusion_loss
from models.scheduler import NoiseScheduler

log = logging.getLogger(__name__)

PRESETS: Dict[str, Dict[str, Any]] = {
    # attention is only used at low resolution levels to keep memory manageable
    "tiny": dict(block_out_channels=[32, 64], down=["DownBlock2D", "CrossAttnDownBlock2D"],
                 up=["CrossAttnUpBlock2D", "UpBlock2D"], norm_num_groups=8, token_dim=64, layers_per_block=1),
    "small": dict(block_out_channels=[64, 128, 256, 256],
                  down=["DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
                  up=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"],
                  norm_num_groups=32, token_dim=256, layers_per_block=2),
    "base": dict(block_out_channels=[96, 192, 384, 384],
                 down=["DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
                 up=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"],
                 norm_num_groups=32, token_dim=384, layers_per_block=2),
}


@dataclass
class ModelConfig:
    resolution: int = 256
    preset: str = "small"
    latent: bool = False                  # False: pixel-space diffusion; True: latent diffusion with a frozen VAE
    vae_path: Optional[str] = None        # diffusers AutoencoderKL directory / hub id (latent mode)
    vae_random_init: bool = False         # tests only: tiny randomly initialised VAE
    unet_pretrained: Optional[str] = None  # diffusers UNet2DConditionModel dir/hub id; conv_in is expanded (zero-init)
    attention_head_dim: int = 8
    image_grid: int = 4
    num_train_timesteps: int = 1000
    beta_schedule: str = "cosine"
    prediction_type: str = "epsilon"
    gradient_checkpointing: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        extra = set(d) - known
        if extra:
            raise UserError(f"Unknown model config key(s): {sorted(extra)}. Valid keys: {sorted(known)}")
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DiffusionPix2Pix(nn.Module):
    def __init__(self, cfg: ModelConfig, num_classes: int):
        super().__init__()
        from diffusers import UNet2DConditionModel
        if cfg.preset not in PRESETS:
            raise UserError(f"Unknown model.preset '{cfg.preset}'. Choose from {sorted(PRESETS)}.")
        self.cfg, self.num_classes = cfg, num_classes
        p = PRESETS[cfg.preset]

        # ---- optional frozen VAE (latent mode) ------------------------------------------------------------------
        self.vae = None
        z_ch, self.scaling = 3, 1.0
        if cfg.latent:
            from diffusers import AutoencoderKL
            if cfg.vae_path:
                self.vae = AutoencoderKL.from_pretrained(cfg.vae_path)
            elif cfg.vae_random_init:
                self.vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(16, 32),
                                         down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2,
                                         layers_per_block=1, norm_num_groups=8, sample_size=cfg.resolution)
            else:
                raise UserError("model.latent=true needs model.vae_path (a diffusers AutoencoderKL, e.g. "
                                "'stabilityai/sd-vae-ft-mse'); download it first or set latent=false.")
            self.vae.requires_grad_(False).eval()
            z_ch = self.vae.config.latent_channels
            self.scaling = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        self.z_ch = z_ch
        in_ch = 2 * z_ch + 1  # [x_t, source, mask]

        # ---- UNet -----------------------------------------------------------------------------------------------
        if cfg.unet_pretrained:
            if not cfg.latent:
                raise UserError("model.unet_pretrained requires model.latent=true (SD-style UNets work on VAE latents).")
            self.unet = UNet2DConditionModel.from_pretrained(cfg.unet_pretrained)
            self._expand_conv_in(in_ch)
            token_dim, glob_dim = int(self.unet.config.cross_attention_dim), None  # no class embedding in SD UNets
            if self.unet.config.out_channels != z_ch:
                raise UserError("Pretrained UNet out_channels does not match the VAE latent channels.")
        else:
            token_dim = p["token_dim"]
            glob_dim = token_dim * 2
            self.unet = UNet2DConditionModel(
                sample_size=cfg.resolution // (8 if cfg.latent else 1), in_channels=in_ch, out_channels=z_ch,
                down_block_types=tuple(p["down"]), up_block_types=tuple(p["up"]),
                block_out_channels=tuple(p["block_out_channels"]), layers_per_block=p["layers_per_block"],
                cross_attention_dim=token_dim, attention_head_dim=cfg.attention_head_dim,
                norm_num_groups=p["norm_num_groups"], class_embed_type="projection",
                projection_class_embeddings_input_dim=glob_dim)
        self.use_global = glob_dim is not None
        if cfg.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        self.cond = ConditionEncoder(num_classes, token_dim, glob_dim, z_ch + 0, cfg.image_grid)
        self.scheduler = NoiseScheduler(cfg.num_train_timesteps, cfg.beta_schedule, prediction_type=cfg.prediction_type)

    # ------------------------------------------------------------------------------------------------------------
    def _expand_conv_in(self, new_in: int) -> None:
        old = self.unet.conv_in
        if old.in_channels == new_in:
            return
        conv = nn.Conv2d(new_in, old.out_channels, old.kernel_size, old.stride, old.padding)
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[:, :old.in_channels] = old.weight
            conv.bias.copy_(old.bias)
        self.unet.conv_in = conv
        self.unet.register_to_config(in_channels=new_in)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vae is not None:
            self.vae.eval()
        return self

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """State dict without a pretrained (re-loadable) VAE, to keep checkpoints small."""
        sd = self.state_dict()
        if self.vae is not None and self.cfg.vae_path:
            sd = {k: v for k, v in sd.items() if not k.startswith("vae.")}
        return sd

    # ------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Image in [-1,1] -> diffusion space (identity in pixel mode)."""
        if self.vae is None:
            return x
        dist = self.vae.encode(x).latent_dist
        return (dist.sample() if sample else dist.mode()) * self.scaling

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return z
        return self.vae.decode(z / self.scaling).sample.clamp(-1, 1)

    @staticmethod
    def mask_features(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return mask if mask.shape[-2:] == like.shape[-2:] else F.interpolate(mask.float(), size=like.shape[-2:], mode="nearest")

    def _unet(self, x_t, t, src_f, mask_f, tokens, glob):
        out = self.unet(torch.cat([x_t, src_f, mask_f.to(x_t.dtype)], 1), t, encoder_hidden_states=tokens,
                        class_labels=glob if self.use_global else None).sample
        return out

    # ------------------------------------------------------------------------------------------------------------
    def training_loss(self, batch: Dict[str, torch.Tensor], cond_dropout: float = 0.1, sev_dropout: float = 0.1,
                      edit_weight: float = 1.0, bg_weight: float = 0.5, min_snr_gamma: Optional[float] = None,
                      generator: Optional[torch.Generator] = None, fixed_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        tgt, src, mask = batch["target"], batch["source"], batch["mask"]
        z0, zs = self.encode(tgt, sample=generator is None), self.encode(src)
        mf = self.mask_features(mask, z0)
        B = z0.shape[0]
        if fixed_t is not None:
            t = fixed_t.to(z0.device)
        else:
            t = torch.randint(0, self.scheduler.T, (B,), device=z0.device)
        noise = torch.randn(z0.shape, generator=generator, device="cpu").to(z0.device, z0.dtype) if generator is not None \
            else torch.randn_like(z0)
        x_t = self.scheduler.add_noise(z0, noise, t)
        if self.training and (cond_dropout > 0 or sev_dropout > 0):
            drop_c = torch.rand(B, device=z0.device) < cond_dropout
            drop_s = torch.rand(B, device=z0.device) < sev_dropout
        else:
            drop_c = drop_s = None
        tokens, glob = self.cond(zs, mf, batch["class_id"], batch["severity"], batch["sev_known"], drop_c, drop_s)
        pred = self._unet(x_t, t, zs, mf, tokens, glob)
        target = self.scheduler.get_target(z0, noise, t)
        return diffusion_loss(pred, target, mask, t, self.scheduler, edit_weight, bg_weight, min_snr_gamma)

    def predict(self, x_t: torch.Tensor, t: torch.Tensor, src_f: torch.Tensor, mask_f: torch.Tensor,
                class_ids: torch.Tensor, severity: torch.Tensor, sev_known: torch.Tensor,
                guidance_scale: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
        """Model output (eps or v). With ``guidance_scale`` != 1 applies classifier-free guidance on the
        disease/severity condition (source image and mask are always kept). ``guidance_scale`` may be a float or a
        per-sample tensor (B,)."""
        tokens, glob = self.cond(src_f, mask_f, class_ids, severity, sev_known)
        g = torch.as_tensor(guidance_scale, device=x_t.device, dtype=x_t.dtype).reshape(-1, 1, 1, 1)
        if bool((g == 1.0).all()):
            return self._unet(x_t, t, src_f, mask_f, tokens, glob)
        B = x_t.shape[0]
        drop = torch.ones(B, dtype=torch.bool, device=x_t.device)
        tok_u, glob_u = self.cond(src_f, mask_f, class_ids, severity, sev_known, drop_class=drop)
        out = self._unet(torch.cat([x_t, x_t]), torch.cat([t, t]) if t.ndim else t, torch.cat([src_f, src_f]),
                         torch.cat([mask_f, mask_f]), torch.cat([tok_u, tokens]),
                         torch.cat([glob_u, glob]) if glob is not None else None)
        o_u, o_c = out.chunk(2)
        return o_u + g * (o_c - o_u)
