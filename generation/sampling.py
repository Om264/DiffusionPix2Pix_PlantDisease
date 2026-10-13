"""DDIM sampling for DiffusionPix2Pix with classifier-free guidance, SDEdit-style ``strength`` and
background-preserving compositing."""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F

from models.diffusion_pix2pix import DiffusionPix2Pix


def gaussian_blur_mask(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a (B,1,H,W) mask (used to feather the composite seam)."""
    if sigma <= 0:
        return mask
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, device=mask.device, dtype=mask.dtype)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    m = F.pad(mask, (r, r, r, r), mode="replicate")
    m = F.conv2d(m, k.view(1, 1, 1, -1))
    return F.conv2d(m, k.view(1, 1, -1, 1))


def noise_from_seeds(shape, seeds: Sequence[int], device, dtype) -> torch.Tensor:
    """One CPU generator per image -> the initial noise of an image depends only on its own seed."""
    out = []
    for s in seeds:
        g = torch.Generator(device="cpu").manual_seed(int(s))
        out.append(torch.randn(shape, generator=g))
    return torch.stack(out).to(device=device, dtype=dtype)


@torch.no_grad()
def ddim_sample(model: DiffusionPix2Pix, source: torch.Tensor, mask: torch.Tensor, class_ids: torch.Tensor,
                severity: torch.Tensor, sev_known: torch.Tensor, seeds: Sequence[int], steps: int = 50,
                guidance_scale: Union[float, torch.Tensor] = 1.0, strength: float = 1.0, eta: float = 0.0, composite: bool = True,
                feather: float = 0.0, amp_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Generate B images in [-1,1], returned on CPU.

    source (B,3,H,W) in [-1,1]; mask (B,1,H,W) {0,1} = region where the model may synthesise disease.
    ``strength`` < 1 starts from the noised *source* (more faithful to the source, less change)."""
    model.eval()
    dev = source.device
    z_src = model.encode(source)
    mf = model.mask_features(mask, z_src)
    noise = noise_from_seeds(z_src.shape[1:], seeds, dev, z_src.dtype)
    ts = model.scheduler.ddim_timesteps(steps, strength).tolist()
    if strength < 1.0:
        t0 = torch.full((z_src.shape[0],), ts[0], device=dev, dtype=torch.long)
        x = model.scheduler.add_noise(z_src, noise, t0)
    else:
        x = noise
    gen = torch.Generator(device="cpu").manual_seed(int(seeds[0]) + 17) if eta > 0 else None
    x0_clip = 1.0 if model.vae is None else None
    for i, t in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        tb = torch.full((x.shape[0],), t, device=dev, dtype=torch.long)
        if amp_dtype is not None:
            with torch.autocast(device_type=dev.type, dtype=amp_dtype):
                out = model.predict(x, tb, z_src, mf, class_ids, severity, sev_known, guidance_scale)
        else:
            out = model.predict(x, tb, z_src, mf, class_ids, severity, sev_known, guidance_scale)
        x = model.scheduler.ddim_step(out.float(), t, t_prev, x.float(), eta, x0_clip, gen)
    img = model.decode(x.to(z_src.dtype) if model.vae is not None else x).clamp(-1, 1)
    if composite:
        w = gaussian_blur_mask(mask.to(img.dtype), feather).clamp(0, 1)
        img = w * img + (1 - w) * source
    return img.float().cpu()
