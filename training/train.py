"""Train DiffusionPix2Pix (single GPU by default).

    python -m training.train --config configs/training.yaml
    python -m training.train --config configs/training.yaml --resume --set train.max_steps=20000
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.config import load_config, save_config
from common.errors import UserError, run_cli
from common.gpu import amp_dtype as get_amp_dtype
from common.gpu import cuda_memory_stats, get_device, log_environment
from common.imageio_utils import save_rgb, tile_images
from common.logging_utils import setup_logging
from common.seed import seed_everything
from dataio.dataset import PlantDiseaseDataset, severity_allowed
from dataio.metadata import load_class_mapping, load_metadata
from dataio.transforms import from_tensor_m11
from models.diffusion_pix2pix import DiffusionPix2Pix, ModelConfig
from training.checkpoint import EMA, load_checkpoint, resolve_checkpoint, save_checkpoint
from training.distributed import is_main_process, maybe_init_distributed
from training.validate import validate

log = logging.getLogger("train")


def build_model_config(cfg: Dict[str, Any]) -> ModelConfig:
    return ModelConfig.from_dict({**cfg["model"], "resolution": cfg["data"]["resolution"]})


def infinite(loader: DataLoader) -> Iterator:
    epoch = 0
    while True:
        for b in loader:
            yield epoch, b
        epoch += 1


def _preview(model, ema, ds, device, path: Path, n: int = 4, steps: int = 20) -> None:
    from generation.sampling import ddim_sample
    items = [ds[i] for i in range(min(n, len(ds)))]
    b = {k: torch.stack([it[k] for it in items]).to(device) for k in items[0]}
    with ema.applied(model):
        out = ddim_sample(model, b["source"], b["mask"], b["class_id"], b["severity"], b["sev_known"],
                          seeds=list(range(len(items))), steps=steps, guidance_scale=1.0, composite=False)
    rows = [[from_tensor_m11(b["source"][i].cpu()), (b["mask"][i, 0].cpu().numpy() * 255).astype(np.uint8),
             from_tensor_m11(out[i]), from_tensor_m11(b["target"][i].cpu())] for i in range(len(items))]
    save_rgb(tile_images(rows), path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/training.yaml")
    ap.add_argument("--resume", nargs="?", const="auto", default=None, help="'auto' (latest.pt) or a checkpoint path")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.max_steps=100 data.resolution=64")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.set)
    out = Path(cfg["paths"]["output_dir"])
    setup_logging(out / "logs" / "training.log")
    save_config(cfg, out / "config_used_training.yaml")
    seed_everything(cfg["seed"])
    local_rank = maybe_init_distributed()
    device = get_device(cfg["device"])
    log_environment(device)
    tc = cfg["train"]

    # ---------------------------------------------------------------- data
    root = cfg["paths"]["processed_dir"]
    meta, c2i = load_metadata(root), load_class_mapping(root)
    dcfg = {**cfg["data"], "severity_mode": cfg["data"].get("severity_mode", "estimated")}
    tr_meta, va_meta = meta[meta["split"] == "train"], meta[meta["split"] == "val"]
    if va_meta.empty:
        log.warning("No validation split found; validating on a subset of the training images (less reliable).")
        va_meta = tr_meta.sample(min(len(tr_meta), 64), random_state=0)
    if meta["mask_path"].isna().all():
        log.warning("No masks found: the whole image will be the editable region. Run preprocessing.prepare_masks "
                    "so that background pixels are protected.")
    n_sev = int(sum(severity_allowed(str(s), dcfg["severity_mode"]) for s in tr_meta["severity_source"]))
    sev_supported = n_sev > 0
    if not sev_supported:
        log.warning("No disease severity labels were found. Synthetic severity values will be estimated "
                    "(run preprocessing.estimate_severity) - training continues WITHOUT severity conditioning.")
    elif dcfg["severity_mode"] == "estimated" and (tr_meta["severity_source"] == "estimated").any():
        log.warning("Severity conditions are ESTIMATED by a colour heuristic, not ground truth.")
    train_ds = PlantDiseaseDataset(tr_meta, root, c2i, cfg["data"]["resolution"], True, dcfg, cfg["seed"])
    val_ds = PlantDiseaseDataset(va_meta, root, c2i, cfg["data"]["resolution"], False, dcfg, cfg["seed"])
    if len(train_ds) < tc["batch_size"]:
        log.warning("Only %d training images (< batch_size %d): lowering batch size.", len(train_ds), tc["batch_size"])
        tc["batch_size"] = max(1, len(train_ds))
    pin = device.type == "cuda"
    tl = DataLoader(train_ds, tc["batch_size"], shuffle=True, num_workers=tc["num_workers"], pin_memory=pin,
                    drop_last=len(train_ds) > tc["batch_size"], persistent_workers=tc["num_workers"] > 0)
    vl = DataLoader(val_ds, tc["batch_size"], shuffle=False, num_workers=tc["num_workers"], pin_memory=pin)
    log.info("classes=%d  train=%d  val=%d  severity-labelled train images=%d", len(c2i), len(train_ds), len(val_ds), n_sev)

    # ---------------------------------------------------------------- model / optim
    mcfg = build_model_config(cfg)
    model = DiffusionPix2Pix(mcfg, len(c2i)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("trainable parameters: %.2fM (preset=%s, latent=%s)", n_params / 1e6, mcfg.preset, mcfg.latent)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    warm, total = tc["warmup_steps"], tc["max_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    ema = EMA(model, tc["ema_decay"])
    amp = get_amp_dtype(tc["precision"], device)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == torch.float16 and device.type == "cuda"))
    loss_kw = dict(edit_weight=tc["edit_weight"], bg_weight=tc["bg_weight"], min_snr_gamma=tc["min_snr_gamma"])

    ckdir = out / "checkpoints"
    step, best = 0, float("inf")
    hist_path = out / "metrics" / "training_history.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    if a.resume:
        rp = ckdir / "latest.pt" if a.resume == "auto" else Path(a.resume)
        if rp.exists():
            ck = load_checkpoint(resolve_checkpoint(rp), "cpu")
            if ck["class_to_idx"] != c2i:
                raise UserError("The checkpoint was trained on different classes than the current processed dataset.")
            model.load_state_dict(ck["model"], strict=False)
            ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
            if ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            step, best = ck["step"], ck["best_val"]
            log.info("Resumed from %s at step %d (best val %.5f)", rp, step, best)
        else:
            log.warning("--resume given but %s does not exist; starting from scratch.", rp)
    if not (a.resume and step > 0) or not hist_path.exists():
        with open(hist_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "train_loss", "lr", "val_loss", "mem_peak_gb", "elapsed_s"])

    def make_state(with_opt: bool) -> Dict[str, Any]:
        st = {"model": model.trainable_state_dict(), "ema": ema.state_dict(), "step": step, "best_val": best,
              "config": cfg, "model_cfg": mcfg.to_dict(), "class_to_idx": c2i, "severity_supported": sev_supported,
              "severity_mode": dcfg["severity_mode"]}
        if with_opt:
            st.update(optimizer=opt.state_dict(), scheduler=sched.state_dict(), scaler=scaler.state_dict())
        return st

    # ---------------------------------------------------------------- loop
    model.train()
    it, t0, run_loss, run_n, epoch = infinite(tl), time.time(), 0.0, 0, 0
    accum = tc["grad_accum"]
    try:
        while step < total:
            opt.zero_grad(set_to_none=True)
            for _ in range(accum):
                epoch, batch = next(it)
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
                    loss = model.training_loss(batch, cond_dropout=tc["cond_dropout"], sev_dropout=tc["sev_dropout"], **loss_kw)
                if not torch.isfinite(loss):
                    raise UserError("Loss became NaN/Inf. Try a lower learning rate, precision: bf16/fp32, or check the data.")
                scaler.scale(loss / accum).backward()
                run_loss += float(loss.detach()); run_n += 1
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, tc["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            ema.update(model, step)

            val_loss = float("nan")
            if step % tc["log_every"] == 0 or step == total:
                mem = cuda_memory_stats(device)
                tl_ = run_loss / max(run_n, 1)
                log.info("step %d/%d ep %d loss %.4f lr %.2e %s", step, total, epoch, tl_, sched.get_last_lr()[0],
                         f"peak-mem {mem['mem_peak_gb']:.2f}/{mem['mem_total_gb']:.0f} GB" if mem else "")
                with open(hist_path, "a", newline="") as f:
                    csv.writer(f).writerow([step, epoch, tl_, sched.get_last_lr()[0], "", mem.get("mem_peak_gb", ""), time.time() - t0])
                run_loss, run_n = 0.0, 0
            if step % tc["val_every"] == 0 or step == total:
                val_loss = validate(model, vl, device, amp, ema, tc["val_batches"], **loss_kw)
                log.info("VALIDATION step %d  val_loss(EMA) %.5f  (best %.5f)", step, val_loss, best)
                with open(hist_path, "a", newline="") as f:
                    csv.writer(f).writerow([step, epoch, "", "", val_loss, "", time.time() - t0])
                if val_loss < best:
                    best = val_loss
                    save_checkpoint(ckdir / "best.pt", {**make_state(False), "val_loss": val_loss})
                    log.info("New best checkpoint saved (val_loss %.5f)", best)
            if step % tc["save_every"] == 0 or step == total:
                save_checkpoint(ckdir / "latest.pt", make_state(True))
            if tc["sample_every"] and step % tc["sample_every"] == 0:
                _preview(model, ema, val_ds, device, out / "visualizations" / f"train_step_{step:07d}.png")
    except KeyboardInterrupt:
        log.warning("Interrupted: saving latest.pt (resume with --resume).")
        save_checkpoint(ckdir / "latest.pt", make_state(True))
        return
    except torch.cuda.OutOfMemoryError:
        raise UserError("GPU VRAM is insufficient. Reduce data.resolution or train.batch_size (raise train.grad_accum), "
                        "set model.gradient_checkpointing=true, use model.preset=small, or precision fp16/bf16.")
    if not (ckdir / "best.pt").exists():
        save_checkpoint(ckdir / "best.pt", {**make_state(False), "val_loss": float("nan")})
    log.info("Training finished at step %d. Best val_loss %.5f. Checkpoints in %s", step, best, ckdir)


if __name__ == "__main__":
    run_cli(main)
