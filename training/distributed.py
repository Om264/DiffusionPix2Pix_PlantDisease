"""Optional multi-GPU helpers. Single-GPU / CPU training is the default and needs none of this.

NOTE: the DDP path (``torchrun --nproc_per_node=N -m training.train ...``) is provided for convenience and has NOT
been tested in this repository; the single-process path has.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_main_process() -> bool:
    return rank() == 0


def maybe_init_distributed() -> int:
    """Initialise the process group if launched with torchrun; returns the local rank (0 if not distributed)."""
    if world_size() > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        local = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        return local
    return 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
