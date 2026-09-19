"""Device selection + seeding shared by every training/eval script.

    python -m src.device          # print what would be used on this machine

The same code runs on the CPU laptop, an Apple-silicon (MPS) machine and the CUDA laptop;
nothing else in the repo branches on hardware. `--device` on train.py accepts
auto | cpu | cuda | mps. `--deterministic` forces CPU + torch.use_deterministic_algorithms.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def get_device(name: str = "auto", deterministic: bool = False) -> torch.device:
    if deterministic:
        # Bit-exact reproducibility is only promised on CPU; do not chase it on GPUs.
        torch.use_deterministic_algorithms(True)
        return torch.device("cpu")
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False. "
                           "Install with requirements-cuda.txt and check `nvidia-smi`.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is not available.")
    return torch.device(name)


def describe_device(dev: torch.device) -> str:
    if dev.type == "cuda":
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        return f"cuda:{i} {p.name} ({p.total_memory / 2**30:.1f} GiB), torch {torch.__version__}, CUDA {torch.version.cuda}"
    if dev.type == "mps":
        return f"mps (Apple silicon), torch {torch.__version__}"
    return f"cpu ({os.cpu_count()} threads), torch {torch.__version__}"


def seed_everything(seed: int) -> torch.Generator:
    """Seed python / numpy / torch (CPU + all CUDA devices) and return a torch.Generator for the
    DataLoader so shuffling is reproducible independently of model init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


if __name__ == "__main__":
    dev = get_device()
    print("selected:", describe_device(dev))
    print("cuda available:", torch.cuda.is_available(),
          "| mps available:", bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
    x = torch.randn(256, 256, device=dev)
    print("smoke matmul ok on", dev, "->", (x @ x).shape)
