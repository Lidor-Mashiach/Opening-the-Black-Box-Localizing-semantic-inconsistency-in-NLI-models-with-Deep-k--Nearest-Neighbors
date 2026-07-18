"""GPU detection - ONE place, loud and actionable.

`torch.cuda.is_available()` is the correct probe, but it only answers yes/no.
When the answer is no, this module also explains WHY - and the two real-world
cases are completely different:

  * `torch.version.cuda is None` -> the installed torch wheel is CPU-ONLY
    (the PyPI default on **Windows**). No detection logic can fix that - the
    wheel itself must be replaced; the exact command is printed.
  * torch WAS built for CUDA but no GPU is visible -> a driver / allocation
    problem (locally: check `nvidia-smi`; on SLURM: did the job request
    `--gpus=1`?).

Optional fail-fast: export NLI_REQUIRE_GPU=1 and a missing GPU raises
immediately instead of silently crawling on CPU. Off by default - the
sbatch jobs request a GPU explicitly, so this is only a belt-and-braces
switch for unattended runs.

Used by EVERY GPU entry point: training / fine-tuning / encoding / inference
(via model_utils.get_device), Optuna tuning, the paraphrase-bank generator,
and the main runner's banner.
"""
import os

import torch

from .logging_utils import log

REINSTALL_HINT = ("run `python setup_env.py` from the repo root - it discovers "
                  "PyTorch's LIVE CUDA wheel indexes, picks the newest one your "
                  "driver supports, and repairs the torch build automatically "
                  "(manual fallback: choose a live cuXXX index yourself from "
                  "https://download.pytorch.org/whl/ )")


def cuda_status():
    """('cuda' | 'cpu', human-readable explanation). Never raises."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024 ** 3
        return "cuda", (f"{torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB, "
                        f"CUDA {torch.version.cuda})")
    if torch.version.cuda is None:
        return "cpu", ("this torch build is CPU-ONLY - no CUDA support compiled "
                       "in (the PyPI default on Windows). Fix: " + REINSTALL_HINT)
    return "cpu", (f"torch was built for CUDA {torch.version.cuda} but no usable "
                   f"GPU is visible - check `nvidia-smi` and the NVIDIA driver; "
                   f"on SLURM, make sure the job requested --gpus=1")


def resolve_device(context="compute"):
    """Detect once, log loudly, honor NLI_REQUIRE_GPU=1, return torch.device."""
    kind, info = cuda_status()
    if kind == "cuda":
        log("DEVICE", f"{context}: cuda - {info}")
    else:
        log("ERROR", f"{context}: no CUDA GPU - {info}")
        if os.environ.get("NLI_REQUIRE_GPU") == "1":
            raise RuntimeError("NLI_REQUIRE_GPU=1 and no CUDA GPU is available - "
                               "aborting instead of running on CPU. " + info)
        log("ERROR", f"{context}: continuing on CPU (very slow). "
            f"Set NLI_REQUIRE_GPU=1 to fail fast instead.")
    return torch.device(kind)
