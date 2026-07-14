"""Device resolution helpers."""

from __future__ import annotations

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device spec to a concrete ``torch.device``.

    ``"auto"`` picks cuda, then mps, then cpu. Explicit specs (``"cpu"``, ``"cuda"``,
    ``"cuda:1"``, ``"mps"``) are validated against availability with a clear error.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {spec!r} but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"Requested device {spec!r} but MPS is not available")
    return device


def amp_supported(device: torch.device) -> bool:
    """Whether autocast + GradScaler mixed precision is supported on ``device``."""
    return device.type == "cuda"
