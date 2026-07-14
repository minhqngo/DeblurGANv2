"""Tensor <-> image helpers and per-sample metric computation for logging."""

from __future__ import annotations

import numpy as np
import torch

from deblurgan.metrics import psnr, ssim


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """A single CHW tensor in [-1, 1] -> HWC uint8 RGB image."""
    arr = t.detach().cpu().float().numpy()
    arr = (np.transpose(arr, (1, 2, 0)) + 1.0) / 2.0 * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def sample_metrics_and_image(
    blur: torch.Tensor, fake: torch.Tensor, sharp: torch.Tensor
) -> tuple[float, float, np.ndarray]:
    """PSNR/SSIM of the first batch element, plus a [blur | fake | sharp] strip."""
    b = tensor_to_uint8(blur[0])
    f = tensor_to_uint8(fake[0])
    s = tensor_to_uint8(sharp[0])
    return psnr(f, s), ssim(f, s), np.concatenate([b, f, s], axis=1)
