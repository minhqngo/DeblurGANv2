"""PSNR / SSIM on uint8 RGB images via scikit-image.

Uses the modern ``channel_axis`` argument (``multichannel=`` was removed in
scikit-image >= 0.19).
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR between two uint8 RGB images (HWC), in dB."""
    return float(peak_signal_noise_ratio(a, b, data_range=255))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM between two uint8 RGB images (HWC)."""
    return float(structural_similarity(a, b, channel_axis=-1, data_range=255))
