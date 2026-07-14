"""Single-image deblurring inference.

The generator architecture is rebuilt from the config embedded in the checkpoint, so no
separate YAML file is needed at inference time. Inputs and outputs are uint8 RGB arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from deblurgan.devices import resolve_device
from deblurgan.models import build_generator
from deblurgan.training.checkpoint import generator_config_from_checkpoint, load_checkpoint

PAD_TO = 32


def _pad_to_multiple(x: torch.Tensor, multiple: int = PAD_TO) -> tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h or pad_w:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, h, w


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str | torch.device = "auto") -> None:
        self.device = device if isinstance(device, torch.device) else resolve_device(device)
        ckpt = load_checkpoint(checkpoint, map_location=self.device)
        gen_cfg = generator_config_from_checkpoint(ckpt)
        self.model = build_generator(gen_cfg).to(self.device)
        self.model.load_state_dict(ckpt["generator"])
        self.model.eval()

    @torch.no_grad()
    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        """Deblur one uint8 RGB (HWC) image, returning a uint8 RGB image of the same size."""
        x = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float().div(255.0).mul(2.0).sub(1.0)
        x = x.unsqueeze(0).to(self.device)
        x, h, w = _pad_to_multiple(x)
        y = self.model(x)
        y = y[..., :h, :w]
        out = (y.squeeze(0).add(1.0).div(2.0).clamp(0, 1).mul(255.0)).cpu().numpy()
        return out.transpose(1, 2, 0).round().astype(np.uint8)
