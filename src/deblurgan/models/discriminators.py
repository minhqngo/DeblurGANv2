"""PatchGAN discriminators.

- ``NLayerDiscriminator``: the standard 70x70 PatchGAN (math unchanged from the legacy
  ``networks.py``; the dead ``use_parallel`` / ``use_sigmoid`` args are dropped).
- ``DoubleScaleDiscriminator``: the paper's two-scale critic (a local patch critic plus
  a deeper full-image critic), as a single ``nn.Module`` instead of the old
  ``{'patch': ..., 'full': ...}`` dict — so ``.to(device)`` / ``.parameters()`` /
  ``.state_dict()`` are single calls.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        kw = 4
        padw = 1
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DoubleScaleDiscriminator(nn.Module):
    """Local patch critic + deeper full-image critic; ``forward`` returns ``(patch, full)``."""

    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        patch_layers: int = 3,
        full_layers: int = 5,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.patch = NLayerDiscriminator(in_channels, ndf, patch_layers, norm_layer)
        self.full = NLayerDiscriminator(in_channels, ndf, full_layers, norm_layer)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.patch(x), self.full(x)
