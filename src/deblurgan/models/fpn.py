"""Backbone-agnostic FPN generator for DeblurGAN-v2.

One generator works with any hierarchical timm backbone: the five feature maps
(strides 2/4/8/16/32) are tapped via timm's ``features_only`` API and their channel
counts are read from ``feature_info`` — no per-backbone hand-wiring. This replaces the
four near-identical copy-pasted FPN files in the legacy ``models/`` directory.
"""

from __future__ import annotations

from typing import Callable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from deblurgan.config import GeneratorConfig
from deblurgan.models.norm import get_norm_layer

REQUIRED_REDUCTIONS = (2, 4, 8, 16, 32)


def _create_backbone(name: str, pretrained: bool) -> tuple[nn.Module, list[int], list[int]]:
    """Create a timm feature backbone and locate the taps at strides 2,4,8,16,32.

    Returns ``(backbone, tap_indices, tap_channels)``. Raises ``ValueError`` for
    backbones (ViT/Swin-style) that cannot supply all five reduction levels.
    """
    backbone = timm.create_model(name, features_only=True, pretrained=pretrained)
    reductions = list(backbone.feature_info.reduction())
    channels = list(backbone.feature_info.channels())
    try:
        tap_indices = [reductions.index(r) for r in REQUIRED_REDUCTIONS]
    except ValueError as exc:
        raise ValueError(
            f"Backbone {name!r} exposes feature strides {reductions}, but the FPN "
            f"generator needs all of {list(REQUIRED_REDUCTIONS)}. Use a hierarchical CNN "
            f"backbone (e.g. resnet50, mobilenetv2_100, efficientnet_b0, densenet121, "
            f"inception_resnet_v2); ViT/Swin-style backbones are not supported."
        ) from exc
    tap_channels = [channels[i] for i in tap_indices]
    return backbone, tap_indices, tap_channels


class FPNHead(nn.Module):
    """Two 3x3 convs with ReLU, applied to each pyramid level before merging."""

    def __init__(self, num_in: int, num_mid: int, num_out: int) -> None:
        super().__init__()
        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.block0(x), inplace=True)
        x = F.relu(self.block1(x), inplace=True)
        return x


def _upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Nearest-upsample ``x`` to ``ref``'s spatial size (robust to odd backbone rounding)."""
    return F.interpolate(x, size=ref.shape[-2:], mode="nearest")


class TimmFPN(nn.Module):
    """timm backbone + lateral 1x1 convs + top-down pathway.

    ``forward`` returns ``(lateral0, map1, map2, map3, map4)``: ``lateral0`` at stride 2
    with ``head_channels``; ``map1..map4`` at strides 4/8/16/32 with ``fpn_channels``.
    """

    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        fpn_channels: int,
        head_channels: int,
        norm_layer: Callable[[int], nn.Module],
        td_blocks: bool,
    ) -> None:
        super().__init__()
        self.backbone, self.tap_indices, ch = _create_backbone(backbone, pretrained)
        c0, c1, c2, c3, c4 = ch

        self.lateral4 = nn.Conv2d(c4, fpn_channels, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(c3, fpn_channels, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(c2, fpn_channels, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(c1, fpn_channels, kernel_size=1, bias=False)
        # lateral0 matches the decoder's smooth output so they can be summed.
        self.lateral0 = nn.Conv2d(c0, head_channels, kernel_size=1, bias=False)

        def make_td() -> nn.Module:
            if not td_blocks:
                return nn.Identity()
            return nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
                norm_layer(fpn_channels),
                nn.ReLU(inplace=True),
            )

        self.td1 = make_td()
        self.td2 = make_td()
        self.td3 = make_td()

        self.freeze()

    def freeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        enc0, enc1, enc2, enc3, enc4 = (feats[i] for i in self.tap_indices)

        lateral0 = self.lateral0(enc0)
        map4 = self.lateral4(enc4)
        map3 = self.td1(self.lateral3(enc3) + _upsample_to(map4, enc3))
        map2 = self.td2(self.lateral2(enc2) + _upsample_to(map3, enc2))
        map1 = self.td3(self.lateral1(enc1) + _upsample_to(map2, enc1))
        return lateral0, map1, map2, map3, map4


class FPNGenerator(nn.Module):
    """DeblurGAN-v2 generator. Input and output are RGB tensors normalized to [-1, 1].

    Contract: callers pad H and W to a multiple of 32 (the Predictor does this);
    the network itself tolerates other sizes via size-targeted interpolation.
    """

    def __init__(self, cfg: GeneratorConfig) -> None:
        super().__init__()
        norm_layer = get_norm_layer(cfg.norm)
        nf = cfg.fpn_channels
        nh = cfg.head_channels
        mid = max(1, nh // 2)

        self.fpn = TimmFPN(
            backbone=cfg.backbone,
            pretrained=cfg.pretrained,
            fpn_channels=nf,
            head_channels=nh,
            norm_layer=norm_layer,
            td_blocks=cfg.td_blocks,
        )

        self.head1 = FPNHead(nf, nh, nh)
        self.head2 = FPNHead(nf, nh, nh)
        self.head3 = FPNHead(nf, nh, nh)
        self.head4 = FPNHead(nf, nh, nh)

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * nh, nh, kernel_size=3, padding=1),
            norm_layer(nh),
            nn.ReLU(inplace=True),
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(nh, mid, kernel_size=3, padding=1),
            norm_layer(mid),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(mid, cfg.out_channels, kernel_size=3, padding=1)

    def freeze(self) -> None:
        self.fpn.freeze()

    def unfreeze(self) -> None:
        self.fpn.unfreeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lateral0, map1, map2, map3, map4 = self.fpn(x)

        # Bring every head output onto map1's (stride-4) grid, then merge.
        h4 = _upsample_to(self.head4(map4), map1)
        h3 = _upsample_to(self.head3(map3), map1)
        h2 = _upsample_to(self.head2(map2), map1)
        h1 = self.head1(map1)

        smoothed = self.smooth(torch.cat([h4, h3, h2, h1], dim=1))
        smoothed = _upsample_to(smoothed, lateral0)
        smoothed = self.smooth2(smoothed + lateral0)
        smoothed = F.interpolate(smoothed, size=x.shape[-2:], mode="nearest")

        out = torch.tanh(self.final(smoothed)) + x  # global residual
        return torch.clamp(out, min=-1.0, max=1.0)
