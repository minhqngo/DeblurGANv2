"""Model factories: build a generator, discriminator, and losses from typed config."""

from __future__ import annotations

import torch.nn as nn

from deblurgan.config import DiscriminatorConfig, GeneratorConfig, LossConfig
from deblurgan.models.discriminators import DoubleScaleDiscriminator, NLayerDiscriminator
from deblurgan.models.fpn import FPNGenerator
from deblurgan.models.losses import AdversarialLoss, build_adversarial_loss, build_content_loss
from deblurgan.models.norm import get_norm_layer

__all__ = [
    "FPNGenerator",
    "NLayerDiscriminator",
    "DoubleScaleDiscriminator",
    "AdversarialLoss",
    "build_generator",
    "build_discriminator",
    "build_losses",
]


def build_generator(cfg: GeneratorConfig) -> FPNGenerator:
    return FPNGenerator(cfg)


def build_discriminator(cfg: DiscriminatorConfig) -> nn.Module | None:
    """Return the discriminator module, or ``None`` for ``kind == 'no_gan'``."""
    if cfg.kind == "no_gan":
        return None
    norm_layer = get_norm_layer(cfg.norm)
    if cfg.kind == "patch_gan":
        return NLayerDiscriminator(n_layers=cfg.num_layers, norm_layer=norm_layer)
    if cfg.kind == "double_gan":
        return DoubleScaleDiscriminator(
            patch_layers=cfg.num_layers, full_layers=5, norm_layer=norm_layer
        )
    raise ValueError(f"Unknown discriminator kind {cfg.kind!r}")


def build_losses(
    cfg: LossConfig, content_pretrained: bool = True
) -> tuple[nn.Module, AdversarialLoss]:
    content = build_content_loss(cfg.content, pretrained=content_pretrained)
    adversarial = build_adversarial_loss(cfg.adversarial)
    return content, adversarial
