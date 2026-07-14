"""Adversarial-training adapters.

Each adapter wires discriminator module(s) to an :class:`AdversarialLoss` and exposes a
uniform ``trains_d`` / ``loss_d`` / ``loss_g`` interface to the Trainer. Being
``nn.Module`` subclasses, ``.to()`` / ``.parameters()`` / ``.state_dict()`` come for free
— replacing the legacy ``eval()``-based ``GANFactory`` and its dummy-parameter hack.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from deblurgan.config import DiscriminatorConfig, LossConfig
from deblurgan.models import build_discriminator
from deblurgan.models.losses import AdversarialLoss, build_adversarial_loss


class AdversarialTrainer(nn.Module):
    trains_d: bool = True

    def loss_d(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss_g(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NoGAN(AdversarialTrainer):
    trains_d = False

    def loss_d(self, fake, real):
        return torch.zeros((), device=fake.device)

    def loss_g(self, fake, real):
        return torch.zeros((), device=fake.device)


class SingleGAN(AdversarialTrainer):
    def __init__(self, net_d: nn.Module, criterion: AdversarialLoss) -> None:
        super().__init__()
        self.net_d = net_d
        self.criterion = criterion

    def loss_d(self, fake, real):
        return self.criterion.loss_d(self.net_d, fake, real)

    def loss_g(self, fake, real):
        return self.criterion.loss_g(self.net_d, fake, real)


class DoubleGAN(AdversarialTrainer):
    """Two-scale critic: average the patch- and full-image adversarial losses."""

    def __init__(self, net_d: nn.Module, criterion: AdversarialLoss) -> None:
        super().__init__()
        self.net_d = net_d
        self.criterion_patch = criterion
        self.criterion_full = copy.deepcopy(criterion)  # independent image pools

    def loss_d(self, fake, real):
        return 0.5 * (
            self.criterion_patch.loss_d(self.net_d.patch, fake, real)
            + self.criterion_full.loss_d(self.net_d.full, fake, real)
        )

    def loss_g(self, fake, real):
        return 0.5 * (
            self.criterion_patch.loss_g(self.net_d.patch, fake, real)
            + self.criterion_full.loss_g(self.net_d.full, fake, real)
        )


def build_adversarial_trainer(
    disc_cfg: DiscriminatorConfig, loss_cfg: LossConfig, device: torch.device
) -> AdversarialTrainer:
    if disc_cfg.kind == "no_gan":
        return NoGAN().to(device)

    net_d = build_discriminator(disc_cfg)
    criterion = build_adversarial_loss(loss_cfg.adversarial)
    if disc_cfg.kind == "patch_gan":
        trainer: AdversarialTrainer = SingleGAN(net_d, criterion)
    elif disc_cfg.kind == "double_gan":
        trainer = DoubleGAN(net_d, criterion)
    else:
        raise ValueError(f"Unknown discriminator kind {disc_cfg.kind!r}")
    return trainer.to(device)
