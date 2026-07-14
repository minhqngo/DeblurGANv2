"""Content and adversarial losses (device-agnostic; no ``.cuda()``, no ``Variable``).

Content losses:
  * ``ContentLoss``   — L1 or L2 in pixel space.
  * ``PerceptualLoss``— 0.006 * MSE(VGG19 conv3_3 features) + 0.5 * MSE(pixels).

Adversarial losses share the :class:`AdversarialLoss` interface — ``loss_d(net_d, fake,
real)`` and ``loss_g(net_d, fake, real)`` — where ``net_d`` is any callable returning a
score map. The relativistic least-squares variant (``ragan-ls``) is the paper default.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from deblurgan.models.registry import Registry

CONTENT_LOSSES: Registry[nn.Module] = Registry("content loss")
ADV_LOSSES: Registry["AdversarialLoss"] = Registry("adversarial loss")

# VGG19 feature index whose output is conv3_3 (inclusive slice endpoint).
_VGG_CONV3_3 = 15


# --------------------------------------------------------------------------- #
# Content losses
# --------------------------------------------------------------------------- #
@CONTENT_LOSSES.register("l1")
class _L1Content(nn.Module):
    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(fake, real)


@CONTENT_LOSSES.register("l2")
class _L2Content(nn.Module):
    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(fake, real)


ContentLoss = _L1Content  # backwards-friendly alias


@CONTENT_LOSSES.register("perceptual")
class PerceptualLoss(nn.Module):
    """VGG19 conv3_3 perceptual loss plus a pixel MSE term. Inputs are in [-1, 1]."""

    def __init__(self, w_feat: float = 0.006, w_pixel: float = 0.5, pretrained: bool = True) -> None:
        super().__init__()
        weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = torchvision.models.vgg19(weights=weights).features[:_VGG_CONV3_3]
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.w_feat = w_feat
        self.w_pixel = w_pixel
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _features(self, x01: torch.Tensor) -> torch.Tensor:
        return self.vgg((x01 - self.mean) / self.std)

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake01 = (fake + 1.0) / 2.0
        real01 = (real + 1.0) / 2.0
        feat = self.w_feat * F.mse_loss(self._features(fake01), self._features(real01).detach())
        pixel = self.w_pixel * F.mse_loss(fake01, real01)
        return feat + pixel


def build_content_loss(name: str, pretrained: bool = True) -> nn.Module:
    if name == "perceptual":
        return CONTENT_LOSSES.create(name, pretrained=pretrained)
    return CONTENT_LOSSES.create(name)


# --------------------------------------------------------------------------- #
# Image pool (buffers past generated images to stabilize the D update)
# --------------------------------------------------------------------------- #
class ImagePool:
    def __init__(self, pool_size: int = 50) -> None:
        self.pool_size = pool_size
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return images
        out = []
        for image in images:
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                out.append(image)
            elif random.random() > 0.5:
                idx = random.randint(0, self.pool_size - 1)
                out.append(self.images[idx].clone())
                self.images[idx] = image
            else:
                out.append(image)
        return torch.cat(out, dim=0)


# --------------------------------------------------------------------------- #
# Adversarial losses
# --------------------------------------------------------------------------- #
class AdversarialLoss(nn.Module):
    """Interface: compute discriminator / generator losses given a critic ``net_d``."""

    def loss_d(self, net_d: nn.Module, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss_g(self, net_d: nn.Module, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _BaseGAN(AdversarialLoss):
    """Non-relativistic GAN with an image pool. Subclasses set the pointwise criterion."""

    def __init__(self, pool_size: int = 50) -> None:
        super().__init__()
        self.pool = ImagePool(pool_size)

    @staticmethod
    def _criterion(pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        raise NotImplementedError

    def loss_d(self, net_d, fake, real):
        fake = self.pool.query(fake.detach())
        loss_real = self._criterion(net_d(real), True)
        loss_fake = self._criterion(net_d(fake), False)
        return 0.5 * (loss_real + loss_fake)

    def loss_g(self, net_d, fake, real):
        return self._criterion(net_d(fake), True)


@ADV_LOSSES.register("gan")
class GANLoss(_BaseGAN):
    @staticmethod
    def _criterion(pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return F.binary_cross_entropy_with_logits(pred, target)


@ADV_LOSSES.register("lsgan")
class LSGANLoss(_BaseGAN):
    @staticmethod
    def _criterion(pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return F.mse_loss(pred, target)


@ADV_LOSSES.register("wgan-gp")
class WGANGPLoss(AdversarialLoss):
    def __init__(self, lambda_gp: float = 10.0) -> None:
        super().__init__()
        self.lambda_gp = lambda_gp

    def _gradient_penalty(self, net_d, real, fake):
        b = real.size(0)
        alpha = torch.rand(b, 1, 1, 1, device=real.device)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        scores = net_d(interp)
        grads = torch.autograd.grad(
            outputs=scores,
            inputs=interp,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads = grads.reshape(b, -1)
        return ((grads.norm(2, dim=1) - 1) ** 2).mean()

    def loss_d(self, net_d, fake, real):
        fake = fake.detach()
        gp = self._gradient_penalty(net_d, real, fake)
        return net_d(fake).mean() - net_d(real).mean() + self.lambda_gp * gp

    def loss_g(self, net_d, fake, real):
        return -net_d(fake).mean()


class _Relativistic(AdversarialLoss):
    """Relativistic average discriminator. Subclasses define the pointwise term."""

    @staticmethod
    def _term(diff: torch.Tensor, positive: bool) -> torch.Tensor:
        raise NotImplementedError

    def loss_d(self, net_d, fake, real):
        c_fake = net_d(fake.detach())
        c_real = net_d(real)
        d_real = c_real - c_fake.mean()
        d_fake = c_fake - c_real.mean()
        return 0.5 * (self._term(d_real, True) + self._term(d_fake, False))

    def loss_g(self, net_d, fake, real):
        c_fake = net_d(fake)
        c_real = net_d(real).detach()
        d_real = c_real - c_fake.mean()
        d_fake = c_fake - c_real.mean()
        return 0.5 * (self._term(d_fake, True) + self._term(d_real, False))


@ADV_LOSSES.register("ragan")
class RelativisticGANLoss(_Relativistic):
    @staticmethod
    def _term(diff, positive):
        target = torch.ones_like(diff) if positive else torch.zeros_like(diff)
        return F.binary_cross_entropy_with_logits(diff, target)


@ADV_LOSSES.register("ragan-ls")
class RelativisticLSGANLoss(_Relativistic):
    @staticmethod
    def _term(diff, positive):
        target = 1.0 if positive else -1.0
        return torch.mean((diff - target) ** 2)


def build_adversarial_loss(name: str) -> AdversarialLoss:
    return ADV_LOSSES.create(name)
