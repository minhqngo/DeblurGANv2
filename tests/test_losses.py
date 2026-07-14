import pytest
import torch

from deblurgan.models.discriminators import NLayerDiscriminator
from deblurgan.models.losses import (
    ADV_LOSSES,
    PerceptualLoss,
    build_adversarial_loss,
    build_content_loss,
)

ADV_NAMES = ["gan", "lsgan", "wgan-gp", "ragan", "ragan-ls"]


def test_content_l1_l2_finite_and_grad():
    for name in ("l1", "l2"):
        loss_fn = build_content_loss(name)
        fake = torch.randn(2, 3, 32, 32, requires_grad=True)
        real = torch.randn(2, 3, 32, 32)
        loss = loss_fn(fake, real)
        assert torch.isfinite(loss)
        loss.backward()
        assert fake.grad is not None


def test_perceptual_finite_and_grad_batch_gt_1():
    # pretrained=False keeps this offline; the bug being guarded against corrupted
    # all but batch element 0, so use batch size 2.
    loss_fn = PerceptualLoss(pretrained=False)
    fake = (torch.rand(2, 3, 64, 64) * 2 - 1).requires_grad_(True)
    real = torch.rand(2, 3, 64, 64) * 2 - 1
    loss = loss_fn(fake, real)
    assert torch.isfinite(loss)
    loss.backward()
    assert fake.grad is not None and torch.isfinite(fake.grad).all()
    # Every batch element must receive gradient (the in-place [0]-only bug is gone).
    assert (fake.grad.flatten(1).abs().sum(dim=1) > 0).all()


@pytest.mark.parametrize("name", ADV_NAMES)
def test_adversarial_d_and_g_losses(name):
    torch.manual_seed(0)
    adv = build_adversarial_loss(name)
    net_d = NLayerDiscriminator(n_layers=2)

    real = torch.rand(2, 3, 64, 64) * 2 - 1
    fake = (torch.rand(2, 3, 64, 64) * 2 - 1).requires_grad_(True)

    # D step: on detached fake. Gradient flows to D, not to the generator's fake.
    fake_detached = fake.detach()
    loss_d = adv.loss_d(net_d, fake_detached, real)
    assert torch.isfinite(loss_d)
    loss_d.backward()
    assert any(p.grad is not None for p in net_d.parameters())

    # G step: on attached fake -> gradient flows back to the generator output.
    net_d.zero_grad()
    loss_g = adv.loss_g(net_d, fake, real)
    assert torch.isfinite(loss_g)
    loss_g.backward()
    assert fake.grad is not None


def test_all_adv_losses_registered():
    assert set(ADV_NAMES).issubset(set(ADV_LOSSES.names()))
