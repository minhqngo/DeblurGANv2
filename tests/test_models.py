import pytest
import torch

from deblurgan.config import DiscriminatorConfig, GeneratorConfig
from deblurgan.models import build_discriminator, build_generator

TINY_BACKBONES = ["mobilenetv3_small_050", "resnet18", "efficientnet_b0"]


@pytest.mark.parametrize("backbone", TINY_BACKBONES)
@pytest.mark.parametrize("size", [(64, 64), (64, 96)])
def test_generator_forward_shape_and_range(backbone, size):
    g = build_generator(GeneratorConfig(backbone=backbone, pretrained=False)).eval()
    h, w = size
    x = torch.randn(1, 3, h, w)
    with torch.no_grad():
        y = g(x)
    assert y.shape == x.shape
    assert y.min() >= -1.0001 and y.max() <= 1.0001


def test_generator_rejects_non_hierarchical_backbone():
    with pytest.raises(ValueError, match="feature strides"):
        build_generator(GeneratorConfig(backbone="vit_base_patch16_224", pretrained=False))


def test_td_blocks_toggle_changes_params():
    with_td = build_generator(GeneratorConfig(backbone="resnet18", pretrained=False, td_blocks=True))
    without_td = build_generator(
        GeneratorConfig(backbone="resnet18", pretrained=False, td_blocks=False)
    )
    assert sum(p.numel() for p in with_td.parameters()) > sum(
        p.numel() for p in without_td.parameters()
    )


def test_freeze_unfreeze_backbone():
    g = build_generator(GeneratorConfig(backbone="resnet18", pretrained=False))
    assert all(not p.requires_grad for p in g.fpn.backbone.parameters())
    g.unfreeze()
    assert all(p.requires_grad for p in g.fpn.backbone.parameters())
    g.freeze()
    assert all(not p.requires_grad for p in g.fpn.backbone.parameters())


def test_discriminator_variants():
    assert build_discriminator(DiscriminatorConfig(kind="no_gan")) is None

    patch = build_discriminator(DiscriminatorConfig(kind="patch_gan", num_layers=3))
    out = patch(torch.randn(1, 3, 128, 128))
    assert out.dim() == 4  # patch score map

    double = build_discriminator(DiscriminatorConfig(kind="double_gan", num_layers=3))
    p, f = double(torch.randn(1, 3, 128, 128))
    assert p.dim() == 4 and f.dim() == 4
