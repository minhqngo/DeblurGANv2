import numpy as np
import torch

from deblurgan.inference.predictor import Predictor
from deblurgan.models import build_generator
from deblurgan.training.checkpoint import save_checkpoint


def _save_tiny_checkpoint(path, tiny_config):
    gen = build_generator(tiny_config.generator)
    save_checkpoint(
        path,
        epoch=0,
        best_psnr=0.0,
        cfg=tiny_config,
        generator=gen,
        adversarial=None,
        optimizer_g=None,
        optimizer_d=None,
        scheduler_g=None,
        scheduler_d=None,
    )


def test_predict_non_multiple_of_32(tmp_path, tiny_config):
    ckpt = tmp_path / "best.pt"
    _save_tiny_checkpoint(ckpt, tiny_config)

    predictor = Predictor(ckpt, device="cpu")
    img = (np.random.rand(70, 90, 3) * 255).astype(np.uint8)  # not multiples of 32
    out = predictor(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_predictor_rebuilds_from_embedded_config(tmp_path, tiny_config):
    ckpt = tmp_path / "best.pt"
    _save_tiny_checkpoint(ckpt, tiny_config)
    predictor = Predictor(ckpt, device="cpu")
    # The backbone must match the one embedded in the checkpoint config.
    assert predictor.model.fpn.backbone is not None
    with torch.no_grad():
        out = predictor(np.zeros((64, 64, 3), np.uint8))
    assert out.shape == (64, 64, 3)


def test_legacy_checkpoint_rejected(tmp_path):
    import torch as _torch

    bad = tmp_path / "old.h5"
    _torch.save({"model": {}}, bad)
    try:
        Predictor(bad, device="cpu")
    except ValueError as e:
        assert "legacy" in str(e).lower() or "incompatible" in str(e).lower()
    else:
        raise AssertionError("expected a ValueError for a legacy checkpoint")
