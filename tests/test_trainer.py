import dataclasses

import torch

from deblurgan.data.dataset import PairedDataset, build_dataloader
from deblurgan.training.checkpoint import load_checkpoint
from deblurgan.training.logger import MetricLogger
from deblurgan.training.trainer import Trainer


def _loaders(cfg, device):
    train_ds = PairedDataset.from_config(cfg.data.train)
    val_ds = PairedDataset.from_config(cfg.data.val)
    train = build_dataloader(train_ds, cfg.data.batch_size, 0, shuffle=True, device=device, drop_last=True)
    val = build_dataloader(val_ds, cfg.data.val_batch_size, 0, shuffle=False, device=device)
    return train, val


def test_end_to_end_smoke(tiny_config, tmp_path):
    device = torch.device("cpu")
    train, val = _loaders(tiny_config, device)
    logger = MetricLogger(tmp_path / "tb")
    trainer = Trainer(tiny_config, train, val, device, logger)

    before = trainer.netG.final.weight.detach().clone()
    trainer.fit()
    after = trainer.netG.final.weight.detach()

    assert not torch.equal(before, after)  # params moved
    out = tmp_path / "runs" / "test"
    assert (out / "last.pt").exists()


def test_warmup_unfreezes_and_rebuilds_optimizer(tiny_config, tmp_path):
    device = torch.device("cpu")
    train, val = _loaders(tiny_config, device)
    trainer = Trainer(tiny_config, train, val, device, MetricLogger(tmp_path / "tb"))

    # At init the backbone is frozen; optimizer_G sees only decoder params.
    assert all(not p.requires_grad for p in trainer.netG.fpn.backbone.parameters())
    n_before = sum(len(g["params"]) for g in trainer.optimizer_g.param_groups)

    trainer._maybe_unfreeze(tiny_config.training.warmup_epochs)
    assert all(p.requires_grad for p in trainer.netG.fpn.backbone.parameters())
    n_after = sum(len(g["params"]) for g in trainer.optimizer_g.param_groups)
    assert n_after > n_before


def test_checkpoint_resume_roundtrip(tiny_config, tmp_path):
    device = torch.device("cpu")
    train, val = _loaders(tiny_config, device)

    # Train the full 2 epochs, then resume for 1 more.
    trainer = Trainer(tiny_config, train, val, device, MetricLogger(tmp_path / "tb1"))
    trainer.fit()
    ckpt_path = tmp_path / "runs" / "test" / "last.pt"
    ckpt = load_checkpoint(ckpt_path)
    assert ckpt["epoch"] == tiny_config.training.epochs - 1
    assert "optimizer_g" in ckpt and ckpt["optimizer_g"] is not None
    assert ckpt["adversarial"] is not None  # double_gan discriminator saved

    resumed_cfg = dataclasses.replace(
        tiny_config,
        training=dataclasses.replace(
            tiny_config.training, epochs=3, resume=str(ckpt_path)
        ),
    )
    trainer2 = Trainer(resumed_cfg, train, val, device, MetricLogger(tmp_path / "tb2"))
    assert trainer2.start_epoch == tiny_config.training.epochs  # continues, not restarts
    assert trainer2.best_psnr == ckpt["best_psnr"]
    trainer2.fit()
