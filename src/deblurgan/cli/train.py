"""``deblurgan-train`` entry point."""

from __future__ import annotations

import argparse
import logging

import torch

from deblurgan.config import load_config
from deblurgan.data.dataset import PairedDataset, build_dataloader
from deblurgan.devices import resolve_device
from deblurgan.training.logger import MetricLogger
from deblurgan.training.trainer import Trainer


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a DeblurGAN-v2 model.")
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument(
        "-o", "--override", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. -o training.epochs=5 (repeatable)",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    cfg = load_config(args.config, overrides=args.override)

    torch.manual_seed(cfg.training.seed)
    device = resolve_device(cfg.training.device)
    logging.getLogger(__name__).info("Using device: %s | backbone: %s", device, cfg.generator.backbone)

    train_ds = PairedDataset.from_config(cfg.data.train)
    val_ds = PairedDataset.from_config(cfg.data.val)
    train_loader = build_dataloader(
        train_ds, cfg.data.batch_size, cfg.data.num_workers, shuffle=True, device=device, drop_last=True
    )
    val_loader = build_dataloader(
        val_ds, cfg.data.val_batch_size, cfg.data.num_workers, shuffle=False, device=device
    )

    log_dir = f"{cfg.training.output_dir}/{cfg.experiment}"
    trainer = Trainer(cfg, train_loader, val_loader, device, MetricLogger(log_dir))
    trainer.fit()


if __name__ == "__main__":
    main()
