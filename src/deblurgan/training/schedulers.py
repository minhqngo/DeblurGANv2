"""Learning-rate scheduler factory built on stdlib PyTorch schedulers.

Replaces the legacy ``schedulers.py`` (private ``_LRScheduler`` subclass, a linear decay
whose denominator never reached ``min_lr``, and an ``sgdr`` branch keyed on the wrong
config field).
"""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim import lr_scheduler

from deblurgan.config import SchedulerConfig


def build_scheduler(
    cfg: SchedulerConfig,
    optimizer: Optimizer,
    total_epochs: int,
    last_epoch: int = -1,
) -> lr_scheduler.LRScheduler:
    """Create the epoch-stepped scheduler for ``optimizer``.

    ``plateau`` returns a ``ReduceLROnPlateau`` (``mode='max'``, stepped with val PSNR);
    all others are stepped once per epoch with no argument.
    """
    if cfg.name == "constant":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _e: 1.0, last_epoch=last_epoch)

    if cfg.name == "linear":
        base_lr = optimizer.defaults["lr"]
        floor = cfg.min_lr / base_lr if base_lr > 0 else 0.0
        span = max(1, total_epochs - cfg.start_epoch)

        def lr_lambda(epoch: int) -> float:
            if epoch < cfg.start_epoch:
                return 1.0
            frac = (epoch - cfg.start_epoch) / span
            return max(floor, 1.0 - frac)

        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)

    if cfg.name == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=cfg.min_lr, last_epoch=last_epoch
        )

    if cfg.name == "plateau":
        # last_epoch is unsupported by ReduceLROnPlateau; it has no schedule to replay.
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=cfg.patience, factor=cfg.factor, min_lr=cfg.min_lr
        )

    raise ValueError(f"Unknown scheduler {cfg.name!r}")


def is_plateau(cfg: SchedulerConfig) -> bool:
    return cfg.name == "plateau"
