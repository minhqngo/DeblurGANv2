"""The training loop.

Fixes carried over from the legacy trainer: device-agnostic (CPU/CUDA/MPS), no
``DataParallel``/``.module`` coupling, exact epoch length (no off-by-one), the D update
on a detached fake (no ``retain_graph``), ``adv_lambda`` applied to the generator term
only, and full checkpoint/resume support.
"""

from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from deblurgan.config import Config
from deblurgan.devices import amp_supported
from deblurgan.models import build_generator, build_losses
from deblurgan.training.adversarial import build_adversarial_trainer
from deblurgan.training.checkpoint import load_checkpoint, save_checkpoint
from deblurgan.training.logger import MetricLogger
from deblurgan.training.schedulers import build_scheduler, is_plateau
from deblurgan.training.visuals import sample_metrics_and_image

log = logging.getLogger(__name__)


def _build_optimizer(cfg, params) -> Optimizer:
    name = cfg.name
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}")


class Trainer:
    def __init__(
        self,
        cfg: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        logger: MetricLogger,
    ) -> None:
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger

        self.netG = build_generator(cfg.generator).to(device)
        self.content_loss, _ = build_losses(cfg.losses)
        self.content_loss = self.content_loss.to(device)
        self.adv = build_adversarial_trainer(cfg.discriminator, cfg.losses, device)

        self.amp = cfg.training.amp and amp_supported(device)
        self.scaler_g = torch.amp.GradScaler(device.type, enabled=self.amp)
        self.scaler_d = torch.amp.GradScaler(device.type, enabled=self.amp)

        self.optimizer_g = self._make_g_optimizer()
        self.optimizer_d = (
            _build_optimizer(cfg.optimizer, self.adv.parameters())
            if self.adv.trains_d
            else None
        )
        self.scheduler_g = build_scheduler(cfg.scheduler, self.optimizer_g, cfg.training.epochs)
        self.scheduler_d = (
            build_scheduler(cfg.scheduler, self.optimizer_d, cfg.training.epochs)
            if self.optimizer_d is not None
            else None
        )

        self.start_epoch = 0
        self.best_psnr = -math.inf
        if cfg.training.resume:
            self._resume(cfg.training.resume)

    # --- construction helpers ------------------------------------------------ #
    def _make_g_optimizer(self):
        """Build optimizer_G over the currently-trainable generator params.

        ``initial_lr`` is stamped on every group so a scheduler can be (re)built with an
        explicit ``last_epoch`` (needed when the optimizer is rebuilt at warmup unfreeze).
        """
        trainable = [p for p in self.netG.parameters() if p.requires_grad]
        opt = _build_optimizer(self.cfg.optimizer, trainable)
        for group in opt.param_groups:
            group.setdefault("initial_lr", self.cfg.optimizer.lr)
        return opt

    def _maybe_unfreeze(self, epoch: int) -> None:
        warmup = self.cfg.training.warmup_epochs
        if warmup and epoch == warmup:
            log.info("Unfreezing backbone at epoch %d and rebuilding optimizer_G", epoch)
            self.netG.unfreeze()
            self.optimizer_g = self._make_g_optimizer()
            self.scheduler_g = build_scheduler(
                self.cfg.scheduler, self.optimizer_g, self.cfg.training.epochs, last_epoch=epoch - 1
            )

    # --- main loop ----------------------------------------------------------- #
    def fit(self) -> None:
        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            self._maybe_unfreeze(epoch)
            self._train_epoch(epoch)
            val_metrics = self._validate(epoch)
            self._step_schedulers(val_metrics.get("psnr", 0.0))

            psnr_now = val_metrics.get("psnr", -math.inf)
            is_best = psnr_now > self.best_psnr
            if is_best:
                self.best_psnr = psnr_now
            self._save(epoch, is_best)
            log.info(
                "epoch %d done | val PSNR %.3f (best %.3f)", epoch, psnr_now, self.best_psnr
            )
        self.logger.close()

    def _epoch_batches(self, loader: DataLoader, limit: int | None):
        n = len(loader) if limit is None else min(len(loader), limit)
        return itertools.islice(loader, n), n

    def _train_epoch(self, epoch: int) -> None:
        self.netG.train()
        self.adv.train()
        self.logger.clear()
        batches, _ = self._epoch_batches(self.train_loader, self.cfg.training.batches_per_epoch)
        for i, batch in enumerate(batches):
            blur = batch["blur"].to(self.device, non_blocking=True)
            sharp = batch["sharp"].to(self.device, non_blocking=True)
            losses, fake = self._train_step(blur, sharp)
            self.logger.add(**losses)
            if i == 0:
                p, s, vis = sample_metrics_and_image(blur, fake, sharp)
                self.logger.add(psnr=p, ssim=s)
                self.logger.add_image(vis)
        self.logger.write(epoch, prefix="train")

    def _train_step(self, blur: torch.Tensor, sharp: torch.Tensor):
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            fake = self.netG(blur)

        loss_d_val = 0.0
        if self.adv.trains_d:
            self.optimizer_d.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                loss_d = self.adv.loss_d(fake.detach(), sharp)
            self.scaler_d.scale(loss_d).backward()
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()
            loss_d_val = loss_d.item()

        self.optimizer_g.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            loss_content = self.content_loss(fake, sharp)
            loss_adv = self.adv.loss_g(fake, sharp)
            loss_g = loss_content + self.cfg.losses.adv_lambda * loss_adv
        self.scaler_g.scale(loss_g).backward()
        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()

        return (
            {
                "G": loss_g.item(),
                "content": loss_content.item(),
                "adv": loss_adv.item(),
                "D": loss_d_val,
            },
            fake,
        )

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        self.netG.eval()
        self.logger.clear()
        batches, _ = self._epoch_batches(
            self.val_loader, self.cfg.training.val_batches_per_epoch
        )
        for i, batch in enumerate(batches):
            blur = batch["blur"].to(self.device)
            sharp = batch["sharp"].to(self.device)
            fake = self.netG(blur)
            loss_content = self.content_loss(fake, sharp)
            p, s, vis = sample_metrics_and_image(blur, fake, sharp)
            self.logger.add(content=loss_content.item(), psnr=p, ssim=s)
            if i == 0:
                self.logger.add_image(vis)
        return self.logger.write(epoch, prefix="val")

    def _step_schedulers(self, val_psnr: float) -> None:
        if is_plateau(self.cfg.scheduler):
            self.scheduler_g.step(val_psnr)
            if self.scheduler_d is not None:
                self.scheduler_d.step(val_psnr)
        else:
            self.scheduler_g.step()
            if self.scheduler_d is not None:
                self.scheduler_d.step()

    # --- checkpoint / resume ------------------------------------------------- #
    def _save(self, epoch: int, is_best: bool) -> None:
        out = Path(self.cfg.training.output_dir) / self.cfg.experiment
        save_checkpoint(
            out / "last.pt",
            epoch=epoch,
            best_psnr=self.best_psnr,
            cfg=self.cfg,
            generator=self.netG,
            adversarial=self.adv if self.adv.trains_d else None,
            optimizer_g=self.optimizer_g,
            optimizer_d=self.optimizer_d,
            scheduler_g=self.scheduler_g,
            scheduler_d=self.scheduler_d,
        )
        if is_best:
            save_checkpoint(
                out / "best.pt",
                epoch=epoch,
                best_psnr=self.best_psnr,
                cfg=self.cfg,
                generator=self.netG,
                adversarial=self.adv if self.adv.trains_d else None,
                optimizer_g=self.optimizer_g,
                optimizer_d=self.optimizer_d,
                scheduler_g=self.scheduler_g,
                scheduler_d=self.scheduler_d,
            )

    def _resume(self, path: str) -> None:
        ckpt = load_checkpoint(path, map_location=self.device)
        self.netG.load_state_dict(ckpt["generator"])
        resumed_epoch = ckpt["epoch"]
        self.start_epoch = resumed_epoch + 1
        self.best_psnr = ckpt["best_psnr"]

        warmup = self.cfg.training.warmup_epochs
        if warmup and self.start_epoch > warmup:
            # Replay the unfreeze so optimizer param groups match the saved state;
            # the scheduler is rebuilt fresh and then restored from state below.
            self.netG.unfreeze()
            self.optimizer_g = self._make_g_optimizer()
            self.scheduler_g = build_scheduler(
                self.cfg.scheduler, self.optimizer_g, self.cfg.training.epochs
            )

        if ckpt.get("optimizer_g"):
            self.optimizer_g.load_state_dict(ckpt["optimizer_g"])
        if ckpt.get("scheduler_g"):
            self.scheduler_g.load_state_dict(ckpt["scheduler_g"])
        if self.adv.trains_d and ckpt.get("adversarial"):
            self.adv.load_state_dict(ckpt["adversarial"])
        if self.optimizer_d is not None and ckpt.get("optimizer_d"):
            self.optimizer_d.load_state_dict(ckpt["optimizer_d"])
        if self.scheduler_d is not None and ckpt.get("scheduler_d"):
            self.scheduler_d.load_state_dict(ckpt["scheduler_d"])
        log.info("Resumed from %s at epoch %d", path, self.start_epoch)
