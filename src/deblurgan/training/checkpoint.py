"""Resume-capable checkpoints.

A checkpoint bundles the generator, discriminator(s), both optimizers and schedulers,
the epoch counter, the best PSNR so far, and the full config (as a YAML string, so the
Predictor can rebuild the generator without a separate config file). The legacy format
saved only ``{'model': generator_state}`` and could not be resumed.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from deblurgan.config import Config, GeneratorConfig, load_config_from_dict

CHECKPOINT_VERSION = 2


def _yaml_safe(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return [_yaml_safe(v) for v in obj]
    if isinstance(obj, list):
        return [_yaml_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _yaml_safe(v) for k, v in obj.items()}
    return obj


def _sd(module) -> dict | None:
    return None if module is None else module.state_dict()


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    best_psnr: float,
    cfg: Config,
    generator,
    adversarial,
    optimizer_g,
    optimizer_d,
    scheduler_g,
    scheduler_d,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "best_psnr": best_psnr,
        "config_yaml": yaml.safe_dump(_yaml_safe(asdict(cfg)), sort_keys=False),
        "generator": generator.state_dict(),
        "adversarial": _sd(adversarial),
        "optimizer_g": _sd(optimizer_g),
        "optimizer_d": _sd(optimizer_d),
        "scheduler_g": _sd(scheduler_g),
        "scheduler_d": _sd(scheduler_d),
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    """Load a checkpoint dict, with a clear error for the incompatible legacy ``.h5``."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "version" not in ckpt:
        raise ValueError(
            f"{path} is not a DeblurGAN-v2 (>=2.0) checkpoint. Legacy '.h5' weights are "
            "incompatible with the refactored architecture and cannot be loaded; retrain "
            "with the new code."
        )
    return ckpt


def config_from_checkpoint(ckpt: dict) -> Config:
    return load_config_from_dict(yaml.safe_load(ckpt["config_yaml"]))


def generator_config_from_checkpoint(ckpt: dict) -> GeneratorConfig:
    return config_from_checkpoint(ckpt).generator
