"""Typed configuration for DeblurGAN-v2.

A small dataclass hierarchy plus a recursive ``from_dict`` loader that validates
YAML against the dataclasses (unknown keys and bad ``Literal`` values are errors,
reported with their dotted path). No pydantic / dacite / Hydra dependency.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

Norm = Literal["instance", "batch", "none"]


class ConfigError(ValueError):
    """Raised when a config file cannot be parsed into the dataclass schema."""


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
@dataclass
class CorruptSpec:
    """One entry in a split's synthetic-corruption pipeline (an albumentations op)."""

    name: str
    prob: float = 0.5
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitConfig:
    """A single data split (train or val).

    ``blur_glob`` and ``sharp_glob`` are kept separate on purpose: the legacy config
    pointed both at the same glob, so the model learned to reconstruct its own input.
    """

    blur_glob: str
    sharp_glob: str
    size: int = 256
    crop: Literal["random", "center"] = "random"
    scope: Literal["none", "weak", "geometric"] = "geometric"
    corrupt: list[CorruptSpec] = field(default_factory=list)
    preload: bool = False
    preload_size: int = 0


@dataclass
class DataConfig:
    train: SplitConfig
    val: SplitConfig
    batch_size: int = 8
    val_batch_size: int = 1
    num_workers: int = 4  # -1 => os.cpu_count()


@dataclass
class GeneratorConfig:
    backbone: str = "mobilenetv2_100"  # any timm model name
    pretrained: bool = True
    fpn_channels: int = 256
    head_channels: int = 128
    td_blocks: bool = True
    norm: Norm = "instance"
    out_channels: int = 3


@dataclass
class DiscriminatorConfig:
    kind: Literal["no_gan", "patch_gan", "double_gan"] = "double_gan"
    num_layers: int = 3
    norm: Norm = "instance"


@dataclass
class LossConfig:
    content: Literal["perceptual", "l1", "l2"] = "perceptual"
    adversarial: Literal["gan", "lsgan", "wgan-gp", "ragan", "ragan-ls"] = "ragan-ls"
    adv_lambda: float = 0.001  # scales the generator's adversarial term only


@dataclass
class OptimizerConfig:
    name: Literal["adam", "adamw", "sgd"] = "adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.5, 0.999)


@dataclass
class SchedulerConfig:
    name: Literal["linear", "cosine", "plateau", "constant"] = "linear"
    start_epoch: int = 50  # linear: hold lr until here, then decay to min_lr at `epochs`
    min_lr: float = 1e-7
    patience: int = 10  # plateau only
    factor: float = 0.5  # plateau only


@dataclass
class TrainingConfig:
    epochs: int = 200
    warmup_epochs: int = 3  # backbone frozen for these epochs, then unfrozen
    batches_per_epoch: int | None = 1000  # None => full dataset each epoch
    val_batches_per_epoch: int | None = None
    device: str = "auto"  # auto | cpu | cuda | cuda:N | mps
    amp: bool = False
    seed: int = 42
    resume: str | None = None  # path to a checkpoint to resume from
    output_dir: str = "runs"


@dataclass
class Config:
    data: DataConfig
    experiment: str = "deblurgan"
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _coerce(hint: Any, value: Any, path: str) -> Any:
    """Coerce ``value`` to the type described by ``hint``, recursing into dataclasses."""
    origin = get_origin(hint)

    # Optional[X] / Union[..., None]
    if origin is Union:
        args = [a for a in get_args(hint) if a is not type(None)]
        if value is None:
            return None
        # Optional[X] -> unwrap to X; genuine multi-member unions are not used here.
        return _coerce(args[0], value, path)

    if is_dataclass(hint):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        return _from_dict(hint, value, path)

    if origin in (list, list.__class__) or origin is list:
        (item_hint,) = get_args(hint) or (Any,)
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        return [_coerce(item_hint, v, f"{path}[{i}]") for i, v in enumerate(value)]

    if origin is tuple:
        item_hints = get_args(hint)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list/tuple, got {type(value).__name__}")
        if len(item_hints) == 2 and item_hints[1] is Ellipsis:
            return tuple(_coerce(item_hints[0], v, f"{path}[{i}]") for i, v in enumerate(value))
        if len(value) != len(item_hints):
            raise ConfigError(f"{path}: expected {len(item_hints)} items, got {len(value)}")
        return tuple(_coerce(h, v, f"{path}[{i}]") for i, (h, v) in enumerate(zip(item_hints, value)))

    if origin is Literal:
        allowed = get_args(hint)
        if value not in allowed:
            raise ConfigError(f"{path}: {value!r} is not one of {list(allowed)}")
        return value

    # Plain scalar. Be permissive between int/float (YAML numbers), strict otherwise.
    if hint is float and isinstance(value, int):
        return float(value)
    if hint is int and isinstance(value, bool):  # bool is a subclass of int; keep them distinct
        raise ConfigError(f"{path}: expected int, got bool")
    return value


def _from_dict(cls: type, data: dict, path: str = "") -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path or cls.__name__}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    fields_by_name = {f.name: f for f in dataclasses.fields(cls)}

    unknown = set(data) - set(fields_by_name)
    if unknown:
        prefix = f"{path}." if path else ""
        raise ConfigError(
            f"unknown config key(s): {', '.join(sorted(prefix + k for k in unknown))}"
        )

    kwargs: dict[str, Any] = {}
    for name, f in fields_by_name.items():
        child_path = f"{path}.{name}" if path else name
        if name in data:
            kwargs[name] = _coerce(hints[name], data[name], child_path)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:  # type: ignore[misc]
            raise ConfigError(f"missing required config key: {child_path}")
    return cls(**kwargs)


def _apply_override(data: dict, dotted_key: str, raw_value: str) -> None:
    """Apply a single ``a.b.c=value`` override in place (value parsed as YAML scalar)."""
    keys = dotted_key.split(".")
    node = data
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ConfigError(f"override {dotted_key}: '{k}' is not a mapping")
    node[keys[-1]] = yaml.safe_load(raw_value)


def load_config(path: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> Config:
    """Load a YAML file into a validated :class:`Config`.

    ``overrides`` is a list of ``dotted.key=value`` strings (value parsed as a YAML
    scalar), e.g. ``["training.epochs=5", "generator.backbone=resnet50"]``.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")

    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"override {override!r} must be of the form key=value")
        key, value = override.split("=", 1)
        _apply_override(data, key.strip(), value.strip())

    cfg = _from_dict(Config, data)
    validate(cfg)
    return cfg


def load_config_from_dict(data: dict) -> Config:
    """Parse and validate a :class:`Config` from an already-loaded mapping."""
    cfg = _from_dict(Config, data)
    validate(cfg)
    return cfg


def config_to_dict(cfg: Config) -> dict:
    """Serialize a :class:`Config` back to plain dicts (for checkpoint embedding)."""
    return dataclasses.asdict(cfg)


# --------------------------------------------------------------------------- #
# Semantic validation (beyond type/Literal checks done during parsing)
# --------------------------------------------------------------------------- #
def validate(cfg: Config) -> None:
    t = cfg.training
    if t.epochs < 1:
        raise ConfigError("training.epochs must be >= 1")
    if not (0 <= t.warmup_epochs < t.epochs):
        raise ConfigError(
            f"training.warmup_epochs ({t.warmup_epochs}) must be in [0, epochs={t.epochs})"
        )
    if cfg.scheduler.name == "linear" and not (0 <= cfg.scheduler.start_epoch < t.epochs):
        raise ConfigError(
            f"scheduler.start_epoch ({cfg.scheduler.start_epoch}) must be in [0, epochs={t.epochs}) "
            "for the linear scheduler"
        )
    if cfg.generator.fpn_channels % 2 != 0:
        raise ConfigError("generator.fpn_channels must be even (fpn_channels//2 is used)")
    if cfg.data.batch_size < 1:
        raise ConfigError("data.batch_size must be >= 1")
    if cfg.losses.adv_lambda < 0:
        raise ConfigError("losses.adv_lambda must be >= 0")
