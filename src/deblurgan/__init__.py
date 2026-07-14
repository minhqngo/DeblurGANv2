"""Backbone-agnostic DeblurGAN-v2."""

from __future__ import annotations

from deblurgan.config import Config, load_config

__version__ = "2.0.0"

__all__ = ["Config", "load_config", "Predictor", "__version__"]


def __getattr__(name: str):
    # Lazy import so `import deblurgan` stays cheap and free of a hard torch/timm
    # dependency at import time (the CLIs pull those in when actually needed).
    if name == "Predictor":
        from deblurgan.inference.predictor import Predictor

        return Predictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
