"""Normalization-layer factory."""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch.nn as nn


def get_norm_layer(norm: str) -> Callable[[int], nn.Module]:
    """Return a callable ``num_features -> nn.Module`` for the requested norm type.

    ``instance`` uses ``affine=True, track_running_stats=False`` (unlike the legacy
    ``affine=False, track_running_stats=True``): with no running stats the module
    behaves identically in train and eval mode, so inference can use ``.eval()``
    instead of the old ``model.train(True)`` hack.
    """
    if norm == "instance":
        return partial(nn.InstanceNorm2d, affine=True, track_running_stats=False)
    if norm == "batch":
        return partial(nn.BatchNorm2d, affine=True)
    if norm == "none":
        return lambda _num_features: nn.Identity()
    raise ValueError(f"Unknown norm type {norm!r} (expected 'instance', 'batch', or 'none')")
