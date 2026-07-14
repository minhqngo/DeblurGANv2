"""Rolling-window metric aggregation + TensorBoard logging.

Uses ``torch.utils.tensorboard`` (not tensorboardX) and has no ``logging.basicConfig``
side effect at construction time (the CLI configures logging once).
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

WINDOW = 100


class MetricLogger:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self._windows: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW))
        self._epoch_values: dict[str, list[float]] = defaultdict(list)
        self._image: np.ndarray | None = None

    def clear(self) -> None:
        self._windows.clear()
        self._epoch_values.clear()
        self._image = None

    def add(self, **metrics: float) -> None:
        for key, value in metrics.items():
            self._windows[key].append(value)
            self._epoch_values[key].append(value)

    def add_image(self, image_rgb: np.ndarray) -> None:
        """Store one RGB HWC image (uint8) for this epoch's summary."""
        self._image = image_rgb

    def running_message(self) -> str:
        return "; ".join(f"{k}={np.mean(v):.4f}" for k, v in self._windows.items() if v)

    def epoch_means(self) -> dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self._epoch_values.items() if v}

    def write(self, epoch: int, prefix: str) -> dict[str, float]:
        means = self.epoch_means()
        for key, value in means.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, epoch)
        if self._image is not None:
            self.writer.add_image(f"{prefix}/sample", self._image, epoch, dataformats="HWC")
        self.writer.flush()
        return means

    def close(self) -> None:
        self.writer.close()
