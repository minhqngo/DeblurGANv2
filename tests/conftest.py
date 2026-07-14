import cv2
import numpy as np
import pytest

from deblurgan.config import (
    Config,
    DataConfig,
    DiscriminatorConfig,
    GeneratorConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    SplitConfig,
    TrainingConfig,
)


def _make_pair(blur_dir, sharp_dir, name, size=128):
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    sharp = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    # A distinct, correlated blur so blur != sharp but they are paired.
    blur = cv2.GaussianBlur(sharp, (7, 7), 0)
    cv2.imwrite(str(blur_dir / name), cv2.cvtColor(blur, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(sharp_dir / name), cv2.cvtColor(sharp, cv2.COLOR_RGB2BGR))


@pytest.fixture
def paired_dirs(tmp_path):
    """Create 6 paired blur/sharp PNGs; return (blur_glob, sharp_glob)."""
    blur_dir = tmp_path / "blur"
    sharp_dir = tmp_path / "sharp"
    blur_dir.mkdir()
    sharp_dir.mkdir()
    for i in range(6):
        _make_pair(blur_dir, sharp_dir, f"img_{i:03d}.png")
    return str(blur_dir / "*.png"), str(sharp_dir / "*.png")


@pytest.fixture
def tiny_config(tmp_path, paired_dirs):
    """A minimal CPU/offline config for smoke tests (no pretrained downloads)."""
    blur_glob, sharp_glob = paired_dirs
    return Config(
        experiment="test",
        data=DataConfig(
            train=SplitConfig(blur_glob=blur_glob, sharp_glob=sharp_glob, size=128, scope="weak"),
            val=SplitConfig(
                blur_glob=blur_glob, sharp_glob=sharp_glob, size=128, scope="none", crop="center"
            ),
            batch_size=2,
            val_batch_size=1,
            num_workers=0,
        ),
        generator=GeneratorConfig(backbone="mobilenetv3_small_050", pretrained=False),
        discriminator=DiscriminatorConfig(kind="double_gan", num_layers=2),
        losses=LossConfig(content="l1", adversarial="ragan-ls", adv_lambda=0.01),
        optimizer=OptimizerConfig(name="adam", lr=1e-4),
        scheduler=SchedulerConfig(name="linear", start_epoch=1),
        training=TrainingConfig(
            epochs=2,
            warmup_epochs=1,
            batches_per_epoch=2,
            val_batches_per_epoch=1,
            device="cpu",
            output_dir=str(tmp_path / "runs"),
        ),
    )
