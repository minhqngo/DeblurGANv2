import cv2
import numpy as np
import pytest

from deblurgan.config import SplitConfig
from deblurgan.data.dataset import PairedDataset


def test_pairing_and_item_shape(paired_dirs):
    blur_glob, sharp_glob = paired_dirs
    ds = PairedDataset.from_config(
        SplitConfig(blur_glob=blur_glob, sharp_glob=sharp_glob, size=64, scope="none")
    )
    assert len(ds) == 6
    item = ds[0]
    assert item["blur"].shape == (3, 64, 64)
    assert item["sharp"].shape == (3, 64, 64)
    assert item["blur"].dtype.is_floating_point
    assert -1.01 <= float(item["blur"].min()) and float(item["blur"].max()) <= 1.01


def test_count_mismatch_raises(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "s").mkdir()
    for i in range(3):
        cv2.imwrite(str(tmp_path / "b" / f"{i}.png"), np.zeros((8, 8, 3), np.uint8))
    for i in range(2):
        cv2.imwrite(str(tmp_path / "s" / f"{i}.png"), np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(ValueError, match="count mismatch"):
        PairedDataset.from_config(
            SplitConfig(blur_glob=str(tmp_path / "b/*.png"), sharp_glob=str(tmp_path / "s/*.png"))
        )


def test_rgb_channel_order(tmp_path):
    # Write an image that is pure red in RGB. cv2 stores BGR, so channel 0 must be hot.
    blur_dir = tmp_path / "blur"
    sharp_dir = tmp_path / "sharp"
    blur_dir.mkdir()
    sharp_dir.mkdir()
    red_rgb = np.zeros((64, 64, 3), np.uint8)
    red_rgb[..., 0] = 255  # R
    for d in (blur_dir, sharp_dir):
        cv2.imwrite(str(d / "x.png"), cv2.cvtColor(red_rgb, cv2.COLOR_RGB2BGR))

    ds = PairedDataset.from_config(
        SplitConfig(
            blur_glob=str(blur_dir / "*.png"),
            sharp_glob=str(sharp_dir / "*.png"),
            size=64,
            scope="none",
            crop="center",
        )
    )
    sharp = ds[0]["sharp"]  # CHW in [-1,1]
    means = sharp.mean(dim=(1, 2))
    assert means[0] > means[1] and means[0] > means[2]  # channel 0 (R) is dominant
