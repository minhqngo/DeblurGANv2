import numpy as np
import pytest

from deblurgan.config import CorruptSpec
from deblurgan.data.aug import (
    get_corrupt_function,
    get_transforms,
    known_corrupt_names,
)


@pytest.mark.parametrize("scope", ["none", "weak", "geometric"])
@pytest.mark.parametrize("crop", ["random", "center"])
def test_paired_transform_applies_to_both(scope, crop):
    transform = get_transforms(32, scope=scope, crop=crop)
    blur = (np.random.rand(48, 48, 3) * 255).astype(np.uint8)
    sharp = blur.copy()
    out_blur, out_sharp = transform(blur, sharp)
    assert out_blur.shape == (32, 32, 3)
    assert out_sharp.shape == (32, 32, 3)
    # Identical inputs + a shared geometric transform => identical outputs.
    assert np.array_equal(out_blur, out_sharp)


def test_unknown_scope_raises():
    with pytest.raises(ValueError, match="scope"):
        get_transforms(32, scope="strong")


def test_corrupt_registry_builds_every_documented_name():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    for name in known_corrupt_names():
        fn = get_corrupt_function([CorruptSpec(name=name, prob=1.0)])
        out = fn(img)
        assert out.shape == img.shape


def test_empty_corrupt_is_none():
    assert get_corrupt_function([]) is None
