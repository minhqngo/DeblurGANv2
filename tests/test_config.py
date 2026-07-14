import textwrap

import pytest

from deblurgan.config import Config, ConfigError, load_config


def _write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(text))
    return p


MINIMAL = """
    experiment: t
    data:
      train: {blur_glob: a/*.png, sharp_glob: b/*.png}
      val: {blur_glob: c/*.png, sharp_glob: d/*.png}
"""


def test_roundtrip_defaults(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert isinstance(cfg, Config)
    assert cfg.experiment == "t"
    assert cfg.generator.backbone == "mobilenetv2_100"
    assert cfg.generator.fpn_channels == 256
    assert cfg.data.train.blur_glob == "a/*.png"
    assert cfg.losses.adversarial == "ragan-ls"


def test_nested_corrupt_parsing(tmp_path):
    cfg = load_config(_write(tmp_path, """
        experiment: t
        data:
          train:
            blur_glob: a/*.png
            sharp_glob: b/*.png
            corrupt:
              - {name: jpeg, prob: 0.3, params: {quality_range: [70, 90]}}
          val: {blur_glob: c/*.png, sharp_glob: d/*.png}
    """))
    assert cfg.data.train.corrupt[0].name == "jpeg"
    assert cfg.data.train.corrupt[0].prob == 0.3
    assert cfg.data.train.corrupt[0].params["quality_range"] == [70, 90]


def test_unknown_key_reports_path(tmp_path):
    with pytest.raises(ConfigError, match="generator.fpn_channelz"):
        load_config(_write(tmp_path, MINIMAL + "    generator: {fpn_channelz: 128}\n"))


def test_literal_violation(tmp_path):
    with pytest.raises(ConfigError, match="losses.adversarial"):
        load_config(_write(tmp_path, MINIMAL + "    losses: {adversarial: not-a-loss}\n"))


def test_missing_required_key(tmp_path):
    with pytest.raises(ConfigError, match="data.train.sharp_glob"):
        load_config(_write(tmp_path, """
            experiment: t
            data:
              train: {blur_glob: a/*.png}
              val: {blur_glob: c/*.png, sharp_glob: d/*.png}
        """))


def test_overrides(tmp_path):
    cfg = load_config(
        _write(tmp_path, MINIMAL),
        overrides=[
            "training.epochs=5",
            "scheduler.start_epoch=2",
            "generator.backbone=resnet50",
            "data.batch_size=2",
        ],
    )
    assert cfg.training.epochs == 5
    assert cfg.generator.backbone == "resnet50"
    assert cfg.data.batch_size == 2


def test_semantic_warmup_ge_epochs(tmp_path):
    with pytest.raises(ConfigError, match="warmup_epochs"):
        load_config(_write(tmp_path, MINIMAL), overrides=["training.epochs=3", "training.warmup_epochs=3"])


def test_semantic_odd_fpn_channels(tmp_path):
    with pytest.raises(ConfigError, match="fpn_channels"):
        load_config(_write(tmp_path, MINIMAL), overrides=["generator.fpn_channels=127"])


def test_betas_tuple(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL), overrides=["optimizer.betas=[0.9, 0.99]"])
    assert cfg.optimizer.betas == (0.9, 0.99)
