# DeblurGAN-v2 (backbone-agnostic refactor)

A modernized, backbone-agnostic implementation of
[DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better](https://arxiv.org/abs/1908.03826)
(Kupyn, Martyniuk, Wu, Wang — ICCV 2019).

The paper's core idea is a Feature Pyramid Network (FPN) generator that "can flexibly
work with a wide range of backbones." This implementation delivers on that: the
generator wraps **any hierarchical [timm](https://github.com/huggingface/pytorch-image-models)
backbone**, inferring the feature-map channel counts automatically. Swapping backbones is
a one-line config change:

```yaml
generator:
  backbone: mobilenetv2_100   # or resnet50, efficientnet_b0, densenet121, inception_resnet_v2, ...
```

![](./doc_images/pipeline.jpg)

## What's here

- **One generic FPN generator** (`deblurgan/models/fpn.py`) over any timm CNN backbone —
  replacing the four copy-pasted per-backbone files of the original.
- **Typed YAML config** parsed into validated dataclasses (`deblurgan/config.py`).
- **Device-agnostic** training/inference (CPU / CUDA / MPS), no `DataParallel`.
- **Resume-capable checkpoints** that embed the model config (inference needs no YAML).
- **Unified CLIs**: `deblurgan-train`, `deblurgan-predict`, `deblurgan-eval`.
- **A pytest suite** that runs on CPU with no network access.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # or: uv pip install -e ".[dev]"
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.2.

## Data layout

Point the config at parallel `blur/` and `sharp/` directories (e.g. the GoPro dataset):

```
data/gopro/train/<sequence>/blur/*.png
data/gopro/train/<sequence>/sharp/*.png
```

`blur_glob` and `sharp_glob` are configured **separately** — matching filenames across the
two globs are verified at load time.

## Train

```bash
deblurgan-train --config configs/gopro.yaml
```

Override any config value on the command line (repeatable `-o key=value`):

```bash
deblurgan-train --config configs/gopro.yaml \
  -o generator.backbone=resnet50 \
  -o training.epochs=100 \
  -o data.batch_size=4
```

Checkpoints and TensorBoard logs are written to `runs/<experiment>/`. Resume with:

```bash
deblurgan-train --config configs/gopro.yaml -o training.resume=runs/<experiment>/last.pt
```

## Predict

```bash
# single image, a directory, or a glob
deblurgan-predict runs/<experiment>/best.pt path/to/blurry.png --output out/
deblurgan-predict runs/<experiment>/best.pt "images/*.png" --output out/

# video (frame-by-frame)
deblurgan-predict runs/<experiment>/best.pt clip.mp4 --video --output clip_deblurred.mp4
```

## Evaluate

```bash
deblurgan-eval runs/<experiment>/best.pt --data data/gopro/test --out-csv metrics.csv
```

Reports mean PSNR / SSIM (scikit-image).

## Tests

```bash
pytest
```

## Changes from the original implementation

This is a clean-break rewrite. Notable behavioral differences:

- **Old `.h5` checkpoints are not loadable** — the generic FPN has different module
  names; retrain with this code.
- Generator/backbone is any timm model; the vendored MobileNet/SENet/Inception code and
  the hardcoded `mobilenetv2.pth.tar` path are gone.
- Training data is **RGB** (the original fed BGR to training but RGB to inference).
- `adv_lambda` scales the **generator's** adversarial term only (the original also scaled
  the discriminator loss).
- Perceptual loss applies ImageNet normalization to the whole batch (the original
  corrupted all but the first batch element in place).
- Default learning rate is `1e-4` (was `0.01`); validation images are not corrupted by
  default; instance norm uses `affine=True, track_running_stats=False`, so inference runs
  in `.eval()` mode.
- DenseNet-style configs now get the global residual + output clamp like every other
  backbone (the original densenet variant returned a bare `tanh`).
```
