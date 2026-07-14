"""``deblurgan-eval`` entry point: PSNR/SSIM over a GoPro-style test set.

Expects paired ``.../blur/<name>`` and ``.../sharp/<name>`` images. Replaces the legacy
``test_metrics.py`` (external ``ssim`` package, ``Variable``, hardcoded paths).
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import cv2
from tqdm import tqdm

from deblurgan.inference.predictor import Predictor
from deblurgan.metrics import psnr, ssim


def _find_pairs(data_root: str, blur_glob: str | None, sharp_glob: str | None):
    if blur_glob and sharp_glob:
        blur = sorted(glob.glob(blur_glob, recursive=True))
        sharp = sorted(glob.glob(sharp_glob, recursive=True))
        return list(zip(blur, sharp))
    blur = sorted(glob.glob(str(Path(data_root) / "**" / "blur" / "*.*"), recursive=True))
    pairs = []
    for b in blur:
        # Derive the sharp path by swapping the parent 'blur' dir for 'sharp'.
        sharp_path = Path(b).parent.parent / "sharp" / Path(b).name
        if sharp_path.exists():
            pairs.append((b, str(sharp_path)))
    return pairs


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PSNR/SSIM of a model on a test set.")
    p.add_argument("checkpoint", help="path to a checkpoint (.pt)")
    p.add_argument("--data", default=None, help="dataset root containing */blur/* and */sharp/*")
    p.add_argument("--blur-glob", default=None, help="explicit blur glob (with --sharp-glob)")
    p.add_argument("--sharp-glob", default=None, help="explicit sharp glob (with --blur-glob)")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:N|mps")
    p.add_argument("--out-csv", default=None, help="write per-image metrics to this CSV")
    p.add_argument("--save-images", default=None, help="directory to write deblurred outputs")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if not args.data and not (args.blur_glob and args.sharp_glob):
        raise SystemExit("provide --data, or both --blur-glob and --sharp-glob")

    predictor = Predictor(args.checkpoint, device=args.device)
    pairs = _find_pairs(args.data, args.blur_glob, args.sharp_glob)
    if not pairs:
        raise SystemExit("no blur/sharp pairs found")

    save_dir = Path(args.save_images) if args.save_images else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    psnr_sum = ssim_sum = 0.0
    for blur_path, sharp_path in tqdm(pairs, desc="eval"):
        blur = cv2.cvtColor(cv2.imread(blur_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        sharp = cv2.cvtColor(cv2.imread(sharp_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        out = predictor(blur)
        p, s = psnr(out, sharp), ssim(out, sharp)
        psnr_sum += p
        ssim_sum += s
        rows.append({"blur": blur_path, "psnr": p, "ssim": s})
        if save_dir:
            cv2.imwrite(str(save_dir / Path(blur_path).name), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    n = len(pairs)
    print(f"images: {n}  mean PSNR: {psnr_sum / n:.4f} dB  mean SSIM: {ssim_sum / n:.4f}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["blur", "psnr", "ssim"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
