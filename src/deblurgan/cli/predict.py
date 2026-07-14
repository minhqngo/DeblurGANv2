"""``deblurgan-predict`` entry point.

Accepts image files, directories, or globs, plus a ``--video`` mode.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2

from deblurgan.inference.predictor import Predictor
from deblurgan.inference.video import process_video

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _expand_inputs(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files += [p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
        elif glob.has_magic(item):
            files += [Path(p) for p in sorted(glob.glob(item, recursive=True))]
        else:
            files.append(path)
    return files


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deblur images or a video with a trained model.")
    p.add_argument("checkpoint", help="path to a checkpoint (.pt)")
    p.add_argument("inputs", nargs="+", help="image file(s), directory, glob, or a video with --video")
    p.add_argument("--output", "-O", default="output", help="output directory (or file for --video)")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:N|mps")
    p.add_argument("--video", action="store_true", help="treat inputs as video files")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    predictor = Predictor(args.checkpoint, device=args.device)

    if args.video:
        out = Path(args.output)
        for item in args.inputs:
            dest = out if out.suffix else out / (Path(item).stem + "_deblurred.mp4")
            process_video(predictor, item, dest)
            print(f"{item} -> {dest}")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _expand_inputs(args.inputs)
    if not files:
        raise SystemExit("no input images found")
    for path in files:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip (unreadable): {path}")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = predictor(rgb)
        dest = out_dir / path.name
        cv2.imwrite(str(dest), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"{path} -> {dest}")


if __name__ == "__main__":
    main()
