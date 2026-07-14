"""Frame-by-frame video deblurring."""

from __future__ import annotations

from pathlib import Path

import cv2
from tqdm import tqdm

from deblurgan.inference.predictor import Predictor


def process_video(predictor: Predictor, in_path: str | Path, out_path: str | Path) -> None:
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    try:
        with tqdm(total=total, desc="frames") as bar:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                out_rgb = predictor(rgb)
                writer.write(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
                bar.update(1)
    finally:
        cap.release()
        writer.release()
