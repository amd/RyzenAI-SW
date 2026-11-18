#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Prepare the DETR-ResNet50 calibration image folder (calibration_dataset/images).

Streams a pinned revision of the Hugging Face dataset "detection-datasets/coco" (COCO 2017, val
split), shuffles the stream (seed 42, buffer 1000) and saves the first 200 images as
0.png ... 199.png, in stream order.
Images are saved unprocessed (no resize/normalize): resizing to the model's static 800x800 input
and normalization happen later, at calibration time, in olive_ryzen_data_scripts.py.

Usage:
    python prepare_calibration_dataset.py [--output-dir calibration_dataset/images] [--count 200] [--seed 42]
"""
import argparse
from pathlib import Path

from datasets import load_dataset

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "images"
DEFAULT_COUNT = 200
DEFAULT_SEED = 42
BUFFER_SIZE = 1000
DATASET_NAME = "detection-datasets/coco"
REVISION = "cf0b22332314a937e9dc8a1957b21725430bb41d"
SPLIT = "val"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder to save calibration images into")
    p.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of images to save")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shuffle seed")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The calibration loader reads every image in the folder.
    for stale in args.output_dir.glob("*.png"):
        stale.unlink()

    print(f"Streaming '{DATASET_NAME}' (split={SPLIT}, revision={REVISION[:12]}) from Hugging Face...")
    dataset = load_dataset(DATASET_NAME, revision=REVISION, split=SPLIT, streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=BUFFER_SIZE)

    saved = 0
    for sample in dataset:
        if saved >= args.count:
            break
        sample["image"].convert("RGB").save(args.output_dir / f"{saved}.png")
        saved += 1

    print(f"Saved {saved} calibration images to '{args.output_dir}/'.")


if __name__ == "__main__":
    main()
