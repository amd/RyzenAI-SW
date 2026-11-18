#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Prepare the ConvNeXt Tiny calibration image folder (calibration_dataset/images).

The 200 images are rows of the train split of the Hugging Face dataset "timm/mini-imagenet"
(revision REVISION). SOURCES maps each parquet file of the train split to {row: image number};
image number N is saved as N.png. Only the parquet row groups that contain these rows are read
(about 250 MB). Each image is decoded the way the datasets library decodes it (EXIF orientation
applied), converted to RGB and saved unprocessed as PNG: resizing and normalization happen later, at
calibration time, in olive_ryzen_data_scripts.py, which reads the images in calib_list.txt order.

Usage:
    python prepare_calibration_dataset.py [--output-dir calibration_dataset/images]
"""
import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Image as ImageFeature
from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "images"
DATASET_NAME = "timm/mini-imagenet"
REVISION = "bd8779f9d33c061ea6e75fdd3bce4e43dd679060"

# parquet file index (data/train-000NN-of-00013.parquet) -> {row in that file: image number}
SOURCES = {
    0: {3: 137, 9: 132, 16: 82, 33: 76, 37: 21, 41: 64, 45: 19, 50: 20, 63: 191, 70: 42, 78: 106, 90: 178, 94: 80, 97: 45},
    1: {5: 149, 11: 197, 12: 17, 23: 188, 28: 95, 33: 198, 40: 143, 43: 3, 45: 113, 56: 115, 66: 121, 75: 41, 77: 49, 85: 5, 89: 94, 93: 180, 103: 142, 110: 123},
    2: {8: 6, 16: 40, 24: 118, 32: 65, 43: 90, 44: 46, 47: 75, 49: 184, 50: 114, 54: 28, 55: 112, 56: 125, 70: 109, 73: 12, 79: 70, 85: 124, 90: 66, 97: 11, 100: 36, 101: 105, 112: 130},
    4: {7: 68, 10: 170, 12: 156, 16: 167, 22: 77, 27: 38, 36: 62, 40: 163, 43: 139, 44: 185, 46: 53, 52: 10, 54: 56, 63: 98, 68: 74, 69: 174, 78: 15, 79: 104, 92: 23, 99: 160, 110: 111},
    5: {3: 146, 9: 9, 11: 119, 15: 57, 19: 51, 21: 141, 33: 168, 35: 43, 43: 154, 55: 33, 56: 78, 57: 128, 63: 129, 65: 2, 66: 107, 74: 61, 76: 127, 79: 134, 80: 91, 81: 165, 101: 176, 105: 182, 111: 158, 112: 150, 117: 183},
    6: {4: 55, 6: 35, 23: 147, 29: 153, 30: 135, 36: 52, 43: 4, 44: 29, 51: 16, 55: 131, 64: 25, 68: 59, 74: 58, 77: 1, 78: 159, 85: 145, 89: 47, 108: 89},
    7: {16: 177, 20: 8, 28: 151, 30: 189, 32: 166, 36: 193, 38: 187, 46: 175, 47: 108, 50: 179, 63: 39, 66: 155, 76: 13, 77: 172, 78: 24, 82: 148, 84: 92, 96: 199, 102: 67, 109: 102},
    8: {7: 126, 8: 0, 12: 73, 13: 99, 18: 71, 19: 101, 23: 96, 40: 162, 43: 120, 46: 69, 55: 133, 62: 84, 66: 79, 75: 50, 83: 18, 95: 194, 101: 30, 105: 190, 108: 100, 109: 117},
    9: {2: 169, 9: 32, 15: 181, 18: 22, 31: 87, 34: 196, 40: 26, 46: 72, 60: 136, 68: 97, 72: 173, 82: 27, 83: 83, 92: 60, 98: 140, 99: 144, 100: 122, 104: 85, 113: 171},
    11: {0: 103, 3: 116, 6: 44, 8: 195, 9: 86, 22: 31, 27: 110, 34: 138, 38: 93, 41: 164, 43: 81, 49: 54, 55: 157, 60: 192, 67: 48, 69: 7, 71: 14, 76: 88, 82: 37, 88: 34, 91: 152, 96: 63, 110: 161, 116: 186},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder to save calibration images into")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading '{DATASET_NAME}' (train split, revision {REVISION[:12]}) from Hugging Face...")
    fs = HfFileSystem()
    decoder = ImageFeature()
    saved = 0
    for file_index, rows in SOURCES.items():
        path = f"datasets/{DATASET_NAME}@{REVISION}/data/train-{file_index:05d}-of-00013.parquet"
        with fs.open(path, "rb") as f:
            parquet = pq.ParquetFile(f)
            group_start = 0
            for group in range(parquet.metadata.num_row_groups):
                group_rows = parquet.metadata.row_group(group).num_rows
                wanted = {r: n for r, n in rows.items() if group_start <= r < group_start + group_rows}
                if wanted:
                    table = parquet.read_row_group(group, columns=["image"]).to_pylist()
                    for row, number in wanted.items():
                        image = decoder.decode_example(table[row - group_start]["image"])
                        image.convert("RGB").save(args.output_dir / f"{number}.png")
                        saved += 1
                group_start += group_rows
    print(f"Saved {saved} calibration images to '{args.output_dir}/'.")

    listed = sorted(line.strip() for line in (HERE / "calib_list.txt").read_text().splitlines() if line.strip())
    present = sorted(p.name for p in args.output_dir.glob("*.png"))
    if present != listed:
        sys.exit(f"WARNING: '{args.output_dir}' has {len(present)} images that do not match the "
                 f"{len(listed)} names in calib_list.txt.")
    print(f"All {len(listed)} images listed in calib_list.txt are present.")


if __name__ == "__main__":
    main()
