#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Prepare the ESRGAN calibration image folder (calibration_dataset/images) from DIV2K.

Draws 10 images from the DIV2K *train* HR split (0001-0800) with random.Random(42).sample() over
the sorted file names, so they never overlap the DIV2K validation images (0801-0900) used by the
accuracy test. Each image is turned into a model input the same way the accuracy test builds its
LR inputs: center-crop to 4x the model input size (1000x1000, bicubic upscale first if smaller),
then bicubic-downsample by 4 to 250x250.

Each image is saved under its DIV2K name (e.g. 0655.png). manifest.txt lists the files in
sampling order; olive_ryzen_data_scripts.py feeds them in that order, which affects the
quantization result.

DIV2K train HR is read from a folder with the 800 PNG files, from DIV2K_train_HR.zip, or, with
--download, from DIV2K_train_HR.zip on the DIV2K server (only the 10 selected images are read
with HTTP range requests, about 45 MB instead of 3.5 GB).

Usage:
    python prepare_calibration_dataset.py --div2k /path/to/DIV2K_train_HR (or DIV2K_train_HR.zip)
    python prepare_calibration_dataset.py --download
"""
import argparse
import io
import random
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

LR_SIZE = 250
SCALE = 4
COUNT = 10
SEED = 42
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "images"
DIV2K_TRAIN_HR_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP range requests."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--div2k", type=Path, help="DIV2K train HR folder (0001.png ...) or DIV2K_train_HR.zip")
    src.add_argument("--download", action="store_true", help="Read the selected images from the DIV2K server")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Calibration folder to write")
    return p.parse_args()


def open_source(args):
    """Returns (sorted PNG file names, function that returns one image as a binary file object)."""
    if args.div2k is not None and args.div2k.is_dir():
        return sorted(p.name for p in args.div2k.glob("*.png")), lambda name: open(args.div2k / name, "rb")
    zf = zipfile.ZipFile(HttpRangeFile(DIV2K_TRAIN_HR_URL) if args.download else args.div2k)
    members = {Path(m).name: m for m in zf.namelist() if m.lower().endswith(".png")}
    return sorted(members), lambda name: io.BytesIO(zf.read(members[name]))


def make_lr(fileobj):
    hr_size = LR_SIZE * SCALE
    img = Image.open(fileobj).convert("RGB")
    w, h = img.size
    if min(w, h) < hr_size:
        s = hr_size / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
        w, h = img.size
    left, top = (w - hr_size) // 2, (h - hr_size) // 2
    hr = img.crop((left, top, left + hr_size, top + hr_size))
    return hr.resize((LR_SIZE, LR_SIZE), Image.BICUBIC)


def main():
    args = parse_args()
    names, open_image = open_source(args)
    if len(names) != 800:
        raise SystemExit(f"Expected the 800 DIV2K train HR files 0001.png ... 0800.png, found {len(names)} PNG files")
    picked = random.Random(SEED).sample(names, COUNT)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("*.png"):
        stale.unlink()

    lines = [f"# ESRGAN calibration set: {COUNT} images extracted from DIV2K (train HR), in calibration order."]
    for name in picked:
        with open_image(name) as f:
            make_lr(f).save(args.output_dir / name)
        lines.append(name)
    (args.output_dir / "manifest.txt").write_text("\n".join(lines) + "\n", newline="\n")
    print(f"Saved {COUNT} calibration images to '{args.output_dir}/'.")


if __name__ == "__main__":
    main()
