#!/usr/bin/env python3
"""
Setup calibration data for ResNet50 INT8 quantization

Downloads a subset of COCO128 images (freely available, no authentication required)
for use as calibration data.

Usage:
    python setup_calib_data.py                        # download ~128 COCO images (default)
    python setup_calib_data.py --num-images 200       # download more images
    python setup_calib_data.py --source local --val-dir /path/to/ILSVRC2012_val
"""

import argparse
import sys
import zipfile
import shutil
import urllib.request
from pathlib import Path


MODEL_DIR = Path(__file__).parent
CALIB_DIR = MODEL_DIR / "calib_data"

COCO128_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"


def count_images(directory):
    count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPEG']:
        count += len(list(directory.glob(ext)))
    return count


def download_coco128(num_images):
    """
    Download COCO128 subset from Ultralytics GitHub release.
    No authentication required.
    """
    print(f"\n[INFO] Downloading COCO128 calibration images...")
    print(f"       Source: {COCO128_URL}")

    temp_dir = MODEL_DIR / "temp_coco_download"
    temp_dir.mkdir(exist_ok=True)
    zip_path = temp_dir / "coco128.zip"

    try:
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(downloaded * 100.0 / total_size, 100)
                sys.stdout.write(f'\r   Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB)')
                sys.stdout.flush()

        urllib.request.urlretrieve(COCO128_URL, zip_path, reporthook=download_progress)
        print("\n[OK] Download complete")

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False

    print(f"\n[INFO] Extracting calibration images...")

    CALIB_DIR.mkdir(exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            image_members = [
                m for m in zip_ref.namelist()
                if 'images' in m and m.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            extracted = 0
            for member in image_members[:num_images]:
                filename = Path(member).name
                if filename:
                    with zip_ref.open(member) as src:
                        with open(CALIB_DIR / filename, 'wb') as dst:
                            dst.write(src.read())
                    extracted += 1
                    if extracted % 50 == 0:
                        print(f"   Extracted {extracted} images...")

        print(f"[OK] Extracted {extracted} images")

    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False

    shutil.rmtree(temp_dir, ignore_errors=True)
    return True


def copy_from_local_ilsvrc(val_dir, num_images):
    """
    Copy images from a locally downloaded ILSVRC2012 validation set.
    Selects images evenly across class folders for representative coverage.
    """
    val_path = Path(val_dir)
    if not val_path.exists():
        print(f"[ERROR] Validation directory not found: {val_path}")
        return False

    CALIB_DIR.mkdir(exist_ok=True)

    all_images = []
    subfolders = [d for d in val_path.iterdir() if d.is_dir()]

    if subfolders:
        # Class-subfolder layout (n01440764/ILSVRC2012_val_*.JPEG)
        for folder in sorted(subfolders):
            imgs = list(folder.glob('*.JPEG')) + list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
            if imgs:
                all_images.append(imgs[0])
    else:
        for ext in ['*.JPEG', '*.jpg', '*.png']:
            all_images.extend(val_path.glob(ext))

    if not all_images:
        print(f"[ERROR] No images found in: {val_path}")
        return False

    import random
    random.seed(42)
    selected = random.sample(all_images, min(num_images, len(all_images)))

    print(f"\n[INFO] Copying {len(selected)} images from {val_path}")
    for i, src in enumerate(selected):
        dst = CALIB_DIR / f"imagenet_val_{i:06d}{src.suffix}"
        shutil.copy(src, dst)

    print(f"[OK] Copied {len(selected)} images to {CALIB_DIR}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download calibration data for ResNet50 quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download COCO128 images (default, no authentication needed)
  python setup_calib_data.py

  # Download more images
  python setup_calib_data.py --num-images 200

  # Copy from a local ILSVRC2012 validation directory
  python setup_calib_data.py --source local --val-dir D:/datasets/ILSVRC2012/val
        """
    )

    parser.add_argument(
        '--num-images', type=int, default=128,
        help='Number of calibration images to download/copy (default: 128)'
    )
    parser.add_argument(
        '--source', type=str, default='coco128', choices=['coco128', 'local'],
        help='Image source: "coco128" (default, no auth) or "local" (ILSVRC2012)'
    )
    parser.add_argument(
        '--val-dir', type=str, default=None,
        help='Path to local ILSVRC2012 validation directory (required when --source=local)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-download even if calibration data already exists'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ResNet50 Calibration Data Setup")
    print("=" * 70)

    if CALIB_DIR.exists():
        existing = count_images(CALIB_DIR)
        if existing > 0 and not args.force:
            print(f"\n[INFO] Calibration data already exists: {existing} images in {CALIB_DIR}")
            print(f"       Use --force to re-download.")
            print("\nNext step:")
            print("  python quantize_resnet.py")
            return 0
        elif args.force:
            print(f"\n[INFO] --force specified, clearing existing {existing} images...")
            shutil.rmtree(CALIB_DIR)

    success = False

    if args.source == 'local':
        if not args.val_dir:
            print("[ERROR] --val-dir is required when using --source=local")
            return 1
        success = copy_from_local_ilsvrc(args.val_dir, args.num_images)
    else:
        success = download_coco128(args.num_images)

    if not success:
        print("\n[ERROR] Failed to set up calibration data.")
        print("\nAlternative: provide images manually:")
        print(f"  mkdir {CALIB_DIR}")
        print(f"  # Copy 100-200 diverse JPEG/PNG images into {CALIB_DIR}")
        print(f"  # Then run: python quantize_resnet.py")
        return 1

    final_count = count_images(CALIB_DIR)
    print(f"\n{'=' * 70}")
    print(f"Setup complete: {final_count} calibration images in {CALIB_DIR}")
    print(f"{'=' * 70}")

    if final_count < 100:
        print(f"\n[NOTE] {final_count} images may give adequate calibration but 100+ is recommended.")

    print(f"\nNext step:")
    print(f"  python quantize_resnet.py")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
