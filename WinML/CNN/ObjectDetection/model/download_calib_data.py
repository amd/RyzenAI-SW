#!/usr/bin/env python3
"""
Download COCO calibration dataset for YOLOv8 quantization

This script downloads a subset of COCO validation images to use as
calibration data for INT8 quantization.

COCO Dataset: https://cocodataset.org/
"""

import sys
import os
import zipfile
import shutil
from pathlib import Path
import urllib.request


def download_coco_val_subset():
    """Download subset of COCO validation images for calibration"""

    model_dir = Path(__file__).parent
    calib_dir = model_dir / "calib_data"

    print("=" * 70)
    print("COCO Calibration Dataset Download")
    print("=" * 70)

    # Check if we already have images
    if calib_dir.exists():
        image_files = list(calib_dir.glob("**/*.jpg")) + list(calib_dir.glob("**/*.png"))
        if image_files and len(image_files) >= 100:
            print(f"\n[OK] Calibration data already exists: {len(image_files)} images")
            print(f"     Location: {calib_dir}")
            return 0

    calib_dir.mkdir(exist_ok=True)

    # Download a subset of COCO validation images
    print("\n[1/3] Downloading COCO validation subset...")
    print("       Downloading from Ultralytics datasets...")

    try:
        # Use Ultralytics datasets download
        from ultralytics.data.utils import download

        # Download COCO val2017 (subset)
        # This will download a manageable subset for calibration
        print("\n[INFO] Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

        print("\n[INFO] Downloading COCO val subset (this may take a few minutes)...")

        # Create a temporary directory for download
        temp_dir = model_dir / "temp_coco_download"
        temp_dir.mkdir(exist_ok=True)

        # Download using wget or urllib
        # Using a pre-selected subset of COCO val images
        coco_subset_url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"

        zip_path = temp_dir / "coco128.zip"

        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(downloaded * 100.0 / total_size, 100)
                sys.stdout.write(f'\r   Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB)')
                sys.stdout.flush()

        urllib.request.urlretrieve(coco_subset_url, zip_path, reporthook=download_progress)
        print("\n[OK] Download complete")

        # Extract images
        print("\n[2/3] Extracting calibration images...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of all image files in the images subdirectory
            image_members = [m for m in zip_ref.namelist()
                           if 'images' in m and m.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # Extract up to 200 images for calibration
            max_images = 200
            extracted_count = 0

            for member in image_members[:max_images]:
                # Extract to calib_dir, flattening directory structure
                filename = Path(member).name
                if filename:  # Skip directory entries
                    with zip_ref.open(member) as source:
                        with open(calib_dir / filename, 'wb') as target:
                            target.write(source.read())
                    extracted_count += 1
                    if extracted_count % 50 == 0:
                        print(f"   Extracted {extracted_count} images...")

            print(f"[OK] Extracted {extracted_count} images")

        # Clean up
        print("\n[3/3] Cleaning up...")
        shutil.rmtree(temp_dir)

        final_images = list(calib_dir.glob("*.jpg")) + list(calib_dir.glob("*.png"))
        print(f"[OK] Calibration dataset ready: {len(final_images)} images")
        print(f"     Location: {calib_dir}")

        return 0

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nAlternative: Manually download COCO validation images")
        print("1. Visit: https://cocodataset.org/#download")
        print("2. Download: 2017 Val images")
        print(f"3. Extract ~200 images to: {calib_dir}")
        print("\nOr download COCO128 subset:")
        print("1. Download: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip")
        print(f"2. Extract images/ folder to: {calib_dir}")

        # Try fallback to example images
        return create_calibration_from_examples()


def create_calibration_from_examples():
    """Fallback: Use example images if COCO unavailable"""

    model_dir = Path(__file__).parent
    calib_dir = model_dir / "calib_data"
    images_dir = model_dir.parent / "images"

    print("\n" + "=" * 70)
    print("[INFO] Attempting to create minimal calibration from example images...")
    print("=" * 70)

    calib_dir.mkdir(exist_ok=True)

    # Copy any example input images
    if images_dir.exists():
        example_images = [img for img in images_dir.glob("*.jpg")
                         if "input" in img.name.lower() or "output" not in img.name.lower()]

        if example_images:
            for img in example_images:
                dest = calib_dir / img.name
                shutil.copy(img, dest)

            final_count = len(list(calib_dir.glob("*.jpg")))
            print(f"\n[OK] Created minimal calibration data: {final_count} images")
            print(f"[WARNING] This is a VERY minimal dataset for testing only")
            print(f"          For production quantization, use COCO validation set")
            print(f"          (100-300 representative images recommended)")
            return 0

    print("\n[ERROR] No calibration images available")
    print("        Please download COCO manually or add images to calib_data/")
    return 1


if __name__ == "__main__":
    result = download_coco_val_subset()

    if result == 0:
        print("\n" + "=" * 70)
        print("SUCCESS: Calibration dataset ready")
        print("=" * 70)
        print("\nNext steps:")
        print("  python quantize_yolov8.py")
    else:
        print("\n" + "=" * 70)
        print("FAILED: Could not download calibration data")
        print("=" * 70)

    sys.exit(result)
