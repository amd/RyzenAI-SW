#!/usr/bin/env python3
"""
Download WIDER FACE calibration dataset for RetinaFace quantization

This script downloads a subset of WIDER FACE validation images to use as
calibration data for INT8 quantization.

WIDER FACE Dataset: http://shuoyang1213.me/WIDERFACE/
"""

import sys
import os
import zipfile
import shutil
from pathlib import Path
import urllib.request


def download_wider_face_val():
    """Download WIDER FACE validation images"""

    model_dir = Path(__file__).parent
    calib_dir = model_dir / "calib_data"

    print("=" * 70)
    print("WIDER FACE Calibration Dataset Download")
    print("=" * 70)

    # Check if we already have face images
    if calib_dir.exists():
        image_files = list(calib_dir.glob("**/*.jpg")) + list(calib_dir.glob("**/*.png"))
        if image_files and len(image_files) >= 50:
            print(f"\n[OK] Calibration data already exists: {len(image_files)} images")
            print(f"     Location: {calib_dir}")

            # Check if images are actually face images (WIDER FACE) not ImageNet
            sample_file = image_files[0]
            if "ILSVRC" in sample_file.name:
                print(f"\n[WARNING] Found ImageNet images, not face images")
                print(f"          Removing ImageNet calibration data...")
                shutil.rmtree(calib_dir)
                calib_dir.mkdir(exist_ok=True)
            else:
                return 0

    calib_dir.mkdir(exist_ok=True)

    # Download WIDER FACE validation images (compressed subset)
    print("\n[1/3] Downloading WIDER FACE validation subset...")
    print("       This may take several minutes (~200 MB)...")

    # Google Drive link for WIDER FACE validation images
    wider_url = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip"

    zip_path = model_dir / "WIDER_val.zip"

    try:
        # Download with progress
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100.0 / total_size, 100)
            sys.stdout.write(f'\r   Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB)')
            sys.stdout.flush()

        urllib.request.urlretrieve(wider_url, zip_path, reporthook=download_progress)
        print("\n[OK] Download complete")

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nAlternative: Manually download WIDER FACE validation set")
        print("1. Visit: http://shuoyang1213.me/WIDERFACE/")
        print("2. Download: WIDER Face Validation Images")
        print(f"3. Extract to: {calib_dir}")
        return 1

    # Extract subset of images (limit to ~300 images for calibration)
    print("\n[2/3] Extracting validation images...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of all image files
            image_members = [m for m in zip_ref.namelist() if m.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # Extract first 300 images
            max_images = 300
            extracted_count = 0

            for member in image_members[:max_images]:
                zip_ref.extract(member, calib_dir)
                extracted_count += 1
                if extracted_count % 50 == 0:
                    print(f"   Extracted {extracted_count} images...")

            print(f"[OK] Extracted {extracted_count} images")

        # Clean up zip file
        print("\n[3/3] Cleaning up...")
        zip_path.unlink()

        # Count final images
        final_images = list(calib_dir.glob("**/*.jpg")) + list(calib_dir.glob("**/*.png"))
        print(f"[OK] Calibration dataset ready: {len(final_images)} face images")
        print(f"     Location: {calib_dir}")

        return 0

    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return 1


def create_symlink_to_examples():
    """Alternative: Create symlinks to example images if WIDER FACE unavailable"""

    model_dir = Path(__file__).parent
    calib_dir = model_dir / "calib_data"
    images_dir = model_dir.parent / "images"

    print("\n[INFO] Creating calibration data from example images...")

    calib_dir.mkdir(exist_ok=True)

    # Copy example images
    example_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    if not example_images:
        print("[WARNING] No example images found")
        return 1

    for img in example_images:
        if "input" in img.name.lower() or "output" not in img.name.lower():
            dest = calib_dir / img.name
            shutil.copy(img, dest)

    final_count = len(list(calib_dir.glob("*.jpg")) + list(calib_dir.glob("*.png")))

    print(f"[OK] Created calibration data: {final_count} images")
    print(f"[WARNING] This is a minimal dataset for testing only")
    print(f"          For production, use WIDER FACE or similar face dataset")

    return 0


if __name__ == "__main__":
    result = download_wider_face_val()

    if result != 0:
        print("\n" + "=" * 70)
        print("Falling back to example images for minimal testing...")
        print("=" * 70)
        result = create_symlink_to_examples()

    if result == 0:
        print("\n" + "=" * 70)
        print("SUCCESS: Calibration dataset ready")
        print("=" * 70)
        print("\nNext steps:")
        print("  python quantize_retinaface.py")

    sys.exit(result)
