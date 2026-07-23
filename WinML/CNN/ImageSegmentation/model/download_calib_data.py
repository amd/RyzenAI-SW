#!/usr/bin/env python3
"""
Download PASCAL VOC 2012 calibration images for DeepLabV3 quantization.

Downloads a subset of PASCAL VOC 2012 validation images to use as
calibration data for INT8 quantization. These images cover all 21
semantic classes the model was trained on.

PASCAL VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
"""

import sys
import shutil
import tarfile
import urllib.request
from pathlib import Path


# VOC 2012 validation image list hosted on a public mirror
VOC_VAL_URL = (
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
)
# Lighter alternative: download just the JPEGImages via torchvision
TORCHVISION_FALLBACK = True


def download_via_torchvision(calib_dir: Path, max_images: int = 200) -> int:
    """Download VOC validation images using torchvision (lighter approach)."""
    try:
        import torchvision.datasets as dsets
        import torchvision.transforms as T
    except ImportError:
        print("[ERROR] torchvision not installed. Run: pip install torchvision")
        return 1

    voc_root = calib_dir.parent / "voc_download"
    print(f"\n[1/3] Downloading PASCAL VOC 2012 validation set via torchvision...")
    print(f"      This may take several minutes (~2 GB download)...")

    try:
        dataset = dsets.VOCSegmentation(
            root=str(voc_root),
            year="2012",
            image_set="val",
            download=True,
            transform=None,
        )
        print(f"[OK]  Downloaded {len(dataset)} validation images")
    except Exception as e:
        print(f"[ERROR] torchvision download failed: {e}")
        return 1

    # Copy a flat subset of JPEG images to calib_data/
    print(f"\n[2/3] Copying up to {max_images} images to calibration directory...")
    calib_dir.mkdir(exist_ok=True)

    copied = 0
    images_root = voc_root / "VOCdevkit" / "VOC2012" / "JPEGImages"
    if not images_root.exists():
        # torchvision may place them differently
        candidates = list(voc_root.rglob("*.jpg"))
        images_root = None
    else:
        candidates = sorted(images_root.glob("*.jpg"))

    for src in candidates[:max_images]:
        shutil.copy(src, calib_dir / src.name)
        copied += 1

    print(f"[OK]  Copied {copied} images to {calib_dir}")

    # Clean up large download directory
    print("\n[3/3] Cleaning up full VOC download...")
    shutil.rmtree(voc_root, ignore_errors=True)
    print("[OK]  Cleanup complete")

    return 0 if copied > 0 else 1


def copy_from_images_dir(calib_dir: Path) -> int:
    """Fallback: copy the example input images for minimal testing."""
    images_dir = Path(__file__).parent.parent / "images"
    calib_dir.mkdir(exist_ok=True)

    examples = [
        p for p in images_dir.glob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        and "output" not in p.name.lower()
        and p.name != "download_sample_image.py"
    ]

    if not examples:
        print("[WARNING] No example images found in images/ directory")
        return 1

    for src in examples:
        shutil.copy(src, calib_dir / src.name)

    count = len(list(calib_dir.glob("*.jpg")) + list(calib_dir.glob("*.png")))
    print(f"[OK]  Copied {count} example images to {calib_dir}")
    print("[WARNING] This is a minimal dataset for testing only.")
    print("          For production quantization, use real PASCAL VOC images.")
    return 0


def main():
    model_dir = Path(__file__).parent
    calib_dir = model_dir / "calib_data"

    print("=" * 70)
    print("PASCAL VOC 2012 Calibration Dataset Download")
    print("=" * 70)

    # Check if we already have enough images
    if calib_dir.exists():
        existing = list(calib_dir.glob("**/*.jpg")) + list(calib_dir.glob("**/*.png"))
        if len(existing) >= 50:
            print(f"\n[OK] Calibration data already exists: {len(existing)} images")
            print(f"     Location: {calib_dir}")
            return 0

    result = download_via_torchvision(calib_dir, max_images=200)

    if result != 0:
        print("\n" + "=" * 70)
        print("Falling back to example images for minimal testing...")
        print("=" * 70)
        result = copy_from_images_dir(calib_dir)

    if result == 0:
        final = list(calib_dir.glob("**/*.jpg")) + list(calib_dir.glob("**/*.png"))
        print("\n" + "=" * 70)
        print("SUCCESS: Calibration dataset ready")
        print("=" * 70)
        print(f"  Images:   {len(final)}")
        print(f"  Location: {calib_dir}")
        print("\nNext step:")
        print("  python quantize_deeplabv3.py")

    return result


if __name__ == "__main__":
    sys.exit(main())
