#!/usr/bin/env python3
"""
Accuracy evaluation for RealESRGAN (x4) ONNX models using PSNR / SSIM.

Creates low-resolution / high-resolution pairs from the DIV2K validation
set (DIV2K_valid_HR.zip from the DIV2K website, 100 high-resolution images,
~2K pixels):
  1. Centre-crop each HR image to the model output size (4× the input size).
  2. Downsample the crop by 4× with bicubic to obtain the LR input.
  3. Run the ONNX model to produce a super-resolved (SR) image.
  4. Compare SR vs. HR using PSNR and SSIM on the Y channel and RGB.

Metrics reported:
  - PSNR (Y)   – Y-channel Peak Signal-to-Noise Ratio (dB)
  - SSIM (Y)   – Y-channel Structural Similarity Index
  - PSNR (RGB) – RGB Peak Signal-to-Noise Ratio (dB)
  - Avg latency (ms/image)

Usage:
    python esrgan_accuracy_test.py -m model.onnx                         # CPU EP
    python esrgan_accuracy_test.py -m model.onnx -n 50                   # fewer images
    python esrgan_accuracy_test.py -m model.onnx -e OV -d CPU            # OpenVINO on CPU
    python esrgan_accuracy_test.py -m model.onnx -e OV -d NPU            # OpenVINO on NPU
    python esrgan_accuracy_test.py -m model.onnx -e DML -d GPU           # DirectML on GPU
    python esrgan_accuracy_test.py -m model.onnx -e QNN -d NPU             # QNN on NPU
"""

import argparse
import platform
import time
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from ort_test_utils import (
    EP_FULL_TO_SHORT,
    add_common_args,
    create_session,
    detect_layout,
    get_input_hw,
    register_providers,
    resolve_ep_name,
    resolve_models,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCALE_FACTOR = 4  # RealESRGAN x4
DIV2K_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
DIV2K_NUM_IMAGES = 100  # validation set: images 0801–0900


# ---------------------------------------------------------------------------
# CoreML support (macOS)
# ---------------------------------------------------------------------------

def is_coreml_model(model_path: Path) -> bool:
    """Return True if the path points to a CoreML model package/file."""
    return model_path.suffix.lower() in {".mlpackage", ".mlmodel"}


def load_coreml_model(model_path: Path):
    """Load a CoreML model and return (model, input_name, layout, input_shape)."""
    if platform.system() != "Darwin":
        raise RuntimeError("CoreML inference is only supported on macOS.")

    try:
        import coremltools as ct
    except ImportError as e:
        raise RuntimeError(
            "coremltools is required for CoreML model evaluation. "
            "Install it with: pip install coremltools==9.0"
        ) from e

    model = ct.models.MLModel(str(model_path))
    spec = model.get_spec()

    if not spec.description.input:
        raise RuntimeError("CoreML model has no inputs.")

    input_desc = spec.description.input[0]
    input_name = input_desc.name
    shape = tuple(input_desc.type.multiArrayType.shape)

    if len(shape) != 4:
        raise RuntimeError(f"Unsupported CoreML input shape {shape}; expected rank-4 image tensor.")

    if shape[1] == 3:
        layout = "NCHW"
    elif shape[3] == 3:
        layout = "NHWC"
    else:
        raise RuntimeError(f"Unable to infer CoreML input layout from shape {shape}.")

    return model, input_name, layout, shape


def run_coreml_inference(coreml_model, input_name: str, inp: np.ndarray) -> np.ndarray:
    """Run one CoreML inference and return the super-resolved output array."""
    outputs = coreml_model.predict({input_name: inp})
    if not outputs:
        raise RuntimeError("CoreML inference produced no outputs.")
    return np.asarray(next(iter(outputs.values())))


# ---------------------------------------------------------------------------
# Image-quality metrics
# ---------------------------------------------------------------------------

def _rgb_to_y(img: np.ndarray) -> np.ndarray:
    """Convert HWC uint8 RGB image to Y channel (ITU-R BT.601).

    Returns a float64 2-D array in [0, 255].
    """
    r, g, b = img[:, :, 0].astype(np.float64), img[:, :, 1].astype(np.float64), img[:, :, 2].astype(np.float64)
    return 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """PSNR between two uint8 images (HWC or 2-D). Returns dB."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """SSIM between two 2-D uint8 images (Y channel).

    Uses the simplified formulation with default constants
    (K1=0.01, K2=0.03, L=255).  Computed per-pixel with an
    11×11 uniform window, matching the common SR evaluation protocol.
    """
    from scipy.ndimage import uniform_filter

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = uniform_filter(img1, size=11)
    mu2 = uniform_filter(img2, size=11)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(img1 * img1, size=11) - mu1_sq
    sigma2_sq = uniform_filter(img2 * img2, size=11) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=11) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def prepare_pair(
    pil_img: Image.Image,
    lr_h: int,
    lr_w: int,
    layout: str = "NCHW",
) -> tuple[np.ndarray, np.ndarray]:
    """Create an LR input tensor and an HR reference uint8 image.

    Steps:
        1. Convert to RGB.
        2. Resize so that the shortest side >= HR size, then centre-crop
           to exactly (hr_h, hr_w) = (4*lr_h, 4*lr_w).
        3. Downsample the HR crop by 4× (bicubic) → LR image.
        4. Normalise LR to [0, 1] float32; arrange as NCHW or NHWC.

    Returns:
        lr_tensor : np.ndarray, shape (1, 3, lr_h, lr_w) or (1, lr_h, lr_w, 3)
        hr_uint8  : np.ndarray, shape (hr_h, hr_w, 3), dtype uint8
    """
    hr_h, hr_w = lr_h * SCALE_FACTOR, lr_w * SCALE_FACTOR
    img = pil_img.convert("RGB")
    w, h = img.size

    # Resize so shortest side >= max(hr_h, hr_w), keeping aspect
    target = max(hr_h, hr_w)
    short = min(w, h)
    if short < target:
        scale = target / short
        img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BICUBIC)
        w, h = img.size

    # Centre-crop to (hr_w, hr_h)
    left = (w - hr_w) // 2
    top = (h - hr_h) // 2
    hr_img = img.crop((left, top, left + hr_w, top + hr_h))
    hr_uint8 = np.asarray(hr_img, dtype=np.uint8)  # (hr_h, hr_w, 3)

    # Downsample by 4× to get LR
    lr_img = hr_img.resize((lr_w, lr_h), Image.BICUBIC)
    lr_arr = np.asarray(lr_img, dtype=np.float32) / 255.0  # HWC, [0, 1]

    if layout == "NHWC":
        lr_tensor = lr_arr[np.newaxis, ...]          # (1, lr_h, lr_w, 3)
    else:
        lr_tensor = lr_arr.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, lr_h, lr_w)

    return lr_tensor, hr_uint8


def postprocess_sr(output: np.ndarray) -> np.ndarray:
    """Convert model output to HWC uint8 RGB image.

    Detects the output layout from the tensor shape (the channel
    dimension is 3, so whichever axis equals 3 after removing the
    batch dimension determines NCHW vs NHWC).

    Clamps to [0, 1], converts to uint8.
    """
    if output.ndim == 4:
        output = output[0]  # remove batch dim

    # Detect layout from output shape: (3, H, W) → NCHW, (H, W, 3) → NHWC
    if output.shape[0] == 3:
        output = output.transpose(1, 2, 0)  # CHW → HWC

    output = np.clip(output, 0.0, 1.0)
    return (output * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_images(num_samples: int, seed: int = 42) -> list:
    """Load HR images from the DIV2K validation set.

    Downloads the 100 HR validation images (0801–0900) from the official
    ETH Zürich server on first use and caches them locally under
    ``~/.cache/esrgan_accuracy_test/DIV2K_valid_HR/``.
    """
    cache_dir = Path.home() / ".cache" / "esrgan_accuracy_test" / "DIV2K_valid_HR"

    if not cache_dir.exists() or not any(cache_dir.glob("*.png")):
        zip_path = cache_dir.parent / "DIV2K_valid_HR.zip"
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading DIV2K validation HR images from {DIV2K_URL} …")
        urllib.request.urlretrieve(DIV2K_URL, str(zip_path))
        print(f"Extracting to {cache_dir} …")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(cache_dir.parent))
        zip_path.unlink(missing_ok=True)
        print("  Download complete.")

    img_files = sorted(cache_dir.glob("*.png"))
    if not img_files:
        raise FileNotFoundError(
            f"No PNG files found in {cache_dir}. "
            f"Try deleting the cache directory and re-running."
        )

    total = min(num_samples, len(img_files))
    print(f"Loading {total} DIV2K validation HR images from {cache_dir} …")
    images = []
    for i in range(total):
        images.append(Image.open(img_files[i]).convert("RGB"))
        if (i + 1) % 50 == 0:
            print(f"  loaded {i + 1}/{total} images")

    print(f"  loaded {len(images)} images total")
    return images


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    model_path: Path,
    images: list,
    ep: str,
    device: str = "CPU",
):
    """Evaluate a RealESRGAN ONNX/CoreML model.  Returns dict with metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}  ({model_path.name})")
    runtime = "CoreML" if is_coreml_model(model_path) else "ONNXRuntime"
    if runtime == "CoreML":
        print("  Runtime: CoreML")
    else:
        print(f"  EP: {ep}  Device: {device}")
    print(f"{'='*60}")

    if not model_path.exists():
        print(f"  ** SKIPPED – model file not found: {model_path}")
        return None

    coreml_model = None
    session = None
    if runtime == "CoreML":
        try:
            coreml_model, input_name, layout, shape = load_coreml_model(model_path)
        except Exception as e:
            print(f"  ** SKIPPED – failed to load CoreML model: {e}")
            return None
        input_shape = shape
        if layout == "NCHW":
            lr_h, lr_w = int(shape[2]), int(shape[3])
        else:
            lr_h, lr_w = int(shape[1]), int(shape[2])
    else:
        try:
            session = create_session(str(model_path), ep, device)
        except Exception as e:
            print(f"  ** SKIPPED – failed to load model: {e}")
            return None

        layout = detect_layout(session)
        lr_h, lr_w = get_input_hw(session)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape

    hr_h, hr_w = lr_h * SCALE_FACTOR, lr_w * SCALE_FACTOR

    print(f"  Input layout : {layout}  (shape: {input_shape})")
    print(f"  LR size      : {lr_w}×{lr_h}")
    print(f"  HR size (×{SCALE_FACTOR}) : {hr_w}×{hr_h}")

    psnr_y_list = []
    psnr_rgb_list = []
    ssim_y_list = []
    elapsed_ms = []
    total = len(images)
    skipped = 0

    for idx, pil_img in enumerate(images):
        pil_img = pil_img.convert("RGB")
        w, h = pil_img.size

        # Skip images that are too small for the required HR crop
        if w < hr_w or h < hr_h:
            skipped += 1
            continue

        lr_tensor, hr_uint8 = prepare_pair(pil_img, lr_h, lr_w, layout=layout)

        t0 = time.perf_counter()
        if runtime == "CoreML":
            sr_output = run_coreml_inference(coreml_model, input_name, lr_tensor)
        else:
            sr_output = session.run([output_name], {input_name: lr_tensor})[0]
        t1 = time.perf_counter()
        elapsed_ms.append((t1 - t0) * 1000)

        sr_uint8 = postprocess_sr(sr_output)

        # Crop border (standard in SR evaluation: ignore SCALE_FACTOR pixels at borders)
        crop = SCALE_FACTOR
        hr_crop = hr_uint8[crop:-crop, crop:-crop, :]
        sr_crop = sr_uint8[crop:-crop, crop:-crop, :]

        # PSNR (RGB)
        psnr_rgb = compute_psnr(sr_crop, hr_crop)
        psnr_rgb_list.append(psnr_rgb)

        # PSNR and SSIM on Y channel
        hr_y = _rgb_to_y(hr_crop)
        sr_y = _rgb_to_y(sr_crop)
        # Quantise Y to uint8 for consistent comparison
        hr_y_u8 = np.clip(hr_y + 0.5, 0, 255).astype(np.uint8)
        sr_y_u8 = np.clip(sr_y + 0.5, 0, 255).astype(np.uint8)

        psnr_y = compute_psnr(sr_y_u8, hr_y_u8)
        ssim_y = compute_ssim(sr_y_u8, hr_y_u8)
        psnr_y_list.append(psnr_y)
        ssim_y_list.append(ssim_y)

        evaluated = len(psnr_y_list)
        if evaluated % 200 == 0:
            print(
                f"  [{evaluated:>5}/{total}]  "
                f"PSNR(Y) {np.mean(psnr_y_list):5.2f} dB  "
                f"SSIM(Y) {np.mean(ssim_y_list):.4f}  "
                f"PSNR(RGB) {np.mean(psnr_rgb_list):5.2f} dB  "
                f"avg {np.mean(elapsed_ms):6.2f} ms/img"
            )

    if not psnr_y_list:
        print("  ** No images evaluated (all too small?)")
        return None

    avg_psnr_y = float(np.mean(psnr_y_list))
    avg_ssim_y = float(np.mean(ssim_y_list))
    avg_psnr_rgb = float(np.mean(psnr_rgb_list))
    avg_ms = float(np.mean(elapsed_ms))

    print(f"\n  Results for {model_name}:")
    print(f"    PSNR (Y)       : {avg_psnr_y:6.2f} dB")
    print(f"    SSIM (Y)       : {avg_ssim_y:.4f}")
    print(f"    PSNR (RGB)     : {avg_psnr_rgb:6.2f} dB")
    print(f"    Avg latency    : {avg_ms:6.2f} ms/image")
    print(f"    Evaluated      : {len(psnr_y_list)}")
    if skipped:
        print(f"    Skipped (small): {skipped}")

    return {
        "model": model_name,
        "ep": "CoreML" if runtime == "CoreML" else ep,
        "device": "NE" if runtime == "CoreML" else device,
        "psnr_y": avg_psnr_y,
        "ssim_y": avg_ssim_y,
        "psnr_rgb": avg_psnr_rgb,
        "avg_ms": avg_ms,
        "total": len(psnr_y_list),
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list):
    """Print a comparison table of all evaluated models."""
    print(f"\n{'='*100}")
    print("SUPER-RESOLUTION ACCURACY SUMMARY  (dataset: DIV2K val)")
    print(f"{'='*100}")
    hdr = (
        f"{'Model':<25} {'EP':<10} {'Dev':<4} "
        f"{'PSNR(Y)':>8} {'SSIM(Y)':>8} {'PSNR(RGB)':>9} "
        f"{'ms/img':>7} {'Imgs':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ep_short = EP_FULL_TO_SHORT.get(r["ep"], r["ep"])
        print(
            f"{r['model']:<25} {ep_short:<10} {r['device']:<4} "
            f"{r['psnr_y']:>8.2f} {r['ssim_y']:>8.4f} {r['psnr_rgb']:>9.2f} "
            f"{r['avg_ms']:>7.1f} {r['total']:>5}"
        )
    print(f"{'='*100}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate RealESRGAN ONNX models (PSNR / SSIM)"
    )
    add_common_args(
        p,
        default_num_samples=100,
        num_samples_help="Number of validation images to use (DIV2K val has 100)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Register EPs via WinML
    print("Registering available EPs...")
    register_providers(verbose=True)

    # Resolve the requested EP
    ep = resolve_ep_name(args.ep)
    device = args.device

    # Build the list of (name, path) pairs to evaluate
    model_items = resolve_models(args)

    print("Models to evaluate:", [name for name, _ in model_items])
    print(f"Validation images : {args.num_samples}")
    print(f"EP: {args.ep} ({ep})")
    print(f"Device: {device}")

    # Load dataset once, reuse for every model
    images = load_images(args.num_samples)

    results = []
    for name, path in model_items:
        r = evaluate_model(
            name, path, images,
            ep=ep, device=device,
        )
        if r is not None:
            results.append(r)

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
