#!/usr/bin/env python3
"""
DeepLabV3 Semantic Image Segmentation Example
Demonstrates pixel-level scene understanding using Windows ML with NPU acceleration.

Model:   DeepLabV3 with ResNet50 backbone (torchvision)
Dataset: PASCAL VOC 2012 (21 semantic classes)
Input:   [1, 3, 520, 520] NCHW, ImageNet-normalized
Output:  [1, 21, 520, 520] per-pixel class logits -> argmax -> segmentation map
"""

import argparse
import sys
import os
import time
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not installed. Run: pip install onnxruntime")
    sys.exit(1)

# ---------------------------------------------------------------------------
# PASCAL VOC 2012 class definitions
# ---------------------------------------------------------------------------
VOC_CLASSES = [
    "background",    # 0
    "aeroplane",     # 1
    "bicycle",       # 2
    "bird",          # 3
    "boat",          # 4
    "bottle",        # 5
    "bus",           # 6
    "car",           # 7
    "cat",           # 8
    "chair",         # 9
    "cow",           # 10
    "dining table",  # 11
    "dog",           # 12
    "horse",         # 13
    "motorbike",     # 14
    "person",        # 15
    "potted plant",  # 16
    "sheep",         # 17
    "sofa",          # 18
    "train",         # 19
    "tv/monitor",    # 20
]

# Visually distinct BGR colors per class (OpenCV uses BGR)
VOC_COLORS_BGR = np.array([
    [128, 128, 128],  # 0  background     - gray
    [128,   0,   0],  # 1  aeroplane      - dark red
    [  0, 128,   0],  # 2  bicycle        - dark green
    [128, 128,   0],  # 3  bird           - olive
    [  0,   0, 128],  # 4  boat           - dark blue
    [128,   0, 128],  # 5  bottle         - purple
    [  0, 128, 128],  # 6  bus            - teal
    [128, 128, 128],  # 7  car            - silver (reuse)
    [  0,  64, 128],  # 8  cat            - slate blue
    [192, 128, 128],  # 9  chair          - rose
    [ 64, 128,   0],  # 10 cow            - yellow-green
    [ 64,   0, 128],  # 11 dining table   - indigo
    [192,   0,   0],  # 12 dog            - red
    [192,   0, 128],  # 13 horse          - magenta-red
    [ 64, 128, 128],  # 14 motorbike      - steel blue
    [192, 128,   0],  # 15 person         - amber
    [ 64,   0,   0],  # 16 potted plant   - maroon
    [192,   0, 128],  # 17 sheep          - violet-red
    [ 64, 128,   0],  # 18 sofa           - lime (reuse)
    [  0,  64,   0],  # 19 train          - forest green
    [  0, 192,   0],  # 20 tv/monitor     - bright green
], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Execution provider registration
# ---------------------------------------------------------------------------
def register_execution_providers():
    """Register Windows ML VitisAI execution provider"""
    try:
        result = subprocess.check_output(
            [sys.executable, "winml_worker.py"], text=True
        )
        ep_paths = json.loads(result)
        for name, lib_path in ep_paths.items():
            if lib_path and Path(lib_path).exists():
                ort.register_execution_provider_library(name, lib_path)
                print(f"Registered: {name}")
    except Exception as e:
        print(f"Warning: Could not register execution providers: {e}")
        print("Falling back to CPU execution")


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
def preprocess_image(image_path: str, input_size: tuple = (520, 520)):
    """
    Preprocess image for DeepLabV3 model.

    Steps:
      1. Load RGB image
      2. Resize to input_size (bilinear)
      3. Normalize with ImageNet mean/std
      4. Transpose HWC -> CHW
      5. Add batch dimension -> NCHW

    Returns:
        input_tensor: float32 array [1, 3, H, W]
        original_size: (W, H) of the original image
    """
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (W, H)

    img_resized = img.resize(input_size, Image.BILINEAR)
    img_array = np.array(img_resized).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    img_array = img_array.transpose((2, 0, 1))       # HWC -> CHW
    img_array = np.expand_dims(img_array, axis=0)    # -> NCHW

    return img_array.astype(np.float32), original_size


# ---------------------------------------------------------------------------
# Post-processing & visualization
# ---------------------------------------------------------------------------
def postprocess_segmentation(
    output: np.ndarray,
    original_image_path: str,
    output_path: str,
    original_size: tuple,
    alpha: float = 0.55,
) -> dict:
    """
    Convert model output to a color segmentation map and save visualization.

    Args:
        output:             Model output [1, 21, 520, 520] logits
        original_image_path: Path to original input image
        output_path:        Path to save the overlay visualization
        original_size:      (W, H) of original image for resizing back
        alpha:              Blend factor for overlay (0=original, 1=mask only)

    Returns:
        dict with class counts and present class names
    """
    # argmax over class dimension -> [1, H, W] -> [H, W]
    seg_map = np.argmax(output[0], axis=0).astype(np.uint8)

    # Build color mask from VOC palette
    color_mask = VOC_COLORS_BGR[seg_map]  # [H, W, 3] BGR

    # Resize both mask and original back to original resolution
    orig_w, orig_h = original_size
    color_mask_resized = cv2.resize(
        color_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
    )

    original_bgr = cv2.imread(original_image_path)
    if original_bgr is None:
        # Fallback: use a white canvas
        original_bgr = np.full((orig_h, orig_w, 3), 255, dtype=np.uint8)
    else:
        original_bgr = cv2.resize(original_bgr, (orig_w, orig_h))

    # Blend original image with color mask
    overlay = cv2.addWeighted(original_bgr, 1.0 - alpha,
                               color_mask_resized, alpha, 0)

    # --- Legend: list detected classes in bottom-left corner ---
    present_ids = sorted(np.unique(seg_map).tolist())
    present_names = [VOC_CLASSES[i] for i in present_ids]

    # Background color behind legend
    legend_x, legend_y = 10, orig_h - (len(present_ids) * 22 + 10)
    legend_y = max(legend_y, 5)

    for idx, class_id in enumerate(present_ids):
        name  = VOC_CLASSES[class_id]
        color = tuple(int(c) for c in VOC_COLORS_BGR[class_id])
        y_pos = legend_y + idx * 22

        # Color swatch
        cv2.rectangle(overlay, (legend_x, y_pos),
                      (legend_x + 16, y_pos + 16), color, -1)
        cv2.rectangle(overlay, (legend_x, y_pos),
                      (legend_x + 16, y_pos + 16), (0, 0, 0), 1)

        # Class label
        cv2.putText(overlay, f"{class_id}: {name}",
                    (legend_x + 22, y_pos + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(overlay, f"{class_id}: {name}",
                    (legend_x + 22, y_pos + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)
    print(f"Segmentation output saved to: {output_path}")

    return {"class_ids": present_ids, "class_names": present_names}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    model_path: str,
    image_path: str,
    ep_policy: str = "NPU",
    output_dir: str = "../images",
    verbose: bool = False,
):
    """Run semantic segmentation inference"""

    print(f"\nModel:     {Path(model_path).name}")
    print(f"Image:     {Path(image_path).name}")
    print(f"EP Policy: {ep_policy}\n")

    register_execution_providers()

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 1  # Errors only

    if hasattr(ort, "OrtExecutionProviderDevicePolicy"):
        if ep_policy == "NPU":
            sess_options.set_provider_selection_policy(
                ort.OrtExecutionProviderDevicePolicy.PREFER_NPU
            )
        else:
            sess_options.set_provider_selection_policy(
                ort.OrtExecutionProviderDevicePolicy.PREFER_CPU
            )

    cache_dir = os.path.abspath("cache_dir")
    provider_options = [{
        "cache_dir": str(cache_dir),
        "cache_key": "deeplabv3_modelcachekey",
        "enable_cache_file_io_in_mem": "0",
    }]

    print("Creating inference session (first run compiles model for NPU ~1-3 min)...")
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        provider_options=provider_options,
    )

    providers = session.get_providers()
    print(f"Execution Providers: {providers}")

    input_name  = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input:  {input_name} {input_shape}")

    output_name  = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape
    print(f"Output: {output_name} {output_shape}")

    print("\nPreprocessing image...")
    input_tensor, original_size = preprocess_image(image_path)
    print(f"  Original size: {original_size[0]}x{original_size[1]}")
    print(f"  Input tensor:  {input_tensor.shape}")

    print("\nWarming up (3 iterations)...")
    for _ in range(3):
        session.run(None, {input_name: input_tensor})

    print("Benchmarking (20 iterations)...")
    iterations = 20
    t0 = time.perf_counter()
    for _ in range(iterations):
        outputs = session.run(None, {input_name: input_tensor})
    t1 = time.perf_counter()

    avg_latency_ms = (t1 - t0) / iterations * 1000
    throughput = 1000 / avg_latency_ms

    print(f"\nPerformance:")
    print(f"  Average latency: {avg_latency_ms:.2f} ms")
    print(f"  Throughput:      {throughput:.2f} images/sec")

    if verbose:
        raw = outputs[0]
        print(f"\nOutput stats: shape={raw.shape}, "
              f"min={raw.min():.3f}, max={raw.max():.3f}")

    print("\nGenerating segmentation visualization...")
    output_path = os.path.join(
        output_dir,
        f"segmentation_{ep_policy.lower()}_output.png"
    )
    result = postprocess_segmentation(
        outputs[0], image_path, output_path, original_size
    )

    print(f"\nDetected {len(result['class_ids'])} classes:")
    for cid, cname in zip(result["class_ids"], result["class_names"]):
        print(f"  [{cid:2d}] {cname}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="DeepLabV3 Semantic Segmentation with Windows ML"
    )
    parser.add_argument(
        "--model",
        default="../model/deeplabv3_resnet50.onnx",
        help="Path to ONNX model (default: ../model/deeplabv3_resnet50.onnx)",
    )
    parser.add_argument(
        "--image",
        default="../images/image_segmentation_input.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--ep_policy",
        default="NPU",
        choices=["NPU", "CPU"],
        help="Execution provider policy (default: NPU)",
    )
    parser.add_argument(
        "--output_dir",
        default="../images",
        help="Directory to save output images (default: ../images)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        print("Run: python ../model/download_model.py")
        sys.exit(1)

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        print("Provide a JPEG/PNG image with PASCAL VOC objects")
        print("(people, vehicles, animals, furniture, etc.)")
        sys.exit(1)

    run_inference(
        args.model,
        args.image,
        args.ep_policy,
        args.output_dir,
        args.verbose,
    )


if __name__ == "__main__":
    main()
