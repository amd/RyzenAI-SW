#!/usr/bin/env python3
"""
Accuracy evaluation for ConvNeXt Tiny ONNX models against ImageNet-1K validation set.

Uses evanarlian/imagenet_1k_resized_256 from Hugging Face, which contains
the full 50,000-image ImageNet validation set (Parquet format, no gated access).

Metrics reported:
  - Top-1 accuracy
  - Top-5 accuracy

Usage:
    python convnext_accuracy_test.py -m model.onnx                       # CPU EP, CPU device
    python convnext_accuracy_test.py -m model.mlpackage                  # CoreML on macOS
    python convnext_accuracy_test.py -m model.onnx -n 5000               # more images
    python convnext_accuracy_test.py -m model.onnx -e OV -d CPU          # OpenVINO on CPU
    python convnext_accuracy_test.py -m model.onnx -e OV -d NPU          # OpenVINO on NPU
    python convnext_accuracy_test.py -m model.onnx -e DML -d GPU         # DirectML on GPU
    python convnext_accuracy_test.py -m model.onnx -e RTX -d GPU         # TensorRT RTX on GPU
    python convnext_accuracy_test.py -m model.onnx -e VITIS -d NPU      # VitisAI on NPU
"""

import argparse
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ort_test_utils import (
    EP_FULL_TO_SHORT,
    add_common_args,
    create_session,
    detect_layout,
    register_providers,
    resolve_ep_name,
    resolve_models,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMG_SIZE = 224
DATASET_NAME = "evanarlian/imagenet_1k_resized_256"  # Parquet-based ImageNet-1K (no loading script)
DATASET_SPLIT = "val"  # 50 000 ImageNet validation images


# ---------------------------------------------------------------------------
# Preprocessing (mirrors the ConvNeXt torchvision transform)
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image, layout: str = "NCHW") -> np.ndarray:
    """Preprocess a PIL image for ConvNeXt Tiny (ImageNet normalisation).

    Pipeline:
        1. Resize shortest edge to 236 (bicubic)
        2. Centre-crop to 224x224
        3. Convert to float32 [0, 1]
        4. Normalise with ImageNet mean/std
        5. Arrange to NCHW or NHWC layout, add batch dim
    """
    # 1. Resize shortest edge to 236 (keeps aspect ratio)
    image = image.convert("RGB")
    w, h = image.size
    short = min(w, h)
    scale = 236 / short
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    image = image.resize((new_w, new_h), Image.BICUBIC)

    # 2. Centre-crop 224x224
    left = (new_w - IMG_SIZE) // 2
    top = (new_h - IMG_SIZE) // 2
    image = image.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

    # 3. To float32 [0, 1]
    arr = np.asarray(image, dtype=np.float32) / 255.0

    # 4. Normalise
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # 5. Arrange layout and add batch dim
    if layout == "NHWC":
        return arr[np.newaxis, ...]           # 1xHxWxC
    else:
        arr = arr.transpose(2, 0, 1)          # CHW
        return arr[np.newaxis, ...]            # 1xCxHxW


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
    """Run one CoreML inference and return logits as a NumPy array."""
    outputs = coreml_model.predict({input_name: inp})
    if not outputs:
        raise RuntimeError("CoreML inference produced no outputs.")

    logits = np.asarray(next(iter(outputs.values())))
    if logits.ndim == 1:
        logits = logits[np.newaxis, :]
    return logits


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_validation_dataset(num_samples: int, seed: int = 42):
    """Load ImageNet-1K validation images from Hugging Face (streaming).

    Uses direct Parquet loading for compatibility with datasets >= 4.x
    (which removed support for custom loading scripts).
    """
    from datasets import load_dataset

    print(f"Loading '{DATASET_NAME}' dataset (streaming, split={DATASET_SPLIT}) …")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=5000)

    images = []
    labels = []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        images.append(sample["image"])
        labels.append(sample["label"])
        if (i + 1) % 500 == 0:
            print(f"  loaded {i + 1}/{num_samples} samples")

    print(f"  loaded {len(images)} samples total")
    return images, labels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    model_path: Path,
    images: list,
    labels: list,
    ep: str,
    device: str = "CPU",
):
    """Evaluate a single ONNX/CoreML model. Returns dict with accuracy & latency."""
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

    if runtime == "CoreML":
        try:
            coreml_model, input_name, layout, input_shape = load_coreml_model(model_path)
        except Exception as e:
            print(f"  ** SKIPPED – failed to load CoreML model: {e}")
            return None

        print(f"  Input layout: {layout}  (shape: {input_shape})")
    else:
        try:
            session = create_session(str(model_path), ep, device)
        except Exception as e:
            print(f"  ** SKIPPED – failed to load model: {e}")
            return None

        layout = detect_layout(session)
        print(f"  Input layout: {layout}  (shape: {session.get_inputs()[0].shape})")

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

    top1_correct = 0
    top5_correct = 0
    total = len(images)
    elapsed_ms = []

    for idx, (pil_img, label) in enumerate(zip(images, labels)):
        inp = preprocess_image(pil_img, layout=layout)

        t0 = time.perf_counter()
        if runtime == "CoreML":
            logits = run_coreml_inference(coreml_model, input_name, inp)
        else:
            logits = session.run([output_name], {input_name: inp})[0]
        t1 = time.perf_counter()
        elapsed_ms.append((t1 - t0) * 1000)

        top5_preds = np.argsort(logits[0])[::-1][:5]
        pred = top5_preds[0]

        if pred == label:
            top1_correct += 1
        if label in top5_preds:
            top5_correct += 1

        if (idx + 1) % 200 == 0:
            print(
                f"  [{idx+1:>5}/{total}]  "
                f"top-1 {top1_correct/(idx+1)*100:5.2f}%  "
                f"top-5 {top5_correct/(idx+1)*100:5.2f}%  "
                f"avg {np.mean(elapsed_ms):6.2f} ms/img"
            )

    top1_acc = top1_correct / total * 100
    top5_acc = top5_correct / total * 100
    avg_ms = float(np.mean(elapsed_ms))

    print(f"\n  Results for {model_name}:")
    print(f"    Top-1 accuracy : {top1_acc:6.2f}%  ({top1_correct}/{total})")
    print(f"    Top-5 accuracy : {top5_acc:6.2f}%  ({top5_correct}/{total})")
    print(f"    Avg latency    : {avg_ms:6.2f} ms/image")

    return {
        "model": model_name,
        "ep": "COREML" if runtime == "CoreML" else ep,
        "device": device or "",
        "runtime": runtime,
        "top1": top1_acc,
        "top5": top5_acc,
        "avg_ms": avg_ms,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list):
    """Print a comparison table of all evaluated models."""
    print(f"\n{'='*90}")
    print("ACCURACY SUMMARY  (dataset: ImageNet-1K val)")
    print(f"{'='*90}")
    hdr = f"{'Model':<14} {'Runtime':<12} {'EP':<10} {'Device':<6} {'Top-1 %':>8} {'Top-5 %':>8} {'Avg ms':>8} {'Samples':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ep_short = EP_FULL_TO_SHORT.get(r['ep'], r['ep'])
        dev = r.get('device', '')
        runtime = r.get('runtime', 'ONNXRuntime')
        print(
            f"{r['model']:<14} {runtime:<12} {ep_short:<10} {dev:<6} {r['top1']:>8.2f} {r['top5']:>8.2f} "
            f"{r['avg_ms']:>8.2f} {r['total']:>8}"
        )
    print(f"{'='*90}")


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate ConvNeXt ONNX/CoreML models on ImageNet-1K validation set"
    )
    add_common_args(p, default_num_samples=1000,
                    num_samples_help="Number of validation images to use")
    return p.parse_args()


def main():
    args = parse_args()
    model_items = resolve_models(args)
    only_coreml = all(is_coreml_model(path) for _, path in model_items)

    ep = resolve_ep_name(args.ep)
    device = args.device

    if not only_coreml:
        print("Registering available EPs...")
        register_providers(verbose=True)

    print("Models to evaluate:", [name for name, _ in model_items])
    print(f"Validation samples: {args.num_samples}")
    if only_coreml:
        print("Runtime: CoreML")
    else:
        print(f"EP: {args.ep} ({ep})")
        print(f"Device: {device}")

    images, labels = load_validation_dataset(args.num_samples, seed=args.seed)

    results = []
    for name, path in model_items:
        r = evaluate_model(
            name, path, images, labels,
            ep=ep, device=device,
        )
        if r is not None:
            results.append(r)

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
