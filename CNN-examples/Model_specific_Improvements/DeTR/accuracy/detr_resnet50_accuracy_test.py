#!/usr/bin/env python3
"""
Accuracy evaluation for DETR-ResNet-50 ONNX models on COCO 2017 validation set.

Uses the COCO 2017 val split (4,952 images) from Hugging Face
(``detection-datasets/coco``, Parquet-based).

DETR is an object-detection model, so the primary metric is COCO mAP
(IoU=0.50:0.95), computed with ``pycocotools``.

Metrics reported:
  - mAP        (IoU 0.50:0.95, area=all, maxDets=100)
  - mAP@50     (IoU 0.50)
  - mAP@75     (IoU 0.75)
  - mAP-small  (area=small)
  - mAP-medium (area=medium)
  - mAP-large  (area=large)
  - Avg latency (ms/image)

Usage:
    python detr_resnet50_accuracy_test.py -m model.onnx                    # CPU EP, CPU device
    python detr_resnet50_accuracy_test.py -m model.onnx -n 500             # fewer images
    python detr_resnet50_accuracy_test.py -m model.onnx -e OV -d CPU       # OpenVINO on CPU
    python detr_resnet50_accuracy_test.py -m model.onnx -e DML -d GPU      # DirectML on GPU
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
IMG_SIZE = 800  # DETR default input resolution
DATASET_NAME = "detection-datasets/coco"  # Parquet-based COCO (no loading script)
DATASET_SPLIT = "val"  # 4,952 COCO val images
SCORE_THRESHOLD = 0.0  # keep all predictions; COCO eval handles ranking

# DETR (facebook/detr-resnet-50) uses 91 object classes + 1 "no-object" class:
#   index 0  = "N/A" (background – never used as GT)
#   indices 1-90 = COCO category IDs 1-90 (identity mapping)
#   index 91 = "no object" (padding class)
# The model's output label IS the COCO category ID; no remapping needed.
NUM_DETR_CLASSES = 91  # 0..90 are object classes, 91 = no-object

# The 80 COCO "things" category IDs (non-contiguous, skips 12,26,29,30,45,
# 66,68,69,71,83 which are "stuff" categories not annotated with instances).
COCO_80_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

# The HF dataset 'detection-datasets/coco' uses ClassLabel with 80 contiguous
# class indices (0="person"..79="toothbrush").  Map those to COCO category IDs.
COCO_80_TO_CATEGORY_ID = COCO_80_CATEGORY_IDS


# ---------------------------------------------------------------------------
# CoreML support (macOS)
# ---------------------------------------------------------------------------

def is_coreml_model(model_path: Path) -> bool:
    """Return True if the path points to a CoreML model package/file."""
    return model_path.suffix.lower() in {".mlpackage", ".mlmodel"}


def load_coreml_model(model_path: Path, compute_units=None):
    """Load a CoreML model and return (model, input_name, layout, input_shape).
    
    Args:
        model_path: Path to .mlpackage or .mlmodel
        compute_units: ct.ComputeUnit option (ALL, CPU_AND_NE, CPU_ONLY)
                      If None, defaults to CPU_AND_NE (safest for DETR).
    """
    if platform.system() != "Darwin":
        raise RuntimeError("CoreML inference is only supported on macOS.")

    try:
        import coremltools as ct
    except ImportError as e:
        raise RuntimeError(
            "coremltools is required for CoreML model evaluation. "
            "Install it with: pip install coremltools==9.0"
        ) from e

    if compute_units is None:
        compute_units = ct.ComputeUnit.CPU_AND_NE
        print(f"[info] Loading CoreML model with compute_units={compute_units}")
    else:
        print(f"[info] Loading CoreML model with compute_units={compute_units}")

    model = ct.models.MLModel(
        str(model_path),
        compute_units=compute_units,
    )
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


def run_coreml_inference(coreml_model, input_name: str, pixel_values: np.ndarray):
    """Run one CoreML inference and return (logits, pred_boxes) arrays.

    The DETR CoreML model exposes two named outputs: ``logits`` and
    ``pred_boxes``.  Falls back to positional order if the names are absent.
    """
    outputs = coreml_model.predict({input_name: pixel_values})
    if not outputs:
        raise RuntimeError("CoreML inference produced no outputs.")

    if "logits" in outputs and "pred_boxes" in outputs:
        logits = np.asarray(outputs["logits"])
        pred_boxes = np.asarray(outputs["pred_boxes"])
    else:
        values = [np.asarray(v) for v in outputs.values()]
        if len(values) < 2:
            raise RuntimeError("CoreML DETR model must produce logits and pred_boxes.")
        # logits has more channels (num_classes+1) than pred_boxes (4)
        values.sort(key=lambda a: a.shape[-1], reverse=True)
        logits, pred_boxes = values[0], values[1]
    return logits, pred_boxes


# ---------------------------------------------------------------------------
# Preprocessing (mirrors DETR / DetrImageProcessor)
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image, img_size: int = IMG_SIZE, layout: str = "NCHW") -> tuple[np.ndarray, tuple[int, int]]:
    """Preprocess a PIL image for DETR.

    Pipeline:
        1. Convert to RGB
        2. Resize to exactly (img_size, img_size)
        3. Normalise with ImageNet mean/std
        4. Arrange to NCHW or NHWC layout, add batch dim

    These ONNX exports are static 800x800 and have no ``pixel_mask`` input, so
    the model cannot tell padded regions from real content. Aspect-preserving
    resize plus zero padding therefore costs a large amount of accuracy (mAP
    0.40 -> 0.16 on the FP32 model), because roughly a third of the canvas is a
    constant region the model treats as image data. Filling the whole canvas
    avoids that, and matches the calibration pre-processing in
    ../olive_ryzen_data_scripts.py.

    Returns:
        pixel_values: np.ndarray of shape (1, 3, img_size, img_size) or (1, img_size, img_size, 3)
        orig_size:    (orig_h, orig_w) before any transforms
    """
    image = image.convert("RGB")
    orig_w, orig_h = image.size

    image = image.resize((img_size, img_size), Image.BILINEAR)

    # To float [0, 1]
    arr = np.asarray(image, dtype=np.float32) / 255.0

    # ImageNet normalise
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    if layout != "NHWC":
        arr = arr.transpose(2, 0, 1)              # HWC → CHW

    return arr[np.newaxis, ...], (orig_h, orig_w)


# ---------------------------------------------------------------------------
# Post-processing (convert DETR outputs → COCO result format)
# ---------------------------------------------------------------------------

def postprocess_detections(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    orig_h: int,
    orig_w: int,
    score_threshold: float = SCORE_THRESHOLD,
) -> list[dict]:
    """Convert DETR logits + boxes to COCO-style detections.

    Args:
        logits:     (num_queries, num_classes+1)  – last class is "no object"
        pred_boxes: (num_queries, 4) in [cx, cy, w, h] normalised to the input canvas
        orig_h, orig_w: original image size (before resize)

    Returns:
        List of dicts with keys: category_id, bbox (COCO xywh), score
    """
    # Softmax over classes
    logits_max = logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits - logits_max)
    probs = exp / exp.sum(axis=-1, keepdims=True)

    # Best class per query (excluding "no object" class at index NUM_DETR_CLASSES)
    scores = probs[:, :NUM_DETR_CLASSES].max(axis=-1)      # (num_queries,)
    labels = probs[:, :NUM_DETR_CLASSES].argmax(axis=-1)    # (num_queries,)

    # preprocess_image resizes the whole image onto the canvas with no padding,
    # so normalised boxes scale straight onto the original size.
    detections = []
    for i in range(len(scores)):
        if scores[i] < score_threshold:
            continue
        # Skip "N/A" background class (index 0)
        if labels[i] == 0:
            continue

        cx, cy, bw, bh = pred_boxes[i]
        # Normalised [0,1] wrt the canvas, which covers the whole original image
        x1 = (cx - bw / 2) * orig_w
        y1 = (cy - bh / 2) * orig_h
        w = bw * orig_w
        h = bh * orig_h

        # Clip to image bounds
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        w = min(w, orig_w - x1)
        h = min(h, orig_h - y1)

        detections.append({
            # DETR label index IS the COCO category ID (identity mapping)
            "category_id": int(labels[i]),
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "score": float(scores[i]),
        })
    return detections


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_coco_val(num_samples: int, seed: int = 42):
    """Load COCO 2017 val images from Hugging Face (streaming).

    Uses ``detection-datasets/coco`` which is Parquet-based (no loading
    script) and compatible with ``datasets >= 4.x``.

    Schema:
        image_id  – int64  (COCO image ID)
        image     – PIL Image
        width     – int64
        height    – int64
        objects   – Sequence of {bbox_id, category (ClassLabel 0-79),
                    bbox [x_min,y_min,x_max,y_max] (PASCAL VOC), area}

    Returns:
        images:      list of PIL.Image
        image_ids:   list of COCO image IDs
        annotations: list of annotation dicts (for building GT)
    """
    from datasets import load_dataset

    print(f"Loading '{DATASET_NAME}' dataset (streaming, split={DATASET_SPLIT}) …")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=5000)

    images = []
    image_ids = []
    all_annotations = []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        images.append(sample["image"])
        image_ids.append(sample["image_id"])

        # Collect ground-truth annotations for this image
        objs = sample.get("objects")
        if objs:
            bboxes = objs.get("bbox", [])
            categories = objs.get("category", [])
            areas = objs.get("area", [])
            n = len(bboxes)
            for j in range(n):
                # category is a ClassLabel index (0-79); map to COCO cat id
                cls_idx = categories[j]
                coco_cat_id = COCO_80_TO_CATEGORY_ID[cls_idx]
                # Dataset stores PASCAL VOC format [x_min, y_min, x_max, y_max];
                # convert to COCO format [x, y, w, h] for pycocotools.
                x_min, y_min, x_max, y_max = bboxes[j]
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = areas[j] if j < len(areas) else float(bbox[2] * bbox[3])
                all_annotations.append({
                    "id": len(all_annotations) + 1,
                    "image_id": sample["image_id"],
                    "category_id": coco_cat_id,
                    "bbox": list(bbox),
                    "area": float(area),
                    "iscrowd": 0,
                })

        if (i + 1) % 500 == 0:
            print(f"  loaded {i + 1}/{num_samples} samples")

    print(f"  loaded {len(images)} images, {len(all_annotations)} annotations")
    return images, image_ids, all_annotations


def build_coco_gt(image_ids: list[int], annotations: list[dict]) -> "COCO":
    """Build a pycocotools COCO ground-truth object from in-memory data."""
    from pycocotools.coco import COCO

    # Build the COCO GT dict
    categories = [{"id": cid, "name": str(cid)} for cid in sorted(set(COCO_80_CATEGORY_IDS))]
    gt_images = [{"id": iid, "width": 0, "height": 0} for iid in sorted(set(image_ids))]

    gt_dict = {
        "images": gt_images,
        "annotations": annotations,
        "categories": categories,
    }

    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()
    return coco_gt


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    model_path: Path,
    images: list,
    image_ids: list[int],
    annotations: list[dict],
    ep: str,
    device: str = "CPU",
    coreml_compute_units: str = "CPU_AND_NE",
):
    """Evaluate a DETR ONNX model using COCO mAP.

    Returns dict with mAP metrics and latency, or None on failure.
    
    Args:
        coreml_compute_units: "CPU_ONLY", "CPU_AND_NE", or "ALL" (GPU)
    """
    from pycocotools.cocoeval import COCOeval

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
            # Convert compute units string to ct.ComputeUnit
            try:
                import coremltools as ct
            except ImportError:
                raise RuntimeError("coremltools required")
            
            cu_map = {
                "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
                "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
                "ALL": ct.ComputeUnit.ALL,
            }
            compute_units = cu_map.get(coreml_compute_units, ct.ComputeUnit.CPU_AND_NE)
            
            coreml_model, input_name, layout, shape = load_coreml_model(model_path, compute_units=compute_units)
        except Exception as e:
            print(f"  ** SKIPPED – failed to load CoreML model: {e}")
            return None
        print(f"  Inputs : ['{input_name}']")
        print(f"  Input layout: {layout}  (shape: {shape})")
    else:
        try:
            session = create_session(str(model_path), ep, device)
        except Exception as e:
            print(f"  ** SKIPPED – failed to load model: {e}")
            return None

        # Determine input/output names
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
        print(f"  Inputs : {input_names}")
        print(f"  Outputs: {output_names}")

        layout = detect_layout(session)
        print(f"  Input layout: {layout}  (shape: {session.get_inputs()[0].shape})")

    all_detections = []
    elapsed_ms = []
    total = len(images)

    for idx, (pil_img, img_id) in enumerate(zip(images, image_ids)):
        pixel_values, (orig_h, orig_w) = preprocess_image(pil_img, layout=layout)

        t0 = time.perf_counter()
        if runtime == "CoreML":
            logits, pred_boxes = run_coreml_inference(coreml_model, input_name, pixel_values)
            logits = logits[0]       # (num_queries, num_classes+1)
            pred_boxes = pred_boxes[0]   # (num_queries, 4)
        else:
            feed = {input_names[0]: pixel_values}
            outputs = session.run(output_names, feed)
            # outputs[0] = logits (1, num_queries, num_classes+1)
            # outputs[1] = pred_boxes (1, num_queries, 4)
            logits = outputs[0][0]       # (num_queries, num_classes+1)
            pred_boxes = outputs[1][0]   # (num_queries, 4)
        t1 = time.perf_counter()
        elapsed_ms.append((t1 - t0) * 1000)

        dets = postprocess_detections(logits, pred_boxes, orig_h, orig_w)
        for det in dets:
            det["image_id"] = int(img_id)
        all_detections.extend(dets)

        if (idx + 1) % 200 == 0:
            print(
                f"  [{idx+1:>5}/{total}]  "
                f"{len(all_detections)} detections so far  "
                f"avg {np.mean(elapsed_ms):6.2f} ms/img"
            )

    avg_ms = float(np.mean(elapsed_ms)) if elapsed_ms else 0.0
    print(f"\n  Inference complete: {len(all_detections)} total detections")
    print(f"  Avg latency: {avg_ms:.2f} ms/image")

    # COCO evaluation
    if not all_detections:
        print("  ** No detections produced, mAP = 0")
        return {
            "model": model_name,
            "ep": "CoreML" if runtime == "CoreML" else ep,
            "device": "NE" if runtime == "CoreML" else device,
            "mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0,
            "mAP_s": 0.0, "mAP_m": 0.0, "mAP_l": 0.0,
            "avg_ms": avg_ms, "total": total,
        }

    coco_gt = build_coco_gt(image_ids, annotations)
    coco_dt = coco_gt.loadRes(all_detections)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats  # 12 standard COCO metrics
    result = {
        "model": model_name,
        "ep": "CoreML" if runtime == "CoreML" else ep,
        "device": "NE" if runtime == "CoreML" else device,
        "mAP": float(stats[0]),       # IoU=0.50:0.95, area=all
        "mAP50": float(stats[1]),     # IoU=0.50
        "mAP75": float(stats[2]),     # IoU=0.75
        "mAP_s": float(stats[3]),     # area=small
        "mAP_m": float(stats[4]),     # area=medium
        "mAP_l": float(stats[5]),     # area=large
        "avg_ms": avg_ms,
        "total": total,
    }
    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list):
    """Print a comparison table of all evaluated models."""
    print(f"\n{'='*105}")
    print("DETECTION ACCURACY SUMMARY  (dataset: COCO 2017 val)")
    print(f"{'='*105}")
    hdr = (
        f"{'Model':<25} {'EP':<10} {'Dev':<4} "
        f"{'mAP':>6} {'mAP50':>6} {'mAP75':>6} "
        f"{'AP_s':>6} {'AP_m':>6} {'AP_l':>6} "
        f"{'ms/img':>7} {'Imgs':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ep_short = EP_FULL_TO_SHORT.get(r["ep"], r["ep"])
        print(
            f"{r['model']:<25} {ep_short:<10} {r['device']:<4} "
            f"{r['mAP']:>6.3f} {r['mAP50']:>6.3f} {r['mAP75']:>6.3f} "
            f"{r['mAP_s']:>6.3f} {r['mAP_m']:>6.3f} {r['mAP_l']:>6.3f} "
            f"{r['avg_ms']:>7.1f} {r['total']:>5}"
        )
    print(f"{'='*105}")


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate DETR-ResNet-50 ONNX models on COCO 2017 val (mAP)"
    )
    add_common_args(p, default_num_samples=1000,
                    num_samples_help="Number of COCO val images to use (the split has 4,952)")
    p.add_argument(
        "-t", "--score-threshold",
        type=float,
        default=SCORE_THRESHOLD,
        help=f"Min detection score to keep (default: {SCORE_THRESHOLD})",
    )
    p.add_argument(
        "--coreml-compute-units",
        type=str,
        default="CPU_AND_NE",
        choices=["CPU_ONLY", "CPU_AND_NE", "ALL"],
        help="CoreML compute units (CPU_AND_NE = NE default, ALL = GPU, CPU_ONLY = CPU) "
             "(default: CPU_AND_NE, safest for DETR on M4)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    global SCORE_THRESHOLD
    SCORE_THRESHOLD = args.score_threshold

    print("Registering available EPs...")
    register_providers(verbose=True)

    ep = resolve_ep_name(args.ep)
    device = args.device
    model_items = resolve_models(args)

    print("Models to evaluate:", [name for name, _ in model_items])
    print(f"Validation images : {args.num_samples}")
    print(f"EP: {args.ep} ({ep})")
    print(f"Device: {device}")
    print(f"CoreML compute units: {args.coreml_compute_units}")

    images, image_ids, annotations = load_coco_val(args.num_samples, seed=args.seed)

    results = []
    for name, path in model_items:
        r = evaluate_model(
            name, path, images, image_ids, annotations,
            ep=ep, device=device,
            coreml_compute_units=args.coreml_compute_units,
        )
        if r is not None:
            results.append(r)

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
