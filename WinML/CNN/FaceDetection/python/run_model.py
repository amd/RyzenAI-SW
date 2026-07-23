#!/usr/bin/env python3
"""
retinaface Face Detection Example
Demonstrates human face detection using Windows ML with NPU acceleration
"""

import argparse
import sys
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import json
import subprocess
import cv2
from anchor_utils import generate_anchors, decode_boxes

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not installed")
    print("Install with: pip install onnxruntime")
    sys.exit(1)


def register_execution_providers():
    """Register Windows ML execution providers"""
    try:
        result = subprocess.check_output([sys.executable, "winml_worker.py"], text=True)
        ep_paths = json.loads(result)

        for name, lib_path in ep_paths.items():
            if lib_path and Path(lib_path).exists():
                ort.register_execution_provider_library(name, lib_path)
                print(f"Registered: {name}")
    except Exception as e:
        print(f"Warning: Could not register execution providers: {e}")
        print("Falling back to CPU execution")


def preprocess_image(image_path: str, input_size: tuple = (640, 640)):
    """Preprocess image for RetinaFace model (NCHW format, 640x640)"""
    img = Image.open(image_path).convert('RGB')
    original_size = img.size

    # Resize to model input size
    img_resized = img.resize(input_size, Image.BILINEAR)

    # Convert to numpy array (HWC format)
    img_array = np.array(img_resized).astype(np.float32)

    # RetinaFace uses mean subtraction (not normalization)
    # BGR mean values for face detection (note: PIL uses RGB, so reverse the mean)
    mean = np.array([123.0, 117.0, 104.0], dtype=np.float32)  # RGB order
    img_array = img_array - mean

    # Convert to NCHW format: [H,W,C] -> [C,H,W]
    img_array = img_array.transpose((2, 0, 1))

    # Add batch dimension: [C,H,W] -> [1,C,H,W]
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, original_size


def postprocess_face_detection(outputs: tuple, original_image_path: str, output_path: str,
                               image_size: tuple = (640, 640), conf_threshold: float = 0.5,
                               nms_threshold: float = 0.4) -> int:
    """
    Process RetinaFace outputs and draw bounding boxes with confidence scores.

    Args:
        outputs: Model outputs (boxes, scores, landmarks)
        original_image_path: Path to input image
        output_path: Path to save output image
        image_size: Input size (width, height)
        conf_threshold: Confidence threshold for filtering
        nms_threshold: IoU threshold for NMS

    Returns:
        Number of detected faces
    """
    # Parse outputs
    box_offsets = outputs[0][0]  # (15960, 4) - anchor box offsets
    scores = outputs[1][0]       # (15960, 2) - [background, face] logits
    landmarks = outputs[2][0]    # (15960, 10) - 5 landmarks x (x, y)

    # Generate anchors
    anchors = generate_anchors(image_size)
    print(f"Generated {len(anchors)} anchors")

    # Decode boxes from anchors and offsets
    boxes = decode_boxes(box_offsets, anchors, variances=[0.1, 0.2])

    # Convert logits to probabilities
    scores_prob = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    face_scores = scores_prob[:, 1]  # Get face class probability

    # Diagnostic: Print score statistics
    print(f"\nScore Statistics:")
    print(f"  Min: {face_scores.min():.6f}, Max: {face_scores.max():.6f}, Mean: {face_scores.mean():.6f}")
    print(f"  Scores > 0.5: {np.sum(face_scores > 0.5)}")
    print(f"  Scores > 0.1: {np.sum(face_scores > 0.1)}")
    print(f"  Scores > 0.02: {np.sum(face_scores > 0.02)}")
    top_10 = np.sort(face_scores)[-10:]
    print(f"  Top 10 scores: {top_10}")

    # Use adaptive threshold based on score distribution
    # For quantized models (NPU), scores may be lower
    if face_scores.max() < 0.6:
        # Quantized INT8 model - use higher floor to reduce false positives
        # For max_score=0.453, this gives max(0.30, 0.317) = 0.317
        adaptive_threshold = max(0.30, face_scores.max() * 0.70)
        print(f"  Using quantized-model threshold: {adaptive_threshold:.3f} (floor=0.30, mult=0.70)")
    else:
        # FP32 model - use standard threshold
        adaptive_threshold = conf_threshold
        print(f"  Using standard threshold: {adaptive_threshold}")

    # Filter by adaptive confidence threshold
    mask = face_scores > adaptive_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = face_scores[mask]
    filtered_landmarks = landmarks[mask]

    # Apply Non-Maximum Suppression
    if len(filtered_boxes) > 0:
        # Convert boxes to xyxy format for NMS
        boxes_xyxy = np.stack([
            filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2,
            filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2,
            filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2,
            filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2
        ], axis=1)

        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            filtered_scores.tolist(),
            adaptive_threshold,
            nms_threshold
        )

        if len(indices) > 0:
            indices = indices.flatten()
            final_boxes = boxes_xyxy[indices]
            final_scores = filtered_scores[indices]
            final_landmarks = filtered_landmarks[indices]
        else:
            final_boxes = np.array([])
            final_scores = np.array([])
            final_landmarks = np.array([])
    else:
        final_boxes = np.array([])
        final_scores = np.array([])
        final_landmarks = np.array([])

    # Draw results on image
    img = cv2.imread(original_image_path)
    img_height, img_width = img.shape[:2]

    for i, (box, score) in enumerate(zip(final_boxes, final_scores)):
        # Convert normalized coordinates to pixel coordinates
        x1 = int(box[0] * img_width)
        y1 = int(box[1] * img_height)
        x2 = int(box[2] * img_width)
        y2 = int(box[3] * img_height)

        # Draw bounding box (green color)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw confidence label
        label = f"Face {i+1}: {score*100:.1f}%"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Save output image
    cv2.imwrite(output_path, img)

    face_count = len(final_boxes)
    print(f"Detected {face_count} faces (after NMS), output saved to: {output_path}")

    return face_count


def run_inference(model_path: str, image_path: str, ep_policy: str = "NPU", verbose: bool = False, output_dir: str = "../images"):
    """Run face detection inference"""

    print(f"\\nModel: {Path(model_path).name}")
    print(f"Image: {Path(image_path).name}")
    print(f"EP Policy: {ep_policy}\\n")

    register_execution_providers()

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 1  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal

    if hasattr(ort, 'OrtExecutionProviderDevicePolicy'):
        if ep_policy == "NPU":
            sess_options.set_provider_selection_policy(ort.OrtExecutionProviderDevicePolicy.PREFER_NPU)
        else:
            sess_options.set_provider_selection_policy(ort.OrtExecutionProviderDevicePolicy.PREFER_CPU)

    cache_dir = os.path.abspath('cache_dir')
    provider_options = [{'cache_dir': str(cache_dir),
                         'cache_key': 'modelcachekey',
                         'enable_cache_file_io_in_mem':'0'}]
    print("Creating inference session...")
    session = ort.InferenceSession(model_path, sess_options=sess_options, provider_options=provider_options)

    providers = session.get_providers()
    print(f"Execution Providers: {providers}")

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input: {input_name} {input_shape}")

    print("\\nPreprocessing image...")
    input_data, original_size = preprocess_image(image_path)

    print("Warming up...")
    for _ in range(3):
        session.run(None, {input_name: input_data})

    print("Benchmarking...")
    iterations = 20
    start = time.perf_counter()
    for _ in range(iterations):
        outputs = session.run(None, {input_name: input_data})
    end = time.perf_counter()

    avg_latency = (end - start) / iterations * 1000
    throughput = 1000 / avg_latency

    print(f"\\nPerformance:")
    print(f"  Average latency: {avg_latency:.2f} ms")
    print(f"  Throughput: {throughput:.2f} images/sec")

    print("\\nProcessing results...")
    output_image_path = os.path.join(output_dir, f'face_detection_{ep_policy.lower()}_output.png')
    face_count = postprocess_face_detection(outputs, image_path, output_image_path)


def main():
    parser = argparse.ArgumentParser(description="Face Detection with Windows ML")
    parser.add_argument("--model", default="../model/retinaface_resnet.onnx", help="Path to ONNX model")
    parser.add_argument("--image", default="../images/face_detection_input.jpg", help="Path to input image")
    parser.add_argument("--ep_policy", default="NPU", choices=["NPU", "CPU"], help="Execution provider policy")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output_dir", default="../images", help="Output directory for results")

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        print("Run: python ../model/download_model.py")
        sys.exit(1)

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    run_inference(args.model, args.image, args.ep_policy, args.verbose, args.output_dir)


if __name__ == "__main__":
    main()
