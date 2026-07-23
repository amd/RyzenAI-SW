#!/usr/bin/env python3
"""
YOLOv8 Object Detection Example
Demonstrates object detection using Windows ML with NPU acceleration
"""

import argparse
import sys
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
import json
import subprocess
import cv2

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not installed")
    print("Install with: pip install onnxruntime")
    sys.exit(1)

# COCO class labels (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


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
    """
    Preprocess image for YOLOv8 model (NCHW format, 640x640)

    Args:
        image_path: Path to input image
        input_size: Model input size (width, height)

    Returns:
        Preprocessed image array and original image size
    """
    img = Image.open(image_path).convert('RGB')
    original_size = img.size

    # Resize to model input size
    img_resized = img.resize(input_size, Image.BILINEAR)

    # Convert to numpy array (HWC format)
    img_array = np.array(img_resized).astype(np.float32)

    # YOLOv8 normalization: [0, 255] -> [0, 1]
    img_array = img_array / 255.0

    # Convert to NCHW format: [H,W,C] -> [C,H,W]
    img_array = img_array.transpose((2, 0, 1))

    # Add batch dimension: [C,H,W] -> [1,C,H,W]
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, original_size


def postprocess_detections(outputs: tuple, original_image_path: str, output_path: str,
                          image_size: tuple = (640, 640), conf_threshold: float = 0.25,
                          nms_threshold: float = 0.45) -> int:
    """
    Process YOLOv8 outputs and draw bounding boxes with labels.

    YOLOv8 output format: [batch, 84, 8400]
    - 84 = 4 (bbox coords: x, y, w, h) + 80 (class scores)
    - 8400 = number of predictions (from different feature map scales)

    Args:
        outputs: Model outputs
        original_image_path: Path to input image
        output_path: Path to save output image
        image_size: Input size (width, height)
        conf_threshold: Confidence threshold for filtering
        nms_threshold: IoU threshold for NMS

    Returns:
        Number of detected objects
    """
    # Parse output: [1, 84, 8400] -> [8400, 84]
    predictions = np.transpose(np.squeeze(outputs[0]))

    boxes = []
    confidences = []
    class_ids = []

    for pred in predictions:
        # First 4 values are bbox coords (x_center, y_center, width, height)
        x_center, y_center, width, height = pred[0:4]

        # Next 80 values are class scores
        class_scores = pred[4:]

        # Get class with highest confidence
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence < conf_threshold:
            continue

        # Store for NMS
        boxes.append([x_center, y_center, width, height])
        confidences.append(float(confidence))
        class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    if len(boxes) > 0:
        # Convert boxes to xyxy format for NMS
        boxes_np = np.array(boxes)
        boxes_xyxy = np.stack([
            boxes_np[:, 0] - boxes_np[:, 2] / 2,  # x1
            boxes_np[:, 1] - boxes_np[:, 3] / 2,  # y1
            boxes_np[:, 0] + boxes_np[:, 2] / 2,  # x2
            boxes_np[:, 1] + boxes_np[:, 3] / 2   # y2
        ], axis=1)

        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            confidences,
            conf_threshold,
            nms_threshold
        )

        if len(indices) > 0:
            indices = indices.flatten()
            final_boxes = boxes_xyxy[indices]
            final_confidences = [confidences[i] for i in indices]
            final_class_ids = [class_ids[i] for i in indices]
        else:
            final_boxes = np.array([])
            final_confidences = []
            final_class_ids = []
    else:
        final_boxes = np.array([])
        final_confidences = []
        final_class_ids = []

    # Draw results on image
    img = cv2.imread(original_image_path)
    img_height, img_width = img.shape[:2]

    # Color map for different classes (for visualization)
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)

    for i, (box, conf, class_id) in enumerate(zip(final_boxes, final_confidences, final_class_ids)):
        # Convert normalized coordinates to pixel coordinates
        # YOLOv8 outputs are already in pixel coordinates relative to 640x640
        x1 = int(box[0] * img_width / image_size[0])
        y1 = int(box[1] * img_height / image_size[1])
        x2 = int(box[2] * img_width / image_size[0])
        y2 = int(box[3] * img_height / image_size[1])

        # Get color for this class
        color = tuple(int(c) for c in colors[class_id])

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label with class name and confidence
        label = f"{COCO_CLASSES[class_id]}: {conf*100:.1f}%"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw label background
        cv2.rectangle(img, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), color, -1)

        # Draw label text
        cv2.putText(img, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save output image
    cv2.imwrite(output_path, img)

    detection_count = len(final_boxes)
    print(f"\nDetected {detection_count} objects, output saved to: {output_path}")

    # Print detection summary
    if detection_count > 0:
        print("\nDetections:")
        class_counts = {}
        for class_id, conf in zip(final_class_ids, final_confidences):
            class_name = COCO_CLASSES[class_id]
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1

        for class_name, count in sorted(class_counts.items()):
            print(f"  - {class_name}: {count}")

    return detection_count


def run_inference(model_path: str, image_path: str, ep_policy: str = "NPU",
                 verbose: bool = False, output_dir: str = "../images"):
    """Run object detection inference"""

    print(f"\nModel: {Path(model_path).name}")
    print(f"Image: {Path(image_path).name}")
    print(f"EP Policy: {ep_policy}\n")

    register_execution_providers()

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 0  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal

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

    print("\nPreprocessing image...")
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

    print(f"\nPerformance:")
    print(f"  Average latency: {avg_latency:.2f} ms")
    print(f"  Throughput: {throughput:.2f} images/sec")

    print("\nProcessing results...")
    output_image_path = os.path.join(output_dir, f'object_detection_{ep_policy.lower()}_output.png')
    detection_count = postprocess_detections(outputs, image_path, output_image_path)


def main():
    parser = argparse.ArgumentParser(description="Object Detection with Windows ML")
    parser.add_argument("--model", default="../model/yolov8m.onnx", help="Path to ONNX model")
    parser.add_argument("--image", default="../images/object_detection_input.jpg", help="Path to input image")
    parser.add_argument("--ep_policy", default="NPU", choices=["NPU", "CPU"], help="Execution provider policy")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output_dir", default="../images", help="Output directory for results")

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        print("Run: python ../model/download_yolov8.py")
        sys.exit(1)

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    run_inference(args.model, args.image, args.ep_policy, args.verbose, args.output_dir)


if __name__ == "__main__":
    main()
