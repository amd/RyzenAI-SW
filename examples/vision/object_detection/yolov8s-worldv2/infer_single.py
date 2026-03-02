# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

from pycocotools.coco import COCO
import envs as ENVS

from eval_on_coco import (
    load_onnx_model,
    preprocess_image,
    postprocess_output,
)

def infer_single_image(
    model_path: str,
    image_path: str,
    providers,
    runtime_seconds: int = 0,
):
    print("providers ",providers)
    session, input_name, input_size_wh, yolo_id_to_cls_map = load_onnx_model(
        model_path, providers
    )

    anno_file_path = ENVS.COCO_DATA_ROOT / "annotations/instances_val2017.json"
    coco = COCO(anno_file_path)
    cats = coco.loadCats(coco.getCatIds())
    coco_id_to_cls_map = {cat["id"]: cat["name"] for cat in cats}

    yolo_id_to_coco_id_map = {
        int(k): coco_id for k, coco_id in enumerate(sorted(coco_id_to_cls_map.keys()))
    }

    img_path = Path(image_path)
    assert img_path.is_file(), f"Image not found: {img_path}"

    img: np.ndarray = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img_h, img_w = img.shape[:2]

    img_resized, pad_top_left, scale = preprocess_image(
        img, input_size_wh, bgr2rgb=True
    )

    latencies = []
    if runtime_seconds > 0:
        print(f"\n=== Running {runtime_seconds}-second performance benchmark ===")
        end_time = time.time() + runtime_seconds
        while time.time() < end_time:
            t0 = time.time()
            _ = session.run(None, {input_name: img_resized})
            t1 = time.time()
            latencies.append(t1 - t0)

    outputs = session.run(None, {input_name: img_resized})
    outputs = outputs[0]

    detections = postprocess_output(
        outputs[0],
        pad_top_left,
        scale,
        yolo_id_to_coco_id_map,
        min_score_thres=0.25,
        nms_iou_thres=0.5,
        img_width=img_w,
        img_height=img_h,
    )

    print("\n=== Detection Results ===")
    if len(detections) == 0:
        print("No objects detected.")
    else:
        for d in detections:
            coco_id = d["category_id"]
            cls_name = coco_id_to_cls_map.get(coco_id, "unknown")
            print(
                f"[{cls_name}]  score={d['score']:.3f},  bbox={d['bbox']}"
            )

    avg_latency, fps = None, None
    if latencies:
        latencies = np.array(latencies)
        avg_latency = latencies.mean() * 1000
        fps = 1.0 / latencies.mean()
        print("\n=== Performance Results (E2E) ===")
        print(f"AVG Latency per inference: {avg_latency:.2f} ms")
        print(f"Throughput (FPS): {fps:.2f}")

    return detections, avg_latency, fps

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOWorld single-image inference + optional performance benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to yoloworld ONNX model",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image to test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference: cpu/gpu/npu",
    )
    parser.add_argument(
        "--runtime-seconds",
        type=int,
        default=0,
        help="Time (seconds) for E2E performance benchmark, 0 to skip",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.device == "npu":
        providers = ["VitisAIExecutionProvider"]
    elif args.device == "gpu":
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    #print(f"[INFO] Using device={args.device}, providers={providers}")

    detections, latency, fps = infer_single_image(
        model_path=args.model,
        image_path=args.image,
        providers=providers,
        runtime_seconds=args.runtime_seconds,
    )

    print("\n=== Finished ===")