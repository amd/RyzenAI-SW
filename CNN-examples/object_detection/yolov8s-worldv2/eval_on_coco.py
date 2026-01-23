# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

import json
from pathlib import Path
import argparse

import sys
import cv2
import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

import envs as ENVS
import json

def load_onnx_model(model_path: str, providers):
    session = ort.InferenceSession(model_path, providers=providers)

    input_tensor = session.get_inputs()[0]
    input_name = input_tensor.name 
    input_shape = input_tensor.shape # BCHW
    in_h, in_w = int(input_shape[2]), int(input_shape[3])

    custom_meta_map: dict = session.get_modelmeta().custom_metadata_map

    id_to_cls_json_str = custom_meta_map.get("id_to_cls", None)
    if id_to_cls_json_str is not None:
        id_to_cls_map = json.loads(id_to_cls_json_str)
    else:
        id_to_cls_map = None

    assert id_to_cls_map is not None

    return session, input_name, (in_w,in_h), id_to_cls_map


def load_coco_dataset(annotations_path):
    coco = COCO(annotations_path)
    img_ids = coco.getImgIds()

    cats = coco.loadCats(coco.getCatIds())
    id_to_name = {cat["id"]: cat["name"] for cat in cats}

    return coco, img_ids, id_to_name


def preprocess_image(img: np.ndarray, input_size_wh, bgr2rgb=False):
    # cv2.imwrite("runs/raw_img.png", img)

    img_height, img_width = img.shape[:2]
    scale = min(input_size_wh[0] / img_width, input_size_wh[1] / img_height)
    new_size = int(img_width * scale), int(img_height * scale)
    img_resized = cv2.resize(img, new_size)

    top = (input_size_wh[1] - new_size[1]) // 2
    bottom = (input_size_wh[1] - new_size[1]) - top
    left = (input_size_wh[0] - new_size[0]) // 2
    right = (input_size_wh[0] - new_size[0]) - left

    img_resized = cv2.copyMakeBorder(
        img_resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    # cv2.imwrite("runs/resized_and_padded.png", img_resized)

    if bgr2rgb:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    img_resized = np.float32(img_resized) / 255.0
    img_resized = img_resized.transpose(2, 0, 1)  # hwc --> chw
    img_resized = np.expand_dims(img_resized, axis=0)  # chw --> 1chw

    # print(f"pad top {top}, left {left}, scale {scale}")

    return img_resized, (top, left), scale


def postprocess_output(
    output: np.ndarray,
    pad_top_left: tuple,
    scale: float,
    yolo_id_to_coco_id_map: dict,
    min_score_thres: float,
    nms_iou_thres: float,
    img_width: int,
    img_height: int,
):
    # output shape: (xyxy + num-cls, num-boxes)

    # shape: (num-boxes, cxcywh + num-cls)
    output = np.transpose(output, (1, 0))

    cxcywh_nx4 = output[:, :4]  # shape: (num-boxes, 4)
    # restore boxes
    cxcywh_nx4[:, 0] -= pad_top_left[1]  # minus pad left from cx
    cxcywh_nx4[:, 1] -= pad_top_left[0]  # minus pad top from cy
    cxcywh_nx4 /= scale  # restore to original image scale
    cx, cy, w, h = (
        cxcywh_nx4[:, 0],
        cxcywh_nx4[:, 1],
        cxcywh_nx4[:, 2],
        cxcywh_nx4[:, 3],
    )
    x0 = cx - w / 2.0
    y0 = cy - h / 2.0

    class_scores_nxc = output[:, 4:]  # shape: (num-boxes, num-cls)
    scores = np.amax(class_scores_nxc, axis=1)  # shape: (num-boxes,)
    class_indices = np.argmax(class_scores_nxc, axis=1)

    # Stack boxes into list of [left, top, width, height]
    boxes_xywh_nx4: np.ndarray = np.stack(arrays=[x0, y0, w, h], axis=1)
    indices = cv2.dnn.NMSBoxes(boxes_xywh_nx4, scores, min_score_thres, nms_iou_thres)

    detections = []
    for i in indices:
        cls_id = class_indices[i]
        score = class_scores_nxc[i, cls_id]
        if score >= min_score_thres:
            detections.append(
                {
                    "category_id": yolo_id_to_coco_id_map[int(cls_id)],
                    "bbox": tuple(round(float(x), 4) for x in boxes_xywh_nx4[i]),
                    "score": round(float(score), 4),
                }
            )

    # keep map 100
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    if len(detections) > 100:
        detections = detections[:100]

    return detections


def draw_detections(
    canvas: np.ndarray,
    img_detections: list,
    id_to_name: dict = None,
    save_path: Path = None,
):
    for pred in img_detections:
        score = pred["score"]
        if score < 0.25:
            continue
        cls_id = pred["category_id"]
        x, y, w, h = np.asarray(pred["bbox"], int)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)

        if id_to_name is not None:
            text = f"{id_to_name[cls_id]}: {score * 100:.1f}%"
        else:
            text = f"{cls_id}: {score * 100:.1f}%"

        cv2.putText(
            canvas,
            text=text,
            org=(x, max(y - 10, 0)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=(255, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    save_path = str(save_path or "runs/debug_postprocess.png")
    cv2.imwrite(save_path, canvas)


def evaluate_model(
    session: ort.InferenceSession,
    input_name: str,
    coco: COCO,
    images_folder: Path,
    img_ids: list,
    yolo_id_to_coco_id_map: dict,
    coco_id_to_cls_map: dict,
    input_size_wh,
    min_score_thres=0.001,
    nms_iou_thresh=0.5,
    num_max_images=None,
    output_root: Path = None,
):
    images_folder = Path(images_folder)
    assert images_folder.is_dir()

    if num_max_images is not None:
        img_ids = img_ids[:num_max_images]

    detections = []
    demo_saved = False
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = images_folder / img_info["file_name"]
        img: np.ndarray = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_height, img_width = img.shape[:2]

        img_resized, pad_top_left, scale = preprocess_image(
            img, input_size_wh, bgr2rgb=True
        )

        # outputs shape: (bs=1, xyxy + num-cls, num-boxes)
        # outputs = session.run(
        #     output_names=["bbox_output", "cls_output"],
        #     input_feed={input_name: img_resized},
        # )
        # outputs = np.concat(outputs, axis=1)
        outputs = session.run(output_names=None, input_feed={input_name: img_resized})
        outputs = outputs[0]

        img_detections = postprocess_output(
            outputs[0],
            pad_top_left,
            scale,
            yolo_id_to_coco_id_map,
            min_score_thres,
            nms_iou_thresh,
            img_width,
            img_height,
        )

        if not demo_saved:
            save_path = None
            if output_root is not None:
                output_root = Path(output_root)
                output_root.mkdir(parents=True, exist_ok=True)
                save_path = output_root / f"predict_of_{img_id}.png"

            draw_detections(img.copy(), img_detections, coco_id_to_cls_map, save_path)
            demo_saved = True

        for det in img_detections:
            det["image_id"] = img_id

        detections.extend(img_detections)

    return detections


def save_detections(detections, output_path="detections.json"):
    with open(output_path, "w") as f:
        json.dump(detections, f, indent=2)


def save_coco_eval_results(
    coco_eval: COCOeval, save_path: str = "coco_eval_results.json"
):
    # Extract summary metrics (12 values: mAP, AR, etc.)
    summary = {
        "mAP": coco_eval.stats[0],
        "mAP50": coco_eval.stats[1],
        "mAP75": coco_eval.stats[2],
        "mAP_small": coco_eval.stats[3],
        "mAP_medium": coco_eval.stats[4],
        "mAP_large": coco_eval.stats[5],
        "AR@1": coco_eval.stats[6],
        "AR@10": coco_eval.stats[7],
        "AR@100": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11],
    }

    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"COCO evaluation results saved to: {save_path}")


def evaluate_coco(coco_gt: COCO, detections_path: str, results_save_path: str):
    coco_dt = coco_gt.loadRes(str(detections_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()

    # Print overall evaluation summary
    coco_eval.summarize()

    # Extract per-category AP (IoU=0.5:0.95, area=all)
    cat_ids = coco_gt.getCatIds()
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco_gt.loadCats(cat_ids)}

    # precision shape: [IoU thresholds, Recall thresholds, Categories, Area range, MaxDets]
    prec = coco_eval.eval["precision"]

    print("\nPer-category AP (IoU=0.5:0.95, area=all):")
    for idx, cat_id in enumerate(cat_ids):
        # Select metrics: IoU=0.5:0.95, area=all (index 0), maxDet=100 (index 2)
        precision = prec[:, :, idx, 0, 2]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        print(f"{cat_id_to_name[cat_id]:<20} AP: {ap:.3f}")

    mAP = coco_eval.stats[0] * 100
    mAP50 = coco_eval.stats[1] * 100
    mAP75 = coco_eval.stats[2] * 100

    print("\nMain COCO Metrics:")
    print(f"mAP     (AP@[IoU=0.50:0.95]): {mAP:.1f}")
    print(f"mAP50   (AP@IoU=0.50)       : {mAP50:.1f}")
    print(f"mAP75   (AP@IoU=0.75)       : {mAP75:.1f}")

    save_coco_eval_results(coco_eval, results_save_path)


def calc_yolo_id_to_coco_map(yolo_id_to_cls_map: dict, coco_id_to_cls_map: dict):
    yolo_cls_names = sorted(yolo_id_to_cls_map.values())
    coco_cls_names = sorted(coco_id_to_cls_map.values())

    assert yolo_cls_names == coco_cls_names

    coco_cls_to_id_map = {v: k for k, v in coco_id_to_cls_map.items()}

    yolo_id_to_coco_id_map = {
        int(k): coco_cls_to_id_map[v] for k, v in yolo_id_to_cls_map.items()
    }

    return yolo_id_to_coco_id_map

def eval_by_coco_tools(args):
    if args.model is not None:
        onnx_model_path = Path(args.model)
        assert onnx_model_path.is_file(), f"Model not found: {onnx_model_path}"
    else:
        onnx_model_path = ENVS.MODELS_DIR / "yolov8s-world.onnx"

    if args.device == "npu":
        providers = ["VitisAIExecutionProvider"]
    elif args.device == "gpu":
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    print(f"Evaluating model: {onnx_model_path}")
    print(f"Using device: {args.device} -> providers={providers}")

    anno_file_path = ENVS.COCO_DATA_ROOT / "annotations/instances_val2017.json"
    coco, img_ids, coco_id_to_cls_map = load_coco_dataset(anno_file_path)

    session, input_name, input_size_wh, yolo_id_to_cls_map = load_onnx_model(
        onnx_model_path, providers
    )

    yolo_id_to_coco_id_map = calc_yolo_id_to_coco_map(
        yolo_id_to_cls_map,
        coco_id_to_cls_map
    )

    coco_val2017_images_folder = ENVS.COCO_DATA_ROOT / "val2017"
    nms_iou_thresh = 0.75

    output_root = (
        ENVS.PROJECT_DIR
        / f"runs/onnx-predict/{onnx_model_path.stem}-{anno_file_path.stem}-iou={nms_iou_thresh:.2f}"
    )
    output_root.mkdir(exist_ok=True, parents=True)

    detections = evaluate_model(
        session,
        input_name,
        coco,
        coco_val2017_images_folder,
        img_ids,
        yolo_id_to_coco_id_map,
        coco_id_to_cls_map,
        input_size_wh=input_size_wh,
        min_score_thres=0.001,
        nms_iou_thresh=nms_iou_thresh,
        output_root=output_root,
        # num_max_images=10,
    )

    pred_json_save_path = output_root / "pred.json"
    save_detections(detections, pred_json_save_path)
    print(f"detectoins saved to: {pred_json_save_path}")

    coco_eval_save_path = output_root / "coco-metrics.json"
    evaluate_coco(coco, pred_json_save_path, ENVS.PROJECT_DIR / coco_eval_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to ONNX model (default: yolov8s-world.onnx in ENVS.MODELS_DIR)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "npu"],
        help="Inference device backend"
    )

    args = parser.parse_args()
    eval_by_coco_tools(args)

