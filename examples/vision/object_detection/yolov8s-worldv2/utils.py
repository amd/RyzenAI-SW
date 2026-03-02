# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
from pathlib import Path

import cv2
import numpy as np
from onnxruntime import InferenceSession

import envs as ENVS

__all__ = [
    "ONNXDetect",
    "vis_check_onnx",
]


# from https://github.com/jahongir7174/YOLOv8-onnx/blob/master/main.py
class ONNXDetect:
    def __init__(
        self, input_size: int, onnx_path, confidence_threshold=0.25, iou_threshold=0.75
    ):
        self.session = InferenceSession(onnx_path)

        self.inputs = self.session.get_inputs()[0].name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

    def __call__(self, image: np.ndarray):
        x, pad, gain = self.resize(image, image.shape)
        x = x.transpose((2, 0, 1))[::-1]
        x = x.astype("float32") / 255
        x = x[np.newaxis, ...]

        # outputs_list = self.session.run(
        #     output_names=["bbox_output", "cls_output"], input_feed={self.inputs: x}
        # )
        # outputs = np.concat(outputs_list, axis=1)

        outputs = self.session.run(output_names=None, input_feed={self.inputs: x})[0]

        outputs = outputs[0].transpose(1, 0)

        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]

        # Extract class scores (all rows, columns 4 onwards)
        class_scores = outputs[:, 4:]  # Shape: (8400, num_classes)

        # Find maximum score and corresponding class ID for each detection
        max_scores = np.amax(class_scores, axis=1)  # Shape: (8400,)
        class_indices = np.argmax(class_scores, axis=1)  # Shape: (8400,)

        # Filter detections based on confidence threshold
        mask = max_scores >= self.confidence_threshold
        if not np.any(mask):
            return []

        # Apply mask to filter valid detections
        outputs = outputs[mask]  # Shape: (N, 4 + num_classes)
        scores = max_scores[mask]  # Shape: (N,)
        class_indices = class_indices[mask]  # Shape: (N,)

        # Extract bounding box coordinates (cx, cy, w, h)
        cx, cy, w, h = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]

        # Calculate scaled bounding box coordinates
        left = ((cx - w / 2) / gain).astype(int)
        top = ((cy - h / 2) / gain).astype(int)
        width = (w / gain).astype(int)
        height = (h / gain).astype(int)

        # Stack boxes into list of [left, top, width, height]
        boxes = np.stack(arrays=[left, top, width, height], axis=1).tolist()
        scores = scores.tolist()
        class_indices = class_indices.tolist()

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.confidence_threshold, self.iou_threshold
        )

        # Iterate over the selected indices after non-maximum suppression
        nms_outputs = []
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_indices[i]
            nms_outputs.append([*box, score, class_id])
        return nms_outputs

    def resize(self, image: np.ndarray, shape):
        r = min(self.input_size / shape[0], self.input_size / shape[1])

        # Compute padding
        pad = int(round(shape[1] * r)), int(round(shape[0] * r))
        w = (self.input_size - pad[0]) / 2  # w padding
        h = (self.input_size - pad[1]) / 2  # h padding

        if shape[::-1] != pad:
            image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
        left, right = int(round(w - 0.1)), int(round(w + 0.1))
        image: np.ndarray = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return (
            image,
            (top, left),
            min(self.input_size / shape[0], self.input_size / shape[1]),
        )


def vis_check_onnx(onnx_model_path: str, input_size: int):
    image_path = ENVS.DATA_DIR / "test.jpg"

    image = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
    canvas = image.copy()
    model = ONNXDetect(input_size=input_size, onnx_path=onnx_model_path)
    outputs = model(image)
    for output in outputs:
        x, y, w, h, score, index = output
        cv2.rectangle(
            canvas, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2
        )

        cv2.putText(
            canvas,
            text=f"{index}: {score * 100:.1f}%",
            org=(x, max(y - 10, 0)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.25,
            color=(255, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    output_dir = ENVS.PROJECT_DIR / "runs/onnx-predict"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(onnx_model_path).stem
    img_path = output_dir / f"{model_name}_output.png"
    cv2.imwrite(img_path.as_posix(), canvas)
    print(f"visualize output write to {img_path}")
