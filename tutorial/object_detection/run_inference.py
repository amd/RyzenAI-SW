import onnxruntime as ort
import numpy as np
import cv2
import argparse
import sys
from pathlib import Path
from utils import evaluate_on_coco, get_npu_info, get_xclbin
import os 

# Load COCO class labels (optional)
COCO_CLASSES = [  # 80 classes
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Preprocessing: resize, normalize, convert to CHW
def preprocess_image(image_path, img_size=640):
    img = cv2.imread(image_path)
    orig = img.copy()
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)
    return img, orig

# Postprocessing: extract detections, draw boxes
def postprocess(outputs, img, conf_thres=0.4):
    predictions = np.transpose(np.squeeze(outputs[0]))

    boxes = []
    confidences = []
    class_ids = []

    for pred in predictions:
        x_center, y_center, width, height = pred[0:4]
        class_scores = pred[4:]  # 80 class scores

        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        if confidence < conf_thres:
            continue

        # Convert xywh (center) to xyxy (corners)
        x1 = int((x_center - width / 2) * img.shape[1] / 640)
        y1 = int((y_center - height / 2) * img.shape[0] / 640)
        x2 = int((x_center + width / 2) * img.shape[1] / 640)
        y2 = int((y_center + height / 2) * img.shape[0] / 640)

        # boxes.append((x1, y1, x2, y2, confidence, class_id))
        boxes.append([x1, y1, x2, y2])
        confidences.append(float(confidence))
        class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, nms_threshold=0.5)

    # Draw boxes
    # for (x1, y1, x2, y2, conf, class_id) in boxes:
    for i in indices:
        x1, y1, x2, y2 = boxes[i]
        conf = confidences[i]
        class_id = class_ids[i]
        label = f"Class {COCO_CLASSES[class_id]}: {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img

def main(args):
    image_path = args.input_image
    onnx_path = args.model_input
    # Preprocess the input image
    input_img, original = preprocess_image(image_path)

    if args.device == 'cpu':
        print('Running Model on CPU')
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    elif args.device == 'npu-int8':
        print('Running INT8 Model on NPU')
        npu_device = get_npu_info()
        provider_options = [{
            'cache_dir': str(Path(__file__).parent.resolve()),
            'cache_key': 'modelcachekey',
            'xclbin': get_xclbin(npu_device)
        }]
        ort_session = ort.InferenceSession(onnx_path, providers=['VitisAIExecutionProvider'], provider_options=provider_options)

    elif args.device == 'npu-bf16':
        print('Running BF16 Model on NPU')
        provider_options = [{
            'config_file': 'vaiml_config.json',
            'cache_dir': str(Path(__file__).parent.resolve()),
            'cache_key': 'modelcachekey'
        }]
        ort_session = ort.InferenceSession(onnx_path, providers=['VitisAIExecutionProvider'], provider_options=provider_options)
    else:
        print("Unsupported device. Please use 'cpu' or 'npu'.")
        sys.exit(1)

    # Get the input name from the ONNX model
    input_name = ort_session.get_inputs()[0].name

    outputs = ort_session.run(None, {input_name: input_img})
    # post process the outputs
    result_img = postprocess(outputs, original)

    # Save or display the result image
    cv2.imwrite(args.output_image, result_img)

    # Evaluate the model if the flag is set
    if args.evaluate:
        print("Model Accuracy:")
        mAP, mAP50, mAP75 = evaluate_on_coco(args.model_input, ort_session, coco_dataset=args.coco_dataset, device=args.device)
        print("{} model accuracy on {}: mAP {:.3f}, mAP50 {:.3f}, mAP75 {:.3f}".format(args.model_input, args.device, mAP, mAP50, mAP75))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize and evaluate ONNX models.")
    parser.add_argument('--model_input', type=str, default='models/resnet50_bf16.onnx', help='Path to the input ONNX model.')
    parser.add_argument('--input_image', type=str, default='test_image.jpg', help='Path to the input image for inference.')
    parser.add_argument('--output_image', type=str, default='test_output.jpg', help='Path to the output image for inference.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'npu-int8', 'npu-bf16'], required=False, help='device to run the model.')
    parser.add_argument('--int', action='store_true', help='Flag to set xclbin if model is INT8 type')
    parser.add_argument('--evaluate', action='store_true', help='Flag to evaluate the model.')
    parser.add_argument('--coco_dataset', type=str, default='datasets/coco', help='Path to the validation dataset.')
    parser.add_argument('--benchmark', action='store_true', help='Flag to benchmark the model.')

    args = parser.parse_args()
    main(args)
