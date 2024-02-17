import sys

sys.path.insert(0, "../../..")
from pathlib import Path
from vitis_customop.preprocess import generic_preprocess as pre
from vitis_customop.postprocess_resnet import generic_post_process as post
import cv2
import argparse


def get_input_shape(img):
    image = cv2.imread(img)
    input_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    print("input image shape:", list(input_data.shape))
    return list(input_data.shape)


if __name__ == "__main__":
    print("-- Add Pre Processing to Yolov8 model ...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to input image.")
    parser.add_argument('--resize_shape', type=int, nargs=3, default=[640, 640, 4], 
                        help='Resize shape as a list of 3 integers in HWC format.')
    parser.add_argument('--mean', type=float, nargs=3, default=[0, 0, 0], 
                        help='Mean for normalization as a list of 3 floats in HWC format.')
    parser.add_argument('--std_dev', type=float, nargs=3, default=[1, 1, 1], 
                        help='Standard deviation for normalization as a list of 3 floats in HWC format.')
    parser.add_argument('--scale', type=float, nargs=3, default=[255.0, 255.0, 255.0], 
                        help='Scale factor as a list of 3 floats in HWC format.')
    args = parser.parse_args()
    onnx_model_name = Path("../models/yolov8_int_si.v14.onnx")
    onnx_pre_model_name = onnx_model_name.with_suffix(suffix=".with_pre.onnx")
    # checking the input shape
    input_shape = get_input_shape(args.img)
    input_h = input_shape[0]
    input_w = input_shape[1]
    if input_h < args.resize_shape[0] or input_w < args.resize_shape[1]:
        print(
            "\n- Warning: Please provide the input image with shape greater than : {} X {} Exiting!\n".format(
                args.resize_shape[0], args.resize_shape[1]
            )
        )
        sys.exit()
    input_node_name = "DetectionModel::input_0"
    preprocessor = pre.PreProcessor(
        onnx_model_name, onnx_pre_model_name, input_node_name
    )
    preprocessor.set_input_shape(input_shape)
    preprocessor.resize(args.resize_shape)
    preprocessor.normalize(args.mean, args.std_dev, args.scale)
    preprocessor.set_yolo_params(args.mean, args.std_dev, args.scale)
    preprocessor.build()
