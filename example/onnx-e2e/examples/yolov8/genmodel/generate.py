#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import sys
import argparse
import cv2

sys.path.insert(0, "../../..")

from vitis_customop.utils.utils import *
from vitis_customop.preprocess import resize_normalize as pre


def get_input_shape(img):
    image = cv2.imread(img)
    input_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return list(input_data.shape)


def get_resize_shape():
    return [640, 640, 4]


def get_output_shape():
    return [1, 3, 640, 640]


def add_pre_processing(input_model_path, output_model_path, args):
    # Load input model
    base_model = load_model(input_model_path)

    # Get output tensors of the model
    base_innames = [intensor.name for intensor in base_model.graph.input]
    base_invalue_info = [intensor for intensor in base_model.graph.input]

    # Create Pre-processing steps
    resize_h = get_resize_shape()[0]
    resize_w = get_resize_shape()[1]

    # checking the input shape
    input_h = get_input_shape(args.img)[0]
    input_w = get_input_shape(args.img)[1]
    if input_h < resize_h or input_w < resize_w:
        print(
            "\n- Warning: Please provide the input image with shape greater than : {} X {} Exiting!\n".format(
                resize_w, resize_h
            )
        )
        sys.exit()

    steps = list()
    steps.append({"name": "Resize", "target_shape": get_resize_shape()})
    steps.append(
        {
            "name": "Slice",
            "starts": [0, 0, 0],
            "ends": [resize_h, resize_w, 3],
            "axes": [0, 1, 2],
        }
    )
    steps.append({"name": "Transpose", "perm": (2, 0, 1)})
    steps.append({"name": "Cast", "to": 1})
    steps.append({"name": "Div", "val": [255.0]})
    steps.append({"name": "Unsqueeze", "axes": [0]})

    div_val1 = [255.0]
    sub_val = [0, 0, 0]
    div_val2 = [1, 1, 1]

    # Compute norm output data range and set precision params
    in_min = 0.0
    in_max = 255.0

    # Compute possible data range
    data_min = ((in_min / div_val1[0]) - min(sub_val)) / max(div_val2)
    data_max = ((in_max / div_val1[0]) - max(sub_val)) / min(div_val2)

    mean = [0.0] * 4
    std_dev = [0.0] * 4

    # Bit width
    bw = 8

    if data_max >= 1.0:
        op_fl = bw - 2
        ip_fl = 8  # op_fl
        alpha_fl = 8  # ip_fl
        beta_fl = bw - 2
    elif data_max >= 2.0:
        op_fl = bw - 3
        ip_fl = op_fl
        alpha_fl = ip_fl
        beta_fl = alpha_fl - 1

    # Apply multiplier in case of div by 255
    if div_val1[0] == 255.0:
        multiplier = 1 << (bw - ip_fl)
        for i, x in enumerate(sub_val):
            mean[i] = round(x * multiplier, 2)
        for i, x in enumerate(div_val2):
            std_dev[i] = round((x * multiplier), 2)
    else:
        for i, x in enumerate(sub_val):
            mean[i] = x
        for i, x in enumerate(div_val2):
            std_dev[i] = x

    # Create pre processing precision and mean, std deviation params
    params = list()
    params.append({"name": "alpha_fbits", "val": alpha_fl})
    params.append({"name": "beta_fbits", "val": beta_fl})
    params.append({"name": "output_fbits", "val": op_fl})
    params.append({"name": "Mean", "val": mean})
    params.append({"name": "StdDev", "val": std_dev})

    # Create Pre-processor
    preprocessor = pre.ResizeNormalize(steps, params, base_innames[0])
    # Set input shape
    preprocessor.set_input_shape(get_input_shape(args.img))
    # Build steps
    preprocessor.build()

    # Create e2e graph
    ## adding pre nodes at the end will have issue in checker due to topological sorting requirements
    # base_model.graph.node.extend([fn_node])
    ## Insert pre nodes at the start to solve the problem
    index = 0
    for pn in preprocessor.fnodes:
        base_model.graph.node.insert(index, pn)
        index = index + 1

    # Create Pre-processor model
    graph = create_graph(
        preprocessor.fnodes, "g_resizenorm", preprocessor.inval_infos, base_invalue_info
    )
    model = create_model(graph, preprocessor.functions)
    model = infer_shapes(model)
    save_model(model, "../models/ResizeNorm.onnx")

    # Append to the output tensor
    base_model.graph.input.append(preprocessor.inval_infos[0])

    # Create final model
    final_model = create_model(base_model.graph, preprocessor.functions)

    # Remove other inputs from final mode
    for graph_in in final_model.graph.input:
        if graph_in.name == "DetectionModel::input_0":
            final_model.graph.input.remove(graph_in)

    check_model(final_model)
    save_model(final_model, output_model_path)

    # Apply shape inference on the final model
    final_model = infer_shapes(final_model)

    # Save
    save_model(final_model, output_model_path)

    # Check that it works
    check_model(final_model)

    # Save
    save_model(final_model, output_model_path)

    # Reload model to show model input/output info
    load_model(output_model_path)


if __name__ == "__main__":
    print("-- Add Pre Processing to Face-detect model ...")
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to input image.")
    args = parser.parse_args()
    onnx_model_name = Path("../models/yolov8_int_si.v14.onnx")

    ## Version convert
    # from onnx import version_converter
    # model = onnx.load(onnx_model_name)
    # cmodel = version_converter.convert_version(model, 14)
    # cmodel_name = onnx_model_name.with_suffix(suffix=".v14.onnx")
    # onnx.save(cmodel, cmodel_name)

    onnx_e2e_model_name = onnx_model_name.with_suffix(
        suffix=".with_pre_processing.onnx"
    )
    add_pre_processing(onnx_model_name, onnx_e2e_model_name, args)
