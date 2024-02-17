#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import sys
import argparse
import cv2

sys.path.insert(0, "../../..")

from vitis_customop.utils.utils import *
from vitis_customop.preprocess import resize_normalize as pre
from vitis_customop.postprocess_resnet import post_process as post


def get_input_shape(img):
    image = cv2.imread(img)
    input_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return list(input_data.shape)


def get_resize_shape():
    return [224, 224, 4]


def get_output_shape():
    return [1, 3, 224, 224]


def add_post_process(model, nodes):
    model.graph.node.extend(nodes)


def remove_extra_io(model):
    # Remove other outputs from final model
    for graph_out in model.graph.output:
        if graph_out.name == "1327":
            model.graph.output.remove(graph_out)


def add_new_io(model, postprocessor):
    # Append to the output tensor
    for valueinfo in postprocessor.outvalue_info:
        model.graph.output.append(valueinfo)


def add_pre_processing(input_model_path, output_model_path, args):
    e2e_post = True
    # Load input model
    base_model = load_model(input_model_path)

    # Get output tensors of the model
    base_innames = [intensor.name for intensor in base_model.graph.input]
    base_invalue_info = [intensor for intensor in base_model.graph.input]

    # Get output tensors of the model
    base_outnames = [outputT.name for outputT in base_model.graph.output]
    base_outvalue_info = [outputT for outputT in base_model.graph.output]

    # Model params
    sub_val = [51.97, 58.39, 61.84]
    div_val2 = [41.65, 41.65, 41.65]
    div_val1 = [1.0, 1.0, 1.0]

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
    steps.append({"name": "Cast", "to": 1})
    steps.append({"name": "Sub", "val": sub_val})
    steps.append({"name": "Div", "val": div_val2})
    steps.append({"name": "Transpose", "perm": (2, 0, 1)})
    steps.append({"name": "Unsqueeze", "axes": [0]})

    # Compute norm output data range and set precision params
    in_min = 0.0
    in_max = 255.0

    # Compute possible data range
    data_min = ((in_min / div_val1[0]) - min(sub_val)) / max(div_val2)
    data_max = ((in_max / div_val1[0]) - max(sub_val)) / min(
        div_val2
    )  # 255 - 123 / 166.7

    mean = [0.0] * 4
    std_dev = [0.0] * 4
    alpha_fl = 1
    beta_fl = 7
    op_fl = 6
    ip_fl = alpha_fl

    # Bit width
    bw = 8

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
        preprocessor.nodes, "g_resizenorm", preprocessor.inval_infos, base_invalue_info
    )

    model = create_model(graph, preprocessor.functions)
    model = infer_shapes(model)
    save_model(model, "../models/ResizeNorm.onnx")

    # Append to the output tensor
    base_model.graph.input.append(preprocessor.inval_infos[0])

    # base_model.graph.node.insert(0, preprocessor.inval_infos[0])
    # nd_score_trans = onnx.helper.make_node("Transpose", perm = (0,2,1), name="T_score_t")
    # base_model.graph.node.insert(1, nd_score_trans)

    # Pre/Post functions
    functions = []
    functions.extend(preprocessor.functions)

    if e2e_post:
        # Create post processor
        postprocessor = post.PostProcess(base_outvalue_info[0])
        # Add post-nodes to base model
        add_post_process(base_model, postprocessor.nodes)
        functions.extend(postprocessor.functions)
        # Create final model
        final_model_e2e = create_model(base_model.graph, funcs=functions)

        # Remove extra input/output
        remove_extra_io(final_model_e2e)

        # # Remove extra input/output
        add_new_io(final_model_e2e, postprocessor)
        # Remove other inputs from final mode
        for graph_in in final_model_e2e.graph.input:
            if graph_in.name == "blob.1":
                final_model_e2e.graph.input.remove(graph_in)
        onnx_e2e_model_name = onnx_model_name.with_suffix(suffix=".e2e.onnx")

        check_model(final_model_e2e)
        save_model(final_model_e2e, onnx_e2e_model_name)

        # Apply shape inference on the final model
        final_model_e2e = infer_shapes(final_model_e2e)

        # Save
        save_model(final_model_e2e, onnx_e2e_model_name)

        # Check that it works
        check_model(final_model_e2e)

        # Save
        save_model(final_model_e2e, onnx_e2e_model_name)

        # Reload model to show model input/output info
        load_model(onnx_e2e_model_name)

    # Create final model
    final_model = create_model(base_model.graph, funcs=functions)
    # Remove other inputs from final mode
    for graph_in in final_model.graph.input:
        if graph_in.name == "blob.1":
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
    print("-- Add Pre Processing to ResNet50 model ...")
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to input image.")
    args = parser.parse_args()
    onnx_model_name = Path("../models/resnet50_pt.v14.onnx")

    onnx_e2e_model_name = onnx_model_name.with_suffix(suffix=".with_pre.onnx")
    add_pre_processing(onnx_model_name, onnx_e2e_model_name, args)
