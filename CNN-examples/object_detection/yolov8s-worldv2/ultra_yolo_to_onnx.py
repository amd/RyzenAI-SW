# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
import json

import onnx

from utils import vis_check_onnx


def split_concat_output(model: onnx.ModelProto):
    import onnx_graphsurgeon as gs

    # build graph
    graph = gs.import_onnx(model)

    # Find the last concat node (assuming it's the final output)
    concat_nodes = [n for n in graph.nodes if n.op == "Concat"]
    concat_node = concat_nodes[-1]
    inputs = concat_node.inputs  # Expecting [bbox_pred, class_pred]

    # Remove the concat node
    concat_node.outputs[0].outputs.clear()  # Remove downstream connection
    graph.nodes.remove(concat_node)

    # Set its inputs as new outputs
    graph.outputs = inputs

    # Rename outputs for clarity
    inputs[0].name = "bbox_output"
    inputs[1].name = "cls_output"

    # Save updated model
    return gs.export_onnx(graph)


def pt_to_onnx(pt_model_name: str, input_size):
    import onnx
    from ultralytics import YOLOWorld

    yolo_model = YOLOWorld(pt_model_name)

    onnx_model_path = yolo_model.export(
        format="onnx",
        nms=False,
        dynamic=False,
        simplify=True,
        # default is 640, but will raise an exception if use 640
        imgsz=input_size,
        opset=20,
    )

    # append id-to-cls map to model
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # onnx_model = split_concat_output(onnx_model)

    id_to_cls_meta = onnx_model.metadata_props.add()
    id_to_cls_meta.key = "id_to_cls"
    id_to_cls_meta.value = json.dumps(yolo_model.names)

    onnx.save_model(onnx_model, onnx_model_path)

    return onnx_model_path


def check_onnx_model(onnx_model_path: str):
    import onnx

    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)

    print("=" * 88)
    print("Model inputs:")
    for input_tensor in model.graph.input:
        name = input_tensor.name
        type_info = input_tensor.type.tensor_type
        shape = [d.dim_value if (d.dim_value > 0) else "?" for d in type_info.shape.dim]
        print(f"  - name: {name}, shape: {shape}")

    print("Model outputs:")
    for output_tensor in model.graph.output:
        name = output_tensor.name
        type_info = output_tensor.type.tensor_type
        shape = [d.dim_value if (d.dim_value > 0) else "?" for d in type_info.shape.dim]
        print(f"  - name: {name}, shape: {shape}")
    print("=" * 88)


def export_pt_to_onnx(pt_model: str, input_size: int):
    onnx_model_path = pt_to_onnx(pt_model, input_size)

    check_onnx_model(onnx_model_path)

    vis_check_onnx(onnx_model_path, input_size)


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pt-model",
        required=True,
        type=str,
        help="Local or remote path to pytorch pt weight",
    )
    parser.add_argument(
        "--input-size",
        required=True,
        type=int,
        help="Input size of model, eg 320, 640",
    )

    args = parser.parse_args()

    print(f"Exporting model: {args.pt_model} with input size: {args.input_size}")

    export_pt_to_onnx(args.pt_model, args.input_size)


if __name__ == "__main__":
    main()
