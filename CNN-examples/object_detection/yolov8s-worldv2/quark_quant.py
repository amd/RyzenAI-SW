# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
# refer https://quark.docs.amd.com/latest/onnx/basic_usage_onnx.html

import argparse
import copy
import logging
from pathlib import Path

import cv2
import onnx
from onnxruntime.quantization import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import custom_config as qcc
from quark.onnx.quantization.config.config import Config

import envs as ENVS
from eval_on_coco import preprocess_image
from utils import vis_check_onnx

COCO_TEST_IMAGES_DIR = ENVS.COCO_DATA_ROOT / "test2017"


def setup_logging():
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("quark_quant.log", mode="a", encoding="utf-8"),
        ],
    )


setup_logging()


def find_exclude_subgraphs_and_nodes(model_path):
    model = onnx.load(model_path)

    concat_nodes = [n for n in model.graph.node if n.op_type == "Concat"]
    reshape_nodes = [n for n in model.graph.node if n.op_type == "Reshape"]

    def get_successor_nodes(target_node: onnx.NodeProto):
        target_outputs = set(target_node.output)

        successor_nodes = []
        for node in model.graph.node:
            for input_name in node.input:
                if input_name in target_outputs:
                    successor_nodes.append(node.name)

        return successor_nodes

    def get_predecessor_nodes(target_node: onnx.NodeProto):
        target_inputs = set(target_node.input)

        predecessor_nodes = []
        for node in model.graph.node:
            for output_name in node.output:
                if output_name in target_inputs:
                    predecessor_nodes.append(node.name)

        return predecessor_nodes

    # get bbox post-process subgraph start-nodes and end-nodes
    # It started from the last Reshape node, ended before second-to-last Concat node
    last_reshape = reshape_nodes[-1]  # before start
    bbox_postprocess_start_nodes = get_successor_nodes(last_reshape)

    second_to_last_concat = concat_nodes[-2]  # after end
    bbox_postprocess_end_nodes = get_predecessor_nodes(second_to_last_concat)

    exclude_subgraphs = [(bbox_postprocess_start_nodes, bbox_postprocess_end_nodes)]

    last_concat = concat_nodes[-1]
    exclude_nodes = [last_concat.name]

    return exclude_subgraphs, exclude_nodes


def find_postprocess_subgraph(model_path):
    """The post-process subgraph started at 3rd-to-last Concat node, ended at output node"""
    model = onnx.load(model_path)

    concat_nodes = [n for n in model.graph.node if n.op_type == "Concat"]

    post_process_start_node_names = [concat_nodes[-3].name]
    post_process_end_node_names = [concat_nodes[-1].name]

    exclude_subgraphs = [
        (post_process_start_node_names, post_process_end_node_names),
    ]

    return exclude_subgraphs


def find_last_concat(model_path):
    model = onnx.load(model_path)

    concat_nodes = [n for n in model.graph.node if n.op_type == "Concat"]

    return [concat_nodes[-1].name]


def get_model_input(input_model_path: str) -> str:
    model = onnx.load(input_model_path)

    input_tensor = model.graph.input[0]

    input_name = input_tensor.name
    shape_dims = input_tensor.type.tensor_type.shape.dim

    height, width = None, None
    if len(shape_dims) >= 3:
        # NCHW
        height_dim = shape_dims[-2]
        width_dim = shape_dims[-1]

        height = height_dim.dim_value if height_dim.dim_value > 0 else None
        width = width_dim.dim_value if width_dim.dim_value > 0 else None

    return input_name, (height, width)


class ImageDataReader(CalibrationDataReader):
    def __init__(self, images_paths: list, input_name: str, input_size_hw: int):
        self._enum_data = None
        self._input_name = input_name
        self._image_1chw_list = self._prepare_images(images_paths, input_size_hw)

    def _prepare_images(self, images_paths: list, input_size_hw):
        in_h, in_w = input_size_hw
        images_1chw_list = []
        for img_path in images_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

            if img is None:
                continue

            img_1chw, *_ = preprocess_image(
                img, input_size_wh=(in_w, in_h), bgr2rgb=True
            )
            assert img_1chw.shape == (1, 3, in_h, in_w)

            images_1chw_list.append(img_1chw)

        print(f"Load {len(images_1chw_list)} calibration images!")

        return images_1chw_list

    def get_next(self):
        if self._enum_data is None:
            self._enum_data = iter(
                [{self._input_name: data} for data in self._image_1chw_list]
            )
        return next(self._enum_data, None)

    def rewind(self):
        self._enum_data = None


def get_calib_images_path(num_calib_images: int):
    """Sample images from coco-test2017 as calibration images."""

    all_images_path = sorted(COCO_TEST_IMAGES_DIR.glob("*.jpg"))

    sampled_images_path = all_images_path[:num_calib_images]

    return sampled_images_path


def quant(
    input_onnx_model_path: Path,
    quant_cfg_name: str,
    num_calib_images: int,
    exclude_post: bool,
    lr: float = 0.1,
    iters: int = 3000,
):
    print(
        f"Quanting model: {input_onnx_model_path} with "
        f"cfg: {quant_cfg_name}, {num_calib_images} calib images "
        f"lr: {lr}, iters: {iters}"
    )

    if exclude_post:
        exclude_last_concat = None
        exclude_post_subgraph = find_postprocess_subgraph(input_onnx_model_path)
        print(f"Excluding {len(exclude_post_subgraph)} subgraphs!")
        for i, subgraph in enumerate(exclude_post_subgraph):
            print(f"ignore subgraph {i}: {subgraph}")
    else:
        exclude_post_subgraph = None
        exclude_last_concat = find_last_concat(input_onnx_model_path)
        print(f"Excluding last concat node: {exclude_last_concat}")

    model_input_name, model_input_shape_hw = get_model_input(input_onnx_model_path)
    calib_images_paths = get_calib_images_path(num_calib_images)

    calib_data_reader = ImageDataReader(
        calib_images_paths, model_input_name, model_input_shape_hw
    )

    quant_config: qcc.QuantizationConfig = copy.deepcopy(
        qcc.get_default_config(quant_cfg_name)
    )

    if exclude_post_subgraph is not None:
        quant_config.subgraphs_to_exclude = exclude_post_subgraph

    if exclude_last_concat is not None:
        quant_config.nodes_to_exclude = exclude_last_concat

    quant_config.execution_providers = ["CPUExecutionProvider"]
    quant_config.extra_op_types_to_quantize = ["Einsum", "ReduceMax"]

    extra_params = None
    if "ADAROUND" in quant_cfg_name:
        extra_params = {
            "LearningRate": lr,
            "NumIterations": iters,
            "OptimDevice": "cuda:0",
            "InferDevice": "cuda:0",
            "BatchSize": 4,
        }

    if "ADAQUANT" in quant_cfg_name:
        extra_params = {
            "LearningRate": lr,
            "NumIterations": iters,
            "OptimDevice": "cuda:0",
            "InferDevice": "cuda:0",
            "BatchSize": 4,
        }

    if extra_params is not None:
        print(f"Set special params for {quant_cfg_name}, params: {extra_params}")
        ft_dict: dict = quant_config.extra_options["FastFinetune"]
        ft_dict.update(extra_params)

    print(f"Quantizing using params: {quant_config}")

    quantization_config = Config(global_quant_config=quant_config)
    quantization_config.global_quant_config.log_severity_level = 0

    post_suffix = "exclude-post" if exclude_post else "exclude-last-concat"
    in_h, in_w = model_input_shape_hw
    quant_onnx_model_path = input_onnx_model_path.with_stem(
        f"{input_onnx_model_path.stem}-{quant_cfg_name}-{in_h}x{in_w}-{post_suffix}"
    )

    quantizer = ModelQuantizer(quantization_config)
    quantizer.quantize_model(
        input_onnx_model_path.as_posix(),
        quant_onnx_model_path.as_posix(),
        calib_data_reader,
    )

    print(f"quantize success, saved to: {quant_onnx_model_path}")

    vis_check_onnx(quant_onnx_model_path, in_h)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument(
        "--quant",
        type=str,
        choices=list(qcc.DefaultConfigMapping.keys()),
        required=True,
    )
    #print(qcc.DefaultConfigMapping.keys())
    parser.add_argument("-exclude-post", action="store_true", default=False)
    #parser.add_argument("-adaround", action="store_true", default=False)

    parser.add_argument("--num-calib-images", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=3000)

    args = parser.parse_args()

    print(f"Quant input args: {args}")

    quant(
        input_onnx_model_path=Path(args.onnx),
        quant_cfg_name=args.quant,
        num_calib_images=args.num_calib_images,
        exclude_post=args.exclude_post,
        lr=args.lr,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
