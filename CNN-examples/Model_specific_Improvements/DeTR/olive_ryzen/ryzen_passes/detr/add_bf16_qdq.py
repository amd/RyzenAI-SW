#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import json
import onnx
import platform
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from argparse import Namespace
from typing import Optional, Union

from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.hardware.accelerator import AcceleratorSpec
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.model import ONNXModelHandler

def get_node_by_name(model, node_name):
    for node in model.graph.node:
        if node.name == node_name:
            return node
    return None

def fp32_to_bf16_raw(arr_fp32: np.ndarray) -> bytes:
    """
    Convert FP32 numpy array to BF16 raw bytes (little endian)
    """
    assert arr_fp32.dtype == np.float32
    uint32 = arr_fp32.view(np.uint32)
    bf16 = (uint32 >> 16).astype(np.uint16)
    return bf16.tobytes()

def add_bfp16_qdq(model, node_name):
    node = get_node_by_name(model, node_name)
    in_tensor_name = node.input[0]
    q_scale = np.array(1.0, dtype=np.float32)
    q_zero_point = np.array(0, dtype=np.float32)
    q_scale_init_name = f"{in_tensor_name}_q_scale"
    q_zero_point_init_name = f"{in_tensor_name}_q_zero_point"
    q_scale_init = onnx.helper.make_tensor(
        name=q_scale_init_name,
        data_type=onnx.TensorProto.FLOAT,
        dims=[],
        vals=q_scale,
        raw=True
    )
    q_zero_point_init = onnx.helper.make_tensor(
        name=q_zero_point_init_name,
        data_type=onnx.TensorProto.BFLOAT16,
        dims=[],
        vals=fp32_to_bf16_raw(q_zero_point),
        raw=True
    )
    model.graph.initializer.extend([q_scale_init, q_zero_point_init])
    q_node_output_name = f"{in_tensor_name}_q_out"
    q_node = onnx.helper.make_node(
        "ExtendedQuantizeLinear",
        [in_tensor_name, q_scale_init_name, q_zero_point_init_name],
        [q_node_output_name],
        f"{in_tensor_name}_ExtendedQuantizeLinear",
        domain="com.amd.quark",
    )

    dq_scale_init_name = f"{in_tensor_name}_dq_scale"
    dq_zero_point_init_name = f"{in_tensor_name}_dq_zero_point"
    dq_scale_init = onnx.helper.make_tensor(
        name=dq_scale_init_name,
        data_type=onnx.TensorProto.FLOAT,
        dims=[],
        vals=q_scale,
        raw=True
    )
    dq_zero_point_init = onnx.helper.make_tensor(
        name=dq_zero_point_init_name,
        data_type=onnx.TensorProto.BFLOAT16,
        dims=[],
        vals=fp32_to_bf16_raw(q_zero_point),
        raw=True
    )
    model.graph.initializer.extend([dq_scale_init, dq_zero_point_init])
    dq_node_output_name = f"{in_tensor_name}_dq_out"
    dq_node = onnx.helper.make_node(
        "ExtendedDequantizeLinear",
        [q_node_output_name, dq_scale_init_name, dq_zero_point_init_name],
        [dq_node_output_name],
        f"{in_tensor_name}_ExtendedDequantizeLinear",
        domain="com.amd.quark",
    )
    node.input[0] = dq_node_output_name
    model.graph.node.extend([q_node, dq_node])
    return model

def add_bfp16_cast(model, node_name):
    node = get_node_by_name(model, node_name)
    in_tensor_name = node.input[0]
    bf16_cast_node_output_name = f"{in_tensor_name}_bfp16_cast_out"
    bfp16_cast_node = onnx.helper.make_node(
        "Cast",
        name=f"{in_tensor_name}_bfp16_cast",
        inputs=[in_tensor_name],
        outputs=[bf16_cast_node_output_name],
        to=onnx.TensorProto.BFLOAT16
    )
    fp32_cast_node_output_name = f"{in_tensor_name}_fp32_cast_out"
    fp32_cast_node = onnx.helper.make_node(
        "Cast",
        name=f"{in_tensor_name}_fp32_cast",
        inputs=[bf16_cast_node_output_name],
        outputs=[fp32_cast_node_output_name],
        to=onnx.TensorProto.FLOAT
    )

    node.input[0] = fp32_cast_node_output_name
    model.graph.node.extend([bfp16_cast_node, fp32_cast_node])
    return model

def add_int8_qdq(model, node_name, scale, zero_point):
    node = get_node_by_name(model, node_name)
    in_tensor_name = node.input[0]
    q_scale = np.array([scale], dtype=np.float32)
    q_zero_point = np.array([zero_point], dtype=np.int8)
    q_scale_init_name = f"{in_tensor_name}_q_scale"
    q_zero_point_init_name = f"{in_tensor_name}_q_zero_point"
    q_scale_init = onnx.helper.make_tensor(
        name=q_scale_init_name,
        data_type=onnx.TensorProto.FLOAT,
        dims=[],
        vals=q_scale.tobytes(),
        raw=True
    )
    q_zero_point_init = onnx.helper.make_tensor(
        name=q_zero_point_init_name,
        data_type=onnx.TensorProto.INT8,
        dims=[],
        vals=q_zero_point.tobytes(),
        raw=True
    )
    model.graph.initializer.extend([q_scale_init, q_zero_point_init])
    q_node_output_name = f"{in_tensor_name}_q_out"
    q_node = onnx.helper.make_node(
        "QuantizeLinear",
        [in_tensor_name, q_scale_init_name, q_zero_point_init_name],
        [q_node_output_name],
        f"{in_tensor_name}_new_QuantizeLinear",
    )

    dq_scale_init_name = f"{in_tensor_name}_dq_scale"
    dq_zero_point_init_name = f"{in_tensor_name}_dq_zero_point"
    dq_scale_init = onnx.helper.make_tensor(
        name=dq_scale_init_name,
        data_type=onnx.TensorProto.FLOAT,
        dims=[],
        vals=q_scale.tobytes(),
        raw=True
    )
    dq_zero_point_init = onnx.helper.make_tensor(
        name=dq_zero_point_init_name,
        data_type=onnx.TensorProto.INT8,
        dims=[],
        vals=q_zero_point.tobytes(),
        raw=True
    )
    model.graph.initializer.extend([dq_scale_init, dq_zero_point_init])
    dq_node_output_name = f"{in_tensor_name}_dq_out"
    dq_node = onnx.helper.make_node(
        "DequantizeLinear",
        [q_node_output_name, dq_scale_init_name, dq_zero_point_init_name],
        [dq_node_output_name],
        f"{in_tensor_name}_new_DequantizeLinear",
    )
    node.input[0] = dq_node_output_name
    model.graph.node.extend([q_node, dq_node])
    return model

def add_u16_qdq(model, node_name, scale, zero_point):
    node = get_node_by_name(model, node_name)
    in_tensor_name = node.input[0]
    q_scale = np.array(scale, dtype=np.float32)
    q_zero_point = np.array(zero_point, dtype=np.uint16)
    q_scale_init_name = f"{in_tensor_name}_q_scale"
    q_zero_point_init_name = f"{in_tensor_name}_q_zero_point"
    q_scale_init = onnx.helper.make_tensor(
        name=q_scale_init_name,
        data_type=onnx.TensorProto.FLOAT,
        dims=[],
        vals=q_scale.tobytes(),
        raw=True
    )
    q_zero_point_init = onnx.helper.make_tensor(
        name=q_zero_point_init_name,
        data_type=onnx.TensorProto.UINT16,
        dims=[],
        vals=q_zero_point.tobytes(),
        raw=True
    )
    model.graph.initializer.extend([q_scale_init, q_zero_point_init])
    q_node_output_name = f"{in_tensor_name}_q_out"
    q_node = onnx.helper.make_node(
        "QuantizeLinear",
        [in_tensor_name, q_scale_init_name, q_zero_point_init_name],
        [q_node_output_name],
        f"{in_tensor_name}_new_QuantizeLinear",
        domain="com.microsoft",
    )

    dq_scale_init_name = f"{in_tensor_name}_dq_scale"
    dq_zero_point_init_name = f"{in_tensor_name}_dq_zero_point"
    dq_scale_init = onnx.helper.make_tensor(
        name=dq_scale_init_name,
        data_type=onnx.TensorProto.FLOAT,
        dims=[],
        vals=q_scale.tobytes(),
        raw=True
    )
    dq_zero_point_init = onnx.helper.make_tensor(
        name=dq_zero_point_init_name,
        data_type=onnx.TensorProto.UINT16,
        dims=[],
        vals=q_zero_point.tobytes(),
        raw=True
    )
    model.graph.initializer.extend([dq_scale_init, dq_zero_point_init])
    dq_node_output_name = f"{in_tensor_name}_dq_out"
    dq_node = onnx.helper.make_node(
        "DequantizeLinear",
        [q_node_output_name, dq_scale_init_name, dq_zero_point_init_name],
        [dq_node_output_name],
        f"{in_tensor_name}_new_DequantizeLinear",
        domain="com.microsoft",
    )
    node.input[0] = dq_node_output_name
    model.graph.node.extend([q_node, dq_node])
    return model

def get_initializer_by_name(model, name):
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return initializer
    return None

def get_i8_qdq_scale_and_zero_point(model, node_name):
    node = get_node_by_name(model, node_name)
    q_scale_name = node.input[1]
    q_zero_point_name = node.input[2]
    q_scale = get_initializer_by_name(model, q_scale_name)
    q_zero_point = get_initializer_by_name(model, q_zero_point_name)
    # return float and int8
    return float(q_scale.float_data[0]), int(q_zero_point.int32_data[0])

def get_u16_qdq_scale_and_zero_point(model, node_name):
    node = get_node_by_name(model, node_name)
    q_scale_name = node.input[1]
    q_zero_point_name = node.input[2]
    q_scale = get_initializer_by_name(model, q_scale_name)
    q_zero_point = get_initializer_by_name(model, q_zero_point_name)
    return float(q_scale.float_data[0]), int(q_zero_point.int32_data[0])

def add_bf16_qdq(model_path):
    model = onnx.load(model_path)
    scale, zero_point = get_i8_qdq_scale_and_zero_point(model, "/model/Reshape_output_0_DequantizeLinear")
    model = add_int8_qdq(model, "/model/Transpose_output_0_QuantizeLinear", scale, zero_point)
    model = add_bfp16_cast(model, "/model/Transpose_output_0_QuantizeLinear")
    node_name = "/model/decoder/layers.5/final_layer_norm/LayerNormalization_output_0_QuantizeLinear"
    scale, zero_point = get_u16_qdq_scale_and_zero_point(model, node_name)
    model = add_u16_qdq(model, node_name, scale, zero_point)
    model = add_bfp16_cast(model, node_name)
    return model

class AddBF16QDQ(Pass):

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "model_name": PassConfigParam(
                type_=str,
                required=True,
                description="file name of the output model, written to tmp_name",
            ),
            "tmp_name": PassConfigParam(
                type_=str,
                default="tmp_olive_custom",
                description="temp dir name, created in working directory if it is relative path",
            ),
        }

    def _run_for_config(self, model: ONNXModelHandler, config, output_model_path):

        src_model_path = model.model_path
        tmp_dir_path = self._create_tmp_dir(config.tmp_name)
        dst_model_path = str(tmp_dir_path / config.model_name)
        model = add_bf16_qdq(src_model_path)
        return model_proto_to_olive_model(model, dst_model_path, config)

    def _create_tmp_dir(self, tmp_name):
        cwd = os.getcwd()
        dir_path = Path(os.path.join(cwd, tmp_name))
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
