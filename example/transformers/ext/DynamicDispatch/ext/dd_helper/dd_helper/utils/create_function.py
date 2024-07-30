##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##
import numpy
import argparse

import onnx_tool
from onnx_tool import Graph
from onnx_tool.fusion import *
from onnx_tool.fusion import create_descs_from_nodenames, FusionPattern
from utils import *

import onnx
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_tensor,
)

import onnxruntime as ort


def get_domain():
    return "com.amd"


def get_opsets():
    return [
        onnx.helper.make_opsetid("", 14),
        onnx.helper.make_opsetid(get_domain(), 1),
        onnx.helper.make_opsetid("com.microsoft", 1),
    ]


o_m = onnx.load("PSF\subgraphs\_tulrv6_encoder_layer.0_attention_self_key_MatMul.onnx")

inputs = o_m.graph.input

model_input = make_tensor_value_info(inputs[0].name, TensorProto.UINT16, [1, 512, 768])
inputs = [model_input]
print(inputs)
nodes = o_m.graph.node
outputs = o_m.graph.output

initializer = o_m.graph.initializer
value_infos = o_m.graph.value_info

input_names = []
init_names = [i.name for i in initializer]
for i in inputs:
    input_names.append(i.name)
input_names.extend(init_names)
inputs.extend(initializer)
# print(inputs)
model_output = make_tensor_value_info(
    outputs[0].name, TensorProto.UINT16, [1, 512, 1152]
)
outputs = [model_output]
output_names = []
for i in outputs:
    output_names.append(i.name)
# out_val_info=[v_info for v_info in value_infos if v_info.name in output_names]
# opset_imports=
funct = onnx.helper.make_function(
    "com.amd",
    inputs=input_names,
    outputs=output_names,
    nodes=nodes,
    fname="QMatMul",
    opset_imports=get_opsets(),
)

f_node = onnx.helper.make_node(
    inputs=input_names,
    outputs=output_names,
    name="QMatMul",
    domain="com.amd",
    op_type="QMatMul",
)

graph = onnx.helper.make_graph(
    nodes=[f_node], inputs=[inputs[0]], outputs=outputs, name="with_function"
)
graph.initializer.extend(initializer)

model = onnx.helper.make_model(graph, functions=[funct], opset_imports=get_opsets())
onnx.save_model(model, "funct.onnx")
onnx.checker.check_model(model)
session_options = ort.SessionOptions()
session_options.log_severity_level = 1
session_options.log_verbosity_level = 1
sess = ort.InferenceSession("funct.onnx", session_options)
