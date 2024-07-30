#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import os
import numpy as np
import onnx
from onnx import numpy_helper
from pathlib import Path

# import sys
# sys.path.insert(0, '..')

from vitis_customop.utils.utils import *


class PostProcess:
    def __init__(self, input_tensor):
        self.innames = [input_tensor.name]
        self.outnames = ["values","indices"]

        self.fname = "PostProcess"
        self.nodes = []
        self.functions = []
        self.fnodes = []

        self.invalue_info = [input_tensor]
        self.outvalue_info = [
            onnx.helper.make_tensor_value_info( self.outnames[0], onnx.TensorProto.FLOAT, [5]),
            onnx.helper.make_tensor_value_info( self.outnames[1], onnx.TensorProto.INT64, [5])
        ]

        # Create constant nodes
        # self.create_constant_nodes()
        # Create  nodes
        self.create_nodes()

        # Create function
        self.functions.append(self.create_func())

        # Create function node
        self.fnodes.append(self.create_func_node())

    def create_func(self):
        # Create function
        return onnx.helper.make_function(
            domain=get_domain(),
            fname=self.fname,
            inputs=self.innames,
            outputs=self.outnames,
            nodes=self.nodes,
            opset_imports=get_opsets(),
        )

    def create_func_node(self):
        # Create a node with the fuction
        return onnx.helper.make_node(
            self.fname,
            inputs=self.innames,
            outputs=self.outnames,
            name="PostProcessSubgraph",
            domain=get_domain(),
        )

    def create_nodes(self):
        input_name = self.innames
        output_name = self.outnames 
        self.nodes.extend(
            [
                onnx.helper.make_node('Squeeze', inputs=input_name, outputs=['squeezed']),
                onnx.helper.make_node('Softmax', inputs=['squeezed'], outputs=['softmax'], axis=0),
                onnx.helper.make_node("Constant", [], ["k_value"],
                value=onnx.helper.make_tensor(name="k", data_type=onnx.TensorProto.INT64, dims=[1],vals=[5])),
                onnx.helper.make_node("TopK", inputs=["softmax", "k_value"], outputs=output_name, axis=0),
            ]
        )

    