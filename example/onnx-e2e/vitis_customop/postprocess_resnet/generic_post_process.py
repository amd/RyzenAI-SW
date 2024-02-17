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


class PostProcessor:
    def __init__(self, input_model_path, output_model_path, output_name):
        self.original_output_name = output_name
        self.base_model = load_model(input_model_path)
        self.output_model_path = output_model_path
        base_outvalue_info = [outputT for outputT in self.base_model.graph.output if outputT.name==output_name]
        input_tensor = base_outvalue_info[0]
        self.innames = [input_tensor.name]
        self.outnames = ["values","indices"]
        self.steps = []

        self.fname = "PostProcess"
        self.nodes = []
        self.functions = self.base_model.functions
        self.fnodes = []

        self.invalue_info = [input_tensor]
        self.outvalue_info = [
            onnx.helper.make_tensor_value_info( self.outnames[0], onnx.TensorProto.FLOAT, [5]),
            onnx.helper.make_tensor_value_info( self.outnames[1], onnx.TensorProto.INT64, [5])
        ]


    def ResNetPostProcess(self): 
        input_name = self.innames
        output_name = self.outnames 
        self.steps.append({"name": "Squeeze", "inputs": input_name, "outputs":['squeezed']})
        self.steps.append({"name": "Softmax", "inputs": ['squeezed'], "outputs": ['softmax'], "axis":0})
        self.steps.append({"name": "Constant", "inputs": [], "outputs": ["k_value"], "value": onnx.helper.make_tensor(name="k", data_type=onnx.TensorProto.INT64, dims=[1],vals=[5])})
        self.steps.append({"name": "TopK", "inputs": ["softmax", "k_value"], "outputs": output_name, "axis":0})

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
            name="ResNetPostSubgraph",
            domain=get_domain(),
        )

    def create_nodes(self):
        for step in self.steps: 
            operator = step.get("name")
            if operator=="Squeeze": 
                self.nodes.append(onnx.helper.make_node('Squeeze', inputs=step.get('inputs'), outputs=step.get('outputs')))
            if operator=="Softmax": 
                self.nodes.append(onnx.helper.make_node('Softmax', inputs=step.get('inputs'), outputs=step.get('outputs'), axis=step.get('axis')))
            if operator=="Constant": 
                self.nodes.append(onnx.helper.make_node('Constant', inputs=step.get('inputs'), outputs=step.get('outputs'), value=step.get('value')))
            if operator=="TopK":
                self.nodes.append(onnx.helper.make_node('TopK', inputs=step.get('inputs'), outputs=step.get('outputs'), axis=step.get('axis')))

    def save_final_model(self, output_model_path): 
        # Remove other outputs from final model
        for graph_out in self.base_model.graph.output:
            if graph_out.name == self.original_output_name:
                self.base_model.graph.output.remove(graph_out)
        # Add new output nodes 
        for valueinfo in self.outvalue_info:
            self.base_model.graph.output.append(valueinfo)
        final_model_e2e = create_model(self.base_model.graph, funcs=self.functions)

        # Apply shape inference on the final model
        final_model_e2e = infer_shapes(final_model_e2e)

        # Save
        save_model(final_model_e2e, output_model_path)
        check_model(final_model_e2e)
        # Reload model to show model input/output info
        load_model(output_model_path)

    def build(self):
        # Create constant nodes
        self.create_nodes()
        # Create function
        self.functions.append(self.create_func())
        # Create function node
        self.fnodes.append(self.create_func_node())

        self.base_model.graph.node.extend(self.nodes)

        #save final model 
        self.save_final_model(self.output_model_path)

    