import os
import numpy as np
import onnx
from pathlib import Path

# import sys
# sys.path.insert(0, '..')

from vitis_customop.utils.utils import *


class ResizeNormalize:
    def __init__(self, steps, params, outname):
        self.steps = steps
        self.params = params
        self.target_shape = None
        self.input_shape = None
        self.innames = [self.get_input_name()]
        self.outnames = [outname]
        self.inval_infos = None
        self.fname = self.get_function_name()
        self.nodes = []
        self.fnodes = []
        self.functions = []
        self.step_indices = {
            "Resize": 1,
            "Cast": 2,
            "Transpose": 3,
            "Div": 4,
            "Sub": 5,
            "Unsqueeze": 6,
            "Mul": 7,
            "LetterBox": 8,
            "Slice": 9,
        }

    def get_input_name(self):
        return "img_data"

    def get_function_name(self):
        return "ResizeNormalize"

    def get_invalue_info(self):
        if not self.input_shape:
            print(
                "- Error: Input shape not defined for ResizeNormalize. Please set using 'set_input_shape' method. "
            )
            exit()
        return onnx.helper.make_tensor_value_info(
            self.get_input_name(), onnx.TensorProto.UINT8, self.input_shape
        )

    def create_nodes(self):
        input_name = self.get_input_name()
        step_index = 0
        for step in self.steps:
            step_index = step_index + 1
            # print ("- Total Steps: {}, Current Step: {}".format(len(self.steps), step_index))
            output_name = None
            if step_index == len(self.steps):
                output_name = self.outnames[0]

            operator = step.get("name")
            if operator == "Resize":
                node_name = "ex_resize_" + str(step_index)
                if not output_name:
                    output_name = "ex_resized_out_" + str(step_index)
                # Target size for resize
                target_shape = step.get("target_shape")
                self.target_shape = target_shape
                sizes = onnx.helper.make_tensor(
                    "sizes",
                    data_type=onnx.TensorProto.INT64,
                    dims=[len(target_shape)],
                    vals=target_shape,
                )
                # Add constant node for target shape for resize
                self.nodes.append(
                    onnx.helper.make_node(
                        "Constant",
                        inputs=[],
                        outputs=["sizes"],
                        value=sizes,
                        name=node_name + "_const_sizes",
                    )
                )
                # Add resize node
                self.nodes.append(
                    onnx.helper.make_node(
                        "Resize",
                        inputs=[input_name, "", "", "sizes"],
                        outputs=[output_name],
                        mode="linear",
                        name=node_name,
                    )
                )
            elif operator == "Cast":
                # Add resize node
                node_name = "ex_cast_" + str(step_index)
                if not output_name:
                    output_name = "ex_cast_out_" + str(step_index)
                val = step.get("to")
                self.nodes.append(
                    onnx.helper.make_node(
                        "Cast",
                        inputs=[input_name],
                        outputs=[output_name],
                        to=val,
                        name=node_name,
                    )
                )
            elif operator == "Transpose":
                node_name = "ex_transpose_" + str(step_index)
                if not output_name:
                    output_name = "ex_transpose_out_" + str(step_index)
                values = step.get("perm")
                self.nodes.append(
                    onnx.helper.make_node(
                        "Transpose",
                        inputs=[input_name],
                        outputs=[output_name],
                        perm=values,
                        name=node_name,
                    )
                )
            elif operator == "Div":
                node_name = "ex_div_" + str(step_index)
                if not output_name:
                    output_name = "ex_div_out_" + str(step_index)
                values = step.get("val")
                # Divisor values
                div_const_name = "ex_divisor_" + str(step_index)
                divisor = onnx.helper.make_tensor(
                    div_const_name,
                    data_type=onnx.TensorProto.FLOAT,
                    dims=[len(values)],
                    vals=values,
                )
                self.nodes.extend(
                    [
                        onnx.helper.make_node(
                            "Constant",
                            inputs=[],
                            outputs=[div_const_name],
                            value=divisor,
                            name=node_name + "_const_divisor",
                        ),
                        onnx.helper.make_node(
                            "Div",
                            inputs=[input_name, div_const_name],
                            outputs=[output_name],
                            name=node_name,
                        ),
                    ]
                )
            elif operator == "Sub":
                node_name = "ex_sub_" + str(step_index)
                if not output_name:
                    output_name = "ex_sub_out_" + str(step_index)
                values = step.get("val")
                # Mean values
                mean = onnx.helper.make_tensor(
                    "mean",
                    data_type=onnx.TensorProto.FLOAT,
                    dims=[len(values)],
                    vals=values,
                )
                self.nodes.extend(
                    [
                        onnx.helper.make_node(
                            "Constant",
                            inputs=[],
                            outputs=["mean"],
                            value=mean,
                            name=node_name + "_const_mean",
                        ),
                        onnx.helper.make_node(
                            "Sub",
                            inputs=[input_name, "mean"],
                            outputs=[output_name],
                            name=node_name,
                        ),
                    ]
                )
            elif operator == "Unsqueeze":
                node_name = "ex_unsqueeze_" + str(step_index)
                if not output_name:
                    output_name = "ex_unsqueeze_out_" + str(step_index)
                # Batch axis
                axes = step.get("axes")
                batch_axis = onnx.helper.make_tensor(
                    "batch_axis",
                    data_type=onnx.TensorProto.INT64,
                    dims=[len(axes)],
                    vals=axes,
                )
                self.nodes.extend(
                    [
                        onnx.helper.make_node(
                            "Constant",
                            inputs=[],
                            outputs=["batch_axis"],
                            value=batch_axis,
                            name=node_name + "_const_batch_axis",
                        ),
                        onnx.helper.make_node(
                            "Unsqueeze",
                            inputs=[input_name, "batch_axis"],
                            outputs=[output_name],
                            name=node_name,
                        ),
                    ]
                )
            elif operator == "Slice":
                node_name = "ex_slice_" + str(step_index)
                if not output_name:
                    output_name = "ex_slice_out_" + str(step_index)
                # Slice inputs
                starts = step.get("starts")
                ends = step.get("ends")
                axes = step.get("axes")
                starts_in = onnx.helper.make_tensor(
                    "starts",
                    data_type=onnx.TensorProto.INT64,
                    dims=[len(starts)],
                    vals=starts,
                )
                ends_in = onnx.helper.make_tensor(
                    "ends",
                    data_type=onnx.TensorProto.INT64,
                    dims=[len(ends)],
                    vals=ends,
                )
                axes_in = onnx.helper.make_tensor(
                    "axes",
                    data_type=onnx.TensorProto.INT64,
                    dims=[len(axes)],
                    vals=axes,
                )
                self.nodes.extend(
                    [
                        onnx.helper.make_node(
                            "Constant",
                            inputs=[],
                            outputs=["starts"],
                            value=starts_in,
                            name=node_name + "_const_batch_axis",
                        ),
                        onnx.helper.make_node(
                            "Constant",
                            inputs=[],
                            outputs=["ends"],
                            value=ends_in,
                            name=node_name + "_const_batch_axis",
                        ),
                        onnx.helper.make_node(
                            "Constant",
                            inputs=[],
                            outputs=["axes"],
                            value=axes_in,
                            name=node_name + "_const_batch_axis",
                        ),
                        onnx.helper.make_node(
                            "Slice",
                            inputs=[input_name, "starts", "ends", "axes", ""],
                            outputs=[output_name],
                            name=node_name,
                        ),
                    ]
                )
            elif operator == "Mul":
                node_name = "ex_mul_" + str(step_index)
                if not output_name:
                    output_name = "ex_mul_out_" + str(step_index)
                # Multiplier name/values
                mul_const_name = "ex_multiplier_" + str(step_index)
                values = step.get("val")
                multiplier = onnx.helper.make_tensor(
                    mul_const_name,
                    data_type=onnx.TensorProto.FLOAT,
                    dims=[len(values)],
                    vals=values,
                )
                self.nodes.extend(
                    [
                        onnx.helper.make_node(
                            "Constant",
                            inputs=[],
                            outputs=[mul_const_name],
                            value=multiplier,
                            name=node_name + "_const_multiplier",
                        ),
                        onnx.helper.make_node(
                            "Mul",
                            inputs=[input_name, mul_const_name],
                            outputs=[output_name],
                            name=node_name,
                        ),
                    ]
                )
            elif operator == "LetterBox":
                pass
            else:
                print(
                    '\n- Error: ResizeNormalize doesn\'t support operator: "{}"'.format(
                        operator
                    )
                )
                exit()

            # Update input name
            input_name = output_name

    def create_func(self):
        # Create function
        self.functions.extend(
            [
                onnx.helper.make_function(
                    domain=get_domain(),
                    fname=self.fname,
                    inputs=self.innames,
                    outputs=self.outnames,
                    nodes=self.nodes,
                    opset_imports=get_opsets(),
                )
            ]
        )

    def create_func_nodes(self):
        # Create a node with the fuction
        node_name = "resize_and_normalize_data"
        attr_steps = [step.get("name") for step in self.steps]

        # check for transpose
        attr_trans = 0
        for step in self.steps:
            if step.get("name") == "Transpose":
                attr_trans = 1
                break
            else:
                attr_trans = 0

        # Fetch precision params and mean, std deviation
        attr_alphafbits = [
            param.get("val")
            for param in self.params
            if param.get("name") == "alpha_fbits"
        ][0]
        attr_betafbits = [
            param.get("val")
            for param in self.params
            if param.get("name") == "beta_fbits"
        ][0]
        attr_outfbits = [
            param.get("val")
            for param in self.params
            if param.get("name") == "output_fbits"
        ][0]
        attr_mean = [
            param.get("val") for param in self.params if param.get("name") == "Mean"
        ][0]
        attr_stddev = [
            param.get("val") for param in self.params if param.get("name") == "StdDev"
        ][0]
        attr_size = self.target_shape

        self.fnodes.extend(
            [
                onnx.helper.make_node(
                    self.fname,
                    inputs=self.innames,
                    outputs=self.outnames,
                    steps=attr_steps,
                    transpose=attr_trans,
                    alpha_fl=attr_alphafbits,
                    beta_fl=attr_betafbits,
                    out_fl=attr_outfbits,
                    mean=attr_mean,
                    stddev=attr_stddev,
                    out_shape=attr_size,
                    name=node_name,
                    domain=get_domain(),
                )
            ]
        )

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.inval_infos = [self.get_invalue_info()]

    def build(self):
        # Create constant nodes
        self.create_nodes()
        # Create function
        self.create_func()
        # Create function node
        self.create_func_nodes()
