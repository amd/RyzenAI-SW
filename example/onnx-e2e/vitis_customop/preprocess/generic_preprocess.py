import os
import numpy as np
import onnx
from pathlib import Path

# import sys
# sys.path.insert(0, '..')

from vitis_customop.utils.utils import *


class PreProcessor:
    def __init__(self, input_model_path, output_model_path, input_name):
        self.output_model_path = output_model_path
        self.base_model = load_model(input_model_path)
        self.target_shape = None
        self.input_shape = None
        self.innames = [self.get_input_name()]
        self.outnames = [input_name]
        self.inval_infos = None
        self.fname = self.get_function_name()
        self.nodes = []
        self.fnodes = []
        self.functions = []
        self.steps = []
        self.params = []

    def resize(self, target_shape):
        self.target_shape = target_shape
        self.steps.append({"name": "Resize", "target_shape": target_shape})

    def normalize(self, mean, std_dev, scale):
        self.steps.append(
            {
                "name": "Slice",
                "starts": [0, 0, 0],
                "ends": [self.target_shape[0], self.target_shape[1], 3],
                "axes": [0, 1, 2],
            }
        )
        self.steps.append({"name": "Cast", "to": 1})
        self.steps.append({"name": "Div", "val": scale})
        self.steps.append({"name": "Sub", "val": mean})
        self.steps.append({"name": "Div", "val": std_dev})
        self.steps.append({"name": "Transpose", "perm": (2, 0, 1)})
        self.steps.append({"name": "Unsqueeze", "axes": [0]})

    def set_resnet_params(self, mean, std_dev, scale): 
        in_min = 0.0
        in_max = 255.0

        # Compute possible data range
        data_min = ((in_min / scale[0]) - min(mean)) / max(std_dev)
        data_max = ((in_max / scale[0]) - max(mean)) / min(std_dev)

        alpha_fl = 1
        beta_fl = 7
        op_fl = 6
        ip_fl = alpha_fl
        bw = 8
        mean_in = [0.0] * 4
        std_dev_in = [0.0] * 4
        # Apply multiplier in case of div by 255
        if scale[0] == 255.0:
            multiplier = 1 << (bw - ip_fl)
            for i, x in enumerate(mean):
                mean_in[i] = round(x * multiplier, 2)
            for i, x in enumerate(std_dev):
                std_dev_in[i] = round((x * multiplier), 2)
        else:
            for i, x in enumerate(mean):
                mean_in[i] = x
            for i, x in enumerate(std_dev):
                std_dev_in[i] = x
        self.params.append({"name": "alpha_fbits", "val": alpha_fl})
        self.params.append({"name": "beta_fbits", "val": beta_fl})
        self.params.append({"name": "output_fbits", "val": op_fl})
        self.params.append({"name": "Mean", "val": mean_in})
        self.params.append({"name": "StdDev", "val": std_dev_in})
    
    def set_yolo_params(self, mean, std_dev, scale): 
        in_min = 0.0
        in_max = 255.0

        # Compute possible data range
        data_min = ((in_min / scale[0]) - min(mean)) / max(std_dev)
        data_max = ((in_max / scale[0]) - max(mean)) / min(std_dev)
        mean_in = [0.0] * 4
        std_dev_in = [0.0] * 4

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
        if scale[0] == 255.0:
            multiplier = 1 << (bw - ip_fl)
            for i, x in enumerate(mean):
                mean_in[i] = round(x * multiplier, 2)
            for i, x in enumerate(std_dev):
                std_dev_in[i] = round((x * multiplier), 2)
        else:
            for i, x in enumerate(mean):
                mean[i] = x
            for i, x in enumerate(std_dev):
                std_dev[i] = x

        # Create pre processing precision and mean, std deviation params
        self.params.append({"name": "alpha_fbits", "val": alpha_fl})
        self.params.append({"name": "beta_fbits", "val": beta_fl})
        self.params.append({"name": "output_fbits", "val": op_fl})
        self.params.append({"name": "Mean", "val": mean_in})
        self.params.append({"name": "StdDev", "val": std_dev_in})

    def save_final_model(self, output_model_path):
        self.base_model.graph.input.append(self.inval_infos[0])
        # Create final model
        final_model = create_model(self.base_model.graph, self.functions)

        # Remove other inputs from final mode
        for graph_in in final_model.graph.input:
            if graph_in.name == self.outnames[0]:
                final_model.graph.input.remove(graph_in)

        # Apply shape inference on the final model
        final_model = infer_shapes(final_model)

        # Save
        save_model(final_model, output_model_path)

        # Check that it works
        check_model(final_model)

        # Reload model to show model input/output info
        load_model(output_model_path)

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
        # Create function nodes
        self.create_func_nodes()
        # Create e2e graph
        ## adding pre nodes at the end will have issue in checker due to topological sorting requirements
        ## Insert pre nodes at the start to solve the problem
        index = 0
        for pn in self.fnodes:
            self.base_model.graph.node.insert(index, pn)
            index = index + 1
        # save final model
        self.save_final_model(self.output_model_path)
