#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import copy
import gc
import logging
import os

#!/bin/env python3
import numpy as np
import RyzenAI
import torch

TILE_SMALL = 256
TILE_MEDIUM = 512
TILE_LARGE = 1024


class Utils:
    node_count = 0
    weight_min, weight_mean, weight_max, weight_stddev = 0.0, 0.0, 0.0, 0.0
    input_min, input_mean, input_max, input_stddev = 0.0, 0.0, 0.0, 0.0
    output_min, output_max = 0.0, 0.0

    # using hooks
    linear_shapes = {}
    linear_inp = {}
    linear_outp = {}

    @classmethod
    def print_model_size(cls, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print("\n**** Model size: {:.3f}MB\n\n".format(size_all_mb))
        return size_all_mb

    @classmethod
    def analyze_weights(cls, model):
        cls.node_count = 0
        cls.weight_min, cls.weight_mean, cls.weight_max, cls.weight_stddev = (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        logging.critical(
            f",[RANGES][WEIGHT],weight.shape,min(weight),mean(weight),max(weight),std_dev(weight),mean(weight)+3*std_dev(weight)"
        )

        def _visit(module, name):
            for attr_str in dir(module):
                target_attr = getattr(module, attr_str)
                _linear_node = target_attr
                if _linear_node.__class__.__name__ == "QLinearExperimentalCPU":
                    weight = _linear_node.weight.detach().numpy().flatten()
                    logging.critical(
                        f",[RANGES][WEIGHT],({_linear_node.weight.shape[0]}x{_linear_node.weight.shape[1]}),{np.min(weight)},{np.mean(weight)},{max(weight)},{np.std(weight)},{np.mean(weight)+3*np.std(weight)}"
                    )
                    cls.node_count += 1

            for name, immediate_child_module in module.named_children():
                _visit(immediate_child_module, name)

        _visit(model, "model")

    @classmethod
    def get_ranges(cls, model):
        cls.node_count = 0
        cls.weight_min, cls.weight_mean, cls.weight_max, cls.weight_stddev = (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        cls.input_min, cls.input_mean, cls.input_max, cls.input_stddev = (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        cls.output_min, cls.output_max = 0.0, 0.0

        def _visit(module, name):
            for attr_str in dir(module):
                target_attr = getattr(module, attr_str)
                _linear_node = target_attr
                if _linear_node.__class__.__name__ == "Linear2":
                    # if type(target_attr) == Linear2:
                    logging.critical(
                        f"[RANGES][WEIGHT]:,{min(_linear_node.weight_min)},{np.mean(_linear_node.weight_min)},{max(_linear_node.weight_min)},{min(_linear_node.weight_max)},{np.mean(_linear_node.weight_max)},{max(_linear_node.weight_max)}"
                    )

                    logging.critical(
                        f"[RANGES][INPUT]:,{min(_linear_node.input_min)},{np.mean(_linear_node.input_min)},{max(_linear_node.input_min)},{min(_linear_node.input_max)},{np.mean(_linear_node.input_max)},{max(_linear_node.input_max)}"
                    )

                    logging.critical(
                        f"[RANGES][OUTPUT]:,{min(_linear_node.output_min)},{np.mean(_linear_node.output_min)},{max(_linear_node.output_min)},{min(_linear_node.output_min)},{np.mean(_linear_node.output_max)},{max(_linear_node.output_min)}"
                    )
                    cls.node_count += 1

                    if min(_linear_node.weight_min) < cls.weight_min:
                        cls.weight_min = min(_linear_node.weight_min)
                    if max(_linear_node.weight_max) > cls.weight_max:
                        cls.weight_max = max(_linear_node.weight_max)

                    if min(_linear_node.input_min) < cls.input_min:
                        cls.input_min = min(_linear_node.input_min)
                    if max(_linear_node.input_max) > cls.input_max:
                        cls.input_max = max(_linear_node.input_max)

                    if min(_linear_node.output_min) < cls.output_min:
                        cls.output_min = min(_linear_node.output_min)
                    if max(_linear_node.output_max) > cls.output_max:
                        cls.output_max = max(_linear_node.output_max)
            for name, immediate_child_module in module.named_children():
                _visit(immediate_child_module, name)

        _visit(model, "model")

    @classmethod
    def get_linear_params(cls, model, quant_mode):
        if quant_mode != 1:
            print(f"use state_dict to extract weights")
            return
        else:
            import os

            import aie2

            i = 0
            try:
                os.mkdir("./quantized_weights")
            except:
                print("weights already dumped in ./quantized_weights folder")
                return
            for idx, (name, module) in enumerate(model.named_modules()):
                if module.__class__.__name__ == "Linear3":
                    print(f"{i} {idx} {name} {module.__class__.__name__}")
                    i += 1
                    f_name = f"./quantized_weights/{str(i)}_{str(idx)}_{name}.npy"
                    np.save(f_name, module.weight_q)
            return

    @classmethod
    def count_layers(cls, model):
        def get_layers(model: torch.nn.Module):
            children = list(model.children())
            return (
                [model]
                if len(children) == 0
                else [ci for c in children for ci in get_layers(c)]
            )

        layers = get_layers(model)
        layer_counts = {"total": len(layers)}
        for layer in layers:
            layer_name = layer.__class__.__name__
            if layer_name not in layer_counts:
                layer_counts[layer_name] = 1
            else:
                layer_counts[layer_name] += 1
        return layer_counts

    @classmethod
    def register_shapes_hook_linear(cls, model, moduletype=torch.nn.Linear):
        cls.linear_shapes = {}

        def generate_hook_fn(name):
            def hook_fn(module, inp, outp):
                # input is a tuple
                inp_shape = tuple((inp[0].shape))
                outp_shape = tuple((outp.shape))
                weight_shape = (inp_shape[-1], outp_shape[-1])
                if cls.linear_shapes.get(name) == None:
                    cls.linear_shapes[name] = {
                        f"in:{inp_shape} wt:{weight_shape} out:{outp_shape}": 1
                    }
                else:
                    if (
                        cls.linear_shapes[name].get(
                            f"in:{inp_shape} wt:{weight_shape} out:{outp_shape}"
                        )
                        == None
                    ):
                        cls.linear_shapes[name][
                            f"in:{inp_shape} wt:{weight_shape} out:{outp_shape}"
                        ] = 1
                    else:
                        cls.linear_shapes[name][
                            f"in:{inp_shape} wt:{weight_shape} out:{outp_shape}"
                        ] += 1

            return hook_fn

        def register_all_layers(model, moduletype):
            for name, module in model.named_modules():
                if isinstance(module, moduletype):
                    # print(f"Registering forward hook to {name} {module}")
                    module.register_forward_hook(generate_hook_fn(name))

        register_all_layers(model, moduletype)

    @classmethod
    def extract_shapes_linear(cls):
        all_shapes = {}
        for key in Utils.linear_shapes.keys():
            shapes_dict = Utils.linear_shapes[key]
            logging.critical(f"Module: {key} Shapes: {shapes_dict}")
            for shape in shapes_dict.keys():
                if all_shapes.get(shape) is None:
                    all_shapes[shape] = 0
                all_shapes[shape] += shapes_dict[shape]
        # for key in all_shapes.keys():
        #    print(f"key: {key} Shapes: {all_shapes[key]}")
        return all_shapes

    @classmethod
    def register_dist_hook_linear(cls, model):
        cls.linear_inp = {}
        cls.linear_outp = {}

        def generate_hook_fn(name):
            def hook_fn(module, inp, outp):
                # input is a tuple
                case = 1
                if case == 1:
                    data = inp[0].detach().numpy().flatten()
                    newd = []
                    upper_ = np.percentile(data, 99.999)
                    lower_ = np.percentile(data, 0.001)
                    for i in range(len(data)):
                        if (data[i] <= upper_) and (data[i] >= lower_):
                            newd.append(data[i])
                    newd = np.array(newd)
                    inp_min = newd.min()
                    inp_max = newd.max()
                    if cls.linear_inp.get(name) is None:
                        cls.linear_inp[name] = [inp_min, inp_max]
                    else:
                        cls.linear_inp[name] += [inp_min, inp_max]

                elif case == 2:
                    inp_mean = inp[0].mean().item()
                    inp_std = inp[0].std().item()
                    inp_min = inp[0].min().item()
                    inp_max = inp[0].max().item()
                    if cls.linear_inp.get(name) is None:
                        cls.linear_inp[name] = [inp_min, inp_max]
                    else:
                        cls.linear_inp[name] += [inp_min, inp_max]

                else:
                    data = list(inp[0].detach().numpy().flatten())
                    if cls.linear_inp.get(name) is None:
                        cls.linear_inp[name] = data
                    else:
                        cls.linear_inp[name] += data

                outp_mean = outp.mean().item()
                outp_std = outp.std().item()
                outp_min = outp_mean - 3 * outp_std
                outp_max = outp_mean + 3 * outp_std
                if cls.linear_outp.get(name) is None:
                    cls.linear_outp[name] = [outp_min, outp_max]
                else:
                    cls.linear_outp[name] += [outp_min, outp_max]

            return hook_fn

        def register_all_layers(model):
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # print(f"Registering forward hook to {name} {module}")
                    module.register_forward_hook(generate_hook_fn(name))

        register_all_layers(model)
