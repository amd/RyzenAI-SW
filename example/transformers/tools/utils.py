#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

#!/bin/env python3
import numpy as np

import logging
import torch
import copy
import gc 
import os
import qlinear # Workaround to avoid RyzenAI import error
import RyzenAI

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
        print('\n**** Model size: {:.3f}MB\n\n'.format(size_all_mb))
        return size_all_mb
        
    @classmethod
    def replace_node(cls, model, old_node, new_node, new_node_args, new_node_kwargs={}):
        cls.node_count = 0
        def _replace(module, name, old_node, new_node, new_node_args, new_node_kwargs={}):
            for attr_str in dir(module):
                target_attr = getattr(module, attr_str)
                if type(target_attr) == old_node:
                    _old = target_attr
                    _new = new_node(*new_node_args, **new_node_kwargs)
                    # logging.critical(f"Replacing node: attr_str={attr_str}, {target_attr} with {_new.__class__.__name__}")
                    if (_old._get_name() == "DynamicQuantizedLinear"):
                        _new.in_features = _old.in_features
                        _new.out_features = _old.out_features
                        _new.weight_bias = _old._packed_params._weight_bias()
                        _new.quantize_weights()
                        del _old
                    elif _old.__class__.__name__ =="Linear":
                        _new.in_features = _old.in_features
                        _new.out_features = _old.out_features
                        _new.bias   = _old.bias
                        _new.weight = _old.weight  
                        #_new.quantize_weights()   
                        del _old
                    elif _old.__class__.__name__ =="WQLinear": # Replaced by QLinearPerGrp, etc
                        _new.in_features = _old.in_features
                        _new.out_features = _old.out_features
                        _new.bias   = _old.bias
                        _new.w_bit = _old.w_bit
                        _new.group_size = _old.group_size
                        _new.qweight = _old.qweight
                        _new.qzeros = _old.qzeros
                        _new.scales = _old.scales
                        #_new.quantize_weights()   
                        del _old
                        gc.collect()
                    elif _old.__class__.__name__ =="Softmax":
                        _new.dim   = _old.dim
                    elif _old.__class__.__name__ == "OPTAttention": # Replaced by OPTFlashAttention
                        replicate_opt_attention_params(_old, _new)
                        del _old
                        _new.initialize_quant_fa() # Merge QKV projections if quant_mode is not None
                        gc.collect()
                    elif _old.__class__.__name__ == "LlamaAttention": # Replaced by LlamaFlashAttention
                        replicate_llama_attention_params(_old, _new)
                        del _old
                        _new.initialize_quant_fa() # Merge QKV projections if quant_mode is not None
                        gc.collect()
                    else:
                        pass
                    setattr(module, attr_str, _new)
                    cls.node_count += 1

            for name, immediate_child_module in module.named_children():
                _replace(immediate_child_module, name, old_node, new_node, new_node_args, new_node_kwargs)
            
        print(f"Model transformation: Replacing {old_node} layers with {new_node} ...")
        _replace(model, 'model', old_node, new_node, new_node_args, new_node_kwargs)
        print(f"Model transformation done!: Replaced {cls.node_count} {old_node} layers with {new_node}.")

    @classmethod
    def analyze_weights(cls, model):
        cls.node_count = 0
        cls.weight_min, cls.weight_mean, cls.weight_max, cls.weight_stddev = 0.0, 0.0, 0.0, 0.0
        logging.critical(f",[RANGES][WEIGHT],weight.shape,min(weight),mean(weight),max(weight),std_dev(weight),mean(weight)+3*std_dev(weight)")
                    
        def _visit(module, name):
            for attr_str in dir(module):
                target_attr = getattr(module, attr_str)
                _linear_node = target_attr
                if _linear_node.__class__.__name__ =="QLinearExperimentalCPU":
                    weight = _linear_node.weight.detach().numpy().flatten()
                    logging.critical(f",[RANGES][WEIGHT],({_linear_node.weight.shape[0]}x{_linear_node.weight.shape[1]}),{np.min(weight)},{np.mean(weight)},{max(weight)},{np.std(weight)},{np.mean(weight)+3*np.std(weight)}")
                    cls.node_count += 1

            for name, immediate_child_module in module.named_children():
                _visit(immediate_child_module, name)
            
        _visit(model, 'model')
    
    @classmethod
    def get_ranges(cls, model):
        cls.node_count = 0
        cls.weight_min, cls.weight_mean, cls.weight_max, cls.weight_stddev = 0.0, 0.0, 0.0, 0.0
        cls.input_min, cls.input_mean, cls.input_max, cls.input_stddev = 0.0, 0.0, 0.0, 0.0
        cls.output_min, cls.output_max = 0.0, 0.0
        def _visit(module, name):
            for attr_str in dir(module):
                target_attr = getattr(module, attr_str)
                _linear_node = target_attr
                if _linear_node.__class__.__name__ =="Linear2":
                #if type(target_attr) == Linear2:
                    logging.critical(f"[RANGES][WEIGHT]:,{min(_linear_node.weight_min)},{np.mean(_linear_node.weight_min)},{max(_linear_node.weight_min)},{min(_linear_node.weight_max)},{np.mean(_linear_node.weight_max)},{max(_linear_node.weight_max)}")

                    logging.critical(f"[RANGES][INPUT]:,{min(_linear_node.input_min)},{np.mean(_linear_node.input_min)},{max(_linear_node.input_min)},{min(_linear_node.input_max)},{np.mean(_linear_node.input_max)},{max(_linear_node.input_max)}")

                    logging.critical(f"[RANGES][OUTPUT]:,{min(_linear_node.output_min)},{np.mean(_linear_node.output_min)},{max(_linear_node.output_min)},{min(_linear_node.output_min)},{np.mean(_linear_node.output_max)},{max(_linear_node.output_min)}")
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
            
        _visit(model, 'model')
        
    @classmethod
    def get_linear_params(cls, model, quant_mode):
        if quant_mode != 1:
            print (f"use state_dict to extract weights")
            return 
        else:
            import aie2
            import os 
            i = 0
            try: 
                os.mkdir("./quantized_weights")
            except:
                print ("weights already dumped in ./quantized_weights folder")
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
            return [model] if len(children) == 0 else [ci for c in children for ci in get_layers(c)]
        layers = get_layers(model)
        layer_counts = {'total': len(layers)}
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
                    cls.linear_shapes[name] = {f"in:{inp_shape} wt:{weight_shape} out:{outp_shape}": 1}
                else:
                    if cls.linear_shapes[name].get(f"in:{inp_shape} wt:{weight_shape} out:{outp_shape}") == None:
                        cls.linear_shapes[name][f"in:{inp_shape} wt:{weight_shape} out:{outp_shape}"] = 1
                    else:
                        cls.linear_shapes[name][f"in:{inp_shape} wt:{weight_shape} out:{outp_shape}"] += 1
            return hook_fn 
        
        def register_all_layers(model, moduletype):
            for name, module in model.named_modules():
                if isinstance(module,moduletype):
                    #print(f"Registering forward hook to {name} {module}")
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
        #for key in all_shapes.keys():
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
                if case==1:
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

                elif case==2:
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
                outp_min = outp_mean - 3*outp_std
                outp_max = outp_mean + 3*outp_std                
                if cls.linear_outp.get(name) is None:
                    cls.linear_outp[name] = [outp_min, outp_max]
                else:
                    cls.linear_outp[name] += [outp_min, outp_max]
            return hook_fn 
        
        def register_all_layers(model):
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    #print(f"Registering forward hook to {name} {module}")
                    module.register_forward_hook(generate_hook_fn(name))

        register_all_layers(model)


def replicate_opt_attention_params(v_op, fa_op):
    fa_op.embed_dim = v_op.embed_dim
    fa_op.num_heads = v_op.num_heads
    fa_op.dropout = v_op.dropout
    fa_op.head_dim = v_op.embed_dim // v_op.num_heads
    fa_op.scaling = v_op.head_dim**-0.5
    fa_op.is_decoder = v_op.is_decoder
    fa_op.k_proj = copy.deepcopy(v_op.k_proj)
    fa_op.v_proj = copy.deepcopy(v_op.v_proj)
    fa_op.q_proj = copy.deepcopy(v_op.q_proj)
    fa_op.out_proj = copy.deepcopy(v_op.out_proj)


def replicate_llama_attention_params(v_op, fa_op):
    fa_op.config = v_op.config
    fa_op.hidden_size = v_op.hidden_size
    fa_op.num_heads = v_op.num_heads
    fa_op.head_dim = v_op.head_dim
    fa_op.num_key_value_heads = v_op.num_key_value_heads
    fa_op.num_key_value_groups = v_op.num_key_value_groups
    fa_op.max_position_embeddings = v_op.max_position_embeddings
    fa_op.rope_theta = v_op.rope_theta
    fa_op.k_proj = copy.deepcopy(v_op.k_proj)
    fa_op.v_proj = copy.deepcopy(v_op.v_proj)
    fa_op.q_proj = copy.deepcopy(v_op.q_proj)
    fa_op.o_proj = copy.deepcopy(v_op.o_proj)


def get_aiegemm(impl) -> None:
    assert impl in ["v0", "v1"], "Unsupported impl!"

    if impl == "v1":
        return RyzenAI.qlinear_2_a8w8acc32("int8", "int8", "int32")
    # impl
    dev = os.getenv("DEVICE")
    if dev is None:
        print("DEVICE environment variable is not set")
        raise SystemExit
    dll_path_base =  os.getenv("PYTORCH_AIE_PATH") + "/dll/" + dev + "/qlinear/"
    dll_path = dll_path_base + "libGemmQnnAie_8x2048_2048x6144.dll"
    dll_token_path = dll_path_base + "libGemmQnnAie_1x2048_2048x6144.dll"
    return RyzenAI.qlinear(
        [dll_token_path, dll_path],
        [(1, 2048), (8, 2048)],
        (2048, 6144),
        2, 2, 4, True,
        f"./logs/log_aiegemm_cpp.csv")


def get_fa_tile_heuristic(l):
    # # Fixed tiles
    # if l <= TILE_MEDIUM:
    #     return TILE_SMALL, TILE_SMALL
    # elif TILE_MEDIUM < l <= TILE_MEDIUM + TILE_SMALL:
    #     return TILE_MEDIUM, TILE_SMALL
    # elif TILE_MEDIUM + TILE_SMALL < l <= TILE_LARGE:
    #     return TILE_MEDIUM, TILE_MEDIUM
    # elif TILE_LARGE < l <= TILE_LARGE + TILE_MEDIUM:
    #     _, x = self.get_lqlk_heuristic(l - TILE_MEDIUM)
    #     return TILE_MEDIUM, x
    # elif TILE_LARGE + TILE_MEDIUM < l <= TILE_LARGE * 2:
    #     return TILE_LARGE, TILE_LARGE
    # else:
    #     _, x = self.get_lqlk_heuristic(l - TILE_LARGE)
    #     return TILE_LARGE, x

    # Tmp solution for simply /2
    x = (l + 1) // 2
    return x, l - x


def generate_attention_test_input(b, H, L, D, attn_type="opt", has_mask=True, dtype=torch.float32):
    hidden_states = torch.rand(b, L, D)
    hidden_states = hidden_states.to(dtype)

    past_key_value = None
    attention_mask = None
    position_ids = None

    # KV cache for token phase
    Lx = L
    if L == 1:
        key_states = torch.rand((b, H, 128, D // H), dtype=dtype)
        value_states = torch.rand((b, H, 128, D // H), dtype=dtype)
        past_key_value = (key_states, value_states)
        Lx += 128

    if has_mask:
        attention_mask = torch.rand(b, 1, L, Lx) * 1.e-2
        attention_mask = attention_mask.to(dtype)
    if attn_type == "llama":
        position_ids = torch.randint(0, L, (b, L)).to(torch.long)

    return hidden_states, attention_mask, position_ids, past_key_value
