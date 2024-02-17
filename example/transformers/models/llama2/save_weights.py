#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import torch
import argparse 
from transformers import pipeline, set_seed

from modeling_llama_amd import LlamaForCausalLM
from transformers import LlamaTokenizer
import os 

import gc 
import smooth

import numpy as np 

set_seed(123)

def save_weights(weights_dir):
    model = LlamaForCausalLM.from_pretrained("./llama-2-wts-hf/%s"%args.model_name) #, torch_dtype=torch.bfloat16)

    if args.quant_mode == "smooth":
        act_scales = torch.load(os.getenv("PYTORCH_AIE_PATH") + "/ext/smoothquant/act_scales/" + "llama2-7b-gateproj.pt")
        smooth.smooth_lm(model, act_scales, 0.5)
        print(f"SmoothQuant enabled ...")

    torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True )
    torch.save(model, "./quantized_llama2_%s.pth"%args.model_name)
    count = 0

    # Save weights for onnx
    for name, module in model.named_modules():
        if isinstance(module, torch.ao.nn.quantized.dynamic.modules.linear.Linear):
            weight_bias = module._packed_params._weight_bias()
            weight_q = torch.int_repr(
                weight_bias[0]).numpy().astype( np.int8)
            weight_scale = weight_bias[0].q_scale()
            
            fname = weights_dir + "/" + name 

            if weight_bias[1] is not None:
                bias = weight_bias[1].detach().numpy()
                print(f"{name} {module._get_name()} {weight_q.shape} {bias.shape} ")
                count += bias.shape[0]
                np.savez(fname, weight_q=weight_q, weight_scale=weight_scale, bias=bias)
            else:
                print(f"{name} {module._get_name()} {weight_q.shape} None ")
                bias = None
                np.savez(fname, weight_q=weight_q, weight_scale=weight_scale)
            
            count += weight_q.shape[0] * weight_q.shape[1]
    print(f"Num of params: {count/(1024*1024)}MB")


def read_weights(weights_dir):
    for path, directories, files in os.walk(weights_dir):
        for i, file_name in enumerate(files):
            file_name = path + "/" + file_name 
            npzfile = np.load(file_name)
            weight_q = npzfile['weight_q']
            weight_scale = npzfile['weight_scale']
            
            if 'bias' in npzfile.files:
                bias = npzfile['bias']
                print(f"{file_name} {weight_q.shape} {bias.shape} {weight_q.min()} {weight_q.max()}")
            else:
                bias = None
                print(f"{file_name} {weight_q.shape} None ")
            

if __name__ == "__main__":
    """
    Description:
    1. Load Llama2 model
    2. Perform Smooth quant
    3. Perform PTDQ
    4. Save pytorch model 
    5. Create weights directory
    6. Dump all integer weights, floating point scale and floating point bias to npz file
    7. Each npz file is the hierarchical name of the layer
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Different Llama model variants", type=str, default="7B_chat", choices=["7B", "7B_chat"])
    parser.add_argument('--quant_mode', help="Quantization mode - smoothquant or pytorch dynamic-quant", type=str, default="smooth", choices=["dyn", "smooth"])
    parser.add_argument('--action', help="save to npz or read from npz", type=str, default="save", choices=["save", "read"])
    args = parser.parse_args()
    print(f"{args}")

    weights_dir = "./weights_%s"%args.model_name
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    if args.action == "save":
        save_weights(weights_dir)
    else:
        read_weights(weights_dir)
            

        
    
