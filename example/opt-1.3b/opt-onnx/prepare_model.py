import torch
import argparse 
from transformers import AutoTokenizer, OPTForCausalLM
import os 

# https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization

import smooth

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

import onnx 
import onnxruntime as ort
import numpy as np

import time 

import string
import random 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Different OPT model sizes", type=str, default="opt-1.3b", choices=["opt-125m", "opt-350m", "opt-1.3b", "opt-2.7b", "opt-6.7b"])
    parser.add_argument("--save", help="load model from huggingface, smoothquant and save model", action='store_true')
    parser.add_argument('--quantize', help="quantize using optimum onnx quantizer with avx512 backend", action='store_true')
    parser.add_argument('--use_cache', help="Enable caching support", action='store_true')
    parser.add_argument('--check', help="compare pytorch, onnx, quantized-onnx outputs", action='store_true')
    args = parser.parse_args()
    print(f"{args}")

    if args.save:
        model = OPTForCausalLM.from_pretrained("facebook/" + args.model_name)
        tokenizer = AutoTokenizer.from_pretrained("facebook/" + args.model_name)
        model.tokenizer = tokenizer 
        
        act_scales = torch.load(os.getenv("PYTORCH_AIE_PATH") + "/ext/smoothquant/act_scales/" + "%s.pt"%args.model_name)
        smooth.smooth_lm(model, act_scales, 0.5)
        print(model)
        
        prompt = ''.join(random.choices(string.ascii_lowercase + " ", k=model.config.max_position_embeddings))
        #inputs = tokenizer(prompt, return_tensors="pt")  # takes a lot of time
        inputs = tokenizer("What is meaning of life", return_tensors="pt") 
        print(f"inputs: {inputs}")
        print(f"inputs.input_ids: {inputs.input_ids}")
        for key in inputs.keys():
            print(inputs[key].shape)
            print(inputs[key])
        model_out = model(inputs.input_ids)
        print(f"{(model_out.logits.shape)=}")
        out_dir = "./%s_smoothquant"%args.model_name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        model.save_pretrained(out_dir+"/pytorch")

    elif args.quantize:
        dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False, use_symmetric_activations=True, 
                                                operators_to_quantize=["MatMul"],
                                                )
        print(dqconfig)
        if(args.use_cache):
            decoder_wp_quantizer = ORTQuantizer.from_pretrained("./%s_smoothquant"%args.model_name + "/onnx", file_name="decoder_with_past_model.onnx")
            decoder_wp_quantizer.quantize(save_dir="./%s_ortquantized"%args.model_name, quantization_config=dqconfig)

        quantizer = ORTQuantizer.from_pretrained("./%s_smoothquant"%args.model_name + "/onnx", file_name="decoder_model.onnx")
        model_quantized_path = quantizer.quantize( save_dir="./%s_ortquantized"%args.model_name, quantization_config=dqconfig )        

    elif args.check:
        model = OPTForCausalLM.from_pretrained("facebook/" + args.model_name)
        tokenizer = AutoTokenizer.from_pretrained("facebook/" + args.model_name)
        model.tokenizer = tokenizer 
        act_scales = torch.load(os.getenv("PYTORCH_AIE_PATH") + "/ext/smoothquant/act_scales/" + "%s.pt"%args.model_name)
        smooth.smooth_lm(model, act_scales, 0.5)
        
        prompt = ''.join(random.choices(string.ascii_lowercase + " ", k=model.config.max_position_embeddings))
        #inputs = tokenizer(prompt, return_tensors="pt") 
        inputs = tokenizer("What is meaning of life", return_tensors="pt") 
        print(f"inputs: {inputs}")
        print(f"inputs.input_ids: {inputs.input_ids}")
        for key in inputs.keys():
            print(inputs[key].shape)
            print(inputs[key])
        
        start = time.time()
        pytorch_out = model(inputs.input_ids).logits
        end = time.time()
        print(f"[PROFILE] Time to compute pytorch_out: {end-start}s")

        model = torch.ao.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8 )
        start = time.time()
        pytorch_quant_out = model(inputs.input_ids).logits
        end = time.time()
        print(f"[PROFILE] Time to compute pytorch_quant_out: {end-start}s")
        
        ort_sess = ort.InferenceSession('./%s_smoothquant/model.onnx'%args.model_name)
        start = time.time()
        onnx_out = ort_sess.run(None, {'onnx::Reshape_0': inputs.input_ids.numpy()})
        end = time.time()
        print(f"[PROFILE] Time to compute onnx_out: {end-start}s")
        onnx_out = torch.tensor(onnx_out[0])

        ort_sess = ort.InferenceSession('./%s_ortquantized/model_quantized.onnx'%args.model_name)
        start = time.time()
        onnx_quant_out = ort_sess.run(None, {'onnx::Reshape_0': inputs.input_ids.numpy()})
        end = time.time()
        print(f"[PROFILE] Time to compute onnx_quant_out: {end-start}s")
        onnx_quant_out = torch.tensor(onnx_quant_out[0])

        print(pytorch_out.shape)
        print(onnx_out.shape)
        print(onnx_quant_out.shape)

        print("***** Pytorch FP32 Output *****")
        print(pytorch_out)
        print("***** ONNX FP32 FP32 Output *****")
        print(onnx_out)
        
        print("***** Pytorch quantized Output *****")
        print(pytorch_quant_out)
        print("***** ONNX quantized Output *****")
        print(onnx_quant_out)
        
        res = torch.allclose(pytorch_out, onnx_out, atol=1)
        print(f"[RESULT] PyTorch FP32 vs ONNX FP32: allclose(atol=1) ? ... {res}")
        
        res = torch.allclose(pytorch_quant_out, onnx_quant_out, atol=5.2)
        print(f"[RESULT] PyTorch Quantized using torch.ao.quantize_dynamic() vs ONNX Quantized using Optimum ORT Quantizer: allclose(atol=5.2) ? ... {res}")
        print(f"[RESULT] Maximum pytorch_quant_out: {pytorch_quant_out.abs().max().item()}")
        print(f"[RESULT] Maximum onnx_quant_out: {onnx_quant_out.abs().max().item()}")
        err = pytorch_quant_out - onnx_quant_out
        print(f"[RESULT] Average err between pytorch_quant_out and onnx_quant_out: {err.abs().mean().item()}")
        print(f"[RESULT] Maximum err between pytorch_quant_out and onnx_quant_out: {err.abs().max().item()}")

    else:
        print(f"Nothing to do, exitting !!")
