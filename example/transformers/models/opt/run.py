#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import torch
import logging 
import time 
import argparse 
import psutil
from transformers import set_seed
from transformers import AutoTokenizer

import os
import builtins

import qlinear 

from utils import Utils
from model_utils import (
    warmup, 
    decode_prompt,
    decode_prompts,
    get_wikitext2,
    perplexity,
)
from profiler import ProfileAIE

import gc 
import smooth
import psutil

set_seed(123)

def check_config(args):
    if  ((args.load == True) and (args.quant_mode == "none")) or \
        ((args.load == True) and (args.amdopt == True)) or \
        ((args.load == True) and (args.quant_mode == "w8a8") and (args.smoothquant == False))  or \
        ((args.dtype == "bfloat16") and (args.target == "cpu") and (args.quant_mode == "w8a8")) or \
        ((args.quant_mode == "none") and (args.target == "aie")) :
        print(f" *** MODE NOT SUPPORTED *** : check help and readme")
        raise SystemExit


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained("facebook/" +  args.model_name)
    if args.load:
        if  (args.quant_mode == "w8a8") and \
            (args.flash_attention == False) and\
            (args.smoothquant == True):
            model = torch.load("./quantized_%s_%s.pth"%(args.model_name, args.dtype))
        else:
            print(f" *** MODE NOT SUPPORTED *** : rerun without --load")
            raise SystemExit
    else:
        if args.amdopt:
            from modeling_opt_amd import OPTForCausalLM, OPTAttention
        else:
            from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTAttention
        class OPTForCausalLMT(OPTForCausalLM):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.tokenizer = None  
        
        if args.dtype == "bfloat16":
            model = OPTForCausalLMT.from_pretrained("facebook/" + args.model_name, torch_dtype=torch.bfloat16)        
        else:
            model = OPTForCausalLMT.from_pretrained("facebook/" + args.model_name) 
                
        model.tokenizer = tokenizer 
        print(model)
        if (args.smoothquant == True):
            act_scales = torch.load(os.getenv("PYTORCH_AIE_PATH") + "/ext/smoothquant/act_scales/" + "%s.pt"%args.model_name)
            smooth.smooth_lm(model, act_scales, 0.5)
            print(f"SmoothQuant enabled ...")
        if args.dtype == "bfloat16":
            model = model.to(torch.bfloat16)
        
        print(model)
        if (args.quant_mode == "w8a8"):
            torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True )
        print(model)
        if (args.flash_attention == True):
            from opt_flash_attention import OPTFlashAttention
            node_args = ()
            if args.quant_mode == "w8a8":
                quant = 1
            else:
                quant = None
            node_kwargs = {
                'quant_mode': quant,
                'device' : args.target,
                'dtype': args.dtype, 
                'impl': args.impl,
                'embed_dim': model.config.hidden_size,
                'num_heads': model.config.num_attention_heads,
                'opt_name': "facebook/" + args.model_name, }
            Utils.replace_node( model,
                                OPTAttention,
                                OPTFlashAttention,
                                node_args, node_kwargs   )
    collected = gc.collect()
    model.eval()
    print(model)
    return model, tokenizer 


def model_transform(model, args):
    if args.quant_mode == "w8a8":
        if (args.target == "aie") :
            node_args = ()
            quant_mode = 1
            node_kwargs = {'device': 'aie', 'quant_mode':args.quant_mode, 'profiler':args.profile, 'dtype':args.dtype, 'impl':args.impl}
            if (args.no_accelerate == False):
                Utils.replace_node(    model, 
                                        torch.ao.nn.quantized.dynamic.modules.linear.Linear,
                                        qlinear.QLinear, 
                                        node_args, node_kwargs )
            else:
                Utils.replace_node(     model, 
                                        torch.ao.nn.quantized.dynamic.modules.linear.Linear, 
                                        qlinear_experimental.QLinearExperimentalPyTiling, 
                                        node_args, node_kwargs )
        elif (args.target == "customcpu"):
            node_args = ()
            node_kwargs = {'quant_mode':1, 'requantize_out_scale':128}
            Utils.replace_node( model, 
                                torch.ao.nn.quantized.dynamic.modules.linear.Linear, 
                                qlinear_experimental.QLinearExperimentalCPU, 
                                node_args, node_kwargs )
        else: #target == "cpu":
            pass 
    else: # quant_mode == None 
        if (args.target == "aie") :
            print(f"*** FP32 MODEL ON AIE NOT SUPPORTED ***")
            raise SystemExit 
        elif (args.target == "customcpu"):
            quant_mode = None
            node_args = ()
            node_kwargs = {'quant_mode':quant_mode}
            Utils.replace_node( model, 
                                torch.nn.Linear, 
                                qlinear_experimental.QLinearExperimentalCPU, 
                                node_args, node_kwargs )
        else: #target == "cpu":
            pass 
        """
        if args.quant_mode == ptsq: - for aie and customcpu
            scales = torch.load(".\calibration_%s\quantized_%s_act_scales.pt"%(args.model_name, args.model_name))
            for name, module in model.named_modules():
                if name in scales.keys():
                    print(name, module.x_scale, scales[name])
                    module.x_scale = scales[name]
                    print(name, module.x_scale, scales[name])
        """
    print(model)
    return model 
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Different OPT model sizes", type=str, default="opt-1.3b", choices=["opt-125m", "opt-350m", "opt-1.3b", "opt-2.7b", "opt-6.7b", "opt-13b", "opt-30b"])
    parser.add_argument("--target", help="cpu, custumcpu, aie", type=str, default="aie", choices=["cpu", "customcpu", "aie"])
    parser.add_argument('--dtype', help="All ops other than linear ops in bfloat16 or float32", type=str, default="float32", choices=["bfloat16", "float32"])
    parser.add_argument('--quant_mode', help="Quantization mode - w8a8", type=str, default="w8a8", choices=["w8a8", "none"]) # ptsq not suported
    parser.add_argument('--smoothquant', help="Enable smoothquant", action='store_true')
    parser.add_argument('--amdopt', help="Use OPT from local folder - with profile instrumentation: use without --load", action='store_true')
    parser.add_argument('--profile', help="Log matmul times for prompt and token phases - supported only for AIE target", action='store_true')
    parser.add_argument('--no-accelerate', help="Use tiling/padding in c++ or python with AIE target", action='store_true')
    parser.add_argument('--dataset', help="Dataset - wikitext2-raw-v1, wikitext2-v1", type=str, default="raw", choices=["non-raw", "raw"])
    parser.add_argument('--load', help="Load quantized weights from checkpoint", action='store_true')
    parser.add_argument('--flash_attention', help="Enable flash attention", action='store_true')
    parser.add_argument('--num_torch_threads', help="Number of torch threads", type=int, default=2, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--task', help="perplexity: Measure perplexity on wikitext2 dataset; benchmark: Benchmark latency w.r.t prompt length; benchmark_long: Benchmark long sequences (compare with flash attn); decode: Decode set of prompts; torchprofile: profile using torch profiler for decode", type=str, default="decode", choices=["decode", "benchmark", "benchmark_long", "perplexity", "torchprofile", "opsprofile"] )
    parser.add_argument('--impl', help="Choose between different implementations for aie target", type=str, default="v0", choices=["v0", "v1"])
    parser.add_argument('--fuse_mlp', help="Enable MLP fusion", action='store_true')
    args = parser.parse_args()
    print(f"{args}")
    check_config(args)

    dev = os.getenv("DEVICE")
    if dev == "stx":
        p = psutil.Process()
        p.cpu_affinity([0, 1, 2, 3])
    torch.set_num_threads(args.num_torch_threads)

    log_dir = "./logs_%s"%args.model_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if args.flash_attention: 
        log_file = log_dir + "/log_%s_%s_%s_%s_flashattn.log"%(args.model_name, args.target, args.quant_mode, args.dtype) 
    else:
        log_file = log_dir + "/log_%s_%s_%s_%s.log"%(args.model_name, args.target, args.quant_mode, args.dtype)
    
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.CRITICAL)
    
    # Set amdopt option for flash attention import
    builtins.amdopt = args.amdopt
    builtins.fuse_mlp = args.fuse_mlp
    builtins.impl = args.impl
    builtins.model_name = args.model_name

    # Step 1 - Load model
    model, tokenizer = load_model(args)

    # Step 2 - Model transformation to use target device
    model = model_transform(model, args) 
    collected = gc.collect()

    # Step 3 - Warmup
    warmup(model, tokenizer)

    # Step 4 - Do one of the following after warmup()
    if (args.task == "decode"):
        decode_prompts(model, tokenizer)
        logging.shutdown()
        out_file = log_file.replace(".log", "_profile.csv")
        out_file = open(out_file, "w")
        ProfileAIE.analyze_profiling(args.profile, args.amdopt, log_file, out_file)
        out_file.close()
        
    elif (args.task == "benchmark") or (args.task == "benchmark_long"):
        trainloader, testenc = get_wikitext2(tokenizer, nsamples=2, seqlen=2048)
        if (args.task == "benchmark"):
            seqlens =  [4, 8, 16, 32, 64, 128, 256]
        else:
            seqlens =  [512, 717, 1024, 1536, 1801, 2000]
        input_ids = next(iter(trainloader))[0][:, :model.config.max_position_embeddings] # max of opt is 2048
        for seqlen in seqlens:
            logging.critical("*"*40)
            print("*"*40)
            print(f"Benchmarking for {seqlen} tokens ...")
            input_ids_test = input_ids[:, :seqlen]
            decode_prompt(model, tokenizer, prompt=None, input_ids = input_ids_test, max_new_tokens=11)
            
        logging.shutdown()
        out_file = log_file.replace(".log", "_profile.csv")
        out_file = open(out_file, "w")
        ProfileAIE.analyze_profiling(args.profile, args.amdopt, log_file, out_file)
        out_file.close()

    elif (args.task == "perplexity"):
        start = time.time()
        perplexity(model, tokenizer, dataset=args.dataset)
        print(f"Time taken to measure ppl on RyzenAI: {time.time() - start}s")
    elif (args.task ==  "torchprofile"):
        with torch.profiler.profile( activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True ) as prof:
            with torch.profiler.record_function("model_inference"):
                decode_prompts(model, tokenizer)                    
        with open(log_file.replace(".log", "_profile_torch.log"), 'w') as f:
            print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=-1), file=f)    
    