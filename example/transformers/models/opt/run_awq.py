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
from modeling_opt_amd import OPTForCausalLM
#from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTAttention
import gc 

from pre_quant import run_awq, apply_awq
from quantizer import pseudo_quantize_model_weight, real_quantize_model_weight
from qmodule import WQLinear


set_seed(123)

def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained("facebook/" +  args.model_name)
    
    if args.awq == "none":
        model = OPTForCausalLM.from_pretrained("facebook/" + args.model_name, torch_dtype=torch.bfloat16) 
        model.tokenizer = tokenizer 
    else:
        if (args.task == "quantize"):
            model = OPTForCausalLM.from_pretrained("facebook/" + args.model_name, torch_dtype=torch.bfloat16) 
            model.tokenizer = tokenizer 
            print(model)
            q_config = {
                    "zero_point": True,
                    "q_group_size": 128,  } # whether to use group quantization
            
            Utils.print_model_size(model)

            if args.awq == 'load':
                print("Loading pre-computed AWQ results from", os.getenv("AWQ_CACHE"))
                awq_results = torch.load(os.getenv("AWQ_CACHE")  + "/%s-w%d-g128.pt"%(args.model_name, args.w_bit), map_location="cpu")
                apply_awq(model, awq_results)
                print("Quantization config:", q_config)
                real_quantize_model_weight(
                            model, w_bit=args.w_bit, q_config=q_config
                        )

                Utils.print_model_size(model)
                
                #for n, m in model.named_modules():
                #    if isinstance(m, WQLinear):
                #        print(f"After AWQ - {n} : {m.qweight.data.min()}  {m.qweight.data.max()}  {m.qweight.data.shape} {m.scales.shape}  {m.qzeros.shape} ")

            elif args.awq == 'run':
                print("Quantization config:", q_config)
                awq_results = run_awq(
                        model, tokenizer,
                        w_bit=args.w_bit, q_config=q_config,
                        n_samples=128, seqlen=512,
                    )
                torch.save(awq_results, "./%s-w%d-g128-generated.pt"%(args.model_name, args.w_bit))
                print(model)
                print("Saved AWQ results in ./%s-w%d-g128-generated.pt"%(args.model_name, args.w_bit))
                raise SystemExit 

            Utils.replace_node( model, 
                            WQLinear, 
                            qlinear.QLinearPerGrp, 
                            (), {'device':'cpu', 'w_bit':args.w_bit, 'group_size':128} )
            print(model)
            gc.collect()

            Utils.print_model_size(model)
            if True: # Quantize lm_head
                Utils.replace_node( model, 
                                    torch.nn.Linear, 
                                    qlinear.QLinearPerGrp, 
                                    (), {'device':'cpu', 'w_bit':args.w_bit, 'group_size':32} )
                print(model)
                gc.collect()
        
            torch.save(model, "pytorch_%s_w_bit_%d_awq_amd.pt"%(args.model_name, args.w_bit))
            print(f"Quantized and saved model: pytorch_{args.model_name}_w_bit_{args.w_bit}_awq_amd.pt")
            raise SystemExit

        else:
            ckpt = "./pytorch_%s_w_bit_%d_awq_amd.pt"%(args.model_name, args.w_bit)
            if not os.path.exists(ckpt):
                print(f"\n\n ***** Run --task quantize first to save quantized model ...!!! \n\n")
                raise SystemExit 
            
            model = torch.load(ckpt)
    Utils.print_model_size(model)
    collected = gc.collect()
    model.eval()
    model = model.to(torch.bfloat16)
    print(model)
    return model, tokenizer 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Different OPT model sizes", type=str, default="opt-125m", choices=["opt-125m", "opt-350m", "opt-1.3b", "opt-2.7b", "opt-6.7b", "opt-13b"])
    parser.add_argument("--target", help="cpu, aie, aie_emu", type=str, default="cpu", choices=["cpu", "aie_emu", "aie"])
    parser.add_argument('--dataset', help="Dataset - wikitext2-raw-v1, wikitext2-v1", type=str, default="raw", choices=["non-raw", "raw"])
    parser.add_argument('--awq', help="load awq scales, clips from pt or run awq", type=str, default="load", choices=["load", "run", "none"]) 
    parser.add_argument('--task', help="quantize: Apply AWQ and save ckpt; perplexity: Measure perplexity on wikitext2 dataset; benchmark: Benchmark latency w.r.t prompt length; benchmark_long: Benchmark long sequences (compare with flash attn); decode: Decode set of prompts;", type=str, default="decode", choices=["quantize", "decode", "benchmark", "benchmark_long", "perplexity"] )
    parser.add_argument('--w_bit', help="weight bit size", type=int, default=3, choices=[3, 4])
    args = parser.parse_args()
    print(f"{args}")
    if (args.w_bit == 4) and (args.target == "aie"):
        print(" ***** NOT IMPLEMENTED IN AIE ***** ")
        raise SystemExit

    dev = os.getenv("DEVICE")
    builtins.fuse_mlp = False

    log_dir = "./logs_%s"%args.model_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_awq_%s.log"%(args.model_name)
    
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.CRITICAL)
    builtins.amdopt = True

    model, tokenizer = load_model(args)

    for n, m in model.named_modules():
        if isinstance(m, qlinear.QLinearPerGrp):
            print(f"Preparing weights of layer : {n}")
            m.device = "aie"
            m.quantize_weights()
    
    print(model)
    Utils.print_model_size(model)

    if (args.task == "decode"):
        decode_prompts(model, tokenizer)
        logging.shutdown()
        out_file = log_file.replace(".log", "_profile.csv")
        out_file = open(out_file, "w")
        ProfileAIE.analyze_profiling(False, True, log_file, out_file)
        out_file.close()
        
    elif (args.task == "benchmark") or (args.task == "benchmark_long"):
        trainloader, testenc = get_wikitext2(tokenizer, nsamples=2, seqlen=2048)
        if (args.task == "benchmark"):
            seqlens =  [4, 8, 16, 32, 64, 128, 256]
        else:
            seqlens =  [512, 1024, 1536, 2000]
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
        ProfileAIE.analyze_profiling(False, True, log_file, out_file)
        out_file.close()

    elif (args.task == "perplexity"):
        start = time.time()
        perplexity(model, tokenizer, dataset=args.dataset)
        print(f"Time taken to measure ppl on RyzenAI: {time.time() - start}s")

    