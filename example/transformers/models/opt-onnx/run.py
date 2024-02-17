#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import argparse
import logging
import time
import gc
import os
import builtins
import sys 
sys.path.append("../opt") 
from model_utils import (
    warmup, 
    decode_prompt,
    decode_prompts,
    get_wikitext2,
    perplexity,
)
from transformers import set_seed
from transformers import AutoTokenizer
import pathlib
from profiler import ProfileAIE

import smooth
import onnxruntime as ort
import psutil

CURRENT_DIR = pathlib.Path(__file__).parent
print(CURRENT_DIR.parent)
config_file_path = CURRENT_DIR / "vaip_config.json"

set_seed(123)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_path",help="Local directory path to ONNX model", default="")
    parser.add_argument("--target", help="cpu, aie", type=str, default="aie", choices=["cpu", "aie"])
    parser.add_argument('--perplexity', help="Calculate perplexity on wikitext2 instead of decoding prompts", action='store_true')
    parser.add_argument("--model_name", help="Different OPT model sizes", type=str, default="opt-1.3b", choices=["opt-125m", "opt-350m", "opt-1.3b", "opt-2.7b"])
    parser.add_argument('--quant_mode', help="Quantization mode - w8a8", type=str, default="w8a8", choices=["w8a8", "none"])
    parser.add_argument('--dataset', help="wikitext2-raw-v1, wikitext2-v1", type=str, default="raw", choices=["non-raw", "raw"])
    parser.add_argument('--amdort', help="Use ORT from local folder - with profile instrumentation", action='store_true')
    parser.add_argument('--profile', help="Log matmul times for prompt and token phases - supported only for AIE target", action='store_true')
    parser.add_argument('--impl', help="Choose between different implementations for aie target", type=str, default="v0", choices=["v0", "v1"])
    parser.add_argument('--task', help="perplexity: Measure perplexity on wikitext2 dataset; benchmark: Benchmark latency w.r.t prompt length; benchmark_long: Benchmark long sequences (compare with flash attn); decode: Decode set of prompts", type=str, default="decode", choices=["decode", "benchmark", "benchmark_long", "perplexity"] )
    parser.add_argument('--num_intra_op_threads', help="Number of intra op num threads", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--num_inter_op_threads', help="Number of inter op num threads", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--intra_op_spinning', help="Disable intra op spinning", type=str, default="0", choices=["0", "1"])
    parser.add_argument('--inter_op_spinning', help="Disable inter op spinning", type=str, default="0", choices=["0", "1"])
    args = parser.parse_args()
    print(f"{args}")

    dev = os.getenv("DEVICE")
    if dev == "stx":
        p = psutil.Process()
        p.cpu_affinity([0, 1, 2, 3])
    
    # Set implementation for aie target
    builtins.impl = args.impl
    builtins.quant_mode = args.quant_mode

    log_dir = "./logs_%s"%args.model_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_%s_%s.log"%(args.model_name, args.target)
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.CRITICAL)

    if args.target == "aie":
        provider = "VitisAIExecutionProvider"
        provider_options = {'config_file': str(config_file_path)} 
    else:
        provider = "CPUExecutionProvider"
        provider_options = {} 
    
    path = "facebook/"
    if args.local_path != "":
        path = args.local_path
       
    if args.amdort:
        from modeling_ort_amd import ORTModelForCausalLM
    else:
        from optimum.onnxruntime import ORTModelForCausalLM
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.num_intra_op_threads
    
    if args.num_inter_op_threads:
        sess_options.execution_mode  = ort.ExecutionMode.ORT_PARALLEL        
    
    sess_options.inter_op_num_threads = args.num_inter_op_threads  
    
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", args.intra_op_spinning)
    sess_options.add_session_config_entry("session.inter_op.allow_spinning", args.inter_op_spinning)
    
    model = ORTModelForCausalLM.from_pretrained(path, provider=provider,use_cache=True, use_io_binding=False, session_options=sess_options, provider_options=provider_options)
    tokenizer = AutoTokenizer.from_pretrained("facebook/" +  args.model_name)

    collected = gc.collect()
    
    warmup(model, tokenizer)

    if (args.task == "decode"):
        decode_prompts(model, tokenizer)  
        logging.shutdown()
        out_file = log_file.replace(".log", "_profile.csv")
        out_file = open(out_file, "w")
        ProfileAIE.analyze_profiling(args.profile, args.amdort, log_file, out_file)
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
        ProfileAIE.analyze_profiling(args.profile, args.amdort, log_file, out_file)
        out_file.close()
        
    elif args.task == "perplexity":
        start = time.time()
        perplexity(model, tokenizer, dataset=args.dataset, framework="onnxrt")
        print(f"Time taken to measure ppl on RyzenAI: {time.time() - start}s")
