#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import argparse
import logging
import time
import gc
import os
import sys 
sys.path.append("../opt-pytorch") 
from model_utils import warmup, decode_prompts, perplexity
from transformers import set_seed
from transformers import AutoTokenizer
import pathlib
from profiler import ProfileAIE

import smooth

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
    parser.add_argument('--dataset', help="wikitext2-raw-v1, wikitext2-v1", type=str, default="raw", choices=["non-raw", "raw"])
    parser.add_argument('--amdort', help="Use ORT from local folder - with profile instrumentation", action='store_true')

    args = parser.parse_args()
    print(f"{args}")

    log_dir = "./logs_%s"%args.model_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_%s_cpu.log"%(args.model_name)
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
    
    model = ORTModelForCausalLM.from_pretrained(path, provider=provider,use_cache=True, use_io_binding=False, provider_options=provider_options)
    tokenizer = AutoTokenizer.from_pretrained("facebook/" +  args.model_name)

    collected = gc.collect()
    
    warmup(model, tokenizer)
    if args.perplexity == True:
        start = time.time()
        perplexity(model, tokenizer, dataset=args.dataset, framework="onnxrt")
        print(f"Time taken to measure ppl on RyzenAI: {time.time() - start}s")
    else:
        decode_prompts(model, tokenizer)
        logging.shutdown()
        out_file = log_file.replace(".log", "_profile.csv")
        out_file = open(out_file, "w")
        ops_profiler = True
        ProfileAIE.analyze_profiling(ops_profiler, args.amdort, log_file, out_file)
        out_file.close()
