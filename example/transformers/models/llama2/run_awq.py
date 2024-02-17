#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#

import torch
import logging
import time
import argparse
import os
import psutil
from transformers import set_seed
from transformers import LlamaTokenizer

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

from modeling_llama_amd import LlamaForCausalLM, LlamaAttention

from pre_quant import run_awq, apply_awq
from quantizer import real_quantize_model_weight
from qmodule import WQLinear

set_seed(123)


def load_model(args):
    tokenizer = LlamaTokenizer.from_pretrained("./llama-2-wts-hf/7B_chat")
    if args.awq == "none":
        model = LlamaForCausalLM.from_pretrained("./llama-2-wts-hf/7B_chat", torch_dtype=torch.bfloat16) 
    
    else:
        ckpt = "pytorch_llama27b_w_bit_{}_awq{}_{}amd.pt".format(args.w_bit, "_fa" if args.flash_attention else "", "lm_" if args.lm_head else "")
        if args.task == "quantize":
            model = LlamaForCausalLM.from_pretrained("./llama-2-wts-hf/7B_chat", torch_dtype=torch.bfloat16)
            print(model)
            
            Utils.print_model_size(model)

            q_config = {
                    "zero_point": True,
                    "q_group_size": 128,  } # whether to use group quantization

            if args.awq == 'load':
                print("Loading pre-computed AWQ results from", os.getenv("AWQ_CACHE"))
                awq_results = torch.load(os.getenv("AWQ_CACHE")  + "/llama-2-7b-chat-w%d-g128.pt"%args.w_bit, map_location="cpu")
                apply_awq(model, awq_results)
                print("Quantization config:", q_config)
                real_quantize_model_weight(
                            model, w_bit=args.w_bit, q_config=q_config
                        )

                Utils.print_model_size(model)

                #for n, m in model.named_modules():
                #    if isinstance(m, WQLinear):
                #        print(f"AWQ Model load : {n} : {m.qweight.data.min()}  {m.qweight.data.max()}  {m.qweight.data.shape} {m.scales.shape} qzeros: {m.qzeros.shape} {m.qzeros.min()} {m.qzeros.max()}")

            elif args.awq == 'run':
                awq_results = run_awq(
                        model, tokenizer,
                        w_bit=args.w_bit, q_config=q_config,
                        n_samples=128, seqlen=512,
                    )
                torch.save(awq_results, "./llama-2-7b-chat-w%d-g128-generated.pt"%args.w_bit)
                print(model)
                print("Saved AWQ results in ./llama-2-7b-chat-w%d-g128-generated.pt"%args.w_bit)
                raise SystemExit
            
            if args.flash_attention:
                from llama_flash_attention import LlamaFlashAttention
                node_args = ()
                node_kwargs = {
                    'config': model.config,
                    'llama_name': "llama-2-wts-hf/7B_chat",
                    'flash_config_path': "../../ops/python/llama_flash_attention_config.json",
                    'device': "cpu", # args.target
                    'max_new_tokens': 11,
                    'quant_mode': "awq"
                }
                Utils.replace_node( model,
                                    LlamaAttention,
                                    LlamaFlashAttention,
                                    node_args, node_kwargs)
            
            Utils.replace_node( model, 
                                WQLinear, 
                                qlinear.QLinearPerGrp, 
                                (), {'device':'cpu', 'w_bit':args.w_bit, 'group_size':128} )
            print(model)
            gc.collect()

            Utils.print_model_size(model)
            if args.lm_head: # Quantize lm_head
                Utils.replace_node( model, 
                                    torch.nn.Linear, 
                                    qlinear.QLinearPerGrp, 
                                    (), {'device':'cpu', 'w_bit':args.w_bit, 'group_size':32} )
                print(model)
                gc.collect()

            torch.save(model, ckpt)
            print(f"Quantized and saved model: {ckpt}")
            raise SystemExit
        else:
            print(f"Loading from ckpt: {ckpt}")
            if not os.path.exists(ckpt):
                print(f"\n\n ***** Run --task quantize (with/without lm_head) first to save quantized model ...!!! \n\n")
                raise SystemExit 
            model = torch.load(ckpt)

    Utils.print_model_size(model)
    _ = gc.collect()
    model.eval()
    model = model.to(torch.bfloat16)
    print(model)
    return model, tokenizer 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="Dataset - wikitext2-raw-v1, wikitext2-v1", type=str, default="raw", choices=["non-raw", "raw"])
    parser.add_argument('--w_bit', help="weight bit size", type=int, default=3, choices=[3, 4])
    parser.add_argument('--awq', help="load awq scales, clips from pt or run awq", type=str, default="load", choices=["load", "run", "none"]) 
    parser.add_argument("--target", help="cpu, aie, aie_emu", type=str, default="cpu", choices=["cpu", "aie_emu", "aie"])
    parser.add_argument('--task', help="quantize: Apply AWQ and save ckpt; perplexity: Measure perplexity on wikitext2 dataset; benchmark: Benchmark latency w.r.t prompt length; benchmark_long: Benchmark long sequences (compare with flash attn); decode: Decode set of prompts;", type=str, default="decode", choices=["quantize", "decode", "benchmark", "benchmark_long", "perplexity"] )
    parser.add_argument('--flash_attention', help="Enable flash attention", action='store_true')
    parser.add_argument('--lm_head', help="Enable PerGrp quantization of lm_head layer", action='store_true')
    parser.add_argument('--num_torch_threads', help="Number of torch threads", type=int, default=8, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    args = parser.parse_args()
    print(f"{args}")
    dev = os.getenv("DEVICE")

    if dev == "stx":
        p = psutil.Process()
        p.cpu_affinity([0, 1, 2, 3])
    torch.set_num_threads(args.num_torch_threads)
    
    log_dir = "./logs_awq_7B_chat"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_awq_7B_chat.log"

    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.CRITICAL)

    model, tokenizer = load_model(args)

    if args.awq != "none":
        for n, m in model.named_modules():
            if isinstance(m, qlinear.QLinearPerGrp):
                print(f"Preparing weights of layer : {n}")
                m.device = "aie"
                m.quantize_weights()

    print(model)
    Utils.print_model_size(model)
    
    warmup(model, tokenizer)

    if (args.task == "decode"):
        decode_prompts(model, tokenizer, max_new_tokens=11)
        logging.shutdown()
        out_file = log_file.replace(".log", "_profile.csv")
        out_file = open(out_file, "w")
        ProfileAIE.analyze_profiling(False, True, log_file, out_file)
        out_file.close()

    elif (args.task == "benchmark") or (args.task == "benchmark_long"):
        #print(model.config.max_position_embeddings) # 2048
        trainloader, testenc = get_wikitext2(tokenizer, nsamples=2, seqlen=4096)
        if (args.task == "benchmark"):
            seqlens =  [4, 8, 16, 32, 64, 128, 256]
        else:
            seqlens =  [512, 1024, 1536, 2048, 3000, 4096] 
        input_ids = next(iter(trainloader))[0][:, :4096]
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
