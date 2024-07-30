#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import argparse
import builtins
import gc
import logging
import os
import time

import llm_eval
import llm_profile
import psutil
import torch
from ryzenai_llm_engine import RyzenAILLMEngine, TransformConfig
from ryzenai_llm_quantizer import QuantConfig, RyzenAILLMQuantizer

from transformers import set_seed

set_seed(123)

supported_models = [
    "facebook/opt-125m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "llama-2-7b",
    "llama-2-7b-chat",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-3b",
]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="mode name",
        type=str,
        default="facebook/opt-125m",
        choices=supported_models,
    )
    parser.add_argument(
        "--target", help="cpu, aie", type=str, default="aie", choices=["cpu", "aie"]
    )
    parser.add_argument(
        "--task",
        help="infershapes: shape inference; quantize:quantize the model; decode: Decode set of prompts; benchmark: Benchmark latency w.r.t prompt length; benchmark_long: Benchmark long sequences (compare with flash attn); perplexity: Measure perplexity on wikitext2 dataset; countgops: profile gops of the model for given workload; profilemodel: find performance metrics;",
        type=str,
        default="decode",
        choices=[
            "infershapes",
            "quantize",
            "decode",
            "benchmark",
            "benchmark_long",
            "countgops",
            "perplexity",
            "profilemodel",
            "mmlu",
        ],
    )
    parser.add_argument(
        "--precision",
        help="w8a8 or w8a16 - used for smoothquant, bf16 - runs on cpu",
        type=str,
        default="w8a8",
        choices=["bf16", "w8a8", "w8a16"],
    )
    parser.add_argument(
        "--flash_attention_plus",
        help="enable attention optimizations",
        action="store_true",
    )
    parser.add_argument(
        "--profilegemm",
        help="Log matmul times for prompt and token phases - supported only for AIE target",
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset - wikitext2-raw-v1, wikitext2-v1",
        type=str,
        default="raw",
        choices=["non-raw", "raw"],
    )
    parser.add_argument(
        "--fast_attention", help="enable fast attention", action="store_true"
    )

    args = parser.parse_args()
    print(f"{args}")

    dev = os.getenv("DEVICE")

    if "opt" in args.model_name:
        from llm_eval import OPTModelEval as CausalLMModel

        from transformers import AutoTokenizer as LMTokenizer
    elif "llama" in args.model_name:
        from llm_eval import LlamaModelEval as CausalLMModel

        from transformers import LlamaTokenizer as LMTokenizer
    elif "bloom" in args.model_name:
        from llm_eval import BloomModelEval as CausalLMModel

        from transformers import AutoTokenizer as LMTokenizer
    else:
        print(
            f"Create a EvalHarness Model for this model:{args.model_name} in llm_eval.py and rerun"
        )
        raise SystemExit

    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_%s.log" % (args.model_name.replace("/", "_"))
    logging.basicConfig(filename=log_file, filemode="w", level=logging.CRITICAL)

    qmodels_dir = "./quantized_models/"
    if not os.path.exists(qmodels_dir):
        os.makedirs(qmodels_dir)
    ckpt = qmodels_dir + "/quantized_%s_smoothquant.pth" % (
        args.model_name.replace("facebook/", "").replace("bigscience/", "")
    )

    ############################################################################################
    ### Step 1 - Model Quantization
    ### Step 2 - Model Transformation & Optimization
    ### Step 3 - Inference
    ############################################################################################

    if args.task == "quantize":
        if not os.path.exists(ckpt):
            model = CausalLMModel.from_pretrained(
                args.model_name, attn_implementation="eager"
            )
            model.tokenizer = LMTokenizer.from_pretrained(args.model_name)
            model.model_name = args.model_name.replace("facebook/", "").replace(
                "bigscience/", ""
            )
            print(model)
            quant_config = QuantConfig(
                quant_mode="smoothquant", model_name=model.model_name, dataset="raw"
            )

            ##############################################################
            ### Step 1 - Model Quantization
            model = RyzenAILLMQuantizer.quantize(model, quant_config=quant_config)
            print(model)
            ##############################################################

            torch.save(model, ckpt)
            print(f"\n\nSaved Quantized Model ... : {ckpt} !!! \n")
        else:
            print(f"\n\nFound quantized Model on disk : {ckpt} - nothing to do\n")

    else:
        if args.precision == "bf16":
            model = CausalLMModel.from_pretrained(
                args.model_name, torch_dtype=torch.bfloat16, attn_implementation="eager"
            )
            model.tokenizer = LMTokenizer.from_pretrained(args.model_name)
            model.model_name = args.model_name.replace("facebook/", "").replace(
                "bigscience/", ""
            )

        else:
            if not os.path.exists(ckpt):
                print(f"\n\nQuantized Model not available ... !!! \n")
                print(
                    f"\n\nRun with --task quantize and generate quantized model first \n"
                )
                raise SystemExit
            model = torch.load(ckpt)
            print(model)
            print(f"model.model_name: {model.model_name}")

        ##############################################################
        ### Step 2 - Model Transformation & Optimization
        transform_config = TransformConfig(
            flash_attention_plus=args.flash_attention_plus,
            fast_mlp=False,
            fast_attention=args.fast_attention,
            precision=args.precision,
            model_name=args.model_name,
            target=args.target,
            profilegemm=args.profilegemm,
        )
        model = RyzenAILLMEngine.transform(model, transform_config)
        print(model)
        ##############################################################

        ##############################################################
        ### Step 3 - Inference
        if args.task == "infershapes":
            if args.target != "cpu":
                print(f"\n\n *** Set --target to CPU to infer shapes *** exiting ... ")
            else:
                llm_eval.infer_linear_shapes(model)

        elif args.task == "decode":
            llm_eval.decode_prompts(model, log_file)

        elif (args.task == "benchmark") or (args.task == "benchmark_long"):
            llm_eval.benchmark(model, args.dataset, args.task, log_file)

        elif args.task == "countgops":
            llm_eval.count_gops(model)

        elif args.task == "perplexity":
            start = time.time()
            llm_eval.perplexity(model, dataset=args.dataset)
            print(f"Time taken to measure ppl on RyzenAI: {time.time() - start}s")

        elif args.task == "profilemodel":
            llm_profile.TorchModuleProfile.register_profile_hooks(model)
            if "code" in model.model_name.lower():
                prompt = "def fibonacci_recursive(n):"
            else:
                prompt = "What is meaning of life?"
            llm_eval.decode_prompt(model, model.tokenizer, prompt, max_new_tokens=11)
            llm_profile.TorchModuleProfile.generate_report(model)
            logging.shutdown()
            out_file = log_file.replace(".log", "_profile.csv")
            out_file = open(out_file, "w")
            llm_profile.ProfileLLM.analyze_profiling(log_file, out_file)
            out_file.close()
        elif args.task == "mmlu":
            start = time.time()
            llm_eval.mmlu(model)
            print(f"Time taken to measure mmlu on RyzenAI: {time.time() - start}s")
