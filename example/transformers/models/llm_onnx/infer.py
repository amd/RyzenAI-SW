##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##

import argparse
import logging
import os
import sys
import time
from pathlib import PurePath

import onnxruntime as ort
import torch
from colorama import Back, Fore, Style

from transformers import AutoTokenizer, LlamaConfig, LlamaTokenizer

try:
    # Try importing wrapper class
    from include.llm import OPTORTModelEval as ORTOPTForCausalLM
    from include.llm import ORTModelEval as ORTModelForCausalLM
except ImportError:
    print(Fore.YELLOW + "[RyzenAI::W] Importing from optimum.onnxruntime" + Fore.RESET)
    from optimum.onnxruntime.modeling_decoder import ORTModelForCausalLM
    from optimum.onnxruntime.modeling_decoder import ORTOPTForCausalLM

from include.arg_parser import ArgumentParser
from include.logger import LINE_SEPARATER, RyzenAILogger, get_cachedir
from include.tasks import Tasks


def main():
    # Get argument parser
    parser = ArgumentParser()
    # Parse args
    args = parser.parse(sys.argv[1:])
    # Display args
    parser.display()

    # Directory with ONNX model
    onnx_model_dir = args.model_dir
    model_dir_base = PurePath(onnx_model_dir).parts[-1]

    # Create session option
    sess_options = ort.SessionOptions()

    # Enable ORT profile trace
    if args.ort_trace:
        sess_options.enable_profiling = True

    # Register Ryzen-AI Custom OP DLL
    if args.dll:
        if os.path.exists(args.dll):
            sess_options.register_custom_ops_library(args.dll)
        else:
            import errno

            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.dll)

    # Huggingface cache dir
    cache_dir = get_cachedir()

    # Max token generation
    default_max_length = 200 if "code" in args.model_name.lower() else 64
    max_length = args.max_length if args.max_length is not None else default_max_length

    # User settings
    if args.tokenizer:
        model_id = args.tokenizer
    else:
        model_id = args.model_name

    # Enable Logging
    logger = RyzenAILogger(model_dir_base)
    # Start logging
    logger.start()

    # ORT Logging level
    # Logging severity -> 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal
    ort.set_default_logger_severity(3)

    if args.target == "aie":
        ep = "VitisAIExecutionProvider"  # running on Ryzen-AI
    else:
        ep = "CPUExecutionProvider"  # running on CPU

    # default model arguments
    model_args = {}
    model_args["use_cache"] = True
    model_args["use_io_binding"] = True
    model_args["trust_remote_code"] = False
    model_args["provider"] = ep
    model_args["session_options"] = sess_options

    if args.target == "aie":
        model_args["provider_options"] = {
            "config_file": "vaip_config_transformers.json"
        }

    # Model/Tokenizer select
    if "llama" in args.model_name.lower():
        Tokenizer = LlamaTokenizer
        CausalLMModel = ORTModelForCausalLM
    elif "opt" in args.model_name.lower():
        Tokenizer = AutoTokenizer
        CausalLMModel = ORTOPTForCausalLM
    elif "qwen" in args.model_name.lower():
        from optimum.utils import NormalizedConfigManager, NormalizedTextConfig

        NormalizedConfigManager._conf["qwen2"] = NormalizedTextConfig
        Tokenizer = AutoTokenizer
        CausalLMModel = ORTModelForCausalLM
        model_args["use_cache"] = True
        model_args["use_io_binding"] = False
        model_args["trust_remote_code"] = True
    elif "chatglm" in args.model_name.lower():
        from optimum.utils import NormalizedConfigManager, NormalizedTextConfig

        NormalizedConfigManager._conf["chatglm"] = NormalizedTextConfig
        Tokenizer = AutoTokenizer
        CausalLMModel = ORTOPTForCausalLM
        model_args["use_cache"] = False
        model_args["use_io_binding"] = False
        model_args["trust_remote_code"] = True
    else:
        raise NotImplementedError("- Error: Only Llama & OPT are currently supported")

    # Tokenizer
    tokenizer = Tokenizer.from_pretrained(
        model_id,
        token=True,
        cache_dir=cache_dir,
        trust_remote_code=model_args["trust_remote_code"],
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Model
    model = CausalLMModel.from_pretrained(onnx_model_dir, **model_args)

    print("\n- Model: " + Fore.MAGENTA + "{}".format(model_dir_base) + Fore.RESET)
    print(LINE_SEPARATER)

    # Run task
    task_handler = Tasks(model, tokenizer)

    if args.task == "benchmark":
        # Run benchmark
        task_handler.benchmark(args.seqlen, args.max_new_tokens)
    elif args.task == "decode":
        if args.prompt:
            prompts = [args.prompt]
        else:
            if "code" in args.model_name.lower():
                from include.user_prompts import prompts_code as prompts
            else:
                from include.user_prompts import prompts as prompts

        # Run Decode
        task_handler.decode(prompts, max_length)
    else:
        task_handler.perplexity()

    # Stop logging
    logger.stop()

    if args.profile:
        from llm_profile import ProfileLLM

        out_file = logger.logfile.replace(".log", "_profile.csv")
        out_file = open(out_file, "w")
        ProfileLLM.analyze_profiling(logger.logfile, out_file)
        out_file.close()


if __name__ == "__main__":
    main()
