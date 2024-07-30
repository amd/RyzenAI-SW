##
## Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
##

import argparse

from colorama import Fore, init

from .logger import LINE_SEPARATER


# Command line parser
class ArgumentParser:
    def __init__(self):
        # Create a parser
        parser = argparse.ArgumentParser(description="LLM Inference on Ryzen-AI")
        # Add args
        # Model/tokenizer paths
        parser.add_argument("--model_dir", required=True, help="Model directory path")
        parser.add_argument(
            "--draft_model_dir",
            required=False,
            default=None,
            help="Draft Model directory path for speculative decoding",
        )

        parser.add_argument(
            "--model_name",
            help="model name",
            type=str,
            required=True,
            choices=[
                "facebook/opt-125m",
                "facebook/opt-1.3b",
                "facebook/opt-2.7b",
                "facebook/opt-6.7b",
                "meta-llama/Llama-2-7b-hf",
                "Qwen/Qwen1.5-7B-Chat",
                "THUDM/chatglm3-6b",
                "codellama/CodeLlama-7b-hf",
            ],
        )

        parser.add_argument(
            "--tokenizer",
            required=False,
            default=None,
            help="Path to the tokenizer (Optional).",
        )
        # Custom OP DLL
        parser.add_argument(
            "--dll",
            required=False,
            default=None,
            help="Path to the Ryzen-AI Custom OP Library",
        )
        # Target
        parser.add_argument(
            "--target",
            required=False,
            default="cpu",
            choices=["cpu", "aie"],
            help="Target device (CPU or Ryzen-AI)",
        )
        # Tasks
        parser.add_argument(
            "--task",
            required=False,
            default="decode",
            choices=["decode", "benchmark", "perplexity"],
            help="Run model with a specified task",
        )
        # Benchmark with sequence length
        parser.add_argument(
            "--seqlen",
            required=False,
            nargs="+",
            type=int,
            default=[8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 4096],
            help="Input Sequence length for benchmarks",
        )
        # New tokens to be generated
        parser.add_argument(
            "--max_new_tokens",
            required=False,
            default=11,
            type=int,
            help="Number of new tokens to be generated",
        )
        # Profiling
        parser.add_argument(
            "--ort_trace",
            required=False,
            default=False,
            action="store_true",
            help="Enable ORT Trace dump",
        )
        # Analyze trace
        parser.add_argument(
            "--view_trace",
            required=False,
            default=False,
            action="store_true",
            help="Display trace summary on console",
        )
        # Decode Max length
        parser.add_argument(
            "--prompt", required=False, default=None, type=str, help="User prompt"
        )

        # Decode Max length
        parser.add_argument(
            "--max_length",
            required=False,
            type=int,
            help="Number of tokens to be generated",
        )
        # Enable profiling summary
        parser.add_argument(
            "--profile",
            required=False,
            default=False,
            action="store_true",
            help="Enable profiling summary",
        )
        # Enable power profiling (internal)
        parser.add_argument(
            "--power_profile",
            required=False,
            default=False,
            action="store_true",
            help="Enable power profiling via AGM",
        )
        # Enable argument display
        parser.add_argument(
            "-v",
            "--verbose",
            required=False,
            default=False,
            action="store_true",
            help="Enable argument display",
        )
        # Save options
        self.parser = parser

    # Parse args
    def parse(self, options):
        self.args = self.parser.parse_args(options)
        return self.args

    # Print args
    def display(self):
        if self.args.verbose:
            print("{}\n- Parsed Args\n{}".format(LINE_SEPARATER, LINE_SEPARATER))
            for arg, value in vars(self.args).items():
                print(
                    "- {:20}: ".format(arg)
                    + Fore.CYAN
                    + "{}".format(value)
                    + Fore.RESET
                )
            print("{}".format(LINE_SEPARATER))
