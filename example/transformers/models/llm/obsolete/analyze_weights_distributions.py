#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import argparse
import logging
import os

import qlinear_experimental
import torch
from utils import Utils

from transformers import AutoTokenizer, OPTForCausalLM

if __name__ == "__main__":
    """
    User QLinearExperimentalCPU class to iterate through weights and analyze them
    This was used for initial version of quantization"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        help="Different OPT model sizes",
        type=str,
        default="opt-1.3b",
        choices=["opt-125m", "opt-350m", "opt-1.3b", "opt-2.7b", "opt-6.7b"],
    )
    args = parser.parse_args()
    print(f"{args}")

    log_dir = "./logs_%s" % args.model_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_%s_analyze_weights.log" % (args.model_name)
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.CRITICAL,
    )

    model = OPTForCausalLM.from_pretrained("facebook/" + args.model_name)
    tokenizer = AutoTokenizer.from_pretrained("facebook/" + args.model_name)

    node_args = ()
    node_kwargs = {"quant_mode": None}
    Utils.replace_node(
        model,
        torch.nn.Linear,
        qlinear_experimental.QLinearExperimentalCPU,
        node_args,
        node_kwargs,
    )
    logging.critical(f"[PROFILE][CPU] Linear num of replaced nodes: {Utils.node_count}")

    print(model)

    Utils.analyze_weights(model)
    logging.critical(
        f"[RANGES] QLinearExperimentalCPU nodes visited : {Utils.node_count}"
    )
