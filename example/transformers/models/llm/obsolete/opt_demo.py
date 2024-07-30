#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import argparse
import gc
import os

import qlinear
import qlinear_experimental
import smooth
import torch
from modeling_opt_amd import OPTForCausalLM
from utils import Utils

from transformers import AutoTokenizer


def warmup(model, tokenizer):
    prompt = "What is the meaning of life?"
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="Different OPT model sizes",
        type=str,
        default="opt-1.3b",
        choices=["opt-125m", "opt-1.3b", "opt-2.7b", "opt-6.7b"],
    )
    parser.add_argument(
        "--target", help="cpu, aie", type=str, default="cpu", choices=["cpu", "aie"]
    )
    parser.add_argument(
        "--quant_mode",
        help="Quantization mode - none, or smoothquant+pytorch dynamic-quant",
        type=str,
        default="w8a8",
        choices=["none", "w8a8"],
    )
    parser.add_argument(
        "--load", help="Load quantized weights from checkpoint", action="store_true"
    )
    args = parser.parse_args()
    print(f"{args}")

    class OPTForCausalLMT(OPTForCausalLM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tokenizer = None

    tokenizer = AutoTokenizer.from_pretrained("facebook/" + args.model_name)

    if args.load:
        if args.quant_mode == "w8a8":
            model = torch.load("./quantized_%s_float32.pth" % args.model_name)
            model.eval()
        else:
            print("Mode not supported: if target=cpu, use without --load")
            raise SystemExit
    else:
        model = OPTForCausalLMT.from_pretrained("facebook/" + args.model_name)
        model.tokenizer = tokenizer
        if args.quant_mode == "w8a8":
            act_scales = torch.load(
                os.getenv("PYTORCH_AIE_PATH")
                + "/ext/smoothquant/act_scales/"
                + "%s.pt" % args.model_name
            )
            smooth.smooth_lm(model, act_scales, 0.5)

            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )

    collected = gc.collect()

    if (args.target == "aie") and (args.quant_mode == "none"):
        print("Mode not supported")
        raise SystemExit

    if args.target == "aie":  # ptdq
        node_args = ()
        quant_mode = "w8a8"
        qprofiler = False
        node_kwargs = {"device": "aie", "quant_mode": quant_mode}

        Utils.replace_node(
            model,
            torch.ao.nn.quantized.dynamic.modules.linear.Linear,
            qlinear.QLinear,
            node_args,
            node_kwargs,
        )
    else:
        pass

    collected = gc.collect()
    print(model)
    while True:
        print("*" * 20)
        prompt = input("Enter prompt or 'exit': ")
        if prompt == "exit":
            raise SystemExit

        length = input("Enter response length (1-1000): ")
        length = int(length)
        if length > 1000:
            length = 100

        inputs = tokenizer(prompt, return_tensors="pt")

        case_dict = {
            0: "Greedy search",
            1: "Stochastic search",
            2: "Contrastive search",
        }
        case = input(
            "Enter 0(greedy search) 1(stochastic search) or 2(contrastive search): "
        )
        case = int(case)
        if case not in [0, 1, 2]:
            case = 2
        print("Setting search to: ", case_dict[case])
        """
        0: determinstic - greedy and beam
        1: stochastic search - nucleus sampling
        2: contrastive search
        """
        if case == 0:
            output_sequences = model.generate(
                input_ids=inputs.input_ids,
                max_length=length,
            )

        elif case == 1:
            output_sequences = model.generate(
                input_ids=inputs.input_ids,
                max_length=length,
                do_sample=True,
                top_k=0,
                top_p=0.92,
            )

        else:  # case == 2:
            output_sequences = model.generate(
                input_ids=inputs.input_ids,
                max_length=length,
                do_sample=True,
                top_k=5,  # top K
                top_p=0.92,  # nucleus sampling
                penalty_alpha=0.6,
            )

        response = tokenizer.batch_decode(
            output_sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print(response)
