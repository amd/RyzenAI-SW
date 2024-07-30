#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import argparse
import logging
import os

import qlinear_experimental
import torch
from utils import Utils

from transformers import AutoTokenizer, OPTForCausalLM, set_seed

set_seed(123)

prompt = "The First Cataract at Aswān, where the riverbed is turned into rapids by a belt of granite, was the country’s only well-defined boundary within a populated area. To the south lay the far less hospitable area of Nubia, in which the river flowed through low sandstone hills that in most regions left only a very narrow strip of cultivable land. Nubia was significant for Egypt’s periodic southward expansion and for access to products from farther south. West of the Nile was the arid Sahara, broken by a chain of oases some 125 to 185 miles (200 to 300 km) from the river and lacking in all other resources except for a few minerals. The eastern desert, between the Nile and the Red Sea, was more important, for it supported a small nomadic population and desert game, contained numerous mineral deposits, including gold, and was the route to the "

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="Different OPT model sizes",
        type=str,
        default="opt-1.3b",
        choices=["opt-125m", "opt-350m", "opt-1.3b", "opt-2.7b", "opt-6.7b"],
    )
    parser.add_argument(
        "--seqlen", help="Enter length of input prompt", type=int, default=8
    )
    args = parser.parse_args()
    print(f"{args}")

    if args.seqlen > 2048:  # max for OPT
        print("Enter a number less than 2048 and try again")
        raise SystemExit

    log_dir = "logs_%s" % args.model_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = log_dir + "/log_%s_extract_shapes.log" % (args.model_name)
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.CRITICAL,
    )

    tokenizer = AutoTokenizer.from_pretrained("facebook/" + args.model_name)
    if False:
        model = OPTForCausalLM.from_pretrained("facebook/" + args.model_name)
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    else:
        model = torch.load("./quantized_%s_float32.pth" % args.model_name)
        model.eval()

    node_args = ()
    node_kwargs = {"quant_mode": 1, "collect_stats": True}
    Utils.replace_node(
        model,
        torch.ao.nn.quantized.dynamic.modules.linear.Linear,
        qlinear_experimental.QLinearExperimentalCPU,
        node_args,
        node_kwargs,
    )

    longprompt = ""
    for i in range(8):
        longprompt += prompt

    inputs = tokenizer(longprompt, return_tensors="pt")
    inputs.input_ids = inputs.input_ids[:, : args.seqlen]
    inputs.attention_mask = inputs.attention_mask[:, : args.seqlen]
    generate_ids = model.generate(inputs.input_ids, max_new_tokens=128)
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    all_shapes = {}
    for name, module in model.named_modules():
        if isinstance(module, qlinear_experimental.QLinearExperimentalCPU):
            logging.critical(
                f"Shapes of {name} {module._get_name()}: {module.mmul_shapes}"
            )
            for key in module.mmul_shapes.keys():
                if all_shapes.get(key) is None:
                    all_shapes[key] = 0
                all_shapes[key] += module.mmul_shapes[key]

    logging.critical("\nCumulative matmul shapes observed in the model ...")
    for key in all_shapes.keys():
        logging.critical(f"Module: {key} Shapes: {all_shapes[key]}")
    logging.critical("\n\n")
    print(f"Extracted shapes to {log_dir} + / log_{args.model_name}_extract_shapes.log")
