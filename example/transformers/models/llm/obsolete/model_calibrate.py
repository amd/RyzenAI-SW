#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#

import argparse
import logging
import os

import numpy as np
import qlinear_experimental
import torch
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, save
from model_utils import calibrate
from utils import Utils

from transformers import AutoTokenizer, OPTForCausalLM, set_seed

set_seed(123)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="Different OPT model sizes",
        type=str,
        default="opt-1.3b",
        choices=[
            "opt-125m",
            "opt-350m",
            "opt-1.3b",
            "opt-2.7b",
            "opt-6.7b",
            "opt-13b",
            "opt-30b",
        ],
    )
    args = parser.parse_args()
    print(f"{args}")

    steps = {"1to5": True, "6to10": True}

    log_dir = "calibration_%s" % args.model_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = log_dir + "/log_calibrate_%s.log" % (args.model_name)
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.CRITICAL,
    )

    tokenizer = AutoTokenizer.from_pretrained("facebook/" + args.model_name)

    if steps["1to5"] is True:
        model = OPTForCausalLM.from_pretrained("facebook/" + args.model_name)

        print(
            f"Step 1 : Insert observers, analyze model to calculate scaling factors of activations using wikitext2-raw test set"
        )

        Utils.register_shapes_hook_linear(model)
        Utils.register_dist_hook_linear(model)

        calibrate(model, tokenizer)
        print(f"Step 2 : Calibrated on wikitext2-raw test set")

        all_shapes = Utils.extract_shapes_linear()
        logging.critical(f"\n\n")
        for key in all_shapes.keys():
            logging.critical(f"Shape: {key} occurances: {all_shapes[key]}")

        keys = Utils.linear_inp.keys()
        graph = figure(
            x_range=list(keys),
            height=1200,
            width=2400,
            title="%s input data ranges" % args.model_name,
        )
        output_file(log_dir + "/calibration_%s_inputs.html" % args.model_name)
        logging.critical(f"\n\n")
        act_scales = {}
        for i, key in enumerate(keys):
            data = np.array(Utils.linear_inp[key])
            case = 1
            if case == 1:
                scale = np.max(np.abs(data)) / 128.0
                idx = [i + 0.5 for k in range(len(data))]
                graph.scatter(idx, data)
            else:
                data_mean = np.mean(data)
                hi = np.percentile(data, 99.9)
                lo = np.percentile(data, 0.1)
                scale = np.max(np.abs([lo, hi])) / 128.0
                newd = [hi, lo, data_mean]
                idx = [i + 0.5 for k in range(len(newd))]
                graph.scatter(idx, newd)
            logging.critical(
                "Input activation: Layer-%d: %s : data.min():%.4f data.max():%.4f scale:%4f"
                % (i, key, data.min(), data.max(), scale)
            )

            act_scales[key] = scale

        print(
            f"Step 3 : Saved input activation range plots in calibration_{args.model_name}_inputs.html"
        )
        torch.save(
            act_scales, log_dir + "/quantized_%s_act_scales.pt" % args.model_name
        )

        graph.xaxis.axis_label = "layers"
        graph.yaxis.axis_label = "input activation ranges"
        graph.xaxis.major_label_orientation = "vertical"
        save(graph)
        print(
            f"Step 4 : Saved activation scale factors in quantized_{args.model_name}_act_scales.pt"
        )

        keys = Utils.linear_outp.keys()
        graph = figure(
            x_range=list(keys),
            height=1200,
            width=2400,
            title="%s output data ranges" % args.model_name,
        )
        output_file(log_dir + "/calibration_%s_outputs.html" % args.model_name)
        logging.critical(f"\n\n")
        for i, key in enumerate(keys):
            data = np.array(Utils.linear_outp[key])
            idx = [i + 0.5 for k in range(len(data))]
            logging.critical(
                "Output activation: Layer-%d: %s : data.min():%.4f data.max():%.4f"
                % (i, key, data.min(), data.max())
            )
            graph.scatter(idx, data)

        graph.xaxis.axis_label = "layers"
        graph.yaxis.axis_label = "output activation ranges"
        graph.xaxis.major_label_orientation = "vertical"
        save(graph)
        print(
            f"Step 5 : Saved output activation data range plots in calibration_{args.model_name}_outputs.html"
        )
        logging.critical(f"\n\n")

    if steps["6to10"] is True:
        if False:
            print(f"Step 6 : Perform PTDQ")
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            print(f"Step 6 : Loading dynamic quantized model")
            model = torch.load("./quantized_%s.pth" % args.model_name)
            model.eval()

        print(
            f"Step 7 : Insert custom Linear node to analyze int32 accumulator overflow"
        )
        node_args = ()
        node_kwargs = {"quant_mode": 1, "collect_stats": True}
        Utils.replace_node(
            model,
            torch.ao.nn.quantized.dynamic.modules.linear.Linear,
            qlinear_experimental.QLinearExperimentalCPU,
            node_args,
            node_kwargs,
        )

        # calibrate
        calibrate(model, tokenizer)
        print(f"Step 8 : Calibrated on wikitext2-raw test set")
        logging.critical(
            "\n\nInserted qlinear_experimental.QLinearExperimentalCPU in place of torch.nn.Linear"
        )

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

        abs_max = 0
        all_int32_min = []
        all_int32_max = []
        layers = []
        layers_idx = []
        i = 0
        keys = []
        for name, module in model.named_modules():
            if isinstance(module, qlinear_experimental.QLinearExperimentalCPU):
                logging.critical(
                    f"Output activations: int32: Module {name} : min_int32:{module.min_int32}  max_int32:{module.max_int32}  abs_max:{abs_max}"
                )
                keys.append(name)
                local_max = np.max(np.abs([module.min_int32, module.max_int32]))
                if local_max > abs_max:
                    abs_max = local_max
                layers.append(name)
                layers_idx.append(i)
                i += 1
                all_int32_min.append(module.min_int32)
                all_int32_max.append(module.max_int32)

        print(f"\tAbs max. of int32 accumulator seen during calibration: {abs_max}")
        logging.critical(
            f"Abs max. of int32 accumulator seen during calibration: {abs_max}"
        )

        line_source = ColumnDataSource(
            data=dict(
                x=layers_idx,
                y1=all_int32_min,
                y2=all_int32_max,
            )
        )
        graph = figure(
            x_range=list(keys),
            height=1200,
            width=2400,
            title="%s int32 accumulator ranges" % args.model_name,
        )
        output_file(
            log_dir + "/calibration_%s_int32_accumulator.html" % args.model_name
        )
        graph.vline_stack(["y1", "y2"], x="x", source=line_source)
        graph.xaxis.axis_label = "layers"
        graph.yaxis.axis_label = "int32 accumulator ranges"
        graph.xaxis.major_label_orientation = "vertical"
        save(graph)
        print(
            f"Step 9 : Saved int32 accumulator data range plot in calibration_{args.model_name}_int32_accumulator.html"
        )

        print(
            f"\trequantize_out_scale: {abs_max/32768} - round to closest 2**r and use r as requantize_out_scale"
        )
        logging.critical(
            f"\trequantize_out_scale: {abs_max/32768} - round to closest 2**r and use r as requantize_out_scale"
        )
        print(
            f"Step 10 : Calculated requantize_out_scale for int32->int16 shift in AIE"
        )
