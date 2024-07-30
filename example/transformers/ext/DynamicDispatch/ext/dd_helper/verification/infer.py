import argparse
import os
import onnx
from onnx import numpy_helper
from onnx import helper
import numpy as np
import time

import onnxruntime as ort
import re


def inference(args):
    # Read args
    original_model_path = args.model
    input_data_dir = args.in_data_dir
    output_data_dir = args.out_data_dir
    ep = args.ep
    config_file_path = args.config
    xclbin = args.xclbin
    iterations = int(args.iters)
    save_as_npz = args.npz

    # Create session options
    sess_options = ort.SessionOptions()
    ort.set_default_logger_severity(3)
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    print("\n- Model: {}".format(original_model_path))

    if ep == "vai":
        print("- VAI-EP " + "-" * 80 + "\n")

        if config_file_path is None:
            raise ValueError("Option --ep vai requies --config-file to be set.")
        elif not os.path.exists(config_file_path):
            raise FileNotFoundError(
                "Unable to locate config file: {}".format(config_file_path)
            )

        if xclbin is None:
            raise ValueError("Option --ep vai requies --xclbin to be set.")
        elif not os.path.exists(xclbin):
            raise FileNotFoundError("Unable to locate xclbin file: {}".format(xclbin))

        # Create ORT Session
        ort_sess = ort.InferenceSession(
            original_model_path,
            providers=["VitisAIExecutionProvider"],
            provider_options=[
                {
                    "config_file": config_file_path,
                    "target": "RyzenAI_transformer_config_2",
                    "xclbin": xclbin,
                }
            ],
            sess_options=sess_options,
        )
    else:
        print("- CPU-EP " + "-" * 80 + "\n")
        ort_sess = ort.InferenceSession(original_model_path, sess_options=sess_options)

    # Print input's / output's name / type / shape
    for inp in ort_sess.get_inputs():
        print(
            "- Input name: {}\n  type: {}, shape: {}\n".format(
                inp.name, inp.type, inp.shape
            )
        )
    for output in ort_sess.get_outputs():
        print(
            "- Output name: {}\n  type: {}, shape: {}\n".format(
                output.name, output.type, output.shape
            )
        )

    # Create output dir
    if output_data_dir is not None:
        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)

    # Get input files
    input_files = os.listdir(input_data_dir)
    # Generate input keys
    keys = list(set([re.search(r"\d+", k).group() for k in input_files]))
    keys.sort()

    if len(ort_sess.get_inputs()) == 1:
        embedding_files = [str(k) + ".raw" for k in keys]
        attn_mask_files = [str(k) + ".raw" for k in keys]
    else:
        # Input files must be stored in one directory with this naming
        embedding_files = ["embeddings_" + str(k) + ".raw" for k in keys]
        attn_mask_files = ["attention_mask_" + str(k) + ".raw" for k in keys]

    # Base names for output files
    output_files = ["output_" + str(k) for k in keys]

    # Run inference for all input samples
    for k, ef, af, of in zip(keys, embedding_files, attn_mask_files, output_files):
        # Input file paths
        emb_fp = os.path.join(input_data_dir, ef)
        atm_fp = os.path.join(input_data_dir, af)
        # Output file path
        if output_data_dir is not None:
            ofp = os.path.join(output_data_dir, of)
            print(
                "- Input Files: {} -> Emb = {}, Mask = {}\n  Output Files: {}".format(
                    k, ef, af, of
                )
            )
        else:
            print("- Input Files: {} -> Emb = {}, Mask = {}".format(k, ef, af))

        # Prepare inputs for inference
        model_inputs = {}
        for inp in ort_sess.get_inputs():
            if "embed" in inp.name:
                model_inputs[inp.name] = np.fromfile(
                    emb_fp, dtype=np.float32, sep=""
                ).reshape(inp.shape)
            elif "attention" in inp.name or "mask" in inp.name:
                model_inputs[inp.name] = np.fromfile(
                    atm_fp, dtype=np.float32, sep=""
                ).reshape(inp.shape)
            elif "image" in inp.name:
                model_inputs[inp.name] = np.fromfile(
                    atm_fp, dtype=np.float32, sep=""
                ).reshape(inp.shape)
            else:
                raise ValueError(
                    "Input file names must have names with 'embeddings' and 'attention_mask'"
                )

        print("- Running Session ...")
        from colorama import Fore

        output_hash = {}

        start = time.perf_counter_ns()
        for i in range(iterations):
            outputs = ort_sess.run(None, {**model_inputs})
            for meta, tensor in zip(ort_sess.get_outputs(), outputs):
                outsum = tensor.flatten().sum()
                if output_hash.get(meta.name, None) is None:
                    output_hash[meta.name] = outsum
                elif outsum != output_hash.get(meta.name):
                    print(
                        Fore.RED
                        + "- Error: Inconsistent output across runs! {} vs {}\n".format(
                            outsum, output_hash.get(meta.name)
                        )
                        + Fore.RESET
                    )
                else:
                    print(Fore.GREEN + "- Consistent output across runs!" + Fore.RESET)
        # Execution time (ms)
        end = time.perf_counter_ns()
        print(
            "- Average execution time (ms): {}\n".format(
                (end - start) / (iterations * 1000000)
            )
        )
        # Save outputs in files
        if output_data_dir is not None:
            if save_as_npz:
                # Save as .npz file
                kwargs = {}
                for meta, tensor in zip(ort_sess.get_outputs(), outputs):
                    kwargs[meta.name] = tensor
                np.savez(ofp, **kwargs)
            else:
                # Save as .raw file
                for i in range(len(ort_sess.get_outputs())):
                    fn = ofp + "_{}.raw".format(i)
                    with open(fn, "wb") as fp:
                        fp.write(outputs[i].data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Original ONNX model", type=str, required=True)
    parser.add_argument("--in-data-dir", help="Input data dir", required=True)
    parser.add_argument(
        "--out-data-dir", help="Output data dir", required=False, default=None
    )
    parser.add_argument(
        "--ep",
        help="Output data dir",
        choices=["cpu", "vai"],
        required=False,
        default="cpu",
    )
    parser.add_argument(
        "--config", help="Config file path", required=False, default=None
    )
    parser.add_argument(
        "--xclbin", help="XCLBIN file path", required=False, default=None
    )
    parser.add_argument(
        "--iters", help="Number of iterations", required=False, default=1
    )
    parser.add_argument(
        "--npz",
        default=False,
        required=False,
        action="store_true",
        help="Store output files as .npz",
    )
    # Parse args
    args = parser.parse_args()
    # Run inference
    inference(args)
