import argparse
import psutil
import os
import onnx
from onnx import numpy_helper
from onnx import helper
import numpy as np
import time

import onnxruntime as ort
import re
from colorama import Fore, Style

# Print Status
from tabulate import tabulate as Tabulate

# Report writing to CSV
import csv

# L2-Norm threshold
L2_NORM_THESHOLD = 0.05
MAX_ERR_THRESHOLD = 10


def relative_l2_norm_error(cpu_out, aie_out):
    if cpu_out.shape != aie_out.shape:
        print("CPU / VAI Shape: {} / {}".format(cpu_out.shape, aie_out.shape))
        raise ValueError("Shape mismatch at output")
    # L2-norm
    l2_norm_diff = np.linalg.norm((cpu_out - aie_out)) / cpu_out.size
    # Max Error (alsolute max)
    max_err = np.amax(np.abs(cpu_out - aie_out))
    return l2_norm_diff, max_err


def create_ort_session(model_path, ep, config=None, xclbin=None):
    # Create ORT Session Options
    sess_options = ort.SessionOptions()

    # Set log severity
    ort.set_default_logger_severity(3)
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    if ep == "cpu":
        print("- Create CPU-EP Session ...")
        session = ort.InferenceSession(model_path, sess_options=sess_options)
    elif ep == "vai":
        print("- Create VAI-EP Session ...")
        if config is None:
            raise ValueError("Option --ep vai requies --config-file to be set.")
        elif not os.path.exists(config):
            raise FileNotFoundError("Unable to locate config file: {}".format(config))

        if xclbin is None:
            raise ValueError("Option --ep vai requies --xclbin to be set.")
        elif not os.path.exists(xclbin):
            raise FileNotFoundError("Unable to locate xclbin file: {}".format(xclbin))

        # Create ORT Session
        session = ort.InferenceSession(
            model_path,
            providers=["VitisAIExecutionProvider"],
            provider_options=[
                {
                    "config_file": config,
                    "target": "RyzenAI_transformer_config_2",
                    "xclbin": xclbin,
                }
            ],
            sess_options=sess_options,
        )
    else:
        raise ValueError("EP: {} is invalid.".format(ep))

    # Return session
    return session


def display_input_tensor_meta(session):
    # Input/Output name/type/shape
    for tm in session.get_inputs():
        print(
            "- Input name: {}\n  type: {}, shape: {}\n".format(
                tm.name, tm.type, tm.shape
            )
        )
    for tm in session.get_outputs():
        print(
            "- Output name: {}\n  type: {}, shape: {}\n".format(
                tm.name, tm.type, tm.shape
            )
        )


def get_data(tensor_meta, data):
    # Get input data
    model_inputs = {}
    for tm in tensor_meta:
        model_inputs[tm.name] = data.get(tm.name, None)
        if model_inputs[tm.name] is None:
            raise ValueError("- Unable to locate data for tensor {}".format(tm.name))
    return model_inputs


def get_files(userpath, ff=".onnx"):
    # Get models from directory
    files = None
    if os.path.isdir(userpath):
        files = os.listdir(userpath)
        files = [os.path.join(userpath, file) for file in files if file.endswith(ff)]
    elif os.path.isfile(userpath) and userpath.endswith(ff):
        files = [userpath]
    else:
        raise ValueError("Invalid input model/dir path.")
    return files


def test_subgraph(model_path, args):
    # Args
    data_dir = args.data
    dll = args.dll
    verbose = args.verbose
    ep = args.ep
    config = args.config
    xclbin = args.xclbin
    repeat = args.repeat
    reports_dir = args.reports_dir
    output_dir = args.out_dir

    # ORT inference session
    session = None
    try:
        if ep == "cpu" or ep == "CPU":
            session = create_ort_session(model_path, "cpu")
        elif ep == "vai" or ep == "VAI":
            session = create_ort_session(model_path, "vai", config, xclbin)
        else:
            raise ValueError(
                "Invalid Execution Provider. Available options [CPU, VAI]."
            )
        print("-" * 80)
    except Exception as e:
        print("\n- Error in creating inference session!")
        print(
            "-" * 80
            + Fore.RED
            + "\n{}\n".format(e)
            + Fore.RESET
            + "-" * 80
            + "\n- Exiting ...\n".format(e)
        )
        exit(1)

    # Get input/output data
    data_files = get_files(data_dir, ".npz")

    # Test Status
    status = None
    sum_l2n = 0

    # Tabulate the status
    floatformat = (None, ".12f", None, ".12f")

    test_results = [["Data File", "L2-Norm", "Status", "Max-err"]]
    test_failures = [["Data File", "L2-Norm", "Status", "Max-err"]]

    # Path and file names
    inpath, infile = os.path.split(model_path)
    # Create reports directory
    os.makedirs(reports_dir, exist_ok=True)
    report_file = os.path.join(reports_dir, infile + ".md")
    # Open report file for writing
    rfp = open(report_file, "w")

    # Output data directory
    npz_fp_base = None
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        npz_fp_base = os.path.join(output_dir, infile)

    # Compare with absolute max
    maximum_err = 0
    # run subgraph for each input file
    for df in data_files:
        # Data file path
        df_path = df
        _, df = os.path.split(df)

        # load data file
        data = np.load(df_path)

        tensors = []
        if verbose >= 2:
            display_input_tensor_meta(session)

        # Get input data
        model_inputs = get_data(session.get_inputs(), data)

        # Get golden output data
        golden_outputs = get_data(session.get_outputs(), data)

        # Run inference
        output_hash = {}
        for i in range(repeat):
            model_outputs = session.run(None, {**model_inputs})
            # Save as npz file
            if output_dir is not None:
                npz_fp = npz_fp_base + "_with_" + df
                kwargs = {}
                for meta, tensor in zip(session.get_outputs(), model_outputs):
                    kwargs[meta.name] = tensor
                np.savez(npz_fp, **kwargs)
            # Check Consistency across runs
            for meta, tensor in zip(session.get_outputs(), model_outputs):
                outsum = tensor.flatten().sum()
                if output_hash.get(meta.name, None) is None:
                    output_hash[meta.name] = outsum
                elif outsum != output_hash.get(meta.name):
                    print(
                        Fore.RED
                        + "- Error: Inconsistent output across runs!\n"
                        + "- Error: EP: {}, data: {}".format(ep.upper(), df)
                        + Fore.RESET
                    )
                    return [], []

        # L2 Norm for each model output
        for meta, tensor in zip(session.get_outputs(), model_outputs):
            l2norm = 0
            # Get output tensors for different sessions
            tensor_gold = golden_outputs[meta.name].astype(np.float32)
            tensor_real = tensor.astype(np.float32)
            # L2 Norm
            l2norm, maxerr = relative_l2_norm_error(tensor_gold, tensor_real)
            sum_l2n += l2norm
            # Compare against threshold
            if l2norm < L2_NORM_THESHOLD and maxerr < MAX_ERR_THRESHOLD:
                status = "Pass"
            else:
                cstatus = Fore.RED + "Fail" + Fore.RESET
                status = "Fail"
                # Update failures
                test_failures.append([df_path, l2norm, status, maxerr])
            # Update results
            test_results.append([df_path, l2norm, status, maxerr])

        # Update max err across runs
        maximum_err = max(maximum_err, maxerr)

    # Compare & report results
    result_head = "# Model Results {}-EP vs Golden ...\n\n".format(ep.upper())
    rfp.write(result_head)
    result_head = "- Model Path: {}\n".format(model_path)
    rfp.write(result_head)
    # Display header
    header_string = "\n## Results {}-EP vs Golden ... ".format(ep.upper())
    # Results
    results = (
        header_string
        + "\n\n"
        + Tabulate(
            test_results, headers="firstrow", tablefmt="github", floatfmt=floatformat
        )
    )
    rfp.write(results)
    # Close report
    rfp.close()

    # Test fails
    if len(test_failures) > 1:
        if verbose:
            failures = (
                header_string
                + "(Max Err > {})\n\n".format(MAX_ERR_THRESHOLD)
                + Tabulate(
                    test_failures,
                    headers="firstrow",
                    tablefmt="github",
                    floatfmt=floatformat,
                )
            )
            print(failures)
        COLOR_TEXT = Fore.RED
    else:
        COLOR_TEXT = Fore.GREEN
    # Summary
    print(
        "\n## Average of {} Inputs Samples -> ".format(len(data_files))
        + COLOR_TEXT
        + "Average L2-Norm: {:.10f}, ".format(sum_l2n / len(data_files))
        + "Max Error Across all samples: {:.6f}".format(maximum_err)
        + Fore.RESET
    )
    # Report path
    print("\n- Test report saved at: {}".format(report_file))
    # return test failures
    return test_failures[1:], test_results[1:]


def test_subgraphs(args):
    # Args
    model_dir = args.model
    data_dir = args.data
    fltr = args.filter
    reports_dir = args.reports_dir

    if not os.path.exists(model_dir):
        raise ValueError("- Unable to locate model directory: {}".format(model_dir))
    if not os.path.exists(data_dir):
        raise ValueError("- Unable to locate data directory: {}".format(data_dir))

    if fltr:
        print("- Filtering models with name: {}".format(fltr))

    # Create reports directory
    os.makedirs(reports_dir, exist_ok=True)
    report_file = os.path.join(reports_dir, "summary.csv")
    failure_file = os.path.join(reports_dir, "failures.csv")

    # Open report file for writing
    rfp = open(report_file, "w", newline="")
    ffp = open(failure_file, "w", newline="")

    # Test Results
    test_results = [["Model", "Data File", "L2-Norm", "Status", "Max-err"]]
    test_failures = [["Model", "Data File", "L2-Norm", "Status", "Max-err"]]

    # Get models from directory
    models = get_files(model_dir, ".onnx")
    # Run inference for each model
    for model_path in models:
        if fltr is not None:
            if fltr.lower() not in model_path.lower():
                continue
        _, model = os.path.split(model_path)
        print("\n- " + Fore.CYAN + "Model Path: " + Fore.RESET + model_path)
        print("-" * 80)

        # Run subgraph
        failures, all_results = test_subgraph(model_path, args)
        print("-" * 80)

        # Tabulate all results
        for row in all_results:
            entry = [model]
            entry.extend(row)
            test_results.append(entry)

        # Tabulate failures
        if len(failures):
            for failed in failures:
                entry = [model]
                entry.extend(failed)
                test_failures.append(entry)

    # Print failure summary
    print("")
    if len(test_failures) > 1:
        print("- " + Fore.RED + "Failure Summary ...\n" + Fore.RESET)
        print(Tabulate(test_failures, headers="firstrow", tablefmt="github") + "\n")
        # Save CSV
        csv_writer = csv.writer(ffp, delimiter=",")
        csv_writer.writerows(test_failures)
        ffp.close()
        print(
            "- Failure Summary saved at: "
            + Fore.CYAN
            + "{}".format(failure_file)
            + Fore.RESET
        )

    # Write all results
    csv_writer = csv.writer(rfp, delimiter=",")
    csv_writer.writerows(test_results)
    rfp.close()
    print(
        "- All Results saved at: " + Fore.CYAN + "{}\n".format(report_file) + Fore.RESET
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Original ONNX model file or directory containing models",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        help="Directory for the outputs to be saved",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument("--data", help="Input data file or directory", required=True)
    parser.add_argument(
        "--dll", help="Custom OP DLL path", required=False, default=None
    )
    parser.add_argument(
        "--filter", help="Filter by file name charactors", required=False, default=None
    )
    parser.add_argument(
        "--ep",
        help="Execution provider",
        choices=["cpu", "CPU", "vai", "VAI"],
        required=False,
        default="cpu",
    )
    parser.add_argument("--config", help="Config JSON file path", required=False)
    parser.add_argument("--xclbin", help="XCLBIN file path", required=False)
    parser.add_argument(
        "--repeat",
        help="Run each session repeat count times",
        required=False,
        default=1,
        type=int,
    )
    parser.add_argument(
        "--reports-dir",
        help="Directory path where the reports should be saved",
        required=False,
        default="reports",
        type=str,
    )
    parser.add_argument(
        "--verbose",
        help="Enable more prints",
        required=False,
        default=False,
        action="store_true",
    )
    # Parse args
    args = parser.parse_args()
    # Run inference
    test_subgraphs(args)
