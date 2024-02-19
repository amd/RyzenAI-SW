# release 16 public

import pandas as pd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import argparse
import subprocess
import time
import importlib_metadata
import sys
import json
import platform
import psutil
import re
import csv
from pathlib import Path

from onnxruntime.quantization.calibrate import CalibrationDataReader
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod, quantize_static
import vai_q_onnx
import random
from PIL import Image

class Colors:
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    DIMMERED_WHITE = "\033[90m"

def show_help():
    print("Usage: python performance_benchmark.py <parameters list>")
    print("Please fill the parameters list. Use -h for help")
    print("i.e.:")
    print("python performance_benchmark.py --model resnet50_1_5_224x224-op13_NHWC_quantized.onnx --device VitisAIEP --num 2000 -i 4 -t 4 -p 1 -j demo.json -r 1")

def parse_args():
    if len(sys.argv) < 2:
        show_help()
        quit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", "-b", type=int, default=1, help="batch size: number of images processed at the same time by the model. VitisAIEP supports batchsize = 1. Default is 1")
    parser.add_argument(
        "--calib",
        type=str,
        default=".\\Imagenet\\val",
        help=f"path to Imagenet database, used for quantization with calibration. Default= .\Imagenet\val ",
    )
    parser.add_argument(
        "--config", "-c", type=str, default="vaip_config.json", help="path to config json file. Default=vaip_config.json"
    )
    parser.add_argument(
        "--core",
        default="1x4",
        type=str,
        help="Which core to use. Possible values are 1x4 and 4x4. Default=1x4",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="CPU",
        choices=["CPU", "iGPU", "dGPU", "ZenDNN", "VitisAIEP"],
        help="Execution Provider selection. Default=CPU",
    )
    parser.add_argument(
        "--infinite",
        type=int,
        default=1,
        help="if 1: Executing an infinite loop, when combined with a time limit, enables the test to run for a specified duration. Default=1",
    )
    parser.add_argument(
        "--instance-count",
        "-i",
        type=int,
        default=1,
        help="This parameter governs the parallelism of job execution. When the Vitis AI EP is selected, this parameter controls the number of DPU runners. The workload is always equally divided per each instance count. Default=1",
    )
    parser.add_argument("--intra-op-num-threads", type=int, default=2, help="Number of CPU threads enabled when an operator is resolved by the CPU. Affects the performances but also the CPU power consumption. Default=2")
    
    parser.add_argument(
        "--json",
        type=str,
        help="Path to the file of parameters.",
    )
    parser.add_argument(
        "--log_csv",
        "-k",
        type=int,
        default=0,
        help="If this option is set to 1, measurement data will appended to a CSV file. Default=0",
    )
    parser.add_argument(
        "--log_json",
        "-j",
        type=str,
        default="report_performance.json",
        help="JSON file name where the measureemnts will be saved. Default = report_performance.json ",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=0,
        help="Minimum time interval (s) for running inferences. Default=0",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="",
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--no-inference",
        type=int,
        default=0,
        help="When set to 1 the benchmark runs without inference for power measurements baseline. Default=0",
    )
    parser.add_argument(
        "--num", "-n", type=int, default=100, help="The number of images loaded into memory and subsequently sent to the model. Default=100"
    )
    parser.add_argument(
        "--renew",
        "-r",
        type=int,
        default=1,
        help="if set to 1 cancel the cache and recompile the model. Set to 0 to keep the old compiled file. Default=1",
    )
    parser.add_argument(
        "--timelimit",
        "-l",
        type=int,
        default=10,
        help="When used in conjunction with the --infinite option, it represents the maximum duration of the experiment. The default value is set to 10 seconds.",
    )
    parser.add_argument(
        "--threads", "-t", type=int, default=1, help="CPU threads. Default=1"
    )
    parser.add_argument(
        "--verbose", "-v", type=int, default=0, help="0 (default): no debug messages, 1: few debug messages, 2: all debug messages"
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=40,
        help="Perform warmup runs, default = 40",
    )
    args, _ = parser.parse_known_args()
    if args.json:
        try:
            with open(args.json, "r") as json_file:
                config = json.load(json_file)
            # Update argparse arguments with values from the JSON file
            for arg_name, value in config.items():
                setattr(args, arg_name, value)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
    
    return args   

def initcsv(filename, R, C):
    data = np.full((R, C), np.nan)
    if os.path.isfile(filename):
        os.remove(filename)
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, header=False)

def plot2D(tablename, ep):
    # Read the CSV file
    data = pd.read_csv(tablename, header=None)

    # find best configuration for throughput
    argmax = np.unravel_index(np.argmax(data, axis=None), data.shape)
    ggprint(f"best processors number {argmax[0] + 1}")
    ggprint(f"best batchsize number {argmax[1] + 1}")

    # Y (instances)
    rows_positions = np.arange(data.shape[0])
    # X (batchsize)
    cols_positions = np.arange(data.shape[1])

    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(data.shape[0]), labels=rows_positions + 1)
    ax.set_xticks(np.arange(data.shape[1]), labels=cols_positions + 1)

    # Loop over data dimensions and create text annotations.
    for y in rows_positions:
        for x in cols_positions:
            text = ax.text(x, y, data[x][y], ha="center", va="center", color="w")

    # ax.set_title(f'{ep} Throughput [fps]')
    ax.set_title(f"{ep}")
    fig.tight_layout()

    plt.xlabel("Batchsize")
    plt.ylabel("Instances")

    # Adjust figure size to fit labels
    fig = plt.gcf()
    fig.set_size_inches(20, 5)

    # Save the plot as a PNG file
    pictname = os.path.splitext(tablename)[0] + ".png"
    plt.savefig(pictname, dpi=300, bbox_inches="tight")

    plt.show()

def plot2D_2(thr_table, lat_table, ep):
    # Read the CSV file
    thr_data = pd.read_csv(thr_table, header=None)
    lat_data = pd.read_csv(lat_table, header=None)

    # find best configuration
    thr_argmax = np.unravel_index(np.argmax(thr_data, axis=None), thr_data.shape)
    ggprint(
        f"Throughput best config: processors: {thr_argmax[0] + 1}, batchsize: {thr_argmax[1] + 1}"
    )
    lat_argmin = np.unravel_index(np.argmin(lat_data, axis=None), lat_data.shape)
    ggprint(
        f"Latency    best config: processors: {lat_argmin[0] + 1}, batchsize: {lat_argmin[1] + 1}"
    )

    # Y (instances)
    rows_positions = np.arange(thr_data.shape[0])
    # X (batchsize)
    cols_positions = np.arange(thr_data.shape[1])

    fig, axs = plt.subplots(1, 2)

    imthr = axs[0].imshow(thr_data, cmap="Blues")
    imlat = axs[1].imshow(lat_data, cmap="Oranges")

    # Show all ticks and label them with the respective list entries
    axs[0].set_yticks(np.arange(thr_data.shape[0]), labels=rows_positions + 1)
    axs[0].set_xticks(np.arange(thr_data.shape[1]), labels=cols_positions + 1)
    axs[1].set_yticks(np.arange(lat_data.shape[0]), labels=rows_positions + 1)
    axs[1].set_xticks(np.arange(lat_data.shape[1]), labels=cols_positions + 1)

    # Loop over data dimensions and create text annotations.
    for y in rows_positions:
        for x in cols_positions:
            thr_text = axs[0].text(
                x, y, thr_data[x][y], ha="center", va="center", color="black"
            )
            lat_text = axs[1].text(
                x, y, lat_data[x][y], ha="center", va="center", color="black"
            )

    axs[0].set_title(f"{ep} Throughput [fps]")
    axs[1].set_title(f"{ep} Latency [ms]")
    fig.tight_layout()

    axs[0].set_xlabel("Batchsize")
    axs[0].set_ylabel("Instances")
    axs[1].set_xlabel("Batchsize")
    axs[1].set_ylabel("Instances")

    # Adjust figure size to fit labels
    fig = plt.gcf()
    fig.set_size_inches(20, 5)

    # Save the plot as a PNG file
    # pictname = os.path.splitext(tablename)[0] + '.png'
    pictname = "thr_lat.png"
    plt.savefig(pictname, dpi=300, bbox_inches="tight")

    plt.show()

def check_package_version(package_name):
    try:
        version = importlib_metadata.version(package_name)
        return version
    except importlib_metadata.PackageNotFoundError:
        return f"{package_name} not found"

def check_env(release):
    data = [
        ("PACKAGE", "STATUS"),
    ]
    ggprint(
        "------------------------------------------------------------------------------"
    )
    ggprint("Preliminary environment check")
    if sys.version_info.major == 3 and sys.version_info.minor == 9:
        data.append(
            (
                f"Python version {sys.version_info.major}.{sys.version_info.minor}",
                Colors.GREEN + "OK" + Colors.RESET,
            )
        )
    else:
        data.append(
            (
                f"Python version {sys.version_info.major}.{sys.version_info.minor}",
                Colors.RED + "OK" + Colors.RESET,
            )
        )

    package_name = "onnxruntime"
    version = check_package_version(package_name)
    # print(f"{package_name} version: {version}")
    if version in ["1.15.1", "1.16.1", "1.16.2", "1.16.3", "1.17.0"]:
        data.append(
            (f"{package_name} version: {version}", Colors.GREEN + "OK" + Colors.RESET)
        )
    else:
        data.append(
            (f"{package_name} version: {version}", Colors.RED + "WRONG" + Colors.RESET)
        )

    package_name = "onnxruntime-vitisai"
    version = check_package_version(package_name)
    # print(f"{package_name} version: {version}")
    if version == "1.15.1":
        data.append(
            (f"{package_name} version: {version}", Colors.GREEN + "OK" + Colors.RESET)
        )
    else:
        data.append(
            (
                f"{package_name} version: {version}" + version,
                Colors.RED + "WRONG" + Colors.RESET,
            )
        )

    package_name = "voe"
    version = check_package_version(package_name)
    # print(f"{package_name} version: {version}")
    if version == "0.1.0":
        data.append(
            (f"{package_name} version: {version}", Colors.GREEN + "OK" + Colors.RESET)
        )
    else:
        data.append(
            (f"{package_name} version: {version}", Colors.RED + "WRONG" + Colors.RESET)
        )

    max_width = max(len(row[0]) for row in data)

    for row in data:
        column1, column2 = row
        # Left-align the text in the first column and pad with spaces
        formatted_column1 = column1.ljust(max_width)
        ggprint(f"{formatted_column1} {column2}")

    silicon = checksilicon()
    ggprint(f"{silicon} silicon present")      
    ggprint(f"Benchmark release: {release}")
    ggprint("------------------------------------------------------------------------------")

def check_args(args):
    assert args.num >= (
        args.batchsize * args.instance_count
    ), "runs must be greater than batches*instance-count"

    total_cpu = os.cpu_count()
    if args.instance_count > total_cpu:
        args.instance_count = total_cpu
        ggprint(f"Limiting instance count to max cpu count ({total_cpu})")


def cancelcache(cache_path):
    #cache_path = r"modelcachekey"
    if os.path.exists(cache_path) and os.path.isdir(cache_path):
        try:
            shutil.rmtree(cache_path)
            ggprint(f"Deleted {cache_path}")
        except Exception as e:
            ggprint(f"Error during cache cancellation: {e}")
    else:
        ggprint(f"{cache_path} does not exist or is not a directory")

def set_ZEN_env():
    os.environ["ZENDNN_LOG_OPTS"] = "ALL:0"
    os.environ["OMP_NUM_THREADS"] = "64"
    os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["ZENDNN_INFERENCE_ONLY"] = "1"
    os.environ["ZENDNN_INT8_SUPPORT"] = "0"
    os.environ["ZENDNN_RELU_UPPERBOUND"] = "0"
    os.environ["ZENDNN_GEMM_ALGO"] = "3"
    os.environ["ZENDNN_ONNXRT_VERSION"] = "1.12.1"
    os.environ["ZENDNN_ONNX_VERSION"] = "1.12.0"
    os.environ["ZENDNN_PRIMITIVE_CACHE_CAPACITY"] = "1024"
    os.environ["ZENDNN_PRIMITIVE_LOG_ENABLE"] = "0"
    os.environ["ZENDNN_ENABLE_LIBM"] = "0"
    os.environ["ZENDNN_CONV_ALGO"] = "3"
    os.environ["ZENDNN_CONV_ADD_FUSION_ENABLE"] = "1"
    os.environ["ZENDNN_RESNET_STRIDES_OPT1_ENABLE"] = "1"
    os.environ["ZENDNN_CONV_CLIP_FUSION_ENABLE"] = "1"
    os.environ["ZENDNN_BN_RELU_FUSION_ENABLE"] = "1"
    os.environ["ZENDNN_CONV_ELU_FUSION_ENABLE"] = "1"
    os.environ["ZENDNN_CONV_RELU_FUSION_ENABLE"] = "1"
    os.environ["ORT_ZENDNN_ENABLE_INPLACE_CONCAT"] = "1"
    os.environ["ZENDNN_ENABLE_MATMUL_BINARY_ELTWISE"] = "1"
    os.environ["ZENDNN_ENABLE_GELU"] = "1"
    os.environ["ZENDNN_ENABLE_FAST_GELU"] = "1"
    os.environ["ZENDNN_REMOVE_MATMUL_INTEGER"] = "1"
    os.environ["ZENDNN_MATMUL_ADD_FUSION_ENABLE"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def ggprint(linea):
    print(Colors.DIMMERED_WHITE + linea + Colors.RESET)

def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(":"))
    return h * 3600 + m * 60 + s

def str_to_sec(date_string):
    date_string = date_string.strip("--[ ]").strip()
    #
    # date_format = "%d.%m.%Y %H:%M:%S.%f"
    time_str = date_string.split(" ")[1]
    seconds = time_to_seconds(time_str)
    return seconds

def meas_init(args, release, total_throughput, average_latency, xclbin_path):
    # dictionary of results
    measurement = {}
    measurement["run"] = {}
    if  args.json:
        temp=""
        with open(args.json, "r") as json_file:
            config = json.load(json_file)
            for key, value in config.items():
                temp = temp + f" --{key} {value}"
        measurement["run"]["command"] = " ".join(sys.argv) + temp
    else:
        measurement["run"]["command"] = " ".join(sys.argv)
    
    measurement["run"]["benchmark_release"] = release
    measurement["run"]["model"] = args.model
    measurement["run"]["device"] = args.device

    measurement["results"] = {}
    measurement["results"]["performance"] = {}
    measurement["results"]["performance"]["total_throughput"] = total_throughput
    measurement["results"]["performance"]["average_latency"] = average_latency
    measurement["results"]["efficency perf/W"] = {}
    measurement["results"]["efficency perf/W"]["apu_perf_pow"] = "N/A"
    measurement["results"]["energy mJ/frame"] = {}
    measurement["results"]["energy mJ/frame"]["apu"] = "N/A"
    measurement["results"]["energy mJ/frame"]["cpu"] = "N/A"
    measurement["results"]["energy mJ/frame"]["ipu"] = "N/A"
    measurement["results"]["energy mJ/frame"]["mem"] = "N/A"

    measurement["vitisai"] = {}
    measurement["vitisai"]["all"] = 0
    measurement["vitisai"]["CPU"] = 0
    measurement["vitisai"]["DPU"] = 0

    measurement["system"] = {}
    measurement["system"]["frequency"] = {}
    measurement["system"]["frequency"]["MPIPUCLK"] = 0
    measurement["system"]["frequency"]["IPUHCLK"] = 0
    measurement["system"]["frequency"]["FCLK"] = 0
    measurement["system"]["frequency"]["LCLK"] = 0
    measurement["system"]["voltages"] = {}
    measurement["system"]["voltages"]["CORE0"] = 0
    measurement["system"]["voltages"]["CORE1"] = 0
    measurement["system"]["voltages"]["CORE2"] = 0
    measurement["system"]["voltages"]["CORE3"] = 0
    measurement["system"]["voltages"]["CORE4"] = 0
    measurement["system"]["voltages"]["CORE5"] = 0
    measurement["system"]["voltages"]["CORE6"] = 0
    measurement["system"]["voltages"]["CORE7"] = 0
    measurement["system"]["hw"] = {}
    measurement["system"]["hw"]["processor"] = platform.processor()
    measurement["system"]["hw"]["num_cores"] = os.cpu_count()
    measurement["system"]["os"] = {}
    measurement["system"]["os"]["os_version"] = platform.platform()

    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    swap_info = psutil.swap_memory()
    measurement["system"]["resources"] = {
        "CPU_usage": f"{cpu_usage}%",
        "Memory": f"{memory_info.available / (1024 ** 3)} GB",
        "Swap_Memory": f"{swap_info.free / (1024 ** 3)} GB",
    }

    powershell_path = os.path.join(os.environ['SystemRoot'], 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')
    if os.path.exists(powershell_path):
        try:
            powershell_command = 'Get-WmiObject Win32_PnPSignedDriver | Where-Object {$_.DeviceName -like "*AMD IPU*"} | Select-Object DeviceName, DriverVersion'
            result = subprocess.check_output([powershell_path, "-Command", powershell_command], text=True)
            lines = result.strip().split("\n")
            driver_version = lines[-1].split()[-1]
        except subprocess.CalledProcessError:
            ggprint("Error: AMD IPU driver not found or PowerShell command failed.")       
    else:
        ggprint("Warning: Could not execute a Powershell command. Please manually verify that the AMD IPU Driver exists")       
        driver_version = ""

    measurement["system"]["driver"] = {}
    measurement["system"]["driver"]["ipu"] = driver_version

    measurement["environment"] = {}
    measurement["environment"]["packages"] = {}

    # info stored in the cache (only with VitisAIEP)
    if args.device == "VitisAIEP":
        measurement["environment"]["xclbin"] = {}
        measurement["environment"]["xclbin"]["xclbin_path"] = xclbin_path
        cache_dir = os.path.join(Path(__file__).parent.resolve(), "cache", os.path.basename(args.model))
        with open(os.path.join(cache_dir, r"modelcachekey\config.json"), "r") as json_file:
            data = json.load(json_file)
        releases = data["version"]["versionInfos"]

        measurement["environment"]["xclbin"]["packages"] = {
            release["packageName"]: {
                "commit": release["commit"],
                "version": release["version"],
            }
            for release in releases
        }

    try:
        output = subprocess.check_output("conda list", shell=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    package_info = {}
    # Parse the output to extract package names and versions
    lines = output.strip().split("\n")
    for line in lines[3:]:  # Skip the first 3 lines which contain header information
        parts = re.split(r"\s+", line.strip())
        if len(parts) >= 2:
            package_name = parts[0]
            package_version = parts[1]
            package_info[package_name] = package_version
    measurement["environment"]["packages"] = package_info

    if args.device == "VitisAIEP":
        measurement["environment"]["vaip_config"] = {}
        with open(args.config, "r") as json_file:
            vaip_conf = json.load(json_file)

        measurement["environment"]["vaip_config"] = vaip_conf

    return measurement

def appendcsv(measurement, csv_file="measurements.csv"):
    fieldnames = [
        "timestamp",
        "command",
        "benchmark_release",
        "model",
        "device",
        "total_throughput",
        "average_latency",
        "apu_perf_pow",
        "energy_apu",
        "energy_cpu",
        "energy_ipu",
        "energy_mem",
        "MPIPUCLK",
        "IPUHCLK",
        "FCLK",
        "LCLK",
        "V_CORE0",
        "V_CORE1",
        "V_CORE2",
        "V_CORE3",
        "V_CORE4",
        "V_CORE5",
        "V_CORE6",
        "V_CORE7",
        "processor",
        "num_cores",
        "os_version",
        "CPU_usage",
        "Memory",
        "Swap_Memory",
        "ipu_driver",
        "xclbin_path",
        "vaip",
        "target_factory",
        "xcompiler",
        "onnxruntime",
        "graph_engine",
        "xrt",
    ]
    # Open the CSV file in append mode and write the measurement
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Check if the file is empty and write the header if needed
        if file.tell() == 0:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": time.strftime("%Y%m%d%H%M%S"),
                "command": measurement["run"]["command"],
                "benchmark_release": measurement["run"]["benchmark_release"],
                "model": measurement["run"]["model"],
                "device": measurement["run"]["device"],
                "total_throughput": measurement["results"]["performance"][
                    "total_throughput"
                ],
                "average_latency": measurement["results"]["performance"][
                    "average_latency"
                ],
                "apu_perf_pow": measurement["results"]["efficency perf/W"][
                    "apu_perf_pow"
                ],
                "energy_apu": measurement["results"]["energy mJ/frame"]["apu"],
                "energy_cpu": measurement["results"]["energy mJ/frame"]["cpu"],
                "energy_ipu": measurement["results"]["energy mJ/frame"]["ipu"],
                "energy_mem": measurement["results"]["energy mJ/frame"]["mem"],
                "MPIPUCLK": measurement["system"]["frequency"]["MPIPUCLK"],
                "IPUHCLK": measurement["system"]["frequency"]["IPUHCLK"],
                "FCLK": measurement["system"]["frequency"]["FCLK"],
                "LCLK": measurement["system"]["frequency"]["LCLK"],
                "V_CORE0": measurement["system"]["voltages"]["CORE0"],
                "V_CORE1": measurement["system"]["voltages"]["CORE1"],
                "V_CORE2": measurement["system"]["voltages"]["CORE2"],
                "V_CORE3": measurement["system"]["voltages"]["CORE3"],
                "V_CORE4": measurement["system"]["voltages"]["CORE4"],
                "V_CORE5": measurement["system"]["voltages"]["CORE5"],
                "V_CORE6": measurement["system"]["voltages"]["CORE6"],
                "V_CORE7": measurement["system"]["voltages"]["CORE7"],
                "processor": measurement["system"]["hw"]["processor"],
                "num_cores": measurement["system"]["hw"]["num_cores"],
                "os_version": measurement["system"]["os"]["os_version"],
                "CPU_usage": measurement["system"]["resources"]["CPU_usage"],
                "Memory": measurement["system"]["resources"]["Memory"],
                "Swap_Memory": measurement["system"]["resources"]["Swap_Memory"],
                "ipu_driver": measurement["system"]["driver"]["ipu"],
                "xclbin_path": measurement["environment"]["xclbin"]["xclbin_path"],
                "vaip": measurement["environment"]["xclbin"]["packages"]["vaip"][
                    "version"
                ],
                "target_factory": measurement["environment"]["xclbin"]["packages"][
                    "target_factory"
                ]["version"],
                "xcompiler": measurement["environment"]["xclbin"]["packages"][
                    "xcompiler"
                ]["version"],
                "onnxruntime": measurement["environment"]["xclbin"]["packages"][
                    "onnxrutnime"
                ]["version"],
                "graph_engine": measurement["environment"]["xclbin"]["packages"][
                    "graph_engine"
                ]["version"],
                "xrt": measurement["environment"]["xclbin"]["packages"]["xrt"][
                    "version"
                ],
            }
        )
    ggprint(f"Data appended to {csv_file}")

def save_result_json(results, filename):
    if filename=="use timestamp":
        timestamp = time.strftime("%Y%m%d%H%M%S")
        filename = "results_" + timestamp + ".json"
        
    with open(filename, "w") as file_json:
        json.dump(results, file_json, indent=4)
    ggprint(f"Data saved in {filename}")

def check_silicon(expected, found):
    if expected != found:
        raise ValueError(f"Mismatch between {expected} xclbin and {found} driver")
    else:
        return expected

def PHX_1x4_setup(silicon):
    expected = "RyzenAI-Phoenix"
    try:
        result = check_silicon(expected, silicon)
    except ValueError as error:
        print(f"Error: {error}")
    else:
        xclbin_path = os.path.join(os.getcwd(),"onnxrt\\1x4.xclbin")
        os.environ['XLNX_VART_FIRMWARE'] = str(xclbin_path)       
        os.environ["XLNX_TARGET_NAME"] = "AMD_AIE2_Nx4_Overlay"
        ggprint("PHOENIX 1x4")
        ggprint(f"Path to xclbin_path = {xclbin_path}")
        ggprint(os.environ["XLNX_TARGET_NAME"])

def PHX_4x4_setup(silicon):
    expected = "RyzenAI-Phoenix"
    try:
        result = check_silicon(expected, silicon)
    except ValueError as error:
        print(f"Error: {error}")
    else:
        xclbin_path = os.path.join(os.getcwd(),"onnxrt\\4x4.xclbin")
        os.environ['XLNX_VART_FIRMWARE'] = str(xclbin_path)       
        os.environ['XLNX_TARGET_NAME'] = "AMD_AIE2_4x4_Overlay"
        ggprint("PHOENIX 4x4")
        ggprint(f"Path to xclbin_path = {xclbin_path}")
        ggprint(os.environ["XLNX_TARGET_NAME"])

def STX_1x4_setup(silicon):
    expected = "RyzenAI-Strix"
    try:
        result = check_silicon(expected, silicon)
    except ValueError as error:
        ggprint(f"Error: {error}")
    else:
        xclbin_path = os.path.join(os.getcwd(),"onnxrt\\1x4.xclbin")
        os.environ['XLNX_VART_FIRMWARE'] = str(xclbin_path)       
        os.environ["XLNX_TARGET_NAME"] = "AMD_AIE2P_Nx4_Overlay"
        ggprint("STRIX 1x4")
        ggprint(f"Path to xclbin_path = {xclbin_path}")
        ggprint(os.environ["XLNX_TARGET_NAME"])

def DEF_setup(silicon):
    # default setup
    try:
        result = subprocess.check_output(
            "conda env config vars list", shell=True, text=True
        )
        ggprint("Using default setup")
        #ggprint(result)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

def checksilicon():
    # Run the shell command
    command = "xbutil examine -f json -o out.json --force"
    try:
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True )
    except subprocess.CalledProcessError as e:
        print(f"Error executing the shell command: {e}")    
    
    # Open the JSON file
    with open('out.json', 'r') as file:
        data = json.load(file)
    # Access the value of "system-host-devices-vbnv"
    silicon = data["system"]["host"]["devices"][0]["vbnv"]
    #print("The value of 'system-host-devices-vbnv' is:", silicon)
    return silicon

def set_engine_shape(case):
    switch_dict = {
        "RyzenAI-Phoenix_1x4": PHX_1x4_setup,
        "RyzenAI-Phoenix_4x4": PHX_4x4_setup,
        "RyzenAI-Strix_1x4": STX_1x4_setup,
    }

    silicon = checksilicon()
    action = switch_dict.get(f'{silicon}_{case}', DEF_setup)
    action(silicon)

def SetCalibDir(source_directory, calib_dir):
    if os.path.exists(calib_dir):
        shutil.rmtree(calib_dir)
    os.makedirs(calib_dir)

    subfolders = [f.path for f in os.scandir(source_directory) if f.is_dir()]
    copied_images = 0
    num_images_to_copy = 200
    random.shuffle(subfolders)
    for subfolder in subfolders:
        files = os.listdir(subfolder)
        random.shuffle(files)

        for file in files:
            if copied_images >= num_images_to_copy:
                break

            source_path = os.path.join(subfolder, file)
            destination_path = os.path.join(calib_dir, file)
        
            # check if the image has three channels
            image = Image.open(source_path)
            image_mode = image.mode
            if image_mode == 'RGB':
                shutil.copy(source_path, destination_path)
                copied_images += 1
    return copied_images

def list_operators(onnx_model_path):
    # Load the ONNX model
    oplist = []
    onnx_model = onnx.load(onnx_model_path)

    # Iterate through the nodes in the model's graph
    for node in onnx_model.graph.node:
        # Print the operator type for each node
        #print("Operator:", node.op_type)
        oplist.append(node.op_type)
    return oplist

def get_input_format(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    input_shapes = []
    for input_node in onnx_model.graph.input:
        shape = [dim.dim_value or dim.dim_param for dim in input_node.type.tensor_type.shape.dim]
        input_shapes.append(tuple(shape))
    return [input_node.name, input_shapes]

def analyze_input_format(input_shapes):
    order = "unknown"
    if len(input_shapes)>1:
        print("I cannot handle multiple inputs models yet")
        quit()
    else:
        min_position = input_shapes[0].index(min(input_shapes[0][1:]))
        if min_position == 1:
            print("NCHW detected")
            order = "NCHW"
        elif min_position == 3:
            print("NHWC detected")
            order = "NHWC"
        else:
            print("Unknown input format")
            quit()
    
    return order
   
def list_files_in_directory(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

class DataReader:
    def __init__(self, calibration_folder, batch_size, target_size, inputname):
        self.calibration_folder = calibration_folder
        self.batch_size = batch_size
        self.target_size = target_size
        self.inputname = inputname
        self.image_paths = self.load_image_paths()
        self.batch_index = 0

    def load_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.calibration_folder):
            for file in files:
                image_paths.append(os.path.join(root, file))
        return image_paths

    def read_batch(self):
        batch = []
        for i in range(self.batch_size):
            if self.batch_index >= len(self.image_paths):
                break
            image_path = self.image_paths[self.batch_index]
            image = Image.open(image_path)

            if image is not None:
                # temp = np.array(image)
                # automatically transpose according to model input shape
                # print(f'Image shape = {temp.shape}')
                # print(f'original model input size = {self.target_size}')

                min_position = self.target_size.index(min(self.target_size[1:]))
                if min_position == 1:
                    # print("NCHW detected")
                    newshape = self.target_size[2:]
                elif min_position == 3:
                    # print("NHWC detected")
                    newshape = self.target_size[1:3]
                else:
                    print("Unknown input format")
                    quit()
                image = np.array(image.resize(newshape))

                # eventually transpose it
                #image = np.transpose(image, (2, 0, 1))
                image = image.astype(np.float32) / 255.0
                batch.append(image)
            self.batch_index += 1

        if not batch:
            return None
        else:
            #print(f'returned read batch data reader shape {np.array(batch).shape}')
            #print(np.array(batch))
            return {self.inputname: np.array(batch)}

    def reset(self):
        self.batch_index = 0
    
    def get_next(self):
        # print(f'returned next data reader  {self.read_batch()["input"].shape}')
        return self.read_batch()

def ggquantize(args):
    input_model_path = args.model
    imagenet_directory = args.calib
    
    # `output_model_path` is the path where the quantized model will be saved.
    output_INT8_NHWC_model_path = "NHWC_qdq_U8S8.onnx"

    # `calibration_dataset_path` is the path to the dataset used for calibration during quantization.
    calibration_dataset_path = "calibration"

    # 1) check if the model is already quantized
    operators = list_operators(input_model_path)
    if "QuantizeLinear" in operators:
        ggprint("The model is already quantized")
        return input_model_path
    else:
        print(Colors.MAGENTA)
        print("The model is not quantized")
        # 2) recognize the model input format
        [inputname, input_shape] = get_input_format(input_model_path)      
        print(f"input data info: {inputname} {input_shape}")
        order = analyze_input_format(input_shape)
        if order == "NHWC":
            nchw_to_nhwc = False
            print("The input format is already NHWC")
        elif order =="NCHW":
            nchw_to_nhwc = True
            print("conversion to NHWC enabled")
        else:
            print("Something unknown happened")
            quit()

        # 3) prepare the calibration directory
        calib_dir = "calibration"
        copied_images = SetCalibDir(imagenet_directory, calib_dir)
        print(f"Successfully copied {copied_images} RGB images to {calib_dir}")

        # Data Reader is a utility class that 
        # reads the calibration dataset and prepares it for the quantization process.
        # Initialize the DataReader
        
        #data_reader = DataReader(calibration_dataset_path, batch_size=1, target_size=(224, 224))
        data_reader = DataReader( calibration_folder=calibration_dataset_path, batch_size=1, target_size=input_shape[0], inputname=inputname)

        # 4) Quantize
        # `quantize_static` is a function that applies static quantization to the model.
        # The parameters of this function are:
        # - `input_model_path`: the path to the original, unquantized model.
        # - `output_model_path`: the path where the quantized model will be saved.
        # - `dr`: an instance of a data reader utility, which provides data for model calibration.
        # - `quant_format`: the format of quantization operators. Need to set to QDQ or QOperator.
        # - `activation_type`: the data type of activation tensors after quantization. In this case, it's QUInt8 (Quantized Unsigned Int 8).
        # - `weight_type`: the data type of weight tensors after quantization. In this case, it's QInt8 (Quantized Int 8).   

        vai_q_onnx.quantize_static(
            input_model_path,
            output_INT8_NHWC_model_path,
            data_reader,
            quant_format=vai_q_onnx.QuantFormat.QDQ,
            calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            enable_dpu=True,
            convert_nchw_to_nhwc=nchw_to_nhwc,
            extra_options={
                'ActivationSymmetric':True,
            }
        )


        print('Calibrated and quantized NHWC model saved at:', output_INT8_NHWC_model_path)
        print(Colors.RESET)
        return output_INT8_NHWC_model_path
