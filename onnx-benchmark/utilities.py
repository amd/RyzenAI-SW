# release 22

import pandas as pd
import numpy as np
import os
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
import inspect
import gc
import textwrap

import onnxruntime
import onnx
from onnx import version_converter, helper, shape_inference
import onnx_tool
import random
import urllib.request
from PIL import Image
from typing import Protocol, Any, Dict, List, Mapping, Sequence, Tuple, Union, Optional

#import quark.onnx.quantization.config.custom_config as customconfig
from onnxruntime.quantization.calibrate import CalibrationDataReader
from quark.onnx.quantization.config import Config, get_default_config
from quark.onnx import ModelQuantizer

class DynamicInputError(RuntimeError):
    """Raised when a model contains unsupported dynamic input dimensions."""
    pass


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

'''
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
                min_position = self.target_size.index(min(self.target_size[1:]))
                if min_position == 1:
                    # print("NCHW detected")
                    newshape = self.target_size[2:]
                    image = np.array(image.resize(newshape))
                    image = np.transpose(image, (2, 0, 1))
                elif min_position == 3:
                    # print("NHWC detected")
                    newshape = self.target_size[1:3]
                    image = np.array(image.resize(newshape))
                else:
                    print("Unknown input format")
                    quit()

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
'''

class PowerMeter:
    def __init__(self, tool="AGM", resdir="results", suffix="log"):

        self.tool = tool.upper()
        self.resdir = resdir
        self.suffix = suffix
        self.timestamp = 0
        self.process = None
        self.agm_log_file = None
        self.hwinfo_log_file = None
        self.energy = 0

        agm_dir = os.getenv("AGM_INSTALLATION_PATH", "")
        hwinfo_dir = os.getenv("HWINFO_INSTALLATION_PATH", "")
        self.agm_path = os.path.join(agm_dir, "AMDGraphicsManager.exe")
        self.hwinfo_path = os.path.join(hwinfo_dir, "HWiNFO64.EXE")

        self.agm_cols = ["Time Stamp", 
                         "CPU0 Power Correlation SOCKET Power",
                         "CPU0 Power Correlation VDDCR_VDD Power",
                         "CPU0 Power Correlation VDDCR_SOC Power",
                         "CPU0 Power Correlation VDD_MEM  Power",
                         "CPU0 Frequencies Actual Frequency MPNPUCLK",
                         "CPU0 Frequencies Actual Frequency NPUHCLK",
                         "CPU0 Frequencies Actual Frequency FCLK",
                         "CPU0 Frequencies Actual Frequency LCLK",
                         ]

        self.hwinfo_cols = ["Time",
                            "CPU Package Power [W]",
                            "CPU Core Power (SVI3 TFN) [W]",
                            "CPU SoC Power (SVI3 TFN) [W]",
                            "NPU Clock [MHz]"
                            ]
                 
        # Ensure results directory exists
        os.makedirs(self.resdir, exist_ok=True)

    def start(self):
        """ Start logging power metrics using the selected tool. """
        self.newtimestamp()    
        self.agm_log_file = f"./{self.resdir}/{self.timestamp}_agm_{self.suffix}.csv"    
        self.hwinfo_log_file = f"./{self.resdir}/{self.timestamp}_hwinfo_{self.suffix}.csv"    

        if self.tool == "AGM":
            self.start_agm()
        elif self.tool == "HWINFO":
            self.start_hwinfo()
        elif self.tool == "BOTH":
            self.start_agm()
            self.start_hwinfo()
        else:
            pass

    def stop(self):
        """ Stop the running power measurement tool. """
        if self.tool == "AGM":
            self.stop_agm()
            time.sleep(2)
            self.filteragm(self.agm_log_file)
            #self.plotme(self.agm_log_file)
        elif self.tool == "HWINFO":
            self.stop_hwinfo()
            time.sleep(2)
            self.filterhwinfo(self.hwinfo_log_file)
            #self.plotme(self.hwinfo_log_file)
        elif self.tool == "BOTH":
            self.stop_agm()
            self.stop_hwinfo()
            time.sleep(2)
            self.filteragm(self.agm_log_file)
            self.plotme(self.agm_log_file)
            self.filterhwinfo(self.hwinfo_log_file)
            self.plotme(self.hwinfo_log_file)
        else:
            pass

    def start_agm(self):
        # Check if AGM is installed
        agm_dir = os.getenv("AGM_INSTALLATION_PATH")
        if not agm_dir or not os.path.exists(os.path.join(agm_dir, "AMDGraphicsManager.exe")):
            print("Error: AGM is not installed or the AGM_INSTALLATION_PATH variable is not set correctly.")
            print("Please install AGM or set the correct environment variable.")
            sys.exit(1)

        """ Start AGM power logging. """
        self.terminate_process("AMDGraphicsManager.exe")  # Ensure no existing process

        # Start AGM logging
        self.process = subprocess.Popen(
            [self.agm_path, "-unilog=PM,CLK", "-unilogsetup=unilogsetup.cfg",
             "-unilogperiod=50", "-unilogstopcheck", f"-unilogoutput={self.agm_log_file}"],
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def stop_agm(self):
        """ Stop AGM logging by creating a termination signal file. """
        try:
            with open("terminate.txt", "w") as file:
                pass  # Creates an empty file
            ggprint("AGM logging stopped.")
        except IOError as e:
            ggprint(f"Error stopping AGM: {e}")

    def start_hwinfo(self):
        # Check if HWINFO is installed
        hwinfo_dir = os.getenv("HWINFO_INSTALLATION_PATH")
        if not hwinfo_dir or not os.path.exists(os.path.join(hwinfo_dir, "HWiNFO64.EXE")):
            print("Error: HWINFO is not installed or the HWINFO_INSTALLATION_PATH variable is not set correctly.")
            print("Please install HWINFO or set the correct environment variable.")
            sys.exit(1)

        """ Start HWInfo power logging. """
        self.process = subprocess.Popen([self.hwinfo_path, f"-l{self.hwinfo_log_file}", "-poll_rate=50"])

    def stop_hwinfo(self):
        """ Stop HWInfo logging. """
        if self.process:
            self.process.terminate()
            ggprint("HWInfo logging stopped.")

    def terminate_process(self, process_name):
        """ Terminate a running process by name. """
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] == process_name:
                    proc.terminate()
                    proc.wait()
                    ggprint(f"Terminated process '{process_name}' (PID: {proc.pid}).")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        ggprint(f"No active process found: '{process_name}'")
        return False
    
    def filteragm(self,csvfile):
        # Read the specific columns
        datapower = pd.read_csv(csvfile, usecols=self.agm_cols)
        datapower = datapower.iloc[:-2]  # Rimuove le ultime 2 righe
        # Convert Time Stamp to seconds
        datapower["Time Stamp"] = datapower["Time Stamp"].apply(self.str_to_sec)
        datapower["Time Stamp"] -= datapower["Time Stamp"].min()
        # Rename columns
        rename_dict = {
            "Time Stamp": "Time",
            "CPU0 Power Correlation SOCKET Power": "APU Power",
            "CPU0 Power Correlation VDDCR_VDD Power": "CPU GPU Power",
            "CPU0 Power Correlation VDDCR_SOC Power": "NPU Power",  
            "CPU0 Power Correlation VDD_MEM  Power": "MEM_PHY Power",
            "CPU0 Frequencies Actual Frequency FCLK": "FCLK",
            "CPU0 Frequencies Actual Frequency NPUHCLK": "NPUHCLK",   
            "CPU0 Frequencies Actual Frequency MPNPUCLK": "MPNPUCLK",
            "CPU0 Frequencies Actual Frequency LCLK": "LCLK"
            }
        datapower.rename(columns=rename_dict, inplace=True)
        datapower.to_csv(csvfile, index=False)
    
    def filterhwinfo(self,csvfile):
        # Read HWINFO file
        datapower = pd.read_csv(csvfile, usecols=self.hwinfo_cols, encoding="latin1")
        # Convert Time format
        initial_time = self.time_to_seconds(datapower.iloc[0, 0])
        datapower["Time"] = datapower.iloc[:, 0].apply(lambda x: self.time_to_seconds(x) - initial_time)
        # Ensure the required power column is named correctly
        rename_dict = {
            "CPU Package Power [W]": "APU Power",
            "CPU Core Power (SVI3 TFN) [W]": "CPU GPU Power",
            "CPU SoC Power (SVI3 TFN) [W]": "NPU Power",
            "NPU Clock [MHz]": "NPUHCLK"
            }
        
        datapower.rename(columns=rename_dict, inplace=True)
        datapower.to_csv(csvfile, index=False)

    def newtimestamp(self):
        self.timestamp = time.strftime("%m%d_%H%M")
        ggprint(f"Timestamp = {self.timestamp}")

    def str_to_sec(self,date_string):
        date_string = date_string.strip("--[ ]").strip()
        # date_format = "%d.%m.%Y %H:%M:%S.%f"
        time_str = date_string.split(" ")[1]
        seconds = self.time_to_seconds(time_str)
        return seconds
     
    def time_to_seconds(self,time_str):
        h, m, s = map(float, time_str.split(":"))
        return h * 3600 + m * 60 + s

    def plotme(self,csvfile):
        try:
            data = pd.read_csv(csvfile)

            # Compute average power
            self.avepower = data["APU Power"].mean()

            # Convert to numpy arrays (float) so trapz sees the right types
            time_vals  = data["Time"].to_numpy(dtype=float)
            power_vals = data["APU Power"].to_numpy(dtype=float)

            # Compute energy (integral of power over time)
            self.energy = np.trapz(y=power_vals, x=time_vals)

            # Plot the graph (Power vs. Time)
            plt.figure(figsize=(10, 6))
            plt.plot(data["Time"], data["APU Power"], label="APU Power")
            plt.axhline(y=self.avepower, color="black", linestyle="--", linewidth=1, label=f"Average Power {self.avepower:.2f} [W]")

            plt.xlabel("Time (s)")
            plt.ylabel("Power (W)")
            plt.title(f"Power vs. Time")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            maxy=80
            miny=0
            plt.ylim(miny, maxy)
            # Add a label at the bottom middle of the graph
            plt.text(x=max(data["Time"]) / 2, y=miny+5, s=f"Energy = {self.energy:.2f} [J]", 
                     fontsize=12, ha="center", color="black")
            # Add a label at the top left
            plt.text(x=0, y=maxy-5, s=f"{os.path.basename(csvfile)}", 
                     fontsize=12, ha="left", color="black")

            save_filepath = csvfile.replace("csv","png")
            plt.savefig(save_filepath, dpi=300, bbox_inches='tight')


        except FileNotFoundError:
            ggprint(f"Error: File {csvfile} not found.")
        except Exception as e:
            ggprint(f"An error occurred: {e}")


def analyze_input_format(input_shape):
    order = "unknown"
    min_position = input_shape.index(min(input_shape[1:]))
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

def appendcsv(measurement, args, csv_file="measurements.csv"):
    fieldnames = [
        "timestamp",
        "command",
        "p_batchsize",
        "p_config",
        "p_core",
        "p_execution_provider",
        "p_infinite",
        #"p_instance_count",
        "p_intra_op_num_threads",
        "p_json",
        "p_min_interval",
        "p_model",
        "p_no_inference",
        "p_num",
        "p_power",
        "p_renew",
        "p_timelimit",
        "p_threads",
        "p_warmup",
        "benchmark_release",
        "model",
        "execution_provider",
        "total_throughput",
        "average_latency",
        "apu_perf_pow",
        "energy_apu",
        "energy_cpu",
        "energy_npu",
        "energy_mem",
        "MPNPUCLK",
        "NPUHCLK",
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
        "npu_driver",
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
                "p_batchsize": args.batchsize,
                "p_config": args.config,
                "p_core": args.core,
                "p_execution_provider": args.execution_provider,
                "p_infinite": args.infinite,
                #"p_instance_count": args.instance_count,
                "p_intra_op_num_threads": args.intra_op_num_threads,
                "p_json": args.json,
                "p_min_interval": args.min_interval,
                "p_model": args.model,
                "p_no_inference": args.no_inference,
                "p_num": args.num,
                "p_power": args.power,
                "p_renew": args.renew,
                "p_timelimit": args.timelimit,
                "p_threads": args.threads,
                "p_warmup": args.warmup,
                "benchmark_release": measurement["run"]["benchmark_release"],
                "model": measurement["run"]["model"],
                "execution_provider": measurement["run"]["execution_provider"],
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
                "energy_npu": measurement["results"]["energy mJ/frame"]["npu"],
                "energy_mem": measurement["results"]["energy mJ/frame"]["mem"],
                "MPNPUCLK": measurement["system"]["frequency"]["MPNPUCLK"],
                "NPUHCLK": measurement["system"]["frequency"]["NPUHCLK"],
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
                "npu_driver": measurement["system"]["driver"]["npu"],
                "xclbin_path": measurement["environment"]["xclbin"]["xclbin_path"] if args.execution_provider == "VitisAIEP" else "",
                #"vaip": measurement["environment"]["xclbin"]["packages"]["vaip"]["version"],
                #"target_factory": measurement["environment"]["xclbin"]["packages"]["target_factory"]["version"],
                #"xcompiler": measurement["environment"]["xclbin"]["packages"]["xcompiler"]["version"],
                #"onnxruntime": measurement["environment"]["xclbin"]["packages"]["onnxrutnime"]["version"],
                #"graph_engine": measurement["environment"]["xclbin"]["packages"]["graph_engine"]["version"],
                #"xrt": measurement["environment"]["xclbin"]["packages"]["xrt"]["version"],
            }
        )
    ggprint(f"Data appended to {csv_file}")


def check_package_version(package_name):
    try:
        version = importlib_metadata.version(package_name)
        return version
    except importlib_metadata.PackageNotFoundError:
        return f"{package_name} not found"

def check_env(release, args):
    data = [
        ("PACKAGE", "STATUS"),
    ]
    ggprint(
        "------------------------------------------------------------------------------"
    )
    ggprint("Preliminary environment check")

    required_vars = [
        "RYZEN_AI_INSTALLATION_PATH",
    ]
    missing_vars = [var for var in required_vars if os.getenv(var) is None]
    if missing_vars:
        ggprint(f"Error: Missing environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    if (
        sys.version_info.major == 3
        and 10 <= sys.version_info.minor <= 12
    ):
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
                Colors.RED + "please use Python 3.10-3.12" + Colors.RESET,
            )
        )

    package_name = "onnxruntime-vitisai"
    version = check_package_version(package_name)
    fields = version.split('.')
    if fields[0] == "1" and fields[1] in ["21","22","23"]:
        data.append(
            (f"{package_name} version: {version}", Colors.GREEN + "OK" + Colors.RESET)
        )
    else:
        data.append(
            (
                f"{package_name} version: {version}" + version,
                Colors.RED + f"please update {package_name}" + Colors.RESET,
            )
        )

    package_name = "voe"
    version = check_package_version(package_name)
    fields = version.split('.')
    if fields[0] == "1" and (fields[1] == "6"):
        data.append(
            (f"{package_name} version: {version}", Colors.GREEN + "OK" + Colors.RESET)
        )
    else:
        data.append(
            (f"{package_name} version: {version}", Colors.RED + f"please update {package_name}" + Colors.RESET)
        )

    # AGM
    package_name = "AGM"
    # Get the path from the environment variable
    agm_path = os.getenv("AGM_INSTALLATION_PATH", r"C:\Program Files\AMD Graphics Manager")
    agmexe_path = os.path.join(agm_path, "AMDGraphicsManager.exe")

    agm_version = get_file_version(agmexe_path)
    if agm_version:
        data.append(
            (f"{package_name} version: {agm_version}", Colors.GREEN + "OK" + Colors.RESET)
        )

    # HWINFO
    package_name = "HWINFO"
    # Get the path from the environment variable
    hwinfo_path = os.getenv("HWINFO_INSTALLATION_PATH", r"C:\Program Files\HWiNFO64")
    hwinfoexe_path = os.path.join(hwinfo_path, "HWiNFO64.EXE")

    hwinfo_version = get_file_version(hwinfoexe_path)
    if hwinfo_version:
        data.append(
            (f"{package_name} version: {hwinfo_version}", Colors.GREEN + "OK" + Colors.RESET)
        )

    max_width = max(len(row[0]) for row in data)

    for row in data:
        column1, column2 = row
        # Left-align the text in the first column and pad with spaces
        formatted_column1 = column1.ljust(max_width)
        ggprint(f"{formatted_column1} {column2}")
    
def check_args(args, defaults):
    assert args.num >= (
        #args.batchsize * args.instance_count
        args.batchsize
    ), "runs must be greater than batches"

    #total_cpu = os.cpu_count()
    #if args.instance_count > total_cpu:
    #    args.instance_count = total_cpu
    #    ggprint(f"Limiting instance count to max cpu count ({total_cpu})")

    if args.execution_provider == "VitisAIEP":
        assert os.path.exists(defaults['config']) or os.path.exists(args.config), (
            f"ERROR: Neither the default config path {defaults['config']} nor the provided config path {args.config} exists."
        )

def check_silicon(expected, found):
    if expected != found:
        raise ValueError(f"Mismatch between {expected} xclbin and {found} driver")
    else:
        return expected

def cancelcache(cache_path):
    ggprint(80*"-")
    ggprint("Cleaning cache")
    if os.path.exists(cache_path) and os.path.isdir(cache_path):
        try:
            shutil.rmtree(cache_path)
            ggprint(f"Deleted {cache_path}")
        except Exception as e:
            ggprint(f"Error during cache cancellation: {e}")
    else:
        ggprint(f"{cache_path} does not exist or is not a directory")


def dbprint(message):
    frame = inspect.currentframe()
    caller_frame = frame.f_back # type: ignore
    file_name = caller_frame.f_code.co_filename # type: ignore
    line_number = caller_frame.f_lineno # type: ignore
    print(Colors.BLUE + f"File: {file_name}, Line: {line_number} - {message}" + Colors.RESET)

def del_file(killme):
    # Check if the file exists before attempting to delete it
    if os.path.exists(killme):
        os.remove(killme)
        #ggprint(f"Old measurement {measfile} deleted successfully")


def extract_flops_mem_params(file_path):
    delimiter_pattern = re.compile(r' {2,}')
    column_name = "Forward_FLOPs"

    # Read all non-empty lines
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        if len(lines) < 2:
            raise ValueError(f"{file_path} must have at least a header and one data row")

    # Parse header
    header_cols = delimiter_pattern.split(lines[0])
    if column_name not in header_cols:
        raise ValueError(f"The column '{column_name}' was not found in the header of {file_path}")

    column_number = header_cols.index(column_name)

    # Parse the last line of data
    data_cols = delimiter_pattern.split(lines[-1])
    try:
        flops  = int(data_cols[column_number    ].replace(',', ''))
        mem    = int(data_cols[column_number + 2].replace(',', ''))
        params = int(data_cols[column_number + 4].replace(',', ''))
    except (IndexError, ValueError) as e:
        raise ValueError(f"Failed to parse data columns at index {column_number} in {file_path}") from e

    return {
        "Forward_FLOPs": flops,
        "Memory":        mem,
        "Params":        params,
    }


def _choose_model(candidates: List[Path]) -> Path:
    """Select the largest ONNX (by file size) to avoid auxiliary artifacts."""
    if not candidates:
        raise ValueError("No candidates provided")
    ordered = sorted(((p.stat().st_size, p) for p in candidates), key=lambda item: (-item[0], str(item[1])))
    return ordered[0][1]


def _find_onnx_in_dir(directory: Path) -> List[Path]:
    """Return all .onnx files under the directory tree, sorted lexicographically."""
    return sorted(directory.rglob('*.onnx'))


def _download_onnx(url: str, out_path: Path) -> None:
    """Download an ONNX model atomically and perform a minimal size sanity check."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + '.part')
    ggprint(f"[Downloader] Fetching: {url}")
    urllib.request.urlretrieve(url, tmp)
    if tmp.stat().st_size < 10_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file looks too small: {url}")
    tmp.replace(out_path)


KNOWN_DOWNLOADABLE_MODELS: Dict[str, Dict[str, str]] = {
        'resnet50': {
            'subdir': 'resnet50',
            'filename': 'resnet50-v2-7.onnx',
            'url': 'https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx',
        },
        'resnet50-v2-7': {
            'subdir': 'resnet50',
            'filename': 'resnet50-v2-7.onnx',
            'url': 'https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx',
        },
        'resnet50v2': {
            'subdir': 'resnet50',
            'filename': 'resnet50-v2-7.onnx',
            'url': 'https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx',
        },
        'resnet50-v1-7': {
            'subdir': 'resnet50',
            'filename': 'resnet50-v1-7.onnx',
            'url': 'https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx',
        },
        'resnet50v1': {
            'subdir': 'resnet50',
            'filename': 'resnet50-v1-7.onnx',
            'url': 'https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-7.onnx',
        },
        'mobilenetv2': {
            'subdir': 'mobilenetv2',
            'filename': 'mobilenetv2-12.onnx',
            'url': 'https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx',
        },
        'mobilenetv2-12': {
            'subdir': 'mobilenetv2',
            'filename': 'mobilenetv2-12.onnx',
            'url': 'https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx',
        },
    }

def get_downloadable_model_selectors() -> List[str]:
    """Return the list of model selectors with built-in download support."""
    return sorted(KNOWN_DOWNLOADABLE_MODELS.keys())

def ensure_model_exists(
    model_selector: Optional[str],
    model_path: Optional[str],
    models_root: str = 'models',
    overwrite: bool = False,
) -> Path:
    """Resolve an ONNX path from either an explicit path or a named selector, downloading if needed."""

    if model_path:
        candidate = Path(model_path)
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            matches = _find_onnx_in_dir(candidate)
            if not matches:
                raise FileNotFoundError(
        f"No local model found for selector '{selector}', and no downloader is configured for it.\n"
        f"- Put the model under './{models_root}/{selector}/' as a .onnx file, OR\n"
        f"- Use --model_path to point to a specific .onnx file or directory, OR\n"
        f"- Extend 'KNOWN_DOWNLOADABLE_MODELS' in ensure_model_exists() with a URL for this selector."
    )

    selector = (model_selector or '').strip()
    if not selector:
        raise ValueError("Either --model_path must be provided or --model selector must be set.")

    selector_dir = Path(models_root) / selector
    if selector_dir.is_dir():
        matches = _find_onnx_in_dir(selector_dir)
        if matches and not overwrite:
            return _choose_model(matches)



    sel_norm = selector.lower()
    if sel_norm in KNOWN_DOWNLOADABLE_MODELS:
        info = KNOWN_DOWNLOADABLE_MODELS[sel_norm]
        out_dir = Path(models_root) / info['subdir']
        out_path = out_dir / info['filename']
        if overwrite or not out_path.exists():
            _download_onnx(info['url'], out_path)
        return out_path

    raise FileNotFoundError(
        f"No local model found for selector '{selector}', and no downloader is configured for it.\n"
        f"- Put the model under './{models_root}/{selector}/' as a .onnx file, OR\n"
        f"- Use --model_path to point to a specific .onnx file or directory, OR\n"
        f"- Extend 'KNOWN_DOWNLOADABLE_MODELS' in ensure_model_exists() with a URL for this selector."
    )


def _describe_dim(dim) -> str:
    if dim.HasField('dim_value') and dim.dim_value > 0:
        return str(dim.dim_value)
    if dim.HasField('dim_param') and dim.dim_param:
        return dim.dim_param
    return '?'



def remove_batchnorm_spatial_attribute(model_path: Union[str, Path]) -> bool:
    """Drop the legacy 'spatial' attribute from BatchNormalization nodes.

    Returns True when the file is rewritten."""
    mp = Path(model_path)
    if not mp.exists():
        raise FileNotFoundError(mp)

    model = onnx.load(mp.as_posix())
    changed = False

    for node in model.graph.node:
        if node.op_type != 'BatchNormalization':
            continue
        keep = [attr for attr in node.attribute if attr.name != 'spatial']
        if len(keep) != len(node.attribute):
            node.ClearField('attribute')
            node.attribute.extend(keep)
            changed = True

    if changed:
        onnx.save(model, mp.as_posix())
        ggprint(f"[INFO] Removed deprecated 'spatial' attribute from BatchNormalization nodes in {mp.name}.")

    return changed


def fix_model_batch1(model_path: str, out_path: Optional[str] = None) -> str:
    """Force batch dimension to 1 when dynamic while rejecting other dynamic shapes."""
    mp = Path(model_path)
    outp = Path(out_path) if out_path else mp.with_name(mp.stem + '_BS1' + mp.suffix)

    model = onnx.load(mp.as_posix())
    init_names = {init.name for init in model.graph.initializer}

    batch_fixed = False

    for value in model.graph.input:
        if value.name in init_names:
            continue
        ttype = value.type.tensor_type
        if not ttype.HasField('shape'):
            continue
        dims = list(ttype.shape.dim)
        if not dims:
            continue

        found_non_batch_dynamic = []
        batch_is_dynamic = False

        for idx, dim in enumerate(dims):
            has_param = dim.HasField('dim_param') and bool(dim.dim_param)
            has_value = dim.HasField('dim_value')
            dim_value = dim.dim_value if has_value else None
            is_dynamic = False

            if has_param:
                is_dynamic = True
            elif has_value and dim_value is not None and dim_value <= 0:
                is_dynamic = True
            elif (not has_param) and (not has_value):
                is_dynamic = True

            if is_dynamic:
                if idx == 0:
                    batch_is_dynamic = True
                else:
                    found_non_batch_dynamic.append((idx, _describe_dim(dim)))

        if found_non_batch_dynamic:
            pretty = ', '.join(f"dim[{idx}]={desc}" for idx, desc in found_non_batch_dynamic)
            raise DynamicInputError(
                f"Input '{value.name}' has unsupported dynamic dimensions ({pretty}). "
                "Only fixed input shapes are supported."
            )

        if batch_is_dynamic:
            d0 = dims[0]
            d0.ClearField('dim_param')
            d0.dim_value = 1
            batch_fixed = True
            ggprint("[WARN] Detected dynamic batch dimension. Only fixed shapes are supported; forcing batch size to 1.")

    if not batch_fixed:
        return mp.as_posix()

    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    onnx.save(model, outp.as_posix())
    ggprint(f"[INFO] Saved batch-size-adjusted model to {outp.as_posix()}")
    return outp.as_posix()



def get_driver_release_number(device_name):
    command = f'powershell -Command "Get-WmiObject Win32_PnPSignedDriver | Where-Object {{ $_.DeviceName -like \'*{device_name}*\' }} | Select-Object DeviceName, DriverVersion"'
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        ggprint("Error running PowerShell command:")
        ggprint(result.stderr)
        return
    return result.stdout.split("\n")[3].split(" ")[4].strip()

def get_file_version(file_path):
    """Retrieve file version from Windows executable properties using PowerShell."""
    try:
        result = subprocess.run(
            ["powershell", "(Get-Item '{}' ).VersionInfo.ProductVersion".format(file_path)],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def ggprint(linea):
    print(Colors.DIMMERED_WHITE + linea + Colors.RESET)

def ggquantize(args):
    input_model_path = getattr(args, "model_path", None) or args.model
    calib_data_folder = args.calib
 
    base_name, extension = os.path.splitext(input_model_path)

    # OPSET update to 17
    # A full list of supported adapters can be found here:
    # https://github.com/onnx/onnx/blob/main/onnx/version_converter.py#L21
    # Apply the version conversion on the original model

    # Preprocessing: load the model to be converted.   
    newopset = 20
    original_model = onnx.load(input_model_path)
    opset_version = original_model.opset_import[0].version
    if opset_version<11:
        ggprint(f'The model OPSET is {opset_version} and should be updated to {newopset}')
        if args.update_opset == '1':
            output_updated = f"{base_name}_opset{newopset}{extension}"
            converted_model = version_converter.convert_version(original_model, newopset)
            onnx.save(converted_model, output_updated)
            ggprint(f'The update model was saved with name {output_updated}')
            input_model_path = output_updated
            base_name, extension = os.path.splitext(input_model_path)
        else:
            ggprint("You opted not to update")
    # free memory
    del original_model
    gc.collect()
   
    # check if the model is already quantized
    # Define ONNX data type mapping
    onnx_data_types = {
        onnx.TensorProto.FLOAT: "FP32",
        onnx.TensorProto.UINT8: "INT8",
        onnx.TensorProto.INT8: "INT8",
        onnx.TensorProto.BFLOAT16: "BF16",
        onnx.TensorProto.FLOAT16: "FP16",
    }
    
    def detect_onnx_precision(model_path):
        """Detects the precision (FP32, INT8, BF16) of an ONNX model."""
        model = onnx.load(model_path)
        tensor_data_types = set()
    
        # Iterate through model initializers and inputs to collect data types
        for initializer in model.graph.initializer:
            tensor_data_types.add(initializer.data_type)
        
        for input_tensor in model.graph.input:
            if input_tensor.type.tensor_type.HasField("elem_type"):
                tensor_data_types.add(input_tensor.type.tensor_type.elem_type)
    
        # Map detected data types to readable format
        detected_types = {onnx_data_types.get(dtype, "UNKNOWN") for dtype in tensor_data_types}

        # Determine model precision
        if "INT8" in detected_types:
            return "INT8"
        elif "BF16" in detected_types:
            return "BF16"
        elif "FP32" in detected_types:
            return "FP32"
        else:
            return "UNKNOWN"
    
    datatype = detect_onnx_precision(input_model_path)
    ggprint(f"The ONNX model data type is: {datatype}")
      
    if datatype in ["INT8"]:
        remove_batchnorm_spatial_attribute(input_model_path)
        return input_model_path
    elif datatype in ["BF16", "FP32"]:
        if args.autoquant=="1":
            ggprint("Quantizing to INT8 with Quark")
            quantized_output_model= f"{base_name}_{args.quarkpreset}{extension}"
            if args.renew == "1":
                cache_dir = os.path.join(Path(__file__).parent.resolve(), "cache", os.path.basename(quantized_output_model))
                cancelcache(cache_dir)

            print(Colors.MAGENTA)
            # NCHW or NHWC? we want NHWC (Channel last)
            nchw_to_nhwc = True
            input_name, input_shape, numinputs = get_input_info(input_model_path)
            if numinputs==1:
                print(f"Model Input Name: {input_name}, Model Input Shape: {input_shape}")

                order = analyze_input_format(input_shape)
                if order == "NHWC":
                    nchw_to_nhwc = False
                    print("The input format is already NHWC - this is the optimal shape")
                    # calibration images are OK
                    # model is OK
                elif order =="NCHW":
                    nchw_to_nhwc = True
                    print("The input format is NCHW - conversion to NHWC enabled")
                    # calibration images will be transposed
                    # model will be turned to NHWC

                data_reader =ImageDataReader(calib_data_folder, input_model_path, args.num_calib, 1, order)
                quant_config = get_default_config(args.quarkpreset)
                quant_config.convert_nchw_to_nhwc= nchw_to_nhwc
                config = Config(global_quant_config=quant_config)
                quantizer = ModelQuantizer(config)
                quantizer.quantize_model(input_model_path, quantized_output_model, data_reader)
                remove_batchnorm_spatial_attribute(quantized_output_model)
                print(f'Quark quantized model {quantized_output_model} saved')
                print(Colors.RESET)
                return quantized_output_model
            else:
                ggprint(f'Auto quantization is not yet supported for multiple inputs models')
                ggprint(Colors.RESET)
                sys.exit(1)

        else:
            ggprint("[WARNING] VAIML will compile the model")
            remove_batchnorm_spatial_attribute(input_model_path)
            return input_model_path
    else:
        ggprint("[ERROR] unknown model data type")
        sys.exit(1)        

# def get_input_format(onnx_model_path):
#     onnx_model = onnx.load(onnx_model_path)
#     input_shapes = []
#     for input_node in onnx_model.graph.input:
#         shape = [dim.dim_value or dim.dim_param for dim in input_node.type.tensor_type.shape.dim]
#         input_shapes.append(tuple(shape))
#     return [input_node.name, input_shapes]
# Replaced by more solid:
def get_input_format(onnx_model_path: str) -> Tuple[List[str], List[Tuple[int, ...]]]:
    model = onnx.load(onnx_model_path)

    input_names:  List[str]                = []
    input_shapes: List[Tuple[int, ...]]    = []

    for node in model.graph.input:
        input_names.append(node.name)
        shape = tuple(
            dim.dim_value if dim.dim_value > 0 else -1
            for dim in node.type.tensor_type.shape.dim
        )
        input_shapes.append(shape)

    if not input_names:
        raise ValueError(f"No inputs found in {onnx_model_path}")

    return input_names, input_shapes


def get_input_info(onnx_model_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    
    # Build a set of all initializer names (these aren’t true graph inputs)
    init_names = {init.name for init in model.graph.initializer}
    
    # Filter out those from the graph.input list
    real_inputs = [
        inp for inp in model.graph.input
        if inp.name not in init_names
    ]
    
    # How many “real” inputs we have
    num_inputs = len(real_inputs)
    
    # Grab name & shape of the first real input (if any)
    if num_inputs > 0:
        first = real_inputs[0]
        input_name  = first.name
        input_shape = [dim.dim_value for dim in first.type.tensor_type.shape.dim]
    else:
        input_name, input_shape = None, []
    
    return input_name, input_shape, num_inputs


def initcsv(filename, R, C):
    data = np.full((R, C), np.nan)
    if os.path.isfile(filename):
        os.remove(filename)
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, header=False)


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

def list_files_in_directory(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def log_battery_percentage(start_time, timestamp, resdir):
    elapsed_time = round(time.time() - start_time)  # Time elapsed since start
    battery = psutil.sensors_battery()
    battery_percentage = battery.percent
    if battery:
        print(f"Battery percentage: {battery_percentage}%")
        print(f"Power plugged: {battery.power_plugged}")
        if battery.power_plugged:
            print("Please unplug the power cord if you want to pair the power consumption estimation with the battery discharge measurement.")
    else:
        print("Battery information not available")   

    file_name = f"./{resdir}/battery_{timestamp}.csv"
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists or os.stat(file_name).st_size == 0:
            writer.writerow(["time", "battery"])

        writer.writerow([elapsed_time, battery_percentage])


def meas_init(args, release, total_throughput, average_latency, xclbin_path, model_input):
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
    measurement["run"]["execution_provider"] = args.execution_provider

    measurement["results"] = {}
    measurement["results"]["performance"] = {}
    measurement["results"]["performance"]["total_throughput"] = total_throughput
    measurement["results"]["performance"]["average_latency"] = average_latency
    measurement["results"]["efficency perf/W"] = {}
    measurement["results"]["efficency perf/W"]["apu_perf_pow"] = "N/A"
    measurement["results"]["energy mJ/frame"] = {}
    measurement["results"]["energy mJ/frame"]["apu"] = "N/A"
    measurement["results"]["energy mJ/frame"]["cpu"] = "N/A"
    measurement["results"]["energy mJ/frame"]["npu"] = "N/A"
    measurement["results"]["energy mJ/frame"]["mem"] = "N/A"

    #measurement["model"] = profile_model(args.model, model_input)
    measurement["model"] = args.model

    measurement["vitisai"] = {}
    measurement["vitisai"]["all"] = 0
    measurement["vitisai"]["CPU"] = 0

    measurement["system"] = {}
    measurement["system"]["frequency"] = {}
    measurement["system"]["frequency"]["MPNPUCLK"] = 0
    measurement["system"]["frequency"]["NPUHCLK"] = 0
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
            driver_release=get_driver_release_number("NPU Compute Accelerator Device")
        except subprocess.CalledProcessError:
            ggprint("Error: AMD NPU driver not found or PowerShell command failed.")
            driver_release = "Unknown"
       
    else:
        ggprint("Warning: Could not execute a Powershell command. Please manually verify that the AMD NPU Driver exists")       
        driver_release = "Unknown"

    measurement["system"]["driver"] = {}
    measurement["system"]["driver"]["npu"] = driver_release

    measurement["environment"] = {}
    measurement["environment"]["packages"] = {}

    # info stored in the cache (only with VitisAIEP)
    #if args.execution_provider == "VitisAIEP" and args.cpp!='1':
    if args.execution_provider == "VitisAIEP":
        measurement["environment"]["xclbin"] = {}
        measurement["environment"]["xclbin"]["xclbin_path"] = xclbin_path
        cache_dir = os.path.join(Path(__file__).parent.resolve(), "cache", os.path.basename(args.model))
        with open(os.path.join(cache_dir, r"modelcachekey\context.json"), "r") as json_file:
            data = json.load(json_file)
        releases = data["config"]["version"]["versionInfos"]

        measurement["environment"]["xclbin"]["packages"] = {
            release["packageName"]: {
                "commit": release["commit"],
                "version": release["version"],
            }
            for release in releases
        }
    
    conda_list = ""
    try:
        conda_list = subprocess.check_output("conda list", shell=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    package_info = {}
    # Parse the output to extract package names and versions
    lines = conda_list.strip().split("\n")
    for line in lines[3:]:  # Skip the first 3 lines which contain header information
        parts = re.split(r"\s+", line.strip())
        if len(parts) >= 2:
            package_name = parts[0]
            package_version = parts[1]
            package_info[package_name] = package_version
    measurement["environment"]["packages"] = package_info

    if args.execution_provider == "VitisAIEP":
        measurement["environment"]["vaip_config"] = {}
        with open(args.config, "r") as json_file:
            vaip_conf = json.load(json_file)

        measurement["environment"]["vaip_config"] = vaip_conf

    return measurement

def getmedians(filename):
    df = pd.read_csv(filename)
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
    medians = {col: np.median(df[col].dropna()) for col in df.columns}
    return medians


def parse_cpp_res(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
       
        data = {}
        current_model = None

        for i, line in enumerate(lines):
           
            if (i+1)>len(lines):
                break
            elif line.startswith('LATENCIES:'):
                current_model = lines[i + 1].split()[0]
                latstr = lines[i + 1].split()[1]
                latency = float(re.findall(r'\d+\.\d+', latstr)[0])   
                # dbprint(current_model, latency )
                data[current_model] = {}
                data[current_model]['latency'] = latency
            elif line.startswith('THROUGHPUT:'):             
                current_model = lines[i + 1].split()[0]
                throughputstr = lines[i + 1].split()[1]
                throughput = float(re.findall(r'\d+\.\d+', throughputstr)[0])   
                # dbprint(current_model, throughput )
                data[current_model]['throughput'] = throughput
        return data

def parse_args(apu_type):

    install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']    
    config_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'vaip_config.json')

    defaults = {
        'calib': ".\\images",
        'config': config_file,
        'core': "STX_1x4",
        'execution_provider': 'CPU',
    }

    if len(sys.argv) < 2:
        show_help()
        quit()

    parser = argparse.ArgumentParser()

    parser.add_argument("--batchsize", "-b", type=int, default=1, help="batch size: number of images processed at the same time by the model. VitisAIEP supports batchsize = 1. Default is 1")
    parser.add_argument(
        "--force_batch",
        type=int,
        default=1,
        choices=[0, 1],
        help="If set to 1 (default) rewrite inputs to force batch=1 when the model uses dynamic batches.",
    )

    parser.add_argument(
        "--calib",
        type=str,
        default=defaults['calib'],
        help=f"path to Imagenet database, used for quantization with calibration. Default= ./Imagenet/val ",
    )   
    parser.add_argument(
        "--config", 
        "-c", 
        type=str, 
        default=defaults['config'],
        help="path to config json file. Default= <release>/vaip_config.json",
    )
    
    if apu_type=="STX":
        parser.add_argument(
            "--core",
            default="STX_4x4",
            type=str,
            choices=["STX_4x4"],
            help="Which core to use with STRIX silicon. Default=STX_4x4",
        )
    elif apu_type=="PHX/HPT":
        parser.add_argument(
            "--core",
            default="PHX_4x4",
            type=str,
            choices=["PHX_4x4"],
            help="Which core to use with PHOENIX silicon. Default=PHX_4x4",
        )
    
    parser.add_argument(
        "--cpp",
        type=str,
        default="0",
        choices=["0", "1"],
        help="if this option is set to 1, the tool will use a C++ benchmark code. Default = 0: use Python code.",
    )

    parser.add_argument(
        "--execution_provider",
        "-e",
        type=str,
        default=defaults['execution_provider'],
        choices=["CPU", "VitisAIEP", "iGPU", "dGPU"],
        help="Execution Provider selection. Default=CPU",
    )

    parser.add_argument(
        "--infinite",
        type=str,
        default="1",
        choices=["0", "1"],
        help="if 1: Executing an infinite loop, when combined with a time limit, enables the test to run for a specified duration. Default=1",
    )
    
    # This parameter has been disabled and hardcoded to 1 in release 21
    #parser.add_argument(
    #    "--instance_count",
    #    "-i",
    #    type=int,
    #    default=1,
    #    help="This parameter governs the parallelism of job execution. When the Vitis AI EP is selected, this parameter controls the number of DPU runners. The workload is always equally divided per each instance count. Default=1",
    #)

    parser.add_argument(
        "--intra_op_num_threads", 
        type=int, 
        default=1, 
        help="In general this parameter controls the total number of INTRA threads to use to run the model. INTRA = parallelize computation inside each operator. Specific for VitisAI EP: number of CPU threads enabled when an operator is resolved by the CPU. Affects the performances but also the CPU power consumption. For best performance: set intra_op_num_threads to 0: INTRA Threads Total = Number of physical CPU Cores. For best power efficiency: set intra_op_num_threads to smallest number (>0) that sustains the close to optimal performance (likely 1 for most cases of models running on DPU). Default=1"
        )
    
    parser.add_argument(
        "--json",
        type=str,
        help="Path to the file of parameters.",
    )

    parser.add_argument(
        "--log_csv",
        "-k",
        type=str,
        default="0",
        help="If this option is set to 1, measurement data will appended to a CSV file. Default=0",
    )

    parser.add_argument(
        "--min_interval",
        type=float,
        default=0,
        help="Minimum time interval (s) for running inferences. Default=0",
    )

    parser.add_argument(
        "--model_path",
        "-mp",
        type=str,
        default="",
        help="Explicit path to an ONNX model (file or directory). Overrides --model selector when provided.",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="",
        help="Path to the ONNX model",
    )

    parser.add_argument(
        "--no_inference",
        type=str,
        default="0",
        choices=["0", "1"],
        help="When set to 1 the benchmark runs without inference for power measurements baseline. Default=0",
    )

    parser.add_argument(
        "--num", 
        "-n", 
        type=int, 
        default=100, 
        help="The number of images loaded into memory and subsequently sent to the model. Default=100"
    )
    
    parser.add_argument(
        "--num_calib", 
        type=int, 
        default=10, 
        help="The number of images for calibration. Default=10"
    )

    parser.add_argument(
        "--power",
        "-p",
        type=str,
        default="0",
        choices=["no power", "AGM", "HWINFO", "BOTH"],
        help="This option controls which tool is used for power measurement. No power measurement by default.",
    )

    # quark presets
    parser.add_argument(
        "--quarkpreset",
        type=str,
        help="This option sets the Quark quantization preset to be used.",
    )

    parser.add_argument(
        "--renew",
        "-r",
        type=str,
        default="1",
        choices=["0", "1"],
        help="if set to 1 cancel the cache and recompile the model. Set to 0 to keep the old compiled file. Default=1",
    )
    
    parser.add_argument(
        "--autoquant",
        type=str,
        default="0",
        choices=["0", "1"],
        help="if set to 1 enables auto-quantization with Quark. If the model is FP32 and this option is set to 0, the model is sent to the CPU. Default=0",
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
        "--update_opset",
        type=str,
        default="0",
        choices=["0", "1"],
        help="if set to 1 and the model has opset < 11, automatically update the model opset to 17. Default=0",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        type=str,
        default="0",
        choices=["0", "1", "2"],
        help="0 (default): no debug messages, 1: few debug messages, 2: all debug messages"
    )

    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=40,
        help="Perform warmup runs, default = 40",
    )

    args = parser.parse_args()

    #args, _ = parser.parse_known_args()
    if args.json:
        ggprint("Loading the file of parameters")
        try:
            with open(args.json, "r") as json_file:
                config = json.load(json_file)
            # Update argparse arguments with values from the JSON file
            for arg_name, value in config.items():
                setattr(args, arg_name, value)
        except Exception as e:
            print(f"Error loading the file of parameters: {e}")
    
    return args, defaults  

def plotefficency(mem, npu, cpu, apu, recordfilename):
    stacked_data = [mem, npu, cpu]
    separate_data = [apu]

    categories = ["MEM", "NPU", "CPU"]
    separate_category = ["APU"]
    bar_width = 0.35
    # Create x-axis values for the bars
    x_stacked = np.arange(len(categories))
    x_separate = np.arange(len(categories), len(categories) + len(separate_category))

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the stacked bars
    stacked_bars = ax.bar(x_stacked, stacked_data, bar_width, label="CPU, NPU, MEM")

    # Create the separate bar
    separate_bars = ax.bar(x_separate, separate_data, bar_width, label="APU efficency")

    # Set x-axis labels and tick positions
    ax.set_xticks(np.concatenate((x_stacked, x_separate)))
    ax.set_xticklabels(categories + separate_category)

    # Add labels and a legend
    ax.set_xlabel("Rails")
    ax.set_ylabel("Efficency [fps/W]")
    ax.set_title("How many FPS per each Watt in the rail")
    ax.legend()

    maxy = max(mem, npu, cpu, apu)*1.1
    ax.set_ylim(0, maxy)

    # Add numbers on top of the bars
    for bar in stacked_bars + separate_bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
        )

    # plt.show()
    save_filepath = recordfilename.replace("_meas", "efficency").rsplit(".", 1)[0] + ".png"
    plt.savefig(save_filepath, dpi=300, bbox_inches='tight')

def plotefficency_hwinfo(npu, cpu, apu, recordfilename):
    stacked_data = [npu, cpu]
    separate_data = [apu]

    categories = ["NPU", "CPU"]
    separate_category = ["APU"]
    bar_width = 0.35
    # Create x-axis values for the bars
    x_stacked = np.arange(len(categories))
    x_separate = np.arange(len(categories), len(categories) + len(separate_category))

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the stacked bars
    stacked_bars = ax.bar(x_stacked, stacked_data, bar_width, label="NPU, CPU")

    # Create the separate bar
    separate_bars = ax.bar(x_separate, separate_data, bar_width, label="APU efficency")

    # Set x-axis labels and tick positions
    ax.set_xticks(np.concatenate((x_stacked, x_separate)))
    ax.set_xticklabels(categories + separate_category)

    # Add labels and a legend
    ax.set_xlabel("Rails")
    ax.set_ylabel("Efficency [fps/W]")
    ax.set_title("How many FPS per each Watt in the rail")
    ax.legend()

    maxy = max(npu, cpu, apu)*1.1
    ax.set_ylim(0, maxy)

    # Add numbers on top of the bars
    for bar in stacked_bars + separate_bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
        )

    # plt.show()
    save_filepath = recordfilename.replace("_meas", "efficency").rsplit(".", 1)[0] + ".png"
    plt.savefig(save_filepath, dpi=300, bbox_inches='tight')

def plotenergy(mem, npu, cpu, apu, recordfilename):
    stacked_data = [mem, npu, cpu]
    separate_data = [apu]

    categories = ["MEM", "NPU", "CPU"]
    separate_category = ["APU"]
    bar_width = 0.35
    # Create x-axis values for the bars
    x_stacked = np.arange(len(categories))
    x_separate = np.arange(len(categories), len(categories) + len(separate_category))

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the stacked bars
    stacked_bars = ax.bar(x_stacked, stacked_data, bar_width, label="CPU, NPU, MEM")

    # Create the separate bar
    separate_bars = ax.bar(
        x_separate, separate_data, bar_width, label="APU overall energy"
    )

    # Set x-axis labels and tick positions
    ax.set_xticks(np.concatenate((x_stacked, x_separate)))
    ax.set_xticklabels(categories + separate_category)

    # Add labels and a legend
    ax.set_xlabel("Rails")
    ax.set_ylabel("Energy [mJ]")
    ax.set_title(" Energy to process one frame")
    ax.legend()

    maxy = max(mem, npu, cpu, apu)*1.1
    ax.set_ylim(0, maxy)

    # Add numbers on top of the bars
    for bar in stacked_bars + separate_bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
        )

    #plt.show()
    save_filepath = recordfilename.replace("_meas", "energy").rsplit(".", 1)[0] + ".png"
    plt.savefig(save_filepath, dpi=300, bbox_inches='tight')

def plotenergy_hwinfo(npu, cpu, apu, recordfilename):
    stacked_data = [npu, cpu]
    separate_data = [apu]

    categories = ["NPU", "CPU"]
    separate_category = ["APU"]
    bar_width = 0.35
    # Create x-axis values for the bars
    x_stacked = np.arange(len(categories))
    x_separate = np.arange(len(categories), len(categories) + len(separate_category))

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the stacked bars
    stacked_bars = ax.bar(x_stacked, stacked_data, bar_width, label="NPU, CPU")

    # Create the separate bar
    separate_bars = ax.bar(
        x_separate, separate_data, bar_width, label="APU overall energy"
    )

    # Set x-axis labels and tick positions
    ax.set_xticks(np.concatenate((x_stacked, x_separate)))
    ax.set_xticklabels(categories + separate_category)

    # Add labels and a legend
    ax.set_xlabel("Rails")
    ax.set_ylabel("Energy [mJ]")
    ax.set_title(" Energy to process one frame")
    ax.legend()

    maxy = max(npu, cpu, apu)*1.1
    ax.set_ylim(0, maxy)

    # Add numbers on top of the bars
    for bar in stacked_bars + separate_bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
        )

    #plt.show()
    save_filepath = recordfilename.replace("_meas", "energy").rsplit(".", 1)[0] + ".png"
    plt.savefig(save_filepath, dpi=300, bbox_inches='tight')


def profile_model(modelpath, model_input):
    #credits: https://github.com/ThanatosShinji/onnx-tool/blob/main/benchmark/examples.py
    # function removed for now
    quit()


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

def show_help():
    print("Usage: python performance_benchmark.py <parameters list>")
    print("Please fill the parameters list. Use -h for help")
    print("i.e.:")
    print("python performance_benchmark.py --model resnet50_1_5_224x224-op13_NHWC_quantized.onnx --execution_provider VitisAIEP --num 2000 -i 4 -t 4 -p 1 -j demo.json -r 1")

def save_result_json(results, filename):
    if filename=="use timestamp":
        timestamp = time.strftime("%Y%m%d%H%M%S")
        filename = "results_" + timestamp + ".json"
        
    with open(filename, "w") as file_json:
        json.dump(results, file_json, indent=4)
    ggprint(f"Data saved in {filename}")


def SetCalibDir(source_directory, calib_dir, num_images_to_copy):
    if os.path.exists(calib_dir):
        shutil.rmtree(calib_dir)
    os.makedirs(calib_dir)

    image_files = []
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_files.append(os.path.join(root, file))

    random.shuffle(image_files)

    copied_images = 0

    for file in image_files:
        if copied_images >= num_images_to_copy:
            break
        #destination_path = os.path.join(calib_dir, file)
        # check if the image has three channels
        image = Image.open(file)
        image_mode = image.mode
        if image_mode == 'RGB':
            shutil.copy(file, calib_dir)
            copied_images += 1
    return copied_images

def str_to_sec(date_string):
    date_string = date_string.strip("--[ ]").strip()
    #
    # date_format = "%d.%m.%Y %H:%M:%S.%f"
    time_str = date_string.split(" ")[1]
    seconds = time_to_seconds(time_str)
    return seconds



def tableprint(title, tabledata, tablew):
    print("-" * tablew)
    print(title.center(tablew))
    colwidth=tablew//2
    for row in tabledata:
        print(f"{{:<{colwidth}}} {{:>{colwidth}}}".format(*row))

def terminate_process(process_name):
    # Iterate over all running processes
    for proc in psutil.process_iter(['pid', 'name']):
        try:
        # Check if the process name matches
            if proc.info['name'] == process_name:
                proc.terminate()  # Terminate the process
                proc.wait()  # Wait for the process to be terminated
                ggprint(f"Process '{process_name}' (PID: {proc.pid}) has been terminated.")
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    ggprint(f"Process '{process_name}' is not running.")
    return False

def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(":"))
    return h * 3600 + m * 60 + s

def time_violin(filename, relevant_cols, title, xlabel, ylabel):
    # ----------------------------------------
    # row 1 = Time
    # row 2 = Violin histogram
    df = pd.read_csv(filename)
    fig, axs = plt.subplots(2, 1, figsize=(16, 6))

    medi = getmedians(filename)

    for meas in relevant_cols:
        axs[0].plot(df["Time"], df[meas], label=meas)
    
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel(ylabel)
    axs[0].set_title(title)
    axs[0].legend()
    axs[0].set_ylim(0)

    data = [df[meas] for meas in relevant_cols]

    axs[1].violinplot(data, showmedians=True)
    axs[1].set_title(f"Histogram")
    axs[1].set_xticks(range(1, len(relevant_cols) + 1))
    axs[1].tick_params(axis="x", labelrotation=45)

    axs[1].set_xticklabels(relevant_cols)
    axs[1].set_ylabel(ylabel)
    axs[1].set_xlabel(xlabel)

    axs[1].set_ylim(0)

    for i, value in enumerate(relevant_cols):
        median_value = medi[value]
        axs[1].text(
            i + 1,
            median_value,
            f"{median_value:.2f} {ylabel}",
            ha="center",
            va="bottom",
            color="black",
            fontsize=10,
        )

    fig.set_size_inches(10, 5)
    plt.tight_layout()
    #plt.show()
    prefix = title + "_violin"
    save_filepath = filename.replace("_meas", prefix).rsplit(".", 1)[0] + ".png"
    plt.savefig(save_filepath, dpi=300, bbox_inches='tight')

def tops_peak(device):
    gp = {"strix_50tops": 50, "strix_55tops": 55, "phoenix": 8, "hawk": 14.4}
    return gp[device]

def _preprocess_images(images_folder: str,
                       height: int,
                       width: int,
                       size_limit=0,
                       batch_size=100,
                       order="NHWC"):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    images = os.listdir(images_folder)
    image_names = []
    for image in images:
        image_names.append(os.path.join(images_folder, image))
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    
    unconcatenated_batch_data = []

    batch_data = []
    for index, image_name in enumerate(batch_filenames):
        pillow_img = Image.new("RGB", (width, height))

        pillow_img.paste(Image.open(image_name).resize((width, height)))  
        
        image_array = np.array(pillow_img) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        image_array = (image_array - mean)
        std = np.array([0.229, 0.224, 0.225])
        nchw_data = image_array / std
        
        shape = np.array(nchw_data).shape

        if order == "NCHW":
            nchw_data = nchw_data.transpose((2, 0, 1))
        
        nchw_data = np.expand_dims(nchw_data, axis=0)
        nchw_data = nchw_data.astype(np.float32)
        unconcatenated_batch_data.append(nchw_data)


        if (index + 1) % batch_size == 0:
            one_batch_data = np.concatenate(unconcatenated_batch_data,
                                               axis=0)
            unconcatenated_batch_data.clear()
            batch_data.append(one_batch_data)

    return batch_data

class ImageDataReader(CalibrationDataReader):

    def __init__(self, calibration_image_folder: str, model_path: str, data_size: int, batch_size: int, order):
        self.enum_data = None

        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_mem_pattern = False
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.add_session_config_entry("session.memory_arena_extend_strategy", "kSameAsRequested")

        session = onnxruntime.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])


        # Use inference session to get input shape.
        #session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        #session = onnxruntime.InferenceSession(model_path, providers=["DmlExecutionProvider"])

        if order == "NHWC":
            (_,height, width, _) = session.get_inputs()[0].shape
        else:
            (_, _, height, width) = session.get_inputs()[0].shape
        
        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(calibration_image_folder, height, width, data_size, batch_size, order)
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)
        self.order = order


    #def get_next(self):
    #    if self.enum_data is None:
    #        self.enum_data = iter([{
    #            self.input_name: nhwc_data
    #        } for nhwc_data in self.nhwc_data_list])
    #    return next(self.enum_data, None)
    #Replaced with more robust:
    def get_next(self) -> Dict[str, Any]:
        """
        Returns the next input batch as a dict,
        or {} when there is no more data.
        """
        if self.enum_data is None:
            # build your list of dicts only once
            self.enum_data = iter(
                [{self.input_name: batch} 
                 for batch in self.nhwc_data_list]
            )
        
        try:
            return next(self.enum_data)
        except StopIteration:
            return {}   # always a dict, so type matches the base class

    def rewind(self):
        self.enum_data = None


    def reset(self):
        self.enum_data = None

def get_apu_info():
    # Run pnputil as a subprocess to enumerate PCI devices
    command = r'pnputil /enum-devices /bus PCI /deviceids '
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # Check for supported Hardware IDs
    apu_type = ''
    if 'PCI\\VEN_1022&DEV_1502&REV_00' in stdout.decode(): apu_type = 'PHX/HPT'
    if 'PCI\\VEN_1022&DEV_17F0&REV_00' in stdout.decode(): apu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_10' in stdout.decode(): apu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_11' in stdout.decode(): apu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_20' in stdout.decode(): apu_type = 'KRK'
    return apu_type

def set_environment_variable(apu_type):
    install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
    os.environ['XLNX_ENABLE_CACHE']='1'
    os.environ['XLNX_ONNX_EP_REPORT_FILE']='vitisai_ep_report.json'

    match apu_type:
        case 'PHX/HPT':
            ggprint("Setting environment for PHX/HPT")
            os.environ['XLNX_VART_FIRMWARE']= os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'phoenix', '4x4.xclbin')
            os.environ['NUM_OF_DPU_RUNNERS']='1'
            os.environ['XLNX_TARGET_NAME']='AMD_AIE2_4x4_Overlay'
        case ('STX' | 'KRK'):
            ggprint("Setting environment for STX")
            os.environ['XLNX_VART_FIRMWARE']= os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'strix', 'AMD_AIE2P_4x4_Overlay.xclbin')
            os.environ['NUM_OF_DPU_RUNNERS']='1'
            os.environ['XLNX_TARGET_NAME']='AMD_AIE2P_4x4_Overlay'
        case _:
            ggprint("Unrecognized APU type. Exiting.")
            sys.exit(1)

# Build a map of ONNX element names → NumPy dtypes,
# safely handling bfloat16 on NumPy <1.20
def _make_type_map():
    m = {
        'float':   np.float32,
        'double':  np.float64,
        'float16': np.float16,
        'int64':   np.int64,
        'int32':   np.int32,
        'int16':   np.int16,
        'int8':    np.int8,
        'uint8':   np.uint8,
        'bool':    np.bool_
    }
    try:
        m['bfloat16'] = np.dtype('bfloat16')
    except TypeError:
        # fallback when numpy doesn't support bfloat16
        m['bfloat16'] = np.float32
    return m

_TYPE_MAP = _make_type_map()




class SessionLike(Protocol):
    def get_inputs(self) -> Sequence[Any]: ...
    def get_outputs(self) -> Sequence[Any]: ...
    def run(self, output_names: Sequence[str], input_feed: Mapping[str, Any]) -> Sequence[Any]: ...


class DummySession:
    def get_inputs(self) -> Sequence[Any]:
        return []

    def get_outputs(self) -> Sequence[Any]:
        return []

    def run(
        self,
        output_names: Sequence[str],
        input_feed: Mapping[str, Any],
    ) -> Sequence[Any]:
        # return an empty sequence to signal “no outputs”
        return []

def generate_dummy_inputs(session, orig_im_size=(480, 640)):
    """
    For each ONNX input, make a dummy NumPy array
    (dynamic dims → 1) with matching dtype.
    Special-cases 'orig_im_size' to use the provided tuple.
    """
    dummy_inputs = {}
    print("-" * 80)
    print("Model inputs inspection:")
    
    for inp in session.get_inputs():
        name = inp.name

        # Special-case orig_im_size
        if name == "orig_im_size":
            array = np.array(orig_im_size, dtype=np.int64)
            dummy_inputs[name] = array
            shape_str = str(list(array.shape))
            print(f"{name:<30}  {shape_str:<15}  {array.dtype}")
            continue

        # Build shape, replacing dynamic dims (None or str) with 1
        shape = []
        for d in inp.shape:
            if isinstance(d, int):
                shape.append(d)
            else:
                # dynamic dimension: replace with 1
                shape.append(1)

        # Determine numpy dtype
        elem = inp.type
        elem = elem[elem.find('(') + 1 : elem.find(')')]
        np_dtype = _TYPE_MAP.get(elem, np.float32)

        # Build the dummy array
        if np.issubdtype(np_dtype, np.floating):
            array = np.random.randn(*shape).astype(np_dtype)
        elif np.issubdtype(np_dtype, np.integer):
            array = np.random.randint(0, 10, size=shape, dtype=np_dtype)
        elif np_dtype == np.bool_:
            array = (np.random.rand(*shape) > 0.5)
        else:
            array = np.zeros(shape, dtype=np_dtype)

        dummy_inputs[name] = array
        print(f"{name:<30}  {str(shape):<15}  {np_dtype}")

    return dummy_inputs

# Example usage
# session = ort.InferenceSession("model_dynamic.onnx")
# inputs = generate_dummy_inputs(session, orig_im_size=(16, 16))

