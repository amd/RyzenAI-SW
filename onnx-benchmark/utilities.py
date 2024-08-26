# release 18 
# STRIX and PHOENIX supported

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
import inspect
import gc

from onnxruntime.quantization.calibrate import CalibrationDataReader
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
from onnx import version_converter, helper
from onnxruntime.quantization import shape_inference
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
        "p_instance_count",
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
                "p_instance_count": args.instance_count,
                "p_intra_op_num_threads": args.intra_op_num_threads,
                "p_json": args.json,
                "p_min_interval": args.min_interval,
                "p_model": args.model,
                "p_no_inference": args.no_inference,
                "p_num": args.num,
                "p_power": "NA",
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

def ask_update():
    while True:
        response = input("Do you want to update? (y/n): ").strip().lower()
        if response in {'y', 'n'}:
            return response
        else:
           print("Invalid input. Please enter 'y' or 'n'.")

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
    if sys.version_info.major == 3 and sys.version_info.minor == 10:
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
                Colors.RED + "please update python" + Colors.RESET,
            )
        )

    package_name = "onnxruntime"
    version = check_package_version(package_name)
    if version in ["1.17.0", "1.17.3"]:
        data.append(
            (f"{package_name} version: {version}", Colors.GREEN + "OK" + Colors.RESET)
        )
    else:
        data.append(
            (f"{package_name} version: {version}", Colors.RED + f"please update {package_name}" + Colors.RESET)
        )

    package_name = "onnxruntime-vitisai"
    version = check_package_version(package_name)
    fields = version.split('.')
    if fields[0] == "1" and fields[1] == "17":
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
    if fields[0] == "1" and fields[1] == "2":
        data.append(
            (f"{package_name} version: {version}", Colors.GREEN + "OK" + Colors.RESET)
        )
    else:
        data.append(
            (f"{package_name} version: {version}", Colors.RED + f"please update {package_name}" + Colors.RESET)
        )

    max_width = max(len(row[0]) for row in data)

    for row in data:
        column1, column2 = row
        # Left-align the text in the first column and pad with spaces
        formatted_column1 = column1.ljust(max_width)
        ggprint(f"{formatted_column1} {column2}")
    

def check_args(args, defaults):
    assert args.num >= (
        args.batchsize * args.instance_count
    ), "runs must be greater than batches*instance-count"

    total_cpu = os.cpu_count()
    if args.instance_count > total_cpu:
        args.instance_count = total_cpu
        ggprint(f"Limiting instance count to max cpu count ({total_cpu})")

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
    caller_frame = frame.f_back
    file_name = caller_frame.f_code.co_filename
    line_number = caller_frame.f_lineno
    print(Colors.BLUE + f"File: {file_name}, Line: {line_number} - {message}" + Colors.RESET)


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


def del_old_meas(measfile):
    # Check if the file exists before attempting to delete it
    if os.path.exists(measfile):
        os.remove(measfile)
        #ggprint(f"Old measurement {measfile} deleted successfully")


def detect_device():
    def get_driver_info(device_name):
        # PowerShell command to get device details
        command = f'powershell -Command "Get-WmiObject Win32_PnPEntity | Where-Object {{ $_.Name -like \'*{device_name}*\' }} | Select-Object Name, DeviceID"'
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        device_info = result.stdout.strip()
    
        return device_info

    device_name = "NPU Compute Accelerator Device"
    device_info = get_driver_info(device_name)
    #print("Device Info:\n", device_info)

    if "17F0" in device_info:
        device = "STRIX"
    elif "1502" in device_info:
        device = "PHOENIX"
    else: device = "ERROR Device Unknown"

    #print(f'Device = {device}')
    return device   

def get_driver_release_number(device_name):
    command = f'powershell -Command "Get-WmiObject Win32_PnPSignedDriver | Where-Object {{ $_.DeviceName -like \'*{device_name}*\' }} | Select-Object DeviceName, DriverVersion"'
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        print("Error running PowerShell command:")
        print(result.stderr)
        return
    return result.stdout.split("\n")[3].split(" ")[4].strip()


def ggprint(linea):
    print(Colors.DIMMERED_WHITE + linea + Colors.RESET)


def ggquantize(args):
    input_model_path = args.model
    imagenet_directory = args.calib

    base_name, extension = os.path.splitext(input_model_path)

    # OPSET update to 17
    # A full list of supported adapters can be found here:
    # https://github.com/onnx/onnx/blob/main/onnx/version_converter.py#L21
    # Apply the version conversion on the original model

    # Preprocessing: load the model to be converted.   
    newopset = 17
    original_model = onnx.load(input_model_path)
    opset_version = original_model.opset_import[0].version

    # temporary fix: after conversion the opset is not updated to 17 but to 1
    if opset_version<11:
        if opset_version==1 :
            ggprint(f'[WARNING] The model opset version is 1, which seems unlikely. This might be the result of a previous opset conversion.')
        else:
            ggprint(f'The model OPSET is {opset_version} and should be updated to {newopset}')
            user_response = ask_update()
            if user_response == 'y':
                output_updated = f"{base_name}_opset17{extension}"
                converted_model = version_converter.convert_version(original_model, newopset)
                #print(f"The model after conversion:\n{converted_model}")
                onnx.save(converted_model, output_updated)
                print(f'The update model was saved with name {output_updated}')
                # verify opset
                updated_model = onnx.load(output_updated)
                ggprint(f"Updated model opset version: {updated_model.opset_import[0].version}")
                input_model_path = output_updated
            else:
                ggprint("You chose not to update")
    # free memory
    del original_model
    gc.collect()

    # check if the model is already quantized
    output_INT8_NHWC_model_path = f"{base_name}_int8{extension}"
    calibration_dataset_path = "calibration"

    operators = list_operators(input_model_path)
    if "QuantizeLinear" in operators:
        ggprint("The model is already quantized")
        return input_model_path
    else:
        if args.renew == "1":
            cache_dir = os.path.join(Path(__file__).parent.resolve(), "cache", os.path.basename(output_INT8_NHWC_model_path))
            cancelcache(cache_dir)

        print(Colors.MAGENTA)
        print("The model is not quantized")
        # 2) recognize the model input format
        input_name, input_shape = get_input_info(input_model_path)
        print(f"Model Input Name: {input_name}, Model Input Shape: {input_shape}")

        order = analyze_input_format(input_shape)
        if order == "NHWC":
            nchw_to_nhwc = False
            print("The input format is already NHWC - this is the optimal shape")
        elif order =="NCHW":
            nchw_to_nhwc = True
            print("The input format is NCHW - conversion to NHWC enabled")
       
        # 3) prepare the calibration directory
        calib_dir = "calibration"
        copied_images = SetCalibDir(imagenet_directory, calib_dir, args.num_calib)
        print(f"Successfully copied {copied_images} RGB images to {calib_dir}")

        # Data Reader is a utility class that reads the calibration dataset and prepares it for the quantization process.
        data_reader = DataReader( calibration_folder=calibration_dataset_path, batch_size=1, target_size=input_shape, inputname=input_name)


        if args.execution_provider == "VitisAIEP":
            
            base_name, extension = os.path.splitext(input_model_path)
            preprocessed_model_path = f"{base_name}_pp{extension}"
            shape_inference.quant_pre_process(
                input_model_path,
                preprocessed_model_path,
                skip_optimization=False,
                skip_onnx_shape= False,
                skip_symbolic_shape= False,
                auto_merge= False,
                int_max= 2**31 - 1,
                guess_output_rank= False,
                verbose= 3,
                save_as_external_data= False,
                all_tensors_to_one_file= False,
                external_data_location= "./",
                external_data_size_threshold= 1024
            )

            vai_q_onnx.quantize_static(
                preprocessed_model_path,
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
                    'RemoveQDQConvLeakyRelu':True,
                    'RemoveQDQConvPRelu':True
                }
            )
        
        elif args.execution_provider == "CPU":
            vai_q_onnx.quantize_static(
                input_model_path,
                output_INT8_NHWC_model_path,
                data_reader,
                activation_type=QuantType.QUInt8,
                calibrate_method=vai_q_onnx.CalibrationMethod.Percentile,
                include_cle=True,
                extra_options={
                    'ReplaceClip6Relu':True,
                    'CLESteps':4,
                }
            )

        print('Calibrated and quantized NHWC model saved at:', output_INT8_NHWC_model_path)
        print(Colors.RESET)
        return output_INT8_NHWC_model_path


def get_input_format(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    input_shapes = []
    for input_node in onnx_model.graph.input:
        shape = [dim.dim_value or dim.dim_param for dim in input_node.type.tensor_type.shape.dim]
        input_shapes.append(tuple(shape))
    return [input_node.name, input_shapes]


def get_input_info(onnx_model_path):
    input_info = {}
    # Load the ONNX model without loading it into memory
    with open(onnx_model_path, "rb") as f:
        model_proto = onnx.load(f)

        # Get the first input node in the graph
        first_input = model_proto.graph.input[0]
        input_name = first_input.name
        input_shape = [d.dim_value for d in first_input.type.tensor_type.shape.dim]
    return input_name, input_shape


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

    measurement["vitisai"] = {}
    measurement["vitisai"]["all"] = 0
    measurement["vitisai"]["CPU"] = 0
    measurement["vitisai"]["DPU"] = 0

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
    if args.execution_provider == "VitisAIEP":
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

    if args.execution_provider == "VitisAIEP":
        measurement["environment"]["vaip_config"] = {}
        with open(args.config, "r") as json_file:
            vaip_conf = json.load(json_file)

        measurement["environment"]["vaip_config"] = vaip_conf

    return measurement


def parse_args(device):

    def_config_path = os.path.join(os.environ.get('VAIP_CONFIG_HOME'), 'vaip_config.json')
    defaults = {
        'calib': ".\\images",
        'config': def_config_path,
        'core': "STX_1x4",
        'execution_provider': 'CPU',
    }

    if len(sys.argv) < 2:
        show_help()
        quit()

    parser = argparse.ArgumentParser()

    parser.add_argument("--batchsize", "-b", type=int, default=1, help="batch size: number of images processed at the same time by the model. VitisAIEP supports batchsize = 1. Default is 1")
    parser.add_argument(
        "--calib",
        type=str,
        default=defaults['calib'],
        help=f"path to Imagenet database, used for quantization with calibration. Default= .\Imagenet\val ",
    )   
    parser.add_argument(
        "--config", 
        "-c", 
        type=str, 
        default=defaults['config'],
        help="path to config json file. Default= <release>\\vaip_config.json",
    )

    if device=="STRIX":
        parser.add_argument(
            "--core",
            default="STX_1x4",
            type=str,
            choices=["STX_1x4","STX_4x4"],
            help="Which core to use with STRIX silicon. Default=STX_1x4",
        )
    elif device=="PHOENIX":
        parser.add_argument(
            "--core",
            default="PHX_1x4",
            type=str,
            choices=["PHX_1x4","PHX_4x4"],
            help="Which core to use with PHOENIX silicon. Default=PHX_1x4",
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
    parser.add_argument(
        "--instance_count",
        "-i",
        type=int,
        default=1,
        help="This parameter governs the parallelism of job execution. When the Vitis AI EP is selected, this parameter controls the number of DPU runners. The workload is always equally divided per each instance count. Default=1",
    )
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
        "--renew",
        "-r",
        type=str,
        default="1",
        choices=["0", "1"],
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

def PHX_1x4_setup(silicon):
    xclbin_path = os.path.join(os.environ.get('XCLBINHOME'), '1x4.xclbin')
    os.environ['XLNX_VART_FIRMWARE'] = str(xclbin_path)       
    os.environ["XLNX_TARGET_NAME"] = "AMD_AIE2_Nx4_Overlay"
    ggprint(80*"-")
    ggprint("Core selection")
    ggprint(f"Path to xclbin_path = {xclbin_path}")
    ggprint(os.environ["XLNX_TARGET_NAME"])


def PHX_4x4_setup(silicon):
    xclbin_path = os.path.join(os.environ.get('XCLBINHOME'), '4x4.xclbin')
    os.environ['XLNX_VART_FIRMWARE'] = str(xclbin_path)       
    os.environ["XLNX_TARGET_NAME"] = "AMD_AIE2_4x4_Overlay"
    ggprint(80*"-")
    ggprint("Core selection")
    ggprint(f"Path to xclbin_path = {xclbin_path}")
    ggprint(os.environ["XLNX_TARGET_NAME"])


def STX_1x4_setup(silicon):
    xclbin_path = os.path.join(os.environ.get('XCLBINHOME'), 'AMD_AIE2P_Nx4_Overlay.xclbin')
    os.environ['XLNX_VART_FIRMWARE'] = str(xclbin_path)       
    os.environ["XLNX_TARGET_NAME"] = "AMD_AIE2P_Nx4_Overlay"
    ggprint(80*"-")
    ggprint("Core selection")
    ggprint(f"Path to xclbin_path = {xclbin_path}")
    ggprint(os.environ["XLNX_TARGET_NAME"])


def STX_4x4_setup(silicon):
    #xclbin_path = os.path.join(os.environ.get('XCLBINHOME'), 'AMD_AIE2P_4x4_Overlay_CFG0.xclbin')
    xclbin_path = os.path.join(os.environ.get('XCLBINHOME'), 'AMD_AIE2P_4x4_Overlay.xclbin')
    os.environ['XLNX_VART_FIRMWARE'] = str(xclbin_path)       
    os.environ["XLNX_TARGET_NAME"] = "AMD_AIE2P_4x4_Overlay"
    ggprint(80*"-")
    ggprint("Core selection")
    ggprint(f"Path to xclbin_path = {xclbin_path}")
    ggprint(os.environ["XLNX_TARGET_NAME"])


def set_engine_shape(case):
    switch_dict = {
        "PHX_1x4": PHX_1x4_setup,
        "PHX_4x4": PHX_4x4_setup,
        "STX_1x4": STX_1x4_setup,
        "STX_4x4": STX_4x4_setup,
    }
    action = switch_dict.get(case, DEF_setup)
    action("none")


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
