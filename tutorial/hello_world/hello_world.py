import torch
import torch.nn as nn
import os
import subprocess
import onnxruntime
import numpy as np
import onnx
import shutil
from timeit import default_timer as timer

torch.manual_seed(0)

# Create a simple model
class SmallModel(nn.Module):
    def __init__(self):
        super(SmallModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = torch.add(x, 1)

        return x

# Instantiate the model
pytorch_model = SmallModel()
pytorch_model.eval()

# Print the model architecture
print(pytorch_model)

# Generate dummy input data
batch_size = 1
input_channels = 3
input_size = 224
dummy_input = torch.rand(batch_size, input_channels, input_size, input_size)

# Prep for ONNX export
inputs = {"x": dummy_input}
dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
tmp_model_path = "models/helloworld.onnx"

# Call export function
torch.onnx.export(
        pytorch_model,
        inputs,
        tmp_model_path,
        export_params=True,
        opset_version=17,  # Recommended opset
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )

# Quantize Model
from quark.onnx.quantization.config import Config, get_default_config
from quark.onnx import ModelQuantizer

# `input_model_path` is the path to the original, unquantized ONNX model.
input_model_path = "models/helloworld.onnx"

# `output_model_path` is the path where the quantized model will be saved.
output_model_path = "models/helloworld_quantized.onnx"

# Use default quantization configuration
quant_config = get_default_config("XINT8")
quant_config.extra_options["UseRandomData"] = True
# Defines the quantization configuration for the whole model
config = Config(global_quant_config=quant_config)
print("The configuration of the quantization is {}".format(config))

# Create an ONNX Quantizer
quantizer = ModelQuantizer(config)

# Quantize the ONNX model
quant_model = quantizer.quantize_model(model_input = input_model_path,
                                       model_output = output_model_path,
                                       calibration_data_path = None)

print('Calibrated and quantized model saved at:', output_model_path)

# Run Model on CPU Run

# Specify the path to the quantized ONNZ Model
quantized_model_path = r'./models/helloworld_quantized.onnx'
model = onnx.load(quantized_model_path)

# Create some random input data for testing
input_data = np.random.uniform(low=-1, high=1, size=(batch_size, input_channels, input_size, input_size)).astype(np.float32)

cpu_options = onnxruntime.SessionOptions()

# Create Inference Session to run the quantized model on the CPU
cpu_session = onnxruntime.InferenceSession(
    model.SerializeToString(),
    providers = ['CPUExecutionProvider'],
    sess_options=cpu_options,
)

# Run Inference
start = timer()
cpu_results = cpu_session.run(None, {'input': input_data})
cpu_total = timer() - start

# Run Model on NPU

# Before running, we need to set the ENV variable for the specific NPU we have
# Run pnputil as a subprocess to enumerate PCI devices
command = r'pnputil /enum-devices /bus PCI /deviceids '
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
# Check for supported Hardware IDs
npu_type = ''
if 'PCI\\VEN_1022&DEV_1502&REV_00' in stdout.decode(): npu_type = 'PHX/HPT'
if 'PCI\\VEN_1022&DEV_17F0&REV_00' in stdout.decode(): npu_type = 'STX'
if 'PCI\\VEN_1022&DEV_17F0&REV_10' in stdout.decode(): npu_type = 'STX'
if 'PCI\\VEN_1022&DEV_17F0&REV_11' in stdout.decode(): npu_type = 'STX'

print(f"APU Type: {npu_type}")

install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
xclbin_file = ''
match npu_type:
    case 'PHX/HPT':
        print("Setting xclbin file for PHX/HPT")
        xclbin_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'phoenix', '4x4.xclbin')
    case 'STX':
        print("Setting xclbin file for STX")
        xclbin_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'strix', 'AMD_AIE2P_4x4_Overlay.xclbin')
    case _:
        print("Unrecognized APU type. Exiting.")
        exit()

# We want to make sure we compile everytime, otherwise the tools will use the cached version
# Get the current working directory
current_directory = os.getcwd()
directory_path = os.path.join(current_directory,  r'cache\hello_cache')
cache_directory = os.path.join(current_directory,  r'cache')

# Check if the directory exists and delete it if it does.
if os.path.exists(directory_path):
    shutil.rmtree(directory_path)
    print(f"Directory '{directory_path}' deleted successfully.")
else:
    print(f"Directory '{directory_path}' does not exist.")


# Compile and run

# Point to the config file path used for the VitisAI Execution Provider
install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
config_file_path = os.path.join(install_dir, 'voe-4.0-win_amd64', 'vaip_config.json') # Path to the NPU config file

aie_options = onnxruntime.SessionOptions()

aie_session = onnxruntime.InferenceSession(
    model.SerializeToString(),
    providers=['VitisAIExecutionProvider'],
    sess_options=aie_options,
    provider_options = [{'config_file': config_file_path,
                         'cacheDir': cache_directory,
                         'cacheKey': 'hello_cache',
                         'xclbin': xclbin_file
                        }]
)

# Run Inference
start = timer()
npu_results = aie_session.run(None, {'input': input_data})
npu_total = timer() - start


print(f"CPU Execution Time: {cpu_total}")
print(f"NPU Execution Time: {npu_total}")
