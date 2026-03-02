# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Import necessary libraries
import os
import torch
import torch.nn as nn
import torchvision
import subprocess
import onnxruntime
import numpy as np
import onnx
import shutil
import time
from timeit import default_timer as timer
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config
from utils_custom import ImageDataReader, evaluate_onnx_model
import json
from classification_utils import calib_data_formatting

# ---------------- Model Setup ---------------- #

# Define directories
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Load pre-trained ResNet50 model
model = torchvision.models.resnet50(weights="IMAGENET1K_V2")

# Save the model
model.to("cpu")
torch.save(model, os.path.join(models_dir, "resnet50.pt"))

# Export model to ONNX
dummy_inputs = torch.randn(1, 3, 224, 224)
input_names = ['input']
output_names = ['output']
dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
tmp_model_path = os.path.join(models_dir, "resnet50.onnx")

torch.onnx.export(
    model,
    dummy_inputs,
    tmp_model_path,
    export_params=True,
    opset_version=17,  # Recommended opset
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
)

print(f" Model exported to ONNX at: {tmp_model_path}")

# ---------------- Quark Quantization ---------------- #

# Define dataset directory
calib_dir = "calib_data"
# fomat val_images and store it in calib_data for calibeeration.
os.makedirs(calib_dir, exist_ok=True)
calib_data_formatting()

# Set input & output ONNX model paths
input_model_path = tmp_model_path
output_model_path = os.path.join(models_dir, "resnet50_quantized.onnx")

# Preprocessing transformations
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
calib_dataset = torchvision.datasets.ImageFolder(root=calib_dir, transform=preprocess)

#Data set
num_calib_data = 600
calib_dataset = torch.utils.data.Subset(calib_dataset, range(num_calib_data))

# Define DataLoader for Calibration
calibration_dataloader = torch.utils.data.DataLoader(calib_dataset, batch_size=10, shuffle=False)

# Configure Quark Quantization
quant_config = get_default_config("XINT8")  # Use XINT8 quantization
config = Config(global_quant_config=quant_config)

# Create an ONNX Quantizer
quantizer = ModelQuantizer(config)

# Perform Quark Quantization
quant_model = quantizer.quantize_model(
    model_input=input_model_path,
    model_output=output_model_path,
    calibration_data_reader=ImageDataReader(calibration_dataloader)  # Use ImageDataReader from utils_custom
)

print(f" Quark Quantized model saved at: {output_model_path}")


# ---------------- Inference & Evaluation ---------------- #

from PIL import Image

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

def preprocess_image(input):
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),
        normalize,
    ])
    img_tensor = transform(input).unsqueeze(0)
    return img_tensor.numpy()

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()

labels = load_labels('data/imagenet-simple-labels.json')
image = Image.open('data/dog.jpg')

print("Image size: ", image.size)
input_data = preprocess_image(image)

# Run inference on CPU
onnx_model_path = output_model_path
cpu_options = onnxruntime.SessionOptions()

cpu_session = onnxruntime.InferenceSession(
    onnx_model_path,
    providers=['CPUExecutionProvider'],
    sess_options=cpu_options,
)

start = timer()
cpu_outputs = cpu_session.run(None, {'input': input_data})
end = timer()

cpu_results = postprocess(cpu_outputs)
inference_time = np.round((end - start) * 1000, 2)
idx = np.argmax(cpu_results)

print('----------------------------------------')
print(f'Final top prediction is: {labels[idx]}')
print('----------------------------------------')
print(f'Inference time: {inference_time} ms')
print('----------------------------------------')

sort_idx = np.flip(np.squeeze(np.argsort(cpu_results)))
print('------------ Top 5 labels are: ----------------------------')
print(labels[sort_idx[:5]])
print('-----------------------------------------------------------')

#iGPU inference
dml_options = onnxruntime.SessionOptions()

# Create Inference Session to run the quantized model on the iGPU
dml_session = onnxruntime.InferenceSession(
    onnx_model_path,
    providers = ['DmlExecutionProvider'],
    provider_options = [{"device_id": "0"}]
)
start = time.time()
dml_outputs = dml_session.run(None, {'input': input_data})
end = time.time()

dml_results = postprocess(dml_outputs)
inference_time = np.round((end - start) * 1000, 2)
idx = np.argmax(dml_results)

print('----------------------------------------')
print('Final top prediction is: ' + labels[idx])
print('----------------------------------------')

print('----------------------------------------')
print('Inference time: ' + str(inference_time) + " ms")
print('----------------------------------------')

sort_idx = np.flip(np.squeeze(np.argsort(dml_results)))
print('------------ Top 5 labels are: ----------------------------')
print(labels[sort_idx[:5]])
print('-----------------------------------------------------------')

#NPU inference

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

print(f"NPU Type: {npu_type}")

install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
match npu_type:
    case 'PHX/HPT':
        print("Setting provider options for PHX/HPT")
        xclbin_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'phoenix', '4x4.xclbin')
        provider_options = [{
              'target': 'X1',
              'xclbin': xclbin_file,
              'ai_analyzer_visualization': True,
              'ai_analyzer_profiling': True,
        }]
    case 'STX':
        print("Setting provider options for STX")
        xclbin_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'strix', 'AMD_AIE2P_4x4_Overlay.xclbin')
        provider_options = [{
              'ai_analyzer_visualization': True,
              'ai_analyzer_profiling': True,
          }]
    case _:
        print("Unrecognized APU type. Exiting.")
        exit()

# Create session options
session_options = ort.SessionOptions()
session_options.log_severity_level = 1  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal

npu_session = onnxruntime.InferenceSession(
    onnx_model_path,
    sess_options = session_options,
    providers = ['VitisAIExecutionProvider'],
    provider_options = provider_options
)

start = time.time()
npu_outputs = npu_session.run(None, {'input': input_data})
end = time.time()

npu_results = postprocess(npu_outputs)
inference_time = np.round((end - start) * 1000, 2)
idx = np.argmax(npu_results)

print('----------------------------------------')
print('Final top prediction is: ' + labels[idx])
print('----------------------------------------')

print('----------------------------------------')
print('Inference time: ' + str(inference_time) + " ms")
print('----------------------------------------')

sort_idx = np.flip(np.squeeze(np.argsort(npu_results)))
print('------------ Top 5 labels are: ----------------------------')
print(labels[sort_idx[:5]])
print('-----------------------------------------------------------')
