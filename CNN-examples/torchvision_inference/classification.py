# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Import necessary libraries
import hashlib
import os
import sys
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

# classification.ipynb writes models/resnet50_<QUANT_TYPE>.onnx and compiles it into cache/.
# Loading that same file keeps the NPU cache key identical, so this script does not recompile.
output_model_path = None
for candidate in (
    os.path.join(models_dir, "resnet50_bf16.onnx"),
    os.path.join(models_dir, "resnet50_xint8.onnx"),
):
    if os.path.isfile(candidate):
        output_model_path = candidate
        print(f"Using ONNX model from classification.ipynb: {output_model_path}")
        break

if output_model_path is None:
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
    print("Dataset contains {} samples".format(len(calib_dataset)))

    # Data set
    num_calib_data = 100
    calib_dataset = torch.utils.data.Subset(calib_dataset, range(num_calib_data))

    # Define DataLoader for Calibration
    calibration_dataloader = torch.utils.data.DataLoader(calib_dataset, batch_size=1, shuffle=False)

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

#iGPU inference. DirectML is part of the Windows Ryzen AI build only.
if 'DmlExecutionProvider' not in onnxruntime.get_available_providers():
    print('DirectML (iGPU) is not available on this platform, skipping iGPU inference.')
else:
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

#NPU inference. Detection matches classification.ipynb: pnputil on Windows, lspci on Linux.
npu_type = ''
if sys.platform == 'win32':
    command = r'pnputil /enum-devices /bus PCI /deviceids '
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    out = stdout.decode(errors='ignore')
    if 'PCI\\VEN_1022&DEV_1502&REV_00' in out:
        npu_type = 'PHX/HPT'
    elif 'PCI\\VEN_1022&DEV_17F0&REV_20' in out:
        npu_type = 'KRK'
    elif any(f'PCI\\VEN_1022&DEV_17F0&REV_{rev}' in out for rev in ('00', '10', '11')):
        npu_type = 'STX'
elif sys.platform.startswith('linux'):
    try:
        out = subprocess.run(['lspci', '-nn'], capture_output=True, text=True).stdout
    except FileNotFoundError:
        print('lspci not found. Install it with: sudo apt install pciutils')
        sys.exit(1)
    if '1022:17f0' in out:
        npu_type = 'STX/KRK'
    elif '1022:1502' in out:
        print('PHX/HPT NPUs are not supported by Ryzen AI 1.7.1 on Linux (STX/KRK only).')
        sys.exit(1)

print(f"NPU Type: {npu_type}")
if npu_type not in ('PHX/HPT', 'STX', 'KRK', 'STX/KRK'):
    print("Unrecognized APU type. Exiting.")
    sys.exit(1)

install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
# Same cache_dir and cache_key as classification.ipynb, so the compiled model is reused.
cache_dir = os.path.join(os.getcwd(), 'cache')
with open(onnx_model_path, 'rb') as model_file:
    model_hash = hashlib.sha256(model_file.read() + onnxruntime.__version__.encode()).hexdigest()[:12]
cache_key = f"{os.path.splitext(os.path.basename(onnx_model_path))[0]}_{model_hash}"
print(f"NPU cache: {os.path.join(cache_dir, cache_key)}")
provider_options = [{
    'cache_dir': cache_dir,
    'cache_key': cache_key,
    'enable_cache_file_io_in_mem': 0,
    'ai_analyzer_visualization': True,
    'ai_analyzer_profiling': True,
}]
if npu_type == 'PHX/HPT':
    print("Setting provider options for PHX/HPT")
    provider_options[0]['target'] = 'X1'
    provider_options[0]['xclbin'] = os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'phoenix', '4x4.xclbin')
else:
    print(f"Setting provider options for {npu_type}")

# Create session options
session_options = onnxruntime.SessionOptions()
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
