# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Get model from torchvision
import os
import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms 

import onnx
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod, quantize_static

import numpy as np 
import json
import time
import subprocess

import vai_q_onnx
from classification_utils import get_directories


_, models_dir = get_directories()

## load model from torchvision
model = resnet50(weights="IMAGENET1K_V2")

## Save the model
model.to("cpu")
torch.save(model, str(models_dir / "resnet50.pt"))

# Export to ONNX
dummy_inputs = torch.randn(1, 3, 224, 224)
input_names = ['input']
output_names = ['output']
dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
tmp_model_path = str(models_dir / "resnet50.onnx")

## Call export function
torch.onnx.export(
        model,
        dummy_inputs,
        tmp_model_path,
        export_params=True,
        opset_version=13,  # Recommended opset
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

# Quantize the model

## Modify the datapath to imagenet folder
data_dir = "imagenet"

## `input_model_path` is the path to the original, unquantized ONNX model.
input_model_path = "models/resnet50.onnx"

## `output_model_path` is the path where the quantized model will be saved.
output_model_path = "models/resnet50_quantized.onnx"

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

calib_dir = os.path.join(data_dir, 'calib_data')
calib_dataset = torchvision.datasets.ImageFolder(root=calib_dir, transform=preprocess)

class ClassificationCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calib_dir: str, batch_size: int = 1):
        super().__init__()
        self.iterator = iter(DataLoader(calib_dir, batch_size))

    def get_next(self) -> dict:
        try:
            images, labels = next(self.iterator)
            return {"input": images.numpy()}
        except Exception:
            return None


def classification_calibration_reader(calib_dir, batch_size=1):
    return ClassificationCalibrationDataReader(calib_dir, batch_size=batch_size)

dr = classification_calibration_reader(calib_dataset)

vai_q_onnx.quantize_static(
    input_model_path,
    output_model_path,
    dr,
    quant_format=vai_q_onnx.QuantFormat.QDQ,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    activation_type=vai_q_onnx.QuantType.QUInt8,
    weight_type=vai_q_onnx.QuantType.QInt8,
    enable_ipu_cnn=True, 
    extra_options={'ActivationSymmetric': True} 
)
print('Calibrated and quantized model saved at:', output_model_path)



# preprocessing

from PIL import Image

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

def preprocess(input):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
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
input_data = preprocess(image)

# CPU inference

# Specify the path to the quantized ONNX Model
onnx_model_path = "models/resnet50_quantized.onnx"

cpu_options = onnxruntime.SessionOptions()

# Create Inference Session to run the quantized model on the CPU
cpu_session = onnxruntime.InferenceSession(
    onnx_model_path,
    providers = ['CPUExecutionProvider'],
    sess_options=cpu_options,
)
start = time.time()
cpu_outputs = cpu_session.run(None, {'input': input_data})
end = time.time()

cpu_results = postprocess(cpu_outputs)
inference_time = np.round((end - start) * 1000, 2)
idx = np.argmax(cpu_results)

print('----------------------------------------')
print('Final top prediction is: ' + labels[idx])
print('----------------------------------------')

print('----------------------------------------')
print('Inference time: ' + str(inference_time) + " ms")
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
apu_type = ''
if 'PCI\\VEN_1022&DEV_1502&REV_00' in stdout.decode(): apu_type = 'PHX/HPT'
if 'PCI\\VEN_1022&DEV_17F0&REV_00' in stdout.decode(): apu_type = 'STX'
if 'PCI\\VEN_1022&DEV_17F0&REV_10' in stdout.decode(): apu_type = 'STX'
if 'PCI\\VEN_1022&DEV_17F0&REV_11' in stdout.decode(): apu_type = 'STX'

print(f"APU Type: {apu_type}")

install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
match apu_type:
    case 'PHX/HPT':
        print("Setting environment for PHX/HPT")
        os.environ['XLNX_VART_FIRMWARE']= os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'phoenix', '1x4.xclbin')
        os.environ['NUM_OF_DPU_RUNNERS']='1'
        os.environ['XLNX_TARGET_NAME']='AMD_AIE2_Nx4_Overlay'
    case 'STX':
        print("Setting environment for STX")
        os.environ['XLNX_VART_FIRMWARE']= os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'strix', 'AMD_AIE2P_Nx4_Overlay.xclbin')
        os.environ['NUM_OF_DPU_RUNNERS']='1'
        os.environ['XLNX_TARGET_NAME']='AMD_AIE2_Nx4_Overlay'
    case _:
        print("Unrecognized APU type. Exiting.")
        exit()
print('XLNX_VART_FIRMWARE=', os.environ['XLNX_VART_FIRMWARE'])
print('NUM_OF_DPU_RUNNERS=', os.environ['NUM_OF_DPU_RUNNERS'])
print('XLNX_TARGET_NAME=', os.environ['XLNX_TARGET_NAME'])


## Point to the config file path used for the VitisAI Execution Provider
config_file_path = "./vaip_config.json"
provider_options = [{
              'config_file': config_file_path,
              'ai_analyzer_visualization': True,
              'ai_analyzer_profiling': True,
          }]

npu_session = onnxruntime.InferenceSession(
    onnx_model_path,
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