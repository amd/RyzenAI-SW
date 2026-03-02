# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\getting-started\manual-installation.mdx:101
import os 
import sys
import subprocess
import numpy as np
import onnxruntime as ort

def get_npu_info():
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
    if 'PCI\\VEN_1022&DEV_17F0&REV_20' in stdout.decode(): npu_type = 'KRK'
    return npu_type

# Get APU type info: PHX/STX/HPT
npu_type = get_npu_info()
install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
model       = os.path.join(install_dir, 'quicktest', 'test_model.onnx')
providers   = ['VitisAIExecutionProvider']
provider_options = [{}]  # Default provider options for STX/KRK and newer devices

if npu_type == 'PHX/HPT':
    print("Setting environment for PHX/HPT")
    xclbin_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'phoenix', '4x4.xclbin')
    provider_options = [{
         'target': 'X1',
         'xlnx_enable_py3_round': 0,
         'xclbin': xclbin_file,
    }]

# Create session options
session_options = ort.SessionOptions()
session_options.log_severity_level = 1  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal

try:
    session = ort.InferenceSession(model,
                             sess_options=session_options,
                             providers=providers,
                             provider_options=provider_options)
except Exception as e:
    print(f"Failed to create an InferenceSession: {e}")
    sys.exit(1)  # Exit the program with a non-zero status to indicate an error

def preprocess_random_image():
    image_array = np.random.rand(3, 32, 32).astype(np.float32)
    return np.expand_dims(image_array, axis=0)

# inference on random image data
input_data = preprocess_random_image()
try:
    outputs = session.run(None, {'input': input_data})
except Exception as e:
    print(f"Failed to run the InferenceSession: {e}")
    sys.exit(1)  # Exit the program with a non-zero status to indicate an error
else:
   print("Test finished")
