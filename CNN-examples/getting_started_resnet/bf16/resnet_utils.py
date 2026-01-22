# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import subprocess
from pathlib import Path

def get_directories():
    current_dir = Path(__file__).resolve().parent

    # models directory for resnet sample
    models_dir = current_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # data directory for resnet sample
    data_dir = current_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # cache directory for resnet sample
    cache_dir = current_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    return current_dir, models_dir, data_dir, cache_dir

def get_npu_info():
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

def get_xclbin(npu_device):
    xclbin_file = ''
    if npu_device == 'STX' or npu_device=='KRK':
        xclbin_file = '{}\\voe-4.0-win_amd64\\xclbins\\strix\\AMD_AIE2P_4x4_Overlay.xclbin'.format(os.environ["RYZEN_AI_INSTALLATION_PATH"])
    if npu_device == 'PHX/HPT':
        xclbin_file = '{}\\voe-4.0-win_amd64\\xclbins\\phoenix\\4x4.xclbin'.format(os.environ["RYZEN_AI_INSTALLATION_PATH"])
    return xclbin_file
