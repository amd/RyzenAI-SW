# Manual Installation

> import CIStatus from '@site/src/components/CIStatus';

import CIStatus from '@site/src/components/CIStatus';

# Installation Instructions

<CIStatus validated={false} />

This page covers Ryzen AI installation on Windows.

## Prerequisites

The Ryzen AI Software supports AMD processors with a Neural Processing Unit (NPU). Refer to the release notes for the full list of [supported configurations](/getting-started/supported-hardware).

The following dependencies must be installed on the system before installing the Ryzen AI Software:

| Dependencies | Version Requirement |
|-------------|---------------------|
| Windows 11 | build >= 22621.3527 |
| Visual Studio | 2022 |
| cmake | version >= 3.26 |
| Python distribution (Miniforge preferred) | Latest version |

:::warning
**IMPORTANT**:

- Visual Studio 2022 Community (Optional for AMD Quark, to support custom op flow): ensure that `Desktop Development with C++` is installed

- Miniforge: ensure that the following path is set in the System PATH variable: `path\to\miniforge3\condabin` or `path\to\miniforge3\Scripts\` or `path\to\miniforge3\` (The System PATH variable should be set in the *System Variables* section of the *Environment Variables* window).
:::

## Install NPU Drivers

- Download and Install the NPU driver version: 32.0.203.280 or newer using the following links:

  - [NPU Driver (Version 32.0.203.280)](https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=NPU_RAI1.5_280_WHQL.zip)

    - NPU driver 32.0.203.280 is production driver for Phoenix, Hawk Point, Strix, Strix Halo, and Krackan Point.
  - [NPU Driver (Version 32.0.203.314)](https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=NPU_RAI1.6.1_314_WHQL.zip)

- Install the NPU drivers by following these steps:

  - Extract the downloaded ZIP file.
  - Open a terminal in administrator mode and execute the `.\npu_sw_installer.exe` file.

- Ensure that NPU driver (Version:32.0.203.280, Date:5/16/2025) is correctly installed by opening Task Manager -> Performance -> NPU0.

## Install Ryzen AI Software

- Download the Ryzen AI Software installer [ryzenai-lt-1.7.0.exe](https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=ryzen-ai-lt-1.7.0.exe).

- Launch the EXE installer and follow the instructions on the installation wizard:

  - Accept the terms of the License agreement
  - Provide the destination folder for Ryzen AI installation (default: `C:\Program Files\RyzenAI\1.7.0`)
  - Specify the name for the conda environment (default: `ryzen-ai-1.7.0`)

The Ryzen AI Software packages are now installed in the conda environment created by the installer.

:::info
NuGet package is available to download at [ryzen-ai-1.7.0-nuget.zip](https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=signed_nuget_1.7.0.zip).
:::

## Test the Installation

The Ryzen AI Software installation folder contains test to verify that the software is correctly installed. This installation test can be found in the `quicktest` subfolder which is expected to work for Strix (STX) or newer devices.

- Open a Conda command prompt (search for "Miniforge Prompt" in the Windows start menu)

- Activate the Conda environment created by the Ryzen AI installer:

```bash
conda activate <env_name>
```

- Run the test:

```bat
cd %RYZEN_AI_INSTALLATION_PATH%\quicktest
python quicktest.py
```

```
INFO: [aiecompiler 77-749] Reading logical device aie2p_8x4_device
Using TXN FORMAT 0.1
Test Passed
```

- Verify NPU activity by opening **Task Manager → Performance → NPU** while the test is running. You should see NPU utilization increase during model inference.

## NPU Offloading with Session Options

This section demonstrates how to enable NPU offloading logs using ONNX Runtime session options. The code also includes changes needed in `quicktest.py` to run on Phoenix/Hawk Point devices.
To view detailed logging information, update the session options in `quicktest.py` as shown below:

```python
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
```

- Run the test:

```bat
cd %RYZEN_AI_INSTALLATION_PATH%\quicktest
python quicktest.py 2>&1 | findstr /i "Operators Subgraphs VITIS_EP_CPU NPU Test"
```

- On a successful run, you will see an output similar to the one shown below. This indicates that the model is running on the NPU and that the installation of the Ryzen AI Software was successful:

```
[Vitis AI EP] No. of Operators :
    NPU   398
    VITIS_EP_CPU     2
[Vitis AI EP] No. of Subgraphs :
  NPU     1
Test finished
```

:::info
- The full path to the Ryzen AI Software installation folder is stored in the `RYZEN_AI_INSTALLATION_PATH` environment variable.
- For Phoenix/Hawk Point hardware, set the `target` to `X1` in the provider options.
:::

---

Ryzen AI is licensed under [MIT License](https://github.com/amd/ryzen-ai-documentation/blob/main/License). Refer to the [LICENSE File](https://github.com/amd/ryzen-ai-documentation/blob/main/License) for the full license text and copyright notice.
