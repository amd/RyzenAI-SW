# Installation

> import CIStatus from '@site/src/components/CIStatus';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import ExpectedOutput from '@site/src/components/ExpectedOutput';

import CIStatus from '@site/src/components/CIStatus';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import ExpectedOutput from '@site/src/components/ExpectedOutput';

# Installation

<CIStatus validated={false} />

This page covers Ryzen AI Software installation on Windows. For Linux LLM setup, see [Running LLM on Linux](/models-tutorials/llms/linux-setup).

## Prerequisites

The Ryzen AI Software supports AMD processors with a Neural Processing Unit (NPU). Refer to the [Supported Hardware](/getting-started/supported-hardware) page for the full list of supported configurations.

The following must be installed before installing the Ryzen AI Software:

| Dependency | Version Requirement |
|------------|---------------------|
| **Windows 11** | Build >= 22621.3527 |
| **Visual Studio 2022** | Optional — only required for AMD Quark custom op flows |
| **cmake** | Version >= 3.26 |
| **Python distribution** | [Miniforge](https://github.com/conda-forge/miniforge) (preferred) |

:::warning
**IMPORTANT**:

- **Visual Studio 2022 Community** (Optional for AMD Quark, to support custom op flow): ensure that `Desktop Development with C++` is installed.

- **Miniforge**: ensure that the following path is set in the System PATH variable: `path\to\miniforge3\condabin` or `path\to\miniforge3\Scripts\` or `path\to\miniforge3\`. The System PATH variable should be set in the *System Variables* section of the *Environment Variables* window.
:::

## Step 1: Install NPU Drivers

Download and install the NPU driver (version 32.0.203.280 or newer):

| Driver Version | Supported Platforms | Download |
|---------------|---------------------|----------|
| 32.0.203.280 | Phoenix, Hawk Point, Strix, Strix Halo, Krackan Point | [Download](https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=NPU_RAI1.5_280_WHQL.zip) |
| 32.0.203.314 | Latest platforms | [Download](https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=NPU_RAI1.6.1_314_WHQL.zip) |

**Installation steps:**

1. Extract the downloaded ZIP file
2. Open a terminal in **administrator mode**
3. Run the installer:

```powershell
.\npu_sw_installer.exe
```

**Verify the driver** by opening **Task Manager → Performance → NPU0**. You should see the NPU device listed.

## Step 2: Install Ryzen AI Software

Download the Ryzen AI Software bundled installer: [ryzenai-lt-1.7.0.exe](https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=ryzen-ai-lt-1.7.0.exe)

Launch the EXE installer and follow the installation wizard:

1. Accept the terms of the License agreement
2. Provide the destination folder for Ryzen AI installation (default: `C:\Program Files\RyzenAI\1.7.0`)
3. Specify the name for the conda environment (default: `ryzen-ai-1.7.0`)

The installer creates the conda environment and installs all Ryzen AI Software packages into it automatically.

:::info
NuGet package is also available: [ryzen-ai-1.7.0-nuget.zip](https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=signed_nuget_1.7.0.zip)
:::

## Step 3: Test the Installation

The Ryzen AI Software installation includes a `quicktest` to verify that everything is correctly installed. This test is expected to work for Strix (STX) or newer devices.

1. Open a Conda command prompt (search for "Miniforge Prompt" in the Windows start menu)

2. Activate the Conda environment created by the Ryzen AI installer:

```bat
conda activate ryzen-ai-1.7.0
```

3. Run the test:

```bat
cd %RYZEN_AI_INSTALLATION_PATH%\quicktest
python quicktest.py
```

<ExpectedOutput>
{`INFO: [aiecompiler 77-749] Reading logical device aie2p_8x4_device
Using TXN FORMAT 0.1
Test Passed`}
</ExpectedOutput>

4. Verify NPU activity by opening **Task Manager → Performance → NPU** while the test is running. You should see NPU utilization increase during model inference.

:::tip
To see detailed NPU offloading logs, run with verbose filtering:

```bat
python quicktest.py 2>&1 | findstr /i "Operators Subgraphs VITIS_EP_CPU NPU Test"
```

<ExpectedOutput label="Expected verbose output (Strix/Krackan Point)">
{`[Vitis AI EP] No. of Operators :
    NPU   398
    VITIS_EP_CPU     2
[Vitis AI EP] No. of Subgraphs :
  NPU     1
Test Passed`}
</ExpectedOutput>
:::

:::info
- The full installation path is stored in the `RYZEN_AI_INSTALLATION_PATH` environment variable.
- For Phoenix/Hawk Point hardware, additional session options are required (`target` set to `X1`). See the [NPU Offloading with Session Options](#npu-offloading-with-session-options) section below.
:::

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
    command = r'pnputil /enum-devices /bus PCI /deviceids '
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    npu_type = ''
    if 'PCI\\VEN_1022&DEV_1502&REV_00' in stdout.decode(): npu_type = 'PHX/HPT'
    if 'PCI\\VEN_1022&DEV_17F0&REV_00' in stdout.decode(): npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_10' in stdout.decode(): npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_11' in stdout.decode(): npu_type = 'STX'
    if 'PCI\\VEN_1022&DEV_17F0&REV_20' in stdout.decode(): npu_type = 'KRK'
    return npu_type

npu_type = get_npu_info()
install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
model       = os.path.join(install_dir, 'quicktest', 'test_model.onnx')
providers   = ['VitisAIExecutionProvider']
provider_options = [{}]

if npu_type == 'PHX/HPT':
    print("Setting environment for PHX/HPT")
    xclbin_file = os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'phoenix', '4x4.xclbin')
    provider_options = [{
         'target': 'X1',
         'xlnx_enable_py3_round': 0,
         'xclbin': xclbin_file,
    }]

session_options = ort.SessionOptions()
session_options.log_severity_level = 1

try:
    session = ort.InferenceSession(model,
                             sess_options=session_options,
                             providers=providers,
                             provider_options=provider_options)
except Exception as e:
    print(f"Failed to create an InferenceSession: {e}")
    sys.exit(1)

def preprocess_random_image():
    image_array = np.random.rand(3, 32, 32).astype(np.float32)
    return np.expand_dims(image_array, axis=0)

input_data = preprocess_random_image()
try:
    outputs = session.run(None, {'input': input_data})
except Exception as e:
    print(f"Failed to run the InferenceSession: {e}")
    sys.exit(1)
else:
   print("Test finished")
```

Run the test with verbose filtering:

```bat
cd %RYZEN_AI_INSTALLATION_PATH%\quicktest
python quicktest.py 2>&1 | findstr /i "Operators Subgraphs VITIS_EP_CPU NPU Test"
```

<ExpectedOutput label="Expected verbose output (Strix/Krackan Point)">
{`[Vitis AI EP] No. of Operators :
    NPU   398
    VITIS_EP_CPU     2
[Vitis AI EP] No. of Subgraphs :
  NPU     1
Test finished`}
</ExpectedOutput>

:::info
- For Phoenix/Hawk Point hardware, set the `target` to `X1` in the provider options.
:::

## Alternative: Standalone LLM Install (pip)

If you only need to run LLMs and prefer not to use the bundled installer, you can set up a standalone environment. See [Running LLM via pip install](/models-tutorials/llms/hybrid-inference#running-llm-via-pip-install) for instructions.

For the high-level Python SDK (Lemonade), see [High-Level Python SDK](/models-tutorials/llms/python-api) which provides a quick PyPI-based setup.

## Next Steps

- [Quickstart](/getting-started/quickstart) -- Run your first LLM on the NPU
- [Supported Hardware](/getting-started/supported-hardware) -- Full hardware compatibility matrix
