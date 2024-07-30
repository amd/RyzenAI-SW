# ONNXRuntime Execution Provider Setup

For users who want to use onnxruntime as the frontend, you will need to install the onnxruntime Vitis-AI Execution Provider(EP).

There are two options to install the required Vitis-AI EP:

> Note: Python 3.9 is required for onnxruntime EP installation

## Download and install from CI prebuilt packages

- Download `install_release.zip` package from [pre-built package](http://xcoartifactory/ui/native/vai-rt-ipu-prod-local/com/amd/onnx-rt/dev/latest/) and extract it.
- Activate your `ryzenai-transformers` conda environment
    ```batch
    conda activate ryzenai-transformers
    ```
- Then execute following command to copy files and install packages:
    ```batch
    cd install_release
    pip install --upgrade --force-reinstall onnxruntime_vitisai-1.15.1-cp39-cp39-win_amd64.whl
    pip install --upgrade --force-reinstall voe-0.1.0-cp39-cp39-win_amd64.whl
    python installer.py
    ```

## Build the onnxruntime EP from scratch

If you need to do some development works and want to keep the onnxruntime EP updated with the latest code, you can build the package from scratch by following these steps:

### Prequisites

- Visual Studio 2019, with individual component **spectre** installed. For adding **Spectre Mitigation** to Visual Studio, please refer to the instructions [here](https://learn.microsoft.com/en-us/visualstudio/msbuild/errors/msb8040?view=vs-2019).
- CMake (version >= 3.26)
- XRT. Download the xrt `test_package.zip` from [the artifactory](http://mkmartifactory.amd.com/artifactory/atg-cvml-generic-local/builds/ipu/GithubVerification/acas-dev-phx/jenkins-CVML-IPU_Driver-ipu-windows-githubPR-verification-3824/Release/) and unzip it for later use.

### Build onnxruntime EP


```
conda activate ryzenai-transformers
cd transformers
setup_phx.bat
cd ext\vai-rt
./build_latest.bat
```

### Some known issues
  - Protobuf related errors: check if you have installed protobuf somewhere already, then remove it (generally in base conda env).
