# ONNX E2E Extensions

## Introduction

ONNX has a rich set of [Operators](https://onnx.ai/onnx/operators/index.html). With this set of operators,
ONNX can describe most ML models. However, when we see any trained model, it doesn not include pre/post processing.
ONNX & ONNXRuntime provide ways to add a new or a custom operator which can be leveraged to add pre/post processing
subgraphs to the ML model.

- [Add a new OP](https://onnx.ai/onnx/repo-docs/AddNewOp.html#adding-new-operator-or-function-to-onnx)
- [Add a Custom OP](https://onnxruntime.ai/docs/extensions/add-op.html)

ONNX E2E Extensions provides a way to create and add pre/post processing operators to the trained model which
enables running end-to-end ML inference on different Execution Providers.

- Create Pre/Post Ops as Functions
  - If the operator can be expressed in terms of existing ONNX operators

- Create Pre/Post Ops as Custom OP
  - If the operator cannot be expressed in terms of existing ONNX operators

The current pre/post processing OPs available can be found in [vitis_customop](./vitis_customop)

## Setup
- [Add an IPU device driver and XRT](https://mkmartifactory.amd.com/artifactory/atg-cvml-generic-local/builds/ipu/Nightly/atg-dev/jenkins-CVML-IPU_Driver-ipu-windows-nightly-877/Release/)

### Install Vitis-AI Execution Provider
- [Download onnx-rt package](http://xcoartifactory/artifactory/vai-rt-ipu-prod-local/com/amd/onnx-rt/phx/dev/603/windows/voe-4.0-win_amd64.zip)
- Install the package
```powershell
pip install --upgrade --force-reinstall onnxruntime_vitisai-1.15.1-cp39-cp39-win_amd64.whl
pip install --upgrade --force-reinstall voe-0.1.0-cp39-cp39-win_amd64.whl
python installer.py
```

### Create conda environment

:pushpin: [Download & Install Anaconda](https://www.anaconda.com/download)

```powershell
conda env create --file setup/env.yml
```

- Activate conda environment

```powershell
conda activate onnx-vai
```

## Examples

[README](examples/README.md) for the steps to try out the examples.
