# ONNX E2E Pre/Post Processing Flow

## Introduction

ONNX has a rich set of [Operators](https://onnx.ai/onnx/operators/index.html). With this set of operators,
ONNX can describe most ML models. However, when we see any trained model, it does not include pre/post processing. In order to run the pre/post-processing operations on the IPU, we present a way to integrate these operations into the original ONNX model. To do this, we leverage the custom-operator insertion method provided by ONNX & ONNXRuntime. 

- [Add a new OP](https://onnx.ai/onnx/repo-docs/AddNewOp.html#adding-new-operator-or-function-to-onnx)
- [Add a Custom OP](https://onnxruntime.ai/docs/extensions/add-op.html)

ONNX E2E Extensions provides a way to create and add pre/post processing operators to the trained model which
enables running end-to-end ML inference on different Execution Providers.

- Create Pre/Post Ops as Functions
  - If the operator can be expressed in terms of existing ONNX operators

- Create Pre/Post Ops as Custom OP
  - If the operator cannot be expressed in terms of existing ONNX operators

We currently support some pre/post processing OPs, which are available in [vitis_customop](./vitis_customop). 

## Setup

### Ensure IPU Driver installation

Ensure [IPU Driver](https://ryzenai.docs.amd.com/en/latest/inst.html#prepare-the-system) is installed as per the official installation page


### Install Ryzen AI Software

1. Download and extract [Ryzen AI Software installation package](https://account.amd.com/en/forms/downloads/ryzen-ai-software-platform-xef.html?filename=ryzen-ai-sw-1.1.zip) 

2. Install [Ryzen AI Software](https://ryzenai.docs.amd.com/en/latest/inst.html#install-the-ryzen-ai-software)

Customize the installation by naming the environment onnx-vai. This can be done by using the -env flag: 

```powershell
.\install.bat -env onnx-vai
```

### Install additional dependencies

Activate the Anaconda environment created by the Ryzen AI software installer

```powershell
conda activate onnx-vai
git clone https://github.com/amd/RyzenAI-SW.git
cd RyzenAI-SW\example\onnx-e2e
conda env update --file setup/env.yml
conda activate onnx-vai
```


## Examples

[README](examples/README.md) for the steps to try out the examples.

