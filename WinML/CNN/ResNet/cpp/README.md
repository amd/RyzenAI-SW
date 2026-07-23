# WinML ResNet Tutorial: C++

This tutorial demonstrates how to use Windows Machine Learning (WinML) for ONNX model
inference in C++ on the AMD Ryzen AI NPU. It builds a ResNet-50 image classifier using the
ONNX Runtime C++ API with WinML's automatic execution provider management.

## Prerequisites

Complete the [C++ Setup](../../../README.md#c-setup) in the main README first (Visual Studio
2022+ with the Desktop C++ workload, the Windows SDK, and the matching Windows App SDK
runtime).

You also need the ResNet ONNX model. Follow [Download Model](../README.md#download-model) in
the ResNet tutorial to produce `../model/resnet50.onnx`.

## Build

On first configure it fetches the required NuGet packages
(`Microsoft.WindowsAppSDK.ML`, `Microsoft.Windows.AI.MachineLearning`, WIL) into the build
tree and generates the C++/WinRT projection headers.

```powershell
cd CNN\ResNet\cpp
cmake -G "Visual Studio 17 2022" -A x64 -S . -B build
cmake --build build --config Release
```

The executable and its runtime DLLs are staged in `build\bin\Release\`.

> **Package versions** are pinned at the top of `CMakeLists.txt` (`MML_WASDK_VERSION`).
> To change them, edit the pin and delete `build\` to force a clean re-fetch. Note this is
> the NuGet *package* version, which is separate from the installed Windows App SDK *runtime*
> version (installed per the main README C++ setup).

## Run

Run inference on the NPU:

```powershell
build\bin\Release\CppResnetBuildDemo.exe --model ..\model\resnet50.onnx --image_path ..\images\dog.jpg --ep_policy NPU
```

Or run with defaults (NPU policy, `resnet50.onnx`, and `dog.jpg` staged beside the exe):

```powershell
build\bin\Release\CppResnetBuildDemo.exe
```

If you quantized the model, pass its path via `--model`.

### Command-Line Arguments

- `--ep_policy <NPU|CPU|DEFAULT>`: Execution provider policy. Default: NPU
- `--model <path>`: Path to the input ONNX model (default: `../model/resnet50.onnx`)
- `--compiled_output <path>`: Path for the compiled output model (default: `../model/resnet50_ctx.onnx`)
- `--image_path <path>`: Path to the input image (default: `dog.jpg` beside the executable)
- `--help`: Show the help message

## API Walkthrough

### Execution Provider Setup

The Windows ML runtime dynamically discovers and registers available execution providers (EPs):

```cpp
#include <winml/onnxruntime_cxx_api.h>
#include <winrt/Microsoft.Windows.AI.MachineLearning.h>

using namespace winrt::Microsoft::Windows::AI::MachineLearning;

// Create the ONNX Runtime environment
auto env = Ort::Env();

// Use WinML to ensure and register the certified execution providers
auto catalog = ExecutionProviderCatalog::GetDefault();
catalog.EnsureAndRegisterCertifiedAsync().get();
```

### Session Configuration

Configure session options to select the execution provider by device policy:

```cpp
Ort::SessionOptions sessionOptions;
sessionOptions.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_NPU);
```

**EP Selection Policies:**
- `OrtExecutionProviderDevicePolicy_PREFER_NPU` — prefer the NPU
- `OrtExecutionProviderDevicePolicy_PREFER_CPU` — prefer the CPU
- `OrtExecutionProviderDevicePolicy_DEFAULT` — default selection

### Model Compilation

Compilation optimizes the model for the selected EP. It is a one-time process; the compiled
model is cached and reused on subsequent runs:

```cpp
if (!std::filesystem::exists(compiledModelPath))
{
    Ort::ModelCompilationOptions compile_options(env, sessionOptions);
    compile_options.SetInputModelPath(modelPath.c_str());
    compile_options.SetOutputModelPath(compiledModelPath.c_str());
    Ort::CompileModel(env, compile_options);
}
```

### Session Creation

Create an inference session with the (compiled) model:

```cpp
Ort::Session session(env, modelPathToUse.c_str(), sessionOptions);
```

## Expected Output

```console
ONNX Version string: 1.25.2
Getting available providers...
Provider:VitisAIExecutionProvider is ready to use.
ONNX providers registered:
CPUExecutionProvider
DmlExecutionProvider
VitisAIExecutionProvider
Using execution provider policy: NPU
ResNet model loaded
Running inference for 20 iterations
....................
Output for the last iteration
Top Predictions:
-------------------------------------------
Label                           Confidence
-------------------------------------------
207,golden retriever                 51.65%
208,Labrador retriever                0.98%
852,tennis ball                       0.57%
205,flat-coated retriever             0.45%
244,Tibetan mastiff                   0.39%
-------------------------------------------
Avg time per iteration : 13 milliseconds
```

## References

- [WinML Documentation](https://learn.microsoft.com/en-us/windows/ai/windows-ml/)
- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/)
- [Windows App SDK](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/)
- [ResNet-50 Model on HuggingFace](https://huggingface.co/microsoft/resnet-50)
