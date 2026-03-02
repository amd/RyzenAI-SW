# DirectML Flow (GPU)

> import CIStatus from '@site/src/components/CIStatus';

import CIStatus from '@site/src/components/CIStatus';

# DirectML Flow

<CIStatus validated={false} />

## Prerequisites

- DirectX12 capable Windows OS (Windows 11 recommended)
- Latest AMD [GPU device driver](https://www.amd.com/en/support) installed
- [Microsoft Olive](https://microsoft.github.io/Olive/how-to/installation.html) for model conversion and optimization
- Latest [ONNX Runtime DirectML EP](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)

You can ensure GPU driver and DirectX version from `Windows Task Manager` -> `Performance` -> `GPU`

## Running models on Ryzen AI GPU

Running models on the Ryzen AI GPU is accomplished in two simple steps:

**Model Conversion and Optimization**: After the model is trained, Microsoft Olive Optimizer can be used to convert the model to ONNX and optimize it for optimal target execution.

For additional information, refer to the [Microsoft Olive Documentation](https://microsoft.github.io/Olive/)

**Deployment**: Once the model is in the ONNX format, the ONNX Runtime DirectML EP (`DmlExecutionProvider`) is used to run the model on the AMD Ryzen AI GPU.

For additional information, refer to the [ONNX Runtime documentation for the DirectML Execution Provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)

## Examples

- Optimizing and running [ResNet on Ryzen AI GPU](/models-tutorials/vision/igpu-getting-started)

## Additional Resources

- Article on how AMD and Black Magic Design worked together to accelerate [Davinci Resolve Studio](https://www.blackmagicdesign.com/products/davinciresolve/studio) workload on AMD hardware:

  - [AI Accelerated Video Editing with DaVinci Resolve 18.6 & AMD Radeon Graphics](https://www.amd.com/en/blogs/2023/ai-accelerated-video-editing-with-davinci-resolve-.html)

- Blog posts on using the Ryzen AI Software for various generative AI workloads on GPU:

  - [Automatic1111 Stable Diffusion WebUI with DirectML Extension on AMD GPUs](https://www.amd.com/en/blogs/2023/-how-to-automatic1111-stable-diffusion-webui-with.html)

  - [Running Optimized Llama2 with Microsoft DirectML on AMD Radeon Graphics](https://www.amd.com/en/blogs/2023/-how-to-running-optimized-llama2-with-microsoft-d.html)

  - [AI-Assisted Mobile Workstation Workflows Powered by AMD Ryzen™ AI](https://www.amd.com/en/blogs/2024/ai-assisted-mobile-workstation-workflows-powered-b.html)
