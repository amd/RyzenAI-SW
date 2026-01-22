###########################
DirectML Flow
###########################

*************
Prerequisites
*************

- DirectX12 capable Windows OS (Windows 11 recommended)
- Latest AMD `GPU device driver <https://www.amd.com/en/support>`_ installed
- `Microsoft Olive <https://microsoft.github.io/Olive/how-to/installation.html>`_ for model conversion and optimization
- Latest `ONNX Runtime DirectML EP <https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html>`_ 

You can ensure GPU driver and DirectX version from ``Windows Task Manager`` -> ``Performance`` -> ``GPU`` 

******************************
Running models on Ryzen AI GPU
******************************

Running models on the Ryzen AI GPU is accomplished in two simple steps: 

**Model Conversion and Optimization**: After the model is trained, Microsoft Olive Optimizer can be used to convert the model to ONNX and optimize it for optimal target execution.  

For additional information, refer to the `Microsoft Olive Documentation <https://microsoft.github.io/Olive/>`_


**Deployment**: Once the model is in the ONNX format, the ONNX Runtime DirectML EP (``DmlExecutionProvider``) is used to run the model on the AMD Ryzen AI GPU. 

For additional information, refer to the `ONNX Runtime documentation for the DirectML Execution Provider <https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html>`_


******** 
Examples
********

- Optimizing and running `ResNet on Ryzen AI GPU <https://github.com/amd/RyzenAI-SW/tree/main/example/iGPU/getting_started>`_


********************
Additional Resources
********************


- Article on how AMD and Black Magic Design worked together to accelerate `Davinci Resolve Studio <https://www.blackmagicdesign.com/products/davinciresolve/studio>`_ workload on AMD hardware:   

  - `AI Accelerated Video Editing with DaVinci Resolve 18.6 & AMD Radeon Graphics <https://community.amd.com/t5/ai/ai-accelerated-video-editing-with-davinci-resolve-18-6-amp-amd/ba-p/638252>`_

|

- Blog posts on using the Ryzen AI Software for various generative AI workloads on GPU:

  - `Automatic1111 Stable Diffusion WebUI with DirectML Extension on AMD GPUs <https://community.amd.com/t5/ai/how-to-automatic1111-stable-diffusion-webui-with-directml/ba-p/649027>`_

  - `Running Optimized Llama2 with Microsoft DirectML on AMD Radeon Graphics <https://community.amd.com/t5/ai/how-to-running-optimized-llama2-with-microsoft-directml-on-amd/ba-p/645190>`_

  - `AI-Assisted Mobile Workstation Workflows Powered by AMD Ryzen™ AI <https://community.amd.com/t5/business/ai-assisted-mobile-workstation-workflows-powered-by-amd-ryzen-ai/ba-p/667234>`_ 
