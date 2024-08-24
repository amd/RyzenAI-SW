# LLMs

The Ryzen AI Software includes support for deploying quantized LLMs on the NPU using an eager execution mode, simplifying the model ingestion process. Instead of compiling and executing as a complete graph, the model is processed on an operator-by-operator basis. Compute-intensive operations, such as GEMM/MATMUL, are dynamically offloaded to the NPU, while the remaining operators are executed on the CPU. Eager mode for LLMs is supported in both PyTorch and the ONNX Runtime. 

A general-purpose flow can be found here: [LLMs on RyzenAI with Pytorch](./models/llm/docs/README.md)

- Applicability: prototyping and early development with a broad set of LLMs
- Performance: functional support only, not to be used for benchmarking
- Supported platforms: PHX, HPT, STX (and onwards)
- Supported frameworks: Pytorch
- Supported models: Many

A set of performance-optimized models is available upon request on the AMD secure download site: [Optimized LLMs on RyzenAI](https://account.amd.com/en/member/ryzenai-sw-ea.html)

- Applicability: benchmarking and deployment of specific LLMs
- Performance: highly optimized
- Supported platforms: STX (and onwards)
- Supported frameworks: Pytorch and ONNX Runtime
- Supported models: Llama2, Llama3, Qwen1.5

This is an early access flow, and expected to be upgraded in upcoming release. 

## Run LLMs

* [LLMs on RyzenAI with Pytorch](./models/llm/docs/README.md)



