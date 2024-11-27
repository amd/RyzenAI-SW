
# LLMs

The Ryzen AI Software includes support for deploying quantized LLMs on the NPU using an eager execution mode, simplifying the model ingestion process. Instead of compiling and executing as a complete graph, the model is processed on an operator-by-operator basis. Compute-intensive operations, such as GEMM/MATMUL, are dynamically offloaded to the NPU, while the remaining operators are executed on the CPU. 

A set of performance-optimized models is available upon request on the AMD secure download site: [Optimized LLMs on RyzenAI](https://account.amd.com/en/member/ryzenai-sw-ea.html)

- Applicability: benchmarking and deployment of specific LLMs
- Performance: highly optimized
- Supported platforms: STX (and onwards)
- Supported frameworks: ONNX Runtime GenAI

A general-purpose flow can be found here: [LLMs on RyzenAI with Pytorch](./models/llm/docs/README.md)

- Applicability: prototyping and early development with a broad set of LLMs
- Performance: functional support only, not to be used for benchmarking
- Supported platforms: PHX, HPT, STX (and onwards)
- Supported frameworks: Pytorch




## Run LLMs

* [LLMs on RyzenAI with Pytorch](./models/llm/docs/README.md)



