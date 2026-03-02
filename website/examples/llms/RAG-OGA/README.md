# README

> <table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI RAG </h1>
    </td>
 </tr>
</table>

Ryzen™ AI RAG 
    
 

## Introduction
Welcome to this repository, a showcase of an **ONNX Runtime GenAI(OGA)‑based RAG LLM sample application** running on a **Ryzen AI processor**.
This repo provides supplemental code to the AMD Blog  [RAG with Hybrid LLM on AMD Ryzen AI Processor](https://www.amd.com/en/developer/resources/technical-articles/2025/rag-with-hybrid-llm-on-amd-ryzen-ai-processors.html).

## What You’ll Find Here

- **Retrieval-Augmented Generation (RAG) pipeline** powered by:
  - A **hybrid LLM** enables disaggregated inference in which the compute-heavy prefill phase runs on the NPU, while the decode phase executes on the GPU.
  - An **embedding model** compiled with **Vitis AI Execution Provider**
- Built using the widely adopted **LangChain** orchestration framework

## Quick Setup

Follow these simple steps to get started:

1. Execute the setup steps outlined below to provision your environment.
2. After setup, this README will guide you through how to run the sample application.
