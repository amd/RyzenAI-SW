<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI RAG </h1>
    </td>
 </tr>
</table>

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

---

## 1. Installation and Setup

### 1.1 Download the ONNX-Based Llama Model from Hugging Face

```sh
git clone https://huggingface.co/amd/Llama-3.2-3B-Instruct-onnx-ryzenai-1.7-hybrid /path/to/your/directory/
```
replace `/path/to/your/directory/` with actual path where you want to download the model.

### 1.2  Activate Ryzen AI Environment

To ensure compatibility with ONNX-based Llama model, activate the ryzen-ai-1.7.0 Conda environment.

Please follow instructions provided in the official AMD documentation to install Ryzen AI 1.7.0:

👉 [Ryzen AI 1.7.0 Installation and Conda Environment Creation](https://ryzenai.docs.amd.com/en/latest/inst.html)

After installation, activate the environment by running:

```sh
conda activate ryzen-ai-1.7.0
```
### 1.3  Install Dependencies

After activating the environment, install the required Python dependencies by running:
```sh
git clone https://github.com/amd/RyzenAI-SW.git
cd RyzenAI-SW/example/llm/RAG-OGA
pip install -r requirements.txt
```

## 2. Demo
To explore the use case, please refer below steps:

### 2.1 Retrieval-Augmented Generation (RAG) Pipeline

This example demonstrates a Retrieval-Augmented Generation (RAG) pipeline orchestrated using the LangChain framework. In this setup, documents are indexed into a Facebook AI Similarity Search(FAISS) vector database and retrieved at inference time to enrich user prompts with relevant contextual information.

The following models are deployed using Ryzen AI 1.7.0:

- **Embedding Model**: [BGE (BAAI General Embedding)](https://huggingface.co/BAAI/bge-large-en-v1.5), compiled using Vitis AI Execution Provider.

- **Hybrid LLM**: [Llama3.2-3B-Instruct](https://huggingface.co/amd/Llama-3.2-3B-Instruct-onnx-ryzenai-1.7-hybrid), a quantized ONNX model, running using the OGA(OnnxRuntime GenAI) framework on Ryzen AI 1.7.0.

By running both critical models on the NPU and/or GPU, this setup enables faster and more efficient inference, delivering a high-performance RAG system optimized for AI PCs.

<p align="center">
  <img src="./image/RAG_Diagram.png" alt="RAG Diagram" width="700" style="border: 1px solid black;"/>
</p>
<p align="center"><b>RAG Workflow with LangChain and ONNX</b></p>

### 2.2 🔑 Key Components of the LangChain-Based RAG Pipeline

This RAG pipeline runs locally on an AMD Ryzen AI PC (with NPU & GPU). It combines LangChain, FAISS, embeddings, and an LLM to deliver fast, on‑device question‑answering.

#### 🔹 Data Embedding
Documents are preprocessed and converted into dense vector representations using the BGE embedding model.

#### 🔹 ONNX Inference on AMD NPU
The embedding model is executed using ONNX Runtime on the NPU (Ryzen AI).

#### 🔹 Vector Store Creation
Document embeddings are stored in a FAISS-based vector database for efficient similarity search.

#### 🔹 Context Retrieval
The vector database returns the most relevant document chunks based on the embedded query.

#### 🔹 LLM Prompt Construction
LangChain constructs a prompt using the user’s query, prompt template, and the retrieved context.

#### 🔹 LLM Response Generation
The retrieved data, along with the user’s query, is fed into a custom LLM, running on a hybrid flow (GPU and NPU), to generate a response from the retrieved data.

### 2.3 Download, Export to ONNX, and Compile the Embedding Model.

Run the following command to perform download, export and compile steps:

```bash
python custom_embedding/export_bge_onnx.py
```
Note : Please ensure that you have activated your ryzen‑ai‑1.7.0 environment and are in the RyzenAI‑SW/example/llm/RAG‑OGA directory.

This script generates a static‑shape, non‑quantized FP32 ONNX model that serves as the baseline for further deployment. 
The compiled BGE (BAAI General Embedding) ONNX model will be stored in the cache folder named ``modelcachekey_bge``.


### 2.4 Run the sample RAG application

The system supports two modes of query handling.
- ``--direct_llm`` mode, where the user's query is directly sent to the LLM without any document retrieval.

-  If ``--direct_llm`` flag is not specified, the query triggers retrieval from a FAISS index, enriching the prompt with relevant context before passing it to the LLM.

#### Required Setup: Update Paths in rag.py
- Dataset Path:
 Replace the placeholder with the dataset provided in this directory used to build the FAISS index.
```
dataset_path = r"./Dataset"
```

- LLM Model Path:
 Replace the path to your LLM model that you downloaded in step 1.1
```
llm = custom_llm(model_path="path/to/llm")
```


## 2.5 Sample Outputs

**Case 1: Direct LLM mode (where no retrieval is being done)**
```sh 
python rag.py --direct_llm
```
Ask any question

**For instance,**
```
Enter your question: what is NPU and tell me the three important feature of NPU.
Direct_llm mode is on. No retrieval has been performed.
LLM_call invoked: 1 time(s)
Answer:
NPU stands for Net Protein Utilization, which is a measure of the proportion of dietary protein that is actually utilized by the body for growth and maintenance of tissues. The three important features of NPU are: (1) It is a measure of protein quality, indicating the extent to which a protein is effective in promoting growth and maintenance of body tissues. (2) It is influenced by factors such as the protein's amino acid composition, digestibility, and bioavailability. (3) NPU is a critical factor in determining the adequacy of protein intake, as it helps to identify the protein sources that are most effective in meeting the body's protein needs.

```

**Case 2: Retrieval mode**

In the **Retrieval mode**, documents most similar to the query are retrieved using FAISS, enabling efficient semantic search based on vector similarity.
You can observe how the model behaves differently between direct mode and retrieval mode:

For instance,
```sh 
python rag.py 
```
**Sample Output**

***Question 1***
```
Enter your question:  what is NPU and tell me the three important feature of NPU.
Retrieval mode is on.
Loading existing FAISS index from disk...
LLM_call invoked: 1 time(s)
Answer:
The NPU (Neural Processing Unit) is a specialized processor designed for neural network processing, specifically for deep learning and artificial intelligence applications.
The three important features of NPU are:
1.  **High Performance**: NPU is designed to provide high-performance computing for deep learning workloads, making it an ideal choice for applications that require fast processing of large amounts of data.
2.  **Energy Efficiency**: NPU is designed to be energy-efficient, which is critical for mobile devices and other applications where power consumption is a major concern.
3.  **Low Latency**: NPU is designed to provide low latency, which is critical for real-time applications such as autonomous vehicles, robotics, and other IoT devices.
```

***Question 2***

```
Enter your question: what are the main feature provided by the AMD analyzer, and how does it help in visualizing model execution on Ryzen AI ?
Retrieval mode is on.
Loading existing FAISS index from disk...
LLM_call invoked: 1 time(s)

Answer:
 ## Step 1: Identify the main features of the AMD AI Analyzer
The AMD AI Analyzer is a tool that supports analysis and visualization of model compilation and inference on Ryzen AI. The main features provided by the AMD AI Analyzer include:

- Graph and operator partitions between the NPU and CPU
- Visualization of graph and operator partitions
- Profiling and visualization of model execution
- Generation of artifacts related to inference profile and graph partitions

## Step 2: Explain how the AMD AI Analyzer helps in visualizing model execution on Ryzen AI
The AMD AI Analyzer helps in visualizing model execution on Ryzen AI by providing a comprehensive view of the model's performance and execution on the NPU. The tool allows users to:

- Visualize graph and operator partitions to understand how the model is processed by the hardware
- Profile and visualize model execution to identify performance bottlenecks
- Generate artifacts related to inference profile and graph partitions to gain deeper insights into the model's behavior

## Step 3: Highlight the benefits of using the AMD AI Analyzer
The AMD AI Analyzer provides several benefits, including:

- Improved understanding of model execution on Ryzen AI
- Identification of performance bottlenecks and optimization opportunities
- Generation of artifacts for further analysis and optimization

The final answer is: The AMD AI Analyzer provides a comprehensive set of features that help in visualizing model execution on Ryzen AI, including graph and operator partitions, profiling and visualization, and generation of artifacts related to inference profile and graph partitions. These features enable users to gain a deeper understanding of the model's performance and behavior on the NPU, identify performance bottlenecks, and optimize the model for better performance and power efficiency.
 

```

***Question 3***
```
Enter your question: In the context of Ryzen AI Software's hybrid inference model, how does the integration of automated
operator assignment, encrypted context caching, and hardware-specific xclbin configurations collectively contribute to 
optimizing performance, ensuring security, and minimizing compilation overhead across varying model types such as transformers
and CNNs?
Retriveval mode is on.
 Loading existing FAISS index from disk...

Answer:
The integration of automated operator assignment, encrypted context caching, and hardware-specific xclbin configurations collectively contributes to optimizing performance, ensuring security, and minimizing compilation overhead across varying model types such as transformers and CNNs in the following ways:

1. **Automated Operator Assignment**: This feature optimizes the placement of operators in the model, ensuring that the most efficient and effective assignments are made, which leads to improved performance and reduced computational overhead.

2. **Encrypted Context Caching**: This feature ensures that sensitive model data is protected from unauthorized access, thereby enhancing security. By caching context information, the model can be efficiently transferred and executed across different environments, reducing the need for manual intervention and minimizing compilation overhead.

3. **Hardware-Specific xclbin Configurations**: These configurations are tailored to the specific capabilities of the target platform, ensuring that INT8 models are optimized for the hardware, which leads to improved performance and reduced power consumption. This also enhances security by protecting sensitive model data from unauthorized access.

Together, these features work synergistically to optimize performance, ensure security, and minimize compilation overhead across varying model types such as transformers and CNNs. This results in faster inference times, reduced power consumption, and improved overall efficiency, making the Ryzen AI Software's hybrid inference model a powerful tool for AI and machine learning applications
```


## 2.6 Profiling

The example code also captures key LLM performance metrics, such as Time to First Token (TTFT), Tokens Per Second (TPS), input prompt length, and total generated tokens, providing a clear view of system responsiveness and throughput.

To enable profiling, run the sample with the ``--profiling`` flag:

```sh
python rag.py --profiling
```
**Note:**   
Actual numbers may vary depending on the LLM used, model version, and specific system configuration.

**Sample output:** 
```
--- Aggregated Profiling Summary ---

Q1:
  Avg Input Tokens              : 1607
  Avg Output Tokens             : 339
  Avg TTFT(Sec)                 : 1.640761
  Avg TPS                       : 31.16

Q2:
  Avg Input Tokens              : 1171
  Avg Output Tokens             : 354
  Avg TTFT(Sec)                 : 1.16953
  Avg TPS                       : 32.74

Q3:
  Avg Input Tokens              : 1458
  Avg Output Tokens             : 1
  Avg TTFT(Sec)                 : 1.393054
  Avg TPS                       : 0.0 
```
  





