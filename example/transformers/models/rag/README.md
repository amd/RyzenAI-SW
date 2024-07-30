# How to run RAG LLM on AMD Ryzen AI NPU 

To leverage AMD Ryzen AI NPU for RAG LLM application involves setting up your environment correctly. The following steps will guide you through preparing your environment, quantizing the model for efficient execution on the NPU, and integrating it into the RAG framework. 

**Note:** This example is intended solely for demonstrating the integration of the Ryzen-AI LLM flow with LlamaIndex for the Retrieval-Augmented Generation (RAG) application. The context passed with RAG could have a prompt length greater than 2048 tokens, for which it is not tuned for performance optimization yet.

### 1. Clone Ryzen AI Transformers Repository 

- Clone the Ryzen AI SW repository. For the initial environment setup, follow the detailed steps provided in the `RyzenAI/example/transformers/models/llm` folder. 

### 2. Quantize the model

- Download the original model weights and place it in the `RyzenAI-SW/example/transformers/models/llm`  directory. Now to generate the quantized model to run on the NPU, follow these steps: 
 

  ```cd RyzenAI-SW/example/transformers/models/llm```
  
  ```python run_awq.py --model_name llama-2-7b-chat --task quantize```

 
If you have followed these instructions, you would now have a quantized model file, such as `quantized_llama-2-7b-chat_w4_g128_awq.pth` in the `RyzenAI-SW/example/transformers/models/llm/quantized_models` folder. 
 

### 3. Prepare the RAG LLM Framework 

Once you have completed the setup and quantization using the Ryzen AI transformers guide, you can integrate the quantized model into the RAG Framework. 

- Within the same conda environment, install additional dependencies required for the RAG Framework: 

  ```cd RyzenAI-SW/example/transformers/models/rag ```
  
  ```pip install -r requirements.txt ```

- Copy the original llama-2-7b-chat model from step 1, into `models/rag/llama-2-7b-chat`
- Copy the quantized model from step 1, into `models/rag/quantized_models`
 

### 4. Configure and run the RAG Application 
Configure the RAG Application to use the quantized model, enabling the optional features like speculative decoding (assisted generation) 

```python run.py --model_name llama-2-7b-chat --target aie --no-direct_llm --quantized --assisted_generation ```

*Note:* `fast_attention` optimization is currently only supported for input prompt/token length <=2048, and is turned off in this RAG example for 1.2 release. 
