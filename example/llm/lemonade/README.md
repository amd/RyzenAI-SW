# Ryzen AI LLM Lemonade Examples

The following table contains a curated list of LLMs that have been validated with the [Lemonade SDK](https://github.com/onnx/turnkeyml/blob/main/docs/lemonade/README.md) on Ryzen AI hybrid execution mode, along with CPU implementations of those same checkpoints. 

The hybrid examples are built on top of OnnxRuntime GenAI (OGA), while the CPU baseline is built on top of Hugging Face (HF) ``transformers``. Validation is defined as running all commands in the example page successfully.

<table class="tg"><thead>
  <tr>
    <th class="tg-invis"></th>
    <th class="tg-top" colspan="2">CPU (Hugging Face bfloat16)</th>
    <th class="tg-top" colspan="2">Ryzen AI Hybrid (OGA int4)</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-heading">Model</td>
    <td class="tg-heading">Example</td>
    <td class="tg-heading">Validation</td>
    <td class="tg-heading">Example</td>
    <td class="tg-heading">Validation</td>
  </tr>
  <tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B">DeepSeek-R1-Distill-Qwen-7B</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/DeepSeek_R1_Distill_Qwen_7B.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/DeepSeek_R1_Distill_Qwen_7B.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B">DeepSeek-R1-Distill-Qwen-1.5B</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/DeepSeek_R1_Distill_Qwen_1_5B.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/DeepSeek_R1_Distill_Qwen_1_5B.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B">DeepSeek-R1-Distill-Llama-8B</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/DeepSeek_R1_Distill_Llama_8B.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/DeepSeek_R1_Distill_Llama_8B.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/meta-llama/CodeLlama-7b-Instruct-hf">CodeLlama-7b-Instruct-hf</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/CodeLlama_7b_Instruct_hf.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/CodeLlama_7b_Instruct_hf.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct">Llama-3.2-1B-Instruct</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/Llama_3_2_1B_Instruct.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/Llama_3_2_1B_Instruct.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct">Llama-3.2-3B-Instruct</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/Llama_3_2_3B_Instruct.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/Llama_3_2_3B_Instruct.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct">Phi-3-mini-4k-instruct</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/Phi_3_mini_4k_instruct.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/Phi_3_mini_4k_instruct.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/Qwen/Qwen1.5-7B-Chat">Qwen1.5-7B-Chat</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/Qwen1_5_7B_Chat.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/Qwen1_5_7B_Chat.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3">Mistral-7B-Instruct-v0.3</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/Mistral_7B_Instruct_v0_3.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/Mistral_7B_Instruct_v0_3.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct">Llama-3.1-8B-Instruct</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/Llama_3_1_8B_Instruct.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/Llama_3_1_8B_Instruct.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/microsoft/Phi-3.5-mini-instruct">Phi-3.5-mini-instruct</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/Phi_3_5_mini_instruct.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/Phi_3_5_mini_instruct.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/meta-llama/Llama-2-7b-hf">Llama-2-7b-hf</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/Llama_2_7b_hf.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/Llama_2_7b_hf.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf">Llama-2-7b-chat-hf</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/Llama_2_7b_chat_hf.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/Llama_2_7b_chat_hf.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B">Meta-Llama-3-8B</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/Meta_Llama_3_8B.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/Meta_Llama_3_8B.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/meta-llama/Llama-3.1-8B">Llama-3.1-8B</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/Llama_3_1_8B.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/Llama_3_1_8B.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/Qwen/Qwen2-1.5B">Qwen2-1.5B</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/Qwen2_1_5B.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/Qwen2_1_5B.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/Qwen/Qwen2-7B">Qwen2-7B</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/Qwen2_7B.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/Qwen2_7B.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr><tr>
        <td class="tg-cell-nowrap"><a href="https://huggingface.co/google/gemma-2-2b">gemma-2-2b</a></td><!--Model-->
        <td class="tg-cell"><a href="cpu/gemma_2_2b.md">Link</a></td><!--cpu Example-->
    <td class="tg-cell">🟢</td><!--cpu validation-->
    <td class="tg-cell"><a href="hybrid/gemma_2_2b.md">Link</a></td><!--hybrid Example-->
    <td class="tg-cell">🟢</td><!--hybrid validation-->
  </tr>
</tbody></table>

# Copyright

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.