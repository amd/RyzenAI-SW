# Llama 2 - Pytorch

Llama 2 model was proposed by Meta, [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/).


**_NOTE:_**  Please ensure that you followed the instructions from the [Transformer folder readme](../../README.md) before following the steps here.


# Features
1. w4a16: AWQ on CPU, AIE **(AWQ for Llama2 his from MIT-Han-lab)**
4. Static memory allocation for token phase speed up
5. Matmul grouping for Llama2Attn
6. Model save and load from checkpoints
7. Profiling instrumentation to measure prefill/token times during mission mode

# Model Variants
Llama-2 has several variants: **llama-2-7b", llama-2-7b-chat, llama-2-13b, llama-2-13b-chat, llama-2-30b, llama-2-30b-chat**. ***For Ryzen-AI, limit to llama-2-7bopt1.3b***. 

Llama2 can support a max. sequence length of 4096 tokens.

# Model Structure 
## llama-2-7b , llama-2-7b-chat 
This image shows model structure - both have same structure
![Model Structure](./llama-2-7b-model.png)


# Perplexity scores
Perplexity is measured using negative log likelihood scores.
***Perplexity ios measured on wikitext2-raw dataset***
***Perplexity measurement takes several hours on both CPU and AIE***
***Lower value is better***

On RyzenAI, it takes ~5mins to calculate perplexity on 2 samples with seqlen of 2048. To calculate it on entire testset (84) samples with sequence length of 4096, it takes ~7 hrs.

| **Precision+Config**                                     | **Device** | **2 samples/seq=2048** | **All(84) samples/seq=4096**
|----------------------------------------------------------|------------|------------------------|------------------------------
FP32                                                       | CPU        |  5.092                 |
**BF16**                                                   | **V100**   |  **6.718**             | **6.859**
FP32 + SmoothQuant                                         | CPU        |  5.092                 |
FP32 + Flash Attention v2                                  | CPU        |  5.092                 |
w4abf16 (AWQ, 3-bit, group size: 128)                      | CPU        |  7.807                 |
w4abf16 (AWQ, 3-bit, group size: 128)                      | **AIE**    |  **7.809**             | 
w4abf16 (AWQ, 4-bit, group size: 128)                      | **AIE**    |  **6.925**             | 

# Support modes on NPU

Currently the supported precision on NPU is "w4abf16" and "w4abf16 + FA". **w4abf16** uses **AWQ** PerGrp quantization

 

# Prepare Llama2 Weights to use with HF

The weights of Llama-2 models can be obtained by requesting permission with Meta. Check this [Huggingface page](https://huggingface.co/docs/transformers/main/model_doc/llama2) on Llama-2 for details. 

Once weights are obtained, use Huggingface's converter to convert the weights to be compatible to be loaded with HF interface. 

```
# directory structure of llama-2 weights
$ls -ltrh llama-2-wts
total 536K
-rw-r--r-- 1 user user 489K Jul 13 15:27 tokenizer.model
-rw-r--r-- 1 user user   50 Jul 13 15:27 tokenizer_checklist.chk
-rw-r--r-- 1 user user 6.9K Jul 14 17:06 LICENSE
-rw-r--r-- 1 user user 4.7K Jul 14 17:06 USE_POLICY.md
drwxr-sr-x 2 user user 4.0K Aug 31 11:12 llama-2-7b
drwxr-sr-x 2 user user 4.0K Aug 31 11:15 llama-2-13b
drwxr-sr-x 2 user user 4.0K Aug 31 11:17 llama-2-70b
drwxr-sr-x 2 user user 4.0K Aug 31 11:17 llama-2-7b-chat
drwxr-sr-x 2 user user 4.0K Aug 31 11:17 llama-2-13b-chat
drwxr-sr-x 2 user user 4.0K Aug 31 11:17 llama-2-70b-chat

# rename llama-2-7b as 7B

$ ls -ltrh llama-2-wts
total 500K
drwxr-sr-x 2 user user 4.0K Sep 28 12:44 7B
-rw-r--r-- 1 user user   50 Sep 28 12:45 tokenizer_checklist.chk
-rw-r--r-- 1 user user 489K Sep 28 12:45 tokenizer.model

# Run the converter
$ python <condainstall>/envs/ryzenai-transformers/lib/python3.9/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ./llama-2-wts/ --model_size 7B --output_dir ./llama-2-wts-hf/7B

# you want to convert llama-2-7b-chat, rename the llama-2-7b-chat to 7B and rerun the converter again as follows

$ python <condainstall>/envs/ryzenai-transformers/lib/python3.9/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ./llama-2-wts/ --model_size 7B --output_dir ./llama-2-wts-hf/7B_chat
```



# Go to the llama2 folder


```
cd models/llama2/
```

Ensure the folder `llama-2-wts-hf` at the current location containing subfolders `7B` and `7B_chat`

# [README for w4abf16 with AWQ, PerGrp Quant](./README_w4abf16.MD)


# Model Computation complexity analysis

```
python run.py --quant_mode none --task opsprofile --target cpu
...

****************************************
Seqlen: 4  GOPs: 52.871958808
Seqlen: 8  GOPs: 105.76046139200001
Seqlen: 16  GOPs: 211.58789660800002
Seqlen: 32  GOPs: 423.44448723200003
Seqlen: 64  GOPs: 847.964549248
Seqlen: 128  GOPs: 1700.232196352
Seqlen: 256  GOPs: 3417.6775828480004
Seqlen: 512  GOPs: 6904.2087249920005
Seqlen: 1024  GOPs: 14083.832485888
Seqlen: 2000  GOPs: 28532.896046240003
Seqlen: 3000  GOPs: 44375.283032240004
Seqlen: 4000  GOPs: 61268.295954240006
```
