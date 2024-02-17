# w4abf16 with AWQ + PerGrp Quantization

AWQ enables 3-bit and 4-bit weights for LLMs. This reduces model size of Llama2 7B from 52-58% of int8 model depending on group size and whether the last layer is quantized. 

AWQ scales are obtained from MIT-Han-Lab. THe script also contains hooks to calculate scales instead of using precomputed scales. All layers other than "lm_head" are quantized using AWQ. This software stack of RyzenAI can also quantize lm_head layer using per group quantization scheme, with group sizes varying from 32-256. 

Linear layers are replaced with QLinearPerGrp custom int4 compute layer afetr AWQ.
![AWQ pic](./awq_lmhead.png)

Matmul grouping in done when flash attention is enabled, which reduces number of dispatches to AIE by grouping matmuls of QKV matrices into a single grouped matmul.
This in addition to static memory allocation in token phases provides 8-10% better performance than vanilla attention.

![AWQ FA](./awq_fa.png)

## Save AWQ checkpoints
4-bit AWQ has higher perplexity than 3-bit AWQ with same performance.

### 4-bit AWQ
```
AWQ - lm_head runs in BF16 
python run_awq.py --w_bit 4 --task quantize 

AWQ - lm_head runs in BF16 
python run_awq.py --w_bit 4 --task quantize --flash_attention

AWQ + Quantize lm_head 
python run_awq.py --w_bit 4 --task quantize --lm_head

AWQ + Quantize lm_head
python run_awq.py --w_bit 4 --task quantize --lm_head --flash_attention
```

### 3-bit AWQ
```
AWQ - lm_head runs in BF16 
python run_awq.py --w_bit 3 --task quantize 

AWQ - lm_head runs in BF16 
python run_awq.py --w_bit 3 --task quantize --flash_attention

AWQ + Quantize lm_head 
python run_awq.py --w_bit 3 --task quantize --lm_head

AWQ + Quantize lm_head
python run_awq.py --w_bit 3 --task quantize --lm_head --flash_attention
```



## Run Llama2 decode examples
```
python run_awq.py --task decode --target aie --w_bit 4

****************************************
prompt: What is the meaning of life?
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
response: What is the meaning of life? This is a question that has puzzled philosophers, theologians, scientists, and many others for
****************************************
prompt: Tell me something you don't know.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
response: Tell me something you don't know.
The only thing I can think of is that I don't know how to make you disappear
****************************************
prompt: What does Xilinx do?
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
response: What does Xilinx do?

Xilinx is a company that specializes in designing and manufacturing electronic products, particularly program
*
```

## Latency : PHX (Ryzen9 7945HS)
```
python run_awq.py --target aie --task benchmark --flash_attention --w_bit 4
...
```

```
python run_awq.py --target aie --task benchmark_long --flash_attention --w_bit 4
...
```

```
python run_awq.py --target aie --task benchmark --flash_attention --w_bit 4 --lm_head
...
```

```
python run_awq.py --target aie --task benchmark_long --flash_attention --w_bit 4 --lm_head
...
```

```
python run_awq.py --target aie --task benchmark --flash_attention --w_bit 3
...
```


