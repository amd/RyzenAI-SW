# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\onnx-model-preparation.mdx:70
cd examples/torch/language_modeling/llm_ptq/

python quantize_quark.py \
     --no_trust_remote_code \
     --model_dir "meta-llama/Llama-2-7b-chat-hf"  \
     --output_dir <quantized safetensor output dir>  \
     --quant_scheme w_uint4_per_group_asym \
     --group_size 128 \
     --num_calib_data 128 \
     --seq_len 512 \
     --quant_algo awq \
     --dataset pileval_for_awq_benchmark \
     --model_export hf_format \
     --data_type <datatype> \
     --exclude_layers []
