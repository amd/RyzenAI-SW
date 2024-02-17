import os
import sys 
CWD = os.getcwd()
AWQ_PATH = CWD + "/../"
sys.path.append(AWQ_PATH + "awq/quantize")
sys.path.append(AWQ_PATH + "awq/utils")
sys.path.append(AWQ_PATH + "scripts")
AWQ_CACHE = CWD + "/../../awq_cache/"

from pre_quant import run_awq, apply_awq
from quantizer import pseudo_quantize_model_weight, real_quantize_model_weight
from qmodule import WQLinear 

from transformers import AutoTokenizer
from transformers.models.opt.modeling_opt import OPTForCausalLM
import torch 

model_name = "opt-125m"
model = OPTForCausalLM.from_pretrained("facebook/" + model_name) 
tokenizer = AutoTokenizer.from_pretrained("facebook/" +  model_name)


for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        print(f"{n} : {m.weight.data.min()}  {m.weight.data.max()}")

quant_path = "opt-125m-awq"
quant_config = {"zero_point": True, "q_group_size": 128}

print("Loading pre-computed AWQ results from", AWQ_CACHE)
awq_results = torch.load(AWQ_CACHE + "/%s-w4-g128.pt"%model_name, map_location="cpu")
apply_awq(model, awq_results)

for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        print(f"After AWQ - {n} : {m.weight.data.min()}  {m.weight.data.max()}")

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": True,
    "q_group_size": 128,  # whether to use group quantization

}
print("Quantization config:", q_config)

# Load AWQ
if True:
    real_quantize_model_weight(
                    model, w_bit=4, q_config=q_config
                )

    print(model)

    for n, m in model.named_modules():
        if isinstance(m, WQLinear):
            print(f"After AWQ - {n} : {m.qweight.data.min()}  {m.qweight.data.max()}  {m.qweight.data.shape} {m.scales.shape}  {m.qzeros.shape} ")

    del model 
    import gc 
    gc.collect()
    model = OPTForCausalLM.from_pretrained("facebook/" + model_name) 

# Run AWQ
if True:
    awq_results = run_awq(
                model, tokenizer,
                w_bit=4, q_config=q_config,
                n_samples=128, seqlen=512,
            )
    torch.save(awq_results, "./%s-w4-g128-generated.pt"%model_name)