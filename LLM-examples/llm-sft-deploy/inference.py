# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import torch 
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import json
import evaluate
from tqdm import tqdm 
import random 
import numpy as np 
import onnxruntime_genai as og 
import argparse
from quark_utils import *

random.seed(0)
np.random.seed(1234)
torch.manual_seed(1234)

def volve_alpaca_eval(inference_file_name):
    bertscore = evaluate.load("bertscore")
    eval_set = datasets.load_dataset("bengsoon/volve_alpaca")["test"].select(range(50))
    with open(inference_file_name, "r") as f:
        generated_outputs = json.load(f)

    predictions= []
    references = []
    for dataset_ex, generated in zip(eval_set, generated_outputs):
        response = generated["output"]
        predictions.append(response)
        references.append(dataset_ex["output"])

    bertscore = bertscore.compute(predictions=predictions, references=references, lang="en")
    bertscore_f1_avg = sum(bertscore["f1"])/len(bertscore["f1"])

    print(f"bert f1 avg: {bertscore_f1_avg}")

def volve_alpaca_generate(model, tokenizer, inference_file_name):
    eval_set = datasets.load_dataset("bengsoon/volve_alpaca")["test"].select(range(50))
    def generate(model, tokenizer, prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output_toks = model.generate(**inputs,max_new_tokens=512)
        return tokenizer.decode(output_toks[0], skip_special_tokens=True).replace(prompt, "").strip()

    def format_prompt(example):
        instruction = example["instruction"]
        input_ = example["input"]
        TEMPLATE = """<|begin_of_text|>Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}


        ### Input:
        {input}


        ### Response:

        """
        formatted_input = TEMPLATE.format(instruction=instruction, input=input_)
        return formatted_input 
    
    eval_set = eval_set.map(lambda ex: {"formatted_prompt": format_prompt(ex)})
    outputs = []
    for example in tqdm(eval_set):
        response = generate(model, tokenizer, example["formatted_prompt"])
        response = response.split("Response:")[1].strip()
        outputs.append(
            {
                "prompt": example["formatted_prompt"],
                "reference_output": example["output"],
                "output": response
            }
        )
    with open(inference_file_name, "w") as f:
        json.dump(outputs, f, indent=2)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp', action="store_true", help="eval full precision, unquantized model")
    parser.add_argument('--quark_safetensors', action="store_true", help="eval quark quantized safetensors")
    parser.add_argument('--model_dir', type=str, help="model path dir of finetuned model")
    parser.add_argument ('--inference_filename', type=str, help="predictions saving file name")
    parser.add_argument('--quant_model_dir', type=str, help="model path dir of quantized model")
    args = parser.parse_args()

    ### evaluating quark quantized model (safetensors) ####
    if args.quark_safetensors:
        base_model_name= "meta-llama/Llama-3.2-1B"
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True).to("cuda")

        quark_model = import_hf_model(base_model, model_info_dir=args.quant_model_dir)
        volve_alpaca_generate(quark_model, tokenizer, args.inference_filename)
        
    ### evaluating fp model (not quark quantized) ###
    elif args.fp:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        volve_alpaca_generate(model, tokenizer, args.inference_filename)

    volve_alpaca_eval(args.inference_filename)
