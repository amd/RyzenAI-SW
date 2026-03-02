# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import torch 
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset 
import datasets
import json
import evaluate
from tqdm import tqdm 
import random 
import numpy as np 
import onnxruntime_genai as og 
import argparse

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

def volve_alpaca_generate_oga(model, tokenizer, inference_file_name, max_new_tokens=512):
    eval_set = datasets.load_dataset("bengsoon/volve_alpaca")["test"].select(range(50))
        
    def generate(model, tokenizer, prompt):
        inputs = tokenizer.encode(prompt)
        search_options = {}
        params = og.GeneratorParams(model)
        search_options["max_length"] = len(inputs) + max_new_tokens 
        params.set_search_options(**search_options)
        generator = og.Generator(model, params)
        generator.append_tokens(inputs)
        tokens = []
        response = ''
        while not generator.is_done():
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            tokens.append(new_token)
            response += tokenizer.decode(new_token)
        del generator
        return response 
        
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
    parser.add_argument('--model_dir', type=str, required=True, help="model path dir")
    parser.add_argument ('--inference_filename', type=str, required=True, help="predictions saving file name")
    args = parser.parse_args()

    ### evaluating oga model
    onnx_model_path = args.model_dir
    oga_model = og.Model(onnx_model_path)
    tokenizer = og.Tokenizer(oga_model)
    volve_alpaca_generate_oga(oga_model, tokenizer, args.inference_filename)
    volve_alpaca_eval(args.inference_filename)
