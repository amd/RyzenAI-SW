# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
import datasets as ds
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
import argparse
from transformers.utils import logging
import os
import wandb
from huggingface_hub import login
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
from accelerate import PartialState
device_string = PartialState().process_index
import random
import numpy as np
from peft import PeftModel

hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
if hf_token:
    login(hf_token)
else:
    raise ValueError("HF API KEY is not set..")
    
class LoggingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        lr = kwargs["optimizer"].param_groups[0]["lr"]
        if state.global_step % args.logging_steps == 0:
            print("----------------------------------------------------------------")
            print(f"Step {state.global_step} / {state.max_steps} completed, lr = {lr}")

def format_prompt(dataset):
    TEMPLATE = """<|begin_of_text|>Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}


    ### Input:
    {input}


    ### Response:
    {output}
    <|end_of_text|>
    """

    formatted_dataset = []
    for example in tqdm(dataset):
        formatted_text=""
        instruction = example["instruction"]
        input_ = example["input"]
        output = example["output"]
        formatted_input = TEMPLATE.format(instruction=instruction, input=input_, output=output)
        formatted_dataset.append({"text": formatted_input})
    
    formatted_dataset = Dataset.from_list(formatted_dataset)
    return formatted_dataset

def finetune_model(args, base_model, tokenizer, training_dataset):

    if args.hf_dir == "local":
        args.hf_dir = "./ft_model/"
        push_to_hub = False 
        hub_private_repo = False 
        report_to = None
    else:
        push_to_hub = True 
        hub_private_repo=True 
        report_to = "wandb"

    training_arguments = SFTConfig(
        output_dir = "./results",
        num_train_epochs = args.eps,
        per_device_train_batch_size = args.bs_per_device,
        gradient_accumulation_steps = args.grad_acc, 
        save_steps = -1,
        save_total_limit=1,
        weight_decay=0.1,
        logging_steps = 50,
        learning_rate = args.lr,
        max_grad_norm = args.max_grad_norm, 
        max_seq_length=args.max_seq_length,
        warmup_ratio = 0.03,
        lr_scheduler_type = args.scheduler,
        report_to = report_to,
        push_to_hub=push_to_hub,
        hub_model_id=args.hf_dir,
        hub_private_repo=hub_private_repo,
        hub_strategy="checkpoint"
    )

    if args.lora:
        if args.lora_all:
            peft_config = LoraConfig(
                lora_alpha = 32,
                r = 16,
                bias = "none",
                task_type = "CAUSAL_LM",
                target_modules= ["v_proj", "k_proj", "q_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
                lora_dropout=0.05
            )
        elif args.lora_qv:
            peft_config = LoraConfig(
                lora_alpha = 32,
                r = 16,
                bias = "none",
                task_type = "CAUSAL_LM",
                target_modules= ["v_proj","q_proj"],
                lora_dropout=0.1
            )
        elif args.lora_kv:
            peft_config = LoraConfig(
                lora_alpha = 32,
                r = 16,
                bias = "none",
                task_type = "CAUSAL_LM",
                target_modules = ["k_proj", "v_proj"],
                lora_dropout=0.1
            )
        elif args.lora_kvq:
            peft_config = LoraConfig(
                lora_alpha = 32,
                r = 16,
                bias = "none",
                task_type = "CAUSAL_LM",
                target_modules = ["k_proj", "v_proj", "q_proj"],
                lora_dropout=0.1
            )

        peft_model = get_peft_model(base_model, peft_config)
        peft_model.print_trainable_parameters()

        sft_trainer = SFTTrainer(
            model = base_model,
            train_dataset = training_dataset,
            peft_config = peft_config,
            tokenizer = tokenizer,
            args = training_arguments,
            callbacks = [LoggingCallback]
        )
    else:
        sft_trainer = SFTTrainer(
            model = base_model,
            train_dataset = training_dataset,
            tokenizer = tokenizer,
            args = training_arguments,
            callbacks = [LoggingCallback]
        )

    sft_trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="model to perform training with")
    parser.add_argument("--hf_dir", type=str, default="username/TEST", help="example: <username>/<reponame>")
    parser.add_argument("--bs_per_device", type=int, default=1, help="batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--eps", type=int, default=5, help="num epochs")
    parser.add_argument("--scheduler", type=str, default="linear", help="lr scheduler type")
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--grad_acc", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_all", action="store_true", help="all layers")
    parser.add_argument("--lora_qv", action="store_true", help="qv")
    parser.add_argument("--lora_kv", action="store_true", help="kv")
    parser.add_argument("--lora_kvq", action="store_true", help="qkv")
    parser.add_argument("--lora", action="store_true", help="lora")
    parser.add_argument("--wandb", action="store_true", help="log training to wandb")
    parser.add_argument("--adapter_model_dir", default="volve-adapter")
    parser.add_argument("--merge_model", action="store_true", help="merge adapter after training")

    args = parser.parse_args()

    base_model_name = args.model_name 
    #load model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code = True).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code = True)
    tokenizer.pad_token = tokenizer.eos_token

    if args.merge_model:
        adapter_model_dir = args.adapter_model_dir
        merged_model = PeftModel.from_pretrained(base_model, adapter_model_dir).to("cuda")
        merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained("llama3_1b_ddr_merged")
        tokenizer.save_pretrained("llama3_1b_ddr_merged")
    else:
        #load dataset
        training_dataset_name = "bengsoon/volve_alpaca"
        training_dataset = load_dataset(training_dataset_name, split = "train")
    
        #format dataset
        print("formatting dataset.....")
        formatted_dataset = format_prompt(training_dataset)

        #initialize wandb
        if args.wandb:
            if torch.cuda.current_device() == 0:
                wandb.init(
                    project="npu-application",
                    config= {
                        "learning_rate": args.lr,
                        "batch_size_per_dev": args.bs_per_device,
                        "epochs": args.eps,
                        "optimizer": "AdamW",
                        "lr_scheduler_type": args.scheduler,
                        "max_grad_norm": args.max_grad_norm
                    }
                )
    
        finetune_model(args, base_model, tokenizer, formatted_dataset)



