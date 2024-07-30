import torch
from datasets import load_dataset
from trl import SFTTrainer

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

"""
A simple example on using SFTTrainer and Accelerate to finetune Phi-3 models. For
a more advanced example, please follow HF alignment-handbook/scripts/run_sft.py

1. Install accelerate:
    conda install -c conda-forge accelerate
2. Setup accelerate config:
    accelerate config
to simply use all the GPUs available:
    python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='bf16')"
check accelerate config:
    accelerate env
3. Run the code:
    accelerate launch sample_finetune.py
"""

###################
# Hyper-parameters
###################
args = {
    "bf16": True,
    "do_eval": False,
    "eval_strategy": "no",
    "learning_rate": 5.0e-06,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "max_steps": -1,
    "output_dir": "./checkpoint_dir",
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 4,
    "per_device_train_batch_size": 8,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 0,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.2,
}

training_args = TrainingArguments(**args)

################
# Modle Loading
################
checkpoint_path = "microsoft/Phi-3-mini-4k-instruct"
# checkpoint_path = "microsoft/Phi-3-mini-128k-instruct"
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.pad_token = (
    tokenizer.unk_token
)  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = "right"


##################
# Data Processing
##################
def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["messages"]
    # Add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return example


raw_dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
column_names = list(raw_dataset["train_sft"].features)

processed_dataset = raw_dataset.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=12,
    remove_columns=column_names,
    desc="Applying chat template",
)
train_dataset = processed_dataset["train_sft"]
eval_dataset = processed_dataset["test_sft"]

###########
# Training
###########
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_seq_length=2048,
    dataset_text_field="text",
    tokenizer=tokenizer,
    packing=True,
)
train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

#############
# Evaluation
#############
tokenizer.padding_side = "left"
metrics = trainer.evaluate()
metrics["eval_samples"] = len(eval_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

############
# Save model
############
trainer.save_model(training_args.output_dir)
