# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""LoRA fine-tune on a frozen Quark-quantized base (quant-aware, dynamic LoRA).

The adapter is trained against the INT4 (Quark AWQ) base rather than the FP model,
so the LoRA correction remains valid when the base is deployed as an INT4 model on
the Ryzen AI NPU. The quantized model is loaded back into PyTorch with Quark's
``import_model_from_safetensors`` (the base stays INT4 and dequantizes to bf16 on
the fly during the forward pass); the base is frozen and only the adapter is trained
and saved unmerged so it can be switched at runtime.

Expects a training file of JSON-lines records, each with ``prompt`` and ``response``
fields (see prepare_data.py). The same script trains any task -- point it at a
different dataset and output name.

Example:
    python finetune_on_quark.py \
        --quark-model ./qwen3-4b-quark-int4 \
        --train-file ./data_spider/train.jsonl \
        --adapter-out ./adapters/sql_spider
"""
import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from quark.torch.export.api import import_model_from_safetensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B",
                    help="FP skeleton matching the quantized base")
    ap.add_argument("--quark-model", required=True,
                    help="Local Quark INT4 dir (config.json + model.safetensors)")
    ap.add_argument("--train-file", required=True,
                    help="JSONL with {'prompt':..., 'response':...} records")
    ap.add_argument("--adapter-out", required=True, help="Output adapter dir")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load the bf16 skeleton, then transform in place into the Quark INT4 model.
    print(f"Loading base skeleton: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    print(f"Importing Quark INT4 weights (in place): {args.quark_model}")
    model = import_model_from_safetensors(model, args.quark_model)
    model.config.use_cache = False

    # Freeze the entire quantized base; PEFT adds the trainable LoRA params.
    for p in model.parameters():
        p.requires_grad = False

    peft_config = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    eos = tok.eos_token or "<|endoftext|>"
    ds = load_dataset("json", data_files={"train": args.train_file})["train"]
    ds = ds.map(lambda ex: {"text": ex["prompt"] + ex["response"] + eos},
                remove_columns=ds.column_names)
    print(f"train rows: {len(ds)}")

    sft = SFTConfig(
        output_dir=os.path.join("runs", os.path.basename(args.adapter_out)),
        num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03, bf16=True, logging_steps=20,
        max_length=args.max_seq_len, packing=False,
        dataset_text_field="text", report_to="none")

    trainer = SFTTrainer(model=model, args=sft, train_dataset=ds,
                         processing_class=tok)
    trainer.train()

    trainer.model.save_pretrained(args.adapter_out)  # unmerged adapter
    tok.save_pretrained(args.adapter_out)
    print(f"\nSaved unmerged adapter to: {args.adapter_out}")


if __name__ == "__main__":
    main()
