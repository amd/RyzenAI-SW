# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Run a Ryzen AI NPU model with dynamic LoRA adapter switching.

Loads the compiled NPU model once, then activates any compiled adapter by name at
runtime. Adapter activation is provided by the Ryzen AI runtime (``ryzenai_onnx_utils``):

    set_active_adapter(name)    # activate the compiled adapter <name>
    set_active_adapter("base")  # revert to the base model (no adapter)

The named adapter files (``<name>_prefill.bin`` / ``<name>_token.bin`` and their
``.pb.bin`` headers) must live in the model directory. Run from the folder that has
the copied Ryzen AI runtime DLLs.

Examples:
    # single adapter
    python run_lora_switch.py -m ./qwen3-4b-npu --adapter sql_spider --task sql

    # switch between two adapters on one loaded base (prints each output)
    python run_lora_switch.py -m ./qwen3-4b-npu --switch sql_spider med_mcqa
"""
import argparse

import onnxruntime_genai as og
from ryzenai_onnx_utils import set_active_adapter

SQL_PROMPT = (
    "You are a text-to-SQL assistant. Given a database schema and a question,\n"
    "write a single valid SQL query that answers it.\n\n"
    "### Schema:\n"
    "CREATE TABLE country (Code TEXT, Continent TEXT);\n"
    "CREATE TABLE countrylanguage (CountryCode TEXT, Language TEXT)\n\n"
    "### Question:\nWhat are the languages spoken in countries in Asia?\n\n### SQL:\n"
)

MED_PROMPT = (
    "You are a medical expert answering a multiple-choice question.\n"
    "Choose the single best answer. Respond with only the letter (A, B, C, or D).\n\n"
    "Question: Which vitamin deficiency causes scurvy?\n"
    "A. Vitamin A\nB. Vitamin C\nC. Vitamin D\nD. Vitamin B12\n\n"
    "Answer:"
)

PROMPTS = {"sql": SQL_PROMPT, "med": MED_PROMPT}


def generate(model, tok, prompt, max_new_tokens=64):
    p = og.GeneratorParams(model)
    p.set_search_options(do_sample=False, max_length=len(tok.encode(prompt)) + max_new_tokens)
    g = og.Generator(model, p)
    g.append_tokens(tok.encode(prompt))
    out = []
    while not g.is_done():
        g.generate_next_token()
        out.append(g.get_next_tokens()[0])
    return tok.decode(out).strip().split("\n")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Compiled NPU model dir")
    ap.add_argument("--adapter", help="Single adapter name to activate (or 'base')")
    ap.add_argument("--switch", nargs="+", help="Adapter names to switch through in sequence")
    ap.add_argument("--task", choices=["sql", "med"], default="sql",
                    help="Built-in prompt to use for --adapter")
    ap.add_argument("--prompt", help="Custom prompt (overrides --task)")
    args = ap.parse_args()

    model = og.Model(args.model)   # loads the base NPU model + Ryzen AI EP (once)
    tok = og.Tokenizer(model)

    if args.switch:
        # Demonstrate switching between multiple adapters on one loaded base.
        # Give each adapter a prompt from its own domain when names hint the task.
        for name in args.switch:
            set_active_adapter(name)
            task = "med" if "med" in name.lower() else "sql"
            prompt = PROMPTS[task]
            print(f"[{name}] {generate(model, tok, prompt)}")
    else:
        name = args.adapter or "base"
        set_active_adapter(name)   # "base" reverts to the base model
        prompt = args.prompt if args.prompt else PROMPTS[args.task]
        print(f"[{name}] {generate(model, tok, prompt)}")


if __name__ == "__main__":
    main()
