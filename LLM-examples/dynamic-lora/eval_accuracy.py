# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Evaluate base vs LoRA accuracy, on the GPU or on the Ryzen AI NPU.

Runs the quantized INT4 base with or without a LoRA adapter over a benchmark and
reports accuracy, so the base-vs-LoRA uplift can be reproduced. Two tasks:

  medmcqa : exact-match accuracy on MedMCQA validation (predicted letter vs key).
            4-way multiple choice, so 25% is chance.
  spider  : execution accuracy on Spider dev (run predicted + gold SQL on the real
            SQLite database, compare returned rows). Execution accuracy rather than
            string match, because a valid query that differs in formatting, quoting,
            or join order is still correct.

Two devices:

  (default)  GPU. Quantized base via transformers + PEFT. Use during development to
             check an adapter helps before spending time on NPU compilation.
  --npu      The compiled model on the Ryzen AI NPU, via ONNX Runtime GenAI.
             Adapters are activated by name. This is the path that produced the
             published NPU numbers.

Run base and lora as two SEPARATE passes over the same questions, then compare.
Set the adapter state once per run rather than toggling per request.

Examples (GPU):
    python eval_accuracy.py --task medmcqa --quark-model ./qwen3-4b-quark-int4 --mode base -n 300
    python eval_accuracy.py --task medmcqa --quark-model ./qwen3-4b-quark-int4 --mode lora \
        --adapter ./adapters/med_mcqa -n 300

    python eval_accuracy.py --task spider --quark-model ./qwen3-4b-quark-int4 --mode base \
        --spider ./spider_data -n 100
    python eval_accuracy.py --task spider --quark-model ./qwen3-4b-quark-int4 --mode lora \
        --adapter ./adapters/sql_spider --spider ./spider_data -n 100

Examples (NPU -- reproduces the published numbers, n=200):
    python eval_accuracy.py --npu --task spider --npu-model .\\qwen3-4b-npu \
        --mode base --spider .\\spider_data -n 200
    python eval_accuracy.py --npu --task spider --npu-model .\\qwen3-4b-npu \
        --mode lora --adapter-name sql_spider --spider .\\spider_data -n 200

    python eval_accuracy.py --npu --task medmcqa --npu-model .\\qwen3-4b-npu \
        --mode base -n 200
    python eval_accuracy.py --npu --task medmcqa --npu-model .\\qwen3-4b-npu \
        --mode lora --adapter-name med_mcqa -n 200
"""
import argparse
import os
import re
import sqlite3

LETTERS = ["A", "B", "C", "D"]


def parse_letter(text):
    """Extract the predicted option letter from a completion.

    Prefers a STANDALONE A/B/C/D, falling back to any letter. The order matters:
    on "The answer is C." a bare [ABCD] search returns the "A" in "answer", not the
    actual prediction. Returns None if the completion has no letter at all.
    """
    m = re.search(r"\b([ABCD])\b", text.upper())
    if m:
        return m.group(1)
    m = re.search(r"[ABCD]", text.upper())
    return m.group(0) if m else None


def clean_sql(text):
    """Pull a single executable SQL statement out of a raw completion.

    Models wrap SQL in markdown fences or prefix it with prose; feeding that
    straight to sqlite3 fails and scores a correct answer as wrong. Strips any
    ```sql fence, seeks the first SELECT/WITH, keeps one line, drops the trailing
    semicolon.
    """
    t = text.strip()
    m = re.search(r"```(?:sql)?\s*(.+?)```", t, re.DOTALL | re.IGNORECASE)
    if m:
        t = m.group(1).strip()
    m = re.search(r"(?is)\b(select|with)\b.+", t)
    if m:
        t = m.group(0)
    return t.strip().split("\n")[0].strip().rstrip(";")

MED_PROMPT = (
    "You are a medical expert answering a multiple-choice question.\n"
    "Choose the single best answer. Respond with only the letter (A, B, C, or D).\n\n"
    "Question: {q}\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
    "Answer:"
)

SQL_PROMPT = (
    "You are a text-to-SQL assistant. Given a database schema and a question,\n"
    "write a single valid SQL query that answers it.\n\n"
    "### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:\n"
)


def load_gpu_model(skeleton, quark_dir, mode, adapter):
    """Load the quantized INT4 base on a GPU (transformers), adapter via PEFT.

    Returns a ``gen(prompt, max_new_tokens) -> str`` function, so the scoring code
    below is identical for GPU and NPU and the two can't drift apart.

    torch / transformers / Quark are imported here rather than at module scope so
    this file still starts on a Ryzen AI PC, which has none of them installed.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from quark.torch.export.api import import_model_from_safetensors

    tok = AutoTokenizer.from_pretrained(skeleton)
    model = AutoModelForCausalLM.from_pretrained(skeleton, torch_dtype=torch.bfloat16)
    model = import_model_from_safetensors(model, quark_dir)  # INT4 in place
    if mode == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"[adapter] attached LoRA: {adapter}")
    else:
        print("[adapter] base (no adapter attached)")
    model = model.eval().cuda()

    def gen(prompt, max_new_tokens):
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

    return gen


def load_npu_model(model_dir, mode, adapter_name, dll):
    """Load the compiled model on the Ryzen AI NPU via ONNX Runtime GenAI.

    Returns a ``gen(prompt, max_new_tokens) -> str`` function, matching
    load_gpu_model so the scoring code below is device-agnostic.

    The base is loaded ONCE and the adapter is activated ONCE, before any
    generation -- adapters are not toggled per request inside an accuracy run.
    """
    import onnxruntime_genai as og
    from ryzenai_onnx_utils import set_active_adapter

    model = og.Model(model_dir)   # loads base + Ryzen AI EP once
    tok = og.Tokenizer(model)

    if mode == "lora":
        set_active_adapter(adapter_name, dll)
        print(f"[adapter] activated LoRA '{adapter_name}'")
    else:
        set_active_adapter("base", dll)
        print("[adapter] base (adapter unloaded)")

    def gen(prompt, max_new_tokens):
        toks = tok.encode(prompt)
        p = og.GeneratorParams(model)
        p.set_search_options(do_sample=False,
                             max_length=min(16384, len(toks) + max_new_tokens))
        g = og.Generator(model, p)
        g.append_tokens(toks)
        out = []
        while not g.is_done():
            g.generate_next_token()
            out.append(g.get_next_tokens()[0])
        return tok.decode(out)

    return gen


def eval_medmcqa(gen, n):
    """Exact-match accuracy. The model emits one letter, so string match IS
    semantic match here. `unparsed` counts completions with no A/B/C/D at all --
    if that number is high, the model is failing to follow the output format and
    the accuracy number is measuring formatting, not medical knowledge."""
    from datasets import load_dataset
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")

    # Drop malformed rows BEFORE taking the first n, matching prepare_data.py's
    # filter. Without this the first n here is a different question set than the
    # first n of a prepared val.jsonl, and the scores are not comparable.
    def usable(ex):
        return (ex.get("question") and ex.get("cop") in (0, 1, 2, 3)
                and all(ex.get(k) for k in ("opa", "opb", "opc", "opd")))

    ds = [e for e in ds if usable(e)][:n]
    correct = 0
    unparsed = 0
    for i, ex in enumerate(ds):
        prompt = MED_PROMPT.format(q=ex["question"], a=ex["opa"], b=ex["opb"],
                                   c=ex["opc"], d=ex["opd"])
        pred = parse_letter(gen(prompt, 8))
        if pred is None:
            unparsed += 1
        correct += (pred == LETTERS[ex["cop"]])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ds)}  running acc={correct/(i+1):.3f}", flush=True)
    print(f"  unparsed (no A/B/C/D in output): {unparsed}/{len(ds)}")
    return correct, len(ds)


def _run_sql(db_path, sql):
    """Execute and return a canonical, order-insensitive form of the result.

    Sorted list rather than a set: duplicate rows are meaningful in SQL, so a
    query that returns a row twice is not equivalent to one that returns it once.
    Order IS discarded, because Spider gold queries rarely pin an ORDER BY and a
    differently-ordered but otherwise identical result is still correct.
    Returns None if the SQL is invalid -- that counts as wrong.
    """
    try:
        con = sqlite3.connect(db_path)
        # Some Spider databases contain undecodable bytes; without this they raise
        # and a correct query would be scored wrong.
        con.text_factory = lambda b: b.decode(errors="ignore")
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        con.close()
        return sorted(str(r) for r in rows)
    except Exception:
        return None


def eval_spider(gen, spider_dir, n, shuffle=False, seed=42):
    """Execution accuracy on Spider dev.

    Examples whose GOLD query fails to execute are skipped entirely rather than
    counted against the model -- they are dataset defects, not model errors, and
    counting them depresses both passes by an arbitrary amount. The denominator is
    therefore what was actually scored, which is what gets reported.

    --shuffle takes a fixed-seed random sample instead of the first n. Spider dev is
    ordered by database, so the first n covers only a handful of schemas; a sample
    is more representative. Both passes use the same seed, so they see identical
    questions either way.
    """
    import json
    with open(os.path.join(spider_dir, "dev.json"), encoding="utf-8") as f:
        data = json.load(f)
    if shuffle:
        import random
        random.Random(seed).shuffle(data)
    data = data[:n] if n > 0 else data
    db_root = os.path.join(spider_dir, "database")

    def schema_for(db_id):
        con = sqlite3.connect(os.path.join(db_root, db_id, db_id + ".sqlite"))
        cur = con.cursor()
        cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql NOT NULL")
        ddl = "\n".join(r[0] for r in cur.fetchall())
        con.close()
        return ddl

    correct = scored = 0
    for ex in data:
        db_id = ex["db_id"]
        db_path = os.path.join(db_root, db_id, db_id + ".sqlite")
        if not os.path.exists(db_path):
            continue
        gold_rows = _run_sql(db_path, ex["query"])
        if gold_rows is None:        # broken gold query -- not the model's fault
            continue
        scored += 1
        prompt = SQL_PROMPT.format(schema=schema_for(db_id).strip(),
                                   question=ex["question"].strip())
        pred_sql = clean_sql(gen(prompt, 200))
        correct += (_run_sql(db_path, pred_sql) == gold_rows)
        if scored % 25 == 0:
            print(f"  {scored}/{len(data)}  running acc={correct/scored:.3f}", flush=True)
    return correct, scored


def main():
    ap = argparse.ArgumentParser(
        description="Base vs LoRA accuracy on a GPU (transformers) or the Ryzen AI NPU (OGA).")
    ap.add_argument("--npu", action="store_true",
                    help="run the compiled model on the Ryzen AI NPU "
                         "(default: quantized base on GPU)")
    ap.add_argument("--task", choices=["medmcqa", "spider"], required=True)
    ap.add_argument("--mode", choices=["base", "lora"], required=True)
    ap.add_argument("-n", type=int, default=300, help="number of examples to score")
    ap.add_argument("--spider", help="Spider dataset dir (for --task spider)")
    ap.add_argument("--shuffle", action="store_true",
                    help="score a fixed-seed random sample of Spider dev instead of "
                         "the first n (used for the published NPU numbers)")

    g = ap.add_argument_group("GPU (default)")
    g.add_argument("--skeleton", default="Qwen/Qwen3-4B",
                   help="unquantized HF repo supplying the model structure")
    g.add_argument("--quark-model", help="Quark INT4 export dir")
    g.add_argument("--adapter", help="LoRA adapter dir (for --mode lora)")

    g = ap.add_argument_group("NPU (--npu)")
    g.add_argument("--npu-model", help="compiled NPU model dir (contains genai_config.json)")
    g.add_argument("--adapter-name",
                   help="adapter name registered in the compiled model (for --mode lora)")
    g.add_argument("--dll", default="onnxruntime_providers_ryzenai.dll",
                   help="Ryzen AI EP library used to switch adapters")
    args = ap.parse_args()

    if args.task == "spider" and not args.spider:
        ap.error("--spider <dir> is required for --task spider")

    if args.npu:
        if not args.npu_model:
            ap.error("--npu-model is required with --npu")
        if args.mode == "lora" and not args.adapter_name:
            ap.error("--adapter-name is required for --mode lora with --npu")
        gen = load_npu_model(args.npu_model, args.mode, args.adapter_name, args.dll)
    else:
        if not args.quark_model:
            ap.error("--quark-model is required when running on GPU (or pass --npu)")
        if args.mode == "lora" and not args.adapter:
            ap.error("--adapter is required for --mode lora on GPU")
        gen = load_gpu_model(args.skeleton, args.quark_model, args.mode, args.adapter)

    if args.task == "medmcqa":
        correct, total = eval_medmcqa(gen, args.n)
    else:
        correct, total = eval_spider(gen, args.spider, args.n, args.shuffle)

    acc = correct / total
    metric = "exact-match" if args.task == "medmcqa" else "execution"
    device = "NPU" if args.npu else "GPU"
    print(f"\n{args.task} {args.mode} ({device}) {metric} accuracy: "
          f"{correct}/{total} = {acc:.3f}")


if __name__ == "__main__":
    main()
