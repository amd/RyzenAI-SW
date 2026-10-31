# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Build prompt/response JSONL for LoRA fine-tuning.

Produces a ``train.jsonl`` of ``{"prompt": ..., "response": ...}`` records that
finetune_on_quark.py consumes directly. Two example tasks are supported:

  medmcqa : medical multiple-choice (downloaded from Hugging Face). Response is the
            correct option letter (A/B/C/D).
  spider  : text-to-SQL. Requires the Spider dataset downloaded locally
            (https://yale-lily.github.io/spider); response is the target SQL query.

Example:
    python prepare_data.py --task medmcqa --out ./data_medmcqa --max-train 8000
    python prepare_data.py --task spider  --out ./data_spider --spider ./spider_data
"""
import argparse
import json
import os
import random

LETTERS = ["A", "B", "C", "D"]

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


def dump(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[wrote] {path}  ({len(rows)} rows)")


def build_medmcqa(out, max_train, seed):
    from datasets import load_dataset
    train = load_dataset("openlifescienceai/medmcqa", split="train")
    dev = load_dataset("openlifescienceai/medmcqa", split="validation")

    def rec(ex):
        return {"prompt": MED_PROMPT.format(q=ex["question"], a=ex["opa"],
                                            b=ex["opb"], c=ex["opc"], d=ex["opd"]),
                "response": " " + LETTERS[ex["cop"]]}

    def ok(ex):
        return (ex.get("question") and ex.get("cop") in (0, 1, 2, 3)
                and all(ex.get(k) for k in ("opa", "opb", "opc", "opd")))

    tr = [rec(e) for e in train if ok(e)]
    random.Random(seed).shuffle(tr)
    tr = tr[:max_train]
    dv = [rec(e) for e in dev if ok(e)]
    dump(tr, os.path.join(out, "train.jsonl"))
    dump(dv, os.path.join(out, "val.jsonl"))


def build_spider(out, spider_dir, max_train, seed):
    import sqlite3
    train_json = os.path.join(spider_dir, "train_spider.json")
    dev_json = os.path.join(spider_dir, "dev.json")
    db_root = os.path.join(spider_dir, "database")

    def schema_for(db_id):
        con = sqlite3.connect(os.path.join(db_root, db_id, db_id + ".sqlite"))
        cur = con.cursor()
        cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql NOT NULL")
        ddl = "\n".join(r[0] for r in cur.fetchall())
        con.close()
        return ddl

    def build(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        cache = {}
        for ex in data:
            db_id = ex["db_id"]
            if db_id not in cache:
                try:
                    cache[db_id] = schema_for(db_id)
                except Exception:
                    cache[db_id] = ""
            rows.append({"prompt": SQL_PROMPT.format(schema=cache[db_id],
                                                     question=ex["question"]),
                         "response": " " + ex["query"].strip()})
        return rows

    tr = build(train_json)
    random.Random(seed).shuffle(tr)
    tr = tr[:max_train]
    dv = build(dev_json)
    dump(tr, os.path.join(out, "train.jsonl"))
    dump(dv, os.path.join(out, "val.jsonl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["medmcqa", "spider"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spider", help="Spider dataset dir (for --task spider)")
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.task == "medmcqa":
        build_medmcqa(args.out, args.max_train, args.seed)
    else:
        if not args.spider:
            ap.error("--spider <dir> is required for --task spider")
        build_spider(args.out, args.spider, args.max_train, args.seed)
    print("done.")


if __name__ == "__main__":
    main()
