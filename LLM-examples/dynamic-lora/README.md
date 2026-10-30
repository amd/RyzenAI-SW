# Dynamic LoRA on Ryzen AI

Fine-tune lightweight **LoRA adapters** for domain tasks and run them on the **AMD
Ryzen AI NPU** with **dynamic adapter switching**: one frozen INT4-quantized base
model, multiple small adapters loaded and switched at runtime.

This example trains **two** adapters on the **same** base and switches between them
live on the NPU:

- **Text-to-SQL**, evaluated by execution accuracy on the [Spider](https://yale-lily.github.io/spider) benchmark.
- **Medical multiple-choice**, evaluated by exact-match accuracy on the [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) benchmark.

Two principles make it work:

1. **Train each LoRA against the *quantized* base**, not the FP model. An adapter
   trained on FP and applied to the INT4 model afterward loses its effect on-device.
2. **Match the training data to the target distribution.** Each adapter is a
   specialist; train on data resembling what you evaluate on.

## Pre-built models (Hugging Face)

To skip the quantization and fine-tuning stages, download the base model and adapters,
then build the NPU model for your Ryzen AI Software version (Step 3):

| Artifact | Hugging Face |
|---|---|
| Quark INT4 base | [amd/qwen-3-4B-quark](https://huggingface.co/amd/qwen-3-4B-quark) |
| Fine-tuned LoRA adapter (Spider SQL) | [amd/qwen-3-4B-spider-sql-lora-adapter](https://huggingface.co/amd/qwen-3-4B-spider-sql-lora-adapter) |
| Fine-tuned LoRA adapter (MedMCQA) | [amd/qwen-3-4B-medmcqa-lora-adapter](https://huggingface.co/amd/qwen-3-4B-medmcqa-lora-adapter) |

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Build `train.jsonl` / `val.jsonl` from MedMCQA or Spider |
| `finetune_on_quark.py` | LoRA fine-tune against the frozen Quark INT4 base |
| `eval_accuracy.py` | Evaluate base vs LoRA accuracy, on GPU or on the NPU (MedMCQA / Spider) |
| `run_lora_switch.py` | Run on the NPU; activate/switch adapters at runtime |
| `requirements.txt` | Training / data-prep dependencies |

## Machines

- A **Linux machine with an AMD (Instinct MI series) or NVIDIA GPU** for quantization
  and fine-tuning (Steps 1-2).
- A **Windows Ryzen AI PC** (Ryzen AI 300-series) for NPU compilation and inference
  (Steps 3-4).

## Step 1: Quantize the base (AMD Quark, INT4 AWQ)

Quantize Qwen3-4B to INT4 (AWQ) with AMD Quark. The `quantize_quark.py` script and its
dependencies come from the AMD Quark LLM PTQ examples. Follow the setup and
quantization steps in the Ryzen AI Software documentation:
[Preparing OGA Models - Quantization](https://ryzenai.docs.amd.com/en/latest/oga_model_prepare.html#quantization).

The same quantized base is reused for both adapters, so this step is done once. Run
from the Quark LLM PTQ example directory:

```
python quantize_quark.py \
    --model_dir "Qwen/Qwen3-4B" \
    --output_dir ./qwen3-4b-quark-int4 \
    --quant_scheme uint4_wo_128 \
    --quant_algo awq \
    --quant_algo_config_file awq ./qwen3_awq_config.json \
    --dataset pileval_for_awq_benchmark \
    --model_export hf_format \
    --data_type bfloat16 --exclude_layers []
```

> **Note on the AWQ config for Qwen3:** Qwen3 is not in the Quark 0.11 built-in AWQ
> registry, so `--quant_algo_config_file awq` points to the custom AWQ configuration
> `qwen3_awq_config.json` supplied in this folder (adapted from the Qwen2 template with
> the decoder-layer path set to `model.layers`). Copy it next to `quantize_quark.py`,
> or pass its full path. If the base model you use is already in Quark's AWQ registry,
> this custom config is not needed.

## Step 2: Prepare data + fine-tune the adapters

Install dependencies (`pip install -r requirements.txt`) plus PyTorch for your GPU
and the AMD Quark wheel. Then prepare data and train one adapter per task. The base
stays INT4 (loaded with Quark's `import_model_from_safetensors`); only the adapter is
trained and saved unmerged.

Quark handles dequantization on the fly. `import_model_from_safetensors` restores each
quantized layer as a `QParamsLinear` (an `nn.Linear` subclass, so PEFT targets it
normally); the weights stay INT4 and are dequantized to bfloat16 inside each forward
pass, so the adapter trains against the same numerics the NPU will run.

LoRA configuration used in this example:

| Parameter | Value | | Parameter | Value |
|---|---|---|---|---|
| rank (r) | 16 | | dropout | 0.05 |
| alpha | 32 | | epochs | 5 |
| target modules | q, k, v, o, gate, up, down | | | |

**Task A - Text-to-SQL (Spider):** download Spider to `./spider_data` first
(https://yale-lily.github.io/spider).

```
python prepare_data.py --task spider --out ./data_spider --spider ./spider_data
python finetune_on_quark.py \
    --quark-model ./qwen3-4b-quark-int4 \
    --train-file ./data_spider/train.jsonl \
    --adapter-out ./adapters/sql_spider
```

**Task B - Medical MCQA (MedMCQA):** downloaded automatically from Hugging Face.

```
python prepare_data.py --task medmcqa --out ./data_medmcqa --max-train 8000
python finetune_on_quark.py \
    --quark-model ./qwen3-4b-quark-int4 \
    --train-file ./data_medmcqa/train.jsonl \
    --adapter-out ./adapters/med_mcqa
```

Each run produces an unmerged adapter directory (`adapter_model.safetensors` +
`adapter_config.json`).

### Optional: evaluate base vs LoRA on GPU

Verify the adapter helps before spending time on NPU compilation. Run base and LoRA as
two passes over the same questions and compare the accuracy:

```
# medical (exact-match accuracy on MedMCQA validation)
python eval_accuracy.py --task medmcqa --quark-model ./qwen3-4b-quark-int4 --mode base -n 300
python eval_accuracy.py --task medmcqa --quark-model ./qwen3-4b-quark-int4 --mode lora --adapter ./adapters/med_mcqa -n 300

# text-to-SQL (execution accuracy on Spider dev)
python eval_accuracy.py --task spider --quark-model ./qwen3-4b-quark-int4 --mode base --spider ./spider_data -n 100
python eval_accuracy.py --task spider --quark-model ./qwen3-4b-quark-int4 --mode lora --adapter ./adapters/sql_spider --spider ./spider_data -n 100
```

The same script evaluates the compiled NPU model with `--npu`; see
[Step 4](#optional-evaluate-the-npu-model).

### Reference results (GPU)

Accuracy of the INT4 base with and without each adapter. Within a row the two passes
score the *same* questions, so the uplift is meaningful; `n` differs between rows.

| Task | n | Metric | Base | + LoRA |
|---|---|---|---|---|
| Text-to-SQL (Spider) | 100 | execution accuracy | 46.0% | **63.0%** |
| Medical MCQA (MedMCQA) | 300 | exact-match accuracy | 45.7% | **51.3%** |

Notes on the metrics:

- **MedMCQA is 4-way multiple choice, so 25% is chance.** The base model is already
  well above chance; the adapter moves it further, it does not teach it the task from
  scratch. The script also reports how many completions contained no A/B/C/D at all.
  That count was 0 in our runs, so the gain is not the adapter merely teaching the
  model to emit a letter.
- **Spider is scored by execution accuracy, not string match.** Predicted and gold SQL
  are both run against the real SQLite database and the returned rows compared, so a
  correct query that differs in formatting, aliasing, or join order still counts as
  correct. Invalid SQL counts as wrong.

## Step 3: Compile the model + adapters for the NPU (Windows Ryzen AI PC)

Run these commands in the Ryzen AI environment created by the installer:

```
"C:\Program Files\RyzenAI\ryzen-ai-<version>\Scripts\activate.bat"
```

`model_generate` is available in that environment. It handles the model conversion
internally, so pass the quantized model directory directly as `--input`. Generate each
adapter into its own output directory:

```
model_generate --npu --token_fusion --mem_optimize ^
    --output .\qwen3-4b-npu ^
    --input .\qwen3-4b-quark-int4 ^
    --extra_options lora lora_adapter=.\adapters\sql_spider\adapter_model.safetensors lora_name=sql_spider lora_scaling_factor=2.0

model_generate --npu --token_fusion --mem_optimize ^
    --output .\qwen3-4b-npu-med ^
    --input .\qwen3-4b-quark-int4 ^
    --extra_options lora lora_adapter=.\adapters\med_mcqa\adapter_model.safetensors lora_name=med_mcqa lora_scaling_factor=2.0
```

> **The LoRA scale is baked in at compile time.** There is no runtime scale dial.
> `lora_scaling_factor` must equal the adapter's `alpha / r`, which is `32 / 16 = 2.0`
> for the configuration above. If you change `--rank` or `--alpha` during training,
> change this to match or the adapter will be applied at the wrong strength.

`--mem_optimize` is optional; it reduces peak memory during compilation.

Each run writes `<lora_name>_prefill.bin`, `<lora_name>_prefill.pb.bin`,
`<lora_name>_token.bin`, and `<lora_name>_token.pb.bin`. To serve both adapters from
one base, copy the second adapter's files into the first model directory:

```
copy .\qwen3-4b-npu-med\med_mcqa_*.* .\qwen3-4b-npu\
```

The runtime loads adapters by name from the model directory, so `qwen3-4b-npu` now
serves both adapters from a single base.

Check that the tokenizer files (`tokenizer.json`, `tokenizer_config.json`, and the
rest) are present in `qwen3-4b-npu`. Loading the model fails without them. They are
normally written by `model_generate`; if the directory was assembled by hand, copy
them from the quantized model directory.

## Step 4: Run with dynamic adapter switching

In the same Ryzen AI conda environment, run from the folder that has the copied
Ryzen AI runtime DLLs.

```
# switch between the two adapters on one loaded base
python run_lora_switch.py -m .\qwen3-4b-npu --switch sql_spider med_mcqa

# a single adapter (or the base model)
python run_lora_switch.py -m .\qwen3-4b-npu --adapter sql_spider --task sql
python run_lora_switch.py -m .\qwen3-4b-npu --adapter base --task sql
```

The SQL adapter produces a SQL query and the medical adapter produces an answer
letter, from the same loaded base, selected purely by which adapter is active:

```
[sql_spider] SELECT T1.language FROM countrylanguage AS T1 JOIN country AS T2 ON ... WHERE T2.continent = 'Asia'
[med_mcqa]   B
```

### Optional: evaluate the NPU model

`eval_accuracy.py --npu` scores the compiled model on the NPU, reusing the same
prompts and scoring logic as the GPU path.

Both benchmarks must be available on the Ryzen AI PC, not just on the training box:
copy (or re-download) `spider_data` there for `--task spider`, and `pip install
datasets` for `--task medmcqa`, which pulls MedMCQA from Hugging Face on first run and
therefore needs network access. Neither is required for Steps 3-4 themselves, only
for scoring.

```
python eval_accuracy.py --npu --task spider --npu-model .\qwen3-4b-npu --mode base --spider .\spider_data -n 200 --shuffle
python eval_accuracy.py --npu --task spider --npu-model .\qwen3-4b-npu --mode lora --adapter-name sql_spider --spider .\spider_data -n 200 --shuffle

python eval_accuracy.py --npu --task medmcqa --npu-model .\qwen3-4b-npu --mode base -n 200
python eval_accuracy.py --npu --task medmcqa --npu-model .\qwen3-4b-npu --mode lora --adapter-name med_mcqa -n 200
```

Results from the commands above:

| Task | n | Metric | Base | + LoRA |
|---|---|---|---|---|
| Text-to-SQL (Spider) | 200 | execution accuracy | 47.5% | **63.0%** |
| Medical MCQA (MedMCQA) | 200 | exact-match accuracy | 42.0% | **53.0%** |

The adapter is activated once at startup and left alone for the whole run, so the
measurement is not affected by switching. `--mode base` reverts to the unmodified base
with `set_active_adapter("base")`.

`--shuffle` scores a fixed-seed random sample of Spider dev rather than the first `n`.
Spider dev is ordered by database, so the first 200 questions cover only a few schemas;
the sample is more representative. Both passes use the same seed and therefore see
identical questions. The figures above were produced with `--shuffle`, so include it to
reproduce them.
