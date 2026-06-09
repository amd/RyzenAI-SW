#!/usr/bin/env python3
"""
Regenerate the Ryzen AI model tables from the official Hugging Face collections.

Writes the block between the MODELS_TABLE markers in
docs/models-tutorials/index.mdx:
  - LLMs: combined Hybrid NPU/GPU, NPU 4K, NPU 16K, and NPU-only LFM2 (ONNX)
    variants, keyed by model name; the model name links to Hugging Face.
  - Vision Models: Stable Diffusion image-generation models (capabilities from
    the amd/sd-sandbox project), incl. SD3 / SD3.5.
  - Audio Models: Whisper + Parakeet ASR models.

Live collections are unioned with small curated supplements (SD3/3.5 capability
metadata, audio models) that aren't expressible from collection metadata alone.

Usage:
  python .github/scripts/fetch_models.py
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

VERSION = "1.7.1"
INDEX = Path(__file__).resolve().parents[2] / "docs" / "models-tutorials" / "index.mdx"
START = "{/* MODELS_TABLE_START"
END = "{/* MODELS_TABLE_END */}"

# (column header, collection slug, id suffix to strip, short cell label)
LLM_COLLECTIONS = [
    ("Hybrid NPU / GPU", "amd/ryzen-ai-171-hybrid", "_rai_1.7.1_hybrid", "Hybrid"),
    ("NPU 4K", "amd/ryzen-ai-171-npu-4k", "_rai_1.7.1_npu_4K", "4K"),
    ("NPU 16K", "amd/ryzen-ai-171-npu-16k", "_rai_1.7.1_npu_16K", "16K"),
]
LFM2 = ("amd/ryzen-ai-171-npu-lfm2-models", "_rai_1.7.1")  # folded into the LLM table
SD_COLLECTION = "amd/ryzen-ai-171-sd-models"

# Curated Vision (Stable Diffusion) capabilities, keyed by HF id.
# Source: amd/sd-sandbox "Supported Models" table.
VISION_META = [
    # (hf_id, display, resolution, t2i, i2i, controlnet)
    ("amd/stable-diffusion-1.5-amdnpu", "SD 1.5", "512x512", True, False, False),
    ("amd/sd-turbo-amdnpu", "SD Turbo", "512x512", True, False, False),
    ("amd/sdxl-base-amdnpu", "SDXL Base", "1024x1024", True, True, False),
    ("amd/sdxl-turbo-amdnpu", "SDXL Turbo", "512x512", True, False, False),
    ("amd/segmind-vega-amdnpu", "Segmind-Vega", "1024x1024", True, True, False),
    ("amd/stable-diffusion-3-medium-amdnpu", "SD3 Medium", "512-1024", True, True, True),
    ("amd/stable-diffusion-3.5-medium-amdnpu", "SD3.5 Medium", "512-1024", True, True, True),
]

# Curated Audio (ASR) models. NOTE: confirm the Parakeet repo id for your release.
AUDIO = [
    # (display, hf_id, params, task)
    ("whisper-base", "amd/whisper-base-onnx-npu", "74M", "Speech-to-text (ASR)"),
    ("whisper-small", "amd/whisper-small-onnx-npu", "244M", "Speech-to-text (ASR)"),
    ("whisper-medium", "amd/whisper-medium-onnx-npu", "769M", "Speech-to-text (ASR)"),
    ("whisper-large-v3-turbo", "amd/whisper-large-v3-turbo-onnx-npu", "809M", "Speech-to-text (ASR)"),
    ("Parakeet-TDT-0.6B", "nvidia/parakeet-tdt-0.6b-v2", "0.6B", "Speech-to-text (ASR)"),
]


def fetch(slug: str) -> list[str]:
    url = f"https://huggingface.co/api/collections/{slug}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [it["id"] for it in data.get("items", []) if it.get("type") == "model"]
    except Exception as e:  # noqa: BLE001
        print(f"WARN: could not fetch {slug}: {e}", file=sys.stderr)
        return []


def hf(idv: str) -> str:
    return f"https://huggingface.co/{idv}"


def base_name(model_id: str, suffix: str) -> str:
    name = model_id.split("/", 1)[1] if "/" in model_id else model_id
    if suffix and name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def yes(flag: bool) -> str:
    return "Yes" if flag else ""


def llm_table() -> str:
    # base name -> {label: model_id}
    rows: dict[str, dict[str, str]] = {}
    for _h, slug, suffix, label in LLM_COLLECTIONS:
        for mid in fetch(slug):
            rows.setdefault(base_name(mid, suffix), {})[label] = mid
    # Fold LFM2 in as the "NPU (ONNX)" column.
    lfm2_slug, lfm2_suffix = LFM2
    for mid in fetch(lfm2_slug):
        rows.setdefault(base_name(mid, lfm2_suffix), {})["ONNX"] = mid

    labels = [l for _h, _s, _x, l in LLM_COLLECTIONS] + ["ONNX"]
    headers = ["Model"] + [h for h, _s, _x, _l in LLM_COLLECTIONS] + ["NPU (ONNX)"]
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for base in sorted(rows, key=str.lower):
        variants = rows[base]
        # Model name links to the first available variant repo.
        first = next((variants[l] for l in labels if l in variants), None)
        name_cell = f"[{base}]({hf(first)})" if first else base
        cells = [name_cell]
        for label in labels:
            mid = variants.get(label)
            tag = {"Hybrid": "Hybrid", "4K": "4K", "16K": "16K", "ONNX": "ONNX"}[label]
            cells.append(f"[{tag}]({hf(mid)})" if mid else "")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def vision_table() -> str:
    known = {hf_id for hf_id, *_ in VISION_META}
    out = ["| Model | Output Resolution | Text-to-Image | Image-to-Image | ControlNet |",
           "| --- | --- | --- | --- | --- |"]
    for hf_id, display, res, t2i, i2i, cn in VISION_META:
        out.append(f"| [{display}]({hf(hf_id)}) | {res} | {yes(t2i)} | {yes(i2i)} | {yes(cn)} |")
    # Append any collection models not already covered by the curated metadata.
    for mid in sorted(fetch(SD_COLLECTION), key=str.lower):
        if mid not in known:
            out.append(f"| [{base_name(mid, '')}]({hf(mid)}) |  | Yes |  |  |")
    return "\n".join(out)


def audio_table() -> str:
    out = ["| Model | Parameters | Task |", "| --- | --- | --- |"]
    for display, hf_id, params, task in AUDIO:
        out.append(f"| [{display}]({hf(hf_id)}) | {params} | {task} |")
    return "\n".join(out)


def build_block() -> str:
    return f"""{START} - generated by .github/scripts/fetch_models.py; do not edit by hand */}}

## LLMs

Hybrid (NPU + GPU), NPU 4K, NPU 16K, and NPU-only LFM2 (ONNX) variants share model names where they overlap. A link in a column means that variant exists; the model name links to Hugging Face.

{llm_table()}

## Vision Models

Stable Diffusion image-generation models for the AMD NPU. Capabilities follow the [amd/sd-sandbox](https://github.com/amd/sd-sandbox) project.

{vision_table()}

## Audio Models

Speech-to-text (ASR) models for the AMD NPU.

{audio_table()}

{END}"""


def main() -> None:
    text = INDEX.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        print(f"ERROR: markers not found in {INDEX}", file=sys.stderr)
        sys.exit(1)
    new = pattern.sub(lambda _m: build_block(), text)
    INDEX.write_text(new, encoding="utf-8")
    print(f"Updated model tables block in {INDEX}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
