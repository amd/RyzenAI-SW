#!/usr/bin/env python3
"""
Discover candidate models to test on AMD hardware.
Outputs a JSON mapping domains (llm, vision, audio) to model IDs.

Model IDs are AMD-specific ONNX models from the actual RyzenAI-SW repo
and HuggingFace collections. Only models that actually exist in the AMD
ecosystem are listed here.
"""

import argparse
import json
from pathlib import Path

# AMD ONNX models that exist on HuggingFace and are referenced in the
# RyzenAI-SW repo examples and documentation.
LLM_CANDIDATES = [
    # From LLM-examples/oga_api (actually used in repo)
    "amd/Llama-2-7b-chat-hf-awq-g128-int4-asym-fp16-onnx-hybrid",
    # From Phi-3.5-mini-instruct-onnx-ryzenai-npu/ directory in repo
    "amd/Phi-3.5-mini-instruct-onnx-ryzenai-npu",
    # From LLM-examples/RAG-OGA
    "amd/Llama-3.2-3B-Instruct-onnx-ryzenai-1.7-hybrid",
    # From LLM-examples/VLM
    "amd/Gemma-3-4b-it-mm-onnx-ryzenai-npu",
    # From LLM-examples/oga_inference
    "amd/gpt-oss-20b-onnx-ryzenai-npu",
]

# Vision models from CNN-examples/ in the repo
VISION_CANDIDATES = [
    # From CNN-examples/getting_started_resnet
    "resnet50-int8",
    "resnet50-bf16",
    # From CNN-examples/object_detection/yolov8m
    "yolov8m-bf16",
    # From examples/super-resolution
    "real-esrgan",
    "sesr-m7",
]

# Audio models from Transformer-examples/ASR and Demos/ASR
AUDIO_CANDIDATES = [
    # From Demos/ASR/Whisper (actual HF model IDs)
    "amd/whisper-tiny-onnx-npu",
    "amd/whisper-base-onnx-npu",
    "amd/whisper-small-onnx-npu",
]


def load_config(config_path: Path) -> dict:
    """Load hardware registry config if it exists."""
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def discover_models(config_path: Path) -> dict:
    """
    Discover candidate models per domain.
    Config is loaded for future hardware-specific filtering.
    """
    _ = load_config(config_path)
    return {
        "llm": LLM_CANDIDATES,
        "vision": VISION_CANDIDATES,
        "audio": AUDIO_CANDIDATES,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover candidate models to test on AMD hardware."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("hardware-registry.json"),
        help="Path to hardware-registry.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model-candidates.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    candidates = discover_models(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    print(f"Wrote {len(candidates['llm'])} LLM, {len(candidates['vision'])} vision, "
          f"{len(candidates['audio'])} audio candidates to {args.output}")


if __name__ == "__main__":
    main()
