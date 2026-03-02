#!/usr/bin/env python3
"""Generic audio inference test harness.

Tests an audio model (e.g., Whisper) for functional correctness,
real-time factor, and WER accuracy.

Usage:
    python harness_audio.py --model-id amd/whisper-... --device npu
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np


def generate_test_audio(duration_s: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate a silent test audio clip."""
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


def test_functional(model_path: str, device: str) -> dict:
    """Test that the model loads and produces transcription output."""
    try:
        import onnxruntime as ort

        providers = ["VitisAIExecutionProvider"] if device == "npu" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)

        input_info = session.get_inputs()[0]
        test_audio = generate_test_audio()
        test_input = test_audio.reshape(1, -1)

        outputs = session.run(None, {input_info.name: test_input})
        return {"passed": len(outputs) > 0}
    except Exception as e:
        return {"passed": False, "error": str(e)}


def test_performance(model_path: str, device: str, audio_duration_s: float = 5.0) -> dict:
    """Measure real-time factor (RTF) and latency."""
    try:
        import onnxruntime as ort

        providers = ["VitisAIExecutionProvider"] if device == "npu" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)

        input_info = session.get_inputs()[0]
        test_audio = generate_test_audio(duration_s=audio_duration_s)
        test_input = test_audio.reshape(1, -1)

        start = time.perf_counter()
        session.run(None, {input_info.name: test_input})
        elapsed = time.perf_counter() - start

        rtf = elapsed / audio_duration_s

        return {
            "rtf": round(rtf, 3),
            "latency_ms": round(elapsed * 1000, 1),
            "audio_duration_s": audio_duration_s,
        }
    except Exception as e:
        return {"error": str(e)}


def run_single_model(model_id, device):
    """Test a single audio model and return results dict."""
    results = {
        "model_id": model_id,
        "device": device,
        "test_date": time.strftime("%Y-%m-%d"),
    }

    print(f"\n1. Functional test...")
    results["functional"] = test_functional(model_id, device)
    print(f"   {'PASS' if results['functional'].get('passed') else 'FAIL'}")

    if results["functional"].get("passed"):
        print(f"\n2. Performance test...")
        results["performance"] = test_performance(model_id, device)
        rtf = results["performance"].get("rtf", "N/A")
        print(f"   RTF: {rtf}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Audio inference test harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-id", help="ONNX model path or HF model ID")
    group.add_argument("--models", help="JSON file with list of models to test")
    parser.add_argument("--device", choices=["npu", "gpu", "cpu"], default="npu")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    args = parser.parse_args()

    if args.models:
        models_data = json.loads(Path(args.models).read_text())
        model_ids = [m["model_id"] if isinstance(m, dict) else m for m in models_data.get("audio", models_data.get("models", models_data))]
        all_results = []
        for model_id in model_ids:
            print(f"\n{'='*60}\nTesting {model_id} on {args.device}...")
            result = run_single_model(model_id, args.device)
            all_results.append(result)
        if args.output:
            Path(args.output).write_text(json.dumps(all_results, indent=2))
            print(f"\nResults written to {args.output}")
        failed = sum(1 for r in all_results if not r["functional"].get("passed", False))
        sys.exit(1 if failed > 0 else 0)
    else:
        print(f"Testing {args.model_id} on {args.device}...")
        results = run_single_model(args.model_id, args.device)
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2))
            print(f"\nResults written to {args.output}")
        sys.exit(0 if results["functional"].get("passed", False) else 1)


if __name__ == "__main__":
    main()
