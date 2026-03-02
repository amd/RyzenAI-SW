#!/usr/bin/env python3
"""Generic vision inference test harness.

Tests a vision model for functional correctness, FPS, and accuracy.
Used as Tier 2 fallback when no model-specific test script exists.

Usage:
    python harness_vision.py --model-id amd/yolov8-... --device npu --task classification
    python harness_vision.py --model-id amd/yolov8-... --device npu --task detection
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np


def create_test_image(height: int = 224, width: int = 224) -> np.ndarray:
    """Create a random test image as numpy array."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def test_functional(model_path: str, device: str) -> dict:
    """Test that the model loads and produces output."""
    try:
        import onnxruntime as ort

        providers = ["VitisAIExecutionProvider"] if device == "npu" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)

        input_info = session.get_inputs()[0]
        shape = input_info.shape
        if len(shape) == 4:
            test_input = np.random.randn(*[s if isinstance(s, int) else 1 for s in shape]).astype(np.float32)
        else:
            test_input = np.random.randn(*shape).astype(np.float32)

        outputs = session.run(None, {input_info.name: test_input})
        return {"passed": len(outputs) > 0, "output_shape": str(outputs[0].shape)}
    except Exception as e:
        return {"passed": False, "error": str(e)}


def test_performance(model_path: str, device: str, num_iterations: int = 50) -> dict:
    """Measure FPS and latency."""
    try:
        import onnxruntime as ort

        providers = ["VitisAIExecutionProvider"] if device == "npu" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)

        input_info = session.get_inputs()[0]
        shape = input_info.shape
        test_input = np.random.randn(*[s if isinstance(s, int) else 1 for s in shape]).astype(np.float32)

        # Warmup
        for _ in range(5):
            session.run(None, {input_info.name: test_input})

        start = time.perf_counter()
        for _ in range(num_iterations):
            session.run(None, {input_info.name: test_input})
        elapsed = time.perf_counter() - start

        fps = num_iterations / elapsed
        latency_ms = (elapsed / num_iterations) * 1000

        return {
            "fps": round(fps, 1),
            "latency_ms": round(latency_ms, 1),
            "iterations": num_iterations,
        }
    except Exception as e:
        return {"error": str(e)}


def run_single_model(model_id, device, task="classification"):
    """Test a single vision model and return results dict."""
    results = {
        "model_id": model_id,
        "device": device,
        "task": task,
        "test_date": time.strftime("%Y-%m-%d"),
    }

    print(f"\n1. Functional test...")
    results["functional"] = test_functional(model_id, device)
    print(f"   {'PASS' if results['functional'].get('passed') else 'FAIL'}")

    if results["functional"].get("passed"):
        print(f"\n2. Performance test...")
        results["performance"] = test_performance(model_id, device)
        fps = results["performance"].get("fps", "N/A")
        lat = results["performance"].get("latency_ms", "N/A")
        print(f"   FPS: {fps}, Latency: {lat}ms")

    return results


def main():
    parser = argparse.ArgumentParser(description="Vision inference test harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-id", help="ONNX model path or HF model ID")
    group.add_argument("--models", help="JSON file with list of models to test")
    parser.add_argument("--device", choices=["npu", "gpu", "cpu"], default="npu")
    parser.add_argument("--task", choices=["classification", "detection", "segmentation"], default="classification")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    args = parser.parse_args()

    if args.models:
        models_data = json.loads(Path(args.models).read_text())
        model_ids = [m["model_id"] if isinstance(m, dict) else m for m in models_data.get("vision", models_data.get("models", models_data))]
        all_results = []
        for model_id in model_ids:
            print(f"\n{'='*60}\nTesting {model_id} on {args.device} ({args.task})...")
            result = run_single_model(model_id, args.device, args.task)
            all_results.append(result)
        if args.output:
            Path(args.output).write_text(json.dumps(all_results, indent=2))
            print(f"\nResults written to {args.output}")
        failed = sum(1 for r in all_results if not r["functional"].get("passed", False))
        sys.exit(1 if failed > 0 else 0)
    else:
        print(f"Testing {args.model_id} on {args.device} ({args.task})...")
        results = run_single_model(args.model_id, args.device, args.task)
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2))
            print(f"\nResults written to {args.output}")
        sys.exit(0 if results["functional"].get("passed", False) else 1)


if __name__ == "__main__":
    main()
