#!/usr/bin/env python3
"""Generic LLM inference test harness.

Tests a model for functional correctness, performance, and accuracy.
Used as Tier 2 fallback when no model-specific test script exists.

Usage:
    python harness_llm.py --model-id amd/Llama-3.2-1B-Instruct-... --device npu
    python harness_llm.py --model-id amd/Llama-3.2-1B-Instruct-... --device gpu
"""

import argparse
import json
import time
import sys


def test_functional(model_path: str, device: str) -> dict:
    """Test that the model loads and generates coherent output."""
    try:
        import onnxruntime_genai as og

        model = og.Model(model_path)
        tokenizer = og.Tokenizer(model)

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=64)

        prompt = "<|user|>\nSay hello in one sentence.\n<|assistant|>\n"
        input_tokens = tokenizer.encode(prompt)
        params.input_ids = input_tokens

        generator = og.Generator(model, params)
        output_tokens = []
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            output_tokens.append(generator.get_next_tokens()[0])

        output_text = tokenizer.decode(output_tokens)
        return {"passed": len(output_text) > 0, "output_preview": output_text[:200]}
    except Exception as e:
        return {"passed": False, "error": str(e)}


def test_performance(model_path: str, device: str, num_tokens: int = 128) -> dict:
    """Measure tokens/second and time to first token."""
    try:
        import onnxruntime_genai as og

        model = og.Model(model_path)
        tokenizer = og.Tokenizer(model)

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=num_tokens)

        prompt = "<|user|>\nExplain what a neural processing unit is.\n<|assistant|>\n"
        input_tokens = tokenizer.encode(prompt)
        params.input_ids = input_tokens

        generator = og.Generator(model, params)

        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0

        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            token_count += 1
            if first_token_time is None:
                first_token_time = time.perf_counter()

        end_time = time.perf_counter()
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        decode_tokens = token_count - 1
        decode_time = end_time - first_token_time if first_token_time else total_time

        return {
            "tokens_per_second_decode": round(decode_tokens / decode_time, 1) if decode_time > 0 else 0,
            "time_to_first_token_ms": round(ttft, 0),
            "total_tokens": token_count,
            "total_time_s": round(total_time, 2),
        }
    except Exception as e:
        return {"error": str(e)}


def test_accuracy(model_path: str, device: str) -> dict:
    """Run accuracy checks against reference prompts."""
    reference_prompts = [
        {"prompt": "What is 2+2?", "expected_contains": "4"},
        {"prompt": "What color is the sky?", "expected_contains": "blue"},
        {"prompt": "What is the capital of France?", "expected_contains": "Paris"},
    ]

    try:
        import onnxruntime_genai as og

        model = og.Model(model_path)
        tokenizer = og.Tokenizer(model)
        correct = 0

        for ref in reference_prompts:
            params = og.GeneratorParams(model)
            params.set_search_options(max_length=64)

            formatted = f"<|user|>\n{ref['prompt']}\n<|assistant|>\n"
            params.input_ids = tokenizer.encode(formatted)

            generator = og.Generator(model, params)
            tokens = []
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                tokens.append(generator.get_next_tokens()[0])

            output = tokenizer.decode(tokens).lower()
            if ref["expected_contains"].lower() in output:
                correct += 1

        return {
            "exact_match_reference": correct,
            "total_reference": len(reference_prompts),
        }
    except Exception as e:
        return {"error": str(e)}


def run_single_model(model_id, device, skip_accuracy=False):
    """Test a single model and return results dict."""
    results = {
        "model_id": model_id,
        "device": device,
        "test_date": time.strftime("%Y-%m-%d"),
    }

    print(f"\n1. Functional test...")
    results["functional"] = test_functional(model_id, device)
    print(f"   {'PASS' if results['functional'].get('passed') else 'FAIL'}")

    if not results["functional"].get("passed"):
        print("   Skipping performance and accuracy (functional test failed)")
        results["performance"] = {}
        results["accuracy"] = {}
    else:
        print(f"\n2. Performance test...")
        results["performance"] = test_performance(model_id, device)
        tps = results["performance"].get("tokens_per_second_decode", "N/A")
        ttft = results["performance"].get("time_to_first_token_ms", "N/A")
        print(f"   Tokens/s (decode): {tps}, TTFT: {ttft}ms")

        if not skip_accuracy:
            print(f"\n3. Accuracy test...")
            results["accuracy"] = test_accuracy(model_id, device)
            match = results["accuracy"].get("exact_match_reference", "N/A")
            total = results["accuracy"].get("total_reference", "N/A")
            print(f"   Reference match: {match}/{total}")

    return results


def main():
    parser = argparse.ArgumentParser(description="LLM inference test harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-id", help="HuggingFace model ID or local path")
    group.add_argument("--models", help="JSON file with list of models to test")
    parser.add_argument("--device", choices=["npu", "gpu", "cpu"], default="npu")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--skip-accuracy", action="store_true")
    args = parser.parse_args()

    if args.models:
        models_data = json.loads(Path(args.models).read_text())
        model_ids = [m["model_id"] if isinstance(m, dict) else m for m in models_data.get("llm", models_data.get("models", models_data))]
        all_results = []
        for model_id in model_ids:
            print(f"\n{'='*60}\nTesting {model_id} on {args.device}...")
            result = run_single_model(model_id, args.device, args.skip_accuracy)
            all_results.append(result)
        if args.output:
            Path(args.output).write_text(json.dumps(all_results, indent=2))
            print(f"\nResults written to {args.output}")
        failed = sum(1 for r in all_results if not r["functional"].get("passed", False))
        sys.exit(1 if failed > 0 else 0)
    else:
        print(f"Testing {args.model_id} on {args.device}...")
        results = run_single_model(args.model_id, args.device, args.skip_accuracy)
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2))
            print(f"\nResults written to {args.output}")
        sys.exit(0 if results["functional"].get("passed", False) else 1)


if __name__ == "__main__":
    from pathlib import Path
    main()
