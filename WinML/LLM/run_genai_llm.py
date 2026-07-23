#!/usr/bin/env python3
"""
LLM Runner script using Windows ML and ONNX GenAI on AMD NPU via VitisAI EP

This script runs Olive-converted LLM models using Windows ML and ONNX Runtime GenAI.
It's designed to work with AMD NPU devices and includes WinML integration.

Usage:
    python run_llm.py --model <path_to_model> --prompt "Your prompt"
    python run_llm.py --model <path_to_model> --interactive
    python run_llm.py --help

Example:
    python run_llm.py --model ./llama3_21binstruct --prompt "What is Python?"
    python run_llm.py --model ./llama3_21binstruct --interactive
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import onnxruntime_genai as og


def run_inference(
    model_path: Path,
    prompt: str,
    use_npu: bool = True,
    max_length: int = 512
) -> None:
    """
    Run single inference on the given prompt.

    Args:
        model_path: Path to the ONNX model directory
        prompt: Text prompt for the model
        use_npu: Whether to use the NPU EP configured in genai_config.json (default: True)
        max_length: Maximum generation length
    """
    print(f"\n{'='*60}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*60}")

    # Create model configuration
    try:
        config = og.Config(str(model_path))

        if not use_npu:
            # Force CPU EP
            config.clear_providers()
            config.append_provider("CPU")
            print("[INFO] Using CPU Execution Provider")
        else:
            print("[INFO] Using NPU Execution Provider (configured in genai_config.json)")

        # Load model
        print("[INFO] Loading model (this may take a minute)...")
        model = og.Model(config)
        print("[INFO] Model loaded successfully")

        # Create tokenizer
        print("[INFO] Creating tokenizer...")
        tokenizer = og.Tokenizer(model)
        print("[INFO] Tokenizer created")

        # Create generator params
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=max_length)

        # Tokenize input
        input_tokens = tokenizer.encode(prompt)
        print(f"[INFO] Tokenized prompt to {len(input_tokens)} tokens")

        # Run inference with streaming
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        print("Response: ", end="", flush=True)

        generator = og.Generator(model, params)
        generator.append_tokens(input_tokens)

        tokenizer_stream = tokenizer.create_stream()

        while not generator.is_done():
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            for token in new_tokens:
                text_chunk = tokenizer_stream.decode(token)
                print(text_chunk, end="", flush=True)

        print(f"\n{'='*60}\n")

    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def interactive_mode(
    model_path: Path,
    use_npu: bool = True,
    max_length: int = 512
) -> None:
    """
    Run model in interactive mode, allowing multiple prompts.

    Args:
        model_path: Path to the ONNX model directory
        use_npu: Whether to use the NPU EP configured in genai_config.json (default: True)
        max_length: Maximum generation length
    """
    print(f"\n{'='*60}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*60}")

    # Load model once
    try:
        config = og.Config(str(model_path))

        if not use_npu:
            config.clear_providers()
            config.append_provider("CPU")
            print("[INFO] Using CPU Execution Provider")
        else:
            print("[INFO] Using NPU Execution Provider (configured in genai_config.json)")

        print("[INFO] Loading model (this may take a minute)...")
        model = og.Model(config)
        print("[INFO] Model loaded successfully")

        print("[INFO] Creating tokenizer...")
        tokenizer = og.Tokenizer(model)
        tokenizer_stream = tokenizer.create_stream()
        print("[INFO] Tokenizer created")

    except Exception as e:
        print(f"\n[ERROR] Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Interactive loop
    print(f"\n{'='*60}")
    print("Interactive Mode")
    print(f"{'='*60}")
    print("Type your prompt and press Enter")
    print("Type 'quit', 'exit', or 'q' to exit")
    print("Press Enter without text for default prompt")
    print(f"{'='*60}\n")

    while True:
        try:
            prompt = input("Prompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nExiting...")
                break

            if not prompt:
                prompt = "What is an AI accelerator?"
                print(f"  (Using default: {prompt})")

            # Create fresh generator for each prompt
            params = og.GeneratorParams(model)
            params.set_search_options(max_length=max_length)

            input_tokens = tokenizer.encode(prompt)
            generator = og.Generator(model, params)
            generator.append_tokens(input_tokens)

            print("\nResponse: ", end="", flush=True)

            while not generator.is_done():
                generator.generate_next_token()
                new_tokens = generator.get_next_tokens()
                for token in new_tokens:
                    text_chunk = tokenizer_stream.decode(token)
                    print(text_chunk, end="", flush=True)

            print(f"\n{'-'*60}\n")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n[ERROR] Inference failed: {e}")
            import traceback
            traceback.print_exc()
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Run ONNX GenAI LLM models on AMD NPU using VitisAI EP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prompt inference (uses VitisAI EP / AMD NPU by default)
  python run_llm.py --model ./llama3_21binstruct --prompt "What is AI?"

  # Interactive mode
  python run_llm.py --model ./llama3_21binstruct --interactive

  # Use CPU instead of NPU
  python run_llm.py --model ./llama3_21binstruct --cpu --prompt "Hello"

  # Longer generation
  python run_llm.py --model ./llama3_21binstruct --max-length 1024 --interactive
"""
    )

    parser.add_argument(
        "--model", "-m",
        type=Path,
        required=True,
        help="Path to the ONNX model directory"
    )

    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Text prompt to send to the model"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (ignores --prompt)"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of VitisAI/NPU"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum generation length (default: 512)"
    )

    args = parser.parse_args()

    # Validate model path
    if not args.model.exists():
        print(f"[ERROR] Model path does not exist: {args.model}")
        sys.exit(1)

    # Check for genai_config.json
    genai_config = args.model / "genai_config.json"
    if not genai_config.exists():
        print(f"[ERROR] genai_config.json not found in model directory: {args.model}")
        sys.exit(1)

    use_npu = not args.cpu

    # Register the AMD execution providers via WinML if requested. Every catalog
    # EP except MIGraphX (non-functional on AMD systems) is made ready and
    # registered with ONNX Runtime GenAI, so models that request VitisAI or
    # RyzenAI in their genai_config.json can load. Keep the bootstrap handle
    # alive for the whole run and close it in the finally below.
    wasdk_handle = None
    if use_npu:
        print(f"\n{'='*60}")
        print("Registering Execution Providers")
        print(f"{'='*60}")
        try:
            from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
                InitializeOptions,
                initialize,
            )
            import winui3.microsoft.windows.ai.machinelearning as winml

            # Pass no version so the bootstrapper matches whatever Windows App
            # SDK runtime is installed on the machine.
            print("[INFO] Initializing WinAppSDK...")
            wasdk_handle = initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI)
            wasdk_handle.__enter__()
            print("[INFO] WinAppSDK initialized")

            providers = winml.ExecutionProviderCatalog.get_default().find_all_providers()
            print("\n[INFO] Available Execution Providers in WinML catalog:")
            for i, provider in enumerate(providers, 1):
                print(f"  {i}. {provider.name} (Status: {provider.ready_state})")
            print()

            registered = []
            for provider in providers:
                if provider.name == "MIGraphXExecutionProvider":
                    continue  # non-functional on AMD systems
                try:
                    if provider.ready_state != winml.ExecutionProviderReadyState.READY:
                        print(f"[INFO] Ensuring {provider.name} (state: {provider.ready_state})...")
                        provider.ensure_ready_async().get()

                    # try_register() activates the EP system-wide. Without it,
                    # library_path may be empty and og.Model() can fail (Error 126).
                    if not provider.try_register() or not provider.library_path:
                        print(f"[WARNING] {provider.name} could not be registered with WinML")
                        continue

                    # WinML makes the EP and its dependent DLLs loadable via this
                    # registration; no manual DLL path handling needed.
                    try:
                        og.register_execution_provider_library(provider.name, provider.library_path)
                        print(f"[INFO] Registered {provider.name}: {provider.library_path}")
                    except RuntimeError as e:
                        # An EP already registered with the runtime is fine.
                        if "already registered" not in str(e).lower():
                            raise
                        print(f"[INFO] {provider.name} already registered")
                    registered.append(provider.name)
                except Exception as e:
                    print(f"[WARNING] {provider.name} registration failed: {e}")

            if not registered:
                print("[WARNING] No execution providers registered, falling back to CPU")
                if wasdk_handle is not None:
                    wasdk_handle.__exit__(None, None, None)
                    wasdk_handle = None
                use_npu = False
        except Exception as e:
            print(f"[WARNING] WinML EP setup failed ({e}), falling back to CPU")
            import traceback
            traceback.print_exc()
            wasdk_handle = None
            use_npu = False

    # Run in appropriate mode
    try:
        if args.interactive:
            interactive_mode(args.model, use_npu=use_npu, max_length=args.max_length)
        elif args.prompt:
            run_inference(args.model, args.prompt, use_npu=use_npu, max_length=args.max_length)
        else:
            # Default: run with a demo prompt
            default_prompt = "Explain what an AMD NPU is in one sentence."
            print("[INFO] No prompt provided, using demo prompt")
            run_inference(args.model, default_prompt, use_npu=use_npu, max_length=args.max_length)
    finally:
        if wasdk_handle is not None:
            try:
                wasdk_handle.__exit__(None, None, None)
            except Exception as e:
                print(f"[WARNING] WinAppSDK bootstrap shutdown failed: {e}")


if __name__ == "__main__":
    main()
