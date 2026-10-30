#!/usr/bin/env python3
"""
Convert Parakeet ONNX models from dynamic shapes to static shapes for NPU.

The AMD Ryzen AI NPU (VitisAI EP) requires all non-batch dimensions to be fixed.
This script creates static-shape versions of the encoder and decoder models.

Usage:
    python convert_static.py                          # Default: 15s chunks (1500 mel frames)
    python convert_static.py --chunk-seconds 30       # 30s chunks (3000 mel frames)
    python convert_static.py --chunk-seconds 10       # 10s chunks (1000 mel frames)
"""

import argparse
import os
import sys
from pathlib import Path


def fix_shapes(model, input_shapes: dict, output_shapes: dict = None):
    """
    Fix dynamic dimensions in an ONNX model's inputs and outputs.

    Args:
        model: ONNX ModelProto
        input_shapes: dict mapping input name -> list of int dimensions
        output_shapes: dict mapping output name -> list of int dimensions (optional)
    """
    import onnx
    from onnx import TensorProto

    # Fix input shapes
    for inp in model.graph.input:
        if inp.name in input_shapes:
            shape = input_shapes[inp.name]
            # Clear existing dimensions
            while len(inp.type.tensor_type.shape.dim) > 0:
                inp.type.tensor_type.shape.dim.pop()
            # Set new fixed dimensions
            for dim_val in shape:
                dim = inp.type.tensor_type.shape.dim.add()
                dim.dim_value = dim_val

    # Fix output shapes if provided
    if output_shapes:
        for out in model.graph.output:
            if out.name in output_shapes:
                shape = output_shapes[out.name]
                while len(out.type.tensor_type.shape.dim) > 0:
                    out.type.tensor_type.shape.dim.pop()
                for dim_val in shape:
                    dim = out.type.tensor_type.shape.dim.add()
                    dim.dim_value = dim_val


def convert_encoder(input_path: str, output_path: str, fixed_frames: int):
    """Convert encoder model to static shapes."""
    import onnx
    from onnx import shape_inference

    print(f"  Loading encoder: {input_path}")
    # Check if there's an external data file (FP32 models use .onnx.data)
    ext_data_path = input_path + ".data"
    has_external_data = os.path.exists(ext_data_path)
    if has_external_data:
        ext_size_gb = os.path.getsize(ext_data_path) / (1024**3)
        print(f"  External data file found: {ext_data_path} ({ext_size_gb:.1f}GB)")
        print(f"  Loading model + external data into memory (may take a minute)...")

    model = onnx.load(input_path, load_external_data=True)

    # Encoder subsampling factor is 8
    # encoded_len = ceil(fixed_frames / 8)
    # But the actual encoded length depends on the model internals.
    # We fix inputs only; outputs remain dynamic for safety.
    encoded_len = (fixed_frames + 7) // 8  # approximate

    input_shapes = {
        "audio_signal": [1, 128, fixed_frames],
        "length": [1],
    }

    # We also fix output shapes to help the NPU compiler
    output_shapes = {
        "outputs": [1, 1024, encoded_len],
        "encoded_lengths": [1],
    }

    print(f"  Fixing encoder input: audio_signal -> [1, 128, {fixed_frames}]")
    print(f"  Fixing encoder output: outputs -> [1, 1024, {encoded_len}]")

    fix_shapes(model, input_shapes, output_shapes)

    # Skip shape inference for large models (>2GB) because infer_shapes()
    # strips all initializers due to protobuf size limits.
    # The VitisAI/NPU compiler handles its own shape propagation.
    if not has_external_data:
        print("  Running shape inference...")
        try:
            model = shape_inference.infer_shapes(model, check_type=False, data_prop=True)
        except Exception as e:
            print(f"  Warning: shape inference had issues (may still work): {e}")
    else:
        print("  Skipping shape inference (model >2GB, would strip weights)")

    print(f"  Saving static encoder: {output_path}")
    if has_external_data:
        # Large model (>2GB) - must save with external data (protobuf limit)
        data_filename = Path(output_path).name + ".data"
        print(f"  Saving with external data: {data_filename}")
        onnx.save(
            model, output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_filename,
            size_threshold=0,  # Save all tensors externally
        )
        # Report combined size
        data_full_path = Path(output_path).parent / data_filename
        data_mb = data_full_path.stat().st_size / (1024 * 1024) if data_full_path.exists() else 0
        onnx_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Done (ONNX: {onnx_mb:.1f} MB + data: {data_mb:.1f} MB)")
    else:
        onnx.save(model, output_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Done ({size_mb:.1f} MB)")


def convert_decoder(input_path: str, output_path: str):
    """Convert decoder model to static shapes."""
    import onnx
    from onnx import shape_inference

    print(f"  Loading decoder: {input_path}")
    model = onnx.load(input_path)

    # Decoder inputs are already effectively fixed in our usage:
    # We always pass [1,1024,1] encoder slice, [1,1] targets, etc.
    input_shapes = {
        "encoder_outputs": [1, 1024, 1],
        "targets": [1, 1],
        "target_length": [1],
        "input_states_1": [2, 1, 640],
        "input_states_2": [2, 1, 640],
    }

    output_shapes = {
        "outputs": [1, 1, 1, 8198],
        "prednet_lengths": [1],
        "output_states_1": [2, 1, 640],
        "output_states_2": [2, 1, 640],
    }

    print("  Fixing decoder inputs: encoder_outputs->[1,1024,1], targets->[1,1], states->[2,1,640]")
    print("  Fixing decoder outputs: logits->[1,1,1,8198], states->[2,1,640]")

    fix_shapes(model, input_shapes, output_shapes)

    # Run shape inference
    print("  Running shape inference...")
    try:
        model = shape_inference.infer_shapes(model, check_type=False, data_prop=True)
    except Exception as e:
        print(f"  Warning: shape inference had issues (may still work): {e}")

    print(f"  Saving static decoder: {output_path}")
    onnx.save(model, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Done ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Parakeet ONNX models to static shapes for Ryzen AI NPU"
    )
    parser.add_argument(
        "--models-dir", default="./models",
        help="Path to models directory (default: ./models)",
    )
    parser.add_argument(
        "--chunk-seconds", type=int, default=15,
        help="Audio chunk size in seconds. Determines fixed encoder input length. (default: 15)",
    )
    parser.add_argument(
        "--output-suffix", default=".static",
        help="Suffix for output filenames (default: .static)",
    )
    parser.add_argument(
        "--precision", choices=["int8", "fp32"], default=None,
        help="Model precision to convert. Auto-detects if not specified.",
    )
    args = parser.parse_args()

    try:
        import onnx
        print(f"ONNX version: {onnx.__version__}")
    except ImportError:
        print("ERROR: 'onnx' package is required. Install with: pip install onnx")
        sys.exit(1)

    models_dir = Path(args.models_dir)

    # Calculate fixed mel frames from chunk duration
    # num_frames = (num_samples - win_length) / hop_length + 1
    # num_samples = chunk_seconds * 16000
    hop_length = 160
    win_length = 400
    sample_rate = 16000
    num_samples = args.chunk_seconds * sample_rate
    fixed_frames = (num_samples - win_length) // hop_length + 1

    print(f"Parakeet Static Shape Converter")
    print(f"================================")
    print(f"Chunk size   : {args.chunk_seconds}s")
    print(f"Fixed frames : {fixed_frames} mel frames")
    print(f"Encoded len  : ~{(fixed_frames + 7) // 8} frames (after 8x subsampling)")
    print()

    # Determine precision (auto-detect if not specified)
    precision = args.precision
    if precision is None:
        # Prefer FP32 if available (better for BF16 NPU path), fall back to INT8
        if (models_dir / "encoder-model.onnx").exists():
            precision = "fp32"
        elif (models_dir / "encoder-model.int8.onnx").exists():
            precision = "int8"
        else:
            print(f"ERROR: No encoder model found in {models_dir}")
            print(f"  Run: python download_models.py --precision fp32")
            sys.exit(1)
        print(f"  Auto-detected precision: {precision}")

    # Find source models based on precision
    if precision == "fp32":
        encoder_src = models_dir / "encoder-model.onnx"
        decoder_src = models_dir / "decoder_joint-model.onnx"
        encoder_dst = models_dir / f"encoder-model.fp32{args.output_suffix}.onnx"
        decoder_dst = models_dir / f"decoder_joint-model.fp32{args.output_suffix}.onnx"
    else:
        encoder_src = models_dir / "encoder-model.int8.onnx"
        decoder_src = models_dir / "decoder_joint-model.int8.onnx"
        encoder_dst = models_dir / f"encoder-model.int8{args.output_suffix}.onnx"
        decoder_dst = models_dir / f"decoder_joint-model.int8{args.output_suffix}.onnx"

    if not encoder_src.exists():
        print(f"ERROR: Encoder model not found: {encoder_src}")
        print(f"  Run: python download_models.py --precision {precision}")
        sys.exit(1)
    if not decoder_src.exists():
        print(f"ERROR: Decoder model not found: {decoder_src}")
        print(f"  Run: python download_models.py --precision {precision}")
        sys.exit(1)

    print(f"[1/2] Converting encoder...")
    convert_encoder(str(encoder_src), str(encoder_dst), fixed_frames)
    print()

    print(f"[2/2] Converting decoder...")
    convert_decoder(str(decoder_src), str(decoder_dst))
    print()

    # Save a config for the static models
    import json
    static_config = {
        "chunk_seconds": args.chunk_seconds,
        "fixed_frames": fixed_frames,
        "encoded_len": (fixed_frames + 7) // 8,
        "precision": precision,
        "encoder_model": encoder_dst.name,
        "decoder_model": decoder_dst.name,
    }
    config_path = models_dir / "static_config.json"
    with open(config_path, "w") as f:
        json.dump(static_config, f, indent=2)

    print(f"Static models saved:")
    print(f"  Encoder: {encoder_dst}")
    print(f"  Decoder: {decoder_dst}")
    print(f"  Config:  {config_path}")
    print()
    print(f"To use with NPU:")
    print(f"  python test_transcribe.py audio.wav --device npu")


if __name__ == "__main__":
    main()
