#!/usr/bin/env python3
"""
Preprocess Parakeet ONNX models for AMD Ryzen AI NPU (VitisAI EP).

The NPU requires static non-batch dimensions. This script fixes shapes on the
encoder and decoder, then (for FP32) applies compiler-oriented graph rewrites:

  - Fuse depthwise Pad -> Conv (VAIML compatibility)
  - Rewrite BOOL attention-mask path for VAIML 1.7.x (avoids unknown type 9 / Slice BOOL)

The FP32 encoder is written to encoder-model.fp32.static.npu.onnx. The decoder is
written to decoder_joint-model.fp32.static.onnx.

Usage:
    python preprocess_for_npu.py                          # Default: 15s chunks (1500 mel frames)
    python preprocess_for_npu.py --chunk-seconds 30       # 30s chunks (3000 mel frames)
    python preprocess_for_npu.py --chunk-seconds 10       # 10s chunks (1000 mel frames)
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import onnx
    from onnx import TensorProto, helper, shape_inference
except ImportError:
    print("ERROR: 'onnx' package is required. Install with: pip install onnx")
    sys.exit(1)


def fix_shapes(model, input_shapes: dict, output_shapes: dict = None):
    """
    Fix dynamic dimensions in an ONNX model's inputs and outputs.

    Args:
        model: ONNX ModelProto
        input_shapes: dict mapping input name -> list of int dimensions
        output_shapes: dict mapping output name -> list of int dimensions (optional)
    """
    # Fix input shapes
    for inp in model.graph.input:
        if inp.name in input_shapes:
            shape = input_shapes[inp.name]
            while len(inp.type.tensor_type.shape.dim) > 0:
                inp.type.tensor_type.shape.dim.pop()
            for dim_val in shape:
                dim = inp.type.tensor_type.shape.dim.add()
                dim.dim_value = dim_val

    if output_shapes:
        for out in model.graph.output:
            if out.name in output_shapes:
                shape = output_shapes[out.name]
                while len(out.type.tensor_type.shape.dim) > 0:
                    out.type.tensor_type.shape.dim.pop()
                for dim_val in shape:
                    dim = out.type.tensor_type.shape.dim.add()
                    dim.dim_value = dim_val


def fuse_pad_to_conv_depthwise(model: onnx.ModelProto) -> int:
    """Fuse depthwise Pad->Conv; mutates model. Returns number of fused pairs."""
    output_to_consumers = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp not in output_to_consumers:
                output_to_consumers[inp] = []
            output_to_consumers[inp].append(node)

    nodes_to_remove = []
    fused = 0

    for node in model.graph.node:
        if node.op_type != "Pad" or "depthwise_conv" not in (node.name or ""):
            continue

        pad_output = node.output[0]
        consumers = output_to_consumers.get(pad_output, [])
        if len(consumers) != 1 or consumers[0].op_type != "Conv":
            continue

        conv_node = consumers[0]
        new_pads = [4, 4]

        found = False
        for attr in conv_node.attribute:
            if attr.name == "pads":
                del attr.ints[:]
                attr.ints.extend(new_pads)
                found = True
                break
        if not found:
            conv_node.attribute.append(onnx.helper.make_attribute("pads", new_pads))

        pad_input = node.input[0]
        for i, inp in enumerate(conv_node.input):
            if inp == pad_output:
                conv_node.input[i] = pad_input

        nodes_to_remove.append(node)
        fused += 1

    remove_names = {n.name for n in nodes_to_remove}
    remaining = [n for n in model.graph.node if n.name not in remove_names]
    del model.graph.node[:]
    model.graph.node.extend(remaining)
    return fused


def _set_cast_to(node, dtype: int):
    del node.attribute[:]
    node.attribute.extend([helper.make_attribute("to", dtype)])


def _clear_attrs(node):
    del node.attribute[:]


def patch_bool_slice_for_171(model: onnx.ModelProto) -> None:
    """Rewrite shared attention mask for VAIML 1.7.x; mutates model."""
    by_name = {node.name: node for node in model.graph.node}

    target = by_name.get("/ConstantOfShape")
    if target is None:
        raise RuntimeError("Could not find /ConstantOfShape node")

    found = False
    for attr in target.attribute:
        if attr.name == "value":
            attr.t.CopyFrom(
                helper.make_tensor(
                    name="value",
                    data_type=TensorProto.INT64,
                    dims=[1],
                    vals=[1],
                )
            )
            found = True
            break

    if not found:
        raise RuntimeError("/ConstantOfShape node has no value attribute")

    cast_4 = by_name["/Cast_4"]
    cast_5 = by_name["/Cast_5"]
    cast_6 = by_name["/Cast_6"]
    and_1 = by_name["/And_1"]
    not_node = by_name["/Not"]

    _set_cast_to(cast_4, TensorProto.FLOAT)
    _set_cast_to(cast_5, TensorProto.FLOAT)
    _set_cast_to(cast_6, TensorProto.FLOAT)

    and_1.op_type = "Mul"
    _clear_attrs(and_1)

    not_node.op_type = "Identity"
    not_node.input[:] = [and_1.output[0]]
    _clear_attrs(not_node)

    for idx in range(24):
        _set_cast_to(by_name[f"/layers.{idx}/self_attn/Cast_1"], TensorProto.FLOAT)
        _set_cast_to(by_name[f"/layers.{idx}/self_attn/Cast_2"], TensorProto.FLOAT)

    one_const_output = "/_MaskOne_output_0"
    one_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[one_const_output],
        name="/_MaskOne",
        value=helper.make_tensor(
            name="value",
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[1.0],
        ),
    )

    new_nodes = []
    inserted_one_const = False
    for node in model.graph.node:
        if not inserted_one_const and node.name == "/layers.0/self_attn/Where":
            new_nodes.append(one_const)
            inserted_one_const = True

        if node.op_type == "Where" and "/self_attn/Where" in node.name and not node.name.endswith("Where_1"):
            layer_prefix = node.name.rsplit("/", 1)[0]
            mask = f"{layer_prefix}/Cast_1_output_0"
            neg_inf = node.input[1]
            logits = node.input[2]

            inv_mask = f"{layer_prefix}/MaskInv_output_0"
            logits_valid = f"{layer_prefix}/MaskValidMul_output_0"
            logits_invalid = f"{layer_prefix}/MaskInvalidMul_output_0"
            logits_masked = f"{layer_prefix}/MaskAdd_output_0"

            new_nodes.extend(
                [
                    helper.make_node(
                        "Sub",
                        [one_const_output, mask],
                        [inv_mask],
                        name=f"{layer_prefix}/MaskInv",
                    ),
                    helper.make_node(
                        "Mul",
                        [logits, mask],
                        [logits_valid],
                        name=f"{layer_prefix}/MaskValidMul",
                    ),
                    helper.make_node(
                        "Mul",
                        [neg_inf, inv_mask],
                        [logits_invalid],
                        name=f"{layer_prefix}/MaskInvalidMul",
                    ),
                    helper.make_node(
                        "Add",
                        [logits_valid, logits_invalid],
                        [logits_masked],
                        name=f"{layer_prefix}/MaskAdd",
                    ),
                ]
            )

            node.op_type = "Identity"
            node.input[:] = [logits_masked]
            _clear_attrs(node)
            new_nodes.append(node)
            continue

        if node.op_type == "Where" and node.name.endswith("Where_1"):
            mask = node.input[0]
            softmax = node.input[2]
            node.op_type = "Mul"
            node.input[:] = [mask, softmax]
            _clear_attrs(node)
            new_nodes.append(node)
            continue

        new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)


def convert_encoder(
    input_path: str,
    output_path: str,
    fixed_frames: int,
    *,
    npu_optimize: bool = False,
):
    """Convert encoder model to static shapes; optionally apply NPU Pad/mask rewrites (FP32)."""
    print(f"  Loading encoder: {input_path}")
    ext_data_path = input_path + ".data"
    has_external_data = os.path.exists(ext_data_path)
    if has_external_data:
        ext_size_gb = os.path.getsize(ext_data_path) / (1024**3)
        print(f"  External data file found: {ext_data_path} ({ext_size_gb:.1f}GB)")
        print(f"  Loading model + external data into memory (may take a minute)...")

    model = onnx.load(input_path, load_external_data=True)

    encoded_len = (fixed_frames + 7) // 8

    input_shapes = {
        "audio_signal": [1, 128, fixed_frames],
        "length": [1],
    }
    output_shapes = {
        "outputs": [1, 1024, encoded_len],
        "encoded_lengths": [1],
    }

    print(f"  Fixing encoder input: audio_signal -> [1, 128, {fixed_frames}]")
    print(f"  Fixing encoder output: outputs -> [1, 1024, {encoded_len}]")

    fix_shapes(model, input_shapes, output_shapes)

    if not has_external_data:
        print("  Running shape inference...")
        try:
            model = shape_inference.infer_shapes(model, check_type=False, data_prop=True)
        except Exception as e:
            print(f"  Warning: shape inference had issues (may still work): {e}")
    else:
        print("  Skipping shape inference (model >2GB, would strip weights)")

    if npu_optimize:
        fused = fuse_pad_to_conv_depthwise(model)
        pad_count = sum(1 for n in model.graph.node if n.op_type == "Pad")
        print(f"  NPU: fused {fused} Pad->Conv pairs ({pad_count} Pad ops left)")
        print("  NPU: patching BOOL/slice attention mask (VAIML 1.7.x)...")
        patch_bool_slice_for_171(model)
        print("  NPU: mask path rewritten to numeric masking")

    print(f"  Saving encoder: {output_path}")
    if has_external_data:
        data_filename = Path(output_path).name + ".data"
        print(f"  Saving with external data: {data_filename}")
        onnx.save(
            model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_filename,
            size_threshold=0,
        )
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
    print(f"  Loading decoder: {input_path}")
    model = onnx.load(input_path)

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
        description="Preprocess Parakeet ONNX for Ryzen AI NPU: static shapes + FP32 graph fixes"
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
        help="Suffix before .npu (FP32 encoder) or in filename (INT8). Default: .static",
    )
    parser.add_argument(
        "--precision", choices=["int8", "fp32"], default=None,
        help="Model precision to convert. Auto-detects if not specified.",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    print(f"ONNX version: {onnx.__version__}")

    hop_length = 160
    win_length = 400
    sample_rate = 16000
    num_samples = args.chunk_seconds * sample_rate
    fixed_frames = (num_samples - win_length) // hop_length + 1

    print(f"Parakeet NPU preprocessing")
    print(f"=========================")
    print(f"Chunk size   : {args.chunk_seconds}s")
    print(f"Fixed frames : {fixed_frames} mel frames")
    print(f"Encoded len  : ~{(fixed_frames + 7) // 8} frames (after 8x subsampling)")
    print()

    precision = args.precision
    if precision is None:
        if (models_dir / "encoder-model.onnx").exists():
            precision = "fp32"
        elif (models_dir / "encoder-model.int8.onnx").exists():
            precision = "int8"
        else:
            print(f"ERROR: No encoder model found in {models_dir}")
            print(f"  Run: python download_models.py --precision fp32")
            sys.exit(1)
        print(f"  Auto-detected precision: {precision}")

    suffix = args.output_suffix
    if precision == "fp32":
        encoder_src = models_dir / "encoder-model.onnx"
        decoder_src = models_dir / "decoder_joint-model.onnx"
        encoder_dst = models_dir / f"encoder-model.fp32{suffix}.npu.onnx"
        decoder_dst = models_dir / f"decoder_joint-model.fp32{suffix}.onnx"
        npu_optimize = True
    else:
        encoder_src = models_dir / "encoder-model.int8.onnx"
        decoder_src = models_dir / "decoder_joint-model.int8.onnx"
        encoder_dst = models_dir / f"encoder-model.int8{suffix}.onnx"
        decoder_dst = models_dir / f"decoder_joint-model.int8{suffix}.onnx"
        npu_optimize = False

    if not encoder_src.exists():
        print(f"ERROR: Encoder model not found: {encoder_src}")
        print(f"  Run: python download_models.py --precision {precision}")
        sys.exit(1)
    if not decoder_src.exists():
        print(f"ERROR: Decoder model not found: {decoder_src}")
        print(f"  Run: python download_models.py --precision {precision}")
        sys.exit(1)

    print(f"[1/2] Converting encoder...")
    convert_encoder(
        str(encoder_src),
        str(encoder_dst),
        fixed_frames,
        npu_optimize=npu_optimize,
    )
    print()

    print(f"[2/2] Converting decoder...")
    convert_decoder(str(decoder_src), str(decoder_dst))
    print()

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
