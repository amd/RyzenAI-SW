#!/usr/bin/env python3
"""
Directly fuse Pad->Conv in the FP32 static encoder for NPU compatibility.

We know from analysis that all 24 depthwise conv Pad ops use:
  pads=[0, 0, 4, 0, 0, 4]  (same-padding for kernel_size=9)
  mode=constant, value=0.0

This script fuses them by:
1. Setting Conv pads=[4, 4] (same-padding built into Conv)
2. Rewiring Conv input to bypass the Pad node
3. Removing the Pad node
"""
import os
import sys
from pathlib import Path

import onnx


def main():
    models_dir = Path("models")
    input_path = models_dir / "encoder-model.fp32.static.onnx"
    output_path = models_dir / "encoder-model.fp32.static.npu.onnx"

    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    ext_data = str(input_path) + ".data"
    has_ext = os.path.exists(ext_data)
    print(f"Loading {input_path}...")
    if has_ext:
        print(f"  External data: {ext_data} ({os.path.getsize(ext_data) / 1e9:.1f}GB)")
    model = onnx.load(str(input_path), load_external_data=True)

    # Build consumer map
    output_to_consumers = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp not in output_to_consumers:
                output_to_consumers[inp] = []
            output_to_consumers[inp].append(node)

    # Find and fuse Pad -> Conv pairs
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

        # We know the padding is [0,0,4, 0,0,4] -> Conv pads=[4,4]
        new_pads = [4, 4]

        # Update Conv pads attribute
        found = False
        for attr in conv_node.attribute:
            if attr.name == "pads":
                del attr.ints[:]
                attr.ints.extend(new_pads)
                found = True
                break
        if not found:
            conv_node.attribute.append(onnx.helper.make_attribute("pads", new_pads))

        # Rewire Conv to take Pad's input directly
        pad_input = node.input[0]
        for i, inp in enumerate(conv_node.input):
            if inp == pad_output:
                conv_node.input[i] = pad_input

        nodes_to_remove.append(node)
        fused += 1

    # Remove fused Pad nodes
    remove_names = {n.name for n in nodes_to_remove}
    remaining = [n for n in model.graph.node if n.name not in remove_names]
    del model.graph.node[:]
    model.graph.node.extend(remaining)

    pad_count = sum(1 for n in model.graph.node if n.op_type == "Pad")
    print(f"Fused {fused} Pad->Conv pairs")
    print(f"Remaining Pad ops: {pad_count}")

    # Save
    print(f"Saving {output_path}...")
    if has_ext:
        data_filename = output_path.name + ".data"
        print(f"  With external data: {data_filename}")
        onnx.save(
            model, str(output_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_filename,
            size_threshold=0,
        )
        data_full = output_path.parent / data_filename
        if data_full.exists():
            print(f"  ONNX: {output_path.stat().st_size / 1e6:.1f} MB")
            print(f"  Data: {data_full.stat().st_size / 1e6:.1f} MB")
    else:
        onnx.save(model, str(output_path))

    # Verify the file is valid
    print("Verifying model loads...")
    test = onnx.load(str(output_path), load_external_data=False)
    print(f"  Nodes: {len(test.graph.node)}")
    print(f"  Inputs: {[i.name for i in test.graph.input]}")
    print(f"  Outputs: {[o.name for o in test.graph.output]}")

    # Update static_config.json to point to the fused model
    config_path = models_dir / "static_config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
        config["encoder_model"] = output_path.name
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Updated {config_path}: encoder_model -> {output_path.name}")

    print("\nDone! Test with:")
    print('  python test_transcribe.py audio.wav --device npu')


if __name__ == "__main__":
    main()
