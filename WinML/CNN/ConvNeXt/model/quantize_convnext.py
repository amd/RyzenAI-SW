#!/usr/bin/env python3
"""
Quantize ConvNeXt Model using Olive and AMD Quark

This script quantizes the FP32 ConvNeXt model to INT8 using Olive workflow
with AMD Quark quantization library for optimal NPU performance.

Requirements:
- olive-ai
- quark-for-amd
- onnxruntime

Usage:
    python quantize_convnext.py
    python quantize_convnext.py --calib-dir /path/to/calibration/images
    python quantize_convnext.py --model convnext_small.onnx --output convnext_small_i8.onnx
"""

import argparse
import sys
import json
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    missing = []

    try:
        import olive
        print(f"[OK] olive-ai version: {olive.__version__}")
    except ImportError:
        missing.append("olive-ai")

    try:
        import quark
        print(f"[OK] quark version: {quark.__version__}")
    except ImportError:
        missing.append("quark-for-amd")

    try:
        import onnxruntime as ort
        print(f"[OK] onnxruntime version: {ort.__version__}")
    except ImportError:
        missing.append("onnxruntime")

    if missing:
        print(f"\n[ERROR] Missing required packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install olive-ai quark-for-amd onnxruntime")
        return False

    return True


def fix_topological_sort(model_path):
    """
    Re-sort ONNX graph nodes in topological order.

    AMD Quark may produce a quantized model whose nodes are not in topological
    order, causing onnx.checker to fail with:
      'Nodes in a graph must be topologically sorted'
    This function reorders the nodes in-place and overwrites the file.
    """
    import onnx

    model = onnx.load(str(model_path))

    known = set()
    for inp in model.graph.input:
        known.add(inp.name)
    for init in model.graph.initializer:
        known.add(init.name)

    original_nodes = list(model.graph.node)
    sorted_nodes = []
    remaining = list(original_nodes)

    while remaining:
        progress = False
        for node in remaining:
            if all(inp == '' or inp in known for inp in node.input):
                sorted_nodes.append(node)
                for out in node.output:
                    if out:
                        known.add(out)
                remaining.remove(node)
                progress = True
                break
        if not progress:
            sorted_nodes.extend(remaining)
            break

    if sorted_nodes != original_nodes:
        del model.graph.node[:]
        model.graph.node.extend(sorted_nodes)
        onnx.save(model, str(model_path))
        print(f"   [OK] Topological sort applied ({len(sorted_nodes)} nodes)")
    else:
        print(f"   [OK] Nodes already in topological order")


def run_quantization(model_path, calib_dir, output_name, config_file):
    """Run Olive quantization workflow"""

    print("=" * 80)
    print("ConvNeXt INT8 Quantization using Olive + AMD Quark")
    print("=" * 80)

    if calib_dir is None:
        calib_dir = Path(__file__).parent / "calib_data"

    if not calib_dir.exists():
        print(f"\n[INFO] Calibration directory not found: {calib_dir}")
        calib_dir.mkdir(exist_ok=True)
        print(f"\n[WARNING] Please add calibration images to: {calib_dir}")
        print(f"          Recommended: 100-200 diverse images (JPG/PNG)")
        print(f"          For now, will use example images from ../images/")

    image_count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_count += len(list(calib_dir.glob(ext)))
        image_count += len(list(calib_dir.glob(f"**/{ext}")))

    if image_count == 0:
        print(f"\n[WARNING] No images found in {calib_dir}")
        print(f"          Will use fallback images from ../images/")
    else:
        print(f"\n[INFO] Found {image_count} calibration images in {calib_dir}")

    if output_name is None:
        output_name = "convnext_small_i8.onnx"
        print(f"\n[INFO] Output model name: {output_name}")

    model_dir = Path(__file__).parent
    output_path = model_dir / output_name
    temp_output_dir = model_dir / "olive_temp_output"

    print(f"\n[1/4] Loading Olive configuration from {config_file}")
    with open(config_file, 'r') as f:
        config = json.load(f)

    config['input_model']['model_path'] = str(model_path)
    config['engine']['output_dir'] = str(temp_output_dir)

    temp_config = Path("olive_config_temp.json")
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"   Model: {model_path}")
    print(f"   Calibration data: {calib_dir}")
    print(f"   Calibration images: {image_count if image_count > 0 else 'using fallback'}")
    print(f"   Output model: {output_name}")

    print(f"\n[2/4] Running Olive quantization...")
    print("   This may take several minutes...")

    import shutil

    try:
        from olive.workflows import run as olive_run

        olive_run(str(temp_config))

        print(f"\n[3/4] Quantization complete!")

        quantized_model = None
        if temp_output_dir.exists():
            onnx_files = list(temp_output_dir.glob("**/*.onnx"))
            if onnx_files:
                quantized_model = onnx_files[0]
                print(f"   Found quantized model: {quantized_model}")

        if quantized_model and quantized_model.exists():
            shutil.copy(quantized_model, output_path)
            print(f"   Copied to: {output_path}")

            if temp_output_dir.exists():
                shutil.rmtree(temp_output_dir)

            temp_config.unlink()

            # Fix node ordering before validation
            fix_topological_sort(output_path)

            return output_path
        else:
            print(f"[ERROR] Could not find quantized model in output directory")
            return None

    except Exception as e:
        print(f"\n[ERROR] Quantization failed: {e}")
        import traceback
        traceback.print_exc()

        if temp_config.exists():
            temp_config.unlink()
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)

        return None


def validate_quantized_model(model_path):
    """Validate the quantized model"""
    print(f"\n[4/4] Validating quantized model...")

    try:
        import onnx

        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)

        input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        output_count = len(model.graph.output)
        opset_version = model.opset_import[0].version

        print(f"   [OK] Model is valid")
        print(f"   - Input shape: {input_shape}")
        print(f"   - Outputs: {output_count}")
        print(f"   - Opset version: {opset_version}")

        quantized_nodes = [n for n in model.graph.node if 'Quant' in n.op_type or 'DequantizeLinear' in n.op_type]
        print(f"   - Quantization nodes: {len(quantized_nodes)}")

        if len(quantized_nodes) > 0:
            print(f"   [OK] Model is quantized (INT8)")
        else:
            print(f"   [WARNING] No quantization nodes found")

        return True

    except Exception as e:
        print(f"   [ERROR] Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Quantize ConvNeXt model using Olive and AMD Quark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize with default settings (outputs convnext_small_i8.onnx)
  python quantize_convnext.py

  # Specify custom calibration directory
  python quantize_convnext.py --calib-dir ./calibration_images

  # Specify custom output filename
  python quantize_convnext.py --output convnext_small_int8.onnx

  # Specify model path explicitly
  python quantize_convnext.py --model convnext_small.onnx
        """
    )

    parser.add_argument('--model', type=str, default=None,
                        help='Path to input FP32 ONNX model (default: convnext_small.onnx in model directory)')
    parser.add_argument('--calib-dir', type=str, default=None,
                        help='Directory containing calibration images (default: ./calib_data/)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output model filename (default: convnext_small_i8.onnx)')
    parser.add_argument('--config', type=str, default='olive_config_convnext.json',
                        help='Olive configuration file (default: olive_config_convnext.json)')

    args = parser.parse_args()

    if args.model is None:
        model_dir = Path(__file__).parent
        model_path = model_dir / "convnext_small.onnx"
        if not model_path.exists():
            print(f"[ERROR] Model not found: {model_path}")
            print("\nRun download_ConvNeXt.py first to get the FP32 model:")
            print("  python download_ConvNeXt.py")
            return 1
        print(f"[INFO] Using model: {model_path.name}")
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"[ERROR] Model not found: {model_path}")
            print("Run download_ConvNeXt.py first to get the FP32 model")
            return 1

    config_file = Path(args.config)
    if not config_file.exists():
        print(f"[ERROR] Config file not found: {config_file}")
        return 1

    calib_dir = Path(args.calib_dir) if args.calib_dir else None
    if calib_dir and not calib_dir.exists():
        print(f"[ERROR] Calibration directory not found: {calib_dir}")
        return 1

    if not check_dependencies():
        return 1

    quantized_model = run_quantization(model_path, calib_dir, args.output, config_file)

    if quantized_model:
        if validate_quantized_model(quantized_model):
            print("\n" + "=" * 80)
            print("SUCCESS: ConvNeXt model quantized to INT8")
            print("=" * 80)
            print(f"\nQuantized model: {quantized_model}")
            print(f"Model size: {quantized_model.stat().st_size / (1024*1024):.1f} MB")
            print("\nTo use the quantized model:")
            print(f"  cd ..")
            print(f"  python run_model.py --model model/{quantized_model.name} --ep_policy NPU")
            print("=" * 80)
            return 0
        else:
            return 1
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
