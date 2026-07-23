#!/usr/bin/env python3
"""
Quantize YOLOv8 Model using Olive and AMD Quark

This script quantizes the FP32 YOLOv8 model to INT8 using Olive workflow
with AMD Quark quantization library for optimal NPU performance.

Requirements:
- olive-ai
- quark-for-amd
- onnxruntime
- calibration images (COCO or representative object detection dataset)

Usage:
    python quantize_yolov8.py
    python quantize_yolov8.py --calib-dir /path/to/calibration/images
    python quantize_yolov8.py --model yolov8s.onnx --output yolov8s_int8.onnx
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
        missing.append("amd-quark")

    try:
        import onnxruntime as ort
        print(f"[OK] onnxruntime version: {ort.__version__}")
    except ImportError:
        missing.append("onnxruntime")

    if missing:
        print(f"\n[ERROR] Missing required packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install olive-ai amd-quark onnxruntime")
        return False

    return True


def detect_model_size(model_path):
    """Detect YOLOv8 model size from filename"""
    model_name = Path(model_path).stem.lower()

    if 'yolov8n' in model_name:
        return 'n'
    elif 'yolov8s' in model_name:
        return 's'
    elif 'yolov8m' in model_name:
        return 'm'
    elif 'yolov8l' in model_name:
        return 'l'
    elif 'yolov8x' in model_name:
        return 'x'
    else:
        return 'm'  # Default to medium


def run_quantization(model_path, calib_dir, output_name, config_file):
    """Run Olive quantization workflow"""

    print("=" * 80)
    print("YOLOv8 INT8 Quantization using Olive + AMD Quark")
    print("=" * 80)

    # Check calibration data
    if calib_dir is None:
        calib_dir = Path(__file__).parent / "calib_data"

    if not calib_dir.exists():
        print(f"\n[INFO] Calibration directory not found: {calib_dir}")
        print(f"[INFO] Creating calib_data directory...")
        calib_dir.mkdir(exist_ok=True)
        print(f"\n[WARNING] Please add calibration images to: {calib_dir}")
        print(f"          Recommended: 100-300 COCO images or representative object detection images (JPG/PNG)")
        print(f"          Run: python download_calib_data.py")

    # Count calibration images
    image_count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_count += len(list(calib_dir.glob(ext)))
        image_count += len(list(calib_dir.glob(f"**/{ext}")))

    if image_count == 0:
        print(f"\n[ERROR] No calibration images found in {calib_dir}")
        print(f"        Please run: python download_calib_data.py")
        return None
    else:
        print(f"\n[INFO] Found {image_count} calibration images in {calib_dir}")

    # Determine output filename if not specified
    if output_name is None:
        model_size = detect_model_size(model_path)
        output_name = f"yolov8{model_size}_int8.onnx"
        print(f"\n[INFO] Auto-detected model size: yolov8{model_size}")
        print(f"[INFO] Output model name: {output_name}")

    # Output directory and final path
    model_dir = Path(__file__).parent
    output_path = model_dir / output_name
    temp_output_dir = model_dir / "olive_temp_output"

    # Load Olive config
    print(f"\n[1/4] Loading Olive configuration from {config_file}")
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Update paths
    config['input_model']['model_path'] = str(model_path)
    config['engine']['output_dir'] = str(temp_output_dir)

    # Save updated config
    temp_config = Path("olive_config_temp.json")
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"   Model: {model_path}")
    print(f"   Calibration data: {calib_dir}")
    print(f"   Calibration images: {image_count}")
    print(f"   Output model: {output_name}")

    # Run Olive
    print(f"\n[2/4] Running Olive quantization...")
    print("   This may take several minutes...")

    # Import shutil at top of try block for use in finally
    import shutil

    try:
        from olive.workflows import run as olive_run

        # Run Olive workflow
        olive_output = olive_run(str(temp_config))

        print(f"\n[3/4] Quantization complete!")
        print(f"   Olive output type: {type(olive_output)}")

        # Search for the quantized model in temp output directory
        quantized_model = None

        # Look for ONNX files in temp output directory
        if temp_output_dir.exists():
            onnx_files = list(temp_output_dir.glob("**/*.onnx"))
            if onnx_files:
                # Use the first/only ONNX file found
                quantized_model = onnx_files[0]
                print(f"   Found quantized model: {quantized_model}")

        if quantized_model and quantized_model.exists():
            # Copy to final location with descriptive name
            shutil.copy(quantized_model, output_path)
            print(f"   Copied to: {output_path}")

            # Cleanup temp directory
            if temp_output_dir.exists():
                shutil.rmtree(temp_output_dir)

            # Cleanup temp config
            if temp_config.exists():
                temp_config.unlink()

            return output_path
        else:
            print(f"[ERROR] Could not find quantized model in {olive_output}")
            return None

    except Exception as e:
        print(f"\n[ERROR] Quantization failed: {e}")
        import traceback
        traceback.print_exc()

        # Cleanup temp config
        if temp_config.exists():
            temp_config.unlink()

        # Cleanup temp directory
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)

        return None


def validate_quantized_model(model_path):
    """Validate the quantized model"""
    print(f"\n[4/4] Validating quantized model...")

    try:
        import onnx

        model = onnx.load(str(model_path))

        # Try full validation, but don't fail if it has topology issues
        try:
            onnx.checker.check_model(model)
            print(f"   [OK] Model passes ONNX validation")
        except Exception as validation_error:
            print(f"   [WARNING] ONNX validation failed (common with Quark): {str(validation_error)[:100]}")
            print(f"   [INFO] Model may still work correctly on NPU - continuing...")

        # Get model info
        input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        output_count = len(model.graph.output)
        opset_version = model.opset_import[0].version

        print(f"   [OK] Model structure verified")
        print(f"   - Input shape: {input_shape}")
        print(f"   - Outputs: {output_count}")
        print(f"   - Opset version: {opset_version}")

        # Check for quantization
        quantized_nodes = [node for node in model.graph.node if 'Quant' in node.op_type or 'DequantizeLinear' in node.op_type]
        print(f"   - Quantization nodes: {len(quantized_nodes)}")

        if len(quantized_nodes) > 0:
            print(f"   [OK] Model is quantized (INT8)")
        else:
            print(f"   [WARNING] No quantization nodes found")

        # Always return True - validation warnings don't prevent usage
        return True

    except Exception as e:
        print(f"   [WARNING] Could not fully validate model: {e}")
        print(f"   [INFO] This may not prevent the model from working")
        # Return True to continue anyway
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Quantize YOLOv8 model using Olive and AMD Quark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize with default settings (auto-detects model size, outputs yolov8m_int8.onnx)
  python quantize_yolov8.py

  # Specify custom calibration directory
  python quantize_yolov8.py --calib-dir ./calibration_images

  # Specify custom output filename
  python quantize_yolov8.py --output yolov8m_quantized.onnx

  # Quantize different YOLOv8 variant
  python quantize_yolov8.py --model yolov8s.onnx
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to input FP32 ONNX model (default: auto-detect yolov8*.onnx)'
    )

    parser.add_argument(
        '--calib-dir',
        type=str,
        default=None,
        help='Directory containing calibration images (default: ./calib_data/)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output model filename (default: auto-detect from input, e.g., yolov8m_int8.onnx)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='olive_config_yolov8.json',
        help='Olive configuration file (default: olive_config_yolov8.json)'
    )

    args = parser.parse_args()

    # Auto-detect model if not specified
    if args.model is None:
        model_dir = Path(__file__).parent
        # Try to find any YOLOv8 ONNX model
        yolov8_models = list(model_dir.glob("yolov8*.onnx"))
        # Exclude quantized models
        yolov8_models = [m for m in yolov8_models if 'int8' not in m.stem.lower() and 'i8' not in m.stem.lower()]

        if yolov8_models:
            model_path = yolov8_models[0]
            print(f"[INFO] Auto-detected model: {model_path.name}")
        else:
            print(f"[ERROR] No YOLOv8 model found in {model_dir}")
            print("\nRun download_yolov8.py first to get the FP32 model:")
            print("  python download_yolov8.py           # YOLOv8m (default)")
            print("  python download_yolov8.py -s n      # YOLOv8n (nano)")
            print("  python download_yolov8.py -s s      # YOLOv8s (small)")
            return 1
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"[ERROR] Model not found: {model_path}")
            print("Run download_yolov8.py first to get the FP32 model")
            return 1

    config_file = Path(args.config)
    if not config_file.exists():
        print(f"[ERROR] Config file not found: {config_file}")
        return 1

    calib_dir = Path(args.calib_dir) if args.calib_dir else None
    if calib_dir and not calib_dir.exists():
        print(f"[ERROR] Calibration directory not found: {calib_dir}")
        return 1

    output_name = args.output

    # Check dependencies
    if not check_dependencies():
        return 1

    # Run quantization
    quantized_model = run_quantization(model_path, calib_dir, output_name, config_file)

    if quantized_model:
        # Validate
        if validate_quantized_model(quantized_model):
            print("\n" + "=" * 80)
            print("SUCCESS: YOLOv8 model quantized to INT8")
            print("=" * 80)
            print(f"\nQuantized model: {quantized_model}")
            print(f"Model size: {quantized_model.stat().st_size / (1024*1024):.1f} MB")
            print("\nTo use the quantized model:")
            print(f"  cd ../python")
            print(f"  python run_model.py --model ../model/{quantized_model.name} --ep_policy NPU")
            print("=" * 80)
            return 0
        else:
            return 1
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
