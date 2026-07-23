#!/usr/bin/env python3
"""
Quantize RetinaFace Model using Olive and AMD Quark

This script quantizes the FP32 RetinaFace model to INT8 using Olive workflow
with AMD Quark quantization library for optimal NPU performance.

Requirements:
- olive-ai
- quark-for-amd
- onnxruntime
- calibration images (default: uses example images from ../images/)

Usage:
    python quantize_retinaface.py
    python quantize_retinaface.py --calib-dir /path/to/calibration/images
    python quantize_retinaface.py --model retinaface_mobilenet.onnx --output olive_output
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
        print("  pip install olive-ai quark-for-amd onnxruntime")
        return False

    return True


def detect_backbone(model_path):
    """Detect backbone type from model filename"""
    model_name = Path(model_path).stem.lower()

    if 'resnet' in model_name or 'r34' in model_name or 'r50' in model_name:
        return 'resnet'
    elif 'mobilenet' in model_name or 'mv1' in model_name or 'mv2' in model_name:
        return 'mobilenet'
    else:
        return 'mobilenet'  # Default


def run_quantization(model_path, calib_dir, output_name, config_file):
    """Run Olive quantization workflow"""

    print("=" * 80)
    print("RetinaFace INT8 Quantization using Olive + AMD Quark")
    print("=" * 80)

    # Check calibration data
    if calib_dir is None:
        calib_dir = Path(__file__).parent / "calib_data"

    if not calib_dir.exists():
        print(f"\n[INFO] Calibration directory not found: {calib_dir}")
        print(f"[INFO] Creating calib_data directory...")
        calib_dir.mkdir(exist_ok=True)
        print(f"\n[WARNING] Please add calibration images to: {calib_dir}")
        print(f"          Recommended: 100-300 face images (JPG/PNG)")
        print(f"          For now, will use example images from ../images/")

    # Count calibration images
    image_count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_count += len(list(calib_dir.glob(ext)))
        image_count += len(list(calib_dir.glob(f"**/{ext}")))

    if image_count == 0:
        print(f"\n[WARNING] No images found in {calib_dir}")
        print(f"          Will use fallback images from ../images/")
    else:
        print(f"\n[INFO] Found {image_count} calibration images in {calib_dir}")

    # Determine output filename if not specified
    if output_name is None:
        backbone = detect_backbone(model_path)
        output_name = f"retinaface_{backbone}_i8.onnx"
        print(f"\n[INFO] Auto-detected backbone: {backbone}")
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

    # DO NOT update data_dir - keep it as-is from the config file
    # Olive will resolve it relative to the user_script location
    # The user_script's load_dataset will use the data_dir parameter as provided

    # Save updated config
    temp_config = Path("olive_config_temp.json")
    with open(temp_config, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"   Model: {model_path}")
    print(f"   Calibration data: {calib_dir}")
    print(f"   Calibration images: {image_count if image_count > 0 else 'using fallback'}")
    print(f"   Output model: {output_name}")
    print(f"   Config data_dir: {config['data_configs'][0]['load_dataset_config']['data_dir']}")

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
        # Olive saves to the temp_output_dir we specified
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

        # Check model
        onnx.checker.check_model(model)

        # Get model info
        input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        output_count = len(model.graph.output)
        opset_version = model.opset_import[0].version

        print(f"   [OK] Model is valid")
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

        return True

    except Exception as e:
        print(f"   [ERROR] Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Quantize RetinaFace model using Olive and AMD Quark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize with default settings (auto-detects backbone, outputs retinaface_mobilenet_i8.onnx)
  python quantize_retinaface.py

  # Specify custom calibration directory
  python quantize_retinaface.py --calib-dir ./calibration_images

  # Specify custom output filename
  python quantize_retinaface.py --output retinaface_mobilenet_int8.onnx

  # Quantize ResNet model (auto-detects backbone, outputs retinaface_resnet_i8.onnx)
  python quantize_retinaface.py --model retinaface_resnet.onnx
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to input FP32 ONNX model (default: auto-detect retinaface_mobilenet.onnx or retinaface_resnet.onnx)'
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
        help='Output model filename (default: auto-detect from input, e.g., retinaface_mobilenet_i8.onnx)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='olive_config_retinaface.json',
        help='Olive configuration file (default: olive_config_retinaface.json)'
    )

    args = parser.parse_args()

    # Auto-detect model if not specified
    if args.model is None:
        model_dir = Path(__file__).parent
        # Try both model names (ResNet first as it's the default)
        resnet_path = model_dir / "retinaface_resnet.onnx"
        mobilenet_path = model_dir / "retinaface_mobilenet.onnx"

        if resnet_path.exists():
            model_path = resnet_path
            print(f"[INFO] Auto-detected model: {model_path.name}")
        elif mobilenet_path.exists():
            model_path = mobilenet_path
            print(f"[INFO] Auto-detected model: {model_path.name}")
        else:
            print(f"[ERROR] No model found. Tried:")
            print(f"  - {resnet_path}")
            print(f"  - {mobilenet_path}")
            print("\nRun download_and_export.py first to get the FP32 model:")
            print("  python download_and_export.py                       # ResNet (default)")
            print("  python download_and_export.py --backbone mobilenet  # MobileNet (optional)")
            return 1
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"[ERROR] Model not found: {model_path}")
            print("Run download_and_export.py first to get the FP32 model")
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
            print("SUCCESS: RetinaFace model quantized to INT8")
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
