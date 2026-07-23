#!/usr/bin/env python3
"""
Quantize DeepLabV3 Model using Olive and AMD Quark

Quantizes the FP32 DeepLabV3 ResNet50 model to INT8 using the Olive workflow
with AMD Quark for optimal NPU performance.

Requirements:
    pip install olive-ai quark-for-amd onnxruntime

Usage:
    python quantize_deeplabv3.py
    python quantize_deeplabv3.py --calib-dir /path/to/images
    python quantize_deeplabv3.py --model deeplabv3_resnet50.onnx --output deeplabv3_i8.onnx
"""

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
def check_dependencies() -> bool:
    missing = []
    try:
        import olive
        print(f"[OK] olive-ai      {olive.__version__}")
    except ImportError:
        missing.append("olive-ai")

    try:
        import quark
        print(f"[OK] quark         {quark.__version__}")
    except ImportError:
        missing.append("quark-for-amd")

    try:
        import onnxruntime as ort
        print(f"[OK] onnxruntime   {ort.__version__}")
    except ImportError:
        missing.append("onnxruntime")

    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print("Install with:")
        print("  pip install olive-ai quark-for-amd onnxruntime")
        return False
    return True


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------
def run_quantization(model_path: Path, calib_dir: Path,
                     output_name: str, config_file: Path) -> Path | None:
    print("=" * 70)
    print("DeepLabV3 INT8 Quantization using Olive + AMD Quark")
    print("=" * 70)

    # Validate / create calibration directory
    if not calib_dir.exists():
        print(f"\n[INFO] Calibration directory not found: {calib_dir}")
        print("[INFO] Creating calib_data directory...")
        calib_dir.mkdir(exist_ok=True)
        print(f"[WARNING] Add calibration images to: {calib_dir}")
        print("          Run: python download_calib_data.py")

    image_count = sum(
        len(list(calib_dir.glob(pat)))
        for pat in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    )
    image_count += sum(
        len(list(calib_dir.glob(f"**/{pat}")))
        for pat in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    )

    if image_count == 0:
        print(f"[WARNING] No images found in {calib_dir} — will use fallback images")
    else:
        print(f"\n[INFO] Found {image_count} calibration images in {calib_dir}")

    if output_name is None:
        output_name = "deeplabv3_resnet50_i8.onnx"

    model_dir = Path(__file__).parent
    output_path = model_dir / output_name
    temp_output_dir = model_dir / "olive_temp_output"

    # Load and patch Olive config
    print(f"\n[1/4] Loading Olive config from {config_file.name}")
    with open(config_file) as f:
        config = json.load(f)

    config["input_model"]["model_path"] = str(model_path)
    config["engine"]["output_dir"] = str(temp_output_dir)

    temp_config = model_dir / "olive_config_temp.json"
    with open(temp_config, "w") as f:
        json.dump(config, f, indent=2)

    print(f"   Input model:  {model_path.name}")
    print(f"   Calib data:   {calib_dir}")
    print(f"   Output model: {output_name}")

    # Run Olive
    print(f"\n[2/4] Running Olive quantization (this may take several minutes)...")
    try:
        from olive.workflows import run as olive_run

        olive_run(str(temp_config))

        print(f"\n[3/4] Quantization complete — searching for output model...")

        quantized_model = None
        if temp_output_dir.exists():
            onnx_files = list(temp_output_dir.glob("**/*.onnx"))
            if onnx_files:
                quantized_model = onnx_files[0]
                print(f"      Found: {quantized_model}")

        if quantized_model and quantized_model.exists():
            shutil.copy(quantized_model, output_path)
            print(f"      Saved to: {output_path}")
        else:
            print("[ERROR] Could not find quantized ONNX model in olive output")
            return None

        # Quark re-introduces symbolic dims — pin them back to static values
        print(f"      Fixing static shapes [1,3,520,520] / [1,21,520,520]...")
        fix_static_shapes(output_path)
        print(f"      Static shapes applied")

        return output_path

    except Exception as e:
        print(f"\n[ERROR] Quantization failed: {e}")
        traceback.print_exc()
        return None

    finally:
        if temp_config.exists():
            temp_config.unlink()
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)


# ---------------------------------------------------------------------------
# Static shape fix
# ---------------------------------------------------------------------------
def fix_static_shapes(model_path: Path,
                      input_shape: list = [1, 3, 520, 520],
                      output_shape: list = [1, 21, 520, 520]) -> None:
    """
    Pin input and output dims to concrete static values.

    Quark re-introduces symbolic dim_param strings (e.g. 'batch_size', 'height',
    'width') during quantization. We directly overwrite only the graph input/output
    dim fields without running shape inference or check_model — both of which can
    trip over Quark's quantized graph structure (topological sort, initializer refs).
    """
    import onnx

    model = onnx.load(str(model_path))

    inp = model.graph.input[0]
    for dim, value in zip(inp.type.tensor_type.shape.dim, input_shape):
        dim.dim_value = value
        dim.ClearField("dim_param")

    out = model.graph.output[0]
    for dim, value in zip(out.type.tensor_type.shape.dim, output_shape):
        dim.dim_value = value
        dim.ClearField("dim_param")

    onnx.save(model, str(model_path))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_model(model_path: Path) -> bool:
    print(f"\n[4/4] Validating quantized model...")
    try:
        import onnx

        model = onnx.load(str(model_path))

        inp = model.graph.input[0]
        inp_shape = [d.dim_value if d.dim_value != 0 else d.dim_param or "?"
                     for d in inp.type.tensor_type.shape.dim]
        out = model.graph.output[0]
        out_shape = [d.dim_value if d.dim_value != 0 else d.dim_param or "?"
                     for d in out.type.tensor_type.shape.dim]
        opset = model.opset_import[0].version

        quant_nodes = [
            n for n in model.graph.node
            if "Quant" in n.op_type or "DequantizeLinear" in n.op_type
        ]

        print(f"   [OK] Model loaded")
        print(f"        Input:             {inp_shape}")
        print(f"        Output:            {out_shape}")
        print(f"        Opset:             {opset}")
        print(f"        Quantized nodes:   {len(quant_nodes)}")

        if quant_nodes:
            print(f"   [OK] Model is quantized (INT8)")
        else:
            print(f"   [WARNING] No quantization nodes found — check config")

        return True

    except Exception as e:
        print(f"   [ERROR] Validation failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Quantize DeepLabV3 model using Olive and AMD Quark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quantize_deeplabv3.py
  python quantize_deeplabv3.py --calib-dir ./calib_data
  python quantize_deeplabv3.py --model deeplabv3_resnet50.onnx --output deeplabv3_i8.onnx
        """,
    )
    parser.add_argument("--model", default=None,
                        help="Input FP32 ONNX model (default: deeplabv3_resnet50.onnx)")
    parser.add_argument("--calib-dir", default=None,
                        help="Calibration image directory (default: ./calib_data)")
    parser.add_argument("--output", default=None,
                        help="Output filename (default: deeplabv3_resnet50_i8.onnx)")
    parser.add_argument("--config", default="olive_config_deeplabv3.json",
                        help="Olive config file (default: olive_config_deeplabv3.json)")
    args = parser.parse_args()

    model_dir = Path(__file__).parent

    # Resolve model path
    if args.model is None:
        model_path = model_dir / "deeplabv3_resnet50.onnx"
    else:
        model_path = Path(args.model)

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("Run: python download_model.py")
        return 1

    # Resolve config
    config_file = Path(args.config)
    if not config_file.exists():
        config_file = model_dir / args.config
    if not config_file.exists():
        print(f"[ERROR] Config not found: {args.config}")
        return 1

    # Resolve calib dir
    calib_dir = Path(args.calib_dir) if args.calib_dir else model_dir / "calib_data"

    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        return 1

    # Quantize
    quantized = run_quantization(model_path, calib_dir, args.output, config_file)

    if quantized is None:
        return 1

    # Validate
    if not validate_model(quantized):
        return 1

    size_mb = quantized.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 70)
    print("SUCCESS: DeepLabV3 model quantized to INT8")
    print("=" * 70)
    print(f"  Quantized model: {quantized}")
    print(f"  Model size:      {size_mb:.1f} MB")
    print("\nRun quantized model:")
    print(f"  cd ../python")
    print(f"  python run_model.py --model ../model/{quantized.name} --ep_policy NPU")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
