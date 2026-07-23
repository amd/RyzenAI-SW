#!/usr/bin/env python3
"""
Download YOLOv8m Object Detection Model and Export to Optimized ONNX

This script downloads and prepares the YOLOv8m ONNX model for WinML deployment:
1. Download YOLOv8m PyTorch weights from Ultralytics
2. Export to ONNX with static shapes and opset 21
3. Optimize for NPU deployment

The output model is ready for both FP32 inference and INT8 quantization.
"""

import sys
import argparse
from pathlib import Path
import urllib.request


def download_and_export_yolov8(model_size='m'):
    """Download YOLOv8 and export to optimized ONNX"""

    model_dir = Path(__file__).parent
    output_filename = f"yolov8{model_size}.onnx"
    output_path = model_dir / output_filename

    if output_path.exists():
        print(f"[OK] Model already exists: {output_path}")
        return 0

    print("=" * 80)
    print(f"YOLOv8{model_size.upper()} Object Detection - Download & Export")
    print("=" * 80)

    # Install ultralytics if needed
    try:
        import ultralytics
        print(f"[OK] Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("\n[INFO] Installing ultralytics package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        import ultralytics

    # Import YOLO after ensuring ultralytics is installed
    from ultralytics import YOLO

    print(f"\n[1/3] Downloading YOLOv8{model_size} PyTorch model...")
    try:
        # This will download the .pt model if not already present
        model = YOLO(f'yolov8{model_size}.pt')
        print(f"[OK] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return 1

    print(f"\n[2/3] Exporting to ONNX...")
    print("  - Setting static input shape: [1, 3, 640, 640]")
    print("  - Opset version: 21")
    print("  - Simplifying graph")

    try:
        # Export to ONNX with optimal settings
        # YOLOv8 export automatically handles:
        # - Static batch size
        # - Graph optimization
        # - Dynamic axes removal
        export_path = model.export(
            format='onnx',
            imgsz=640,
            opset=21,
            simplify=True,
            dynamic=False,  # Static batch size
            half=False,     # FP32 (not FP16)
        )

        print(f"[OK] Exported to: {export_path}")

        # Move to desired location if needed
        import shutil
        exported_file = Path(export_path)
        if exported_file != output_path:
            shutil.move(str(exported_file), str(output_path))
            print(f"[OK] Moved to: {output_path}")

    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\n[3/3] Verifying ONNX model...")
    try:
        import onnx
        model = onnx.load(str(output_path))

        # Check input/output shapes
        print(f"\nModel Information:")
        print(f"  - Opset: {model.opset_import[0].version}")
        print(f"  - Inputs:")
        for inp in model.graph.input:
            shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in inp.type.tensor_type.shape.dim]
            print(f"    - {inp.name}: {shape}")

        print(f"  - Outputs:")
        for out in model.graph.output:
            shape = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in out.type.tensor_type.shape.dim]
            print(f"    - {out.name}: {shape}")

        final_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  - Size: {final_size:.2f} MB")
        print(f"  - Total nodes: {len(model.graph.node)}")

        print("\n" + "=" * 80)
        print("SUCCESS: YOLOv8 ONNX Model Ready")
        print("=" * 80)
        print(f"Model: {output_path}")
        print(f"Format: ONNX FP32, Opset 21")
        print(f"Ready for NPU deployment and INT8 quantization!")
        print("=" * 80)

        return 0

    except ImportError:
        print(f"[WARNING] Cannot verify model - onnx package not installed")
        print(f"          Model export completed, but verification skipped")
        return 0
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='Download and export YOLOv8 to optimized ONNX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script automates the YOLOv8 download and export process:
  1. Download YOLOv8 PyTorch weights from Ultralytics
  2. Export to ONNX with static shapes (640x640)
  3. Simplify graph for optimal NPU performance
  4. Verify model structure

The output is ready for WinML deployment and quantization!

Examples:
  python download_yolov8.py           # YOLOv8m (default, recommended)
  python download_yolov8.py -s n      # YOLOv8n (nano - fastest)
  python download_yolov8.py -s s      # YOLOv8s (small - balanced)
  python download_yolov8.py -s l      # YOLOv8l (large - most accurate)
        """
    )

    parser.add_argument(
        '-s', '--size',
        type=str,
        default='m',
        choices=['n', 's', 'm', 'l', 'x'],
        help='Model size: n(ano), s(mall), m(edium), l(arge), x(large) (default: m)'
    )

    args = parser.parse_args()
    sys.exit(download_and_export_yolov8(args.size))


if __name__ == "__main__":
    main()