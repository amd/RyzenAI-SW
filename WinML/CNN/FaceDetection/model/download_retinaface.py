#!/usr/bin/env python3
"""
Download RetinaFace ONNX Model and Convert to Opset 21

This script downloads a pre-exported RetinaFace ONNX model and converts it to:
- Opset 21
- Static batch size (1)
- NCHW format for NPU compatibility

Supported backbones:
- mobilenet (default): MobileNetV1 0.50 - lightweight, fast
- resnet: ResNet34 - more accurate, larger model

Source: https://github.com/yakhyo/retinaface-pytorch/releases
"""

import sys
import argparse
from pathlib import Path
import urllib.request

# Model configurations
MODEL_CONFIGS = {
    'mobilenet': {
        'url': 'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1_0.50.onnx',
        'filename': 'retinaface_mv1_0.50_original.onnx',
        'name': 'MobileNetV1 0.50',
        'size': '6.5 MB',
        'num_anchors': 16800,  # For 640x640 input
    },
    'resnet': {
        'url': 'https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r34.onnx',
        'filename': 'retinaface_r34_original.onnx',
        'name': 'ResNet34',
        'size': '109 MB',
        'num_anchors': 16800,  # For 640x640 input
    }
}

def download_and_convert(backbone='mobilenet'):
    """Download ONNX model and convert to opset 21 with static shapes"""

    # Validate backbone
    if backbone not in MODEL_CONFIGS:
        print(f"[ERROR] Unknown backbone: {backbone}")
        print(f"Available options: {', '.join(MODEL_CONFIGS.keys())}")
        return 1

    config = MODEL_CONFIGS[backbone]
    model_dir = Path(__file__).parent
    downloaded_model = model_dir / config['filename']
    final_model = model_dir / "retinaface_mobilenet.onnx"

    if final_model.exists():
        print(f"[OK] Model already exists: {final_model}")
        return 0

    print("=" * 80)
    print(f"RetinaFace {config['name']} - Direct ONNX Download & Conversion")
    print("=" * 80)

    # Step 1: Download pre-exported ONNX model
    print(f"\n[1/2] Downloading pre-exported {config['name']} ONNX model...")
    print(f"   Expected size: ~{config['size']}")

    if not downloaded_model.exists():
        try:
            # Download ONNX model directly from GitHub releases
            url = config['url']
            print(f"   Downloading from: {url}")
            print(f"   This may take a few minutes for larger models...")

            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(downloaded_model, 'wb') as out_file:
                out_file.write(response.read())

            size_mb = downloaded_model.stat().st_size / (1024 * 1024)
            print(f"[OK] ONNX model downloaded: {size_mb:.1f} MB")

        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            print(f"\nManual download:")
            print(f"1. Download from: {url}")
            print(f"2. Save as: {downloaded_model}")
            return 1
    else:
        size_mb = downloaded_model.stat().st_size / (1024 * 1024)
        print(f"[OK] Model already downloaded: {size_mb:.1f} MB")

    # Step 2: Convert to opset 21 and fix dynamic shapes
    print("\n[2/2] Converting ONNX model to opset 21 with static batch...")

    try:
        import onnx
        from onnx import version_converter, helper, shape_inference

        print("   Loading ONNX model...")
        model = onnx.load(str(downloaded_model))

        # Display original model info
        print(f"\n   Original Model Info:")
        print(f"   - Opset Version: {model.opset_import[0].version}")
        print(f"   - Inputs:")
        for inp in model.graph.input:
            shape = [d.dim_value if d.dim_value > 0 else d.dim_param
                    for d in inp.type.tensor_type.shape.dim]
            print(f"     {inp.name}: {shape}")

        # Convert to opset 21 if needed
        target_opset = 21
        current_opset = model.opset_import[0].version

        if current_opset != target_opset:
            print(f"\n   Converting opset {current_opset} -> {target_opset}...")
            model = version_converter.convert_version(model, target_opset)
            print(f"   [OK] Converted to opset {target_opset}")
        else:
            print(f"   [OK] Already at opset {target_opset}")

        # Fix input and output sizes to static dimensions
        print(f"\n   Setting static input size [1, 3, 640, 640]...")

        # Set input to fixed size: [1, 3, 640, 640] (NCHW)
        for inp in model.graph.input:
            if inp.type.tensor_type.shape.dim:
                dims = inp.type.tensor_type.shape.dim
                # Set batch=1, channels=3, height=640, width=640
                if len(dims) == 4:
                    dims[0].dim_value = 1
                    dims[0].ClearField('dim_param')
                    dims[1].dim_value = 3
                    dims[1].ClearField('dim_param')
                    dims[2].dim_value = 640
                    dims[2].ClearField('dim_param')
                    dims[3].dim_value = 640
                    dims[3].ClearField('dim_param')
                    print(f"     Input '{inp.name}': [1, 3, 640, 640]")

        print(f"\n   Setting static output sizes...")
        # Fix all output dimensions to static values
        for out in model.graph.output:
            if out.type.tensor_type.shape.dim:
                dims = out.type.tensor_type.shape.dim
                # Set batch dimension to 1
                if len(dims) > 0:
                    dims[0].dim_value = 1
                    dims[0].ClearField('dim_param')
                # Set other dimensions to static values if they exist
                for i in range(1, len(dims)):
                    if dims[i].dim_param or dims[i].dim_value == 0:
                        # Clear dynamic dimension
                        dims[i].ClearField('dim_param')
                        # Will be inferred by shape inference

                shape_str = [d.dim_value if d.dim_value > 0 else "?" for d in dims]
                print(f"     Output '{out.name}': {shape_str}")

        # Run shape inference to propagate static shapes
        print("   Running shape inference to propagate static shapes...")
        model = shape_inference.infer_shapes(model)

        # Calculate output dimensions for 640x640 input
        # RetinaFace has 3 pyramid levels (stride 8, 16, 32) with 2 anchors each
        # Level 1 (stride 8): 80x80 = 6400, Level 2 (stride 16): 40x40 = 1600, Level 3 (stride 32): 20x20 = 400
        # Total: (6400 + 1600 + 400) * 2 anchors = 16800 anchors
        num_anchors = config['num_anchors']

        # Explicitly set output dimensions based on RetinaFace architecture
        print("   Setting fixed output dimensions...")

        # Expected output shapes in order: loc, conf, landms
        output_shapes_list = [
            [1, num_anchors, 4],      # bounding boxes: [batch, anchors, 4_coords]
            [1, num_anchors, 2],      # confidence: [batch, anchors, 2_classes]
            [1, num_anchors, 10]      # landmarks: [batch, anchors, 10_coords (5 points * 2)]
        ]

        # Get value_info from shape inference to check inferred shapes
        value_info_map = {}
        for value in model.graph.value_info:
            if value.type.tensor_type.shape.dim:
                shape = [d.dim_value for d in value.type.tensor_type.shape.dim]
                value_info_map[value.name] = shape

        # Fix output shapes
        for idx, out in enumerate(model.graph.output):
            if out.type.tensor_type.shape.dim:
                dims = out.type.tensor_type.shape.dim

                # First try to use inferred shape from value_info
                if out.name in value_info_map:
                    inferred_shape = value_info_map[out.name]
                    print(f"     Using inferred shape for '{out.name}': {inferred_shape}")
                    for i, dim_val in enumerate(inferred_shape):
                        if i < len(dims) and dim_val > 0:
                            dims[i].dim_value = dim_val
                            dims[i].ClearField('dim_param')
                else:
                    # Use expected shape based on index
                    if idx < len(output_shapes_list):
                        target_shape = output_shapes_list[idx]
                        print(f"     Setting fixed shape for '{out.name}': {target_shape}")
                        for i, target_dim in enumerate(target_shape):
                            if i < len(dims):
                                dims[i].dim_value = target_dim
                                dims[i].ClearField('dim_param')
                    else:
                        # Fallback: at least fix batch=1 and clear dynamic params
                        if len(dims) > 0:
                            dims[0].dim_value = 1
                            dims[0].ClearField('dim_param')
                        for dim in dims:
                            if dim.dim_param:
                                dim.ClearField('dim_param')

                # Final check: forcefully set any remaining dynamic dimensions
                for i, dim in enumerate(dims):
                    if dim.dim_value <= 0:
                        # Last resort: use expected shape if available
                        if idx < len(output_shapes_list) and i < len(output_shapes_list[idx]):
                            dim.dim_value = output_shapes_list[idx][i]
                            dim.ClearField('dim_param')
                            print(f"       Forcing dimension {i} to {output_shapes_list[idx][i]}")

                # Display final shape
                final_shape = [d.dim_value if d.dim_value > 0 else "?" for d in dims]
                if "?" in [str(s) for s in final_shape]:
                    print(f"     WARNING: Output '{out.name}' still has dynamic dimensions: {final_shape}")
                    print(f"       Manual fix may be required")
                else:
                    print(f"     ✓ Output '{out.name}': {final_shape}")

        # Validate model
        print("   Validating model...")
        onnx.checker.check_model(model)

        # Save converted model
        print(f"   Saving converted model...")
        onnx.save(model, str(final_model))

        size_mb = final_model.stat().st_size / (1024 * 1024)
        print(f"[OK] Model converted and saved: {size_mb:.1f} MB")

        # Display final model info
        print(f"\n   Final Model Info:")
        print(f"   - Opset Version: {model.opset_import[0].version}")
        print(f"   - Inputs:")
        for inp in model.graph.input:
            shape = [d.dim_value if d.dim_value > 0 else d.dim_param
                    for d in inp.type.tensor_type.shape.dim]
            print(f"     {inp.name}: {shape}")
        print(f"   - Outputs:")
        for out in model.graph.output:
            shape = [d.dim_value if d.dim_value > 0 else d.dim_param
                    for d in out.type.tensor_type.shape.dim]
            print(f"     {out.name}: {shape}")

        print("\n" + "=" * 80)
        print("SUCCESS: RetinaFace ONNX model ready for NPU deployment")
        print("=" * 80)
        print(f"Model Details:")
        print(f"  - Format: ONNX FP32")
        print(f"  - Opset Version: 21")
        print(f"  - Backbone: {config['name']}")
        print(f"  - Batch Size: Static (1)")
        print(f"  - Input Shape: [1, 3, 640, 640]")
        print(f"  - Location: {final_model}")
        print(f"\nThe FP32 model will be automatically converted to BF16 during NPU compilation.")
        print("=" * 80)

        return 0

    except ImportError:
        print("[ERROR] ONNX package not installed")
        print("Install with: pip install onnx")
        return 1
    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    parser = argparse.ArgumentParser(
        description='Download and convert RetinaFace ONNX models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download MobileNetV1 model (default, lightweight)
  python download_retinaface.py

  # Download ResNet34 model (more accurate, larger)
  python download_retinaface.py --backbone resnet

Available backbones:
  mobilenet  - MobileNetV1 0.50 (~6.5 MB, fast)
  resnet     - ResNet34 (~109 MB, accurate)
        """
    )
    parser.add_argument(
        '--backbone',
        type=str,
        default='mobilenet',
        choices=['mobilenet', 'resnet'],
        help='Model backbone architecture (default: mobilenet)'
    )

    args = parser.parse_args()
    sys.exit(download_and_convert(args.backbone))

if __name__ == "__main__":
    main()
