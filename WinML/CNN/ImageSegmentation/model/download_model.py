#!/usr/bin/env python3
"""
Download and Export DeepLabV3 ResNet50 Model to ONNX

This script downloads the DeepLabV3+ model with ResNet50 backbone from torchvision
and exports it to ONNX FP32 format for NPU deployment via WinML.

Steps performed:
1. Download pretrained DeepLabV3 ResNet50 weights from torchvision
2. Wrap model for direct segmentation output (argmax of logits)
3. Export PyTorch model to ONNX format with static shapes
4. Fix static input/output dimensions (shape inference)
5. Verify exported model structure

The FP32 model will be automatically converted to BF16 during NPU compilation
for optimal performance.

Model: torchvision.models.segmentation.deeplabv3_resnet50
Classes: 21 (PASCAL VOC 2012 semantic segmentation classes)
Input:  [1, 3, 520, 520] NCHW, ImageNet-normalized
Output: [1, 21, 520, 520] class logits per pixel
"""

import sys
import subprocess
from pathlib import Path


def install_dependencies():
    """Install required packages for model export"""
    print("\n[1/4] Installing dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "torch", "torchvision", "onnx", "onnxruntime"],
            check=True, capture_output=True
        )
        print("[OK] Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Failed to install dependencies: {e}")
        print("   Continuing anyway...")


def download_and_export():
    """Download DeepLabV3 and export to ONNX"""
    import torch
    import torch.nn as nn
    import torchvision.models.segmentation as seg_models
    import onnx

    model_dir = Path(__file__).parent
    output_path = model_dir / "deeplabv3_resnet50.onnx"

    if output_path.exists():
        print(f"[OK] Model already exists: {output_path}")
        return 0

    print("\n[2/4] Downloading pretrained DeepLabV3 ResNet50...")
    print("   Source: torchvision pretrained weights (PASCAL VOC)")
    print("   This may take a few minutes (~167 MB)...")

    try:
        weights = seg_models.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        model = seg_models.deeplabv3_resnet50(weights=weights)
        model.eval()
        print("[OK] Model downloaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        return 1

    # Wrap model to output logits directly (single tensor output for ONNX)
    class DeepLabWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)['out']

    wrapped = DeepLabWrapper(model)
    wrapped.eval()

    print("\n[3/4] Exporting to ONNX format...")
    input_size = (520, 520)
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    try:
        torch.onnx.export(
            wrapped,
            dummy_input,
            str(output_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,   # static shapes for NPU
            do_constant_folding=True,
        )
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"[OK] Model exported: {output_path.name} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"[ERROR] ONNX export failed: {e}")
        return 1

    # Fix static shapes and verify model
    try:
        model_proto = onnx.load(str(output_path))

        # Run shape inference so all intermediate tensors get concrete dims
        model_proto = onnx.shape_inference.infer_shapes(model_proto)

        # Explicitly pin input dims to [1, 3, 520, 520]
        inp = model_proto.graph.input[0]
        for dim, value in zip(inp.type.tensor_type.shape.dim, [1, 3, 520, 520]):
            dim.dim_value = value
            dim.ClearField("dim_param")

        # Explicitly pin output dims to [1, 21, 520, 520]
        out = model_proto.graph.output[0]
        for dim, value in zip(out.type.tensor_type.shape.dim, [1, 21, 520, 520]):
            dim.dim_value = value
            dim.ClearField("dim_param")

        onnx.checker.check_model(model_proto)
        onnx.save(model_proto, str(output_path))

        inp_shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        out_shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"[OK] Model verified:")
        print(f"     Input:  {inp.name} {inp_shape}")
        print(f"     Output: {out.name} {out_shape}")
        print(f"     Nodes:  {len(model_proto.graph.node)}")
    except Exception as e:
        print(f"[WARNING] Model verification/shape-fix failed: {e}")

    print("\n" + "=" * 60)
    print("DeepLabV3 model ready for deployment!")
    print(f"  Model:  {output_path}")
    print(f"  Input:  [1, 3, 520, 520]  (NCHW, ImageNet-normalized)")
    print(f"  Output: [1, 21, 520, 520] (class logits per pixel)")
    print("\nPASCAL VOC classes (21 total):")
    classes = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "dining table", "dog",
        "horse", "motorbike", "person", "potted plant", "sheep",
        "sofa", "train", "tv/monitor"
    ]
    for i, c in enumerate(classes):
        print(f"  {i:2d}: {c}")
    print("=" * 60)
    return 0


def main():
    print("=" * 60)
    print("DeepLabV3 ResNet50 Model Download & Export")
    print("=" * 60)
    install_dependencies()
    return download_and_export()


if __name__ == "__main__":
    sys.exit(main())
