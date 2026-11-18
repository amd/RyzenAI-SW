#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Export Hugging Face facebook/detr-resnet-50 (revision no_timm) to the FP32 ONNX model that the Olive recipe quantizes.

Usage (from this folder):
    python export_fp32.py                 # writes detr_resnet50_fp32.onnx next to this script
    python export_fp32.py --out-dir DIR

Static 1x3x800x800 input, opset 20. The TensorQuantOverrides in olive_ryzen_config.json refer to
initializer names produced by this export (onnx::MatMul_46xx ...), so use the pinned torch /
transformers versions. Expected SHA-256 with the pinned environment: 4d11385bcd82...
"""
import argparse
from pathlib import Path

import torch
from transformers import DetrForObjectDetection

HERE = Path(__file__).resolve().parent
OPSET = 20


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=HERE)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").eval()
    path = args.out_dir / "detr_resnet50_fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            torch.randn(1, 3, 800, 800, dtype=torch.float32),
            str(path),
            opset_version=OPSET,
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits", "pred_boxes", "last_hidden_state", "encoder_last_hidden_state"],
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
