#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Export torchvision ConvNeXt-Tiny (IMAGENET1K_V1) to the FP32 ONNX model that the Olive recipe quantizes.

Usage (from this folder):
    python export_fp32.py                 # writes convnext_fp32.onnx next to this script
    python export_fp32.py --out-dir DIR

Opset 20 keeps a native Gelu op; opset 17 decomposes it into Erf and several elementwise operators.
The custom fold pass in olive_ryzen/ is written for the opset 20 graph.
Expected SHA-256 with the pinned environment: ee1812af9ff9...
"""
import argparse
from pathlib import Path

import torch
from torchvision import models

HERE = Path(__file__).resolve().parent
OPSET = 20


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=HERE)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1).eval()
    path = args.out_dir / "convnext_fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            torch.randn(1, 3, 224, 224, dtype=torch.float32),
            str(path),
            opset_version=OPSET,
            external_data=False,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
