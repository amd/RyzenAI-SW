#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Export Real-ESRGAN x4plus (RealESRGAN_x4plus.pth, RRDBNet from basicsr) to the FP32 ONNX model that the
Olive recipe quantizes.

Usage (from this folder):
    python export_fp32.py                 # writes esrgan_fp32.onnx next to this script
    python export_fp32.py --out-dir DIR

Downloads the weights to torch_model/ on first use. Static 1x3x250x250 input (1x3x1000x1000 output),
opset 20. Expected SHA-256 with the pinned environment: 6ea2f4d61fa0...
"""
import argparse
import sys
import urllib.request
from pathlib import Path

import torch
import torchvision.transforms.functional

# basicsr 1.4.2 imports torchvision.transforms.functional_tensor, which torchvision 0.17+
# removed; point it at the module that now holds the same functions.
sys.modules.setdefault("torchvision.transforms.functional_tensor", torchvision.transforms.functional)
from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: E402

HERE = Path(__file__).resolve().parent
OPSET = 20
WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=HERE)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    weights = HERE / "torch_model" / "RealESRGAN_x4plus.pth"
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_URL, weights)
    state = torch.load(weights, map_location="cpu")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model.load_state_dict(state["params_ema" if "params_ema" in state else "params"], strict=True)
    model.eval()
    path = args.out_dir / "esrgan_fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            torch.randn(1, 3, 250, 250, dtype=torch.float32),
            str(path),
            opset_version=OPSET,
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
