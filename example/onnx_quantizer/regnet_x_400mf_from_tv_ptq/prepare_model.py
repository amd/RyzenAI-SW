#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import sys
import onnx
import torch
import torchvision
from torchvision.models import get_model

if len(sys.argv) < 4:
    print(
        "Usage: python prepare_model.py <tv_model_name> <tv model weights name> <onnx path>"
    )
    sys.exit(1)

model_name = sys.argv[1]
weights_name = sys.argv[2]
model_out = sys.argv[3]
model = get_model(model_name, weights=weights_name)
model = model.eval()

weights = torchvision.models.get_weight(weights_name)
input_size = weights.transforms.keywords['crop_size']

device = torch.device("cpu")
batch_size = 1
random_input = torch.randn(1, 3, input_size, input_size).to(device)

torch.onnx.export(
    model,
    random_input,
    model_out,
    export_params=True,
    do_constant_folding=True,
    opset_version=17,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {
            0: 'batch_size'
        },
        'output': {
            0: 'batch_size'
        }
    },
    verbose=True,
)
onnx_model = onnx.load(model_out)
onnx.checker.check_model(onnx_model)

print("Converting to ONNX complete")
