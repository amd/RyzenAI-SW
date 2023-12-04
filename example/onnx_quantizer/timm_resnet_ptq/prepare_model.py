#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import timm
import torch
import sys

if len(sys.argv) < 3:
    print("Usage: python prepare_model.py <timm_model_name> <onnx path>")
    sys.exit(1)

model_name = sys.argv[1]
model_out = sys.argv[2]
model = timm.create_model(model_name, pretrained=True)
model = model.eval()

data_config = timm.data.resolve_model_data_config(
    model,
    use_test_size=True,
)
transforms = timm.data.create_transform(**data_config, is_training=False)

device = torch.device("cpu")
batch_size = 1
random_input = torch.randn((batch_size,) +
                           tuple(data_config['input_size'])).to(device)

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

print("Converting to ONNX complete")
