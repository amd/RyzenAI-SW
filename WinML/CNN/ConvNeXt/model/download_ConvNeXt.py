import sys
import io

# Fix Unicode encoding issues on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import torchvision.models as models
import torch.onnx
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

# Load a pre-trained ConvNeXt model
model = models.convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
model.eval()  # Set the model to evaluation mode

# Create a dummy input tensor with the same size as the model's input
dummy_input = torch.randn(1, 3, 224, 224)

# Define the path where the ONNX model will be saved
onnx_model_path = "convnext_small.onnx"

# Export the model to ONNX format
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    dynamo=False  # Disable dynamo to avoid dynamic_axes warning
)

print(f"Model has been successfully exported to {onnx_model_path}")
