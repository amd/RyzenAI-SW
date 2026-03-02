import torch  
import torchvision.models as models  
import torch.onnx  
from torchvision.models import resnet50, ResNet50_Weights  

# Load a pre-trained ResNet model  
model = models.resnet50(weights=ResNet50_Weights.DEFAULT) 
model.eval()  # Set the model to evaluation mode  
  
# Create a dummy input tensor with the same size as the model's input  
dummy_input = torch.randn(1, 3, 224, 224)  
  
# Define the path where the ONNX model will be saved  
onnx_model_path = "resnet50.onnx"

# Export the model to ONNX format  
torch.onnx.export(  
    model,   
    dummy_input,   
    onnx_model_path,   
    opset_version=17,   
    input_names=['input'],   
    output_names=['output'],  
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  
)  
  
print(f"Model has been successfully exported to {onnx_model_path}")  
