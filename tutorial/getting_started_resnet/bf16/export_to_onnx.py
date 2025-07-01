#!/usr/bin/env python3
import os
import argparse
import torch
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor
from transformers.modeling_utils import PreTrainedModel
from PIL import Image
import numpy as np
import timm

def is_timm_model(model_name: str) -> bool:
    """
    Check if the model name is a timm model
    """
    return model_name.startswith('timm/')

def is_vision_model(model_config) -> bool:
    """
    Determine if the model is a vision model based on its config
    """
    # Check for common vision model architectures
    model_type = getattr(model_config, 'model_type', '').lower()
    vision_architectures = {'vit', 'swin', 'deit', 'beit', 'convnext', 'resnet'}
    return any(arch in model_type for arch in vision_architectures)

def prepare_vision_input(model, processor=None):
    """
    Prepare dummy input for vision models
    """
    # Create a dummy image (black square)
    dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    
    if processor is not None:
        # For HF models with processor
        inputs = processor(images=dummy_image, return_tensors="pt")
    else:
        # For timm models - use standard normalization
        dummy_input = torch.zeros(1, 3, 224, 224)
        inputs = {'pixel_values': dummy_input}
    
    return inputs

def prepare_text_input(tokenizer):
    """
    Prepare dummy input for text models
    """
    return tokenizer("Hello, world!", return_tensors="pt", padding=True, truncation=True, max_length=8)

def load_model(model_name: str):
    """
    Load model based on its type (timm or huggingface)
    """
    if is_timm_model(model_name):
        # Remove 'timm/' prefix for timm.create_model
        timm_model_name = model_name.replace('timm/', '', 1)
        model = timm.create_model(timm_model_name, pretrained=True)
        return model, None, True
    else:
        model = AutoModel.from_pretrained(model_name)
        is_vision = is_vision_model(model.config)
        
        # Load appropriate processor
        if is_vision:
            try:
                processor = AutoImageProcessor.from_pretrained(model_name)
            except Exception:
                processor = AutoFeatureExtractor.from_pretrained(model_name)
        else:
            processor = AutoTokenizer.from_pretrained(model_name)
            
        return model, processor, is_vision

def export_model_to_onnx(model_name: str, output_dir: str, opset_version: int = 19) -> str:
    """
    Export a Hugging Face PyTorch model to ONNX format.
    
    Args:
        model_name (str): Name or path of the Hugging Face model
        output_dir (str): Directory to save the ONNX model
        opset_version (int): ONNX opset version to use
        
    Returns:
        str: Path to the saved ONNX model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {model_name}")
    model, processor, is_vision = load_model(model_name)
    
    # Prepare input based on model type
    dummy_input = prepare_vision_input(model, processor) if is_vision else prepare_text_input(processor)
    
    # Set model to evaluation mode
    model.eval()
    
    # Prepare output path
    model_name_safe = os.path.basename(model_name)
    output_path = os.path.join(output_dir, f"{model_name_safe}.onnx")
    
    # Export the model
    print(f"Exporting model to ONNX format (opset version: {opset_version})")
    with torch.no_grad():
        if is_vision:
            # For vision models, typically only pixel_values is needed
            print("Configuring export for vision model")
            input_names = ['pixel_values']
            dynamic_axes = {
                'pixel_values': {0: 'batch_size'}, #, 2: 'height', 3: 'width'},
                'output': {0: 'batch_size'}
            }
            inputs = (dummy_input['pixel_values'],)
        else:
            # For text models, we need input_ids and attention_mask
            print("Configuring export for text model")
            input_names = ['input_ids', 'attention_mask']
            dynamic_axes = {
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size', 1: 'sequence'}
            }
            inputs = (dummy_input['input_ids'], dummy_input['attention_mask'])
        
        torch.onnx.export(
            model,                     # PyTorch model
            inputs,                    # model input
            output_path,              # output path
            opset_version=opset_version,
            input_names=input_names,  # model input names
            output_names=['output'],   # model output names
            dynamic_axes=dynamic_axes
        )
    
    print(f"Model exported successfully to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Export Hugging Face PyTorch model to ONNX')
    parser.add_argument('--model', type=str, required=True,
                        help='Name or path of the Hugging Face model')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for the ONNX model')
    parser.add_argument('--opset', type=int, default=19,
                        help='ONNX opset version to use')
    
    args = parser.parse_args()
    
    output_path = export_model_to_onnx(
        model_name=args.model,
        output_dir=args.output_dir,
        opset_version=args.opset
    )
    
    print(f"ONNX model path: {output_path}")

if __name__ == "__main__":
    main() 
