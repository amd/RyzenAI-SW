import torch
import torch.nn as nn
import os
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
from PIL import Image
import numpy as np

class CLIPONNXWrapper(nn.Module):
    """Wrapper for CLIP model to ensure proper ONNX export"""
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return (
            outputs.logits_per_image,
            outputs.logits_per_text,
            outputs.text_embeds,
            outputs.image_embeds
        )

def export_model_to_onnx():
    """
    Download CLIP ViT-Base-Patch16 model and export to ONNX format
    """
    # Define paths
    model_dir = Path("./model")
    model_dir.mkdir(exist_ok=True)
    onnx_model_path = model_dir / "model.onnx"

    # Model name
    model_name = 'openai/clip-vit-base-patch16'
    print(f"Downloading model: {model_name}")

    # Load the pre-trained model and processor using safetensors format
    # This avoids the CVE-2025-32434 vulnerability in torch.load
    # Set attn_implementation to 'eager' to avoid scaled_dot_product_attention ONNX export issues
    clip_model = CLIPModel.from_pretrained(
        model_name,
        use_safetensors=True,
        attn_implementation="eager"
    )
    processor = CLIPProcessor.from_pretrained(model_name)

    print(f"Model downloaded successfully")

    # Wrap model for ONNX export
    model = CLIPONNXWrapper(clip_model)
    model.eval()

    # Create dummy inputs for export
    # Create a dummy image (224x224 RGB)
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    # Create dummy text prompts - 10 diverse prompts for export
    # This will create fixed input shapes: [1, 3, 224, 224] for image and [10, 77] for text
    dummy_text = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a bird",
        "a photo of a person",
        "a photo of a car",
        "a photo of a bicycle",
        "a photo of a chair",
        "a photo of a bottle",
        "a photo of an airplane",
        "a photo of a boat"
    ]

    # Process inputs
    # max_length=77 is CLIP's standard text context length (fixed in the architecture)
    inputs = processor(
        text=dummy_text,
        images=dummy_image,
        return_tensors="pt",
        padding="max_length",
        max_length=77,
        truncation=True
    )

    print(f"Exporting model to ONNX format...")
    # print(f"Input shapes - pixel_values: {inputs['pixel_values'].shape}, input_ids: {inputs['input_ids'].shape}")

    # Export to ONNX with fixed shapes (1 image, 10 text prompts, 77 tokens)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs['pixel_values'], inputs['input_ids'], inputs['attention_mask']),
            str(onnx_model_path),
            export_params=True,
            input_names=['pixel_values', 'input_ids', 'attention_mask'],
            output_names=['logits_per_image', 'logits_per_text', 'text_embeds', 'image_embeds'],
            opset_version=14,
            do_constant_folding=True,
            verbose=False
        )

    print(f"Model successfully exported to: {onnx_model_path}")
    # print(f"Model file size: {onnx_model_path.stat().st_size / (1024*1024):.2f} MB")

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_model_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation: PASSED")
    except ImportError:
        print("Note: Install 'onnx' package to validate the exported model")
    except Exception as e:
        print(f"Warning: Model validation failed: {e}")

    return str(onnx_model_path)

if __name__ == "__main__":
    try:
        model_path = export_model_to_onnx()
        print(f"\nDownload and export complete!")
        print(f"You can now use the model with: python run_clip.py")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
