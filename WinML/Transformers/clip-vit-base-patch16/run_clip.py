import subprocess
import json
import sys
import os
import onnxruntime as ort
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests
from io import BytesIO
import argparse
import numpy as np

def register_execution_providers():
    worker_script = os.path.abspath('winml.py')
    result = subprocess.check_output([sys.executable, worker_script], text=True)
    paths = json.loads(result)
    print("paths:", paths)
    for name, lib_path in paths.items():
        if not lib_path or not os.path.exists(lib_path):
            print(f"Skipping execution provider {name}: invalid or missing library path")
            continue
        ort.register_execution_provider_library(name, lib_path)
        print(f"Registered execution provider: {name} with library path: {lib_path}")

def load_image_from_url(url):
    """Load image from URL and return PIL Image"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        raise RuntimeError(f"Failed to load image from URL: {url}\nError: {e}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run CLIP-ViT inference with NPU or CPU')
    parser.add_argument('--ep_policy', type=str.upper, default='NPU', choices=['NPU', 'CPU', 'GPU', 'DEFAULT'],
                        help='Set execution provider policy (NPU, CPU, GPU, DEFAULT). Default: NPU')
    parser.add_argument('--model', type=str, default='./model/model.onnx',
                        help='Path to ONNX model (default: ./model/model.onnx)')
    parser.add_argument('--image_url', type=str,
                        default='http://images.cocodataset.org/val2017/000000039769.jpg',
                        help='URL of image to classify (default: COCO cat image)')
    args = parser.parse_args()

    onnx_model_path = args.model

    # Validate model path exists
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"Model file not found: {onnx_model_path}")

    print("Registering execution providers ...")
    register_execution_providers()

    print(f"Model path: {onnx_model_path}")
    print(f"Image URL: {args.image_url}")

    # Load image
    print("Loading image ...")
    image = load_image_from_url(args.image_url)

    # Define text prompts
    # NOTE: Must be exactly 10 prompts to match quantized model shape [10, 77]
    # max_length=77 is CLIP's standard text context length
    text_prompts = [
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

    # Load processor and prepare inputs
    print("Preparing inputs ...")
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16', use_fast=False)
    inputs = processor(
        text=text_prompts,
        images=image,
        return_tensors="pt",
        padding="max_length",
        max_length=77,  # CLIP's fixed text context length
        truncation=True
    )

    print("Creating session ...")
    session_options = ort.SessionOptions()

    # Set execution provider policy
    policy_map = {
        'NPU': ort.OrtExecutionProviderDevicePolicy.PREFER_NPU,
        'CPU': ort.OrtExecutionProviderDevicePolicy.PREFER_CPU,
        'DEFAULT': ort.OrtExecutionProviderDevicePolicy.DEFAULT
    }

    policy = policy_map.get(args.ep_policy)
    if policy:
        session_options.set_provider_selection_policy(policy)
        print(f"Set provider selection policy to: {args.ep_policy}")

    session = ort.InferenceSession(
        onnx_model_path,
        sess_options=session_options,
    )

    # Report active execution providers
    try:
        eps = session.get_providers()
        print(f"Active execution providers (priority order): {eps}")
        if eps:
            print(f"Primary provider (highest priority): {eps[0]}")
    except Exception as e:
        print(f"Warning: unable to query active execution providers: {e}")

    print("Running inference ...")

    # Prepare ONNX inputs
    onnx_inputs = {
        "pixel_values": inputs['pixel_values'].cpu().numpy(),
        "input_ids": inputs['input_ids'].cpu().numpy().astype(np.int64),
        "attention_mask": inputs['attention_mask'].cpu().numpy().astype(np.int64)
    }

    # Run inference
    outputs = session.run(None, onnx_inputs)
    logits_per_image = outputs[0]  # Shape: [1, num_text_prompts]

    # Apply softmax to get probabilities
    probs = F.softmax(torch.tensor(logits_per_image[0]), dim=0)

    # Display results
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    for i, (prompt, prob) in enumerate(zip(text_prompts, probs)):
        print(f"{prompt:30s} : {prob.item()*100:5.2f}%")

    # Find most likely category
    max_idx = torch.argmax(probs).item()
    print("="*60)
    print(f"Most likely: {text_prompts[max_idx]} ({probs[max_idx].item()*100:.2f}%)")
    print("="*60 + "\n")

    # Validation against PyTorch reference
    print("Validating against PyTorch reference ...")
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16', use_safetensors=True).eval()
    with torch.no_grad():
        pt_outputs = model(**inputs)
        pt_logits = pt_outputs.logits_per_image[0]
        pt_probs = F.softmax(pt_logits, dim=0)

    # Calculate similarity metrics
    similarity = F.cosine_similarity(probs.unsqueeze(0), pt_probs.unsqueeze(0)).item()
    max_diff = torch.max(torch.abs(probs - pt_probs)).item()

    print(f"Probability distribution similarity: {similarity:.6f}")
    print(f"Maximum element-wise difference: {max_diff:.6f}")

    if similarity > 0.95:
        print("Validation: PASSED (similarity > 0.95)")
    else:
        print(f"Warning: Low similarity ({similarity:.6f}), expected > 0.95")

if __name__ == "__main__":
    main()
