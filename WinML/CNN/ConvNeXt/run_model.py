# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from PIL import Image
import numpy as np
import onnxruntime as ort
import subprocess
import sys
import json
import argparse
import os

def register_execution_providers():
    worker_script = os.path.abspath('winml_worker.py')
    proc = subprocess.run(
        [sys.executable, worker_script],
        capture_output=True,
        text=True
    )

    if proc.returncode != 0 or not proc.stdout.strip():
        print(f"[ERROR] winml_worker.py failed (exit code {proc.returncode})")
        if proc.stderr.strip():
            print(f"        stderr: {proc.stderr.strip()}")
        if proc.stdout.strip():
            print(f"        stdout: {proc.stdout.strip()}")
        print("\n[HINT] Possible causes:")
        print("  1. Windows App SDK version mismatch — check 'conda list | findstr wasdk'")
        print("     and install the matching Windows App SDK runtime.")
        print("  2. VitisAI EP not installed — reinstall with:")
        print("     pip install --pre --upgrade -r requirements.txt")
        raise RuntimeError("Failed to retrieve execution provider paths from winml_worker.py")

    # winml_worker may print diagnostic lines before the JSON; find the JSON line.
    json_line = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith('{'):
            json_line = line
            break

    if not json_line:
        raise RuntimeError(
            f"winml_worker.py produced no JSON output.\nstdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
        )

    paths = json.loads(json_line)
    for name, lib_path in paths.items():
        if not lib_path or not os.path.exists(lib_path):
            print(f"Skipping execution provider {name}: invalid or missing library path")
            continue
        ort.register_execution_provider_library(name, lib_path)
        print(f"Registered execution provider: {name} with library path: {lib_path}")


def vitisAI_provider_options(session_options):
    # Enumerate and filter EP devices
    ep_devices = ort.get_ep_devices()
    selected_ep_devices = [
        d for d in ep_devices 
        if d.ep_name == "VitisAIExecutionProvider"
        and d.device.type == ort.OrtHardwareDeviceType.NPU]
    
    if not selected_ep_devices:
        raise RuntimeError("VitisAIExecutionProvider is not available on this system.")
    
    # Configure provider-specific options in "vaiml_config.json" file
    provider_options = {'config_file': 'vaiml_config.json'}
    session_options.add_provider_for_devices([selected_ep_devices[0]], provider_options)
    

def load_and_preprocess_image(image_path):

    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)
    img_array = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std  # HWC
    img_array = img_array.transpose((2, 0, 1))  # CHW
    img_array = np.expand_dims(img_array, axis=0)  # NCHW
    return img_array.astype(np.float32)

def load_labels(label_file):
    labels = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2:
                labels.append(parts[1])
            else:
                labels.append(parts[0])
    return labels

def print_results(labels, results):

    scores = np.asarray(results, dtype=np.float32)
    if scores.ndim > 1:
        scores = scores.reshape(-1)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)

    top_k = min(5, probs.size)
    top_indices = np.argsort(probs)[-top_k:][::-1]
    print("Top-5 (softmax probabilities):")
    for i, idx in enumerate(top_indices, start=1):
        label = labels[idx] if isinstance(labels, (list, tuple)) and idx < len(labels) else f"class {idx}"
        print(f"  Top-{i}: {label} (id={idx}, p={probs[idx]:.6f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Windows ML Python Inference Script")
    parser.add_argument('--ep_policy', type=str.upper, default='NPU', choices=['NPU', 'CPU', 'GPU', 'DEFAULT'],
                        help='Set execution provider policy (NPU, CPU, GPU, DEFAULT). Default: NPU')
    parser.add_argument('--model', type=str, default=None, help='Path to the input ONNX model (default: convnext_small.onnx in model directory)')
    parser.add_argument('--compiled_output', type=str, default=None, help='Path for compiled output model (default: convnext_small_ctx.onnx in model directory)')
    parser.add_argument('--image_path', type=str, default=None, help='Path to the input image (default: all images in images folder)')
    args = parser.parse_args()

    print("Registering execution providers ...")
    register_execution_providers()

    print("Creating session ...")

    resource_path = Path(__file__).parent
    # Set model paths based on args
    model_path = Path(args.model) if args.model else (resource_path / "model" / "convnext_small.onnx")
    compiled_model_path = Path(args.compiled_output) if args.compiled_output else (resource_path / "model" / "convnext_small_ctx.onnx")
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 2  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal

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

    # VitisAI specific Advanced Provider option (optional)
    # vitisAI_provider_options(session_options)

    # Compile model if compiled model does not exist
    if not compiled_model_path.exists():
        print(f"Compiling model to {compiled_model_path} ...")
        model_compiler = ort.ModelCompiler(session_options, model_path)
        try:
            model_compiler.compile_to_file(compiled_model_path)
            print("Model compiled successfully")
        except Exception as e:
            print("Model compilation failed:", e)
            print("Falling back to uncompiled model")

    model_path_to_use = compiled_model_path if compiled_model_path.exists() else model_path

    # Create session with fallback if compiled model requires an EP that isn't active
    def create_session(path):
        return ort.InferenceSession(path, sess_options=session_options)

    try:
        session = create_session(model_path_to_use)
    except Exception as e:
        if model_path_to_use == compiled_model_path:
            print(f"Failed to load compiled model due to: {e}. Falling back to original model: {model_path}")
            session = create_session(model_path)
            model_path_to_use = model_path
        else:
            raise

    # Report active execution providers
    try:
        eps = session.get_providers()
        print(f"Active execution providers (priority order): {eps}")
        if eps:
            print(f"Primary provider (highest priority): {eps[0]}")
    except Exception as e:
        print(f"Warning: unable to query active execution providers: {e}")

    # Load labels
    labels_path = resource_path / "model" / "ConvNeXt.Labels.txt"
    if not labels_path.exists():
        print(f"Warning: Labels file not found at {labels_path}. Proceeding without labels.")
        labels = []
    else:
        labels = load_labels(labels_path)

    # Use image_path argument if provided, else use all images in folder
    if args.image_path:
        image_files = [Path(args.image_path).resolve()]
    else:
        images_folder = resource_path / "images"
        if not images_folder.exists():
            print(f"Warning: images folder not found at {images_folder}.")
            image_files = []
        else:
            exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = [p for p in images_folder.iterdir() if p.is_file() and p.suffix.lower() in exts]

    if not image_files:
        print("No images to process. Provide --image_path or place images in the images folder.")
    for image_file in image_files:
        print(f"Running inference on image: {image_file}")
        print("Preparing input ...")
        img_array = load_and_preprocess_image(image_file)
        print("Running inference ...")
        input_name = session.get_inputs()[0].name
    results = session.run(None, {input_name: img_array})[0]
    print_results(labels, results)
