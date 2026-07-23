#!/usr/bin/env python3
"""
Download and Export RetinaFace MobileNet Model to ONNX

This script downloads the RetinaFace PyTorch model with MobileNet backbone from
yakhyo/retinaface-pytorch and exports it to ONNX FP32 format for NPU deployment.

Steps performed:
1. Clone yakhyo/retinaface-pytorch repository (if needed)
2. Download pretrained MobileNetV1 0.25 weights from GitHub releases
3. Export PyTorch model to ONNX format
4. Convert model from NHWC to NCHW format for NPU compatibility
5. Save final model as retinaface_mobilenet_static.onnx

The FP32 model will be automatically converted to BF16 during NPU compilation
for optimal performance.

Repository: https://github.com/yakhyo/retinaface-pytorch
"""

import sys
import os
import subprocess
from pathlib import Path
import shutil

def download_model():
    """Download RetinaFace PyTorch model and export to ONNX format"""

    model_dir = Path(__file__).parent
    final_model_path = model_dir / "retinaface_mobilenet.onnx"

    # Check if final model already exists
    if final_model_path.exists():
        print(f"[OK] Model already exists: {final_model_path}")
        return 0

    print("=" * 70)
    print("RetinaFace MobileNet Model Download & Export")
    print("=" * 70)

    # Step 1: Clone repository
    repo_dir = model_dir / "retinaface-pytorch"
    if not repo_dir.exists():
        print("\n[1/4] Cloning yakhyo/retinaface-pytorch repository...")
        try:
            subprocess.run([
                "git", "clone",
                "https://github.com/yakhyo/retinaface-pytorch.git",
                str(repo_dir)
            ], check=True, capture_output=True)
            print("[OK] Repository cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to clone repository: {e}")
            print("\nManual steps:")
            print("1. Clone: git clone https://github.com/yakhyo/retinaface-pytorch.git")
            print(f"2. Place in: {model_dir}")
            return 1
    else:
        print(f"\n[1/4] Repository already exists: {repo_dir}")

    # Step 2: Download pretrained weights
    print("\n[2/4] Downloading pretrained MobileNetV1 0.25 weights...")
    weights_dir = repo_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    weights_path = weights_dir / "mobilenetv1_0.25.pth"

    if not weights_path.exists():
        try:
            import urllib.request
            # Download from GitHub releases
            weights_url = "https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/mobilenetv1_0.25.pth"
            print(f"   Downloading from: {weights_url}")
            print(f"   This may take a few minutes...")

            req = urllib.request.Request(weights_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(weights_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)

            if weights_path.exists() and weights_path.stat().st_size > 0:
                size_mb = weights_path.stat().st_size / (1024 * 1024)
                print(f"[OK] Weights downloaded: {size_mb:.1f} MB")
            else:
                print("[ERROR] Download failed - file is empty or missing")
                return 1

        except Exception as e:
            print(f"[ERROR] Failed to download weights: {e}")
            print("\nManual download:")
            print(f"1. Download from: https://github.com/yakhyo/retinaface-pytorch/releases")
            print(f"2. Save as: {weights_path}")
            return 1
    else:
        print(f"[OK] Weights already exist: {weights_path}")

    # Step 3: Install dependencies
    print("\n[3/4] Installing dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "torch", "torchvision", "onnx"
        ], check=True, capture_output=True)
        print("[OK] Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Failed to install dependencies: {e}")
        print("   Continuing anyway...")

    # Step 4: Export to ONNX
    print("\n[4/4] Exporting PyTorch model to ONNX format...")
    onnx_export_path = repo_dir / "retinaface_mobilenet.onnx"

    try:
        # Change to repo directory to run export script
        original_dir = os.getcwd()
        os.chdir(repo_dir)

        # Run ONNX export command
        export_cmd = [
            sys.executable, "-m", "scripts.onnx_export",
            "-w", str(weights_path),
            "-n", "mobilenetv1_025",
            "--height", "640",
            "--width", "640"
        ]

        print(f"   Running: {' '.join(export_cmd)}")
        result = subprocess.run(export_cmd, capture_output=True, text=True, check=True)
        print(result.stdout)

        os.chdir(original_dir)

        # Find the exported ONNX file
        exported_files = list(repo_dir.glob("*.onnx"))
        if exported_files:
            # Copy to final location
            shutil.copy(exported_files[0], final_model_path)
            size_mb = final_model_path.stat().st_size / (1024 * 1024)
            print(f"[OK] Model exported successfully: {size_mb:.1f} MB")
            print(f"   Location: {final_model_path}")

            # Convert to NCHW format with static batch
            print(f"\n[INFO] Converting model to NCHW format for NPU compatibility...")
            try:
                result = subprocess.run([sys.executable, "fix_dynamic_shapes.py"],
                                      capture_output=True, text=True, check=True,
                                      cwd=model_dir)
                print(f"[OK] Model converted successfully")
                print(f"   Final model: retinaface_mobilenet_static.onnx")
            except subprocess.CalledProcessError as e:
                print(f"[WARNING] Model conversion failed: {e}")
                print(f"   You can manually run: python fix_dynamic_shapes.py")
                print(f"   Original model is still usable: {final_model_path}")

            # Simplify model - remove Shape-Gather-Reshape patterns
            static_model_path = model_dir / "retinaface_mobilenet_static.onnx"
            if static_model_path.exists():
                print(f"\n[INFO] Simplifying model (removing dynamic shape operations)...")
                try:
                    result = subprocess.run([
                        sys.executable, "simplify_model.py",
                        "--model", "retinaface_mobilenet_static.onnx",
                        "--output", "retinaface_mobilenet_static.onnx"
                    ], capture_output=True, text=True, check=True, cwd=model_dir)
                    # Print last few lines (summary)
                    output_lines = result.stdout.strip().split('\n')
                    for line in output_lines[-10:]:
                        print(f"   {line}")
                except subprocess.CalledProcessError as e:
                    print(f"[WARNING] Model simplification failed: {e}")
                    print(f"   Continuing with unsimplified model...")

            # Convert LeakyReLU to ReLU for NPU optimization
            static_model_path = model_dir / "retinaface_mobilenet_static.onnx"
            if static_model_path.exists():
                print(f"\n[INFO] Converting LeakyReLU to ReLU for NPU optimization...")
                try:
                    result = subprocess.run([
                        sys.executable, "replace_leakyrelu.py",
                        "--model", "retinaface_mobilenet_static.onnx",
                        "--output", "retinaface_mobilenet_relu.onnx"
                    ], capture_output=True, text=True, check=True, cwd=model_dir)
                    print(result.stdout)
                    relu_model_path = model_dir / "retinaface_mobilenet_relu.onnx"
                    if relu_model_path.exists():
                        size_mb = relu_model_path.stat().st_size / (1024 * 1024)
                        print(f"[OK] ReLU model created: {size_mb:.1f} MB")
                        print(f"   Ready for quantization: retinaface_mobilenet_relu.onnx")
                except subprocess.CalledProcessError as e:
                    print(f"[WARNING] LeakyReLU conversion failed: {e}")
                    print(f"   You can manually run: python replace_leakyrelu.py")

            return 0
        else:
            print("[ERROR] ONNX export completed but no .onnx file found")
            return 1

    except subprocess.CalledProcessError as e:
        os.chdir(original_dir)
        print(f"[ERROR] ONNX export failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        print("\nManual export steps:")
        print(f"1. cd {repo_dir}")
        print(f"2. python -m scripts.onnx_export -w {weights_path} -n mobilenetv1_025 --height 640 --width 640")
        print(f"3. Copy output .onnx file to {final_model_path}")
        return 1
    except Exception as e:
        os.chdir(original_dir)
        print(f"[ERROR] Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(download_model())
