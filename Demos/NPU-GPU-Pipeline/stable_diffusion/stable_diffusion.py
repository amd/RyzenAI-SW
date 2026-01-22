# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict

import config
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from packaging import version
from user_script import get_base_model_name

from olive.common.utils import set_tempdir
from olive.workflows import run as olive_run

file_path = str(Path(__file__).resolve().parent)
sys.path.append(file_path)
from  PIL import Image

def update_config_with_provider(config: Dict, provider: str):
    print("Printing config:",config)
    if provider == "dml":
        # DirectML EP is the default, so no need to update config.
        return config
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def optimize(
    model_id: str,
    provider: str,
    unoptimized_model_dir: Path,
    optimized_model_dir: Path,
):
    from google.protobuf import __version__ as protobuf_version

    # protobuf 4.x aborts with OOM when optimizing unet
    if version.parse(protobuf_version) > version.parse("3.20.3"):
        print("This script requires protobuf 3.20.3. Please ensure your package version matches requirements.txt.")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent

    # Clean up previously optimized models, if any.
    shutil.rmtree(script_dir / "footprints", ignore_errors=True)
    shutil.rmtree(unoptimized_model_dir, ignore_errors=True)
    shutil.rmtree(optimized_model_dir, ignore_errors=True)

    # The model_id and base_model_id are identical when optimizing a standard stable diffusion model like
    # runwayml/stable-diffusion-v1-5. These variables are only different when optimizing a LoRA variant.
    base_model_id = get_base_model_name(model_id)

    # Load the entire PyTorch pipeline to ensure all models and their configurations are downloaded and cached.
    # This avoids an issue where the non-ONNX components (tokenizer, scheduler, and feature extractor) are not
    # automatically cached correctly if individual models are fetched one at a time.
    print("Download stable diffusion PyTorch pipeline...")
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(base_model_id, torch_dtype=torch.float32)
    config.vae_sample_size = pipeline.vae.config.sample_size
    config.cross_attention_dim = pipeline.unet.config.cross_attention_dim
    config.unet_sample_size = pipeline.unet.config.sample_size

    model_info = {}

    submodel_names = ["vae_encoder", "vae_decoder", "unet", "text_encoder"]

    # has_safety_checker = getattr(pipeline, "safety_checker", None) is not None
    has_safety_checker = False

    if has_safety_checker:
        submodel_names.append("safety_checker")

    for submodel_name in submodel_names:
        print(f"\nOptimizing {submodel_name}")

        olive_config = None
        with (script_dir / f"config_{submodel_name}.json").open() as fin:
            olive_config = json.load(fin)
        olive_config = update_config_with_provider(olive_config, provider)

        if submodel_name in ("unet", "text_encoder"):
            olive_config["input_model"]["config"]["model_path"] = model_id
        else:
            # Only the unet & text encoder are affected by LoRA, so it's better to use the base model ID for
            # other models: the Olive cache is based on the JSON config, and two LoRA variants with the same
            # base model ID should be able to reuse previously optimized copies.
            olive_config["input_model"]["config"]["model_path"] = base_model_id

        run_res = olive_run(olive_config)
        from ort_util_img2img import save_optimized_onnx_submodel
        save_optimized_onnx_submodel(submodel_name, provider, model_info)
    from ort_util_img2img import save_onnx_pipeline
    save_onnx_pipeline(
        has_safety_checker, model_info, optimized_model_dir, unoptimized_model_dir, pipeline, submodel_names
    )

    return model_info


def parse_common_args(raw_args):
    parser = argparse.ArgumentParser("Common arguments")

    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5", type=str)
    parser.add_argument(
        "--provider", default="dml", type=str, choices=["dml"], help="Execution provider to use"
    )
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--clean_cache", action="store_true", help="Deletes the Olive cache")
    parser.add_argument("--test_unoptimized", action="store_true", help="Use unoptimized model for inference")
    parser.add_argument("--batch_size", default=1, type=int, help="Number of images to generate per batch")
    parser.add_argument(
        "--prompt",
        default=(
            "castle surrounded by water and nature, village, volumetric lighting, photorealistic, "
            "detailed and intricate, fantasy, epic cinematic shot, mountains, 8k ultra hd"
        ),
        type=str,
    )
    parser.add_argument(
        "--guidance_scale",
        default=7.5,
        type=float,
        help="Guidance scale as defined in Classifier-Free Diffusion Guidance",
    )
    parser.add_argument("--num_images", default=1, type=int, help="Number of images to generate")
    parser.add_argument("--num_inference_steps", default=50, type=int, help="Number of steps in diffusion process")
    parser.add_argument("--interactive", action="store_true", help="Run with a GUI")
    parser.add_argument("--tempdir", default=None, type=str, help="Root directory for tempfile directories and files")
    parser.add_argument("--image_path", default=None, type=str, help="Root directory for tempfile directories and files")
    parser.add_argument(
        "--strength",
        default=1.0,
        type=float,
        help="Value between 0.0 and 1.0, that controls the amount of noise that is added to the input image. "
        "Values that approach 1.0 enable lots of variations but will also produce images "
        "that are not semantically consistent with the input.",
    )
    parser.add_argument("--image_size", default=512, type=int, help="Width and height of the images to generate")

    return parser.parse_known_args(raw_args)


def parse_ort_args(raw_args):
    parser = argparse.ArgumentParser("ONNX Runtime arguments")

    parser.add_argument(
        "--static_dims",
        action="store_true",
        help="DEPRECATED (now enabled by default). Use --dynamic_dims to disable static_dims.",
    )
    parser.add_argument("--dynamic_dims", action="store_true", help="Disable static shape optimization")

    return parser.parse_known_args(raw_args)


def main(raw_args=None):
    common_args, extra_args = parse_common_args(raw_args)

    provider = common_args.provider
    model_id = common_args.model_id

    script_dir = Path(__file__).resolve().parent
    unoptimized_model_dir = script_dir / "models" / "unoptimized" / model_id
    optimized_dir_name = f"optimized-{provider}"
    optimized_model_dir = script_dir / "models" / optimized_dir_name / model_id

    if common_args.clean_cache:
        shutil.rmtree(script_dir / "cache", ignore_errors=True)

    guidance_scale = common_args.guidance_scale

    if model_id == "stabilityai/sd-turbo" and guidance_scale > 0:
        guidance_scale = 0.0
        print(f"WARNING: Classifier free guidance has been forcefully disabled since {model_id} doesn't support it.")

    ort_args, extra_args = parse_ort_args(extra_args)

    if common_args.optimize or not optimized_model_dir.exists():
        set_tempdir(common_args.tempdir)

        # TODO(jstoecker): clean up warning filter (mostly during conversion from torch to ONNX)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ort_util_img2img import validate_args
            validate_args(ort_args, common_args.provider)
            optimize(common_args.model_id, common_args.provider, unoptimized_model_dir, optimized_model_dir)

if __name__ == "__main__":
    main()
