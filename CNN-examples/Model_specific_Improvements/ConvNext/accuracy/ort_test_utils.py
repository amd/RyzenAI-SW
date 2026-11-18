"""
Shared utilities for ONNX accuracy tests.

This module contains the common ORT / WinML infrastructure used by all
accuracy test scripts:

  - WinML EP registration
  - EP short-name ↔ full-name mapping
  - Device-type resolution
  - ``InferenceSession`` creation via ``ort.get_ep_devices()``
  - Input layout detection (NCHW / NHWC)
  - Input spatial-size extraction
  - Common CLI argument helpers
  - Model-path resolution
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path


# ---------------------------------------------------------------------------
# WinML provider registration
# ---------------------------------------------------------------------------

def register_providers(verbose: bool = True) -> list[str]:
    """Use WinML to register execution providers with ONNX Runtime.

    Calls ``WinML().register_execution_providers_to_ort()`` which probes
    the hardware and registers all supported EPs.  Returns the list of
    registered EP full names, or an empty list if WinML is unavailable.

    See https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/get-started?tabs=python
    """
    try:
        from winml_helper import WinML
        registered_eps = WinML().register_execution_providers_to_ort()
        eps = list(registered_eps) if registered_eps else []
    except (ImportError, AttributeError, Exception):
        eps = []

    if verbose:
        print("Registered EPs :", eps if eps else "(WinML unavailable)")
    return eps


# ---------------------------------------------------------------------------
# EP short-name ↔ full-name mapping
# ---------------------------------------------------------------------------
EP_SHORT_MAP: OrderedDict[str, str] = OrderedDict(
    [
        ("CPU",      "CPUExecutionProvider"),
        ("OV",       "OpenVINOExecutionProvider"),
        ("VITIS",    "VitisAIExecutionProvider"),
        ("QNN",      "QNNExecutionProvider"),
        ("MIGRAPHX", "MIGraphXExecutionProvider"),
        ("RTX",      "NvTensorRTRTXExecutionProvider"),
        ("DML",      "DmlExecutionProvider"),
    ]
)

EP_FULL_TO_SHORT = {v: k for k, v in EP_SHORT_MAP.items()}

DEVICE_TYPES = ["CPU", "GPU", "NPU"]


def resolve_ep_name(name: str) -> str:
    """Accept a short name (CPU, OV, DML …) or a full ORT provider name."""
    upper = name.upper()
    if upper in EP_SHORT_MAP:
        return EP_SHORT_MAP[upper]
    if name.endswith("ExecutionProvider"):
        return name
    raise ValueError(
        f"Unknown EP '{name}'. Use one of: {', '.join(EP_SHORT_MAP.keys())} "
        f"or a full ORT provider name."
    )


def get_device_type(device: str):
    """Map a device string to ``ort.OrtHardwareDeviceType``."""
    import onnxruntime as ort
    mapping = {
        "CPU": ort.OrtHardwareDeviceType.CPU,
        "GPU": ort.OrtHardwareDeviceType.GPU,
        "NPU": ort.OrtHardwareDeviceType.NPU,
    }
    key = device.upper()
    if key not in mapping:
        raise ValueError(f"Unknown device type: '{device}'. Use one of: {list(mapping.keys())}")
    return mapping[key]


# ---------------------------------------------------------------------------
# Inference session
# ---------------------------------------------------------------------------

def create_session(
    model_path: str,
    ep_full: str,
    device: str,
):
    """Create an ONNX Runtime inference session.

    Uses ``ort.get_ep_devices()`` to find the EP+device pair that matches
    the requested execution provider and device type, then binds it to
    the session via ``session_options.add_provider_for_devices()``.
    """
    import onnxruntime as ort

    device_type = get_device_type(device)

    session_options = ort.SessionOptions()

    ep_devices = ort.get_ep_devices()
    matched = False
    ep_options = {}
    if ep_full == "VitisAIExecutionProvider":
        # The VitisAI EP's default compile-cache key ignores weights/scales, so two
        # quantizations of the same graph would silently share one compiled NPU binary.
        import hashlib
        digest = hashlib.md5(Path(model_path).read_bytes()).hexdigest()[:16]
        ep_options["cache_key"] = f"{Path(model_path).stem}_{digest}"
    for ep_device in ep_devices:
        if ep_device.ep_name == ep_full and ep_device.device.type == device_type:
            print(f"  Using {ep_device.ep_name} on {device}")
            if ep_options:
                print(f"  EP options: {ep_options}")
            session_options.add_provider_for_devices([ep_device], ep_options)
            matched = True
            break

    if not matched:
        available = [(d.ep_name, d.device.type) for d in ep_devices]
        raise RuntimeError(
            f"No EP+device match for {ep_full} on {device}.\n"
            f"Available ep_devices: {available}"
        )

    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
    )

    print(f"  Active providers: {session.get_providers()}")
    return session


# ---------------------------------------------------------------------------
# Input-shape helpers
# ---------------------------------------------------------------------------

def detect_layout(session) -> str:
    """Detect input layout (NCHW or NHWC) from the model's first input shape."""
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) == 4:
        if input_shape[1] == 3:
            return "NCHW"
        if input_shape[3] == 3:
            return "NHWC"
    print(f"  WARNING: Could not detect layout from input shape {input_shape}, assuming NCHW")
    return "NCHW"


def get_input_hw(session) -> tuple[int, int]:
    """Return (height, width) of the model input."""
    shape = session.get_inputs()[0].shape  # e.g. [1,3,H,W] or [1,H,W,3]
    if len(shape) == 4:
        if shape[1] == 3:
            return int(shape[2]), int(shape[3])
        if shape[3] == 3:
            return int(shape[1]), int(shape[2])
    raise RuntimeError(f"Cannot determine H,W from input shape {shape}")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_num_samples: int = 1000,
    num_samples_help: str = "Number of validation images to use",
) -> None:
    """Add the standard ``-m -n -e -d -s`` arguments to *parser*."""

    parser.add_argument(
        "-m", "--model-path",
        nargs="+",
        type=str,
        required=True,
        help="One or more paths to model files to evaluate (typically ONNX).",
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=default_num_samples,
        help=f"{num_samples_help} (default: {default_num_samples})",
    )
    ep_choices = list(EP_SHORT_MAP.keys())
    parser.add_argument(
        "-e", "--ep",
        type=str,
        default="CPU",
        help=(
            "Execution provider short name. "
            f"Choices: {', '.join(ep_choices)}. "
            "Full ORT names also accepted. (default: CPU)"
        ),
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        choices=DEVICE_TYPES,
        default="CPU",
        help="Target device type: CPU, GPU, or NPU. (default: CPU)",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for dataset shuffling (default: 42)",
    )


def resolve_models(args: argparse.Namespace) -> list[tuple[str, Path]]:
    """Build a list of ``(model_name, model_path)`` from parsed CLI args.

    Exits with code 1 if no valid model files are found.
    """
    model_items: list[tuple[str, Path]] = []
    for p in args.model_path:
        mp = Path(p)
        if not mp.exists():
            print(f"WARNING: model file not found, skipping: {mp}")
            continue
        model_items.append((mp.stem, mp))

    if not model_items:
        print("No valid model files found.")
        sys.exit(1)

    return model_items
