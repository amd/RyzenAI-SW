"""Verify NPU is available and print device info."""
import onnxruntime as ort

providers = ort.get_available_providers()
print("Available ONNX Runtime providers:")
for p in providers:
    print(f"  - {p}")

npu_available = "VitisAIExecutionProvider" in providers
print(f"\nNPU (VitisAI EP) available: {npu_available}")
