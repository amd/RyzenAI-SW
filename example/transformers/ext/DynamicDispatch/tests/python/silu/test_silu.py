
import sys
import numpy as np

from ryzenai_dynamic_dispatch import load_meta_json
from ryzenai_dynamic_dispatch import FusionRuntime

def silu(x):
    return x / (1 + np.exp(-x))

def float_to_bfloat(x):
    # View the array as int16
    int16_view = x.view(np.int16)

    # Discard every other int16 element to truncate to bfloat16
    # Since each float32 is composed of two int16 elements, we need to take every other int16 element
    return np.ascontiguousarray(int16_view[..., 1::2])

def bfloat_to_float(x):
    float32_reconstructed_int = np.zeros(x.shape, dtype=np.int32)
    float32_reconstructed_int[:] = x.astype(np.int32) << 16
    return float32_reconstructed_int.view(np.float32)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python test_silu.py <path_to_meta_json>")
        sys.exit(1)

    np.random.seed(42)

    # Generate a random float32 array
    float32_input = np.random.rand(1, 1, 11008).astype(np.float32)

    bfloat16_input = float_to_bfloat(float32_input)
    golden_input = bfloat_to_float(bfloat16_input)

    inputs = [bfloat16_input]
    outputs = [np.random.randint(low=-42, high=42, size=(1, 1, 11008)).astype(np.int16)]

    #"test_silu_abf16/model_silu_meta.json"
    meta_json = sys.argv[1]
    xclbin = "xclbin/stx/llama2_mladf_2x4x4_gemmbfp16_silu_mul_mha_rms_rope.xclbin"

    meta_data = load_meta_json(meta_json)
    rt = FusionRuntime(xclbin)
    rt.init(meta_data)
    rt.execute(inputs, outputs)

    device_output = bfloat_to_float(outputs[0])
    golden_output = silu(golden_input)

    print("golden output:", golden_output)
    print("device output:", device_output)

    # Desired relative tolerance and absolute tolerance
    rtol = 0.1    # 10%
    atol = 0.08   # Optional, but can be used for very small numbers

    # Check if all elements are close within the specified tolerance
    within_threshold = np.allclose(device_output, golden_output, rtol=rtol, atol=atol)

    print("All elements within threshold:", within_threshold)

    if not within_threshold:
        sys.exit(1)

    sys.exit(0)
