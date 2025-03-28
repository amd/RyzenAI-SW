import os
import argparse
import onnx
import time
import numpy as np
from PIL import Image
from pathlib import Path
import onnxruntime as ort
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantType
from quark.onnx import ModelQuantizer, PowerOfTwoMethod
from quark.onnx.quantization.config import Config, get_default_config
from utils import ImageDataReader, evaluate_onnx_model

def benchmark_model(session, runs=100):
    input_shape = session.get_inputs()[0].shape
    input_shape = tuple(1 if isinstance(dim, str) else dim for dim in input_shape)
    input_data = np.random.rand(*input_shape).astype(np.float32)
    start_time = time.time()
    for _ in range(runs):
        outputs = session.run(None, {session.get_inputs()[0].name: input_data})
    end_time = time.time()
    avg_time = (end_time - start_time) / runs
    print('Average inference time over {} runs: {} ms'.format(runs, avg_time * 1000))

def main(args):
    # Setup the Input model
    input_model_path = args.model_input
    output_model_path = args.model_output
    calibration_dataset_path = args.calib_data

    # Quantize the ONNX model if the flag is set
    # Get quantization configuration
    from quark.onnx import ModelQuantizer, VitisQuantFormat, VitisQuantType
    from quark.onnx.quantization.config import QuantizationConfig
    # Define the calibration data reader
    num_calib_data = 100
    calibration_dataset = ImageDataReader(calibration_dataset_path, input_model_path, data_size=num_calib_data, batch_size=1)

    if args.quantize == 'bf16':
        # Defines the quantization configuration for the whole model
        quant_config = get_default_config("BF16")
        quant_config.extra_options["BF16QDQToCast"] = True
        config = Config(global_quant_config=quant_config)
        print("The configuration of the quantization is {}".format(config))
    elif args.quantize == 'int8':
        # Defines the quantization configuration for the whole model
        quant_config = get_default_config("XINT8")
        config = Config(global_quant_config=quant_config)
        print("The configuration of the quantization is {}".format(config))
    else:
        # Use BF16 as the default quantization options
        print("Invalid quantization option. Please choose from 'BF16' or 'INT8.")

    # Quantize the ONNX model if the flag is set
    if args.quantize:
        # Create an ONNX Quantizer
        quantizer = ModelQuantizer(config)
        quant_model = quantizer.quantize_model(model_input=input_model_path,
                                               model_output=output_model_path,
                                               calibration_data_reader=calibration_dataset)
        print("Model Size:")
        print("Float32 model size: {:.2f} MB".format(os.path.getsize(input_model_path)/(1024 * 1024)))
        print("Quantized {} quantized model size: {:.2f} MB".format(args.quantize, os.path.getsize(output_model_path)/(1024 * 1024)))

    # Benchmark the float/quantized models on CPU/NPU
    if args.benchmark:
        # Run the float model on CPU
        model = onnx.load(input_model_path)
        provider = ['CPUExecutionProvider']
        cache_dir = Path(__file__).parent.resolve()
        provider_options = [{
                'config_file': 'vaiml_config.json',
                'cacheDir': str(cache_dir),
                'cacheKey': 'modelcachekey'
            }]
        session = ort.InferenceSession(model.SerializeToString(), providers=provider,
                                       provider_options=provider_options)
        print('Benchmarking CPU float model:')
        benchmark_model(session)

        # Run the BF16 model on CPU and Register BF16 operations
        import onnxruntime
        from quark.onnx import get_library_path
        sess_options = onnxruntime.SessionOptions()
        # sess_options.register_custom_ops_library(get_library_path(device='cpu'))
        session = onnxruntime.InferenceSession(output_model_path, sess_options, providers=provider)
        # session = onnxruntime.InferenceSession(bf16_model.SerializeToString(), sess_options, providers=provider)
        print('Benchmarking CPU BF16 model:')
        benchmark_model(session)

        # Run quantized model on NPU
        quant_model = onnx.load(output_model_path)
        provider = ['VitisAIExecutionProvider']
        cache_dir = Path(__file__).parent.resolve()
        provider_options = [{
                    'config_file': 'vaiml_config.json',
                    'cacheDir': str(cache_dir),
                    'cacheKey': 'modelcachekey'
                }]
        session = ort.InferenceSession(quant_model.SerializeToString(), providers=provider,
                                       provider_options=provider_options)
        print('Benchmarking NPU BF16 model:')
        benchmark_model(session)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize and evaluate ONNX models.")
    parser.add_argument('--model_input', type=str, default='models/resnet50.onnx', help='Path to the input ONNX model.')
    parser.add_argument('--model_output', type=str, default='models/resnet50_quant.onnx', help='Path to save the quantized ONNX model.')
    parser.add_argument('--calib_data', type=str, default='calib_data', help='Path to the calibration dataset.')
    parser.add_argument('--quantize', type=str, choices=['bf16', 'int8'], required=False, help='options to quantize the model.')
    parser.add_argument('--benchmark', action='store_true', help='Flag to benchmark the model.')

    args = parser.parse_args()
    main(args)