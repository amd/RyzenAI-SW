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

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32)/255
    image_array = np.transpose(image_array, (2, 0, 1))
    input_data = np.expand_dims(image_array, axis=0)
    return input_data

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
    calibration_dataset_path = args.calib_data

    # Benchmark the float/quantized models on CPU/NPU
    model = onnx.load(input_model_path)
    provider = ['CPUExecutionProvider']
    cache_dir = Path(__file__).parent.resolve()
    provider_options = [{
                    'config_file': 'vaiml_config.json',
                    'cacheDir': str(cache_dir),
                    'cacheKey': 'modelcachekey'
                }]
    if args.device == 'cpu':
        # Run the float model on CPU
        session = ort.InferenceSession(model.SerializeToString(), providers=provider)
        print('Benchmarking model on CPU:')
        benchmark_model(session)
    
    elif args.device == 'npu':
        # Run quantized model on NPU
        quant_model = onnx.load(input_model_path)
        provider = ['VitisAIExecutionProvider']
        session = ort.InferenceSession(model.SerializeToString(), providers=provider,
                                       provider_options=provider_options)
        print('Benchmarking model on NPU:')
        benchmark_model(session)

    # Evaluate the model if the flag is set
    if args.evaluate:
        print("Model Accuracy:")
        top1_acc, top5_acc = evaluate_onnx_model(input_model_path, imagenet_data_path=calibration_dataset_path, device=args.device)
        print("{} model accuracy on {}: Top1 {:.3f}, Top5 {:.3f} ".format(args.model_input, args.device, top1_acc, top5_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize and evaluate ONNX models.")
    parser.add_argument('--model_input', type=str, default='models/resnet50_bf16.onnx', help='Path to the input ONNX model.')
    parser.add_argument('--calib_data', type=str, default='calib_data', help='Path to the calibration dataset.')
    parser.add_argument('--device', type=str, choices=['cpu', 'npu'], required=False, help='device to run the model.')
    parser.add_argument('--evaluate', action='store_true', help='Flag to evaluate the model.')

    args = parser.parse_args()
    main(args)
