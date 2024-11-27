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
from quark.onnx import ModelQuantizer  
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
    output_model_path = args.model_output  
    calibration_dataset_path = args.calib_data  
  
    # Get quantization configuration  
    quant_config = get_default_config("XINT8")  
  
    # Defines the quantization configuration for the whole model  
    config = Config(global_quant_config=quant_config)  
    print("The configuration of the quantization is {}".format(config))  
  
    # Define the calibration data reader  
    num_calib_data = 100  
    calibration_dataset = ImageDataReader(calibration_dataset_path, input_model_path, data_size=num_calib_data, batch_size=32)  
  
    # Create an ONNX Quantizer  
    quantizer = ModelQuantizer(config)  
  
    # Quantize the ONNX model if the flag is set  
    if args.quantize:  
        quant_model = quantizer.quantize_model(model_input=input_model_path,   
                                               model_output=output_model_path,   
                                               calibration_data_reader=calibration_dataset)  
        print("Model Size:")  
        print("Float32 model size: {:.2f} MB".format(os.path.getsize(input_model_path)/(1024 * 1024)))  
        print("Int8 quantized model size: {:.2f} MB".format(os.path.getsize(output_model_path)/(1024 * 1024)))  
  
    # Evaluate the model if the flag is set  
    if args.evaluate:  
        print("Model Accuracy:")  
        top1_acc, top5_acc = evaluate_onnx_model(input_model_path, imagenet_data_path=calibration_dataset_path)  
        print("Float32 model accuracy: Top1 {:.3f}, Top5 {:.3f} ".format(top1_acc, top5_acc))  
        top1_acc, top5_acc = evaluate_onnx_model(output_model_path, imagenet_data_path=calibration_dataset_path)  
        print("Int8 quantized model accuracy: Top1 {:.3f}, Top5 {:.3f} ".format(top1_acc, top5_acc))  
        top1_acc, top5_acc = evaluate_onnx_model(output_model_path, imagenet_data_path=calibration_dataset_path, device='npu')  
        print("Int8 quantized model accuracy (NPU): Top1 {:.3f}, Top5 {:.3f} ".format(top1_acc, top5_acc))  
  
    # Benchmark the float/quantized models on CPU/NPU
    if args.benchmark:
        # Run the float model on CPU
        model = onnx.load(input_model_path)  
        provider = ['CPUExecutionProvider']  
        session = ort.InferenceSession(model.SerializeToString(), providers=provider)
        print('Benchmarking CPU float model:')  
        benchmark_model(session)  
  
        # Run the quantized model on CPU  
        quant_model = onnx.load(output_model_path)  
        session = ort.InferenceSession(quant_model.SerializeToString(), providers=provider)  
        print('Benchmarking CPU quantized model:')  
        benchmark_model(session)  
  
        # Run quantized model on NPU  
        quant_model = onnx.load(output_model_path)  
        provider = ['VitisAIExecutionProvider']  
        cache_dir = Path(__file__).parent.resolve()  
        provider_options = [{  
                    'config_file': 'vaip_config.json',  
                    'cacheDir': str(cache_dir),  
                    'cacheKey': 'modelcachekey'  
                }]    
        session = ort.InferenceSession(quant_model.SerializeToString(), providers=provider,  
                                       provider_options=provider_options)  
        print('Benchmarking NPU quantized model:')  
        benchmark_model(session)   
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Quantize and evaluate ONNX models.")  
    parser.add_argument('--model_input', type=str, default='models/resnet50.onnx', help='Path to the input ONNX model.')  
    parser.add_argument('--model_output', type=str, default='models/resnet50_quant.onnx', help='Path to save the quantized ONNX model.')  
    parser.add_argument('--calib_data', type=str, default='calib_data', help='Path to the calibration dataset.')  
    parser.add_argument('--quantize', action='store_true', help='Flag to quantize the model.')  
    parser.add_argument('--evaluate', action='store_true', help='Flag to evaluate the model.') 
    parser.add_argument('--benchmark', action='store_true', help='Flag to benchmark the model.') 

    args = parser.parse_args()  
    main(args)  
