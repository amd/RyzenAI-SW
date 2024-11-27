import os  
import argparse  
import onnxruntime  
from quark.onnx import ModelQuantizer  
from quark.onnx.quantization.config import Config, get_default_config  
from quark.onnx.quantization.config.config import QuantizationConfig  
from onnxruntime.quantization.calibrate import CalibrationMethod  
from onnxruntime.quantization.quant_utils import QuantType, QuantFormat  
from quark.onnx import ModelQuantizer, PowerOfTwoMethod, QuantType  
from quark.onnx.quant_utils import PowerOfTwoMethod, VitisQuantType, VitisQuantFormat  
from utils import top1_accu, ImageDataReader, evaluate_onnx_model  
  
def main(args):  
    # Setup the Input model  
    input_model_path = args.model_input  
    output_model_path = args.model_output  
    calibration_dataset_path = args.calib_data  
  
    # Select quantization configuration based on arguments 
    if args.fast_finetune:  
        quant_config = QuantizationConfig(calibrate_method=PowerOfTwoMethod.MinMSE,
                                          activation_type=QuantType.QUInt8,
                                          weight_type=QuantType.QInt8,
                                          enable_npu_cnn=True,
                                          include_fast_ft=True,
                                          extra_options={'ActivationSymmetric': True})
    elif args.cross_layer_equalization:
        quant_config = QuantizationConfig(calibrate_method=PowerOfTwoMethod.MinMSE,
                                          activation_type=QuantType.QUInt8,
                                          weight_type=QuantType.QInt8,
                                          enable_npu_cnn=True,
                                          include_cle=True,
                                          extra_options={
                                              'ActivationSymmetric': True})
    else:  
        quant_config = get_default_config("XINT8")  
  
    # Defines the quantization configuration for the whole model  
    config = Config(global_quant_config=quant_config)  
    print("The configuration of the quantization is {}".format(config))  
  
    # Define the calibration data reader  
    num_calib_data = 1000
    calibration_dataset = ImageDataReader(calibration_dataset_path, input_model_path, data_size=num_calib_data, batch_size=1)
  
    # Create an ONNX Quantizer  
    quantizer = ModelQuantizer(config)  
  
    # Quantize the ONNX model  
    quant_model = quantizer.quantize_model(model_input=input_model_path,   
                                           model_output=output_model_path,   
                                           calibration_data_reader=calibration_dataset)  
  
    print("Model Size:")  
    print("Float32 model size: {:.2f} MB".format(os.path.getsize(input_model_path)/(1024 * 1024)))  
    print("Int8 quantized model size: {:.2f} MB".format(os.path.getsize(output_model_path)/(1024 * 1024)))  
  
    # Evaluate the model  
    print("Model Accuracy:")  
    top1_acc, top5_acc = evaluate_onnx_model(input_model_path, imagenet_data_path=calibration_dataset_path)  
    print("Float32 model accuracy: Top1 {:.3f}, Top5 {:.3f} ".format(top1_acc, top5_acc))  
    top1_acc, top5_acc = evaluate_onnx_model(output_model_path, imagenet_data_path=calibration_dataset_path)  
    print("Int8 quantized model accuracy: Top1 {:.3f}, Top5 {:.3f} ".format(top1_acc, top5_acc))  
    top1_acc, top5_acc = evaluate_onnx_model(output_model_path, imagenet_data_path=calibration_dataset_path, device='npu')  
    print("Int8 quantized model accuracy (NPU): Top1 {:.3f}, Top5 {:.3f} ".format(top1_acc, top5_acc))  

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Quantize and evaluate ONNX models.")    
    parser.add_argument('--model_input', type=str, default='models/mobilenetv2.onnx', help='Path to the input ONNX model.')    
    parser.add_argument('--model_output', type=str, default='models/mobilenetv2_quant.onnx', help='Path to save the quantized ONNX model.')    
    parser.add_argument('--calib_data', type=str, default='calib_data', help='Path to the calibration dataset.')    
    parser.add_argument('--fast_finetune', action='store_true', help='Use fast fine-tuning configuration.')    
    parser.add_argument('--cross_layer_equalization', action='store_true', help='Use cross-layer equalization configuration.')   
  
    args = parser.parse_args()    
    main(args)    





