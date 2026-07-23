import os
import argparse
import onnxruntime
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config
from quark.onnx.quantization.config.legacy import QuantizationConfig
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantType
from quark.onnx.quantization.quant_utils import ExtendedQuantFormat
from utils import top1_accu, ImageDataReader, evaluate_onnx_model

# Custom quantization parameters for advanced fine-tuning algorithms
# These parameters follow the pattern defined in quark.onnx.quantization.config.custom_config

DEFAULT_ADAROUND_PARAMS = {
    "DataSize": 1000,
    "FixedSeed": 1705472343,
    "BatchSize": 2,
    "NumIterations": 1000,
    "LearningRate": 0.1,
    "OptimAlgorithm": "adaround",
    "OptimDevice": "cpu",
    "InferDevice": "cpu",
    "EarlyStop": True,
}

DEFAULT_ADAQUANT_PARAMS = {
    "DataSize": 1000,
    "FixedSeed": 1705472343,
    "BatchSize": 2,
    "NumIterations": 1000,
    "LearningRate": 0.00001,
    "OptimAlgorithm": "adaquant",
    "OptimDevice": "cpu",
    "InferDevice": "cpu",
    "EarlyStop": True,
}

# Custom A8W8 quantization configurations following quark.onnx.quantization.config.custom_config pattern
# A8W8: 8-bit symmetric activations, 8-bit symmetric weights

A8W8_CONFIG = QuantizationConfig(
    calibrate_method=CalibrationMethod.MinMax,
    quant_format=ExtendedQuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    extra_options={
        "ActivationSymmetric": True,
        "AlignSlice": False,
        "FoldRelu": True,
        "AlignConcat": True,
    },
)

A8W8_ADAROUND_CONFIG = QuantizationConfig(
    calibrate_method=CalibrationMethod.MinMax,
    quant_format=ExtendedQuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    include_fast_ft=True,
    extra_options={
        "ActivationSymmetric": True,
        "AlignSlice": False,
        "FoldRelu": True,
        "AlignConcat": True,
        "FastFinetune": DEFAULT_ADAROUND_PARAMS,
    },
)

A8W8_ADAQUANT_CONFIG = QuantizationConfig(
    calibrate_method=CalibrationMethod.MinMax,
    quant_format=ExtendedQuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    include_fast_ft=True,
    extra_options={
        "ActivationSymmetric": True,
        "AlignSlice": False,
        "FoldRelu": True,
        "AlignConcat": True,
        "FastFinetune": DEFAULT_ADAQUANT_PARAMS,
    },
)

def main(args):
    # Setup the Input model
    input_model_path = args.model_input
    output_model_path = args.model_output
    calibration_dataset_path = args.calib_data
    quant_type = 'A8W8'

    # Select quantization configuration based on arguments
    # Using custom configurations defined above following quark.onnx.quantization.config.custom_config pattern
    if args.adaround:
        quant_config = A8W8_ADAROUND_CONFIG
        print("Using custom A8W8_ADAROUND_CONFIG")
    elif args.adaquant:
        quant_config = A8W8_ADAQUANT_CONFIG
        print("Using custom A8W8_ADAQUANT_CONFIG")
    else:
        quant_config = A8W8_CONFIG
        print("Using custom A8W8_CONFIG")

    # Enable NPU CNN optimizations
    quant_config.enable_npu_cnn = True

    # Optionally enable cross-layer equalization
    if args.cross_layer_equalization:
        quant_config.include_cle = True
        print("Cross-layer equalization enabled")

    # Defines the quantization configuration for the whole model
    config = Config(global_quant_config=quant_config)
    print("\nQuantization Configuration:")
    print(f"  Calibration Method: {quant_config.calibrate_method}")
    print(f"  Activation Type: {quant_config.activation_type}")
    print(f"  Weight Type: {quant_config.weight_type}")
    print(f"  Quant Format: {quant_config.quant_format}")
    print(f"  NPU CNN Enabled: {quant_config.enable_npu_cnn}")
    print(f"  Fast Fine-tune: {quant_config.include_fast_ft}")
    if args.cross_layer_equalization:
        print(f"  Cross-layer Equalization: {quant_config.include_cle}")

    # Define the calibration data reader
    num_calib_data = 100
    calibration_dataset = ImageDataReader(calibration_dataset_path, input_model_path, data_size=num_calib_data, batch_size=1)

    # Create an ONNX Quantizer
    quantizer = ModelQuantizer(config)

    # Quantize the ONNX model
    quant_model = quantizer.quantize_model(model_input=input_model_path,
                                           model_output=output_model_path,
                                           calibration_data_reader=calibration_dataset)

    print("Model Size:")
    print("Float32 model size: {:.2f} MB".format(os.path.getsize(input_model_path)/(1024 * 1024)))
    print("{} quantized model size: {:.2f} MB".format(quant_type, os.path.getsize(output_model_path)/(1024 * 1024)))

    # Evaluate the model
    print("Model Accuracy:")
    top1_acc, top5_acc = evaluate_onnx_model(input_model_path, imagenet_data_path=calibration_dataset_path)
    print("Float32 model accuracy: Top1 {:.3f}, Top5 {:.3f} ".format(top1_acc, top5_acc))
    top1_acc, top5_acc = evaluate_onnx_model(output_model_path, imagenet_data_path=calibration_dataset_path)
    print("{} quantized model accuracy: Top1 {:.3f}, Top5 {:.3f} ".format(quant_type, top1_acc, top5_acc))
    top1_acc, top5_acc = evaluate_onnx_model(output_model_path, imagenet_data_path=calibration_dataset_path, device='npu')
    print("{} quantized model accuracy (NPU): Top1 {:.3f}, Top5 {:.3f} ".format(quant_type, top1_acc, top5_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize and evaluate ONNX models.")
    parser.add_argument('--model_input', type=str, default='models/mobilenetv2.onnx', help='Path to the input ONNX model.')
    parser.add_argument('--model_output', type=str, default='models/mobilenetv2_quant.onnx', help='Path to save the quantized ONNX model.')
    parser.add_argument('--calib_data', type=str, default='calib_data', help='Path to the calibration dataset.')
    parser.add_argument('--cross_layer_equalization', action='store_true', help='Use cross-layer equalization configuration.')
    parser.add_argument('--adaround', action='store_true', help='Use ADAROUND configuration.')
    parser.add_argument('--adaquant', action='store_true', help='Use ADAQUANT configuration.')

    args = parser.parse_args()
    main(args)





