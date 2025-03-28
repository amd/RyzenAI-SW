from quark.onnx.quantization.config import Config, get_default_config
from quark.onnx.quantization.config.config import QuantizationConfig
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantType, QuantFormat
import argparse
import os

def main(args):
    # Use default quantization configuration
    quant_config = get_default_config("BF16")
    quant_config.extra_options["BF16QDQToCast"] = True
    config = Config(global_quant_config=quant_config)
    config.global_quant_config.extra_options["UseRandomData"] = True
    print("The configuration of the quantization is {}".format(config))

    from quark.onnx import ModelQuantizer
    # Create an ONNX Quantizer
    quantizer = ModelQuantizer(config)

    input_model_path = os.path.join(args.model_path)
    output_model_path = os.path.join(args.output_dir,"gte-large-en-v1.5-bf16.onnx")

    # Quantize the ONNX model
    quant_model = quantizer.quantize_model(model_input = input_model_path,
                                       model_output = output_model_path,
                                       calibration_data_path = None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export from Huggingface to ONNX model')
    parser.add_argument('--model_path', type=str, required=True,default="models/model.onnx", help='Name or path of the Hugging Face model')
    parser.add_argument('--output_dir', type=str, required=True,default ='models',help='Output directory for the ONNX model')

    args = parser.parse_args()

    main(args)