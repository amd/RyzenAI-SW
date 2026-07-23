<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Quantization Tutorial </h1>
    </td>
 </tr>
</table>

## Ryzen AI Model Quantization and Deployment

Welcome to the Ryzen AI Model Quantization tutorial. In this guide, you'll learn how to quantize your deep learning models for efficient deployment on NPU.

**AMD Quark** is a comprehensive cross-platform deep learning toolkit designed to simplify and enhance the quantization of deep learning models. Quark enables developers to optimize their models for deployment on a wide range of hardware backends, achieving significant performance gains without compromising accuracy.

Quark helps restore model accuracy that may be lost after quantization through various post training quantization techniques.

For more details refer to the [AMD Quark official documentation](https://quark.docs.amd.com/latest/index.html)

## Quantization Tutorial

In this tutorial we will provide step by step guide for quantization of CNN models using [AMD Quark](https://quark.docs.amd.com/latest/index.html) quantization API. **Quark** for ONNX leverages the power of the ONNX Runtime Quantization tool, providing a robust and flexible solution for quantizing ONNX models. We will demonstrate how to quantize and run the [ResNet50](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx) model ONNX Runtime on Ryzen AI PCs.

### Setup Environment

Create a clone of the Ryzen AI installation conda environment to add required python packages

### Create and Activate Conda Environment

```python
set RYZEN_AI_CONDA_ENV_NAME=ryzen-ai-<version>
conda create --name quark_quantization --clone %RYZEN_AI_CONDA_ENV_NAME%
conda activate quark_quantization
```

Add python packages needed for the tutorial:

```bash
cd ryzenai-sw\tutorial\quark_quantization
pip install -r requirements.txt
```

Input model
-----------

Download the pre-trained float ONNX models for [ResNet50](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx)

The downloaded model should be placed in the models folder, and the script ``quark_quantize.py`` will use it as the input model for quantization as shown below.
```python
input_model_path = "models\\resnet50.onnx"
output_model_path = "models\\resnet50_quant.onnx"
```

Alternatively you can use the python script to download the pre-trained ResNet50 model

```python
cd models
python download_ResNet.py
```

ImageNet Dataset
----------------
If you already have an ImageNet datasets, you can directly use your dataset path.

To prepare the test data, please check the download section of the main website: https://huggingface.co/datasets/ILSVRC/imagenet-1k/tree/script/data. You need to register and download **val_images.tar.gz**.

Then, create the validation dataset and calibration dataset:

```python
mkdir val_data && tar -xzf val_images.tar.gz -C val_data
python prepare_data.py val_data calib_data
```

Quantization configuration
--------------------------
Users can apply their own **Customer Settings** for the ``QuantizationConfig``, ``CalibrationMethod``, or use the some of predefined configurations  ``A8W8``  or ``A16W8`` they can refer to the Quark's User Guide for help. ``XINT8`` is the special case within ``A8W8`` with power of two scales and zero offset for improved performance, which can be used for models with minimal accuracy drop after quantization. The quantization configuration is defined as below:

```python
from quark.onnx.quantization.config import Config, get_default_config
from quark.onnx.quantization.config.config import QuantizationConfig
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantType, QuantFormat

# Use default quantization configuration
quant_config = get_default_config("XINT8")

# Define custom quantization configuration
# quant_config = QuantizationConfig(calibrate_method=PowerOfTwoMethod.MinMSE,
#                                   activation_type=QuantType.QUInt8,
#                                   weight_type=QuantType.QInt8,
#                                   enable_npu_cnn=True,
#                                   extra_options={'ActivationSymmetric': True})

# Defines the quantization configuration for the whole model
config = Config(global_quant_config=quant_config)
print("The configuration of the quantization is {}".format(config))
```

Calibration Dataset
-------------------
The calibration dataset is used to compute the quantization parameters for the activations. Use the ``ImageDataReader`` class to setup the calibration dataset

```python
from utils import ImageDataReader

num_calib_data = 1000
calibration_dataset = ImageDataReader(calibration_dataset_path, input_model_path, data_size=num_calib_data, batch_size=1)
```

Model Quantization
------------------
The model quantizer save the quantized model model at ``output_model_path``

```python
from quark.onnx import ModelQuantizer
# Create an ONNX Quantizer
quantizer = ModelQuantizer(config)

# Quantize the ONNX model
quant_model = quantizer.quantize_model(model_input = input_model_path,
                                       model_output = output_model_path,
                                       calibration_data_path = None)
```

Model Size
----------
Print the original and quantized models.

```python
print("Model Size:")
print("Float32 model size: {:.2f} MB".format(os.path.getsize(input_model_path)/(1024 * 1024)))
print("{} quantized model size: {:.2f} MB".format(args.config, os.path.getsize(output_model_path)/(1024 * 1024)))
```

Model Evaluation
----------------
Print the original and quantized models accuracy

```python
from utils import evaluate_onnx_model

print("Model Accuracy:")
top1_acc, top5_acc = evaluate_onnx_model(input_model_path, imagenet_data_path='calib_data')
print("Float32 model accuracy: Top1 {:.3f}, Top5 {:.3f} ".format(top1_acc, top5_acc))
top1_acc, top5_acc = evaluate_onnx_model(output_model_path, imagenet_data_path='calib_data')
print("{} quantized model accuracy: Top1 {:.3f}, Top5 {:.3f} ".format(args.config, top1_acc, top5_acc))
top1_acc, top5_acc = evaluate_onnx_model(output_model_path, imagenet_data_path='calib_data', device='npu')
print("{} quantized model accuracy (NPU): Top1 {:.3f}, Top5 {:.3f} ".format(args.config, top1_acc, top5_acc))
```

Accuracy Summary
----------------

Quantize and Evaluation the top1 and top5 accuracy of the models on ImageNet validation dataset using the ``quark_quantize.py`` script.

```python
python quark_quantize.py --model_input 'models\resnet50.onnx' --model_output 'models\resnet50_a8w8.onnx' --config A8W8 --quantize --evaluate
```

ResNet50: Using ``A8W8`` configuration

<div align="center">

| ResNet50      | Model Size | Top-1 Accuracy | Top-5 Accuracy |
|---------------|------------|----------------|----------------|
| Float 32      |  97.41 MB  | 88.0%          | 98.0%          |
| A8W8 (CPU)    |  24.54 MB  | 83.0%          | 95.0%          |
| A8W8 (NPU)    |  24.54 MB  | 81.0%          | 96.0%          |

</div>

```python
python quark_quantize.py --model_input 'models\resnet50.onnx' --model_output 'models\resnet50_xint8.onnx' --config XINT8 --quantize --evaluate
```

ResNet50: Using ``XINT8`` configuration

<div align="center">

| ResNet50       | Model Size | Top-1 Accuracy | Top-5 Accuracy |
|----------------|------------|----------------|----------------|
| Float 32       |  97.41 MB  | 88.0%          | 98.0%          |
| XINT8 (CPU)    |  24.46 MB  | 85.0%          | 97.0%          |
| XINT8 (NPU)    |  24.46 MB  | 85.0%          | 98.0%          |

</div>

```python
python quark_quantize.py --model_input 'models\resnet50.onnx' --model_output 'models\resnet50_a16w8.onnx' --config A16W8 --quantize --evaluate
```

ResNet50: Using ``A16W8`` configuration

<div align="center">

| ResNet50      | Model Size | Top-1 Accuracy | Top-5 Accuracy |
|---------------|------------|----------------|----------------|
| Float 32      |  97.41 MB  | 80.0%          | 98.0%          |
| A16W8 (CPU)   |  24.54 MB  | 83.0%          | 97.0%          |
| A16W8 (NPU)   |  24.54 MB  | 83.0%          | 97.0%          |

</div>

Run model on CPU
----------------

```python
quant_model = onnx.load(output_model_path)
provider = ['CPUExecutionProvider']

session = ort.InferenceSession(quant_model.SerializeToString(), providers=provider)
input_data = preprocess_image('test_image.jpg')
# run the model in onnxruntime session
outputs = session.run(None, {session.get_inputs()[0].name: input_data})
predicted_class = np.argmax(outputs[0])
print('CPU quantized outputs:')
print(predicted_class)
```


Run model on NPU
----------------

```python
quant_model = onnx.load(output_model_path)
provider = ['VitisAIExecutionProvider']
cache_dir = Path(__file__).parent.resolve()
print(cache_dir)
provider_options = [{
                    'cacheDir': str(cache_dir),
                    'cacheKey': 'modelcachekey',
                    'enable_cache_file_io_in_mem':'0'
                }]
# For PHX/HPT, xclbin is required
if npu_device == 'PHX/HPT':
   provider_options[0]['target'] = 'X1'
   provider_options[0]['xclbin'] = get_xclbin(npu_device)

session = ort.InferenceSession(quant_model.SerializeToString(), providers=provider,
                               provider_options=provider_options)

input_data = preprocess_image('test_image.jpg')
outputs = session.run(None, {session.get_inputs()[0].name: input_data})
predicted_class = np.argmax(outputs[0])
print('NPU quantized outputs:')
print(predicted_class)
```

Inference Performance
---------------------

Evaluate the inference performance of the float and quantized models on CPU and NPU

```python
python quark_quantize.py --config XINT8 --quantize --evaluate --benchmark
```

<div align="center">

| ResNet50       | Model Size  | Top-1 Accuracy | Top-5 Accuracy  | Inference Time |
|----------------|-------------|----------------|-----------------|----------------|
| Float 32       |  97.41 MB   | 88.0%          | 98.0%           | 21.18 ms       |
| XINT8 (CPU)    |  24.46 MB   | 85.0%          | 97.0%           | 47.04 ms       |
| XINT8 (NPU)    |  24.46 MB   | 85.0%          | 98.0%           | 16.41 ms       |

</div>

Advancted Quantization Tools
----------------------------

While the default quantization configurations work well for many popular models, more sophisticated models might experience a decline in accuracy due to errors introduced during the quantization process. To address this, Quark APIs offer advanced tools to help recover lost accuracy. Some of these tools are highlighted in the [Advanced Quantization Tools](docs/advanced_quant_readme.md) tutorial.

