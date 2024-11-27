<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Advanced Quantization Tools </h1>
    </td>
 </tr>
</table>

## Advanced Quantization Tools

In this section, we explore the advanced quantization capabilities of the Quark quantizer, designed to recover the lost accuracy in quantized models. While basic quantization configurations are effective for many models, advanced and optimized models often require sophisticated techniques to enhance the accuracy of the quantized versions. This guide will walk you through these advanced methods, ensuring your models maintain high performance even after quantization.

This tutorials takes [MobileNetV2](https://github.com/onnx/models/blob/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx) onnx model as an example, which can be challenging to quantize with minimal accuracy loss, using the advanced quark quantization tools.

ImageNet Dataset
----------------

Please ensure to setup the validation and calibration datase using the instruction from [Quark Quantization Tutorial](./onnx/cnn_quant_readme.md)


Model Evaluation
----------------

MobileNet: Using ``XINT8`` configuration, we see a drop of ~3% in the Top-1 accuracy. Optimized models like MobileNetV2 tend to be more difficult to quantize. To bridge the gap between float and quantized accuracy of the model, we can use some advanced quantization configurations or techniques. 

```python
python advanced_quark_quantize.py --model_input models/mobilenetv2.onnx --model_output models/mobilenetv2_quant.onnx 
```

<div align="center">

| MobileNetV2   | Model Size | Top-1 Accuracy | Top-5 Accuracy |
|---------------|------------|----------------|----------------|  
| Float 32      |  13.34 MB  | 71.3%          | 90.6%          |  
| INT8 (CPU)    |   3.44 MB  | 64.0%          | 86.5%          |  
| INT8 (NPU)    |   3.44 MB  | 63.7%          | 87.0%          |  

</div>

ResNet50: Using ``XINT8`` configuration

```python
python advanced_quark_quantize.py --model_input models/resnet50.onnx --model_output models/resnet50_quant.onnx 
```

<div align="center">

| ResNet50      | Model Size | Top-1 Accuracy | Top-5 Accuracy |
|---------------|------------|----------------|----------------|  
| Float 32      |  97.41 MB  | 80.0%          | 96.1%          |  
| INT8 (CPU)    |  24.46 MB  | 77.3%          | 94.9%          |  
| INT8 (NPU)    |  24.46 MB  | 77.4%          | 95.2%          |  

</div>

### Fast Fine Tuning

Fast fine-tuning involves adjusting a pre-trained model to enhance its accuracy after quantization. This approach helps recover accuracy lost during quantization, making the model more suitable for deployment.

```python
INT8_CNN_ACCURATE_CONFIG = QuantizationConfig(calibrate_method=CalibrationMethod.Percentile,
                                              activation_type=QuantType.QUInt8,
                                              weight_type=QuantType.QInt8,
                                              include_fast_ft=True,
                                              extra_options={
                                                  'Percentile': 99.9999,
                                                  'FastFinetune': DEFAULT_ADAROUND_PARAMS
                                              })
config = Config(global_quant_config=INT8_CNN_ACCURATE_CONFIG)
``` 

MobileNet: Using ``INT8_CNN_ACCURATE`` configuration, which improve the accuracy of the model through ``Fast Fine-tuning`` and ``Histogram Percentile`` based techniques.

```python
python advanced_quark_quantize.py --model_input models/mobilenetv2.onnx --model_output models/mobilenetv2_quant.onnx --fast_finetune
```

<div align="center">

| MobileNetV2   | Model Size | Top-1 Accuracy | Top-5 Accuracy |
|---------------|------------|----------------|----------------|  
| Float 32      |  13.32     | 71.3%          | 90.6%          |  
| INT8 (CPU)    |   3.43     | 70.5%          | 90.3%          |  
| INT8 (NPU)    |   3.43     | 69.6%          | 89.4%          |  

</div>

ResNet50: Using ```Fast Fine-tuning`` configuration

```python
python advanced_quark_quantize.py --model_input models/resnet50.onnx --model_output models/resnet50_quant.onnx --fast_finetune
```

<div align="center">

| ResNet50      | Model Size | Top-1 Accuracy | Top-5 Accuracy |
|---------------|------------|----------------|----------------|  
| Float 32      |  97.41 MB  | 80.0%          | 96.1%          |  
| INT8 (CPU)    |  24.46 MB  | 79.3%          | 96.2%          |  
| INT8 (NPU)    |  24.46 MB  | 77.4%          | 95.2%          | 

</div>

### Cross Layer Equalization (CLE)

Cross-Layer Equalization (CLE) optimizes neural networks for quantization by balancing weight distributions across layers, reducing quantization errors. This technique helps maintain model accuracy while enabling efficient quantization.

```python
INT8_CLE_CONFIG = QuantizationConfig(calibrate_method=PowerOfTwoMethod.MinMSE,
                                    activation_type=QuantType.QUInt8,
                                    weight_type=QuantType.QInt8,
                                    enable_npu_cnn=True,
                                    include_cle=True,
                                    extra_options={'ActivationSymmetric': True})

config = Config(global_quant_config=INT8_CLE_CONFIG)
``` 

MobileNet: Using ``Cross Layer Equalization`` configuration

```python
python advanced_quark_quantize.py --model_input models/mobilenetv2.onnx --model_output models/mobilenetv2_quant.onnx --cross_layer_equalization
```

<div align="center">

| MobileNetV2   | Model Size | Top-1 Accuracy | Top-5 Accuracy |
|---------------|------------|----------------|----------------|  
| Float 32      |  13.32     | 71.3%          | 90.6%          |  
| INT8 (CPU)    |   3.43     | 62.7%          | 85.4%          |
| INT8 (NPU)    |   3.43     | 63.7%          | 86.4%          |

</div>

ResNet50: Using ``Cross Layer Equalization`` configuration

```python
python advanced_quark_quantize.py --model_input models/resnet50.onnx --model_output models/resnet50_quant.onnx --cross_layer_equalization
```

<div align="center">

| ResNet50      | Model Size | Top-1 Accuracy | Top-5 Accuracy |
|---------------|------------|----------------|----------------|  
| Float 32      |  97.41 MB  | 80.0%          | 96.1%          |  
| INT8 (CPU)    |  24.46 MB  | 77.7%          | 95.7%          |
| INT8 (NPU)    |  24.46 MB  | 78.2%          | 95.6%          |

</div>

Reference
---------

For more details on the Quark API features in the [Quark Documentation](https://quark.docs.amd.com/latest/index.html)