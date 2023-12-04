<table style="width:100%">
  <tr>

<th width="100%" colspan="6"><img src="https://github.com/Xilinx/Image-Collateral/blob/main/xilinx-logo.png?raw=true" width="30%"/><h1>Ryzen AI Quantization Tutorial</h1>
</th>

  </tr>
  <tr>
    <td width="17%" align="center"><a href=../README.md>1.Introduction</td>
    <td width="17%" align="center">2.ONNX Quantization Tutorial</a>  </td>
    <td width="16%" align="center"><a href="./PT_README.md">3. Pytorch Quantization Tutorial</a></td>
    <td width="17%" align="center"><a href="./TF_README.md">4.Tensorflow1.x quantization tutorial</a></td>
    <td width="17%" align="center"><a href="./TF2_README.md"> 5.Tensorflow2.x quantization tutorial<a></td>

</tr>

</table>


## 2. ONNX Quantization Tutorial

### Introduction

This tutorials takes Resnet50 onnx model as an example and shows how to generate quantized onnx model with Ryzen AI quantizer. Then you can run it with onnxruntime on Ryzen AI PCs. 


### Test Environment
VAI_Q_ONNX operates under the following test environment requirements:
- Python3.8
- onnx>=1.12.0
- onnxruntime>=1.15.0
- onnxruntime-extensions>=0.4.2

### Create and Activate Conda Environment
```shell
$ cd <RyzenAI-SW>/tutorial/RyzenAI_quant_tutorial/onnx_example
$ conda env create --name ${env_name} --file=env.yaml
$ conda activate ${env_name}
``` 

### Installation
You can easily install VAI_Q_ONNX by following the steps below:

1. Build from source code

    Build VAI_Q_ONNX using the provided build script:
    ```bash
    $ cd tutorial/RyzenAI_quant_tutorial/onnx_example
    $ sh build.sh
    $ pip install pkgs/*.whl
    ```


2. Install from pre-built wheel packge
  If you can't build the wheel package successfully, you can also use the existing package we've built for you. Follow these steps to install it:

    ```shell
    $ cd tutorial/RyzenAI_quant_tutorial/onnx_example
    $ pip install pkgs/*.whl
    ```

3. For native Windows users, We recommend the following commands in *command prompt* for a minimal installation that not include custom operations library:
    ```shell
    $ cd tutorial/RyzenAI_quant_tutorial/onnx_example
    $ python3 setup.py bdist_wheel --release    --dist-dir=pkgs
    $ pip install pkgs/*.whl
    ```
  ### Quick Start 
 1. Prepare data and model 
 To Prepare the test data, please check the download section of the main website: https://huggingface.co/datasets/imagenet-1k/tree/main/data
You need to register and download val_images.tar.gz, Then, create the validation dataset and calibration dataset.
    ```shell
    $ cd tutorial/RyzenAI_quant_tutorial/onnx_example/onnx_model_ptq
    $ mkdir val_data && tar -xzf val_images.tar.gz -C val_data
    $ python prepare_data.py val_data calib_data
    ```

## Running vai_q_onnx

Quantization in ONNX Runtime refers to the linear quantization of an ONNX model. We have developed the vai_q_onnx tool as a plugin for ONNX Runtime to support more post-training quantization(PTQ) functions for quantizing a deep learning model. Post-training quantization(PTQ) is a technique to convert a pre-trained float model into a quantized model with little degradation in model accuracy. A representative dataset is needed to run a few batches of inference on the float model to obtain the distributions of the activations, which is also called quantized calibration.


vai_q_onnx supports static quantization and the usage is as follows.




1.  #### Preparing the Float Model and Calibration Set

Before running vai_q_onnx, prepare the float model and calibration set, including the files listed in the following table.


- Dataset Root Directory
  - Category 1 Folder
    - Image1.jpg
    - Image2.jpg
    - ...
  - Category 2 Folder
    - Image1.jpg
    - Image2.jpg
    - ...
    - ...
  - Category N Folder
    - Image1.jpg
    - Image2.jpg
    - ...

If you already have an ImageNet datasets with the above structure, you can also directly use your dataset path instead of val_data.

Finally, download onnx model from the link.

```shell 
$ mkdir models && wget https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-12.onnx
```


### Quntization
Quantization tool takes the pre-processed float32 model and produce a quantized model.
For CPU config:

```shell
python quantize_onnx_model_cpu.py models/resnet50-v1-12.onnx models/resnet50-v1-12.U8S8.fs.onnx calib_data
```
This will generate quantized model using QDQ quant format and Int8 activation type and Int8 weight type with float scale to models/resnet50-v1-12.U8S8.fs.onnx

For IPU config:
```shell
$ python quantize_onnx_model_ipu.py models/resnet50-v1-12.onnx models/resnet50-v1-12.U8S8.pof2s.onnx calib_data
```


### Evalution

Table 1. Input files for vai_q_onnx
Test the accuracy of a float model on ImageNet val dataset (Prec@1 76.130 Prec@5 92.862):


```shell
$ python onnx_validate.py val_data --onnx-input models/resnet50-v1-12.onnx
```
For a float model, you can use GPU to accelerate the evaluation.

```shell
$ python onnx_validate.py val_data --onnx-input models/resnet50-v1-12.onnx --gpu
```

Test the accuracy of a CPU quantized config model on ImageNet val dataset (Prec@1 72.55% Prec@5 90.95%):
```shell
$ python onnx_validate.py val_data --onnx-input models/resnet50-v1-12.U8S8.fs.onnx
```
Test the accuracy of a IPU quantized config model on ImageNet val dataset (Prec@1 73.52% Prec@5 91.29%):

```shell
$ python onnx_validate.py val_data --onnx-input models/resnet50-v1-12.U8S8.pof2s.onnx
```


|  | Float Model | Quantized Model For IPU |Quantized Model For CPU|
| --- | --- | --- | --- |
| Model | models/resnet50-v1-12.onnx | models/resnet50-v1-12.U8S8.pof2s.onnx |resnet50-v1-12.U8S8.pof2s.onnx | models/resnet50-v1-12.U8S8.fs.onnx |
| Model Size | 97.82 MB | 24.48 MB|24.96 MB|
| Prec@1| 74.11%|73.52%|72.55%|
|Prec@5	|91.71 %	|91.29 %	|90.95 %|

### VAI_Q_ONNX APIs

The post-training quantization process is a process to convert the pretrained deep learning models from higher precision (e.g., 32-bit floating-point) to lower precision (e.g., 8-bit integers). This reduces the memory requirements and computational complexity of the model. It needs a set of representative data (called calibration data) to calculate the quantize parameters for the activation tensors. Here is an example of running PTQ with IPU_CNN configurations. We also support quantization without real calibration data for rapid validation of deployment or performance benchmarking.


```python
import vai_q_onnx

vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader, # Set calibration_data_reader=None to do random data quantization
    quant_format=vai_q_onnx.QuantFormat.QDQ,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    activation_type=vai_q_onnx.QuantType.QUInt8,
    weight_type=vai_q_onnx.QuantType.QInt8,
    enable_dpu=True,
    extra_options={'ActivationSymmetric':True}
)
```

**Arguments**

* **model_input**: (String) This parameter represents the file path of the model to be quantized.
* **model_output**: (String) This parameter represents the file path where the quantized model will be saved.
* **calibration_data_reader**: (Object or None) This parameter is a calibration data reader. It enumerates the calibration data and generates inputs for the original model. If you wish to use random data for a quick test, you can set calibration_data_reader to None. The default value is None.
* **quant_format**: (String) This parameter is used to specify the quantization format of the model. It has the following options:
  -  vai_q_onnx.QuantFormat.QOperator: This option quantizes the model directly using quantized operators.
  -  vai_q_onnx.QuantFormat.QDQ: This option quantizes the model by inserting QuantizeLinear/DeQuantizeLinear into the tensor. It supports 8-bit quantization only.
  -  vai_q_onnx.VitisQuantFormat.QDQ: This option quantizes the model by inserting customized VitisQuantizeLinear/VitisDeQuantizeLinear into the tensor. It supports a wider range of bit-widths and precisions.
  -  vai_q_onnx.VitisQuantFormat.FixNeuron (Experimental): This option quantizes the model by inserting FixNeuron (a combination of QuantizeLinear and DeQuantizeLinear) into the tensor. This quant format is currently experimental and cannot use for actual deployment.
* **calibrate_method**: (String) The method used in calibration, default to vai_q_onnx.PowerOfTwoMethod.MinMSE.

    For IPU_CNN platforms, power-of-two methods should be used, options are:
  -  vai_q_onnx.PowerOfTwoMethod.NonOverflow: This method get the power-of-two quantize parameters for each tensor to make sure min/max values not overflow.
  -  vai_q_onnx.PowerOfTwoMethod.MinMSE: This method get the power-of-two quantize parameters for each tensor to minimize the mean-square-loss of quantized values and float values. This takes longer time but usually gets better accuracy.

    For IPU_Transformer or CPU platforms, float scale methods should be used, options are:
  -  vai_q_onnx.CalibrationMethod.MinMax: This method obtains the quantization parameters based on the minimum and maximum values of each tensor.
  -  vai_q_onnx.CalibrationMethod.Entropy: This method determines the quantization parameters by considering the entropy algorithm of each tensor's distribution.
  -  vai_q_onnx.CalibrationMethod.Percentile: This method calculates quantization parameters using percentiles of the tensor values.
* **activation_type**: (QuantType) Specifies the quantization data type for activations.
* **weight_type**: (QuantType) Specifies the quantization data type for weights, For DPU/IPU devices, this must be set to QuantType.QInt8.
* **enable_dpu**: (Boolean) This parameter is a flag that determines whether to generate a quantized model that adapts the approximations and constraints the DPU/IPU. If set to True, the quantization process will consider the specific limitations and requirements of the DPU/IPU.
* **extra_options**:  (Dictionary or None) Contains key-value pairs for various options in different cases.
  -  ActivationSymmetric: (Boolean) If True, symmetrize calibration data for activations. For DPU/IPU, this need be set to True.
  For more details of the extra_options parameters, please refer to the [extra_options](#extra_options).


### Recommended Configurations

#### Configurations For IPU_CNN  

For CNN-based models, this configuration used power-of-2 scale, symmeric activation, symmetric weights and applied operation approximation and constraints adaptation for IPU_CNN targets.

```python
import vai_q_onnx

vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader,
    quant_format=vai_q_onnx.QuantFormat.QDQ,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    activation_type=vai_q_onnx.QuantType.QUInt8,
    weight_type=vai_q_onnx.QuantType.QInt8,
    enable_dpu=True,
    extra_options={'ActivationSymmetric':True}
)
```

#### Configurations For IPU_Transformer 

For Transformer-based models, this configuration used float scale, asymmeric activation, symmetric weights for IPU_Transformer targets.

```python
import vai_q_onnx

vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader,
    quant_format=vai_q_onnx.QuantFormat.QDQ,
    calibrate_method=vai_q_onnx.CalibrationMethod.MinMax,
    activation_type=vai_q_onnx.QuantType.QInt8,
    weight_type=vai_q_onnx.QuantType.QInt8,
)
```

#### Configurations For CPU  

This configuration is used to accelerate inference on CPU. It used float scale, asymmeric activation, symmetric weights and bias. It quantized activation to Uint8, weights to Int8 while biases to Int32.

```python
import vai_q_onnx

vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader,
    quant_format=vai_q_onnx.QuantFormat.QDQ,
    calibrate_method=vai_q_onnx.CalibrationMethod.MinMax,
    activation_type=vai_q_onnx.QuantType.QUInt8,
    weight_type=vai_q_onnx.QuantType.QInt8
)
```

## Running VAI_Q_ONNX Post-Training Quantization(PTQ)

Quantization in ONNX Runtime refers to the linear quantization of an ONNX model. We have developed the VAI_Q_ONNX tool as a plugin for ONNX Runtime to support more post-training quantization(PTQ) functions for quantizing a deep learning model. Post-training quantization(PTQ) is a technique to convert a pre-trained float model into a quantized model with little degradation in model accuracy. A representative dataset is needed to run a few batches of inference on the float model to obtain the distributions of the activations, which is also called quantized calibration.

Here are the steps for running PTQ.

### 1. Preparing the Float Model and Calibration DataSet

Before running VAI_Q_ONNX, prepare the float model and calibration set, including the files listed in the following table.

Table 1. Input files for VAI_Q_ONNX

| No. | Name | Description |
| ------ | ------ | ----- |
| 1 | float model | Floating-point ONNX models in onnx format. |
| 2 | calibration dataset | A subset of the training dataset or validation dataset to represent the input data distribution, usually 100 to 1000 images are enough. |

* Exporting PyTorch Models to ONNX

For PyTorch models, it is recommended to use the TorchScript-based onnx exporter for exporting ONNX models. Please refer to the [PyTorch documentation for guidance](https://pytorch.org/docs/stable/onnx_torchscript.html#torchscript-based-onnx-exporter). 

Tips:
1. Before exporting, please perform the model.eval().
2. Models with opset 17 are recommended.
3. For IPU_CNN platforms, dynamic input shapes are currently not supported and only a batch size of 1 is allowed. Please ensure that the shape of input is a fixed value, and the batch dimension is set to 1.

Example code:
```python
torch.onnx.export(
    model,
    input,
    model_output_path,
    opset_version=17,
    input_names=['input'],
    output_names=['output'],
)
```

* **Opset Versions**: Models with opset 17 are recommended. Models must be opset 10 or higher to be quantized. Models with opset lower than 10 should be reconverted to ONNX from their original framework using a later opset. Alternatively, you can refer to the usage of the version converter for [ONNX Version Converter](https://github.com/onnx/onnx/blob/main/docs/VersionConverter.md). Opset 10 does not support some node fusions and may not get the best performance. We recommend to update the model to opset 17 for better performance. Moreover, per channel quantization is supported for opset 13 or higher versions.

* **Large Models > 2GB**: Due to the 2GB file size limit of Protobuf, for ONNX models exceeding 2GB, additional data will be stored separately. Please ensure that the .onnx file and the data file are placed in the same directory. Also, please set the use_external_data_format parameter to True for large models when quantizing.

### 2. Quantizing Using the VAI_Q_ONNX API
The static quantization method first runs the model using a set of inputs called calibration data. During these runs, we compute the quantization parameters for each activation. These quantization parameters are written as constants to the quantized model and used for all inputs. VAI_Q_ONNX quantization tool has expanded calibration methods to power-of-2 scale/float scale quantization methods. Float scale quantization methods include MinMax, Entropy, and Percentile. Power-of-2 scale quantization methods include MinMax and MinMSE.

```python
vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader,
    quant_format=vai_q_onnx.QuantFormat.QDQ,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    input_nodes=[],
    output_nodes=[],
    enable_dpu=True,
    extra_options={'ActivationSymmetric':True}
)
```

**Arguments**

* **model_input**: (String) This parameter represents the file path of the model to be quantized.
* **model_output**: (String) This parameter represents the file path where the quantized model will be saved.
* **calibration_data_reader**: (Object or None) This parameter is a calibration data reader. It enumerates the calibration data and generates inputs for the original model. If you wish to use random data for a quick test, you can set calibration_data_reader to None. The default value is None.
* **quant_format**: (String) This parameter is used to specify the quantization format of the model. It has the following options:
  -  vai_q_onnx.QuantFormat.QOperator: This option quantizes the model directly using quantized operators.
  -  vai_q_onnx.QuantFormat.QDQ: This option quantizes the model by inserting QuantizeLinear/DeQuantizeLinear into the tensor. It supports 8-bit quantization only.
  -  vai_q_onnx.VitisQuantFormat.QDQ: This option quantizes the model by inserting VitisQuantizeLinear/VitisDeQuantizeLinear into the tensor. It supports a wider range of bit-widths and configurations.
  -  vai_q_onnx.VitisQuantFormat.FixNeuron (Experimental): This option quantizes the model by inserting FixNeuron (a combination of QuantizeLinear and DeQuantizeLinear) into the tensor. This quant format is currently experimental and cannot use for actual deployment.
* **calibrate_method**: (String) The method used in calibration, default to vai_q_onnx.PowerOfTwoMethod.MinMSE.

    For IPU_CNN platforms, power-of-two methods should be used, options are:
  -  vai_q_onnx.PowerOfTwoMethod.NonOverflow: This method get the power-of-two quantize parameters for each tensor to make sure min/max values not overflow.
  -  vai_q_onnx.PowerOfTwoMethod.MinMSE: This method get the power-of-two quantize parameters for each tensor to minimize the mean-square-loss of quantized values and float values. This takes longer time but usually gets better accuracy.

    For IPU_Transformer or CPU platforms, float scale methods should be used, options are:
  -  vai_q_onnx.CalibrationMethod.MinMax: This method obtains the quantization parameters based on the minimum and maximum values of each tensor.
  -  vai_q_onnx.CalibrationMethod.Entropy: This method determines the quantization parameters by considering the entropy algorithm of each tensor's distribution.
  -  vai_q_onnx.CalibrationMethod.Percentile: This method calculates quantization parameters using percentiles of the tensor values.
* **activation_type**: (QuantType) Specifies the quantization data type for activations.
* **weight_type**: (QuantType) Specifies the quantization data type for weights, For DPU/IPU devices, this must be set to QuantType.QInt8.
* **enable_dpu**: (Boolean) This parameter is a flag that determines whether to generate a quantized model that adapts the approximations and constraints the DPU/IPU. If set to True, the quantization process will consider the specific limitations and requirements of the DPU/IPU.
* **input_nodes**:  (List of Strings) This parameter is a list of the names of the starting nodes to be quantized. Nodes in the model before these nodes will not be quantized. For example, this argument can be used to skip some pre-processing nodes or stop the first node from being quantized. The default value is an empty list ([]).
* **output_nodes**: (List of Strings) This parameter is a list of the names of the end nodes to be quantized. Nodes in the model after these nodes will not be quantized. For example, this argument can be used to skip some post-processing nodes or stop the last node from being quantized. The default value is an empty list ([]).
* **enable_dpu**:  (Boolean) This parameter is a flag that determines whether to generate a quantized model that is suitable for the DPU/IPU. If set to True, the quantization process will consider the specific limitations and requirements of the DPU/IPU, thus creating a model that is optimized for DPU/IPU computations
* **extra_options**:  (Dictionary or None) Contains key-value pairs for various options in different cases.
  -  ActivationSymmetric: (Boolean) If True, symmetrize calibration data for activations. For DPU/IPU, this need be set to True.
  For more details of the extra_options parameters, please refer to the [extra_options](#extra_options).

### 3. Quantizing to Other Precisions

In addition to the INT8/UINT8, the VAI_Q_ONNX supports quantizing models to other data formats, including INT16/UINT16, INT32/UINT32, Float16 and BFloat16, which can provide better accuracy or be used for experimental purposes. These new data formats are achieved by a customized version of QuantizeLinear and DequantizeLinear named "VitisQuantizeLinear" and "VitisDequantizeLinear", which expands onnxruntime's UInt8 and Int8 quantization to support UInt16, Int16, UInt32, Int32, Float16 and BFloat16. This customized Q/DQ was implemented by a custom operations library in VAI_Q_ONNX using onnxruntime's custom operation C API.

The custom operations library was developed based on Linux and does not currently support compilation on Windows. If you want to run the quantized model that has the custom Q/DQ on Windows, it is recommended to switch to WSL as a workaround.

To use this feature, the "quant_format" should be set to VitisQuantFormat.QDQ. You may have noticed that in both the recommended IPU_CNN and IPU_Transformer configurations, the "quant_format" is set to QuantFormat.QDQ. IPU targets that support acceleration for models quantized to INT8/UINT8, do not support other precisions.

#### 3.1 Quantizing Float32 Models to Int16 or Int32

The quantizer supports quantizing float32 models to Int16 or Int32 data formats. To enable this, you need to set the "activation_type" and "weight_type" in the quantize_static API to the new data types. Options are VitisQuantType.QInt16/VitisQuantType.QUInt16 or VitisQuantType.QInt32/VitisQuantType.QUInt32.

```python
vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    quant_format=vai_q_onnx.VitisQuantFormat.QDQ,
    activation_type=vai_q_onnx.VitisQuantType.QInt16,
    weight_type=vai_q_onnx.VitisQuantType.QInt16,
)
```

#### 3.2 Quantizing Float32 Models to Float16 or BFloat16

Besides interger data formats, the quantizer also supports quantizing float32 models to float16 or bfloat16 data formats, just set the "activation_type" and "weight_type" to VitisQuantType.QFloat16 or VitisQuantType.QBFloat16.

```python
vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    quant_format=vai_q_onnx.VitisQuantFormat.QDQ,
    activation_type=vai_q_onnx.VitisQuantType.QFloat16,
    weight_type=vai_q_onnx.VitisQuantType.QFloat16,
)
```

#### 3.3 Quantizing Float32 Models to Mixed Data Formats

The quantizer even supports setting the activation and weight to different precisions. For example, activation is Int16 while weight is Int8. This can be used when pure Int8 quantization can not meet accuracy requirements.

```python
vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    quant_format=vai_q_onnx.VitisQuantFormat.QDQ,
    activation_type=vai_q_onnx.VitisQuantType.QInt16,
    weight_type=QuantType.QInt8,
)
```

### 4. Quantizing Float16 Models
For models in float16, we recommend setting convert_fp16_to_fp32 to True. This will first convert your float16 model to a float32 model before quantization, reducing redundant nodes such as cast in the model.

```python
vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader,
    quant_format=QuantFormat.QDQ,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    enable_dpu=True,
    convert_fp16_to_fp32=True,
    extra_options={'ActivationSymmetric':True}
)
```

### 5. Converting NCHW Models to NHWC and Quantize

NHWC input shape typically yields better acceleration performance compared to NCHW on IPU. VAI_Q_ONNX facilitates the conversion of NCHW input models to NHWC input models and do quantization by simply setting "convert_nchw_to_nhwc" to True. Please note that the convsersion steps will be skipped if the model is already NHWC or have non-convertable input shapes.

```python
vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader,
    quant_format=QuantFormat.QDQ,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    enable_dpu=True,
    extra_options={'ActivationSymmetric':True},
    convert_nchw_to_nhwc=True,
)
```

### 6. Quantizing Using CrossLayerEqualization(CLE)

CrossLayerEqualization (CLE) is a technique used to improve PTQ accuracy. It can equalize the weights of consecutive convolution layers, making the model weights easier to perform per-tensor quantization. Experiments show that using CLE technique can improve the PTQ accuracy of some models, especially for models with depthwise_conv layers, such as Mobilenet. Here is an example showing how to enable CLE using VAI_Q_ONNX.

```python
vai_q_onnx.quantize_static(
    model_input,
    model_output,
    calibration_data_reader,
    quant_format=QuantFormat.QDQ,
    calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    enable_dpu=True,
    include_cle=True,
    extra_options={
        'ActivationSymmetric':True,
        'ReplaceClip6Relu': True,
        'CLESteps': 1,
        'CLEScaleAppendBias': True,
        },
)
```

**Arguments**

* **include_cle**:  (Boolean) This parameter is a flag that determines whether to optimize the models using CrossLayerEqualization; it can improve the accuracy of some models. The default is False.

* **extra_options**:  (Dictionary or None) Contains key-value pairs for various options in different cases. Options related to CLE are:
  -  ReplaceClip6Relu: (Boolean) If True, Replace Clip(0,6) with Relu in the model. The default value is False.
  -  CLESteps: (Int): Specifies the steps for CrossLayerEqualization execution when include_cle is set to true, The default is 1, When set to -1, an adaptive CrossLayerEqualization steps will be conducted. The default value is 1.
  -  CLEScaleAppendBias: (Boolean) Whether the bias be included when calculating the scale of the weights, The default value is True.
  


### 7. Optional Steps

#### 7.1 (Optional) Pre-processing on the Float Model

Pre-processing is to transform a float model to prepare it for quantization. It consists of the following three optional steps:

* Symbolic shape inference: This is best suited for transformer models.
* Model Optimization: This step uses ONNX Runtime native library to rewrite the computation graph, including merging computation nodes, and eliminating redundancies to improve runtime efficiency.
* ONNX shape inference.

The goal of these steps is to improve quantization quality. ONNX Runtime quantization tool works best when the tensor’s shape is known. Both symbolic shape inference and ONNX shape inference help figure out tensor shapes. Symbolic shape inference works best with transformer-based models, and ONNX shape inference works with other models.

Model optimization performs certain operator fusion that makes the quantization tool’s job easier. For instance, a Convolution operator followed by BatchNormalization can be fused into one during the optimization, which can be quantized very efficiently.

Unfortunately, a known issue in ONNX Runtime is that model optimization can not output a model size greater than 2GB. So for large models, optimization must be skipped.

Pre-processing API is in the Python module onnxruntime.quantization.shape_inference, function quant_pre_process().

```python
from onnxruntime.quantization import shape_inference

shape_inference.quant_pre_process(
     input_model_path: str,
    output_model_path: str,
    skip_optimization: bool = False,
    skip_onnx_shape: bool = False,
    skip_symbolic_shape: bool = False,
    auto_merge: bool = False,
    int_max: int = 2**31 - 1,
    guess_output_rank: bool = False,
    verbose: int = 0,
    save_as_external_data: bool = False,
    all_tensors_to_one_file: bool = False,
    external_data_location: str = "./",
    external_data_size_threshold: int = 1024,)
```

**Arguments**

* **input_model_path**: (String) This parameter specifies the file path of the input model that is to be pre-processed for quantization.
* **output_model_path**: (String) This parameter specifies the file path where the pre-processed model will be saved.
* **skip_optimization**:  (Boolean) This flag indicates whether to skip the model optimization step. If set to True, model optimization will be skipped, which may cause ONNX shape inference failure for some models. The default value is False.
* **skip_onnx_shape**:  (Boolean) This flag indicates whether to skip the ONNX shape inference step. The symbolic shape inference is most effective with transformer-based models. Skipping all shape inferences may reduce the effectiveness of quantization, as a tensor with an unknown shape cannot be quantized. The default value is False.
* **skip_symbolic_shape**:  (Boolean) This flag indicates whether to skip the symbolic shape inference step. Symbolic shape inference is most effective with transformer-based models. Skipping all shape inferences may reduce the effectiveness of quantization, as a tensor with an unknown shape cannot be quantized. The default value is False.
* **auto_merge**: (Boolean) This flag determines whether to automatically merge symbolic dimensions when a conflict occurs during symbolic shape inference. The default value is False.
* **int_max**:  (Integer) This parameter specifies the maximum integer value that is to be considered as boundless for operations like slice during symbolic shape inference. The default value is 2**31 - 1.
* **guess_output_rank**: (Boolean) This flag indicates whether to guess the output rank to be the same as input 0 for unknown operations. The default value is False.
* **verbose**: (Integer) This parameter controls the level of detailed information logged during inference. A value of 0 turns off logging, 1 logs warnings, and 3 logs detailed information. The default value is 0.
* **save_as_external_data**: (Boolean) This flag determines whether to save the ONNX model to external data. The default value is False.
* **all_tensors_to_one_file**: (Boolean) This flag indicates whether to save all the external data to one file. The default value is False.
* **external_data_location**: (String) This parameter specifies the file location where the external file is saved. The default value is "./".
* **external_data_size_threshold**:  (Integer) This parameter specifies the size threshold for external data. The default value is 1024.


#### 7.2 (Optional) Evaluating the Quantized Model
If you have scripts to evaluate float models, like the models in Xilinx Model Zoo, you can replace the float model file with the quantized model for evaluation. Note that if customized Q/DQ is used in the quantized model, it is necessary to register the custom operations library to onnxruntime inference session before evaluation. For example:

```python
import onnxruntime as ort

so = ort.SessionOptions()
so.register_custom_ops_library(vai_q_onnx.get_library_path())
sess = ort.InferenceSession(quantized_model, so)
```

#### 7.3 (Optional) Dumping the Simulation Results
Sometimes after deploying the quantized model, it is necessary to compare the simulation results on the CPU/GPU and the output values on the DPU/IPU.
You can use the dump_model API of VAI_Q_ONNX to dump the simulation results with the quantized_model.
Currently, only models containing FixNeuron nodes support this feature. For models using QuantFormat.QDQ, you can set 'dump_float' to True to save float data for all nodes' results.

```python
# This function dumps the simulation results of the quantized model,
# including weights and activation results.
vai_q_onnx.dump_model(
    model,
    dump_data_reader=None,
    random_data_reader_input_shape=[],
    dump_float=False,
    output_dir='./dump_results',)
```

**Arguments**

* **model**: (String) This parameter specifies the file path of the quantized model whose simulation results are to be dumped.
* **dump_data_reader**:  (CalibrationDataReader or None) This parameter is a data reader that is used for the dumping process. The first batch will be taken as input. If you wish to use random data for a quick test, you can set dump_data_reader to None. The default value is None.
* **random_data_reader_input_shape**: (List or Tuple of Int) If dynamic axes of inputs require specific value, users should provide its shapes when using internal random data reader (That is, set dump_data_reader to None). The basic format of shape for single input is list (Int) or tuple (Int) and all dimensions should have concrete values (batch dimensions can be set to 1). For example, random_data_reader_input_shape=[1, 3, 224, 224] or random_data_reader_input_shape=(1, 3, 224, 224) for single input. If the model has multiple inputs, it can be fed in list (shape) format, where the list order is the same as the onnxruntime got inputs. For example, random_data_reader_input_shape=[[1, 1, 224, 224], [1, 2, 224, 224]] for 2 inputs. Moreover, it is possible to use dict {name : shape} to specify a certain input, for example, random_data_reader_input_shape={"image" : [1, 3, 224, 224]} for the input named "image". The default value is [].
* **dump_float**: (Boolean) This flag determines whether to dump the floating-point value of nodes' results. If set to True, the float values will be dumped. Note that this may require a lot of storage space. The default value is False.
* **output_dir**: (String) This parameter specifies the directory where the dumped simulation results will be saved. After successful execution of the function, dump results are generated in this specified directory. The default value is './dump_results'.

Note: The batch_size of the dump_data_reader will be better to set to 1 for DPU debugging.

Dump results of each FixNeuron node (including weights and activation) are generated in output_dir after the command has been successfully executed.

For each quantized node, results are saved in *.bin and *.txt formats (\* represents the output name of the node). 
If "dump_float" is set to True, output of all nodes are saved in *_float.bin and *_float.txt (\* represents the output name of the node), please note that this may require a lot of storage space.

Examples of dumping results are shown in the following table. Due to considerations for the storage path, the '/' in the node name will be replaced with '\_'.

Table 2. Example of Dumping Results

| Quantized | Node Name | Saved Weights or Activations |
| ------ | ------ | ----- |
| Yes | /conv1/Conv_output_0_DequantizeLinear | {output_dir}/dump_results/_conv1_Conv_output_0_DequantizeLinear_Output.bin <br> {output_dir}/dump_results/_conv1_Conv_output_0_DequantizeLinear_Output.txt |
| Yes | onnx::Conv_501_DequantizeLinear | {output_dir}/dump_results/onnx::Conv_501_DequantizeLinear_Output.bin <br> {output_dir}/dump_results/onnx::Conv_501_DequantizeLinear_Output.txt |
| No | /avgpool/GlobalAveragePool | {output_dir}/dump_results/_avgpool_GlobalAveragePool_output_0_float.bin <br> {output_dir}/dump_results/_avgpool_GlobalAveragePool_output_0_float.txt |



:arrow_forward:**Next Topic:**  [3. Pytorch Quantization Tutorial](./PT_README.md) 

:arrow_backward:**Previous Topic:**  [1. Introduction](../README.md)
<hr/>

