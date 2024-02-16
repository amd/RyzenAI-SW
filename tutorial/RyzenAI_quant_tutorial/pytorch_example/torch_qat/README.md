# Quantization Aware Training

Quantization Aware Training (QAT) offers higher accuracy compared to Post Training Quantization (PTQ). In Vitis AI Quantization tool, user can perform QAT under different methods through configuration files. All weights and activations are “fake quantized” during forward and backward passes of training. 

## Overview 

### Scale factor data types

- **Float**: The scale factor is a float point number. 
- **Power of two**: The scale factor is restricted to 2**n where n is an integer number,

### Scale factor updating 

- **Learnable**: The scale factor is trained in conjunction with network parameters and updated by gradients. 
- **Unlearnable**: The scale factor is updated by statistics based methods such as min-max.

### Quantization granularity

- **Per-tensor**: All elements in a tensor share the same scale factor.
- **Per-channel**: The elements of each channel (weight tensor only) share the same scale factor.

Vitis AI Quantizer provides several quantization methods. 

<table>
  <thead>
    <tr>
      <th>Scale Type</th>
      <th>Power of Two</th>
      <th>Float</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=1>Learnable scale</td>
      <td rowspan=1><a href="https://arxiv.org/abs/1903.08066">TQT</a></td>
      <td rowspan=1><a href="https://arxiv.org/abs/1902.08153">LSQ</a></td>
    </tr>
    <tr>
      <td rowspan=2>Unlearnable scale</td>
      <td>Min-max</td>
      <td>Min-max</td>
    </tr>
    <tr>
      <td>Pertentile</td>
      <td>Pertentile</td>
    </tr>
  </tbody>
</table>


In **learnable** scale factor method: This tool provides two typical algorithms, called TQT and LSQ, respectively. In **TQT**, the scale factor is in power of 2 format. In quantization schema, quantization range is signed and is symmetry and pre-tensor quantized. See [Link](https://arxiv.org/abs/1903.08066) for more. In **LSQ**,  the scale factor is in float format. In quantization schema, quantization range is fixed for  signed and is symmetry quantized. See [Link](https://arxiv.org/abs/1902.08153) for more. 

In **unlearnable** scale factor method: During training, for every quantized weight/bias/activation tensor, there is a model called observer that does the statistic. The statistical results are used to calculate the scale factor and zero-point. The scale factor is in float format.   In this mode, the quantization range, asymmetric/symmetric, narrow range, per-tensor/per-channel, and method to observer tensor should be specified. This means, that giving more flexibility needs more experience.

## Usage

1. #### Step1: Prepare floating point model

   ```python
   from torchvision.models.resnet import resnet18
   from pytorch_nndct.nn.modules.quant_stubs import QuantStub, DeQuantStub
   
   # Use QuantStub and DeQuantStub to specify quantization scope
   class ModelWithQuantStub(nn.Module):
     def __init__(self, pretrained) -> None:
       super().__init__()
       self._model = resnet18(pretrained)
       self.quant_stub = QuantStub()
       self.dequant_stub = DeQuantStub()

     def forward(self, imgs):
       imgs = self.quant_stub(imgs)
       out = self._model(imgs)
       return self.dequant_stub(out)
   ```
   
2. #### Step2: Prepare training loop
   ```python
   def train(model, train_loader, val_loader, device_ids):
     # training loop here
   ```

3. #### Step3: Init QAT processor, generate quantized model and training

   ```python
   from pytorch_nndct import QatProcessor
   
   inputs = torch.randn([batch_size, 3, 224, 224]) 
  
   # Use default quantization config
   qat_processor = QatProcessor(model, inputs)
   # Users can also specify their quantization config if they don't want to use the default one
   # qat_processor = QatProcessor(model, inputs, config_file='cfg.json')
   
   # Get trainable quantized model
   quantized_model = qat_processor.trainable_model() 
   
   # Train the quantized model 
   train(quantized_model, train_loader, val_loader, criterion,device_ids)
   
   ```

   For more configuration and quantization method, user can refer the demonstrate code in quick start.

##  Quick Start

This demonstrate code supplies the quantization for ResNet-18 and MobileNet V2. This demonstrate supplies several configurations under different quantization schemas. Including methods like TQT, LSQ, and statistic-based float/poweroftwo scale quantization. 

- Train a quantized model	

```shell
python main.py \
        --model_name=[model] \
        --pretrained=[model_weight_path]
        --mode=train \
        --config_file=[configs_file_path]
```

- Deploy a quantized model (Not all types quantized model support export deploy model, under developing )

```shell
python main.py \
    --model_name=[model] \
    --qat_ckpt=[model_weight_path]
    --mode=deploy \
    --config_file=[configs_file_path]
```

- model_name:  MobileNetV2, resnet18

- pretrained: pre-trained fp32 model file path

- qat_ckpt: quantized model weight saved after training
- config_file:  e.g: ./config_files/int_pof2_tqt.json  (Different quantization method config file can be seen in fold ***configs*** )

## Supported Quantizers

### TQT

This quantizer supports learnable scale and the scale type is power of two. The scale factor is initial by weight/activation distributions. During training, the weight/activation is fake quantized and collaborated trained with scale factor. 
**Note**: 

- Under mode power of two,  `./configs/int_pof2_tqt.json` gives an example.
- Users assign the `scale_type` to `poweroftwo` in the config file is enough, and other parameters like `symmetry`, `round_mode`, `method`, `signed`, and `narrow_range` do not need to be specified, because the TQT method's restriction.
- Refer Paper: [Trained Quantization Thresholds for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks](https://arxiv.org/abs/1903.08066)

### LSQ

This quantizer supports learnable scale and the scale type is float. The scale factor is initial by weight/activation distributions. During training, the weight/activation is fake quantized and collaborated trained with scale factor. 

- Under mode lsq, `./configs/int_float_scale_lsq.json` gives an example.

- Users need to assign the `method=lsq`, and  `scale_type=float`, and other parameters like `per_channel`, `round_mode`, and `signed` are optional to specify.

```json
"overall_quantize_config": {
  "datatype": "int",
  "bit_width": 8,
  "scale_type": "float",
  "method": "lsq",
  "calib_statistic_method": "mean"
}
    ```

- Note: `calib_statistic_method=mean` is used for avoiding code error, not effect in QAT.

- Refer Paper: [Learned Step Size Quantization](https://arxiv.org/abs/1902.08153)

### Unlearnable scale

Overview： During training in each quantizer a model called observer does the statistic for tensor(weight/bias/activation). The statistical results are used to calculate the scale factor and zero-point. The scale factor can be in float or power-of-two formats (saved in FP32). User should specify the config not but limited to `method`, `rounding_mode`, `symmetric`, `per_channel`, `unsigned`, `narrow_range`.



|            Unlearnable scale QAT             |                         value                          |
| :------------------------------------------: | :----------------------------------------------------: |
|              scale factor type               |                   float, poweroftwo                    |
| method (do statistics to compute scale & zp) |      maxmin, percentile, std + w/wo [moving_avg]       |
|                  symmetric                   |                       true/false                       |
|                 per_channel                  |       true/false (not supported for activation)        |
|          narrow_range(only signed)           |                       true/false                       |
|                rounding_mode                 | half_even, half_up, floor, std_round, stochastic_round |
|               signed/unsigned                |                       true/false                       |
|                  bit width                   |                        4, 6, 8                         |
|   ch_axis (effect under per_channel mode)    |           axis index to perform quantization           |

- ```json
  "overall_quantize_config": {
    "datatype": "int",
    "bit_width": 8,
    "round_mode": "half_even",
    "scale_type": "float",
    "calib_statistic_method":"mean"，
    "symmetry": true,
    "per_channel": false,
    "method": "maxmin",
    "signed": true,
    "narrow_range": true
  }
  ```

- For weight: Recommend to use Per-Channel, Symmetric configuration. For activation: Recommend to use Pre-Tensor, affine (asymmetric ) configuration

- Percentile method to observe Tensor is more stable in low bit quantization.

- Float Scale config:  `./configs/int_float_scale_min_max.json` and `./configs/int_float_scale_percentile.json` give examples.

- Power of Two scale:   `./configs/int_pof2_percentile.json` and `./configs/int_pof2_std.json` give examples.

### Note：

1. For fine-grained quantization, users can assign different configs in the config file for `tensor` and `layers`, respectively.
2. Training you own model, look at the code in network.py, your model code may need modification;
3. Load pre-trained weight state_dict for better Qat performance.

# Auto module

## Overview 

In QAT, tensors are quantized by manipulation of torch.nn.Module objects. For parameter quantization, quantizer replaces modules with parameters with quantized version of the modules(for instance, replacing nn.Conv2d with QuantizedConv2d and replacing nn.Linear with QuantizedLinear), in which quantizers for parameters are inserted. For input and output quantization, the quantizer adds quantization step in module's forward_hook and forward_pre_hook. But for non-module operations, the quantizer cannot directly modify their behavior. For example, if we have a model: 

```python
import torch
from torch import nn

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
  
  def forward(self, x, y, z):
    tmp = x + y
    ret = tmp * z
    return ret
```

How do we quantize intputs and outputs of operator "+" and "*" in this model? One way is to replace the native operators with modules mannually like below.

```python
import torch
from torch import nn

class Add(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x, y):
    return x + y

class Mul(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x, y):
    return x * y

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.add = Add()
    self.Mul = Mul()

  def forward(self, x, y, z):
    tmp = self.add(x, y)
    ret = self.mul(tmp, z)
    return ret
```

After the replacement, QAT can insert quantizers for tensor x, y, tmp and z via forward_hook and forward_pre_hook.

QAT now provides a tool called **auto_module**, which could be used to replace the non-module operators with modules automatically.

## Usage

Users can use auto_module by a simple function call. Here's an example.

```python
import torch
from torch import nn
from pytorch_nndct.quantization.auto_module import wrap

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
  
  def forward(self, x, y, z):
    tmp = x + y
    ret = tmp * z
    return ret

model = MyModel()
x = torch.rand((2, 2))
y = torch.rand((2, 2))
z = torch.rand((2, 2))
wrapped_model = wrap(model, x, y, z)
```
`wrapped_model` is the model with all native operators("+" and "*" in this example) replaced with modules, which can be processed by QatProcessor like normal models.

The signature of function wrap is as follows.

```python
def wrap(model: nn.Module, *args, **kwargs) -> nn.Module
  """
  Args:
    model (nn.Module): The model to be processed by auto_module
    args, kwargs: The inputs of model. There is model inference in this function. 
      `args` and `kwargs` will be fed to model directly during model inference.
  """
```
