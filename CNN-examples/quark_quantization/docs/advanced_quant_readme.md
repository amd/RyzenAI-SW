<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Advanced Quantization Tools </h1>
    </td>
 </tr>
</table>

## Advanced Quantization Tools

In this section, we explore the advanced quantization capabilities of the Quark quantizer, designed to recover the lost accuracy in quantized models. While basic quantization configurations are effective for many models, advanced and optimized models often require sophisticated techniques to enhance the accuracy of the quantized versions. This guide will walk you through these advanced methods, ensuring your models maintain high performance even after quantization.

This tutorial uses the [MobileNetV3 Large](https://pytorch.org/vision/stable/models/mobilenetv3.html) model as an example, which can be challenging to quantize with minimal accuracy loss. We demonstrate how advanced Quark quantization techniques can recover accuracy compared to basic quantization.

ImageNet Dataset
----------------

Please ensure to setup the validation and calibration datase using the instruction from [AMD Quark Quantization Tutorial](quark_quant_readme.md)


Model Evaluation
----------------

Download the MobileNetV3 Large model:

```bash
cd models
python download_MobileNetV3.py
```

### Basic A8W8 Quantization

Using the basic ``A8W8`` configuration without advanced techniques, we observe significant accuracy degradation. Optimized models like MobileNetV3 are particularly challenging to quantize. To bridge the gap between float and quantized accuracy, we can use advanced quantization techniques.

```python
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
```

```bash
python advanced_quark_quantize.py --model_input models/mobilenetv3_large.onnx --model_output models/mobilenetv3_large_a8w8.onnx
```

<div align="center">

| MobileNetV3 Large   | Model Size | Top-1 Accuracy | Top-5 Accuracy |
|---------------------|------------|----------------|----------------|
| Float 32            |  20.91 MB  | 77.0%          | 95.0%          |
| A8W8 (CPU)          |   5.50 MB  | 15.0%          | 36.0%          |
| A8W8 (NPU)          |   5.50 MB  | 12.0%          | 33.0%          |

</div>

> **Note:** Basic quantization results in severe accuracy loss (~57% drop), demonstrating the need for advanced techniques.

### ADAROUND (Adaptive Rounding)

**ADAROUND** (Adaptive Rounding) is a quantization algorithm that optimizes the rounding of weights by minimizing the reconstruction error layer-by-layer. Instead of using traditional rounding methods, ADAROUND learns optimal rounding decisions for each weight, ensuring better accuracy retention in post-training quantization.

**Key Features:**
- Layer-wise weight rounding optimization
- Minimizes reconstruction error between float and quantized outputs
- Fast convergence with early stopping
- Improves accuracy over naive quantization

```python
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
```

Apply ADAROUND to MobileNetV3 Large:

```bash
python advanced_quark_quantize.py --model_input models/mobilenetv3_large.onnx --model_output models/mobilenetv3_large_a8w8_adaround.onnx --adaround
```

<div align="center">

| MobileNetV3 Large        | Model Size | Top-1 Accuracy | Top-5 Accuracy | Improvement |
|--------------------------|------------|----------------|----------------|-------------|
| Float 32                 |  20.91 MB  | 77.0%          | 95.0%          | -           |
| A8W8 (Basic)             |   5.50 MB  | 15.0%          | 36.0%          | Baseline    |
| **A8W8 + ADAROUND (CPU)**|   5.81 MB  | **70.0%**      | **91.0%**      | **+55.0%**  |
| **A8W8 + ADAROUND (NPU)**|   5.81 MB  | **69.0%**      | **89.0%**      | **+57.0%**  |

</div>

> **Result:** ADAROUND recovers ~95% of the original float32 accuracy (only 4.8% degradation vs 57.5% with basic quantization).

### ADAQUANT (Adaptive Quantization)

**ADAQUANT** (Adaptive Quantization) is a post-training quantization algorithm that jointly optimizes both quantization parameters and weight values by minimizing layer-wise reconstruction errors. Unlike ADAROUND which only optimizes rounding, ADAQUANT adapts both the quantization scales and weight values.

**Key Features:**
- Joint optimization of quantization parameters and weights
- Layer-wise reconstruction error minimization
- Adapts to activation distributions during calibration
- More comprehensive than rounding-only methods

```python
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
```

Apply ADAQUANT to MobileNetV3 Large:

```bash
python advanced_quark_quantize.py --model_input models/mobilenetv3_large.onnx --model_output models/mobilenetv3_large_a8w8_adaquant.onnx --adaquant
```

<div align="center">

| MobileNetV3 Large        | Model Size | Top-1 Accuracy | Top-5 Accuracy | Improvement |
|--------------------------|------------|----------------|----------------|-------------|
| Float 32                 |  20.91 MB  | 77.0%          | 95.0%          | -           |
| A8W8 (Basic)             |   5.50 MB  | 15.0%          | 36.0%          | Baseline    |
| **A8W8 + ADAQUANT (CPU)**|   5.50 MB  | **52.0%**      | **83.0%**      | **+37.0%**  |
| **A8W8 + ADAQUANT (NPU)**|   5.50 MB  | **51.0%**      | **84.0%**      | **+39.0%**  |

</div>

> **Result:** ADAQUANT provides substantial improvement over basic quantization but performs less effectively than ADAROUND for this model (~70% accuracy recovery vs 95% with ADAROUND).


### Cross Layer Equalization (CLE)

**Cross-Layer Equalization (CLE)** optimizes neural networks for quantization by balancing weight distributions across consecutive layers. It equalizes the ranges of weights between layers, reducing quantization errors caused by imbalanced distributions.

**Key Features:**
- Pre-processing technique applied before quantization
- Balances weight ranges across consecutive convolution/linear layers
- Works by redistributing quantization difficulty across layers
- Can be combined with other techniques like ADAROUND or ADAQUANT
- Particularly effective for models with highly variable weight distributions

**Implementation:**

```python
quant_config = A8W8_CONFIG  # Start with base configuration
quant_config.enable_npu_cnn = True
# Enable cross-layer equalization
quant_config.include_cle = True
```

Apply Cross Layer Equalization to MobileNetV3 Large:

```bash
python advanced_quark_quantize.py --model_input models/mobilenetv3_large.onnx --model_output models/mobilenetv3_large_a8w8_cle.onnx --cross_layer_equalization
```

<div align="center">

| MobileNetV3 Large        | Model Size | Top-1 Accuracy | Top-5 Accuracy |
|--------------------------|------------|----------------|----------------|
| Float 32                 |  20.91 MB  | 77.0%          | 95.0%          |
| A8W8 (Basic)             |   5.50 MB  | 15.0%          | 36.0%          |
| **A8W8 + CLE (CPU)**     |   5.81 MB  | **15.0%**      | **36.0%**      |
| **A8W8 + CLE (NPU)**     |   5.81 MB  | **12.0%**      | **33.0%**      |

</div>

> **Result:** CLE provides no improvement for MobileNetV3 when used alone. It's most effective when combined with other techniques like ADAROUND.

## Technique Comparison Summary

<div align="center">

| Technique           | Top-1 Accuracy (NPU) | Accuracy Recovery | Quantization Time | Best Use Case |
|---------------------|----------------------|-------------------|-------------------|---------------|
| Basic A8W8          | 13.12%              | Baseline          | ~4 seconds        | Quick prototyping |
| **ADAROUND**        | **69.24%**          | **95% recovery**  | ~6 minutes        | **Production (recommended)** |
| ADAQUANT            | 50.84%              | 70% recovery      | ~6 minutes        | Alternative to ADAROUND |
| CLE (alone)         | 14.68%              | Minimal           | ~4 seconds        | Combine with other methods |

</div>

## Recommendations

Based on the MobileNetV3 Large evaluation:

1. **For Production Deployment:** Use **ADAROUND** - it provides the best accuracy recovery (95% of float32 performance) with reasonable quantization time.

2. **For Experimentation:** Try ADAQUANT if ADAROUND doesn't perform well on your specific model architecture.

3. **Cross Layer Equalization:** While CLE alone shows minimal improvement for MobileNetV3, it can be beneficial when combined with ADAROUND or ADAQUANT for models with highly imbalanced weight distributions.

4. **Model-Specific Tuning:** Different models respond differently to quantization techniques. Always evaluate multiple methods on your target model.


Reference
---------

For more details on the Quark API features, see the [Quark Documentation](https://quark.docs.amd.com/latest/index.html)
