Quantization Strategies
=======================

AMD Quark for Pytorch offers three distinct quantization strategies tailored to meet the requirements of various hardware backends:

-  **Post Training Weight-Only Quantization**: The weights are quantized ahead of time, but the activations are not quantized (using the original float data type) during inference.

-  **Post Training Static Quantization**: Quantizes both the weights and activations in the model. To achieve the best results, this process necessitates calibration with a dataset that accurately represents the actual data, which allows for precise determination of the optimal quantization parameters for activations.

- **Post Training Dynamic Quantization**: Quantizes the weights ahead of time, while the activations are quantized dynamically at runtime. This method allows for a more flexible approach, especially when the activation distribution is not well-known or varies significantly during inference.

Here is one sample example for different quant strategies:

.. code:: python

   # 1. Set model
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
   model.eval()
   tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

   # 2. Set quantization configuration
   from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
   from quark.torch.quantization.config.config import Config, QuantizationSpec, QuantizationConfig
   from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver

   # 2-1. For weight only quantization, please uncomment the following lines.
   DEFAULT_UINT4_PER_GROUP_ASYM_SPEC = QuantizationSpec(dtype=Dtype.uint4,
                                                       observer_cls=PerChannelMinMaxObserver,
                                                       symmetric=False,
                                                       scale_type=ScaleType.float,
                                                       round_method=RoundType.half_even,
                                                       qscheme=QSchemeType.per_group,
                                                       ch_axis=0,
                                                       is_dynamic=False,
                                                       group_size=32)
   DEFAULT_W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=DEFAULT_UINT4_PER_GROUP_ASYM_SPEC)
   quant_config = Config(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG)

   # 2-2. For dynamic quantization, please uncomment the following lines.
   # INT8_PER_TENSER_DYNAMIC_SPEC = QuantizationSpec(dtype=Dtype.int8,
   #                                                 qscheme=QSchemeType.per_tensor,
   #                                                 observer_cls=PerTensorMinMaxObserver,
   #                                                 symmetric=True,
   #                                                 scale_type=ScaleType.float,
   #                                                 round_method=RoundType.half_even,
   #                                                 is_dynamic=True)
   # DEFAULT_W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSER_DYNAMIC_SPEC,
   #                                                                      weight=INT8_PER_TENSER_DYNAMIC_SPEC)
   # quant_config = Config(global_quant_config=DEFAULT_W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG)

   # 2-3. For static quantization , please uncomment the following lines.
   # FP8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.fp8_e4m3,
   #                                        qscheme=QSchemeType.per_tensor,
   #                                        observer_cls=PerTensorMinMaxObserver,
   #                                        is_dynamic=False)
   # DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
   #                                                            weight=FP8_PER_TENSOR_SPEC)
   # quant_config = Config(global_quant_config=DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG)

   # 3. Define calibration dataloader (still need this step for weight only and dynamic quantization)
   from torch.utils.data import DataLoader
   text = "Hello, how are you?"
   tokenized_outputs = tokenizer(text, return_tensors="pt")
   calib_dataloader = DataLoader(tokenized_outputs['input_ids'])

   # 4. In-place replacement with quantized modules in model
   from quark.torch import ModelQuantizer
   quantizer = ModelQuantizer(quant_config)
   quant_model = quantizer.quantize_model(model, calib_dataloader)

The strategies share the same user API.
You simply need to set the strategy through the quantization configuration, as demonstrated in the previous example.
For more details about setting quantization configuration, refer to the "Configuring AMD Quark for PyTorch" chapter.
