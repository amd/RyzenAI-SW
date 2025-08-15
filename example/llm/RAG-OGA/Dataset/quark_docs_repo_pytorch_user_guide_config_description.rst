Configuring PyTorch Quantization
================================

This topic describes the steps on how to set the quantization configuration in AMD Quark for PyTorch.

Configuration of quantization in ``AMD Quark for PyTorch`` is set using Python ``dataclass`` because it is rigorous and helps you avoid typos. The class ``Config`` in ``quark.torch.quantization.config.config`` is provided for configuration. There are several steps to set up the configuration:

- **Step 1**: Configure :py:class:`.QuantizationSpec` for ``torch.Tensors``. Specify attributes such as ``dtype``, ``observer_cls``, etc.
- **Step 2**: Establish ``QuantizationConfig`` for ``nn.Module``. Define the ``QuantizationSpec`` of ``input_tensors``, ``output_tensors``, ``weight``, and ``bias``.
- **Step 3** [Optional]: Set ``AlgoConfig`` for the model.
- **Step 4**: Set up the overall ``Config`` for the model. This includes:


.. toctree::
  :hidden:
  :maxdepth: 1

  Calibration Methods <calibration_methods.rst>
  Calibration Datasets <calibration_datasets.rst>
  Quantization Strategies <quantization_strategies.rst>
  Quantization Schemes <quantization_schemes.rst>
  Quantization Symmetry <quantization_symmetry.rst>

Step 1: Configuring ``QuantizationSpec`` for torch.Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The class :py:class:`.QuantizationSpec` aims to describe the quantization specification for each tensor, including dtype, observer_cls, qscheme, is_dynamic, symmetric, etc. For example:

.. code-block:: python

   from quark.torch.quantization.config.config import QuantizationSpec
   from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType
   from quark.torch.quantization.observer.observer import PlaceholderObserver, PerTensorMinMaxObserver, PerGroupMinMaxObserver

   BFLOAT16_SPEC = QuantizationSpec(dtype=Dtype.bfloat16, observer_cls=PlaceholderObserver)

   FP8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.fp8_e4m3,
                                          qscheme=QSchemeType.per_tensor,
                                          observer_cls=PerTensorMinMaxObserver,
                                          is_dynamic=False)

   INT8_PER_TENSOR_SPEC = Int8PerTensorSpec(observer_method="min_max",
                                           symmetric=True,
                                           scale_type=ScaleType.float,
                                           round_method=RoundType.half_even,
                                           is_dynamic=False).to_quantization_spec()

   UINT4_PER_GROUP_ASYM_SPEC = QuantizationSpec(dtype=Dtype.uint4,
                                                observer_cls=PerGroupMinMaxObserver,
                                                symmetric=False,
                                                scale_type=ScaleType.float,
                                                round_method=RoundType.half_even,
                                                qscheme=QSchemeType.per_group,
                                                ch_axis=1,
                                                is_dynamic=False,
                                                group_size=128)

Details about each parameters of :py:class:`.QuantizationSpec` as well as of each utility classes (as :py:class:`.Int8PerTensorSpec` to define more easily the quantization spec) are available in :doc:`the API documentation <../autoapi/quark/torch/quantization/config/config/index>`.

Step 2: Establishing ``QuantizationConfig`` for nn.Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The class :py:class:`.QuantizationConfig` is used to describe the global, layer-type-wise, or layer-wise quantization information for each ``nn.Module``, such as ``nn.Linear``. For example:

.. code-block:: python

   from quark.torch.quantization.config.config import QuantizationConfig

   W_FP8_A_FP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                                      weight=FP8_PER_TENSOR_SPEC)

   W_INT8_A_INT8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC,
                                                        weight=INT8_PER_TENSOR_SPEC)

   W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)

Details about each parameters of :py:class:`.QuantizationConfig` are available in :doc:`the API documentation <../autoapi/quark/torch/quantization/config/config/index>`.

Step 3: [Optional] Setting ``AlgoConfig`` for the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use AMD Quark's advanced algorithms such as AWQ, you should set up the required configuration.

You should possess a thorough understanding of the methods and hyperparameters associated with the algorithms before configuring them. Algorithms only support certain ``QuantizationSpec``. Ensure compatibility before running.

Here is the algorithms configuration of Llama2-7b as an example:

.. code-block:: python

   from quark.torch.algorithm.awq.awq import AwqProcessor
   from quark.torch.algorithm.awq.smooth import SmoothQuantProcessor
   from quark.torch.algorithm.gptq.gptq import GptqProcessor
   from quark.torch.quantization.config.config import AWQConfig, SmoothQuantConfig, GPTQConfig

   ALGORITHM_CONFIG=AWQConfig(
     scaling_layers=[
       {'prev_op': 'input_layernorm', 'layers': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'], 'inp': 'self_attn.q_proj', 'module2inspect': 'self_attn'},
       {'prev_op': 'self_attn.v_proj', 'layers': ['self_attn.o_proj'], 'inp': 'self_attn.o_proj'},
       {'prev_op': 'post_attention_layernorm', 'layers': ['mlp.gate_proj', 'mlp.up_proj'], 'inp': 'mlp.gate_proj', 'module2inspect': 'mlp', 'help': 'linear 1'},
       {'prev_op': 'mlp.up_proj', 'layers': ['mlp.down_proj'], 'inp': 'mlp.down_proj',  'help': 'linear 2'}],
     model_decoder_layers='model.layers')

   ALGORITHM_CONFIG=SmoothQuantConfig(
     alpha=0.5,
     scale_clamp_min=0.001,
     scaling_layers=[
       {'prev_op': 'input_layernorm', 'layers': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'], 'inp': 'self_attn.q_proj', 'module2inspect': 'self_attn'},
       {'prev_op': 'self_attn.v_proj', 'layers': ['self_attn.o_proj'], 'inp': 'self_attn.o_proj'},
       {'prev_op': 'post_attention_layernorm', 'layers': ['mlp.gate_proj', 'mlp.up_proj'], 'inp': 'mlp.gate_proj', 'module2inspect': 'mlp', 'help': 'linear 1'},
       {'prev_op': 'mlp.up_proj', 'layers': ['mlp.down_proj'], 'inp': 'mlp.down_proj',   'help': 'linear 2'}],
     model_decoder_layers='model.layers')

   ALGORITHM_CONFIG = GPTQConfig(
       damp_percent=0.01,
       desc_act=True,
       static_groups=True,
       true_sequential=True,
       inside_layer_modules=['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj'],
       model_decoder_layers='model.layers'
   )

   ALGORITHM_CONFIG = RotationConfig(
       scaling_layers = {
           "first_layer": [
               {"prev_modules": ["model.embed_tokens"],
                "norm_module": "model.layers.layer_id.input_layernorm",
                "next_modules": ["model.layers.layer_id.self_attn.q_proj", "model.layers.layer_id.self_attn.k_proj", "model.layers.layer_id.self_attn.v_proj"]},
               {"prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
                "norm_module": "model.layers.layer_id.post_attention_layernorm",
                "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"]}],
           "middle_layers": [
               {"prev_modules": ["model.layers.pre_layer_id.mlp.down_proj"],
                "norm_module": "model.layers.layer_id.input_layernorm",
                "next_modules": ["model.layers.layer_id.self_attn.q_proj", "model.layers.layer_id.self_attn.k_proj", "model.layers.layer_id.self_attn.v_proj"]},
               {"prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
                "norm_module": "model.layers.layer_id.post_attention_layernorm",
                "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"]}],
           "last_layer": [
               {"prev_modules": ["model.layers.layer_id.mlp.down_proj"],
                "norm_module": "model.norm",
                "next_modules": ["lm_head"]}]
       }
   )

   ALGORITHM_CONFIG = QuaRotConfig(
       scaling_layers = {
           "first_layer": [
               {"prev_modules": ["model.embed_tokens"],
                "norm_module": "model.layers.layer_id.input_layernorm",
                "next_modules": ["model.layers.layer_id.self_attn.q_proj", "model.layers.layer_id.self_attn.k_proj", "model.layers.layer_id.self_attn.v_proj"]},
               {"prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
                "norm_module": "model.layers.layer_id.post_attention_layernorm",
                "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"]}],
           "middle_layers": [
               {"prev_modules": ["model.layers.pre_layer_id.mlp.down_proj"],
                "norm_module": "model.layers.layer_id.input_layernorm",
                "next_modules": ["model.layers.layer_id.self_attn.q_proj", "model.layers.layer_id.self_attn.k_proj", "model.layers.layer_id.self_attn.v_proj"]},
               {"prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
                "norm_module": "model.layers.layer_id.post_attention_layernorm",
                "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"]}],
           "last_layer": [
               {"prev_modules": ["model.layers.layer_id.mlp.down_proj"],
                "norm_module": "model.norm",
                "next_modules": ["lm_head"]}]
       }
   )

For AWQ, AMD Quark for PyTorch only supports ``AWQ`` with quantization data type as ``uint4/int4`` and ``per group``, running on ``Linux`` with the ``GPU mode`` for now. More details are available in the :py:class:`.AWQConfig` documentation.


For SmoothQuant, more details are available in the :py:class:`.SmoothQuantConfig` documentation. A high-level explanation about SmoothQuant is available in :doc:`Activation/weight smoothing (SmoothQuant) documentation <smoothquant>`.

For GPTQ, AMD Quark for PyTorch only supports ``GPTQ`` with quantization
data type as ``uint4/int4`` and ``per group``, running on ``Linux`` with
the ``GPU mode`` for now. More details are available in the :py:class:`.GPTQConfig` documentation.


Step 4: Setting up the overall ``Config`` for the model.
--------------------------------------------------------

In :py:class:`.quark.torch.quantization.config.config.Config`, you should set instances for all information of quantization (all instances are optional except ``global_quant_config``).

For example:

.. code-block:: python

   # Example 1: W_INT8_A_INT8_PER_TENSOR
   quant_config = Config(global_quant_config=W_INT8_A_INT8_PER_TENSOR_CONFIG)

   # Example 2: W_UINT4_PER_GROUP with advanced algorithm
   quant_config = Config(global_quant_config=W_UINT4_PER_GROUP_CONFIG, algo_config=ALGORITHM_CONFIG)
   EXCLUDE_LAYERS = ["lm_head"] # For language models
   quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)

   # Example 3: W_FP8_A_FP8_PER_TENSOR with KV_CACHE_FP8
   quant_config = Config(global_quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG)
   KV_CACHE_CFG = {
       "*v_proj":
       QuantizationConfig(input_tensors=quant_config.global_quant_config.input_tensors,
                          weight=quant_config.global_quant_config.weight,
                          output_tensors=FP8_PER_TENSOR_SPEC),
       "*k_proj":
       QuantizationConfig(input_tensors=quant_config.global_quant_config.input_tensors,
                          weight=quant_config.global_quant_config.weight,
                          output_tensors=FP8_PER_TENSOR_SPEC),
   }
   quant_config = replace(quant_config, layer_quant_config=KV_CACHE_CFG)

More details are available in the :py:class:`.quark.torch.quantization.config.config.Config` documentation.