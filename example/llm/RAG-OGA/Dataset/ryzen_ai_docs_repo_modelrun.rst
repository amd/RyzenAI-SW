.. include:: /icons.txt

################################
Model Compilation and Deployment
################################

*****************
Introduction
*****************

The Ryzen AI Software supports compiling and deploying quantized model saved in the ONNX format. The ONNX graph is automatically partitioned into multiple subgraphs by the VitisAI Execution Provider (EP). The subgraph(s) containing operators supported by the NPU are executed on the NPU. The remaining subgraph(s) are executed on the CPU. This graph partitioning and deployment technique across CPU and NPU is fully automated by the VAI EP and is totally transparent to the end-user.

|memo| **NOTE**: Models with ONNX opset 17 are recommended. If your model uses a different opset version, consider converting it using the `ONNX Version Converter <https://github.com/onnx/onnx/blob/main/docs/VersionConverter.md>`_

Models are compiled for the NPU by creating an ONNX inference session using the Vitis AI Execution Provider (VAI EP):

.. code-block:: python

    providers = ['VitisAIExecutionProvider']
    session = ort.InferenceSession(
        model,
        sess_options = sess_opt,
        providers = providers,
        provider_options = provider_options
    )


The ``provider_options`` parameter allows passing special options to the Vitis AI EP.

.. list-table::
   :widths: 20 35
   :header-rows: 1

   * - Provider Options
     - Description
   * - config_file
     - Configuration file to pass certain compile-specific options, used for BF16 compilation.
   * - xclbin
     - NPU binary file to specify NPU configuration, used for INT8 models.
   * - cache_dir
     - The path and name of the cache directory.
   * - cache_key
     - The subfolder in the cache directory where the compiled model is stored.
   * - encryptionKey
     - Used for generating an encrypted compiled model.

Detailed usage of these options is discussed in the following sections of this page.


.. _compile-bf16:

**************************
Compiling BF16 models
**************************

|memo| **NOTE**: For compiling large BF16 models a machine with at least 32GB of memory is recommended. The machine does not need to have an NPU. It is also possible to compile BF16 models on a Linux workstation. More details can be found here: :doc:`rai_linux`

When compiling BF16 models, a compilation configuration file must be provided through the ``config_file`` provider options.

.. code-block:: python

    providers = ['VitisAIExecutionProvider']

    provider_options = [{
        'config_file': 'vai_ep_config.json'
    }]

    session = ort.InferenceSession(
        "resnet50.onnx",
        providers=providers,
        provider_options=provider_options
    )


By default, the configuration file for compiling BF16 models should contain the following:

.. code-block:: json

   {
    "passes": [
        {
            "name": "init",
            "plugin": "vaip-pass_init"
        },
        {
            "name": "vaiml_partition",
            "plugin": "vaip-pass_vaiml_partition",
            "vaiml_config": {}
        }
    ]
   }


Additional options can be specified in the ``vaiml_config`` section of the configuration file, as described below.

**Performance Optimization**

The default compilation optimization level is 1. The optimization level can be changed as follows:

.. code-block:: json

    "vaiml_config": {"optimize_level": 2}

Supported values: 1 (default), 2


**Automatic FP32 to BF16 Conversion**

If a FP32 model is used, the compiler will automatically cast it to BF16 if this option is enabled. For better control over accuracy, it is recommended to quantize the model to BF16 using Quark.

.. code-block:: json

    "vaiml_config": {"enable_f32_to_bf16_conversion": true}

Supported values: false (default), true


**Optimizations for Transformer-Based Models**

By default, the compiler vectorizes the data to optimize performance for CNN models. However, transformers perform best with unvectorized data. To better optimize transformer-based models, set:

.. code-block:: json

    "vaiml_config": {"preferred_data_storage": "unvectorized"}

Supported values: "vectorized" (default), "unvectorized"


.. _compile-int8:

**************************
Compiling INT8 models
**************************

When compiling INT8 models, the NPU configuration must be specified through the ``xclbin`` provider option. This option is not required for BF16 models.

There are two types of NPU configurations for INT8 models: standard and benchmark. Setting the NPU configuration involves specifying a specific ``.xclbin`` binary file, which is located in the Ryzen AI Software installation tree.

Depending on the target processor and binary type (standard/benchmark), the following ``.xclbin`` files should be used:

**For STX/KRK APUs**:

- Standard binary: ``%RYZEN_AI_INSTALLATION_PATH%\voe-4.0-win_amd64\xclbins\strix\AMD_AIE2P_Nx4_Overlay.xclbin``
- Benchmark binary: ``%RYZEN_AI_INSTALLATION_PATH%\voe-4.0-win_amd64\xclbins\strix\AMD_AIE2P_4x4_Overlay.xclbin``

**For PHX/HPT APUs**:

- Standard binary: ``%RYZEN_AI_INSTALLATION_PATH%\voe-4.0-win_amd64\xclbins\phoenix\1x4.xclbin``
- Benchmark binary: ``%RYZEN_AI_INSTALLATION_PATH%\voe-4.0-win_amd64\xclbins\phoenix\4x4.xclbin``

Python example selecting the standard NPU configuration for STX/KRK:

.. code-block:: python

    providers = ['VitisAIExecutionProvider']

    provider_options = [{
        'xclbin': '{}\\voe-4.0-win_amd64\\xclbins\\strix\\AMD_AIE2P_Nx4_Overlay.xclbin'.format(os.environ["RYZEN_AI_INSTALLATION_PATH"])
    }]

    session = ort.InferenceSession(
        "resnet50.onnx",
        providers=providers,
        provider_options=provider_options
    )

|

By default, the Ryzen AI Conda environment automatically sets the standard binary for all inference sessions through the ``XLNX_VART_FIRMWARE`` environment variable. However, explicitly passing the xclbin option in the provider options overrides the environment variable.

.. code-block::

    > echo %XLNX_VART_FIRMWARE%
      C:\Program Files\RyzenAI\1.4.0\voe-4.0-win_amd64\xclbins\strix\AMD_AIE2P_Nx4_Overlay.xclbin



|

************************************
Managing Compiled Models
************************************

To avoid the overhead of recompiling models, it is very advantageous to save the compiled models and use these pre-compiled versions in the final application. Pre-compiled models can be loaded instantaneously and immediately executed on the NPU. This greatly improves the session creation time and overall end-user experience.

The RyzenAI Software supports two mechanisms for saving and reloading compiled models:

- VitisAI EP Cache
- OnnxRuntime EP Context Cache

.. _vitisai-ep-cache:

VitisAI EP Cache
================

The VitisAI EP includes a built-in caching mechanism. This mechanism is enabled by default. When a model is compiled for the first time, it is automatically saved in the VitisAI EP cache directory. Any subsequent creation of an ONNX Runtime session using the same model will load the precompiled model from the cache directory, thereby reducing session creation time.

The location of the VitisAI EP cache is specified with the ``cache_dir`` and ``cache_key`` provider options:

- ``cache_dir`` - Specifies the path and name of the cache directory.
- ``cache_key`` - Specifies the subfolder in the cache directory where the compiled model is stored.

Python example:

.. code-block:: python

    from pathlib import Path

    providers = ['VitisAIExecutionProvider']
    cache_dir = Path(__file__).parent.resolve()
    provider_options = [{'cache_dir': str(cache_dir),
                        'cache_key': 'compiled_resnet50'}]

    session = ort.InferenceSession(
        "resnet50.onnx",
        providers=providers,
        provider_options=provider_options
    )


In the example above, the cache directory is set to the absolute path of the folder containing the script being executed. Once the session is created, the compiled model is saved inside a subdirectory named ``compiled_resnet50`` within the specified cache folder.

Default Settings
----------------
In the current release, if ``cache_dir`` is not set, the default cache location is determined by the type of model:

- INT8 models - ``C:\temp\%USERNAME%\vaip\.cache``
- BF16 models - The directory where the script or program is executed


Disabling the Cache
-------------------
To ignore cached models and force recompilation, unset the ``XLNX_ENABLE_CACHE`` environment variable before running the application:

.. code-block::

    set XLNX_ENABLE_CACHE=



VitisAI EP Cache Encryption
---------------------------

The contents of the VitisAI EP cache folder can be encrypted using AES256. Cache encryption is enabled by passing an encryption key through the VAI EP provider options. The same key must be used to decrypt the model when loading it from the cache. The key is a 256-bit value represented as a 64-digit string.

Python example:

.. code-block:: python

    session = onnxruntime.InferenceSession(
        "resnet50.onnx",
        providers=["VitisAIExecutionProvider"],
        provider_options=[{
            "config_file":"/path/to/vaip_config.json",
            "encryptionKey": "89703f950ed9f738d956f6769d7e45a385d3c988ca753838b5afbc569ebf35b2"
    }])

C++ example:

.. code-block:: cpp

    auto onnx_model_path = "resnet50.onnx"
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "resnet50");
    auto session_options = Ort::SessionOptions();
    auto options = std::unorderd_map<std::string,std::string>({});
    options["config_file"] = "/path/to/vaip_config.json";
    options["encryptionKey"] = "89703f950ed9f738d956f6769d7e45a385d3c988ca753838b5afbc569ebf35b2";

    session_options.AppendExecutionProvider("VitisAI", options);
    auto session = Ort::Experimental::Session(env, model_name, session_options);

As a result of encryption, the model generated in the cache directory cannot be opened with Netron. Additionally, dumping is disabled to prevent the leakage of sensitive information about the model.

.. _ort-ep-context-cache:

OnnxRuntime EP Context Cache
============================

The Vitis AI EP supports the ONNX Runtime EP context cache feature. This features allows dumping and reloading a snapshot of the EP context before deployment. Currently, this feature is only available for INT8 models.

The user can enable dumping of the EP context by setting the ``ep.context_enable`` session option to 1.

The following options can be used for additional control:

- ``ep.context_file_path`` – Specifies the output path for the dumped context model.
- ``ep.context_embed_mode`` – Embeds the EP context into the ONNX model when set to 1.

For further details, refer to the official ONNX Runtime documentation: https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html


EP Context Encryption
---------------------

By default, the generated context model is unencrypted and can be used directly during inference. If needed, the context model can be encrypted using one of the methods described below. 

User-managed encryption
~~~~~~~~~~~~~~~~~~~~~~~
After the context model is generated, the developer can encrypt the generated file using a method of choice. At runtime, the encrypted file can be loaded by the application, decrypted in memory and passed as a serialized string to the inference session. This method gives complete control to the developer over the encryption process.

EP-managed encryption
~~~~~~~~~~~~~~~~~~~~~~~
The Vitis AI EP encryption mechanism can be used to encrypt the context model. This is enabled by passing an encryption key via the ``encryptionKey`` provider option (discussed in the previous section). The model is encrypted using AES256. At runtime, the same encryption key must be provided to decrypt and load the context model. With this method, encryption and decryption is seamlessly managed by the VitisAI EP.

Python example:

.. code-block:: python

    # Compilation session
    session_options = ort.SessionOptions()
    session_options.add_session_config_entry('ep.context_enable', '1')
    session_options.add_session_config_entry('ep.context_file_path', 'context_model.onnx')
    session_options.add_session_config_entry('ep.context_embed_mode', '1')
    session = ort.InferenceSession(
        path_or_bytes='resnet50.onnx',
        sess_options=session_options,
        providers=['VitisAIExecutionProvider'],
        provider_options=[{'encryptionKey': '89703f950ed9f738d956f6769d7e45a385d3c988ca753838b5afbc569ebf35b2'}]
    )

    # Inference session
    session_options = ort.SessionOptions()
    session = ort.InferenceSession(
        path_or_bytes='context_model.onnx',
        sess_options=session_options,
        providers=['VitisAIExecutionProvider'],
        provider_options=[{'encryptionKey': '89703f950ed9f738d956f6769d7e45a385d3c988ca753838b5afbc569ebf35b2'}]
    )


**NOTE**: When compiling with encryptionKey, ensure that any existing cache directory (either the default cache directory or the directory specified by the ``cache_dir`` provider option) is deleted before compiling.

|

**************************
Operator Assignment Report
**************************


Vitis AI EP generates a file named ``vitisai_ep_report.json`` that provides a report on model operator assignments across CPU and NPU. This file is automatically generated in the cache directory if no explicit cache location is specified in the code. This report includes information such as the total number of nodes, the list of operator types in the model, and which nodes and operators runs on the NPU or on the CPU. Additionally, the report includes node statistics, such as input to a node, the applied operation, and output from the node.


.. code-block::

  {
   "deviceStat": [
   {
    "name": "all",
    "nodeNum": 400,
    "supportedOpType": [
     "::Add",
     "::Conv",
     ...
    ]
   },
   {
    "name": "CPU",
    "nodeNum": 2,
    "supportedOpType": [
     "::DequantizeLinear",
     "::QuantizeLinear"
    ]
   },
   {
    "name": "NPU",
    "nodeNum": 398,
    "supportedOpType": [
     "::Add",
     "::Conv",
     ...
    ]
    ...

..
  ------------

  #####################################
  License
  #####################################

 Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
