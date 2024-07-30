.. _quantization-with-olive:


############################
Vitis AI Quantizer for Olive 
############################


*************
Prerequisites
*************

Ensure that Olive is correctly installed. For more information, see :ref:`Olive installation instructions <install-olive>`.


********************
Describing the Model 
********************

Olive requires information about your model, such as loading instructions, the names and shapes of input tensors, target hardware selection, and a list of optimizations you want to perform on the model. You can provide this information in a JSON file as input to Olive. For more details on using Olive and creating the Olive configuration file, refer to the `Microsoft Olive Documentation <https://microsoft.github.io/Olive/>`_.


*********************************
Configuring the Quantization Pass
*********************************

The JSON configuration file must include a ``passes`` key, which is a dictionary containing information about passes executed by the engine. The passes are executed in the order defined within the dictionary, where the key of the dictionary represents the name of the pass. 

To quantize the model for Ryzen AI, use the ``VitisAIQuantization`` pass. In the following example, two passes are used, converting to ONNX and quantizing using Vitis AI. 

.. code-block:: 

    "passes": {
        "onnx_conversion": {
            "type": "OnnxConversion",
            "config": {
                "target_opset": 13
            },
            "host": "local_system",
            "evaluator": "common_evaluator"
        },
        "vitis_ai_quantization": {
            "type": "VitisAIQuantization",
            "disable_search": true,
            "config": {
                "user_script": "user_script.py",
                "data_dir": "data",
                "dataloader_func": "resnet_calibration_reader"
            },
            "clean_run_cache": false
        }


Note: The ``target_opset`` configuration of ``onnx_conversion`` pass must be above 10.

For a complete description of the ``VitisAIQuantization`` pass, refer to the `VitisAIQuantization pass reference guide <https://microsoft.github.io/Olive/api/passes.html#vitis-ai-quantization>`_.


**************************
Checking the Configuration
**************************

Before running quantization with Olive, you can optionally execute a setup mode. This helps identify additional packages that might need to be installed to support the passes set in the configuration JSON file.

.. code-block:: 

   python -m olive.workflows.run --config resnet_static_config.json --setup


********************
Quantizing the Model
********************

To quantize the model, run Olive with the JSON configuration file as follows:


.. code-block:: 

   python -m olive.workflows.run --config resnet_static_config.json 


Here is the typical output:

.. code-block::

  [2023-05-29 01:03:07,086] [WARNING] [engine.py:97:__init__] No accelerators specified for target system. Using CPU.
  [2023-05-29 01:03:07,098] [DEBUG] [engine.py:539:resolve_goals] Resolving goals: {'accuracy': {<AccuracySubType.ACCURACY_SCORE: 'accuracy_score'>:     MetricGoal(type='max-degradation', value=0.01)}, 'latency': {'avg': MetricGoal(type='percent-min-improvement', value=20.0)}}
  [2023-05-29 01:03:07,101] [DEBUG] [engine.py:549:resolve_goals] Computing baseline for metrics ...
  [2023-05-29 01:03:07,101] [DEBUG] [engine.py:898:_evaluate_model] Evaluating model ...
  [2023-05-29 01:03:11,740] [DEBUG] [footprint.py:90:resolve_metrics] There is no goal set for metric: {metric_name}.
  [2023-05-29 01:03:11,740] [DEBUG] [footprint.py:90:resolve_metrics] There is no goal set for metric: {metric_name}.
  [2023-05-29 01:03:11,741] [DEBUG] [engine.py:562:resolve_goals] Baseline: {'accuracy-accuracy_score': 0.8729838728904724, 'latency-avg': 31.98742}
  [2023-05-29 01:03:11,741] [DEBUG] [engine.py:585:resolve_goals] Resolved goals: {'accuracy-accuracy_score': 0.8629838728904724, 'latency-avg': 25.589936}
  [2023-05-29 01:03:11,743] [DEBUG] [engine.py:460:run_search] Step 1 with search point {'OnnxConversion': {}, 'VitisAIQuantization': {}} ...
  [2023-05-29 01:03:11,743] [DEBUG] [engine.py:725:_run_passes] Running pass OnnxConversion
  ============== Diagnostic Run torch.onnx.export version 2.0.1+cpu ==============
  verbose: False, log level: Level.ERROR
  ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

  [2023-05-29 01:03:12,689] [DEBUG] [engine.py:725:_run_passes] Running pass VitisAIQuantization
  [2023-05-29 01:03:12,691] [INFO] [vitis_ai_quantization.py:256:_run_for_config] Preprocessing model for quantization
  Finding optimal threshold for each tensor using PowerOfTwoMethod.MinMSE algorithm ...
  [2023-05-29 01:03:53,389] [DEBUG] [engine.py:898:_evaluate_model] Evaluating model ...
  [2023-05-29 01:03:58,156] [DEBUG] [engine.py:765:_run_passes] Signal: {'accuracy-accuracy_score': 0.8145161271095276, 'latency-avg': 28.5457}
  [2023-05-29 01:03:58,157] [WARNING] [search_strategy.py:133:_next_search_group] No models in this search group ['OnnxConversion', 'VitisAIQuantization'] met the   goals. Sorting the models without applying goals...
  [2023-05-29 01:03:58,159] [INFO] [footprint.py:168:get_pareto_frontier] pareto frontier points: 1_VitisAIQuantization-0-5eced571581e0d511ed3467faeee47b8-cpu-cpu   {'accuracy-accuracy_score': 0.8145161271095276, 'latency-avg': 28.5457}
  [2023-05-29 01:03:58,159] [INFO] [engine.py:475:run_search] Output all 1 models
  [2023-05-29 01:03:58,161] [INFO] [engine.py:318:run] No packaging config provided, skip packaging artifacts



At the end of the Quantization process, the model is saved in the ``[model]``.onnx format. 

..
  ------------

  #####################################
  License
  #####################################

 Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
