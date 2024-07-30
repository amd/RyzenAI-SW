####################
ONNX End-to-End Flow
####################

AI models often require pre/post-processing operations that are not part of the model itself. To run the pre/post-processing operations on the NPU, we have developed a mechanism for integrating them into the original ONNX model. This allows for end-to-end model inference using Vitis AI Execution Provider. The feature is built by leveraging the ONNX Runtime feature `ONNXRuntime-Extensions <https://onnxruntime.ai/docs/extensions/>`_. Typical pre-processing (or post-processing) tasks, such as resizing, normalization, etc can be expressed as custom operators. The pre-trained model can then be extended by absorbing these custom operators. The resulting model that contains the pre/post-processing operations can then be run on NPU. This helps improve end-to-end latency and facilitates PC power saving by reducing CPU utilization.


We provide the ``vitis_customop`` library that supports some common tasks such as resizing, normalization, NMS, etc. These operations are accessible through high-level API calls. The user will need to specify the following:

- pre/postprocessing operations
- operation-specific parameters

We will continue to expand the library with more supported operations. 

The following steps describe how to use the pre and post-processor APIs:

**Step 1:**


Create PreProcessor and PostProcessor instances:

.. code-block:: python

  from vitis_customop.preprocess import generic_preprocess as pre
  from vitis_customop.postprocess_resnet import generic_post_process as post
  input_node_name = "blob.1"
  preprocessor = pre.PreProcessor(input_model_path, output_model_path, input_node_name)
  output_node_name = "1327"
  postprocessor = post.PostProcessor(onnx_pre_model_name, onnx_e2e_model_name, output_node_name)


**Step 2:**

Specify the operations to perform, and pass the required parameters. 

.. code-block:: python

  preprocessor.resize(resize_shape)
  preprocessor.normalize(mean, std_dev, scale)
  preprocessor.set_resnet_params(mean, std_dev, scale)
  postprocessor.ResNetPostProcess()


**Step 3:**

Generate and save the new model

.. code-block:: python

  preprocessor.build()
  postprocessor.build()


Examples to utilize the ONNX end-to-end flow can be found `here <https://github.com/amd/RyzenAI-SW/tree/main/example/onnx-e2e>`_.

..
  ------------

  #####################################
  License
  #####################################

 Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
