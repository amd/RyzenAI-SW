Quick Start for Ryzen AI
========================

.. note::

    In this documentation, **AMD Quark** is sometimes referred to simply as **"Quark"** for ease of reference. When you  encounter the term "Quark" without the "AMD" prefix, it specifically refers to the AMD Quark quantizer unless otherwise stated. Please do not confuse it with other products or technologies that share the name "Quark".

Following the :doc:`Basic usage <../../basic_usage>` guideline page, this document will go through all four steps to quantize a model using a minimalistic approach and build on it to show off some advanced features of AMD Quark. Towards the end an evaluation will be performed to assess the quality of the resulting quantization.


1. Prepare the original float model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For this quick start guide, we will take the **ResNet-50** as example. **ResNet-50** is a Convolutional Neural Network (CNN) that's commonly used for image classification. It's part of the ResNet (Residual Networks) family of models, which were developed to address challenges with training deep neural networks.  ResNet-50 is made up of 50 weight layers, including convolution layers, residual blocks, and fully connected layers. The model has skip connections that allow the model to skip one or more layers, preventing vanishing gradients. Typically, ResNet-50 is trained on over a million images from the **ImageNet** database and applications in real life include medical imaging, anomaly detection, and inventory management.

To get started, first, download the model from the `onnx/models <https://github.com/onnx/models>`__ repo directly:

.. code-block:: bash

    wget -P models https://github.com/onnx/models/raw/new-models/vision/classification/resnet/model/resnet50-v1-12.onnx

.. note::

    In the quantization, graph optimization will be automatically performed.

2. Prepare calibration data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Typically, quantization can be performed without calibration data. However, feeding a representative dataset during the calibration stage yields better results. Here we are going to show how to quantize models with and without calibration data for learning purposes.

2.1. Quantization without Calibration Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Models can be quantized without calibration data. For such, AMD Quark provides an API to perform quantization using auto generated random data. The command line below shows how to quantize a float model without calibration data. Here we are going to use the default quantization config is **A8W8**, you can also use **XINT8**, **A16W8**, and so on. Refer to :doc:`Quark-ONNX Configuration page <../../onnx/user_guide_config_description>` to learn more about the supported data type and quantization configuration.

.. code-block:: bash

    python -m quark.onnx.tools.random_quantize --input_model_path models/resnet50-v1-12.onnx --quantized_model_path models/resnet50-v1-12_random_quantized.onnx --config A8W8

.. note::

    Since the calibration data is an automatically generated tensor with values in the range [0, 1], errors may occur when models require integer input. In such cases, this tool cannot be used and real calibration data must be provided. Similarly, if you want to achieve good quantization accuracy, you must use calibration data.

2.2. Quantization with Calibration Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we are going to perform calibration to learn characteristics of the user input and yield better accuracy. Users must provide a representative dataset in this step. ResNet-50 expects a calibration data folder with images in PNG or JPG formats. For example, you can download images from `Microsoft ONNX Runtime test images <https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/cpu/test_images>`_.

.. code-block:: bash

    mkdir calib_data
    wget -O calib_data/daisy.jpg https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/test_images/daisy.jpg?raw=true

Next, implement the calibration data reader API as shown:

.. code-block:: python

    import os
    import cv2
    import numpy as np
    from torchvision import transforms

    calib_data_folder = "calib_data"
    model_input_name = 'data'

    # You can define your preprocess method
    def preprocess_image(image_path):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.numpy()
        return image

    class CalibrationDataReader:
        def __init__(self, calib_data_folder: str, model_input_name: str):
            super().__init__()
            self.input_name = model_input_name
            self.processed_data = []
            self.data = self._load_calibration_data(calib_data_folder)
            self.index = 0

        def _load_calibration_data(self, data_folder: str):
            for image_filename in os.listdir(data_folder):
                if image_filename.lower().endswith(('.jpg', '.png')):
                    image_path = os.path.join(data_folder, image_filename)
                    image = preprocess_image(image_path)
                    self.processed_data.append(image)
            return self.processed_data

        def get_next(self):
            if self.index < len(self.processed_data):
                input_data = {self.input_name: self.processed_data[self.index]}
                self.index += 1
                return input_data
            return None

        # Instantiate the calibration data reader
    calib_data_reader = CalibrationDataReader(calib_data_folder, model_input_name)

3. Set the quantization configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The code below shows how to quantize a float model with **A8W8**. For more detailed information about basic quantization, please see :doc:`Basic Usage <../../onnx/basic_usage_onnx>`.

.. code-block:: python

    from quark.onnx.quantization.config import Config, get_default_config
    from quark.onnx import ModelQuantizer

    # Set up quantization with a specified configuration
    # For example, use "A8W8" for Ryzen AI INT8 quantization
    a8w8_config = get_default_config("A8W8")
    quantization_config = Config(global_quant_config=a8w8_config)
    quantizer = ModelQuantizer(quantization_config)

.. note::

    The A8W8 configuration is our default setup. To minimize quantization time, accuracy-improvement strategies such as AdaRound or AdaQuant are not applied by default, which may lead to suboptimal accuracy in some cases. For better quantization accuracy, please refer to Section **How to Improve Quantization Accuracy** of :doc:`Float Scales (A8W8 and A16W8) Quantization <tutorial_a8w8_and_a16w8_quantize>` page for details.

4. Quantize the model
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    float_model_path = "models/resnet50-v1-12.onnx"

    quantized_model_path = "models/resnet50-v1-12_quantized.onnx"

    # Quantize the ONNX model and save to specified path
    quantizer.quantize_model(float_model_path, quantized_model_path, calib_data_reader)

4.1 Quantize the model with Advanced Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By this point, the model has been quantized and a certain level of performance was observed. many times the achieved performance is not sufficient and users might be interested in using Advanced Features to improve the results. AMD Quark advanced features include **ADAROUND** and **ADAQUANT**. Compared to basic quantization, the user only needs to update the quantization configuration. For example, user could replace **A8W8** with **A8W8_ADAROUND** or **A8W8_ADAQUANT**.

Let's try replacing the above corresponding two lines with the following a few lines of code.

.. code-block:: python

    a8w8_adaround_config = get_default_config("A8W8_ADAROUND")
    # a8w8_adaquant_config = get_default_config("A8W8_ADAQUANT")
    quantization_config = Config(global_quant_config=a8w8_adaround_config)
    # quantization_config = Config(global_quant_config=a8w8_adaquant_config)

For more detailed information about AdaRound and AdaQuant, please see :doc:`Quantization Using AdaQuant and AdaRound <../../onnx/accuracy_algorithms/ada>`.

5. Evaluation
~~~~~~~~~~~~~

Now that the model is quantized, let's measure how good the model performs. Let's take an image in calibration data folder as input and dump the output ``NumPy`` tensor.

.. code-block:: python

    import os
    import numpy as np
    import cv2
    import onnx
    from torchvision import transforms
    from onnxruntime import InferenceSession

    def preprocess_image(image_path):
	transform = transforms.Compose([
	    transforms.ToPILImage(),
	    transforms.Resize((224, 224)),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = transform(image)
	image = image.unsqueeze(0)
	return image

    def load_onnx_model(model_path):
	session = InferenceSession(model_path)
	return session

    def infer_on_image(session, image):
	input_name = session.get_inputs()[0].name
	output_name = session.get_outputs()[0].name
	result = session.run([output_name], {input_name: image.numpy()})
	return result[0]

    def process_images_and_infer(input_folder, onnx_model_path, output_folder):
	if not os.path.exists(output_folder):
	    os.makedirs(output_folder)
	session = load_onnx_model(onnx_model_path)
	for image_filename in os.listdir(input_folder):
	    if image_filename.lower().endswith(('.jpg', '.png')):
		image_path = os.path.join(input_folder, image_filename)
		print(f"Processing {image_path}...")
		image = preprocess_image(image_path)
		result = infer_on_image(session, image)
		output_filename = os.path.splitext(image_filename)[0] + '_output.npy'
		output_path = os.path.join(output_folder, output_filename)
		np.save(output_path, result)
		print(f"Saved result to {output_path}")

    input_folder = "calib_data"
    onnx_model_path = "models/resnet50-v1-12.onnx" # Replace with "models/resnet50-v1-12_random_quantized.onnx" or "models/resnet50-v1-12_quantized.onnx"
    output_folder = "float_output" # Repalce with "random_quantized_output" or "quantized_output"
    process_images_and_infer(input_folder, onnx_model_path, output_folder)

Quark provides a tool to compare the differences between float and quantized models using ``L2 Loss`` and other metrics. For example:

.. code-block:: bash

    python -m quark.onnx.tools.evaluate --baseline_results_folder float_output --quantized_results_folder random_quantized_output

6. Results
~~~~~~~~~~

As shown in the table below, random quantization results in a very large L2 loss. Using calibration data can significantly reduce the loss, and advanced features can further minimize it.

.. list-table::
   :header-rows: 1

   * -
     - Float Model
     - Quantized Model without Calibration Data
     - Quantized Model with A8W8 Config
     - Quantized Model with A8W8 + AdaRound Config
     - Quantized Model with A8W8 + AdaQuant Config
   * - Model Size
     - 99 MB
     - 25 MB
     - 25 MB
     - 25 MB
     - 25 MB
   * - L2 Loss (compared with float model)
     - 0
     - 30.26
     - 9.78
     - 1.43
     - 1.15

.. raw:: html

   <!-- omit in toc -->

License
~~~~~~~

Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
