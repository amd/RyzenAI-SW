Adding Calibration Datasets
===========================

Class DataReader for AMD Quark Quantizer
----------------------------------------

AMD Quark for ONNX utilizes ONNX Runtime's `CalibrationDataReader` for normalization during quantization calibration. The following code is an example of how to define the class for the calibration data loader.

.. code-block:: python

   import onnxruntime
   from onnxruntime.quantization.calibrate import CalibrationDataReader

   class ImageDataReader(CalibrationDataReader):

       def __init__(self, calibration_image_folder: str, input_name: str,
        input_height: int, input_width: int):
           self.enum_data = None

           self.input_name = input_name

           self.data_list = self._preprocess_images(
                   calibration_image_folder, input_height, input_width)

       # The pre-processing of calibration images should be defined by users.
       # Recommended batch_size is 1.
       def _preprocess_images(self, image_folder: str, input_height: int, input_width: int, batch_size: int = 1):
           data_list = []
           '''
           The pre-processing for each image
           '''
           return data_list

       def get_next(self):
           if self.enum_data is None:
               self.enum_data = iter([{self.input_name: data} for data in self.data_list])
           return next(self.enum_data, None)

       def rewind(self):
           self.enum_data = None

   input_model_path = "path/to/your/resnet50.onnx"
   output_model_path = "path/to/your/resnet50_quantized.onnx"
   calibration_image_folder = "path/to/your/images"

   input_name = 'input_tensor_name'
   input_shape = (1, 3, 224, 224)
   calib_datareader = ImageDataReader(calibration_image_folder, input_name,
    input_shape[2], input_shape[3])


Calibration Data Path for AMD Quark Quantizer
---------------------------------------------

AMD Quark for ONNX supports specifying the path to calibration datasets, making it easy to load them for quantization. Currently, this feature only supports data in `.npy` format.
For detailed guidance on creating calibration datasets in NPY format, see :doc:`Generating NPY Calibration Data<./user_guide_onnx_model_inference_save_input_npy>`.

.. note::
    No preprocessing is applied to the calibration datasets after loading. Ensure that the calibration data is stored in the following format:

For Single-Input Models:
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Place the calibration data files in a directory as follows:

.. code-block::

   calibration_data/
     calib_000001.npy
     calib_000002.npy
     calib_000003.npy
     calib_000004.npy
     calib_000005.npy
     ...

For Multi-Input Models:
~~~~~~~~~~~~~~~~~~~~~~~

Organize the calibration data in sub-directories named after the input models:

.. code-block::

   calibration_data/
     input1_name/
       calib_000001.npy
       calib_000002.npy
       calib_000003.npy
       calib_000004.npy
       calib_000005.npy
       ...
     input2_name/
       calib_000001.npy
       calib_000002.npy
       calib_000003.npy
       calib_000004.npy
       calib_000005.npy
       ...
     ...

Example Code:
~~~~~~~~~~~~~~~

.. code-block:: python

   import onnxruntime
   from quark.onnx import ModelQuantizer
   from quark.onnx.quantization.config import Config, get_default_config

   input_model_path = "path/to/your/resnet50.onnx"
   output_model_path = "path/to/your/resnet50_quantized.onnx"
   calib_data_path= "path/to/your/calib/data/folder"

   quant_config = get_default_config("XINT8")
   config = Config(global_quant_config=quant_config)

   quantizer = ModelQuantizer(config)
   quantizer.quantize_model(input_model_path, output_model_path, calibration_data_reader=None, calibration_data_path=calib_data_path)

Using Random Data for AMD Quark Quantizer
-----------------------------------------

Random Data Calibration uses random numbers when no calibration data is available. To enable this feature, set the `UseRandomData` parameter to `True`. This option is useful for testing but might yield worse quantization results than using a real calibration dataset. It is recommended to use a real calibration dataset when performing static quantization.

Example Code:
~~~~~~~~~~~~~

.. code-block:: python

   import onnxruntime
   from quark.onnx import ModelQuantizer
   from quark.onnx.quantization.config import Config, get_default_config

   input_model_path = "path/to/your/resnet50.onnx"
   output_model_path = "path/to/your/resnet50_quantized.onnx"

   quant_config = get_default_config("XINT8")
   quant_config.extra_options['UseRandomData'] = True
   config = Config(global_quant_config=quant_config)

   quantizer = ModelQuantizer(config)
   quantizer.quantize_model(input_model_path, output_model_path, calibration_data_reader=None)
