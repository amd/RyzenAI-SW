########################
Getting Started Tutorial
########################

This tutorial uses a fine-tuned version of the ResNet model (using the CIFAR-10 dataset) to demonstrate the process of preparing, quantizing, and deploying a model using Ryzen AI Software. The tutorial features deployment using both Python and C++ ONNX runtime code. 

.. note::
   In this documentation, "NPU" is used in descriptions, while "IPU" is retained in the tool's language, code, screenshots, and commands. This intentional 
   distinction aligns with existing tool references and does not affect functionality. Avoid making replacements in the code.

- The source code files can be downloaded from `this link <https://github.com/amd/RyzenAI-SW/tree/main/tutorial/getting_started_resnet>`_. Alternatively, you can clone the RyzenAI-SW repo and change the directory into the tutorial code directory. 

.. code-block::

    git clone https://github.com/amd/RyzenAI-SW.git
    cd tutorial/getting_started_resnet

|

The following are the steps and the required files to run the example: 

.. list-table:: 
   :widths: 20 25 25
   :header-rows: 1

   * - Steps 
     - Files Used
     - Description
   * - Installation
     - ``requirements.txt``
     - Install the necessary package for this example.
   * - Preparation
     - ``prepare_model_data.py``,
       ``resnet_utils.py``
     - The script ``prepare_model_data.py`` prepares the model and the data for the rest of the tutorial.

       1. To prepare the model the script converts pre-trained PyTorch model to ONNX format.
       2. To prepare the necessary data the script downloads and extract CIFAR-10 dataset. 

   * - Pretrained model
     - ``models/resnet_trained_for_cifar10.pt``
     - The ResNet model trained using CIFAR-10 is provided in .pt format.
   * - Quantization 
     - ``resnet_quantize.py``
     - Convert the model to the NPU-deployable model by performing Post-Training Quantization flow using VitisAI ONNX Quantization.
   * - Deployment - Python
     - ``predict.py``
     -  Run the Quantized model using the ONNX Runtime code. We demonstrate running the model on both CPU and NPU. 
   * - Deployment - C++
     - ``cpp/resnet_cifar/.``
     -  This folder contains the source code ``resnet_cifar.cpp`` that demonstrates running inference using C++ APIs. We additionally provide the infrastructure (required libraries, CMake files and headerfiles) required by the example. 


|
|

************************
Step 1: Install Packages
************************

* Ensure that the Ryzen AI Software  is correctly installed. For more details, see the :doc:`installation instructions <inst>`.

* Use the conda environment created during the installation for the rest of the steps. This example requires a couple of additional packages. Run the following command to install them:


.. code-block:: 

   python -m pip install -r requirements.txt

|
|


**************************************
Step 2: Prepare dataset and ONNX model
**************************************

In this example, we utilize the a custom ResNet model finetuned using the CIFAR-10 dataset

The ``prepare_model_data.py`` script downloads the CIFAR-10 dataset in pickle format (for python) and binary format (for C++). This dataset will be used in the subsequent steps for quantization and inference. The script also exports the provided PyTorch model into ONNX format. The following snippet from the script shows how the ONNX model is exported:

.. code-block:: 

    dummy_inputs = torch.randn(1, 3, 32, 32)
    input_names = ['input']
    output_names = ['output']
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    tmp_model_path = str(models_dir / "resnet_trained_for_cifar10.onnx")
    torch.onnx.export(
            model,
            dummy_inputs,
            tmp_model_path,
            export_params=True,
            opset_version=13,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

Note the following settings for the onnx conversion:

- Ryzen AI supports a batch size=1, so dummy input is fixed to a batch_size =1 during model conversion
- Recommended ``opset_version`` setting 13 is used. 

Run the following command to prepare the dataset and export the ONNX model:

.. code-block:: 

   python prepare_model_data.py 

* The downloaded CIFAR-10 dataset is saved in the current directory at the following location: ``data/*``.
* The ONNX model is generated at models/resnet_trained_for_cifar10.onnx

|
|

**************************
Step 3: Quantize the Model
**************************

Quantizing AI models from floating-point to 8-bit integers reduces computational power and the memory footprint required for inference. This example utilizes the Vitis AI ONNX quantizer workflow. Quantization tool takes the pre-trained float32 model from the previous step (``resnet_trained_for_cifar10.onnx``) and produces a quantized model.

.. code-block::

   python resnet_quantize.py

This generates a quantized model using QDQ quant format and UInt8 activation type and Int8 weight type. After the completion of the run, the quantized ONNX model ``resnet.qdq.U8S8.onnx`` is saved to models/resnet.qdq.U8S8.onnx. 

The :file:`resnet_quantize.py` file has ``quantize_static`` function (line 95) that applies static quantization to the model. 

.. code-block::

   from onnxruntime.quantization import QuantFormat, QuantType
   import vai_q_onnx

   vai_q_onnx.quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=vai_q_onnx.QuantFormat.QDQ,
        calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
        activation_type=vai_q_onnx.QuantType.QUInt8,
        weight_type=vai_q_onnx.QuantType.QInt8,
        enable_dpu=True, 
        extra_options={'ActivationSymmetric': True} 
    )

The parameters of this function are:

* **input_model_path**: (String) The file path of the model to be quantized.
* **output_model_path**: (String) The file path where the quantized model is saved.
* **dr**: (Object or None) Calibration data reader that enumerates the calibration data and producing inputs for the original model. In this example, CIFAR10 dataset is used for calibration during the quantization process.
* **quant_format**: (String) Specifies the quantization format of the model. In this example we have used the QDQ quant format.
* **calibrate_method**: (String) In this example this parameter is set to ``vai_q_onnx.PowerOfTwoMethod.MinMSE`` to apply power-of-2 scale quantization. 
* **activation_type**: (String) Data type of activation tensors after quantization. In this example, it's set to QInt8 (Quantized Integer 8).
* **weight_type**: (String) Data type of weight tensors after quantization. In this example, it's set to QInt8 (Quantized Integer 8).
* **enable_dpu**: (Boolean) Determines whether to generate a quantized model that is suitable for the NPU/DPU. If set to True, the quantization process will create a model that is optimized for NPU/DPU computations.
* **extra_options**: (Dict or None) Dictionary of additional options that can be passed to the quantization process. In this example, ``ActivationSymmetric`` is set to True. It means calibration data for activations is symmetrized. 

|
|

************************
Step 4: Deploy the Model  
************************

We demonstrate deploying the quantized model using both Python and C++ APIs. 

* :ref:`Deployment - Python <dep-python>`
* :ref:`Deployment - C++ <dep-cpp>`

.. note::
   During the Python and C++ deployment, the compiled model artifacts are saved in the cache folder named ``<run directory>/modelcachekey``. Ryzen-AI does not support the complied model artifacts across the versions, so if the model artifacts exist from the previous software version, ensure to delete the folder ``modelcachekey`` before the deployment steps. 


.. _dep-python:

Deployment - Python
===========================

The ``predict.py`` script is used to deploy the model. It extracts the first ten images from the CIFAR-10 test dataset and converts them to the .png format. The script then reads all those ten images and classifies them by running the quantized custom ResNet model on CPU or NPU. 

Deploy the Model on the CPU
----------------------------

By default, ``predict.py`` runs the model on CPU. 

.. code-block::
  
        > python predict.py

Typical output

.. code-block:: 

        Image 0: Actual Label cat, Predicted Label cat
        Image 1: Actual Label ship, Predicted Label ship
        Image 2: Actual Label ship, Predicted Label airplane
        Image 3: Actual Label airplane, Predicted Label airplane
        Image 4: Actual Label frog, Predicted Label frog
        Image 5: Actual Label frog, Predicted Label frog
        Image 6: Actual Label automobile, Predicted Label automobile
        Image 7: Actual Label frog, Predicted Label frog
        Image 8: Actual Label cat, Predicted Label cat
        Image 9: Actual Label automobile, Predicted Label automobile
        
                
Deploy the Model on the Ryzen AI NPU
------------------------------------

To successfully run the model on the NPU, run the following setup steps:

- Ensure that the ``XLNX_VART_FIRMWARE`` environment variable is correctly pointing to the :file:`1x4.xclbin` file located in the ``voe-4.0-win_amd64`` folder of the Ryzen AI software installation package. If you installed the Ryzen AI software using automatic installer, this variable is already correctly set. However, if you did the installation manually, you must set the variable as follows: 

.. code-block:: bash 

   set XLNX_VART_FIRMWARE=path\to\RyzenAI\installation\files\ryzen-ai-sw-1.1\voe-4.0-win_amd64\1x4.xclbin

- Copy the :file:`vaip_config.json` runtime configuration file from the ``voe-4.0-win_amd64`` folder of the Ryzen AI software installation package to the current directory. The :file:`vaip_config.json` is used by the :file:`predict.py` script to configure the Vitis AI Execution Provider.


The following section of the :file:`predict.py` script shows how ONNX Runtime is configured to deploy the model on the Ryzen AI NPU:


.. code-block::

  parser = argparse.ArgumentParser()
  parser.add_argument('--ep', type=str, default ='cpu',choices = ['cpu','ipu'], help='EP backend selection')
  opt = parser.parse_args()
  
  providers = ['CPUExecutionProvider']
  provider_options = [{}]

  if opt.ep == 'ipu':
     providers = ['VitisAIExecutionProvider']
     cache_dir = Path(__file__).parent.resolve()
     provider_options = [{
                'config_file': 'vaip_config.json',
                'cacheDir': str(cache_dir),
                'cacheKey': 'modelcachekey'
                }]

  session = ort.InferenceSession(model.SerializeToString(), providers=providers,
                                 provider_options=provider_options)


Run the ``predict.py`` with the ``--ep ipu`` switch to run the custom ResNet model on the Ryzen AI NPU:


.. code-block::

    >python predict.py --ep ipu

Typical output

.. code-block::

    I20231129 12:50:18.631383 16736 vitisai_compile_model.cpp:336] Vitis AI EP Load ONNX Model Success
    I20231129 12:50:18.631383 16736 vitisai_compile_model.cpp:337] Graph Input Node Name/Shape (1)
    I20231129 12:50:18.631383 16736 vitisai_compile_model.cpp:341]   input : [-1x3x32x32]
    I20231129 12:50:18.631383 16736 vitisai_compile_model.cpp:347] Graph Output Node Name/Shape (1)
    I20231129 12:50:18.631383 16736 vitisai_compile_model.cpp:351]   output : [-1x10]
    I20231129 12:50:18.631383 16736 vitisai_compile_model.cpp:226] use cache key modelcachekey
    I20231129 12:50:23.717264 16736 compile_pass_manager.cpp:352] Compile mode: aie
    I20231129 12:50:23.717264 16736 compile_pass_manager.cpp:353] Debug mode: performance
    I20231129 12:50:23.717264 16736 compile_pass_manager.cpp:357] Target architecture: AMD_AIE2_Nx4_Overlay
    I20231129 12:50:23.717264 16736 compile_pass_manager.cpp:540] Graph name: main_graph, with op num: 438
    I20231129 12:50:23.717264 16736 compile_pass_manager.cpp:553] Begin to compile...
    W20231129 12:50:27.786000 16736 RedundantOpReductionPass.cpp:663] xir::Op{name = /avgpool/GlobalAveragePool_output_0_DequantizeLinear_Output_vaip_315, type = pool-fix}'s input and output is unchanged, so it will be removed.              
    I20231129 12:50:27.945919 16736 PartitionPass.cpp:6142] xir::Op{name = output_, type = fix2float} is not supported by current target. Target name: AMD_AIE2_Nx4_Overlay, target type: IPU_PHX. Assign it to CPU.
    I20231129 12:50:29.098559 16736 compile_pass_manager.cpp:565] Total device subgraph number 3, CPU subgraph number 1
    I20231129 12:50:29.098559 16736 compile_pass_manager.cpp:574] Total device subgraph number 3, DPU subgraph number 1
    I20231129 12:50:29.098559 16736 compile_pass_manager.cpp:583] Total device subgraph number 3, USER subgraph number 1
    I20231129 12:50:29.098559 16736 compile_pass_manager.cpp:639] Compile done.
    .... 
    [Vitis AI EP] No. of Operators :   CPU     2    IPU   398  99.50% 
    [Vitis AI EP] No. of Subgraphs :   CPU     1    IPU     1 Actually running on IPU     1  
    ...
    Image 0: Actual Label cat, Predicted Label cat
    Image 1: Actual Label ship, Predicted Label ship
    Image 2: Actual Label ship, Predicted Label ship
    Image 3: Actual Label airplane, Predicted Label airplane
    Image 4: Actual Label frog, Predicted Label frog 
    Image 5: Actual Label frog, Predicted Label frog 
    Image 6: Actual Label automobile, Predicted Label truck
    Image 7: Actual Label frog, Predicted Label frog
    Image 8: Actual Label cat, Predicted Label cat
    Image 9: Actual Label automobile, Predicted Label automobile 
   

.. _dep-cpp:

Deployment - C++
===========================

Prerequisites
-------------

1. Visual Studio 2019 Community edition, ensure "Desktop Development with C++" is installed
2. cmake (version >= 3.26)
3. opencv (version=4.6.0) required for the custom resnet example

Install OpenCV 
--------------

It is recommended to build OpenCV from the source code and use static build. The default installation localtion is "\install" , the following instruction installs OpenCV in the location "C:\\opencv" as an example. You may first change the directory to where you want to clone the OpenCV repository.

.. code-block:: bash

   git clone https://github.com/opencv/opencv.git -b 4.6.0
   cd opencv
   cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -G "Visual Studio 16 2019" "-DCMAKE_INSTALL_PREFIX=C:\opencv" "-DCMAKE_PREFIX_PATH=C:\opencv" -DCMAKE_BUILD_TYPE=Release -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_WITH_STATIC_CRT=OFF -B build
   cmake --build build --config Release
   cmake --install build --config Release

Build and Run Custom Resnet C++ sample
--------------------------------------

.. note::
   In this documentation, **"NPU"** is used in descriptions, while **"IPU"** is retained in the tool's language, code, screenshots, and commands. This intentional 
   distinction aligns with existing tool references and does not affect functionality. Avoid making replacements in the code.

The C++ source files, CMake list files and related artifacts are provided in the ``cpp/resnet_cifar/*`` folder. The source file ``cpp/resnet_cifar/resnet_cifar.cpp`` takes 10 images from the CIFAR-10 test set, converts them to .png format, preprocesses them, and performs model inference. The example has onnxruntime dependencies, that are provided in ``cpp/resnet_cifar/onnxruntime/*``. 

Run the following command to build the resnet example. Assign ``-DOpenCV_DIR`` to the OpenCV installation directory.

.. code-block:: bash

   cd getting_started_resnet/cpp
   cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -DCMAKE_INSTALL_PREFIX=. -DCMAKE_PREFIX_PATH=. -B build -S resnet_cifar -DOpenCV_DIR="C:/opencv" -G "Visual Studio 16 2019"

This should generate the build directory with the ``resnet_cifar.sln`` solution file along with other project files. Open the solution file using Visual Studio 2019 and build to compile. You can also use "Developer Command Prompt for VS 2019" to open the solution file in Visual Studio.

.. code-block:: bash 

   devenv build/resnet_cifar.sln

Now to deploy our model, we will go back to the parent directory (getting_started_resnet) of this example. After compilation, the executable should be generated in ``cpp/resnet_cifar/build/Release/resnet_cifar.exe``. We will copy this application over to the parent directory:

.. code-block:: bash 

   cd ..
   xcopy cpp\build\Release\resnet_cifar.exe .

Additionally, we will also need to copy the onnxruntime DLLs from the Vitis AI Execution Provider package to the current directory. The following commands copy the required files in the current directory: 

.. code-block:: bash 

   xcopy %RYZEN_AI_INSTALLER%\onnxruntime\bin\onnxruntime.dll .
   xcopy %RYZEN_AI_INSTALLER%\onnxruntime\bin\onnxruntime_vitisai_ep.dll .

Here, ``RYZEN_AI_INSTALLER`` is an environment variable that should point to the ``path\to\ryzen-ai-sw-xx\ryzen-ai-sw-xx``. If you installed Ryzen-AI software using the automatic installer, this variable should already be set. Ensure that the Ryzen-AI software package has not been moved post installation, in which case ``RYZEN_AI_INSTALLER`` will have to be set again. 


The C++ application that was generated takes 3 arguments: 

#. Path to the quantized ONNX model generated in Step 3 
#. The execution provider of choice (cpu or NPU) 
#. vaip_config.json (pass None if running on CPU) 


Deploy the Model on the CPU
****************************

To run the model on the CPU, use the following command: 

.. code-block:: bash 

   resnet_cifar.exe models\resnet.qdq.U8S8.onnx cpu None

Typical output: 

.. code-block:: bash 

   model name:models\resnet.qdq.U8S8.onnx
   ep:cpu
   Input Node Name/Shape (1):
           input : -1x3x32x32
   Output Node Name/Shape (1):
           output : -1x10
   Final results:
   Predicted label is cat and actual label is cat
   Predicted label is ship and actual label is ship
   Predicted label is ship and actual label is ship
   Predicted label is airplane and actual label is airplane
   Predicted label is frog and actual label is frog
   Predicted label is frog and actual label is frog
   Predicted label is truck and actual label is automobile
   Predicted label is frog and actual label is frog
   Predicted label is cat and actual label is cat
   Predicted label is automobile and actual label is automobile

Deploy the Model on the NPU
****************************

To successfully run the model on the NPU:

- Ensure that the ``XLNX_VART_FIRMWARE`` environment variable is correctly pointing to the XCLBIN file included in the ONNX Vitis AI Execution Provider package. If you installed Ryzen-AI software by automatic installer, the NPU binary path is already set, however if you did the installation manually, ensure the NPU binary path is set using the following command: 

.. code-block:: bash 

   set XLNX_VART_FIRMWARE=path\to\RyzenAI\installation\ryzen-ai-sw-1.1\ryzen-ai-sw-1.1\voe-4.0-win_amd64\1x4.xclbin


- Copy the ``vaip_config.json`` runtime configuration file from the Vitis AI Execution Provider package to the current directory. The ``vaip_config.json`` is used by the source file ``resnet_cifar.cpp`` to configure the Vitis AI Execution Provider.


The following code block from ``reset_cifar.cpp`` shows how ONNX Runtime is configured to deploy the model on the Ryzen AI NPU:

.. code-block:: bash 

    auto session_options = Ort::SessionOptions();

    auto config_key = std::string{ "config_file" };
    auto cache_dir = std::filesystem::current_path().string(); 
 
    if(ep=="ipu")
    {
    auto options =
        std::unordered_map<std::string, std::string>{ {config_key, json_config}, {"cacheDir", cache_dir}, {"cacheKey", "modelcachekey"} };
    session_options.AppendExecutionProvider("VitisAI", options);
    }

    auto session = Ort::Experimental::Session(env, model_name, session_options);

To run the model on the NPU, we will pass the ipu flag and the vaip_config.json file as arguments to the C++ application. Use the following command to run the model on the NPU: 

.. code-block:: bash 

   resnet_cifar.exe models\resnet.qdq.U8S8.onnx ipu vaip_config.json

Typical output: 

.. code-block::

   I20231129 13:19:47.882169 14796 vitisai_compile_model.cpp:336] Vitis AI EP Load ONNX Model Success
   I20231129 13:19:47.882169 14796 vitisai_compile_model.cpp:337] Graph Input Node Name/Shape (1)
   I20231129 13:19:47.882169 14796 vitisai_compile_model.cpp:341]   input : [-1x3x32x32]
   I20231129 13:19:47.882169 14796 vitisai_compile_model.cpp:347] Graph Output Node Name/Shape (1)
   I20231129 13:19:47.882169 14796 vitisai_compile_model.cpp:351]   output : [-1x10]
   I20231129 13:19:53.161406 14796 compile_pass_manager.cpp:352] Compile mode: aie
   I20231129 13:19:53.161406 14796 compile_pass_manager.cpp:353] Debug mode: performance
   I20231129 13:19:53.161406 14796 compile_pass_manager.cpp:357] Target architecture: AMD_AIE2_Nx4_Overlay
   I20231129 13:19:53.161406 14796 compile_pass_manager.cpp:540] Graph name: main_graph, with op num: 438
   I20231129 13:19:53.161406 14796 compile_pass_manager.cpp:553] Begin to compile...
   W20231129 13:19:57.223416 14796 RedundantOpReductionPass.cpp:663] xir::Op{name = /avgpool/GlobalAveragePool_output_0_DequantizeLinear_Output_vaip_315, type = pool-fix}'s input and output is unchanged, so it will be removed.
   I20231129 13:19:57.389281 14796 PartitionPass.cpp:6142] xir::Op{name = output_, type = fix2float} is not supported by current target. Target name: AMD_AIE2_Nx4_Overlay, target type: IPU_PHX. Assign it to CPU.
   I20231129 13:19:58.546655 14796 compile_pass_manager.cpp:565] Total device subgraph number 3, CPU subgraph number 1
   I20231129 13:19:58.546655 14796 compile_pass_manager.cpp:574] Total device subgraph number 3, DPU subgraph number 1
   I20231129 13:19:58.546655 14796 compile_pass_manager.cpp:583] Total device subgraph number 3, USER subgraph number 1
   I20231129 13:19:58.547658 14796 compile_pass_manager.cpp:639] Compile done.
   I20231129 13:19:58.583139 14796 anchor_point.cpp:444] before optimization:
   ... 
   [Vitis AI EP] No. of Operators :   CPU     2    IPU   398  99.50%
   [Vitis AI EP] No. of Subgraphs :   CPU     1    IPU     1 Actually running on IPU     1
   ...
   Final results:   
   Predicted label is cat and actual label is cat
   Predicted label is ship and actual label is ship
   Predicted label is ship and actual label is ship
   Predicted label is airplane and actual label is airplane
   Predicted label is frog and actual label is frog
   Predicted label is frog and actual label is frog
   Predicted label is truck and actual label is automobile
   Predicted label is frog and actual label is frog
   Predicted label is cat and actual label is cat
   Predicted label is automobile and actual label is automobile                                                                                                                                                                
..
  ------------

  #####################################
  License
  #####################################

 Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
