#################################
Vitis AI Quantizer for TensorFlow
#################################

.. note:: 

    All TensorFlow related documentation is applicable to the TensorFlow 2 version. 



*********************
Enabling Quantization
*********************

Ensure that the Vitis AI Quantizer for TensorFlow is correctly installed. For more information, see the :ref:`installation instructions <install-pt-tf>`.

To enable the Vitis AI Quantizer for TensorFlow, activate the conda environment in the Vitis AI Pytorch TensorFlow 2 container:

.. code-block::

     conda activate vitis-ai-tensorflow2
     

**************************
Post-Training Quantization
**************************

Post-Training Quantization requires the following files:

1. Float model : Floating-point TensorFlow models, either in h5 format or a saved model format.
2. Calibration dataset: A subset of the training dataset containing 100 to 1000 images.
 
 
A complete example of Post-Training Quantization is available at `Vitis AI GitHub <https://github.com/Xilinx/Vitis-AI/blob/v3.0/src/vai_quantizer/vai_q_tensorflow2.x/tensorflow_model_optimization/python/examples/quantization/keras/vitis/mnist_cnn_ptq.py>`__.
     
Vitis AI Quantization APIs
==========================     

Vitis AI provides the ``vitis_activation`` module into Tensorflow library for quantization. The following code shows the usage:

.. code-block::

   model = tf.keras.models.load_model(‘float_model.h5’)
   from tensorflow_model_optimization.quantization.keras import vitis_quantize
   quantizer = vitis_quantize.VitisQuantizer(model)
   quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset,calib_steps=100,calib_batch_size=10, **kwargs)
   

- calib_dataset: Used as a representative calibration dataset for calibration. 
- calib_steps: Total number of steps for calibration. 
- calib_batch_size: Number of samples per batch for calibration. 
- input_shape: Input shape for each input layer. 
- kwargs: Dictionary of the user-defined configurations of quantize strategy. 

Exporting the Model for Deployment
==================================

After the quantization, the quantized model can be saved into ONNX to deploy with ONNX Runtime Vitis AI Execution Provider: 

.. code-block::

   quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset, 
                                              output_format='onnx', 
                                              onnx_opset_version=11, 
                                              output_dir='./quantize_results', 
                                              **kwargs)

Fast Finetuning
===============

After post-training quantization, usually there is a small accuracy loss. If the accuracy loss is large, a fast finetuning approach based on the `AdaQuant Algorithm <https://arxiv.org/abs/2006.10518>`__ can be tried instead of quantization aware training. This approach uses a small unlabeled data to calibrate the activations and finetuning the weights. 

.. code-block::

   quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset, calib_steps=None, calib_batch_size=None, 
                                              include_fast_ft=True, fast_ft_epochs=10)
                                              
Fast finetuning related parameters are as follows:

- include_fast_ft: indicates whether to do fast finetuning or not.
- fast_ft_epochs: indicates the number of finetuning epochs for each layer.


***************************
Quantization Aware Training
***************************


An example of the Quantization Aware Training is available in the `Vitis Github <https://github.com/Xilinx/Vitis-AI/blob/v3.0/src/vai_quantizer/vai_q_tensorflow2.x/tensorflow_model_optimization/python/examples/quantization/keras/vitis/mnist_cnn_qat.py>`__ repo. 


The general steps are as follows:

1. Prepare the floating point model, training dataset, and training script.
2. Modify the training by using ``VitisQuantizer.get_qat_model`` to convert the model into a quantized model and then proceed to training/finetuning it:

.. code-block::

   model = tf.keras.models.load_model(‘float_model.h5’)
   # Call Vai_q_tensorflow2 api to create the quantize training model
   from tensorflow_model_optimization.quantization.keras import vitis_quantize
   quantizer = vitis_quantize.VitisQuantizer(model)
   qat_model = quantizer.get_qat_model(init_quant=True, 
                                       calib_dataset=calib_dataset)
                                       
   # Then run the training process with this qat_model to get the quantize finetuned model.
   # Compile the model
   qat_model.compile(optimizer= RMSprop(learning_rate=lr_schedule),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=keras.metrics.SparseTopKCategoricalAccuracy())
   
   # Start the training/finetunin
   qat_model.fit(train_dataset)

3. Call ``model.save()`` to save the trained model or use callbacks in ``model.fit()`` to save the model periodically.

.. code-block::
 
    # save model manually
    qat_model.save(‘trained_model.h5’)
    
    # save the model periodically during fit using callbacks
    qat_model.fit(train_dataset,
                  callbacks = [
                         keras.callbacks.ModelCheckpoint(
                         filepath=’./quantize_train/’
                         save_best_only=True,
                         monitor="sparse_categorical_accuracy",
                         verbose=1,
                  )])
                  
5. Convert the model to a deployable state by ``get_deploy_model`` API.

.. code-block::

   quantized_model = vitis_quantizer.get_deploy_model(qat_model)
   quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset, 
                                              output_format='onnx', 
                                              onnx_opset_version=11, 
                                              output_dir='./quantize_results',**kwargs)

..
  ------------

  #####################################
  License
  #####################################

 Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
