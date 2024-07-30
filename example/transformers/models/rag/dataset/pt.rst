##############################
Vitis AI Quantizer for PyTorch
##############################


*********************
Enabling Quantization
*********************

Ensure that the Vitis AI Quantizer for PyTorch is correctly installed. For more information, see the :ref:`installation instructions <install-pt-tf>`.

To enable the Vitis AI Quantizer for PyTorch, activate the conda environment in the Vitis AI Pytorch Docker container:

.. code-block::

     conda activate vitis-ai-pytorch
     
 
**************************
Post-Training Quantization
**************************

Post-Training Quantization requires the following files:

1. model.pth : Pre-trained PyTorch model, generally a .pth file.
2. model.py : A Python script including float model definition.
3. calibration dataset: A subset of the training dataset containing 100 to 1000 images.

A complete example of Post-Training Quantization is available in the `Vitis AI GitHub <https://github.com/Xilinx/Vitis-AI/blob/v3.0/src/vai_quantizer/vai_q_pytorch/example/resnet18_quant.py>`__ repo.


Vitis AI Quantization APIs
==========================

Vitis AI provides ``pytorch_nndct`` module with Quantization related APIs. 

1. Import the vai_q_pytorch module:

.. code-block:: 

    from pytorch_nndct.apis import torch_quantizer, dump_xmodel

2. Generate a quantizer with quantization needed input and get the converted model:

.. code-block::

   input = torch.randn([batch_size, 3, 224, 224])
   quantizer = torch_quantizer(quant_mode, model, (input))
   quant_model = quantizer.quant_model

3. Forward a neural network with the converted model:

.. code-block:: 

    acc1_gen, acc5_gen, loss_gen = evaluate(quant_model, val_loader, loss_fn)

4. Output the quantization result and deploy the model.

.. code-block:: 
 
    quantizer.export_quant_config()

5. Export the quantized model for deployment.

.. code-block::

    quantizer.export_onnx_model()
    
    
Quantization Output
===================

If this quantization command runs successfully, two important files are generated in the output directory ``./quantize_result``:

* ``<model>.onnx``: Quantized ONNX model
* ``Quant_info.json``: Quantization steps of tensors. Retain this file for evaluating quantized models.


Hardware-Aware Quantization
===========================

To enable hardware-aware quantization provide the ``target`` to the NPU specific archietecture as follows: 

.. code-block::

   quantizer = torch_quantizer(quant_mode=quant_mode,
                               module=model,
                               input_args=(input),
                               device=device,
                               quant_config_file=config_file,
                               target=target)
                               
The ``target`` of current version of NPU is ``AMD_AIE2_Nx4_Overlay_cfg0``


Partial Quantization
====================

Partial quantization can be enabled by using ``QuantStab`` and ``DeQuantStub`` operator from the ``pytorch_nndct`` library. In the following example, we are quantizing the layers ``subm0`` and ``subm2``, but not the ``subm1``: 

.. code-block::

   from pytorch_nndct.nn import QuantStub, DeQuantStub

   class WholeModule(torch.nn.module):
      def __init__(self,...):
         self.subm0 = ...
         self.subm1 = ...
         self.subm2 = ...

         # define QuantStub/DeQuantStub submodules
         self.quant = QuantStub()
         self.dequant = DeQuantStub()
         
      def forward(self, input):
          input = self.quant(input) # begin of part to be quantized
          output0 = self.subm0(input)
          output0 = self.dequant(output0) # end of part to be quantized

          output1 = self.subm1(output0)

          output1 = self.quant(output1) # begin of part to be quantized
          output2 = self.subm2(output1)
          output2 = self.dequant(output2) # end of part to be quantized


Fast Finetuning
===============

After post-training quantization, there is usually a small accuracy loss. If the accuracy loss is large, a fast-finetuning approach, which is based on the `AdaQuant Algorithm <https://arxiv.org/abs/2006.10518>`__, can be tried instead of the quantization aware training. The fast finetuning uses a small unlabeled data to calibrate the activations and finetuning the weights. 


.. code-block:: 

  # fast finetune model or load finetuned parameter before test
  
  if fast_finetune == True:
      ft_loader, _ = load_data(
                 subset_len=5120,
                 train=False,
                 batch_size=batch_size,
                 sample_method='random',
                 data_dir=args.data_dir,
                 model_name=model_name)
                 
  if quant_mode == 'calib':
      quantizer.fast_finetune(evaluate, (quant_model, ft_loader, loss_fn))
  elif quant_mode == 'test':
      quantizer.load_ft_param()


***************************
Quantization Aware Training
***************************

An example of Quantization Aware Training is available at the `Vitis Github <https://github.com/Xilinx/Vitis-AI/blob/v3.0/src/vai_quantizer/vai_q_pytorch/example/resnet18_qat.py>`__.

General approaches are:

1. If some non-module operations are needed to be quantized, convert them into module operations. For example, ResNet18 uses the ``+`` operator to add two tensors, which can be replaced by ``pytorch_nndct.nn.modules.functional.Add``. 

2. If some modules are called multiple times, uniqify them by defining multiple such modules and call them separately in the foward pass.

3. Insert ``QuantStub`` and ``DeQuantStub``. Any sub-network from QuantStub to DeQuantStub in a forward pass will be quantized. Multiple QuantStub-DeQuantStub pairs are allowed.

4. Create Quantizer module from the ``QatProcessor`` library:


.. code-block::

   from pytorch_nndct import QatProcessor
   qat_processor = QatProcessor(model, inputs, bitwidth=8)
   quantized_model = qat_processor.trainable_model()
   optimizer = torch.optim.Adam(
                     quantized_model.parameters(),
                     lr,
                      weight_decay=weight_decay)

5. For testing after the training, get the deployable model: 

.. code-block::

   output_dir = 'qat_result'
   deployable_model = qat_processor.to_deployable(quantized_model,output_dir)
   validate(val_loader, deployable_model, criterion, gpu)
   
6. Export ONNX model for prediction:

.. code-block::

     qat_processor.export_onnx_model()
     
..
  ------------

  #####################################
  License
  #####################################

 Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ . Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_ for the full license text and copyright notice.
