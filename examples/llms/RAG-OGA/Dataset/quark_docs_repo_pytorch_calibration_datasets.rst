Adding Calibration Datasets
===========================

AMD Quark utilizes the `PyTorch Dataloader <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`__ for normalization during quantization calibration. The PyTorch Dataloader accepts instances of the PyTorch Dataset class as input. PyTorch datasets can be formatted as torch.Tensors, lists, or other types, provided they conform to specific rules. For the official guide on creating a Dataset, please refer to this `link <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`__.

We provide an example of quantizing large language models using typical datasets such as ``pileval`` and ``wikitext``. You can find the example in the path ``quark/examples/torch/language_modeling/data_preparation.py``. We provide a detailed example of how to set up a dataloader and how to convert a Hugging Face dataset to a PyTorch dataloader.

For large language models, the input data for PyTorch models is often represented as either a torch.Tensor or a dictionary. Here we provide three types of input PyTorch Dataset formats for the dataloader as examples. You can define your own PyTorch Dataset for the Dataloader, which must be compatible with PyTorch model input.

Dataloader with Dataset as torch.Tensor
---------------------------------------

If the Dataset format is torch.Tensor, the method of generating a PyTorch Dataloader is simple. For example:

.. code-block:: python

   input_tensor = torch.rand(128, 128)
   calib_dataloader = DataLoader(input_tensor, batch_size=4, shuffle=False)

Dataloader with List[Dict[str, torch.Tensor]] or List[torch.Tensor]
-------------------------------------------------------------------

If the Dataset format is a list of dictionaries or a list of tensors:

.. code-block:: python

   input_list = [{'input_ids': torch.rand(128, 128)}, {'input_ids': torch.rand(128, 128)}]
   calib_dataloader = DataLoader(input_list, batch_size=None, shuffle=False)

Dataloader with Dict[str, torch.Tensor]
---------------------------------------

If the Dataset format is a dictionary, you should define the function `collate_fn`, for example:

.. code-block:: python

   def my_collate_fn(blocks: List[Dict[str, List[List[str]]]]) -> Dict[str, torch.Tensor]:
       data_batch = {}
       data_batch["input_ids"] = torch.Tensor([block["input_ids"] for block in blocks])
       if device:
           data_batch["input_ids"] = data_batch["input_ids"].to(device)
       return data_batch

   input_dict = {'input_ids': torch.rand(128, 128)}
   calib_dataloader = DataLoader(input_dict, batch_size=4, collate_fn=my_collate_fn)

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
