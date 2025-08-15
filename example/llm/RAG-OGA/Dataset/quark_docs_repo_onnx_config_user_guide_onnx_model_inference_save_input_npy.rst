Using ONNX Model Inference and Saving Input Data in NPY Format
==============================================================

This topic explains how to perform inference with an ONNX model using floating-point inputs and save the input data in `.npy` format. This approach facilitates data storage and reuse and can serve as a **calibration dataset** during model quantization, provided that the data adequately reflects the typical distribution of the model inputs.

Through an example, we demonstrate how to define a simple dataset class (`InputDataset`), perform inference using an ONNX model, and save input data in `.npy` format to support subsequent model quantization.

Detailed Code
-------------

.. code-block:: python

    import onnxruntime as ort
    import numpy as np
    import os
    from torch.utils.data import Dataset, DataLoader


    # A simple dataset with two inputs (`input1`, `input2`) and random tensors.
    # Users can customize data generation to match their model's needs.
    class InputDataset(Dataset):
        def __init__(self, num_samples):
            super(InputDataset, self).__init__()
            self.num_samples = num_samples
            self.input1 = [np.random.rand(3, 224, 224).astype(np.float32) for _ in range(num_samples)]
            self.input2 = [np.random.rand(10).astype(np.float32) for _ in range(num_samples)]
            self.labels = [np.random.randint(0, 2) for _ in range(num_samples)]

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return {
                "input1": self.input1[idx],
                "input2": self.input2[idx],
                "label": self.labels[idx]
            }


    dataset = InputDataset(num_samples=10)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    onnx_model_path = "path/to/your/float_model.onnx"
    session = ort.InferenceSession(onnx_model_path)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    enable_data_caching = True
    calibration_cache_dir = "calibration_data/"

    if enable_data_caching:
        for name in input_names:
            input_folder_path = os.path.join(calibration_cache_dir, name)
            os.makedirs(input_folder_path, exist_ok=True)

    for batch_idx, batch in enumerate(data_loader):
        input_feed = {}

        for name in input_names:
            input_data = batch[name].numpy()
            input_feed[name] = input_data

            # If `enable_data_caching` is True, save input data as `.npy` files by input name for each batch.
            if enable_data_caching:
                file_path = os.path.join(calibration_cache_dir, name, f"calib_{batch_idx+1:06d}.npy")
                np.save(file_path, input_data)
                print(f"Saved input data for {name} to {file_path}")

        outputs = session.run(output_names, input_feed)

        predictions = np.argmax(outputs[0], axis=1)

        print(f"Predictions for batch {batch_idx}: {predictions}")


The input data saved during ONNX inference can serve as a calibration dataset for model quantization. For instructions on how to use the saved NPY data as a calibration dataset, refer to :doc:`Calibration Data Path for AMD Quark Quantizer <./calibration_datasets>`. The output data format saved during inference is as follows:

For Single-Input Models
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

   calibration_data/
     calib_000001.npy
     calib_000002.npy
     calib_000003.npy
     calib_000004.npy
     calib_000005.npy
     ...

For Multi-Input Models
~~~~~~~~~~~~~~~~~~~~~~

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

