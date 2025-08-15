Calibration Methods
===================

AMD Quark for PyTorch supports the following calibration methods:

-  **MinMax Calibration Method**: The ``MinMax`` calibration method for
   computing the quantization parameters based on the running min and
   max values. This method uses the tensor min/max statistics to compute
   the quantization parameters. The module records the running minimum
   and maximum of incoming tensors and uses these statistics to compute
   the quantization parameters.

-  **Percentile Calibration Method**: The ``Percentile`` calibration
   method, often used in robust scaling, involves scaling features based
   on percentile information from a static histogram, rather than using
   the absolute minimum and maximum values. This method is particularly
   useful for managing outliers in data.

-  **MSE Calibration Method**: The ``MSE`` (Mean Squared Error)
   calibration method refers to a method where calibration is performed
   by minimizing the mean squared error between the predicted outputs
   and the actual outputs. This method is typically used in regression
   contexts where the goal is to adjust model parameters or data
   transformations to reduce the average squared difference between
   estimated values and the true values. MSE calibration helps in
   refining model accuracy by fine-tuning predictions to be as close as
   possible to the real data points.
