Calibration Methods
===================

AMD Quark for ONNX supports these types of calibration methods:

MinMax Calibration Method
~~~~~~~~~~~~~~~~~~~~~~~~~
The MinMax calibration method computes the quantization parameters based on the running minimum and maximum values. This method uses the tensor min/max statistics to compute the quantization parameters. The module records the running minimum and maximum of incoming tensors and uses these statistics to compute the quantization parameters.

Percentile Calibration Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Percentile calibration method, often used in robust scaling, involves scaling features based on percentile information from a static histogram, rather than using the absolute minimum and maximum values. This method is particularly useful for managing outliers in data.

MSE Calibration Method
~~~~~~~~~~~~~~~~~~~~~~
The MSE (Mean Squared Error) calibration method involves performing calibration by minimizing the mean squared error between the predicted outputs and the actual outputs. This method is typically used in regression contexts where the goal is to adjust model parameters or data transformations to reduce the average squared difference between estimated values and the true values. MSE calibration helps in refining model accuracy by fine-tuning predictions to be as close as possible to the real data points.

Entropy Calibration Method
~~~~~~~~~~~~~~~~~~~~~~~~~~
The Entropy calibration method determines the quantization parameters by considering the entropy algorithm of each tensor’s distribution.

NonOverflow Calibration Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The NonOverflow calibration method obtains the power-of-two quantization parameters for each tensor to ensure that min/max values do not overflow.
