Quantization Strategies
=======================

AMD Quark for ONNX offers three distinct quantization strategies tailored to meet the requirements of various hardware backends:

-  **Post Training Weight-Only Quantization**: Quantizes the weights ahead of time, but the activations are not quantized (using the original float data type) during inference.

-  **Post Training Static Quantization**: Quantizes both the weights and activations in the model. To achieve the best results, this process necessitates calibration with a dataset that accurately represents the actual data, which allows for precise determination of the optimal quantization parameters for activations.

- **Post Training Dynamic Quantization**: Quantizes the weights ahead of time, while the activations are quantized dynamically at runtime. This method allows for a more flexible approach, especially when the activation distribution is not well-known or varies significantly during inference.

The strategies share the same API. You simply need to set the strategy through the quantization configuration, as demonstrated in the previous example. For more details about setting quantization configuration, refer to the "Configuring AMD Quark for ONNX" chapter.
