## Prerequisite

**Note:** Ensure that you are following the instructions for model and dataset setup and compilation from [BF16 setup](../README.md)

In this section we deploy our model on Python for inferencing purpose.

## Model Deployment on CPU

```bash
   python predict.py

   Expected output:
    execution started on CPU
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
```

## Model Deployment on NPU

```bash
    python predict.py --ep npu
    Expected output:
    execution started on NPU
    WARNING: Logging before InitGoogleLogging() is written to STDERR
    I20250929 15:43:35.866091  1152 register_ssmlp.cpp:124] Registering Custom Operator: com.amd:SSMLP
    I20250929 15:43:35.866091  1152 register_matmulnbits.cpp:110] Registering Custom Operator: com.amd:MatMulNBits
    I20250929 15:43:36.021569  1152 vitisai_compile_model.cpp:1266] Vitis AI EP Load ONNX Model Success
    I20250929 15:43:36.021569  1152 vitisai_compile_model.cpp:1267] Graph Input Node Name/Shape (1)
    I20250929 15:43:36.021569  1152 vitisai_compile_model.cpp:1271]          input : [-1x3x32x32]
    I20250929 15:43:36.021569  1152 vitisai_compile_model.cpp:1277] Graph Output Node Name/Shape (1)
    I20250929 15:43:36.021569  1152 vitisai_compile_model.cpp:1281]          output : [-1x10]
    
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
```
