## Prerequisite

**Note:** Ensure that you are following the instructions for model and dataset setup and compilation from [BF16 setup](../README.md)

In this section we deploy our model on Python for inferencing purpose.

## Model Deployment on CPU

```bash
   python predict.py

   Expected output:
    [Vitis AI EP] No. of Operators :   CPU     5  VAIML   119
    [Vitis AI EP] No. of Subgraphs : VAIML     1
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
    [Vitis AI EP] No. of Operators : VAIML   124
    [Vitis AI EP] No. of Subgraphs : VAIML     1
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