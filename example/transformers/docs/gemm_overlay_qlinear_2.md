# QLINEAR_2 Class

Documentation to use qlinear_2 class to offload matmul ops to IPU/AIE.

## qlinear_2 c++ class
qlinear_2 c++ class is the C++ wrapper class to offload tiled matrix multiplications to IPU/AIE.
Features:
1. Tiled matrix multiplication
2. Stationary weights
3. Uses lite runtime stack
4. python inerface

doxygen type documentation is available in the source code [here](../ops/cpp/qlinear_2/qlinear_2.hpp).

## Environment setup
Follow the instructions [here](../README.md) to setup the transformers repo.

```pip install ops\cpp``` will compile and install the python package with qlinear_2 class in the conda env

## C++ unit test cases
To compile and run C++ unit tests,
```
cd tests\cpp
mkdir build
cd build
cmake ..
cmake --build . --config=Release

# To execute all tests
.\Release\cpp_tests.exe --gtest_filter=Qlinear_2Test*

# To execute individual tests
.\Release\cpp_tests.exe --gtest_filter=Qlinear_2Test.<test_name>
```

source code for all c++ unit tests can be found [here.](../tests/cpp/test_qlinear_2.cpp)

## Python unit test cases
To run qlinear_2 python unit tests,
```
cd tests\python
pytest test_qlinear_2.py
```

## OPT1.3b model
Follow instructions [here](../models/llm/docs/opt.md) to execute OPT1.3b model

To execute OP1.3b model using qlinear_2 class, pass ```--qlinear_2``` to the run.py script. For example:
```
python run.py --model_name opt-1.3b --load --smoothquant --quant_mode ptdq --target aie --task decode --qlinear_2
```

## Profiling and logging
There are two profiling mechanisms available:
1. Profiling from python:

    a. Refer to instructions [here](../models/llm/docs/opt_w8a8.md#profiling-latency-phx) to use this feature.
2. C++ kernel level logging:

    a. This feature helps in the analysis of various functionalities performed at the C++ wrapper layer.

    b. Compile time flag ```RYZENAI_LOGGING``` has to be enabled [here](../ops/cpp/utils/logging.h)

Both profiling and logging features generate CSV files which can be analyzed offline.

## Debugging
Debug capability includes writing all matricies passed to the qlinear_2 class to files.

To use this feature, call the debug method after the qlinear_2 object is instantiated.
For ex:
```
qlin = ryzenai::qlinear_2(kernel_x_shape, kernel_y_shape);
qlin.debug(true);
```
Please see QlinearDebug test case in [test_qlinear_2.hpp](../tests/cpp/test_qlinear_2.cpp) for a complete example.

The matricies will be stored in "logs" directory with the following naming convention: ryzenai_qlinear2_<object_id>\_<execute_num>\_<matrix_name>.txt

## Tips & FAQ
1. Start with C++ unit tests. Qlinear_2Test.QlinearBasic is the simplest unit test case.
2. Ensure all C++ unit tests are passing
3. Enure all python unit tests are passing
4. Execute OPT1.3b model e2e after all unit tests are passing
5. To create pull requests, please follow the contribution guidelines [here.](../README.md#code-contribution-guidelines)
