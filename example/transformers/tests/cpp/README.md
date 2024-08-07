# Unit test guidelines

We use GTEST infrastructure to test C++ code. Please follow the below steps to build and execute the test cases.

C++ tests are compiled from the root directory of the repo during `pip install ops\cpp` step.

If new tests are written here, developer can compile the tests in this folder by following commands

## Compile
Navigate to transformers root directory
```
cd <transformers>
```
Build without RyzenAI Perf logging
```
cmake -B build\
cmake --build build\ --config=Release
```
Build with RyzenAI Perf logging
```
cmake -B build\ -DRYZENAI_PERF_=1
cmake --build build\ --config=Release
```

## Execute tests

NOTE: Running tests without gtest_filter will put the board in a bad state. Please follow the steps below as is.

### Tests for Phoenix
```
cd <transformers>\build\tests\cpp
.\Release\cpp_tests.exe --gtest_filter=Linear*
.\Release\cpp_tests.exe --gtest_filter=Qlinear_2Testw8a8*
.\Release\cpp_tests.exe --gtest_filter=Qlinear_2Testw4a16*
.\Release\cpp_tests.exe --gtest_filter=Qlinear_2Testw3a16*
```

### Tests for Strix
```
cd <transformers>\build\tests\cpp
.\Release\cpp_tests.exe --gtest_filter=Qlinear_2Testw8a8*
.\Release\cpp_tests.exe --gtest_filter=Qlinear_2Testw8a16*
.\Release\cpp_tests.exe --gtest_filter=Qlinear_2Testw4a16*
.\Release\cpp_tests.exe --gtest_filter=Qlinear_2Testw3a16*
```

## Output snapshot
```
[----------] Global test environment tear-down
[==========] 5 tests from 1 test suite ran. (10652 ms total)
[  PASSED  ] 5 tests.
```

## Summarize Performance

:pushpin: RyzenAI Perf logging needs to be enabled for generating the performance summary

Set the environment variable to run the test for multiple iteration (Ideally 5000)

*On Anaconda Command Prompt*
```
SET "NUM_EXECUTE_ITERATIONS=5000"
```
*On Anaconda Powershell*
```
$env:NUM_EXECUTE_ITERATIONS="5000"
```
Follow the above steps to run tests and run the python script by passing the RyzenAI log file as an argument.

```
python summarize_perf.py --file <Path to the RyzenAI log file>
```
This script will generate a csv file with the summary of the performance by calculating the median latencies for each unique run.

## Compare Performance

 This section is for comparing the performance of the current branch verses a golden.
 Follow the above steps to generate the performance data using the 'summarize_perf.py' script.

```
 python perf_compare.py --new_perf <path to the csv file generated by summarize_perf.py> --golden <path to the golden data>
```

This script generates a csv file with the percentage improvement in execute latency compared to golden for each shape in the golden file and outputs the number of shapes failed, if any.

:pushpin: Performance degradation of more than 1 percent is considered as a fail.

# Benchmark test guidelines

We use [Google benchmark](https://github.com/google/benchmark) infrastructure to run microbenchmarks. It is recommended to run benchmarks and check their results when your changes may affect performance of other modules, in particular, kernels.

## Compile

The same compile commands above build the benchmark executables.

## Execute benchmark tests
```
.\Release\cpp_bm_tests.exe
```

## Output snapshot
```
Running C:\repos\transformers\build\tests\cpp\Release\cpp_bm_tests.exe
Run on (16 X 3992 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 1024 KiB (x8)
  L3 Unified 16384 KiB (x1)
-------------------------------------------------------------------------------
Benchmark                                     Time             CPU   Iterations
-------------------------------------------------------------------------------
BM_Create/2x3x4/iterations:100         13572750 ns       156250 ns          100
BM_Run/2x3x4                             614736 ns         6250 ns        10000
BM_Create/7x2000x3000/iterations:100  137502116 ns    124062500 ns          100
BM_Run/7x2000x3000                      2998601 ns       109375 ns         1000
BM_Create/4x2048x2048/iterations:100   93918158 ns     78437500 ns          100
BM_Run/7x2000x3000                       884573 ns        42188 ns        10000
```
