# Unit test guidelines

We use GTEST infrastructure to test C++ code. Please follow the below steps to build and execute the test cases.

C++ tests are compiled from the root directory of the repo during pip install ops\cpp step.

If new tests are written here, developer can compile the tests in this folder by following commands

## Compile
```
cd <transformers>

cmake -B build\
cmake --build build\ --config=Release
```

## Execute tests

NOTE: Running tests without gtest_filter will put the board in a bad state. Please follow the steps below as is.

### Tests for Phoenix
```
cd <transformers>\build\tests\cpp
.\Release\cpp_tests.exe --gtest_filter=QlinearTest*
.\Release\cpp_tests.exe --gtest_filter=dyqlinear*
.\Release\cpp_tests.exe --gtest_filter=Qlinear_2Testw8a8*
.\Release\cpp_tests.exe --gtest_filter=Qlinear_2Testw4a16*
.\Release\cpp_tests.exe --gtest_filter=Qlinear_2Testw3a16*
```

### Tests for Strix
```
cd <transformers>\build\tests\cpp
.\Release\cpp_tests.exe --gtest_filter=QlinearTest*
.\Release\cpp_tests.exe --gtest_filter=dyqlinear*
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