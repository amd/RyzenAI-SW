# Unit test guidelines

We use GTEST infrastructure to test C++ code. Please follow the below steps to build and execute the test cases.

C++ tests are compiled from the root directory of the repo.

## Build test suite
```
mkdir build
cd build
cmake ..
cmake --build . --config=Release
```

NOTE: Running tests without gtest_filter will put the board in a bad state. Please follow the steps below as is.

## Execute tests
```
.\Release\cpp_tests.exe --gtest_filter=QlinearTest*
.\Release\cpp_tests.exe --gtest_filter=dy*
.\Release\cpp_tests.exe --gtest_filter=Qlinear_2*
```

## Output snapshot
```
[----------] Global test environment tear-down
[==========] 5 tests from 1 test suite ran. (10652 ms total)
[  PASSED  ] 5 tests.

```
