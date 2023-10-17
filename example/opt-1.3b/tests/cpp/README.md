# Unit test guidelines

We use GTEST infrastructure to test C++ code. Please follow the below steps to build and execute the test cases.

C++ tests are compiled from the root directory of the repo.

## Build test suite
```
cd <transfomers>
mkdir build
cd build
cmake ..
cmake --build . --config=Release
```

## Execute tests
```
cd tests\cpp\
.\Release\cpp_tests.exe
```

## Output snapshot
```
[----------] Global test environment tear-down
[==========] 5 tests from 1 test suite ran. (10652 ms total)
[  PASSED  ] 5 tests.

(ryzenai-transformers) C:\Users\Administrator\Desktop\tejuss\transformers\tests\cpp\build>
```