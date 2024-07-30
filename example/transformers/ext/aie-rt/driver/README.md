# AI Engine Driver

The userspace library for ai-engine provides APIs to configure AIE registers.

## Hardware support

The user space library is supports both AIE, AIEML and AIE2IPU

| BRANCH     	| DEVICE         		|
|------------	|------------------------	|
| master     	| AIE, AIE ML and AIE2IPU	|
| master-aie 	| AIE only       		|

## IO Backends

The userspace library can be compiled with various IO backends. The library
can execute the low level register IO operations for the following backends:

1. Simulation (-D__AIESIM__): IO operation are executed by simulation functions
			      provided by aie-tools.
2. CDO generator(-D__AIECDO__): IO operations are executed by cdo functions
			      provided by aie-tools.
3. Baremetal(-D_AIEBAREMETAL__): IO operations are executed by baremetal
				 functions.
4. Debug: This is the default backend when other flags are not passed. The
	  debug backend prints the register address and corresponding values to
	  stdout.

## Compilation
### Compile library
	make -f Makefile.Linux
### Generate Documentation
	make -f Makefile.Linux doc-generate

HTML docs are populated in ./tmp/api/

PDF doc will be available at ./tmp/latex/refman.pdf

### CMake Alternative
	cmake -S . -B build
	cmake --build ./build --parallel

### Backend for Cmake
	-DSOCKET_BACKEND=on
	-DDEBUG_BACKEND=on

### Build Unit Tests with CMake
We use `CppUTest` for unit testing. Use CMake option `-DWITH_TESTS=ON` to turn
on building the unit testing.

You can specify to use your external cpputest directory with the following
option `-DWITH_TESTS=ON -DCPPUTEST_DIR=<compiled_cpputest_dir>`

Here is the repo for cpputest:
`https://github.com/cpputest/cpputest.git`

The testing executable will be in `<BUILD_DIR>/tests/`.
It will not run the tests by default during build. If you want to run the tests
in the end of the build, you can use CMake option `-DWITH_TESTS_EXEC=ON`.

Example:
	cd aie-rt/driver
	mkdir build; cd build
	cmake ../ -DWITH_TESTS=on -DDEBUG_BACKEND=on -DCPPUTEST_DIR=/path/to/cpputest
