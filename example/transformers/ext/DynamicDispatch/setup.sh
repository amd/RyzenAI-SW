#!/bin/bash

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$(pwd)
export PATH=$PATH:$(pwd)/build/aie-rt/Release
export DOD_ROOT=$(pwd)
export DEVICE=stx
