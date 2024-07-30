#!/usr/bin/env bash

set -e

ROOT=$PWD

mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$ROOT/install -DCMAKE_PREFIX_PATH=$ROOT/install -DBUILD_THIRD_PARTY=ON -DBUILD_GRAPH_ENGINE=ON -DBUILD_SINGLE_LIBRARY=ON -B. -S..

cmake --build . --parallel 4 

# The problem with the above is that the install directory will include header files for all our deps like glog, protobuf, boost... 
# To avoid we can rerun the build and change the install prefix ans set build third party to off

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$ROOT/artifacts -DCMAKE_PREFIX_PATH=$ROOT/install -DBUILD_THIRD_PARTY=ON -DBUILD_GRAPH_ENGINE=ON -DBUILD_SINGLE_LIBRARY=ON -B. -S..

cmake --build . --parallel 4 
