#!/usr/bin/env bash
#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
set -e
set -x

root=vai_q_onnx
cmake_build_dir=${root}/operators/custom_ops/build
mkdir -p ${cmake_build_dir}

gen_files=${root}/gen_files
mkdir -p $gen_files

# cmake args
declare -a args

# parse options
options=$(getopt -a -n 'parse-options' -o h \
		 -l help,clean,fast,use_cuda,use_rocm,conda,type:,cmake-options: \
		 -- "$0" "$@")
[ $? -eq 0 ] || {
    echo "Failed to parse arguments! try --help"
    exit 1
}
eval set -- "$options"
while true; do
    case "$1" in
	-h | --help) show_help=true; break;;
	--clean) clean=true;;
	--fast) fast=true;;
	--use_cuda) use_cuda=true;;
	--use_rocm) use_rocm=true;;
	--conda) conda=true;;
	--type)
	    shift
	    case "$1" in
		release) build_type=Release;;
		debug) build_type=Debug;;
		*) echo "Invalid build type \"$1\"! try --help"; exit 1;;
	    esac
	    ;;
	--cmake-options) shift; args+=($1);;
	--) shift; break;;
    esac
    shift
done

#GIT_VERSION=$(git rev-parse --short HEAD)
#sed -e "s/COMMIT_SED_MASK/${GIT_VERSION}/" version.py > ${gen_files}/version.py

if [ ${fast:=false} == true ]; then
  pushd ${cmake_build_dir}
  echo "Skip cmake and run make directly"
  make -j
  popd
else
  os=`lsb_release -a | grep "Distributor ID" | sed 's/^.*:\s*//'`
  os_version=`lsb_release -a | grep "Release" | sed 's/^.*:\s*//'`
  arch=`uname -p`

  # set build type
  if [ ${build_type:=release} == "release" ]; then
      args+=(-DDEBUG=OFF)
      target_info=${os}.${os_version}.${arch}.Release
  else
      args+=(-DDEBUG=ON)
      target_info=${os}.${os_version}.${arch}.Debug
  fi

  if [ ${conda:=true} == true ]; then
    install_prefix=${CONDA_PREFIX}
  else
    install_prefix=$HOME/.local/${target_info}
  fi
  #echo "Set CMAKE_INSTALL_PREFIX=$install_prefix"
  #args+=(-DCMAKE_INSTALL_PREFIX=${install_prefix})

  # set cuda or rocm
  if [ ${use_cuda:=false} == true ]; then
    args+=(-DUSE_CUDA=ON)
    #sed -i 's/DEVICE_SED_MASK/cuda/' ${gen_files}/version.py
    sed -e 's/DEVICE_SED_MASK/cuda/' ${cmake_build_dir}/../__init__.py > ${gen_files}/__init__.py
    echo "Build vai_q_onnx with CUDA"
  elif [ ${use_rocm:=false} == true ]; then
    args+=(-DUSE_ROCM=ON)
    #sed -i 's/DEVICE_SED_MASK/rocm/' ${gen_files}/version.py
    sed -e 's/DEVICE_SED_MASK/rocm/' ${cmake_build_dir}/../__init__.py > ${gen_files}/__init__.py
    echo "Build vai_q_onnx with ROCM"
  else
    #sed -i 's/DEVICE_SED_MASK/cpu/' ${gen_files}/version.py
    sed -e 's/DEVICE_SED_MASK/cpu/' ${cmake_build_dir}/../__init__.py > ${gen_files}/__init__.py
    echo "Build vai_q_onnx with CPU"
  fi

  if [ ${show_help:=false} == true ]; then
    echo "./build.sh [options]"
    echo "    --help                    show help"
    echo "    --clean                   discard build dir before build"
    echo "    --fast                    run make directly"
    echo "    --use_cuda                build with cuda"
    echo "    --use_rocm                build with rocm"
    echo "    --conda                   search lib path in conda env"
    echo "    --type[=TYPE]             build type for library. VAR {release(default), debug}"
    echo "    --cmake-options[=OPTIONS] append more cmake options"
    exit 0
  else
    if ${clean:=false}; then
      echo "Discard build dir before build"
      rm -fr ./build/* ${cmake_build_dir}/* $gen_files/*.so
    fi
    pushd ${cmake_build_dir}
    echo cmake "${args[@]}" ..
    cmake "${args[@]}" ..
    make -j
    popd
  fi
fi

# copy gen files
cp ${cmake_build_dir}/lib*.so ${gen_files}/


bash ./pip_pkg.sh ./pkgs/ --release
