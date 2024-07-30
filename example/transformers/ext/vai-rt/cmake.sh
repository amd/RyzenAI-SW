#!/bin/bash
#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -e

function    usage() {
    echo "./cmake.sh [options]"
    echo "    --help                    show help"
    echo "    --clean                   discard build dir before build"
    echo "    --ninja=yes|no            use ninja or not"
    echo "    --build-only              build only, will not install"
    echo "    --build-python            build python. if --pack is declared, will build conda package"
    echo "    --type[=TYPE]             build type. VAR {release, debug(default)}"
    echo "    --pack[=FORMAT]           enable packing and set package format. VAR {deb, rpm}"
    echo "    --build-dir[=DIR]         set customized build directory. default directory is ${build_dir_default}"
    echo "    --src-dir[=DIR]           set source directory. default directory is ${PWD}"
    echo "    --cmake-dir[=DIR]         where CMakeLists.txt exits relative to src_dir"
    echo "    --install-prefix[=PREFIX] set customized install prefix. default prefix is ${install_prefix_default}"
    echo "    --cmake-options[=OPTIONS] append more cmake options"
    echo "    --cmake-options-file[=FILENAME] read cmake options from a file."
    exit 0
}

# cmake args
declare -a args

args=(-DBUILD_SHARED_LIBS=ON)
args+=(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
args+=(-DCMAKE_BUILD_TYPE=Debug)
build_type=Debug
if which ninja >/dev/null; then
    use_ninja=yes
else
    use_ninja=no
fi
cmake_dir=.
# parse options
options=$(getopt -a -n 'parse-options' -o h \
		         -l help,ninja:,build-python,clean,build-only,type:,pack:,build-dir:,src-dir:,install-prefix:,cmake-options:,cmake-options-file:,cmake-dir:,home,user \
		         -- "$0" "$@")
[ $? -eq 0 ] || {
    echo "Failed to parse arguments! try --help"
    exit 1
}
eval set -- "$options"
while true; do
    case "$1" in
        -h | --help) show_help=true; usage; break;;
	    --clean) clean=true;;
	    --build-only) build_only=true;;
	    --type)
	        shift
	        case "$1" in
                release)
                    build_type=Release;
                    args+=(-DCMAKE_BUILD_TYPE=${build_type:="Release"});;
                debug)
                    build_type=Debug;
                    args+=(-DCMAKE_BUILD_TYPE=${build_type:="Debug"});;
		        *) echo "Invalid build type \"$1\"! try --help"; exit 1;;
	        esac
	        ;;
	    --pack)
	        shift
                build_package=true
                cpack_generator=
	        case "$1" in
		        deb)
                            cpack_generator=DEB;
                            args+=(-DCPACK_GENERATOR=${cpack_generator});;
		        rpm)
                            cpack_generator=RPM;
                            args+=(-DCPACK_GENERATOR=${cpack_generator});;
		        *) echo "Invalid pack format \"$1\"! try --help"; exit 1;;
	        esac
	        ;;
	    --build-dir) shift; build_dir=$1;;
	    --src-dir) shift; src_dir=$1;;
	    --install-prefix) shift; install_prefix=$1;;
	    --cmake-options) shift; args+=($1);;
        --ninja) shift; use_ninja=$1;;
        --build-python) args+=(-DBUILD_PYTHON=ON);;
	    --user) args+=(-DINSTALL_USER=ON);;
	    --home) args+=(-DINSTALL_HOME=ON);;
        --cmake-options-file)
            shift;
            while IFS= read -r arg; do args+=($arg); done < $1
            ;;
        --cmake-dir) shift; cmake_dir=$1;;
	    --) shift; break;;
    esac
    shift
done

if [ x$use_ninja == x"yes" ] ;then
   args+=(-G Ninja)
fi
os=Linux
if [ "$(uname)" == "Darwin" ]; then
    os=MacOs
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    os=Linux
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
    os=Windows
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
    os=Windows
fi
echo $os

src_dir_default=$(realpath $PWD)
[ -z ${src_dir:+x} ] && src_dir=${src_dir_default}

# detect target & set install prefix
if [ "$os" == "Windows" ]; then
    target_info=${os}.${build_type}
    prefix_path_default=$HOME/.local/${target_info}
    install_prefix_default=$HOME/.local/${target_info}/$(basename $src_dir)
    args+=(-DCMAKE_PREFIX_PATH=${prefix_path:="${prefix_path_default}"})
    args+=(-DBUILD_TEST=OFF)
elif [ -z ${OECORE_TARGET_SYSROOT:+x} ]; then
    echo "Native-platform building..."
    os=`lsb_release -a | grep "Distributor ID" | sed 's/^.*:\s*//'`
    os_version=`lsb_release -a | grep "Release" | sed 's/^.*:\s*//'`
    arch=`uname -p`
    target_info=${os}.${os_version}.${arch}.${build_type}
    install_prefix_default=$HOME/.local/${target_info}
    args+=(-DBUILD_TEST=ON)
else
    echo "Cross-platform building..."
    echo "Found target sysroot ${OECORE_TARGET_SYSROOT}"
    target_info=${OECORE_TARGET_OS}.${OECORE_SDK_VERSION}.${OECORE_TARGET_ARCH}.${build_type}
    install_prefix=${OECORE_TARGET_SYSROOT}/install/${build_type}
    args+=(-DCMAKE_TOOLCHAIN_FILE=${OECORE_NATIVE_SYSROOT}/usr/share/cmake/OEToolchainConfig.cmake)
    args+=(-DCMAKE_PREFIX_PATH=/install/${build_type})
    args+=(-DBUILD_TEST=ON)
fi
args+=(-DCMAKE_INSTALL_PREFIX=${install_prefix:="${install_prefix_default}"})

# set build dir
build_dir_default=$HOME/build/build.${target_info}/$(basename $src_dir)
[ -z ${build_dir:+x} ] && build_dir=${build_dir_default}

if [ x${clean:=false} == x"true" ] && [ -d ${build_dir} ];then
    echo "cleaning: rm -fr ${build_dir}"
    rm -fr "${build_dir}"
fi

mkdir -p ${build_dir}
cd -P ${build_dir}
echo "cd $PWD"
args+=(-B "$build_dir" -S "$src_dir/$cmake_dir")
echo cmake "${args[@]}"
cmake "${args[@]}"
if [ -z ${OECORE_TARGET_SYSROOT:+x} ]; then
    echo cp -av "${build_dir}/compile_commands.json" "$src_dir"
    cp -av "${build_dir}/compile_commands.json" "$src_dir" || true
fi

if [ "$os" == "Windows" ]; then
    cmake --build . -j $(nproc) --config ${build_type}
else
    cmake --build . -j $(nproc)
fi
${build_only:=false} || cmake --install .

if ! [ "$os" == "Windows" ]; then
    ${build_package:=false} && cpack -G ${cpack_generator}
fi

exit 0
