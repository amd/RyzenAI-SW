#!/bin/bash
pass_name=$1;
PassName=$(echo $pass_name | sed -r 's/(^|_)(\w)/\U\2/g')
PASS_NAME=$(echo $pass_name | tr '[a-z]' '[A-Z'])

if ! grep "add_subdirectory(vaip_pass_$pass_name)" CMakeLists.txt; then
    sed -i "/^# end passes.*/i add_subdirectory(vaip_pass_$pass_name)" CMakeLists.txt
fi

mkdir -p vaip_pass_$pass_name
mkdir -p vaip_pass_$pass_name/src

cat <<EOF >vaip_pass_$pass_name/CMakeLists.txt
# Copyright 2022 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
cmake_minimum_required(VERSION 3.5)
project(
  vaip
  VERSION 1.0.0
  LANGUAGES C CXX)

vai_add_library(NAME pass_$pass_name INCLUDE_DIR
                include SRCS src/pass_main.cpp)

target_link_libraries(pass_$pass_name
                      PRIVATE vaip::core vart::util glog::glog)
EOF


cat <<EOF >vaip_pass_$pass_name/src/pass_main.cpp
#include <glog/logging.h>
#include "vaip/vaip.hpp"
#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_$PASS_NAME, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(DEBUG_$PASS_NAME) >= n)
namespace {
using namespace vaip_core;
struct $PassName {
  static std::unique_ptr<Rule> create_rule() {
    auto builder = PatternBuilder();
    std::shared_ptr<Pattern> pat_xxx = builder.wildcard();
    return Rule::create_rule(
        pat_xxx,
        [=](onnxruntime::Graph* graph, binder_t& binder) -> bool {
          return false;
        });
  }
  $PassName(IPass& self): self_{self} {
  }
  void process(IPass& self, Graph& graph) {
    create_rule()->apply(&graph);
  }
  IPass& self_;
};
} // namespace
DEFINE_VAIP_PASS(${PassName}, ${pass_name})

EOF

echo ok.
