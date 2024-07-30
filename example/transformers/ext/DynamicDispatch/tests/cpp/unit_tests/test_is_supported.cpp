// Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.

#include <array>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

#include "ops/op_builder.hpp"

#include "test_common.hpp"

struct OpConfig {
  std::string name;
  std::vector<std::string> types;
  std::map<std::string, std::any> attr;

  friend void PrintTo(const OpConfig &config, std::ostream *os) {
    *os << config.name << "(";
    for (const auto &type : config.types) {
      *os << type << ",";
    }
    *os << ")";
  }
};

class IsSupportedFixture : public testing::TestWithParam<OpConfig> {};
class IsNotSupportedFixture : public testing::TestWithParam<OpConfig> {};

TEST_P(IsSupportedFixture, IsSupported) {
  const auto &config = GetParam();
  EXPECT_TRUE(OpsFusion::OpBuilder::is_supported(config.name, config.types,
                                                 config.attr));
}

TEST_P(IsNotSupportedFixture, IsNotSupported) {
  const auto &config = GetParam();
  EXPECT_FALSE(OpsFusion::OpBuilder::is_supported(config.name, config.types,
                                                  config.attr));
}

// not exhaustive but could be
const std::array kSupportedConfigs{
    OpConfig{"MatMul", {"uint8", "uint8", "uint8"}, {}},
    OpConfig{"MatMul", {"uint16", "uint8", "uint16"}, {}},
    OpConfig{"LRN", {"bfloat16", "uint16", "uint8"}, {}},
    OpConfig{"LayerNorm", {"bfloat16", "uint16", "uint16"}, {}},
    OpConfig{"MatMulAdd", {"uint8", "uint8", "uint8"}, {}},
    OpConfig{"MatMulAdd", {"uint16", "uint8", "uint16"}, {}},
    OpConfig{"ADD", {"uint8", "", "bfloat16"}, {}},
    OpConfig{"Add", {"uint16", "", "bfloat16"}, {}},
    OpConfig{"MHAGRPB", {"uint8", "uint8", "uint8"}, {}},
    OpConfig{"MHAGRPB", {"uint16", "uint8", "uint16"}, {}},
    OpConfig{"MatMulAddGelu", {"uint8", "uint8", "uint8"}, {}},
    OpConfig{"MatMulAddGelu", {"uint16", "uint8", "uint16"}, {}},
    OpConfig{"square", {}, {}},
    OpConfig{"cube", {}, {}},
    OpConfig{"PM_LOAD", {}, {}},
};

const std::array kNotSupportedConfigs{
    OpConfig{"MatMul", {"float", "float", "float"}},
    OpConfig{"MatMul", {"uint32", "uint32", "uint32"}},
};

INSTANTIATE_TEST_CASE_P(IsSupported, IsSupportedFixture,
                        testing::ValuesIn(kSupportedConfigs));
INSTANTIATE_TEST_CASE_P(IsNotSupported, IsNotSupportedFixture,
                        testing::ValuesIn(kNotSupportedConfigs));
