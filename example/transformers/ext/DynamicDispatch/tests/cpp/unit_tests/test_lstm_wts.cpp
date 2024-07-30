/*
 * Copyright � 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <ops/lstm/lstm.hpp>
#include <ops/ops_common/help_file.hpp>

#include "enable_perf.hpp"

static std::string GetTestSubFolderName(std::string prefix, int Mi0, int Mi1,
                                        int Mi2) {
  return prefix + "_" + std::to_string(Mi0) + "_" + std::to_string(Mi1) + "_" +
         std::to_string(Mi2);
}

static int CompareFileContents(const std::string &input_file_path,
                               const std::string &output_file_path) {
  // Open the input file and read the contents into a string
  std::ifstream input_file(input_file_path);
  if (!input_file.is_open()) {
    std::cerr << "Failed to open input file: " << input_file_path << std::endl;
    return -1;
  }

  std::ifstream temp1(input_file_path);
  std::ifstream temp2(output_file_path);
  int32_t val1, val2;
  int errcount = 0;
  while (temp1 >> val1 && temp2 >> val2) {
    if (val1 != val2) {
      std::cout << "ERROR: Expected: " << val1 << ", "
                << "Received: " << val2 << "\n";
      std::cout << val1 << " " << val2 << std::endl;
      errcount++;
    }
  }
  std::cout << "Errcount : " << errcount << std::endl;
  return errcount;
}

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_lstm(int Mi0, int Mi1, int Mi2, int Mo0, int Mo1, int Mo2,
              bool debug = false, const std::string &ifmDtype = "uint16",
              const std::string &weightDtype = "uint16",
              const std::string &ofmDtype = "uint16",
              const int modelNum = 320) {

  int err_count = 0;
  std::string fileName, testDataFolder, generatedFileName;

  testDataFolder = OpInterface::get_dod_base_dir() + "\\" + "tests" + "\\" +
                   "cpp" + "\\" + "unit_tests" + "\\" + "testDataMladf" + "\\" +
                   "lstm" + "_" + std::to_string(modelNum);

  std::vector<size_t> weightShape = {1, 1, 672512};
  fileName = testDataFolder + "\\" + "wts" + ".bin";

  std::vector<WgT> weight = OpsFusion::read_bin_file<WgT>(fileName);

  if (weight.size() != (weightShape[0] * weightShape[1] * weightShape[2])) {
    std::cout << "Weight parameter file is not proper. Expected size = "
              << (weightShape[0] * weightShape[1] * weightShape[2])
              << ", Actual Size = " << weight.size() << std::endl;
  }

  std::vector<size_t> ifmShape = {static_cast<size_t>(Mi0),
                                  static_cast<size_t>(Mi1),
                                  static_cast<size_t>(Mi2)};
  auto ifmSize = ifmShape[0] * ifmShape[1] * ifmShape[2];
  fileName = testDataFolder + "\\" + "ifm" + ".bin";
  std::vector<InT> ifm = OpsFusion::read_bin_file<InT>(fileName);
  if (ifm.size() != ifmSize) {
    std::cout << "ifm sample file is not proper. Expected size = " << ifmSize
              << ", Actual Size = " << ifm.size() << std::endl;
  }

  std::vector<size_t> ofmShape = {static_cast<size_t>(Mo0),
                                  static_cast<size_t>(Mo1),
                                  static_cast<size_t>(Mo2)};
  int32_t garbage_value = 0xAAAABBBB;
  std::vector<OuT> ofm(Mo0 * Mo1 * Mo2, garbage_value);

  ryzenai::lstm lstm_ =
      ryzenai::lstm<InT, WgT, OuT>(ifmDtype, weightDtype, ofmDtype, false);
  debug = true;
  lstm_.debug(debug);
  lstm_.set_params(modelNum, ifmShape, weightShape, ofmShape);

  std::vector<Tensor> const_Tensor;
  const_Tensor = {{weight.data(), weightShape, weightDtype}};
  lstm_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{ifm.data(), ifmShape, ifmDtype}};
  std::vector<Tensor> output_Tensor = {{ofm.data(), ofmShape, ofmDtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("Mi0=" << Mi0 << ", Mi1=" << Mi1 << ", Mo0=" << Mo0
                  << ", Mo1=" << Mo0);
  PROFILE_THIS(lstm_.execute(input_Tensor, output_Tensor));
#else
  lstm_.execute(input_Tensor, output_Tensor);
#endif
  generatedFileName = testDataFolder + "\\" + "ofmOut" + ".txt";
  write32BitHexTxtFile(generatedFileName, (OuT *)ofm.data(), ofm.size());

  fileName = testDataFolder + "\\" + "ofm32_ref.txt";
  if (CompareFileContents(fileName, generatedFileName)) {
    std::cout << "Error: ofm output doesn't match" << std::endl;
    err_count++;
  }
  return err_count;
}

/* PSO2-320 */
TEST(LstmTesta16w16c16, Kernel1) {
  int err_count = test_lstm<uint16_t, uint16_t, uint16_t>(
      80, 1, 64, 80, 2, 128, false, "uint16", "uint16", "uint16", 320);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* PSO2-640 */
TEST(LstmTesta16w16c16, Kernel2) {
  int err_count = test_lstm<uint16_t, uint16_t, uint16_t>(
      160, 1, 64, 160, 2, 128, false, "uint16", "uint16", "uint16", 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* PSO2-1280 */
TEST(LstmTesta16w16c16, Kernel3) {
  int err_count = test_lstm<uint16_t, uint16_t, uint16_t>(
      320, 1, 64, 320, 2, 128, false, "uint16", "uint16", "uint16", 1280);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
