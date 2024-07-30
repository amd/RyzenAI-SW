/*
 * Copyright � 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <ops/lstm/lstm.hpp>
#include <ops/ops_common/help_file.hpp>

#include "enable_perf.hpp"

static std::string GetTestSubFolderName(std::string prefix, int modelNum,
                                        int Mi0, int Mi1, int Mi2) {
  return prefix + "_" + std::to_string(modelNum) + "_" + std::to_string(Mi0) +
         "_" + std::to_string(Mi1) + "_" + std::to_string(Mi2);
}

static int CompareFileContents(const std::string &input_file_path,
                               const std::string &output_file_path) {
  // Open the input file and read the contents into a string
  std::ifstream input_file(input_file_path);
  if (!input_file.is_open()) {
    std::cerr << "Failed to open input file: " << input_file_path << std::endl;
    return -1;
  }

  std::string input_file_contents((std::istreambuf_iterator<char>(input_file)),
                                  std::istreambuf_iterator<char>());
  input_file.close();

  // Open the output file and read the contents into a string
  std::ifstream output_file(output_file_path);
  if (!output_file.is_open()) {
    std::cerr << "Failed to open output file: " << output_file_path
              << std::endl;
    return -1;
  }

  std::string output_file_contents(
      (std::istreambuf_iterator<char>(output_file)),
      std::istreambuf_iterator<char>());
  output_file.close();

  // Compare the two file contents
  if (input_file_contents == output_file_contents) {
    return 0;
  } else {
    return -1;
  }
}

template <typename InT = uint16_t, typename WgT = uint16_t,
          typename OuT = uint16_t>
int test_lstm(int Mi0, int Mi1, int Mi2, int Mo0, int Mo1, int Mo2,
              bool debug = false, const std::string &ifmDtype = "uint16",
              const std::string &weightDtype = "uint16",
              const std::string &ofmDtype = "uint16", const int modelNum) {

  int err_count = 0;
  std::string fileName, testDataFolder, generatedFileName;

  testDataFolder = OpInterface::get_dod_base_dir() + "\\" + "tests" + "\\" +
                   "cpp" + "\\" + "unit_tests" + "\\" + "testDataMladf" + "\\" +
                   GetTestSubFolderName("lstm_", modelNum, Mi0, Mi1, Mi2);

  std::vector<size_t> w_weightShape = {2, 512, Mi2};
  fileName = testDataFolder + "\\" + "w_weight" + ".const";
  std::vector<WgT> w_weight = OpsFusion::read_bin_file<WgT>(fileName);
  if (w_weight.size() !=
      (w_weightShape[0] * w_weightShape[1] * w_weightShape[2])) {
    std::cout << "Weight parameter file is not proper. Expected size = "
              << (w_weightShape[0] * w_weightShape[1] * w_weightShape[2])
              << ", Actual Size = " << w_weight.size() << std::endl;
  }

  std::vector<size_t> r_weightShape = {2, 512, 128};
  fileName = testDataFolder + "\\" + "r_weight" + ".const";
  std::vector<WgT> r_weight = OpsFusion::read_bin_file<WgT>(fileName);
  if (r_weight.size() !=
      (r_weightShape[0] * r_weightShape[1] * r_weightShape[2])) {
    std::cout << "Weight parameter file is not proper. Expected size = "
              << (r_weightShape[0] * r_weightShape[1] * r_weightShape[2])
              << ", Actual Size = " << r_weight.size() << std::endl;
  }

  std::vector<size_t> b_weightShape = {2, 1024};
  fileName = testDataFolder + "\\" + "b_weight" + ".const";
  std::vector<WgT> b_weight = OpsFusion::read_bin_file<WgT>(fileName);
  if (b_weight.size() != (b_weightShape[0] * b_weightShape[1])) {
    std::cout << "Weight parameter file is not proper. Expected size = "
              << (b_weightShape[0] * b_weightShape[1])
              << ", Actual Size = " << b_weight.size() << std::endl;
  }

  std::vector<size_t> ifmShape = {Mi0, Mi1, Mi2};
  fileName = testDataFolder + "\\" + "ifm" + ".const";
  std::vector<InT> ifm = OpsFusion::read_bin_file<InT>(fileName);
  if (ifm.size() != ifmSize) {
    std::cout << "ifm sample file is not proper. Expected size = " << ifmSize
              << ", Actual Size = " << ifm.size() << std::endl;
  }

  std::vector<size_t> ofmShape = {Mo0, Mo1, Mo2};
  int32_t garbage_value = 0xAAAABBBB;
  std::vector<OuT> ofm(Mo0 * Mo1 * Mo2, garbage_value);

  ryzenai::lstm lstm_ =
      ryzenai::lstm<InT, WgT, OuT>(ifmDtype, weightDtype, ofmDtype, false);
  debug = true;
  lstm_.debug(debug);
  // lstm_.set_params(modelNameLowerCase);

  std::vector<Tensor> const_Tensor;
  const_Tensor = {{w_weight.data(), w_weightShape, weightDtype},
                  {r_weight.data(), r_weightShape, weightDtype},
                  {b_weight.data(), b_weightShape, weightDtype}};
  lstm_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{ifm.data(), ifmShape, ifmDtype}};
  std::vector<Tensor> output_Tensor = {{ofm.data(), ofmShape, ofmDtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("Mi0=" << Mi0 << ", Mi1=" << Mi1 << ", F0=" << F0 << ", F1=" << F1
                  << ", K=" << K << ", N=" << N << ", Mo0=" << Mo0
                  << ", Mo1=" << Mo0);
  PROFILE_THIS(conv_.execute(input_Tensor, output_Tensor));
#else
  conv_.execute(input_Tensor, output_Tensor);
#endif

  generatedFileName = testDataFolder + "\\" + "ofmOut" + ".txt";
  write32BitHexTxtFile(generatedFileName, (OuT *)ofm.data(), ofm.size());

  fileName = testDataFolder + "\\" + "ofmRef.txt";
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
TEST(LstmTesta16w16c16, Kernel2) {
  int err_count = test_lstm<uint16_t, uint16_t, uint16_t>(
      80, 1, 256, 80, 2, 128, false, "uint16", "uint16", "uint16", 320);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
/* PSO2-640 */
TEST(LstmTesta16w16c16, Kernel3) {
  int err_count = test_lstm<uint16_t, uint16_t, uint16_t>(
      160, 1, 64, 160, 2, 128, false, "uint16", "uint16", "uint16", 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
TEST(LstmTesta16w16c16, Kernel4) {
  int err_count = test_lstm<uint16_t, uint16_t, uint16_t>(
      160, 1, 256, 160, 2, 128, false, "uint16", "uint16", "uint16", 640);
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
