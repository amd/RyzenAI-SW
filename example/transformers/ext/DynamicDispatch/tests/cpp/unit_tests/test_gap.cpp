/*
 * Copyright � 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <ops/gap/gap.hpp>
#include <ops/ops_common/help_file.hpp>

#include "enable_perf.hpp"

static std::string GetTestSubFolderName(std::string prefix, int zeroPoint,
                                        int K, int Mi0, int Mi1) {
  return prefix + "_" + std::to_string(zeroPoint) + "_" + std::to_string(Mi0) +
         "_" + std::to_string(Mi1) + "_" + std::to_string(K);
}

static std::string GetParamKey(std::string prefix, int64_t zp, int64_t K,
                               int64_t Mi0, int64_t Mi1) {
  return prefix + "_" + std::to_string(zp) + "_" + std::to_string(Mi0) + "_" +
         std::to_string(Mi1) + "_" + std::to_string(K);
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

/* Gap has only 3 parameters
Input : K x Mi0 x Mi1
Filter : No Filter, only layer parameters are required
Output : K x 1 x 1 */
template <typename InT = uint16_t, typename OuT = uint16_t>
int test_gap(int Mi0, int Mi1, int K, bool debug = false,
             const std::string &ifmDtype = "uint16",
             const std::string &ofmDtype = "uint16", int zeroPoint = 1,
             const std::string &modelName = "psi") {
  int err_count = 0;
  std::string fileName, testDataFolder, generatedFileName;
  std::string modelNameLowerCase = modelName;

  std::transform(modelNameLowerCase.begin(), modelNameLowerCase.end(),
                 modelNameLowerCase.begin(), ::tolower);
  testDataFolder =
      OpInterface::get_dod_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
      "unit_tests" + "\\" + "testDataMladf" + "\\" +
      GetTestSubFolderName(modelNameLowerCase, zeroPoint, K, Mi0, Mi1);

  std::vector<size_t> ifmShape = {
      1, static_cast<size_t>(Mi0) * static_cast<size_t>(Mi1),
      static_cast<size_t>(K)};
  size_t ifmSize = 1 * Mi0 * Mi1 * K;
  fileName = testDataFolder + "\\" + "ifm" + ".const";
  std::vector<InT> ifm = OpsFusion::read_bin_file<InT>(fileName);
  if (ifm.size() != ifmSize) {
    std::cout << "ifm sample file is not proper. Expected size = " << ifmSize
              << ", Actual Size = " << ifm.size() << std::endl;
  }

  std::vector<size_t> ofmShape = {1, static_cast<size_t>(K)};
  int32_t garbage_value = 0xAABB;
  std::vector<OuT> ofm(K, garbage_value);

  std::map<std::string, std::any> attr;
  attr["input_shape"] = std::vector<int>{1, K, Mi0 * Mi1, 1};
  attr["output_shape"] = std::vector<int>{1, K, 1, 1};
  attr["zero_point"] = std::vector<int>{zeroPoint};

  ryzenai::gap gap_ = ryzenai::gap<InT, OuT>(ifmDtype, ofmDtype, false, attr);
  debug = true;
  gap_.debug(debug);
  gap_.set_params(modelNameLowerCase);

  std::vector<Tensor> const_Tensor;
  gap_.initialize_const_params(const_Tensor);

  if (debug == true) {
    fileName = testDataFolder + "\\" + "wtsRef.txt";
    std::string weightGeneratedFolder =
        OpInterface::get_dod_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
        "unit_tests" + "\\" + "testDataMladf" + "\\" + "GeneratedWeights";
    generatedFileName =
        weightGeneratedFolder + "\\" +
        GetParamKey("wtsGenerated", zeroPoint, K, (Mi0 * Mi1), 1) + ".txt";
    if (CompareFileContents(fileName, generatedFileName)) {
      std::cout << "Error: the weight generated are not proper" << std::endl;
      err_count++;
    }
  }

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{ifm.data(), ifmShape, ifmDtype}};
  std::vector<Tensor> output_Tensor = {{ofm.data(), ofmShape, ofmDtype}};

#ifdef UNIT_TEST_PERF
  // LOG_THIS("Mi0=" << Mi0 << ", Mi1=" << Mi1 << ", F0=" << F0 << ", F1=" << F1
  //                 << ", K=" << K << ", N=" << N << ", Mo0=" << Mo0
  //                 << ", Mo1=" << Mo0);
  PROFILE_THIS(gap_.execute(input_Tensor, output_Tensor));
#else
  gap_.execute(input_Tensor, output_Tensor);
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

#if 1
TEST(GapTesta16w8c16_, Kernel53) {
  int err_count = test_gap<uint16_t, uint16_t>(49, 1, 1024, false, "uint16",
                                               "uint16", 35881, "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
#endif
