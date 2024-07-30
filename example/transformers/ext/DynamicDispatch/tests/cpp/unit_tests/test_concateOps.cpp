/*
 * Copyright � 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <ops/concateOps/concateOps.hpp>
#include <ops/ops_common/help_file.hpp>

#include "enable_perf.hpp"

static std::string GetTestSubFolderName(std::string prefix, int zeroPoint,
                                        int K, int N, int F0) {
  return prefix + "_" + std::to_string(zeroPoint) + "_" + std::to_string(F0) +
         "_" + std::to_string(K) + "_" + std::to_string(N);
}

static std::string GetParamKey(std::string prefix, int64_t zp, int64_t K,
                               int64_t N, int64_t F0) {
  return prefix + "_" + std::to_string(zp) + "_" + std::to_string(F0) + "_" +
         std::to_string(K) + "_" + std::to_string(N);
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

// Helper function to read ifm data from a file
static Tensor
GetOfmTensor(const std::tuple<int, int, int, int, int, int, int, int, int, bool,
                              std::string, std::string, std::string, int,
                              std::string, std::string> &params,
             std::vector<std::vector<uint16_t>> &ofmDataContainer) {
  /*Extract and set parameters */
  int Mi0 = std::get<0>(params);
  int Mi1 = std::get<1>(params);
  int F0 = std::get<2>(params);
  int F1 = std::get<3>(params);
  int K = std::get<4>(params);
  int N = std::get<5>(params);
  int Mo0 = std::get<6>(params);
  int Mo1 = std::get<7>(params);
  int groupId = std::get<8>(params);
  bool debug = std::get<9>(params);
  int zeroPoint = std::get<13>(params);
  const std::string &opOfmDtype = std::get<12>(params);
  const std::string &opModelName = std::get<14>(params);
  const std::string &operatorName = std::get<15>(params);

  std::vector<size_t> ofmShape = {
      1, (static_cast<size_t>(Mo0) * static_cast<size_t>(Mo1)),
      static_cast<size_t>(N)};
  int32_t garbage_value = 0xAAAABBBB;
  /* Sandip TBD : Need to replace vector type from opOfmDtype */
  std::vector<uint16_t> ofm(N * (Mo0) * (Mo1), garbage_value);

  /* Add the ofm data to the ofmDataContainer */
  ofmDataContainer.push_back(std::move(ofm));

  /* Get a reference to the last element of the weightDataContainer which
  contains the weight data just added */
  std::vector<uint16_t> &ofm_data_ref = ofmDataContainer.back();

  return {ofm_data_ref.data(), ofmShape, opOfmDtype};
}

// Helper function to read ifm data from a file
static Tensor
GetIfmTensor(const std::tuple<int, int, int, int, int, int, int, int, int, bool,
                              std::string, std::string, std::string, int,
                              std::string, std::string> &params,
             std::vector<std::vector<uint16_t>> &ifmDataContainer) {
  /*Extract and set parameters */
  int Mi0 = std::get<0>(params);
  int Mi1 = std::get<1>(params);
  int F0 = std::get<2>(params);
  int F1 = std::get<3>(params);
  int K = std::get<4>(params);
  int N = std::get<5>(params);
  int Mo0 = std::get<6>(params);
  int Mo1 = std::get<7>(params);
  int groupId = std::get<8>(params);
  bool debug = std::get<9>(params);
  int zeroPoint = std::get<13>(params);
  const std::string &opIfmDtype = std::get<10>(params);
  const std::string &opModelName = std::get<14>(params);
  const std::string &operatorName = std::get<15>(params);

  std::string opModelNameLowerCase = opModelName;
  std::transform(opModelNameLowerCase.begin(), opModelNameLowerCase.end(),
                 opModelNameLowerCase.begin(), ::tolower);
  std::string testDataFolder =
      OpInterface::get_dod_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
      "unit_tests" + "\\" + "testDataMladf" + "\\" +
      GetTestSubFolderName(opModelNameLowerCase, zeroPoint, K, N, F0);

  std::vector<size_t> ifmShape;
  size_t ifmSize;
  if ((zeroPoint == 29172) && (opModelNameLowerCase == "psi")) {
    /* This is a specific case required for layer 1 of PSI model only */
    ifmShape = {1, (static_cast<size_t>(Mi0) * static_cast<size_t>(Mi1)),
                static_cast<size_t>(K + 1)}; // activate
    ifmSize = (K + 1) * Mi0 * Mi1;
  } else {
    ifmShape = {1, (static_cast<size_t>(Mi0) * static_cast<size_t>(Mi1)),
                static_cast<size_t>(K)}; // activate
    ifmSize = K * Mi0 * Mi1;
  }
  std::string fileName = testDataFolder + "\\" + "ifm" + ".const";
  /* Sandip TBD : Need to replace vector type from opIfmDtype */
  std::vector<uint16_t> ifm = OpsFusion::read_bin_file<uint16_t>(fileName);
  if (ifm.size() != ifmSize) {
    std::cout << "ifm sample file is not proper. Expected size = " << ifmSize
              << ", Actual Size = " << ifm.size() << std::endl;
  }
  /* Add the ifm data to the ifmDataContainer */
  ifmDataContainer.push_back(std::move(ifm));
  /* Get a reference to the last element of the ifmDataContainer which contains
   * the ifm data just added */
  std::vector<uint16_t> &ifm_data_ref = ifmDataContainer.back();

  return {ifm_data_ref.data(), ifmShape, opIfmDtype};
}

// Helper function to read weight data from a file
static Tensor
read_weight_data(const std::tuple<int, int, int, int, int, int, int, int, int,
                                  bool, std::string, std::string, std::string,
                                  int, std::string, std::string> &params,
                 std::vector<std::vector<uint16_t>> &weightDataContainer) {
  /* Extract and set parameters */
  int K = std::get<4>(params);
  int N = std::get<5>(params);
  int F0 = std::get<2>(params);
  int F1 = std::get<3>(params);
  int groupId = std::get<8>(params);
  const std::string &opWtsDtype = std::get<11>(params);
  int zeroPoint = std::get<13>(params);
  const std::string &opModelName = std::get<14>(params);

  /* Read weight data from a file */
  std::string opModelNameLowerCase = opModelName;
  std::transform(opModelNameLowerCase.begin(), opModelNameLowerCase.end(),
                 opModelNameLowerCase.begin(), ::tolower);
  std::string testDataFolder =
      OpInterface::get_dod_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
      "unit_tests" + "\\" + "testDataMladf" + "\\" +
      GetTestSubFolderName(opModelNameLowerCase, zeroPoint, K, N, F0);
  std::vector<size_t> weightShape =
      (groupId == 1)
          ? std::vector<size_t>{static_cast<size_t>(N), static_cast<size_t>(K),
                                static_cast<size_t>(F0),
                                static_cast<size_t>(F1)}
          : std::vector<size_t>{static_cast<size_t>(N), 1,
                                static_cast<size_t>(F0),
                                static_cast<size_t>(F1)};
  std::string fileName = testDataFolder + "\\" + "weight" + ".const";
  /* Sandip TBD: uint16_t is hardcoded. Should be modified with the use of
   * opWtsDtype */
  std::vector<uint16_t> weight = OpsFusion::read_bin_file<uint16_t>(fileName);
  if (weight.size() !=
      (weightShape[0] * weightShape[1] * weightShape[2] * weightShape[3])) {
    std::cout << "Weight parameter file is not proper. Expected size = "
              << (weightShape[0] * weightShape[1] * weightShape[2] *
                  weightShape[3])
              << ", Actual Size = " << weight.size() << std::endl;
  }
  /* Add the weight data to the weightDataContainer */
  weightDataContainer.push_back(std::move(weight));
  /* Get a reference to the last element of the weightDataContainer which
  contains the weight data just added */
  std::vector<uint16_t> &weight_data_ref = weightDataContainer.back();
  return {weight_data_ref.data(), weightShape, opWtsDtype};
}

/* Helper function to get attributes for each operator test is executing */
static std::map<std::string, std::any>
GetAttr(const std::tuple<int, int, int, int, int, int, int, int, int, bool,
                         std::string, std::string, std::string, int,
                         std::string, std::string> &params) {
  /* Extract and set parameters */
  int K = std::get<4>(params);
  int N = std::get<5>(params);
  int F0 = std::get<2>(params);
  int F1 = std::get<3>(params);
  int groupId = std::get<8>(params);
  const std::string &opIfmDtype = std::get<10>(params);
  const std::string &opWtsDtype = std::get<11>(params);
  const std::string &opOfmDtype = std::get<12>(params);
  int zeroPoint = std::get<13>(params);
  const std::string &operatorType = std::get<15>(params);

  /* Store Attributes */
  std::map<std::string, std::any> attr;
  attr["opType"] = operatorType;
  attr["opIfmDtype"] = opIfmDtype;
  attr["opWtsDtype"] = opWtsDtype;
  attr["opOfmDtype"] = opOfmDtype;

  attr["group"] = std::vector<int>{groupId};
  attr["input_shape"] =
      std::vector<int>{1, K, std::get<0>(params), std::get<1>(params)};
  attr["output_shape"] =
      std::vector<int>{1, N, std::get<6>(params), std::get<7>(params)};
  int weightThirdDim = (groupId == 1) ? K : 1;
  attr["weight_shape"] = std::vector<int>{N, weightThirdDim, F0, F1};
  attr["zero_point"] = std::vector<int>{zeroPoint};

  return attr;
}

static std::string GetParamKey(std::string prefix, int64_t graphId,
                               int64_t inChannels, int64_t outChannels) {
  return prefix + "_" + std::to_string(graphId) + "_" +
         std::to_string(inChannels) + "_" + std::to_string(outChannels);
}

template <typename InT = uint16_t, typename OuT = uint16_t>
static int test_concatenate(
    const std::vector<std::tuple<int, int, int, int, int, int, int, int, int,
                                 bool, std::string, std::string, std::string,
                                 int, std::string, std::string>> &paramsVec,
    const int graphId = 320, const int inChannels = 8,
    const int outChannels = 16, const std::string &modelName = "pso2") {
  int total_err_count = 0;
  std::vector<std::map<std::string, std::any>> attributesVec;
  std::vector<Tensor> const_Tensor;
  std::vector<Tensor> input_Tensor;
  std::vector<Tensor> output_Tensor;
  std::vector<std::vector<uint16_t>> weightDataContainer;
  std::vector<std::vector<uint16_t>> ifmDataContainer;
  std::vector<std::vector<uint16_t>> ofmDataContainer;
  bool debugFlag = true;

  for (const auto &params : paramsVec) {
    int err_count = 0;
    attributesVec.push_back(GetAttr(params));
    /* Call the helper function to read weight data with tuple as the argument
     */
    const_Tensor.push_back(read_weight_data(params, weightDataContainer));
  }

  if (!paramsVec.empty()) {
    const auto &first_params = paramsVec.front();
    const auto &last_params = paramsVec.back();

    input_Tensor.push_back(GetIfmTensor(first_params, ifmDataContainer));
    output_Tensor.push_back(GetOfmTensor(last_params, ofmDataContainer));
  }
  ryzenai::concateOps concatenate_ = ryzenai::concateOps<InT, OuT>(
      graphId, inChannels, outChannels, attributesVec);
  concatenate_.set_params(modelName, debugFlag);
  concatenate_.get_buffer_reqs(input_Tensor, output_Tensor);
  concatenate_.initialize_const_params(const_Tensor);

  std::string testDataFolder =
      OpInterface::get_dod_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
      "unit_tests" + "\\" + "testDataMladf" + "\\" +
      GetParamKey("concatenate", graphId, inChannels, outChannels);
  std::string weightGeneratedFolder =
      OpInterface::get_dod_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
      "unit_tests" + "\\" + "testDataMladf" + "\\" + "GeneratedWeights";
  std::string fileName, generatedFileName;
  if (debugFlag) {
    fileName = testDataFolder + "\\" + "wtsRef.txt";
    generatedFileName =
        weightGeneratedFolder + "\\" +
        GetParamKey("wtsGenerated", graphId, inChannels, outChannels) + ".txt";
    if (CompareFileContents(fileName, generatedFileName)) {
      std::cout << "Error: the weight generated is not proper" << std::endl;
      total_err_count++;
    }
  }
  concatenate_.execute(input_Tensor, output_Tensor);

  fileName = testDataFolder + "\\" + "ofmRef.bin";
  generatedFileName = weightGeneratedFolder + "\\" + "ofmOut.bin";
  write_bin_file(generatedFileName, (char *)output_Tensor.at(0).data,
                 output_Tensor.at(0).shape[0] * output_Tensor.at(0).shape[1] *
                     output_Tensor.at(0).shape[2] * 2);
  if (CompareFileContents(fileName, generatedFileName)) {
    std::cout << "Error: the ofm generated is not proper" << std::endl;
    total_err_count++;
  }
  return total_err_count;
}

#if 0
TEST(ConcatenateTesta16w16c16_, PsoTest) {
  std::vector<
      std::tuple<int, int, int, int, int, int, int, int, int, bool, std::string,
                 std::string, std::string, int, std::string, std::string>>
      paramsVec = {
          std::make_tuple(60, 320, 3, 3, 8, 16, 32, 160, 1, false, "uint16",
                          "uint16", "uint16", 40597, "PSO2", "conv"),
          std::make_tuple(32, 160, 3, 3, 16, 32, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 32705, "PSO2", "conv"),
          std::make_tuple(16, 80, 1, 1, 32, 16, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 36423, "PSO2", "conv"),
          std::make_tuple(16, 80, 3, 3, 16, 32, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 33409, "PSO2", "conv"),
          std::make_tuple(16, 80, 1, 1, 32, 128, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 29586, "PSO2", "conv"),
          std::make_tuple(16, 80, 1, 1, 128, 16, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 25513, "PSO2", "conv"),
          std::make_tuple(16, 80, 3, 3, 16, 32, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 31530, "PSO2", "conv"),
          std::make_tuple(16, 80, 1, 1, 32, 128, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 32591, "PSO2", "conv"),
          std::make_tuple(8, 80, 1, 1, 128, 32, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 31990, "PSO2", "conv"),
          std::make_tuple(8, 80, 3, 3, 32, 48, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 36064, "PSO2", "conv"),
          std::make_tuple(8, 80, 1, 1, 48, 256, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 35326, "PSO2", "conv"),
          std::make_tuple(8, 80, 1, 1, 256, 32, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 34702, "PSO2", "conv"),
          std::make_tuple(8, 80, 3, 3, 32, 48, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 30051, "PSO2", "conv"),
          std::make_tuple(8, 80, 1, 1, 48, 256, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 35719, "PSO2", "conv"),
          std::make_tuple(4, 80, 1, 1, 256, 64, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 26536, "PSO2", "conv"),
          std::make_tuple(4, 80, 3, 3, 64, 80, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 22444, "PSO2", "conv"),
          std::make_tuple(4, 80, 1, 1, 80, 512, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 32234, "PSO2", "conv"),
          std::make_tuple(4, 80, 1, 1, 512, 64, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 33891, "PSO2", "conv"),
          std::make_tuple(4, 80, 3, 3, 64, 80, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 33497, "PSO2", "conv"),
          std::make_tuple(4, 80, 1, 1, 80, 512, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 31960, "PSO2", "conv"),
          std::make_tuple(4, 80, 1, 1, 512, 16, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 33774, "PSO2", "conv")};

  int err_count =
      test_concatenate<uint16_t, uint16_t>(paramsVec, 320, 8, 16, "pso2");
}
#endif
