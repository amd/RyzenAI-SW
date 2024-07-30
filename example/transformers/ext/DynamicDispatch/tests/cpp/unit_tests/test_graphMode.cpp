/*
 * Copyright � 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <ops/concateOps/concateOps.hpp>
#include <ops/ops_common/help_file.hpp>

#include "enable_perf.hpp"

/* ParamsStruct to store extracted parameters */
struct ParamsStruct {
  int Mi0, Mi1, F0, F1, K, N, Mo0, Mo1, groupId, width;
  bool debug, useTxnBinWithZp;
  std::string opIfmDtype, opWtsDtype, opOfmDtype;
  int zeroPoint;
  std::string opModelName, operatorName;
};

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
  /* Open the input file and read the contents into a string */
  std::ifstream input_file(input_file_path);
  if (!input_file.is_open()) {
    std::cerr << "Failed to open input file: " << input_file_path << std::endl;
    return -1;
  }

  /* Open the output file and read the contents into a string */
  std::ifstream output_file(output_file_path);
  if (!output_file.is_open()) {
    std::cerr << "Failed to open output file: " << output_file_path
              << std::endl;
    return -1;
  }

  /* Compare the two file contents line by line */
  std::string input_line, output_line;
  std::size_t line_number = 1;
  bool files_are_different = false;

  while (std::getline(input_file, input_line) &&
         std::getline(output_file, output_line)) {
    if (input_line != output_line) {
      std::cout << "Mismatch at line " << line_number << ":" << std::endl;
      std::cout << "  Input file:  " << input_line << std::endl;
      std::cout << "  Output file: " << output_line << std::endl;
      files_are_different = true;
    }
    ++line_number;
  }

  if (input_file.bad() || output_file.bad()) {
    std::cerr << "Error while reading the files." << std::endl;
    return -1;
  }

  if (std::getline(input_file, input_line) ||
      std::getline(output_file, output_line)) {
    std::cerr << "Files have different number of lines." << std::endl;
    return -1;
  }

  input_file.close();
  output_file.close();

  if (files_are_different) {
    return -1;
  } else {
    return 0;
  }
}

/* Helper function to convert a string to lowercase */
static std::string toLowercase(const std::string &input) {
  std::string lowercase = input;
  std::transform(lowercase.begin(), lowercase.end(), lowercase.begin(),
                 ::tolower);
  return lowercase;
}

/* Helper function to extract parameters */
static ParamsStruct ExtractParameters(
    const std::tuple<int, int, int, int, int, int, int, int, int, bool,
                     std::string, std::string, std::string, int, std::string,
                     bool, int, std::string> &params) {
  ParamsStruct ps;
  ps.Mi0 = std::get<0>(params);
  ps.Mi1 = std::get<1>(params);
  ps.F0 = std::get<2>(params);
  ps.F1 = std::get<3>(params);
  ps.K = std::get<4>(params);
  ps.N = std::get<5>(params);
  ps.Mo0 = std::get<6>(params);
  ps.Mo1 = std::get<7>(params);
  ps.groupId = std::get<8>(params);
  ps.debug = std::get<9>(params);
  ps.opIfmDtype = toLowercase(std::get<10>(params));
  ps.opWtsDtype = toLowercase(std::get<11>(params));
  ps.opOfmDtype = toLowercase(std::get<12>(params));
  ps.zeroPoint = std::get<13>(params);
  ps.opModelName = toLowercase(std::get<14>(params));
  ps.useTxnBinWithZp = std::get<15>(params);
  ps.width = std::get<16>(params);
  ps.operatorName = toLowercase(std::get<17>(params));
  return ps;
}

/* Helper function to initialize ofm data buffer with garbase values */
static Tensor
GetOfmTensor(const ParamsStruct &ps,
             std::vector<std::vector<uint16_t>> &ofmDataContainer) {
  std::vector<size_t> ofmShape = {
      1, (static_cast<size_t>(ps.Mo0) * static_cast<size_t>(ps.Mo1)),
      static_cast<size_t>(ps.N)};
  int32_t garbage_value = 0xAAAABBBB;
  /* Sandip TBD : Need to replace vector type from opOfmDtype */
  std::vector<uint16_t> ofm(ps.N * (ps.Mo0) * (ps.Mo1), garbage_value);

  /* Add the ofm data to the ofmDataContainer */
  ofmDataContainer.push_back(std::move(ofm));

  /* Get a reference to the last element of the weightDataContainer which
  contains the weight data just added */
  std::vector<uint16_t> &ofm_data_ref = ofmDataContainer.back();
  return {ofm_data_ref.data(), ofmShape, ps.opOfmDtype};
}

/* Helper function to read ifm data from a file */
static Tensor
GetIfmTensor(const ParamsStruct &ps,
             std::vector<std::vector<uint16_t>> &ifmDataContainer) {
  std::string testDataFolder =
      OpInterface::get_dod_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
      "unit_tests" + "\\" + "testDataMladf" + "\\" +
      GetTestSubFolderName(ps.opModelName, ps.zeroPoint, ps.K, ps.N, ps.F0);

  std::vector<size_t> ifmShape;
  size_t ifmSize;
  if ((ps.zeroPoint == 29172) && (ps.opModelName == "psi")) {
    /* This is a specific case required for layer 1 of PSI model only */
    ifmShape = {1, (static_cast<size_t>(ps.Mi0) * static_cast<size_t>(ps.Mi1)),
                static_cast<size_t>(ps.K + 1)};
    ifmSize = (ps.K + 1) * ps.Mi0 * ps.Mi1;
  } else if ((ps.zeroPoint == 40597) &&
             ((ps.opModelName == "pso2640") || (ps.opModelName == "pso21280") ||
              (ps.opModelName == "pso22560"))) {
    /* This is a specific case required for layer 1 of PSO640 model only */
    ifmShape = {1, (static_cast<size_t>(ps.Mi0) * static_cast<size_t>(ps.Mi1)),
                static_cast<size_t>(ps.K - 4)}; // activate
    ifmSize = (ps.K - 4) * ps.Mi0 * ps.Mi1;
  } else {
    ifmShape = {1, (static_cast<size_t>(ps.Mi0) * static_cast<size_t>(ps.Mi1)),
                static_cast<size_t>(ps.K)};
    ifmSize = ps.K * ps.Mi0 * ps.Mi1;
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

  return {ifm_data_ref.data(), ifmShape, ps.opIfmDtype};
}

static bool GetOperateorHasWeights(std::string operatorName) {
  if (operatorName == "maxpool") {
    return false;
  } else {
    return true;
  }
}

/* Helper function to read weight data from a file */
static Tensor
read_weight_data(const ParamsStruct &ps,
                 std::vector<std::vector<uint16_t>> &weightDataContainer) {
  if (!GetOperateorHasWeights(ps.operatorName)) {
    /* Return an empty Tensor with nullptr data, empty shape, and empty dtype */
    return {nullptr, std::vector<size_t>{}, ""};
  } else {
    /* Read weight data from a file */
    std::string opModelNameLowerCase = ps.opModelName;
    std::transform(opModelNameLowerCase.begin(), opModelNameLowerCase.end(),
                   opModelNameLowerCase.begin(), ::tolower);
    std::string testDataFolder =
        OpInterface::get_dod_base_dir() + "\\" + "tests" + "\\" + "cpp" + "\\" +
        "unit_tests" + "\\" + "testDataMladf" + "\\" +
        GetTestSubFolderName(opModelNameLowerCase, ps.zeroPoint, ps.K, ps.N,
                             ps.F0);
    std::vector<size_t> weightShape =
        (ps.groupId == 1) ? std::vector<size_t>{static_cast<size_t>(ps.N),
                                                static_cast<size_t>(ps.K),
                                                static_cast<size_t>(ps.F0),
                                                static_cast<size_t>(ps.F1)}
                          : std::vector<size_t>{static_cast<size_t>(ps.N), 1,
                                                static_cast<size_t>(ps.F0),
                                                static_cast<size_t>(ps.F1)};
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
    return {weight_data_ref.data(), weightShape, ps.opWtsDtype};
  }
}

/* Helper function to get attributes for each operator test is executing */
static std::map<std::string, std::any> GetAttr(const ParamsStruct &ps) {
  /* Store Attributes */
  std::map<std::string, std::any> attr;
  attr["opType"] = ps.operatorName;
  attr["opIfmDtype"] = ps.opIfmDtype;
  attr["opWtsDtype"] = ps.opWtsDtype;
  attr["opOfmDtype"] = ps.opOfmDtype;

  attr["group"] = std::vector<int>{ps.groupId};
  attr["input_shape"] = std::vector<int>{1, ps.K, ps.Mi0, ps.Mi1};
  attr["output_shape"] = std::vector<int>{1, ps.N, ps.Mo0, ps.Mo1};
  int weightThirdDim = (ps.groupId == 1) ? ps.K : 1;
  attr["weight_shape"] = std::vector<int>{ps.N, weightThirdDim, ps.F0, ps.F1};
  attr["zero_point"] = std::vector<int>{ps.zeroPoint};

  if (ps.opModelName != "pso2") {
    attr["width"] = std::vector<int>{ps.width};
  }

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
                                 int, std::string, bool, int, std ::string>>
        &paramsVec,
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

  for (const auto &paramsTuple : paramsVec) {
    int err_count = 0;
    ParamsStruct ps = ExtractParameters(paramsTuple);
    attributesVec.push_back(GetAttr(ps));
    /* Call the helper function to read weight data with tuple as the argument
     */
    const_Tensor.push_back(read_weight_data(ps, weightDataContainer));
  }

  if (!paramsVec.empty()) {
    const auto &first_params = ExtractParameters(paramsVec.front());
    const auto &last_params = ExtractParameters(paramsVec.back());

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
#if 0
  /* Many times the simulation team provides ofm data in txt format. One time below code is used to convert this in bin format */
  fileName = testDataFolder + "\\" + "ofmRef.txt";
  size_t dataSize = output_Tensor.at(0).shape[0] *
                    output_Tensor.at(0).shape[1] * output_Tensor.at(0).shape[2];
  uint16_t *dataPtr = (uint16_t *)malloc(dataSize * sizeof(uint16_t));
  readTxtFileHex<uint16_t>(fileName, dataPtr, dataSize * sizeof(uint16_t));
  generatedFileName = testDataFolder + "\\" + "ofmRef.bin";
  write_bin_file(generatedFileName, (char *)dataPtr,
                 output_Tensor.at(0).shape[0] * output_Tensor.at(0).shape[1] *
                     output_Tensor.at(0).shape[2] * 2);
#endif
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
TEST(ConcatenateTesta16w16c16_, PsoTest1) {
  static constexpr int64_t NO_ZP = 0xFFFFFFFFFFFFFFFF;
  std::vector<std::tuple<int, int, int, int, int, int, int, int, int, bool,
                         std::string, std::string, std::string, int,
                         std::string, bool, int, std ::string>>
      paramsVec = {
          std::make_tuple(60, 320, 3, 3, 8, 16, 64, 320, 1, false, "uint16",
                          "uint16", "uint16", 40597, "PSO2", true, 320,
                          "conv"), // layer1
          std::make_tuple(64, 320, 2, 2, 16, 16, 32, 160, 1, false, "uint16",
                          "uint16", "uint16", NO_ZP, "PSO2", true, 320,
                          "maxpool"),

          std::make_tuple(32, 160, 3, 3, 16, 32, 32, 160, 1, false, "uint16",
                          "uint16", "uint16", 32705, "PSO2", true, 320,
                          "conv"), // layer2
          std::make_tuple(32, 160, 2, 2, 32, 32, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", NO_ZP, "PSO2", true, 320,
                          "maxpool"),

          std::make_tuple(16, 80, 1, 1, 32, 16, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 36423, "PSO2", true, 320,
                          "conv"), // layer3
          std::make_tuple(16, 80, 3, 3, 16, 32, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 33409, "PSO2", true, 320, "conv"),
          std::make_tuple(16, 80, 1, 1, 32, 128, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 29586, "PSO2", true, 320, "conv"),
          std::make_tuple(16, 80, 1, 1, 128, 16, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 25513, "PSO2", true, 320, "conv"),
          std::make_tuple(16, 80, 3, 3, 16, 32, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 31530, "PSO2", true, 320, "conv"),
          std::make_tuple(16, 80, 1, 1, 32, 128, 16, 80, 1, false, "uint16",
                          "uint16", "uint16", 32591, "PSO2", true, 320,
                          "conv"), // layer8
          std::make_tuple(16, 80, 2, 1, 128, 128, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", NO_ZP, "PSO2", true, 320,
                          "maxpool"),

          std::make_tuple(8, 80, 1, 1, 128, 32, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 31990, "PSO2", true, 320, "conv"),
          std::make_tuple(8, 80, 3, 3, 32, 48, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 36064, "PSO2", true, 320, "conv"),
          std::make_tuple(8, 80, 1, 1, 48, 256, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 35326, "PSO2", true, 320, "conv"),
          std::make_tuple(8, 80, 1, 1, 256, 32, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 34702, "PSO2", true, 320, "conv"),
          std::make_tuple(8, 80, 3, 3, 32, 48, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 30051, "PSO2", true, 320, "conv"),
          std::make_tuple(8, 80, 1, 1, 48, 256, 8, 80, 1, false, "uint16",
                          "uint16", "uint16", 35719, "PSO2", true, 320,
                          "conv"), // layer14
          std::make_tuple(8, 80, 2, 1, 256, 256, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", NO_ZP, "PSO2", true, 320,
                          "maxpool"),

          std::make_tuple(4, 80, 1, 1, 256, 64, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 26536, "PSO2", true, 320, "conv"),
          std::make_tuple(4, 80, 3, 3, 64, 80, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 22444, "PSO2", true, 320, "conv"),
          std::make_tuple(4, 80, 1, 1, 80, 512, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 32234, "PSO2", true, 320, "conv"),
          std::make_tuple(4, 80, 1, 1, 512, 64, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 33891, "PSO2", true, 320, "conv"),
          std::make_tuple(4, 80, 3, 3, 64, 80, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 33497, "PSO2", true, 320, "conv"),
          std::make_tuple(4, 80, 1, 1, 80, 512, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 31960, "PSO2", true, 320, "conv"),
          std::make_tuple(4, 80, 1, 1, 512, 16, 4, 80, 1, false, "uint16",
                          "uint16", "uint16", 33774, "PSO2", true, 320,
                          "conv")};

  int err_count =
      test_concatenate<uint16_t, uint16_t>(paramsVec, 320, 8, 16, "pso2");
}

TEST(ConcatenateTesta16w16c16_, PsoTest2) {
  static constexpr int64_t NO_ZP = 0xFFFFFFFFFFFFFFFF;

  std::vector<std::tuple<int, int, int, int, int, int, int, int, int, bool,
                         std::string, std::string, std::string, int,
                         std::string, bool, int, std ::string>>
      paramsVec = {std::make_tuple(60, 640, 3, 3, 8, 16, 32, 320, 1, false,
                                   "uint16", "uint16", "uint16", 40597,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(32, 320, 3, 3, 16, 32, 16, 160, 1, false,
                                   "uint16", "uint16", "uint16", 32705,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(16, 160, 1, 1, 32, 16, 16, 160, 1, false,
                                   "uint16", "uint16", "uint16", 36423,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(16, 160, 3, 3, 16, 32, 16, 160, 1, false,
                                   "uint16", "uint16", "uint16", 33409,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(16, 160, 1, 1, 32, 128, 16, 160, 1, false,
                                   "uint16", "uint16", "uint16", 29586,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(16, 160, 1, 1, 128, 16, 16, 160, 1, false,
                                   "uint16", "uint16", "uint16", 25513,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(16, 160, 3, 3, 16, 32, 16, 160, 1, false,
                                   "uint16", "uint16", "uint16", 31530,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(16, 160, 1, 1, 32, 128, 8, 160, 1, false,
                                   "uint16", "uint16", "uint16", 32591,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(8, 160, 1, 1, 128, 32, 8, 160, 1, false,
                                   "uint16", "uint16", "uint16", 31990,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(8, 160, 3, 3, 32, 48, 8, 160, 1, false,
                                   "uint16", "uint16", "uint16", 36064,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(8, 160, 1, 1, 48, 256, 8, 160, 1, false,
                                   "uint16", "uint16", "uint16", 35326,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(8, 160, 1, 1, 256, 32, 8, 160, 1, false,
                                   "uint16", "uint16", "uint16", 34702,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(8, 160, 3, 3, 32, 48, 8, 160, 1, false,
                                   "uint16", "uint16", "uint16", 30051,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(8, 160, 1, 1, 48, 256, 4, 160, 1, false,
                                   "uint16", "uint16", "uint16", 35719,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(4, 160, 1, 1, 256, 64, 4, 160, 1, false,
                                   "uint16", "uint16", "uint16", 26536,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(4, 160, 3, 3, 64, 80, 4, 160, 1, false,
                                   "uint16", "uint16", "uint16", 22444,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(4, 160, 1, 1, 80, 512, 4, 160, 1, false,
                                   "uint16", "uint16", "uint16", 32234,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(4, 160, 1, 1, 512, 64, 4, 160, 1, false,
                                   "uint16", "uint16", "uint16", 33891,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(4, 160, 3, 3, 64, 80, 4, 160, 1, false,
                                   "uint16", "uint16", "uint16", 33497,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(4, 160, 1, 1, 80, 512, 4, 160, 1, false,
                                   "uint16", "uint16", "uint16", 31960,
                                   "PSO2640", true, 640, "conv"),
                   std::make_tuple(4, 160, 1, 1, 512, 16, 4, 160, 1, false,
                                   "uint16", "uint16", "uint16", 33774,
                                   "PSO2640", true, 640, "conv")};

  int err_count =
      test_concatenate<uint16_t, uint16_t>(paramsVec, 640, 8, 16, "pso2");
}

TEST(ConcatenateTesta16w16c16_, PsoTest3) {
  static constexpr int64_t NO_ZP = 0xFFFFFFFFFFFFFFFF;

  std::vector<std::tuple<int, int, int, int, int, int, int, int, int, bool,
                         std::string, std::string, std::string, int,
                         std::string, bool, int, std ::string>>
      paramsVec = {std::make_tuple(60, 1280, 3, 3, 8, 16, 32, 640, 1, false,
                                   "uint16", "uint16", "uint16", 40597,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(32, 640, 3, 3, 16, 32, 16, 320, 1, false,
                                   "uint16", "uint16", "uint16", 32705,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(16, 320, 1, 1, 32, 16, 16, 320, 1, false,
                                   "uint16", "uint16", "uint16", 36423,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(16, 320, 3, 3, 16, 32, 16, 320, 1, false,
                                   "uint16", "uint16", "uint16", 33409,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(16, 320, 1, 1, 32, 128, 16, 320, 1, false,
                                   "uint16", "uint16", "uint16", 29586,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(16, 320, 1, 1, 128, 16, 16, 320, 1, false,
                                   "uint16", "uint16", "uint16", 25513,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(16, 320, 3, 3, 16, 32, 16, 320, 1, false,
                                   "uint16", "uint16", "uint16", 31530,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(16, 320, 1, 1, 32, 128, 8, 320, 1, false,
                                   "uint16", "uint16", "uint16", 32591,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(8, 320, 1, 1, 128, 32, 8, 320, 1, false,
                                   "uint16", "uint16", "uint16", 31990,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(8, 320, 3, 3, 32, 48, 8, 320, 1, false,
                                   "uint16", "uint16", "uint16", 36064,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(8, 320, 1, 1, 48, 256, 8, 320, 1, false,
                                   "uint16", "uint16", "uint16", 35326,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(8, 320, 1, 1, 256, 32, 8, 320, 1, false,
                                   "uint16", "uint16", "uint16", 34702,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(8, 320, 3, 3, 32, 48, 8, 320, 1, false,
                                   "uint16", "uint16", "uint16", 30051,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(8, 320, 1, 1, 48, 256, 4, 320, 1, false,
                                   "uint16", "uint16", "uint16", 35719,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(4, 320, 1, 1, 256, 64, 4, 320, 1, false,
                                   "uint16", "uint16", "uint16", 26536,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(4, 320, 3, 3, 64, 80, 4, 320, 1, false,
                                   "uint16", "uint16", "uint16", 22444,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(4, 320, 1, 1, 80, 512, 4, 320, 1, false,
                                   "uint16", "uint16", "uint16", 32234,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(4, 320, 1, 1, 512, 64, 4, 320, 1, false,
                                   "uint16", "uint16", "uint16", 33891,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(4, 320, 3, 3, 64, 80, 4, 320, 1, false,
                                   "uint16", "uint16", "uint16", 33497,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(4, 320, 1, 1, 80, 512, 4, 320, 1, false,
                                   "uint16", "uint16", "uint16", 31960,
                                   "PSO21280", true, 1280, "conv"),
                   std::make_tuple(4, 320, 1, 1, 512, 16, 4, 320, 1, false,
                                   "uint16", "uint16", "uint16", 33774,
                                   "PSO21280", true, 1280, "conv")};

  int err_count =
      test_concatenate<uint16_t, uint16_t>(paramsVec, 1280, 8, 16, "pso2");
}

TEST(ConcatenateTesta16w16c16_, PsoTest4) {
  static constexpr int64_t NO_ZP = 0xFFFFFFFFFFFFFFFF;

  std::vector<std::tuple<int, int, int, int, int, int, int, int, int, bool,
                         std::string, std::string, std::string, int,
                         std::string, bool, int, std ::string>>
      paramsVec = {std::make_tuple(60, 2560, 3, 3, 8, 16, 32, 1280, 1, false,
                                   "uint16", "uint16", "uint16", 40597,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(32, 1280, 3, 3, 16, 32, 16, 640, 1, false,
                                   "uint16", "uint16", "uint16", 32705,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(16, 640, 1, 1, 32, 16, 16, 640, 1, false,
                                   "uint16", "uint16", "uint16", 36423,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(16, 640, 3, 3, 16, 32, 16, 640, 1, false,
                                   "uint16", "uint16", "uint16", 33409,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(16, 640, 1, 1, 32, 128, 16, 640, 1, false,
                                   "uint16", "uint16", "uint16", 29586,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(16, 640, 1, 1, 128, 16, 16, 640, 1, false,
                                   "uint16", "uint16", "uint16", 25513,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(16, 640, 3, 3, 16, 32, 16, 640, 1, false,
                                   "uint16", "uint16", "uint16", 31530,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(16, 640, 1, 1, 32, 128, 8, 640, 1, false,
                                   "uint16", "uint16", "uint16", 32591,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(8, 640, 1, 1, 128, 32, 8, 640, 1, false,
                                   "uint16", "uint16", "uint16", 31990,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(8, 640, 3, 3, 32, 48, 8, 640, 1, false,
                                   "uint16", "uint16", "uint16", 36064,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(8, 640, 1, 1, 48, 256, 8, 640, 1, false,
                                   "uint16", "uint16", "uint16", 35326,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(8, 640, 1, 1, 256, 32, 8, 640, 1, false,
                                   "uint16", "uint16", "uint16", 34702,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(8, 640, 3, 3, 32, 48, 8, 640, 1, false,
                                   "uint16", "uint16", "uint16", 30051,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(8, 640, 1, 1, 48, 256, 4, 640, 1, false,
                                   "uint16", "uint16", "uint16", 35719,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(4, 640, 1, 1, 256, 64, 4, 640, 1, false,
                                   "uint16", "uint16", "uint16", 26536,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(4, 640, 3, 3, 64, 80, 4, 640, 1, false,
                                   "uint16", "uint16", "uint16", 22444,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(4, 640, 1, 1, 80, 512, 4, 640, 1, false,
                                   "uint16", "uint16", "uint16", 32234,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(4, 640, 1, 1, 512, 64, 4, 640, 1, false,
                                   "uint16", "uint16", "uint16", 33891,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(4, 640, 3, 3, 64, 80, 4, 640, 1, false,
                                   "uint16", "uint16", "uint16", 33497,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(4, 640, 1, 1, 80, 512, 4, 640, 1, false,
                                   "uint16", "uint16", "uint16", 31960,
                                   "PSO22560", true, 2560, "conv"),
                   std::make_tuple(4, 640, 1, 1, 512, 16, 4, 640, 1, false,
                                   "uint16", "uint16", "uint16", 33774,
                                   "PSO22560", true, 2560, "conv")};

  int err_count =
      test_concatenate<uint16_t, uint16_t>(paramsVec, 2560, 8, 16, "pso2");
}
#endif
