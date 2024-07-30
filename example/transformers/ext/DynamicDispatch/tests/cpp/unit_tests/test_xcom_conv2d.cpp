/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include <ops/xcom/conv/conv.hpp>

#include "test_common.hpp"
template <typename InT, typename WgT, typename BiasT, typename OuT, bool DWC>
int test_conv2d(size_t H, size_t W, size_t C_in, size_t kernel_size,
                size_t stride, size_t C_out, bool bias_en,
                const std::string &a_dtype, const std::string &b_dtype,
                const std::string &bias_dtype, const std::string &c_dtype) {

  // assume transpose has been done to convert from [NCHW] to [NHWC]
  std::vector<size_t> activations_shape = {1, H, W, C_in};
  // use ONNX format for passing in shape
  std::vector<size_t> weights_shape = {C_out, C_in, kernel_size, kernel_size};
  std::vector<size_t> bias_shape = {C_out, 1, 1, 1};

  DOD_ASSERT(H % stride == 0, "H is divisible by stride");
  DOD_ASSERT(W % stride == 0, "W is divisible by stride");
  DOD_ASSERT(kernel_size == 1 || kernel_size == 3,
             "Expect kernel size to be 1x1 or 3x3");

  // Expect convolution to maintain size when stride is 1
  // for 1x1 this is default behaviour, for 3x3 assumes zero padding dim of 1
  size_t H_out = H / stride;
  size_t W_out = W / stride;

  std::vector<size_t> out_shape = {1, H_out, W_out, C_out};

  std::vector<InT> activations(H * W * C_in);
  std::vector<WgT> weights(C_out * kernel_size * kernel_size * C_in);
  std::vector<BiasT> bias(C_out, 0);
  std::vector<OuT> cpu_q_out(H_out * W_out * C_out);
  std::vector<OuT> aie_out(H_out * W_out * C_out);

  constexpr size_t LAYER_PARAM_SIZE = 64;
  std::vector<std::uint32_t> qdq_params(LAYER_PARAM_SIZE /
                                        sizeof(std::uint32_t));

  ryzenai::xcom::conv_qdq_info_t param;

  param.H = H;
  param.W = W;
  param.C_in = C_in;
  param.C_out = C_out;
  param.kernel_size_y = kernel_size;
  param.kernel_size_x = kernel_size;
  param.stride_y = stride;
  param.stride_x = stride;

  static_assert(LAYER_PARAM_SIZE == sizeof(param));

  memcpy(qdq_params.data(), &param, sizeof(param));

  std::vector<size_t> qdq_params_shape = {LAYER_PARAM_SIZE /
                                          sizeof(std::uint32_t)};

  srand(0xABCD);
  initialize_random<InT>(activations, activations.size(), 5, -5);
  initialize_random<WgT>(weights, weights.size(), 5, -5);

  if (bias_en) {
    initialize_random<BiasT>(bias, bias.size(), 5, -5);
  }

  std::string folder_name = "/bin/conv_case/";
  std::string activation_path =
      OpInterface::get_dod_base_dir() + folder_name + "/ifm.bin";
  std::vector<std::uint8_t> input_override =
      OpsFusion::read_bin_file<std::uint8_t>(activation_path);

  // Input activations dont get shuffled, just catch if any padding is done
  DOD_ASSERT(activations.size() * sizeof(InT) == input_override.size(),
             "Data buffers size mismatch");
  memcpy(activations.data(), input_override.data(), input_override.size());

  // For now skip reading param into weights, since param contains
  //[layer_params, weights (shuffled/padded), bias]
  // instead, this is overwritten in conv2d internally when populating xrt::BO

  std::string golden_path =
      OpInterface::get_dod_base_dir() + folder_name + "/golden_0.bin";
  std::vector<std::uint8_t> golden_override =
      OpsFusion::read_bin_file<std::uint8_t>(golden_path);

  DOD_ASSERT(cpu_q_out.size() * sizeof(OuT) == golden_override.size(),
             "Data buffers size mismatch");
  memcpy(cpu_q_out.data(), golden_override.data(), golden_override.size());

  bool load_xrt = false;
  ryzenai::xcom::conv2d conv2d_ =
      ryzenai::xcom::conv2d<InT, WgT, BiasT, OuT, DWC>(
          a_dtype, b_dtype, bias_dtype, c_dtype, load_xrt);

  std::vector<Tensor> const_Tensor = {
      {weights.data(), weights_shape, b_dtype},
      {bias.data(), bias_shape, bias_dtype},
      {qdq_params.data(), qdq_params_shape, "uint32"},
  };

  std::vector<Tensor> input_Tensor;
  struct Tensor act_T = {activations.data(), activations_shape, a_dtype};
  input_Tensor.push_back(act_T);

  std::vector<Tensor> output_Tensor;
  struct Tensor out_T = {aie_out.data(), out_shape, c_dtype};
  output_Tensor.push_back(out_T);

  ryzenai::xcom::conv_params_t params;

  // TO DO: check whether this is actually configurable just on layer params
  // params.relu = true;

  // QDQ params - typically affects txn bin
  // params.zero_point = 0;
  // params.scale = 0;

  // will be used to look up transaction binary
  params.static_params.kernel_x = kernel_size;
  params.static_params.kernel_y = kernel_size;
  params.static_params.stride_x = stride;
  params.static_params.stride_y = stride;
  // should drive whether bias reads in data or zeroed out
  // assume kernel always has bias buffer
  params.static_params.bias = bias_en;

  params.static_params.shape_info.H = H;
  params.static_params.shape_info.W = W;
  params.static_params.shape_info.C_in = C_in;
  params.static_params.shape_info.C_out = C_out;

  conv2d_.set_params(params);
  conv2d_.initialize_const_params(const_Tensor);

#ifdef UNIT_TEST_PERF
  LOG_THIS("H = " << H << ", W = " << W << ", C_in = " << C_in
                  << ", stride = " << stride << ", C_out = " << C_out);
  PROFILE_THIS(conv2d_.execute(input_Tensor, output_Tensor));
#else
  conv2d_.execute(input_Tensor, output_Tensor);
#endif

  int err_status =
      memcmp(aie_out.data(), cpu_q_out.data(), aie_out.size() * sizeof(OuT));

  return err_status;
}

// Conv2D
TEST(XCOM_CONV2D_a8w8out8, Kernel1) {
  int err_status = test_conv2d<int8_t, int8_t, int8_t, int8_t, false>(
      320, 320, 48, 1, 1, 24, true, "int8", "int8", "int8", "int8");
  EXPECT_TRUE(err_status == 0) << "Error status = " << err_status;
}
