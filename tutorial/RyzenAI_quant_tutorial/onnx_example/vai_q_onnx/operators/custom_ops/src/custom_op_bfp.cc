//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// #include "bfp.h"

#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT
#include "custom_op_bfp.h"

#include <vector>
#include <cmath>
#include <mutex>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include "bfp/bfp.h"

#ifdef USE_CUDA
#include "cuda_runtime_api.h"
#endif

BFPFixNeuronKernel::BFPFixNeuronKernel(
    const OrtApi& ort_api, 
    const OrtKernelInfo* k_info, 
    std::string bfp_method,
    int64_t bit_width, 
    int64_t block_size, 
    int64_t rounding_mode,
    int64_t sub_block_size,
    int64_t sub_block_shift_bits) : 
      ort_(ort_api), 
      bfp_method_(bfp_method),
      bit_width_(bit_width), 
      block_size_(block_size), 
      rounding_mode_(rounding_mode),
      sub_block_size_(sub_block_size),
      sub_block_shift_bits_(sub_block_shift_bits) {
  Ort::ConstKernelInfo info{k_info};
  info_copy_ = info.Copy();
}

Ort::Value BFPFixNeuronKernel::do_bfp(Ort::Value &input) {
  std::vector<int64_t> dimensions = input.GetTensorTypeAndShapeInfo().GetShape();
  size_t element_count = input.GetTensorTypeAndShapeInfo().GetElementCount();
  Buffer b(element_count * 4);
  tmp_buffers_.push_back(b);
  auto output = Ort::Value::CreateTensor<float>(
    input.GetTensorMemoryInfo(), (float*)b.get_data_ptr(), element_count, dimensions.data(), dimensions.size());
  
  if (bfp_method_ == "to_bfp") {
    to_bfp(input, bit_width_, block_size_, rounding_mode_, output);
  } else if (bfp_method_ == "to_bfp_v2") {
    to_bfp_v2(input, bit_width_, block_size_, rounding_mode_, output);
  } else if (bfp_method_ == "to_bfp_prime_shared") {
    to_bfp_prime_shared(
      input, bit_width_, block_size_, 
      sub_block_size_, sub_block_shift_bits_, 
      rounding_mode_, output);
  } else {
    throw std::invalid_argument(
      "Invalid bfp_method, valid bfp_method should be one of [\"to_bfp\", \"to_bfp_v2\", \"to_bfp_prime_shared\"], current bfp_method is " + bfp_method_);
  }
  return output;
}

void BFPFixNeuronKernel::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);
  auto input_value = ctx.GetInput(0);
  std::vector<int64_t> dimensions = input_value.GetTensorTypeAndShapeInfo().GetShape();
  auto input_tensor = Ort::Value::CreateTensor<float>(
    input_value.GetTensorMemoryInfo(),
    const_cast<float*>(input_value.GetTensorData<float>()), 
    input_value.GetTensorTypeAndShapeInfo().GetElementCount(), 
    dimensions.data(), dimensions.size());

  Ort::Value ret{nullptr};
  if (dimensions.size() == 1) {
    auto padded_tensor = pad(context, input_tensor, block_size_);
    auto bfp_tensor = do_bfp(padded_tensor);
    ret = slice(context, bfp_tensor, dimensions[0]);
  } else {
    auto transposed_tensor = transpose(context, input_tensor, 1, dimensions.size() - 1);
    auto padded_tensor = pad(context, transposed_tensor, block_size_);
    auto bfp_tensor = do_bfp(padded_tensor);
    // auto bfp_tensor = std::move(padded_tensor);
    auto sliced_tensor = slice(context, bfp_tensor, dimensions[1]);
    ret = transpose(context, sliced_tensor, 1, dimensions.size() - 1);
  }
  auto output = ctx.GetOutput(0, ret.GetTensorTypeAndShapeInfo().GetShape());

#ifdef USE_CUDA
  cudaMemcpy(
    output.GetTensorMutableRawData(), ret.GetTensorMutableRawData(), 
    output.GetTensorTypeAndShapeInfo().GetElementCount() * 4, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
#else
  memcpy(output.GetTensorMutableRawData(), ret.GetTensorMutableRawData(), 
    output.GetTensorTypeAndShapeInfo().GetElementCount() * 4);
#endif

  for (Buffer b : tmp_buffers_) {
    b.release();
  }
  tmp_buffers_.clear();
}

BFPFixNeuronKernel::~BFPFixNeuronKernel() {
}

void BFPFixNeuronKernel::create_pad_op() {
  const char* add_type_constraint_names[] = {"T", "T", "T"};
  ONNXTensorElementDataType add_type_constraint_values[] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
  std::string mode_str = "constant";
  auto mode = Ort::OpAttr("mode", mode_str.c_str(), 1, OrtOpAttrType::ORT_OP_ATTR_STRING);
  Ort::OpAttr attrs[1] = {std::move(mode)};
  op_pad_ = Ort::Op::Create(info_copy_, "Pad", "", 18,
                            add_type_constraint_names,
                            add_type_constraint_values,
                            3, attrs, 1, 3, 1);
  op_pad_init_ = true;
}

Ort::Value BFPFixNeuronKernel::pad(OrtKernelContext* context, Ort::Value &input, int block_size) {
  std::vector<int64_t> dimensions = input.GetTensorTypeAndShapeInfo().GetShape();
  if (!op_pad_init_) {
    create_pad_op();
  }
  int channels_to_pad = dimensions[dimensions.size() - 1] % block_size == 0 ? 
    0 : block_size - dimensions[dimensions.size() - 1] % block_size;
  int64_t pad_size = 2 * dimensions.size();
  int64_t pad[pad_size];
  for (int i = 0; i < pad_size - 1; i++) pad[i] = 0;
  pad[2 * dimensions.size() - 1] = channels_to_pad;
  auto pad_tensor = Ort::Value::CreateTensor<int64_t>(
    input.GetTensorMemoryInfo(), &pad[0], pad_size, &pad_size, 1);
  
  int64_t const_value_size = 1;
  float const_value = 0;
  auto const_value_tensor = Ort::Value::CreateTensor<float>(
    input.GetTensorMemoryInfo(), &const_value, 1, &const_value_size, 1);
  
  dimensions[dimensions.size() - 1] += channels_to_pad;
  size_t element_count = 1;
  for (int i : dimensions) {
    element_count *= i;
  }
  Buffer b(element_count * 4);
  tmp_buffers_.push_back(b);

  auto output = Ort::Value::CreateTensor<float>(
    input.GetTensorMemoryInfo(), (float*)b.get_data_ptr(), element_count, dimensions.data(), dimensions.size());
  
  const OrtValue* inputs[3] = {input, pad_tensor, const_value_tensor};
  OrtValue* outputs[1] = {output};
  op_pad_.Invoke(context, inputs, 3, outputs, 1);
  return output;
}

void BFPFixNeuronKernel::create_transpose_op(size_t num_dims, int from, int to) {
  const char* add_type_constraint_names[1] = {"T"};
  ONNXTensorElementDataType add_type_constraint_values[1] = {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
  int64_t perm[num_dims];
  for (int i = 0; i < num_dims; i++) {
    perm[i] = i;
  }
  int64_t tmp = perm[from];
  perm[from] = perm[to];
  perm[to] = tmp;
  auto perm_attr = Ort::OpAttr("perm", &perm, num_dims, OrtOpAttrType::ORT_OP_ATTR_INTS);
  Ort::OpAttr attrs[1] = {std::move(perm_attr)};

  op_transpose_ = Ort::Op::Create(info_copy_, "Transpose", "", 13,
                            add_type_constraint_names,
                            add_type_constraint_values,
                            1, attrs, 1, 1, 1);
  op_transpose_init_ = true;
}

Ort::Value BFPFixNeuronKernel::transpose(OrtKernelContext* context, Ort::Value &input, int from, int to) {
  std::vector<int64_t> dimensions = input.GetTensorTypeAndShapeInfo().GetShape();
  if (!op_transpose_init_) {
    create_transpose_op(dimensions.size(), from, to);
  }
  int64_t tmp = dimensions[from];
  dimensions[from] = dimensions[to];
  dimensions[to] = tmp;

  size_t element_count = input.GetTensorTypeAndShapeInfo().GetElementCount();
  Buffer b(element_count * 4);
  tmp_buffers_.push_back(b);

  auto output = Ort::Value::CreateTensor<float>(
    input.GetTensorMemoryInfo(), (float*)b.get_data_ptr(), element_count, dimensions.data(), dimensions.size());

  const OrtValue* inputs[1] = {input};
  OrtValue* outputs[1] = {output};
  op_transpose_.Invoke(context, inputs, 1, outputs, 1);
  return output;
}

void BFPFixNeuronKernel::create_slice_op() {
  const char* add_type_constraint_names[] = {"T", "T", "T", "T", "T"};
  ONNXTensorElementDataType add_type_constraint_values[] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64};
  op_slice_ = Ort::Op::Create(info_copy_, "Slice", "", 13,
                            add_type_constraint_names,
                            add_type_constraint_values,
                            5, nullptr, 0, 5, 1);
  op_slice_init_ = true;
}

Ort::Value BFPFixNeuronKernel::slice(OrtKernelContext* context, Ort::Value &input, size_t last_dim) {
  if (!op_slice_init_) {
    create_slice_op();
  }
  std::vector<int64_t> dimensions = input.GetTensorTypeAndShapeInfo().GetShape();

  int64_t num_dims = dimensions.size();
  dimensions[num_dims - 1] = last_dim;
  int64_t start[num_dims];
  int64_t axes[num_dims];
  int64_t steps[num_dims];
  for (int i = 0; i < num_dims; i++) {
    start[i] = 0;
    axes[i] = i;
    steps[i] = 1;
  }
  auto start_tensor = Ort::Value::CreateTensor<int64_t>(
    input.GetTensorMemoryInfo(), &start[0], num_dims, &num_dims, 1);
  auto end_tensor = Ort::Value::CreateTensor<int64_t>(
    input.GetTensorMemoryInfo(), &dimensions[0], num_dims, &num_dims, 1);
  auto axes_tensor = Ort::Value::CreateTensor<int64_t>(
    input.GetTensorMemoryInfo(), &axes[0], num_dims, &num_dims, 1);
  auto steps_tensor = Ort::Value::CreateTensor<int64_t>(
    input.GetTensorMemoryInfo(), &steps[0], num_dims, &num_dims, 1);
  
  size_t element_count = 1;
  for (auto i : dimensions) {
    element_count *= i;
  }
  Buffer b(element_count * 4);
  tmp_buffers_.push_back(b);

  auto output = Ort::Value::CreateTensor<float>(
    input.GetTensorMemoryInfo(), (float*)b.get_data_ptr(), element_count, dimensions.data(), dimensions.size());

  const OrtValue* inputs[] = {input, start_tensor, end_tensor, axes_tensor, steps_tensor};
  OrtValue* outputs[] = {output};
  op_slice_.Invoke(context, inputs, 5, outputs, 1);
  return output;
}
