//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_c_api.h"
#ifdef USE_CUDA
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct Buffer {
  Buffer(size_t size) {
  #ifdef USE_CUDA
    cudaMalloc(&data_, size);
  #else
    data_ = new char[size];
  #endif
  }

  void* get_data_ptr() {
    return data_;
  }

  void release() {
  #ifdef USE_CUDA
    cudaFree(data_);
  #else
    delete[] data_;
  #endif
  }
private:
  char* data_;
};


struct BFPFixNeuronKernel {
  BFPFixNeuronKernel(
    const OrtApi& ort_api, 
    const OrtKernelInfo* info, 
    std::string bfp_method,
    int64_t bit_width, 
    int64_t block_size, 
    int64_t rounding_mode,
    int64_t sub_block_size,
    int64_t sub_block_shift_bits);
  ~BFPFixNeuronKernel();
  void Compute(OrtKernelContext* context);

  Ort::Value transpose(OrtKernelContext* context, Ort::Value &input, int from, int to);

  void create_transpose_op(size_t num_dims, int from, int to);

  Ort::Value pad(OrtKernelContext* context, Ort::Value &input, int block_size);

  void create_pad_op();

  Ort::Value slice(OrtKernelContext* context, Ort::Value &input, size_t last_dim);

  void create_slice_op();

  Ort::Value do_bfp(Ort::Value &input);

 private:

  const OrtApi& ort_;
  Ort::KernelInfo info_copy_{nullptr};
  Ort::Op op_transpose_{nullptr};
  bool op_transpose_init_ = false;
  Ort::Op op_pad_{nullptr};
  bool op_pad_init_ = false;
  Ort::Op op_slice_{nullptr};
  bool op_slice_init_ = false;
  std::vector<Buffer> tmp_buffers_;

  std::string bfp_method_ = "to_bfp";
  int64_t bit_width_ = 15;
  int64_t block_size_ = 16;
  int64_t rounding_mode_ = 0;
  int64_t sub_block_size_ = 2;
  int64_t sub_block_shift_bits_ = 1;
};

struct BFPFixNeuron : Ort::CustomOpBase<BFPFixNeuron, BFPFixNeuronKernel> {
  explicit BFPFixNeuron() {}

  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const { 
    Ort::InitApi(&api);

    std::string bfp_method;
    bfp_method.resize(50);
    size_t bfp_method_len;
    int64_t bit_width;
    int64_t block_size;
    int64_t rounding_mode;
    int64_t sub_block_size;
    int64_t sub_block_shift_bits;

    auto status = api.KernelInfoGetAttribute_string(info, "bfp_method", &bfp_method[0], &bfp_method_len);
    if (status != nullptr) bfp_method = "to_bfp";
    else bfp_method.resize(bfp_method_len - 1);
    status = api.KernelInfoGetAttribute_int64(info, "bit_width", &bit_width);
    if (status != nullptr) bit_width = 15;
    status = api.KernelInfoGetAttribute_int64(info, "block_size", &block_size);
    if (status != nullptr) block_size = 16;
    status = api.KernelInfoGetAttribute_int64(info, "rounding_mode", &rounding_mode);
    if (status != nullptr) rounding_mode = 0;
    status = api.KernelInfoGetAttribute_int64(info, "sub_block_size", &sub_block_size);
    if (status != nullptr) sub_block_size = 2;
    status = api.KernelInfoGetAttribute_int64(info, "sub_block_shift_bits", &sub_block_shift_bits);
    if (status != nullptr) sub_block_shift_bits = 1;

    return new BFPFixNeuronKernel(
      api, info, bfp_method, bit_width, 
      block_size, rounding_mode, sub_block_size, sub_block_shift_bits); 
  };
  const char* GetName() const { return "BFPFixNeuron"; };

  const char* GetExecutionProviderType() const { 
  #ifdef USE_CUDA
    return "CUDAExecutionProvider"; 
  #else
    return "CPUExecutionProvider";
  #endif
  };

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
};

#ifdef __cplusplus
}
#endif

