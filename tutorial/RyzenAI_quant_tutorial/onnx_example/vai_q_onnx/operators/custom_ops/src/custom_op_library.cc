//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>

#include "custom_op_qdq.h"
#include "custom_op_bfp.h"

#define ORT_TRY try
#define ORT_CATCH(x) catch (x)
#define ORT_RETHROW throw;

#define ORT_HANDLE_EXCEPTION(func) func()


static const char* c_OpDomain = "com.vai.quantize";
static const int64_t c_OpVersion = 1;

struct CustomQuantizeLinear : Ort::CustomOpBase<CustomQuantizeLinear, vai_q::KernelCustomQuantizeLinear> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return std::make_unique<vai_q::KernelCustomQuantizeLinear>(api, info).release();
  };

  const char* GetName() const { return "VitisQuantizeLinear"; };

#ifdef USE_CUDA
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; };
#endif

  size_t GetInputTypeCount() const { return 3; };
  ONNXTensorElementDataType GetInputType(size_t index) const {
    if (index == 0)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else if (index == 1)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };
#if 0
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
    // The third input (index == 2) is optional
    if (index == 2)
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;

    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  };
#endif
  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t index) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; };
#if 0
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }
#endif
};

struct CustomDequantizeLinear : Ort::CustomOpBase<CustomDequantizeLinear, vai_q::KernelCustomDequantizeLinear> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return std::make_unique<vai_q::KernelCustomDequantizeLinear>(api, info).release();
  };

  const char* GetName() const { return "VitisDequantizeLinear"; };

#ifdef USE_CUDA
  const char* GetExecutionProviderType() const { return "CUDAExecutionProvider"; };
#endif

  size_t GetInputTypeCount() const { return 3; };
  ONNXTensorElementDataType GetInputType(size_t index) const {
    if (index == 0)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    else if (index == 1)
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    else
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };
#if 0
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
    // The third input (index == 2) is optional
    if (index == 2)
      return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;

    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  };
#endif
  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t index) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
#if 0
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }
#endif
};


static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);

  static const CustomQuantizeLinear c_CustomOpOne;
  static const CustomDequantizeLinear c_CustomOpTwo;
  static const BFPFixNeuron c_BFPFixNeuron;

  OrtStatus* result = nullptr;

  ORT_TRY {
    Ort::CustomOpDomain domain{c_OpDomain};
    domain.Add(&c_CustomOpOne);
    domain.Add(&c_CustomOpTwo);
    domain.Add(&c_BFPFixNeuron);

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      Ort::Status status{e};
      result = status.release();
    });
  }
  return result;
}

OrtStatus* ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api) {
  return RegisterCustomOps(options, api);
}
