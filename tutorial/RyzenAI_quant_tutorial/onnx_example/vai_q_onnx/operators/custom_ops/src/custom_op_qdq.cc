//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "custom_op_qdq.h"
#include "quantize_linear.h"

#include <vector>
#include <cmath>


#include "core/framework/float16.h"

#ifndef USE_NATIVE_OP_CAST
namespace onnxruntime {

struct MLFloat16;

// Since onnxruntime does not implement functions for MLFloat16,
// we added a fake implements here to avoid running error.

MLFloat16::MLFloat16(float f) {
  val = 0;
}

float MLFloat16::ToFloat() const {
  return 0;
}

}
#endif


namespace vai_q {

/*
* Return the total number of elements up to the specified axis.
* If the axis interval is empty (axis == 0), return 1.
*/
static int64_t SizeToDimension(std::vector<int64_t> input_shape, int64_t axis_no_neg) {
  int64_t size = 1;
  for (size_t index = 0; index < input_shape.size(); index++)
    if (index < static_cast<size_t>(axis_no_neg))
      size = size * input_shape[index];
  return size;
}

/*
* Return the total number of elements from the specified axis to the end of the tensor shape.
* If the axis interval is empty (axis == shape.size()), return 1.
*/
static int64_t SizeFromDimension(std::vector<int64_t> input_shape, int64_t axis_no_neg) {
  int64_t size = 1;
  for (size_t index = 0; index < input_shape.size(); index++)
    if (index >= static_cast<size_t>(axis_no_neg))
      size = size * input_shape[index];
  return size;
}

/*
* Handle a potentially negative axis. Enforces negative axis is valid.
*/
static int64_t HandleNegativeAxis(int64_t axis, int64_t tensor_rank) {
  return axis < 0 ? axis + tensor_rank : axis;
}

/*
* Returns true if given tensor is a scalar or 1D tensor of size 1.
*/
static bool IsScalarOr1ElementVector(std::vector<int64_t>& dims) {
  if (dims.empty() || dims.size() == 1)
    return true;
  else
    return false;
}

/*
* Calculate block count and size according to per-tensor or per-channel
*/
static void PrepareForQDQ(const Ort::ConstValue& input,
                          const Ort::ConstValue& scale,
                          const Ort::ConstValue* pzero_point,
                          int64_t axis,
                          int64_t& block_count,
                          int64_t& broadcast_dim,
                          int64_t& block_size) {
  auto input_info = input.GetTensorTypeAndShapeInfo();
  auto scale_info = scale.GetTensorTypeAndShapeInfo();

  std::vector<int64_t> input_shape = input_info.GetShape();
  std::vector<int64_t> scale_shape = scale_info.GetShape();
  std::vector<int64_t> zp_shape;
  if (pzero_point) zp_shape = pzero_point->GetTensorTypeAndShapeInfo().GetShape();

  if (IsScalarOr1ElementVector(scale_shape)) {  // per-tensor QuantizeLinear/DequantizeLinear
    block_count = 1;
    broadcast_dim = 1;
    block_size = static_cast<int64_t>(input_info.GetElementCount());

    // enforce that zero point are scalars
    if ((pzero_point == nullptr || IsScalarOr1ElementVector(zp_shape)) == false)
      ORT_CXX_API_THROW("PrepareForQDQ zero_point must be null or a scalar or 1D tensor or size 1.",
                        OrtErrorCode::ORT_INVALID_GRAPH);
  } else {  // per-channel QuantizeLinear/DequantizeLinear
    int64_t tensor_rank = static_cast<int64_t>(input_info.GetDimensionsCount());
    if ((axis >= -tensor_rank && axis <= tensor_rank - 1) == false)
      ORT_CXX_API_THROW("PrepareForQDQ axis should be in the valid range.",
                        OrtErrorCode::ORT_INVALID_GRAPH);

    const int64_t axis_no_neg = HandleNegativeAxis(axis, tensor_rank);
    if ((axis_no_neg >= 0 && axis_no_neg < input_shape.size()) == false)
      ORT_CXX_API_THROW("PrepareForQDQ axis_no_neg must be within the input shape dimensions.",
                        OrtErrorCode::ORT_INVALID_GRAPH);

    block_count = SizeToDimension(input_shape, axis_no_neg);
    broadcast_dim = input_shape[static_cast<size_t>(axis_no_neg)];
    block_size = SizeFromDimension(input_shape, axis_no_neg + 1);

    // if an axis was specified, ensure the scale and zero point are compatible
    if ((scale_info.GetDimensionsCount() == 1 && scale_shape[0] == broadcast_dim) == false)
      ORT_CXX_API_THROW("PrepareForQDQ scale must be 1D tensor with the size equals broadcast_dim.",
                        OrtErrorCode::ORT_INVALID_GRAPH);

    if ((pzero_point == nullptr ||
        (pzero_point->GetTensorTypeAndShapeInfo().GetDimensionsCount() == 1 &&
	 zp_shape[0] == broadcast_dim)) == false)
      ORT_CXX_API_THROW("PrepareForQDQ zero_point must be null or 1D tensor with the size equals broadcast_dim.",
                        OrtErrorCode::ORT_INVALID_GRAPH);
  }
}


Ort::Op NativeCastOpCreate(Ort::KernelInfo& info,
                           ONNXTensorElementDataType in_type,
                           ONNXTensorElementDataType out_type) {
  const char* add_type_constraint_names[] = {"T"};
  ONNXTensorElementDataType add_type_constraint_values[] = {in_type};
  size_t type_constraint_count = 0;  // TODO : set 1 here, it will raise error

  int64_t to = static_cast<int64_t>(out_type);
  Ort::OpAttr attr = Ort::OpAttr("to", &to, 1, OrtOpAttrType::ORT_OP_ATTR_INT);

  return Ort::Op::Create(info, "Cast", "", 13,
                         add_type_constraint_names,
                         add_type_constraint_values,
                         type_constraint_count,
                         &attr, 1, 1, 1);
}

void NativeCastOpInvokeToFloat16(OrtKernelContext* context, Ort::Op& op) {
  Ort::KernelContext ctx(context);

  auto input = ctx.GetInput(0);

  // clip input to avoid overflow
  float* clipped_input = (float*)malloc(input.GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(float));
  memcpy(clipped_input, input.GetTensorData<float>(), input.GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(float));
  for (size_t k = 0; k < input.GetTensorTypeAndShapeInfo().GetElementCount(); k ++) {
      if (clipped_input[k] < -65504) clipped_input[k] = -65504;
      if (clipped_input[k] > 65504) clipped_input[k] = 65504;
  }

  std::vector<int64_t> dimensions = input.GetTensorTypeAndShapeInfo().GetShape();
  auto input_value = Ort::Value::CreateTensor(input.GetTensorMemoryInfo(),
                     //(void*)input.GetTensorData<float>(),
                     (void*)clipped_input,
                     input.GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(float),
                     dimensions.data(), dimensions.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  auto output = ctx.GetOutput(0, dimensions);
  auto output_value = Ort::Value::CreateTensor(input.GetTensorMemoryInfo(),
                      (void*)output.GetTensorMutableData<onnxruntime::MLFloat16>(),
                      input.GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(onnxruntime::MLFloat16),
                      dimensions.data(), dimensions.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

  op.Invoke(context, &input_value, 1, &output_value, 1);

  // free memory
  if (clipped_input) {
    free(clipped_input);
    clipped_input = 0;
  }
}

void NativeCastOpInvokeFromFloat16(OrtKernelContext* context, Ort::Op& op) {
  Ort::KernelContext ctx(context);

  auto input = ctx.GetInput(0);
  std::vector<int64_t> dimensions = input.GetTensorTypeAndShapeInfo().GetShape();
  auto input_value = Ort::Value::CreateTensor(input.GetTensorMemoryInfo(),
                     (void*)input.GetTensorData<onnxruntime::MLFloat16>(),
                     input.GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(onnxruntime::MLFloat16),
                     dimensions.data(), dimensions.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

  auto output = ctx.GetOutput(0, dimensions);
  auto output_value = Ort::Value::CreateTensor(input.GetTensorMemoryInfo(),
                      (void*)output.GetTensorMutableData<float>(),
                      input.GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(float),
                      dimensions.data(), dimensions.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  op.Invoke(context, &input_value, 1, &output_value, 1);
}


QUANTIZE_LINEAR_APPLY(float)
QUANTIZE_LINEAR_APPLY_FP16(float)

void KernelCustomQuantizeLinear::ComputeBase(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  const size_t num_inputs = ctx.GetInputCount();
  if (num_inputs < 3)
    ORT_CXX_API_THROW("CustomQuantizeLinear insufficient inputs.",
                      OrtErrorCode::ORT_INVALID_GRAPH);

  auto x = ctx.GetInput(0);
  auto y_scale = ctx.GetInput(1);
  auto y_zero_point = ctx.GetInput(2);

  int64_t N;
  int64_t broadcast_dim;
  int64_t block_size;
  PrepareForQDQ(x, y_scale, &y_zero_point, axis_, N, broadcast_dim, block_size);

  const float* x_data = x.GetTensorData<float>();
  const float* y_scale_data = y_scale.GetTensorData<float>();

  auto dimensions = x.GetTensorTypeAndShapeInfo().GetShape();
  auto y = ctx.GetOutput(0, dimensions);

  ONNXTensorElementDataType zp_type = y_zero_point.GetTensorTypeAndShapeInfo().GetElementType();
  switch(zp_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
      const uint8_t* y_zero_point_data = y_zero_point.GetTensorData<uint8_t>();
      uint8_t* y_data = y.GetTensorMutableData<uint8_t>();
      QuantizeLinearApply<uint8_t>().op(N, broadcast_dim, block_size,
		    x_data, y_scale_data, y_data, y_zero_point_data);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
      const int8_t* y_zero_point_data = y_zero_point.GetTensorData<int8_t>();
      int8_t* y_data = y.GetTensorMutableData<int8_t>();
      QuantizeLinearApply<int8_t>().op(N, broadcast_dim, block_size,
		    x_data, y_scale_data, y_data, y_zero_point_data);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
      const uint16_t* y_zero_point_data = y_zero_point.GetTensorData<uint16_t>();
      uint16_t* y_data = y.GetTensorMutableData<uint16_t>();
      QuantizeLinearApply<uint16_t>().op(N, broadcast_dim, block_size,
		    x_data, y_scale_data, y_data, y_zero_point_data);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
      const int16_t* y_zero_point_data = y_zero_point.GetTensorData<int16_t>();
      int16_t* y_data = y.GetTensorMutableData<int16_t>();
      QuantizeLinearApply<int16_t>().op(N, broadcast_dim, block_size,
		    x_data, y_scale_data, y_data, y_zero_point_data);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
      const uint32_t* y_zero_point_data = y_zero_point.GetTensorData<uint32_t>();
      uint32_t* y_data = y.GetTensorMutableData<uint32_t>();
      QuantizeLinearApply<uint32_t>().op(N, broadcast_dim, block_size,
                    x_data, y_scale_data, y_data, y_zero_point_data);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      const int32_t* y_zero_point_data = y_zero_point.GetTensorData<int32_t>();
      int32_t* y_data = y.GetTensorMutableData<int32_t>();
      QuantizeLinearApply<int32_t>().op(N, broadcast_dim, block_size,
                    x_data, y_scale_data, y_data, y_zero_point_data);
      break;
    }
#ifdef USE_NATIVE_OP_CAST
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      NativeCastOpInvokeToFloat16(context, cast_to_float16_);
      break;
    }
#else
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      const onnxruntime::MLFloat16* y_zero_point_data = y_zero_point.GetTensorData<onnxruntime::MLFloat16>();
      onnxruntime::MLFloat16* y_data = y.GetTensorMutableData<onnxruntime::MLFloat16>();
      QuantizeLinearApplyFp16<onnxruntime::MLFloat16>().op(N, broadcast_dim, block_size,
		    x_data, y_scale_data, y_data, y_zero_point_data);
      break;
    }
#endif
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: {
      const onnxruntime::BFloat16* y_zero_point_data = y_zero_point.GetTensorData<onnxruntime::BFloat16>();
      onnxruntime::BFloat16* y_data = y.GetTensorMutableData<onnxruntime::BFloat16>();
      QuantizeLinearApplyFp16<onnxruntime::BFloat16>().op(N, broadcast_dim, block_size,
		    x_data, y_scale_data, y_data, y_zero_point_data);
      break;
    }
    default: {
      ORT_CXX_API_THROW("CustomQuantizeLinear does not support this data type yet.",
                        OrtErrorCode::ORT_INVALID_GRAPH);
      break;
    }
  }
};


DEQUANTIZE_LINEAR_APPLY(float)
DEQUANTIZE_LINEAR_APPLY_FP16(float)

void KernelCustomDequantizeLinear::ComputeBase(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  const size_t num_inputs = ctx.GetInputCount();
  if (num_inputs < 3)
    ORT_CXX_API_THROW("CustomDequantizeLinear insufficient inputs.",
		      OrtErrorCode::ORT_INVALID_GRAPH);

  auto x = ctx.GetInput(0);
  auto x_scale = ctx.GetInput(1);
  auto x_zero_point = ctx.GetInput(2);

  int64_t N;
  int64_t broadcast_dim;
  int64_t block_size;
  PrepareForQDQ(x, x_scale, &x_zero_point, axis_, N, broadcast_dim, block_size);

  const float* x_scale_data = x_scale.GetTensorData<float>();

  auto dimensions = x.GetTensorTypeAndShapeInfo().GetShape();
  auto y = ctx.GetOutput(0, dimensions);
  float* y_data = y.GetTensorMutableData<float>();

  ONNXTensorElementDataType zp_type = x_zero_point.GetTensorTypeAndShapeInfo().GetElementType();
  switch(zp_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
      const uint8_t* x_data = x.GetTensorData<uint8_t>();
      const uint8_t* x_zero_point_data = x_zero_point.GetTensorData<uint8_t>();
      DequantizeLinearApply<uint8_t>().op(N, broadcast_dim, block_size,
                    x_data, x_scale_data, y_data, x_zero_point_data);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
      const int8_t* x_data = x.GetTensorData<int8_t>();
      const int8_t* x_zero_point_data = x_zero_point.GetTensorData<int8_t>();
      DequantizeLinearApply<int8_t>().op(N, broadcast_dim, block_size,
                    x_data, x_scale_data, y_data, x_zero_point_data);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
      const uint16_t* x_data = x.GetTensorData<uint16_t>();
      const uint16_t* x_zero_point_data = x_zero_point.GetTensorData<uint16_t>();
      DequantizeLinearApply<uint16_t>().op(N, broadcast_dim, block_size,
                    x_data, x_scale_data, y_data, x_zero_point_data);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
      const int16_t* x_data = x.GetTensorData<int16_t>();
      const int16_t* x_zero_point_data = x_zero_point.GetTensorData<int16_t>();
      DequantizeLinearApply<int16_t>().op(N, broadcast_dim, block_size,
                    x_data, x_scale_data, y_data, x_zero_point_data);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
      const uint32_t* x_data = x.GetTensorData<uint32_t>();
      const uint32_t* x_zero_point_data = x_zero_point.GetTensorData<uint32_t>();
      DequantizeLinearApply<uint32_t>().op(N, broadcast_dim, block_size,
                    x_data, x_scale_data, y_data, x_zero_point_data);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      const int32_t* x_data = x.GetTensorData<int32_t>();
      const int32_t* x_zero_point_data = x_zero_point.GetTensorData<int32_t>();
      DequantizeLinearApply<int32_t>().op(N, broadcast_dim, block_size,
                    x_data, x_scale_data, y_data, x_zero_point_data);
      break;
    }
#ifdef USE_NATIVE_OP_CAST
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      NativeCastOpInvokeFromFloat16(context, cast_from_float16_);
      break;
    }
#else
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      const onnxruntime::MLFloat16* x_data = x.GetTensorData<onnxruntime::MLFloat16>();
      const onnxruntime::MLFloat16* x_zero_point_data = x_zero_point.GetTensorData<onnxruntime::MLFloat16>();
      DequantizeLinearApplyFp16<onnxruntime::MLFloat16>().op(N, broadcast_dim, block_size,
                    x_data, x_scale_data, y_data, x_zero_point_data);
      break;
    }
#endif
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: {
      const onnxruntime::BFloat16* x_data = x.GetTensorData<onnxruntime::BFloat16>();
      const onnxruntime::BFloat16* x_zero_point_data = x_zero_point.GetTensorData<onnxruntime::BFloat16>();
      DequantizeLinearApplyFp16<onnxruntime::BFloat16>().op(N, broadcast_dim, block_size,
                    x_data, x_scale_data, y_data, x_zero_point_data);
      break;
    }
    default: {
      ORT_CXX_API_THROW("CustomDequantizeLinear does not support this data type yet.",
                        OrtErrorCode::ORT_INVALID_GRAPH);
      break;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////

KernelCustomQuantizeLinear::KernelCustomQuantizeLinear(const OrtApi& api,
		                                       const OrtKernelInfo* info) : api_(api) {
  Ort::ConstKernelInfo const_info{info};
  info_ = const_info.Copy();

  auto status = api_.KernelInfoGetAttribute_int64(info_, "axis", &axis_);
  if (status != nullptr) axis_ = 1;

#ifdef USE_NATIVE_OP_CAST
  cast_to_float16_ = NativeCastOpCreate(info_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
		                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
#endif
};

KernelCustomQuantizeLinear::~KernelCustomQuantizeLinear() {
};

void KernelCustomQuantizeLinear::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  auto scale = ctx.GetInput(1);
  ONNXTensorElementDataType scale_type = scale.GetTensorTypeAndShapeInfo().GetElementType();
  if (scale_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    ORT_CXX_API_THROW("CustomQuantizeLinear does not support non-float input.",
                      OrtErrorCode::ORT_INVALID_GRAPH);

  return ComputeBase(context);
}


KernelCustomDequantizeLinear::KernelCustomDequantizeLinear(const OrtApi& api,
		                                           const OrtKernelInfo* info) : api_(api) {
  Ort::ConstKernelInfo const_info{info};
  info_ = const_info.Copy();

  auto status = api_.KernelInfoGetAttribute_int64(info_, "axis", &axis_);
  if (status != nullptr) axis_ = 1;

#ifdef USE_NATIVE_OP_CAST
  cast_from_float16_ = NativeCastOpCreate(info_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
		                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
#endif
};

KernelCustomDequantizeLinear::~KernelCustomDequantizeLinear() {
};

void KernelCustomDequantizeLinear::Compute(OrtKernelContext* context) {
  Ort::KernelContext ctx(context);

  auto scale = ctx.GetInput(1);
  ONNXTensorElementDataType scale_type = scale.GetTensorTypeAndShapeInfo().GetElementType();
  if (scale_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    ORT_CXX_API_THROW("CustomDequantizeLinear does not support non-float output.",
                      OrtErrorCode::ORT_INVALID_GRAPH);

  return ComputeBase(context);
};

}  // namespace vai_q
