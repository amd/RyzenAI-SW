//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"

namespace vai_q {

// InT : [float]; T : [uint8/int8, uint16/int16]
// formula is Y = X / Scale + ZeroPoint
#define QUANTIZE_LINEAR_APPLY(InT)                                                        \
  template <typename T>                                                                   \
  struct QuantizeLinearApply {                                                            \
    void op(int64_t N, int64_t broadcast_dim, int64_t block_size,                         \
     		    const InT* input, const InT* scale, T* output, const T* zero_point) { \
      for (size_t n = 0; n < static_cast<size_t>(N); n++) {                               \
        for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {              \
          auto sc = static_cast<float>(scale[bd]);                                        \
          int64_t zp = zero_point ? static_cast<int64_t>(zero_point[bd]) : 0;             \
          for (size_t bs = 0; bs < static_cast<size_t>(block_size); bs++) {               \
            int64_t temp = static_cast<int64_t>(std::nearbyint(static_cast<float>(*input++) / sc)) + zp; \
            if (temp < std::numeric_limits<T>::lowest()) temp = std::numeric_limits<T>::lowest();        \
            if (temp > std::numeric_limits<T>::max()) temp = std::numeric_limits<T>::max();              \
            *output++ = static_cast<T>(temp);                                                            \
          }                                                                               \
        }                                                                                 \
      }                                                                                   \
    }                                                                                     \
  };                                                                                      \

// InT : [float]; T : [float8]
// formula is Y = X / Scale
#define QUANTIZE_LINEAR_APPLY_FP8(InT)                                               \
  template <typename T>                                                              \
  struct QuantizeLinearApplyFp8 {                                                    \
    void op(int64_t N, int64_t broadcast_dim, int64_t block_size,                    \
                    const InT* input, const InT* scale, T* output, const T*) {       \
      for (size_t n = 0; n < static_cast<size_t>(N); n++) {                          \
        for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {         \
          auto sc = static_cast<float>(scale[bd]);                                   \
          for (size_t bs = 0; bs < static_cast<size_t>(block_size); bs++)          { \
            *output++ = T(static_cast<float>(*input++) / sc);                        \
          }                                                                          \
        }                                                                            \
      }                                                                              \
    }                                                                                \
  };                                                                                 \

// InT : [float]; T : [float16/bfloat16]
// formula is Y = X / Scale + ZeroPoint
#define QUANTIZE_LINEAR_APPLY_FP16(InT)                                              \
  template <typename T>                                                              \
  struct QuantizeLinearApplyFp16 {                                                   \
    void op(int64_t N, int64_t broadcast_dim, int64_t block_size,                    \
         const InT* input, const InT* scale, T* output, const T* zero_point) {       \
      for (size_t n = 0; n < static_cast<size_t>(N); n++) {                          \
        for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {         \
          auto sc = static_cast<float>(scale[bd]);                                   \
          T zp = zero_point ? zero_point[bd] : T(0);                                 \
          for (size_t bs = 0; bs < static_cast<size_t>(block_size); bs++)          { \
            float temp = static_cast<float>(*input++) / sc + zp.ToFloat();           \
            *output++ = T(temp);                                                     \
          }                                                                          \
        }                                                                            \
      }                                                                              \
    }                                                                                \
  };                                                                                 \


// T : [uint8/int8, uint16/int16, uint32/int32]; OutT : [float]
// formula is Y = (X - ZeroPoint) * Scale
#define DEQUANTIZE_LINEAR_APPLY(OutT)                                                                    \
  template <typename T>                                                                                  \
  struct DequantizeLinearApply {                                                                         \
    void op(int64_t N, int64_t broadcast_dim, int64_t block_size,                                        \
  		    const T* input, const OutT* scale, OutT* output, const T* zero_point) {              \
      for (size_t n = 0; n < static_cast<size_t>(N); n++) {                                              \
        for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {                             \
          auto sc = static_cast<float>(scale[bd]);                                                       \
          int64_t zp = zero_point ? static_cast<int64_t>(zero_point[bd]) : 0;                            \
          for (size_t bs = 0; bs < static_cast<size_t>(block_size); bs++) {                              \
            *output++ = static_cast<OutT>(static_cast<float>(static_cast<int64_t>(*input++) - zp) * sc); \
          }                                                                                              \
        }                                                                                                \
      }                                                                                                  \
    }                                                                                                    \
  };                                                                                                     \

// T : [float8]; OutT : [float]
// formula is Y = X * Scale
#define DEQUANTIZE_LINEAR_APPLY_FP8(OutT)                                            \
  template <typename T>                                                              \
  struct DequantizeLinearApplyFp8 {                                                  \
    void op(int64_t N, int64_t broadcast_dim, int64_t block_size,                    \
                    const T* input, const OutT* scale, OutT* output, const T*) {     \
      for (size_t n = 0; n < static_cast<size_t>(N); n++) {                          \
        for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {         \
          auto sc = static_cast<float>(scale[bd]);                                   \
          for (size_t bs = 0; bs < static_cast<size_t>(block_size); bs++, input++) { \
            *output++ = static_cast<OutT>(input->ToFloat() * sc);                    \
          }                                                                          \
        }                                                                            \
      }                                                                              \
    }                                                                                \
  };                                                                                 \

// T : [float16/bfloat16]; OutT : [float]
// formula is Y = (X - ZeroPoint) * Scale
#define DEQUANTIZE_LINEAR_APPLY_FP16(OutT)                                           \
  template <typename T>                                                              \
  struct DequantizeLinearApplyFp16 {                                                 \
    void op(int64_t N, int64_t broadcast_dim, int64_t block_size,                    \
         const T* input, const OutT* scale, OutT* output, const T* zero_point) {     \
      for (size_t n = 0; n < static_cast<size_t>(N); n++) {                          \
        for (size_t bd = 0; bd < static_cast<size_t>(broadcast_dim); bd++) {         \
          auto sc = static_cast<float>(scale[bd]);                                   \
          T zp = zero_point ? zero_point[bd] : T(0);                                 \
          for (size_t bs = 0; bs < static_cast<size_t>(block_size); bs++, input++) { \
            *output++ = static_cast<OutT>((input->ToFloat() - zp.ToFloat()) * sc);   \
          }                                                                          \
        }                                                                            \
      }                                                                              \
    }                                                                                \
  };                                                                                 \

}  // namespace vai_q
