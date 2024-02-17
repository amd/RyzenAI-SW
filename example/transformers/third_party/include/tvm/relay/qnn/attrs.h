/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relay/qnn/attrs.h
 * \brief Auxiliary attributes for qnn operators.
 */
#ifndef TVM_RELAY_QNN_ATTRS_H_
#define TVM_RELAY_QNN_ATTRS_H_

#include <tvm/ir/attrs.h>

#include <string>

namespace tvm {
namespace relay {
namespace qnn {

/*! \brief Attribute for requantize operator */
struct RequantizeAttrs : public tvm::AttrsNode<RequantizeAttrs> {
  int axis;
  std::string rounding;
  std::string compute_dtype;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(RequantizeAttrs, "relay.attrs.RequantizeAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe(
            "The output channel axis for channel wise quantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
    TVM_ATTR_FIELD(rounding).set_default("None").describe(
        "Defines the rounding direction when the value is midway between"
        "two representable values. There are two supported modes - UPWARD"
        "or TONEAREST. Both modes behave exactly same except at the"
        "midpoints between the two representable values. At the midpoint,"
        "UPWARD rounds towards positive infinity (for example -1.5 will be"
        "rounded to -1). TONEAREST is the standard rounding where the"
        "value is rounded away from zero at midpoints (for example, -1.5"
        "rounds to -2). More context can be found at following gblic manual"
        "https://www.gnu.org/software/libc/manual/html_node/Rounding.html.");
    TVM_ATTR_FIELD(compute_dtype)
        .set_default("None")
        .describe(
            "Specifies the data type used during requantize. Supported "
            "options: \"int64\", \"float32\", \"float64\"");
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attribute for quantize operator */
struct QuantizeAttrs : public tvm::AttrsNode<QuantizeAttrs> {
  DataType out_dtype;
  int axis;
  std::string rounding_method;

  TVM_DECLARE_ATTRS(QuantizeAttrs, "relay.attrs.QuantizeAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe("Output data type, can be one of [int8 or uint8].");
    TVM_ATTR_FIELD(axis)
        .describe(
            "The output channel axis for channel wise quantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
    TVM_ATTR_FIELD(rounding_method)
        .set_default("round")
        .describe(
            "indicates how to find the \"nearest\" in quantize method"
            "Default option: round"
            "Available options are round and floor");
  }
};

struct SimulatedQuantizeAttrs : public tvm::AttrsNode<SimulatedQuantizeAttrs> {
  int axis;

  TVM_DECLARE_ATTRS(SimulatedQuantizeAttrs, "relay.attrs.SimulatedQuantizeAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe(
            "The output channel axis for channel wise quantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
  }
};

/*! \brief Attribute for dequantize operator */
struct DequantizeAttrs : public tvm::AttrsNode<DequantizeAttrs> {
  DataType out_dtype;
  int axis;

  TVM_DECLARE_ATTRS(DequantizeAttrs, "relay.attrs.DequantizeAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe("Output data type, can be one of [float16, float32].");
    TVM_ATTR_FIELD(axis)
        .describe(
            "The channel axis for channel wise dequantization. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
  }
};

/*! \brief Attributes used in qnn layer_norm operator */
struct QnnAieLayerNormAttrs : public tvm::AttrsNode<QnnAieLayerNormAttrs> {
  int axis;
  double epsilon;
  bool has_affine;
  double scale_in;
  int zp_in;
  double scale_ln;
  int zp_ln;
  double scale_w;
  int zp_w;
  double scale_b;
  int zp_b;
  double scale_affine;
  int zp_affine;

  bool center;
  bool scale;
  int affine_shift;
  int shift_w;
  int shift_b;

  TVM_DECLARE_ATTRS(QnnAieLayerNormAttrs, "relay.attrs.QnnAieLayerNormAttrs") {
    TVM_ATTR_FIELD(axis).set_default(-1).describe("Specify which shape axis denotes the channel.");
    TVM_ATTR_FIELD(epsilon).set_default(1e-5).describe(
        "Small float added to variance to avoid dividing by zero");
    TVM_ATTR_FIELD(has_affine).set_default(false).describe("flag if has affine part.");
    TVM_ATTR_FIELD(scale_in).set_default(1.0).describe("float scale of ln input.");
    TVM_ATTR_FIELD(zp_in).set_default(0).describe("zero point of ln input.");
    TVM_ATTR_FIELD(scale_ln).set_default(1.0).describe("float scale of ln output after divide op.");
    TVM_ATTR_FIELD(zp_ln).set_default(0).describe("zero point of ln output after divide op.");
    TVM_ATTR_FIELD(scale_w).set_default(1.0).describe("float scale of weights input.");
    TVM_ATTR_FIELD(zp_w).set_default(0).describe("zero point of weights input.");
    TVM_ATTR_FIELD(scale_b).set_default(1.0).describe("float scale of bias input.");
    TVM_ATTR_FIELD(zp_b).set_default(0).describe("zero point of bias input.");
    TVM_ATTR_FIELD(scale_affine)
        .set_default(1.0)
        .describe("float scale of affine part output after bias_add op.");
    TVM_ATTR_FIELD(zp_affine).set_default(0).describe(
        "zero point of affine part output after bias_add op.");
    TVM_ATTR_FIELD(center).set_default(true).describe(
        "If true, add offset of beta to normalized tensor; "
        "otherwise, beta is ignored.");
    TVM_ATTR_FIELD(scale).set_default(true).describe(
        "If true, multiply by gamma; otherwise, gamma is ignored.");
    TVM_ATTR_FIELD(affine_shift)
        .set_default(0)
        .describe("shift value for quantization of affine result.");
    TVM_ATTR_FIELD(shift_w).set_default(0).describe("shift value for quantization of ln result.");
    TVM_ATTR_FIELD(shift_b).set_default(0).describe(
        "shift value for quantization of affine result.");
  }
};  // struct QnnAieLayerNormAttrs

/*! \brief Attributes used in softmax asr operators */
struct QnnAieSoftmaxAttrs : public tvm::AttrsNode<QnnAieSoftmaxAttrs> {
  int axis;

  TVM_DECLARE_ATTRS(QnnAieSoftmaxAttrs, "relay.attrs.QnnAieSoftmaxAttrs") {
    TVM_ATTR_FIELD(axis).set_default(-1).describe("The axis to sum over when computing softmax.");
  }
};  // struct QnnAieSoftmaxAttrs

/*! \brief Attributes for aie transposed dense operator */
struct QnnAieMatmulAttrs : public tvm::AttrsNode<QnnAieMatmulAttrs> {
  IndexExpr units;
  DataType out_dtype;
  Array<String> weight_packing;
  bool transpose_a;
  bool transpose_b;
  tvm::String auto_scheduler_rewritten_layout;  // The layout after auto-scheduler's layout rewrite

  TVM_DECLARE_ATTRS(QnnAieMatmulAttrs, "relay.attrs.QnnAieMatmulAttrs") {    
    TVM_ATTR_FIELD(units).describe("Number of hidden units of the dense transformation.");

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");

    TVM_ATTR_FIELD(weight_packing)
        .set_default(Array<String>())
        .describe("The weight packing identifier as an array of dimensional identifiers (TN, N0, TK, K0 etc).");

    TVM_ATTR_FIELD(transpose_a)
        .set_default(false)
        .describe("Whether the first input tensor is in transposed format.");

    TVM_ATTR_FIELD(transpose_b)
        .set_default(false)
        .describe("Whether the second input tensor is in transposed format.");
  }
};


/*! \brief Attributes for aie transposed dense operator */
struct QnnAieMatmulQuantAttrs : public tvm::AttrsNode<QnnAieMatmulQuantAttrs> {
  IndexExpr units;
  DataType out_dtype;
  bool transpose_a;
  bool transpose_b;

  tvm::String auto_scheduler_rewritten_layout;  // The layout after auto-scheduler's layout rewrite

  TVM_DECLARE_ATTRS(QnnAieMatmulQuantAttrs, "relay.attrs.QnnAieMatmulQuantAttrs") {    
    TVM_ATTR_FIELD(units).describe("Number of hidden units of the dense transformation.");

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");

    TVM_ATTR_FIELD(transpose_a)
        .set_default(false)
        .describe("Whether the first input tensor is in transposed format.");

    TVM_ATTR_FIELD(transpose_b)
        .set_default(false)
        .describe("Whether the second input tensor is in transposed format.");
  }
};
  
/*! \brief Attribute for broadcast operator */
struct BroadcastAttrs : public tvm::AttrsNode<BroadcastAttrs> {
  int lhs_axis;
  int rhs_axis;

  TVM_DECLARE_ATTRS(BroadcastAttrs, "relay.attrs.BroadcastAttrs") {
    TVM_ATTR_FIELD(lhs_axis)
        .describe(
            "The channel axis for channel wise broadcast. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
    TVM_ATTR_FIELD(rhs_axis)
        .describe(
            "The channel axis for channel wise broadcast. Default value is -1,"
            "which corresponds to the last axis.")
        .set_default(-1);
  }
};

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_QNN_ATTRS_H_
