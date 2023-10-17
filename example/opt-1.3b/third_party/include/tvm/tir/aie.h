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
 * \file tvm/tir/aie.h
 * \brief TIR expressions.
 */

#ifndef TVM_TIR_AIE_H_
#define TVM_TIR_AIE_H_

#include <tvm/tir/expr.h>


namespace tvm {
namespace tir {

/*! \brief None. */
class NoneNode : public PrimExprNode {
 public:
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
  }

  bool SEqualReduce(const NoneNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype);
  }

  void SHashReduce(SHashReducer hash_reduce) const {}

  static constexpr const char* _type_key = "tir.NoneExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(NoneNode, PrimExprNode);
};

/*!
 * \brief Managed reference to NoneNode
 * \sa NoneNode
 */
class NoneExpr : public PrimExpr {
 public:
  TVM_DLL NoneExpr();

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(NoneExpr, PrimExpr, NoneNode);
};

} // namespace tir
} // namespace tvm

#endif
