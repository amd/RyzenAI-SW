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
 * \brief AIE related runtime headers
 */
#ifndef TVM_RUNTIME_CONTRIB_AIE_H_
#define TVM_RUNTIME_CONTRIB_AIE_H_

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/profiling.h>

namespace tvm {
namespace runtime {
namespace aie {

/*! \brief AIE executor configuration options. */
class AieExecutorConfigNode : public Object {
 public:
  /*! 
   * \brief Whether internal storage for tensors should be disabled. This can be used if outside buffers are
   * being used instead (e.g. XRT:BOs in eager mode execution).
   */
  bool disable_internal_storage{false};

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "aie.AieExecutorConfig";
  TVM_DECLARE_FINAL_OBJECT_INFO(AieExecutorConfigNode, Object);
  
};

/*!
 * \brief Managed reference to AieExecutorConfigNode.
 * \sa AieExecutorConfigNode
 */
class AieExecutorConfig : public ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param disable_internal_storage Todo
   */
  AieExecutorConfig(bool disable_internal_storage);

  TVM_DEFINE_OBJECT_REF_METHODS(AieExecutorConfig, ObjectRef, AieExecutorConfigNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AieExecutorConfigNode);
};

}  // namespace aie
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_AIE_H_
