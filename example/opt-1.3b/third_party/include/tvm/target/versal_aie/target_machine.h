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
 * \file tvm/auto_scheduler/auto_schedule.h
 * \brief The user interface of the auto scheduler.
 */

#ifndef TVM_TARGET_VERSAL_AIE_TARGET_MACHINE_H_
#define TVM_TARGET_VERSAL_AIE_TARGET_MACHINE_H_

#include <tvm/auto_scheduler/measure.h>
#include <tvm/auto_scheduler/search_policy.h>

#include <utility>

namespace tvm {
namespace versal_aie {

/*! \brief Target machine information */
class TargetMachineFacadeNode : public Object {
 public:
  /*! \brief The number of build processes to run in parallel */
  int n_parallel;
  /*! \brief Timeout of a build */
  int timeout;

  /*!
   * \brief Build programs and return results.
   * \param inputs An Array of MeasureInput.
   * \param verbose Verbosity level. 0 for silent, 1 to output information during program
   * building.
   * \return An Array of MeasureResult.
   */
  // virtual Array<BuildResult> Build(const Array<MeasureInput>& inputs, int verbose) = 0;

  static constexpr const char* _type_key = "versal_aie.TargetMachineFacade";
  TVM_DECLARE_BASE_OBJECT_INFO(TargetMachineFacadeNode, Object);
};

/*!
 * \brief Managed reference to TargetMachineFacadeNode.
 * \sa TargetMachineFacadeNode
 */
class TargetMachineFacade : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TargetMachineFacade, ObjectRef, TargetMachineFacadeNode);
};

}  // namespace versal_aie
}  // namespace tvm

#endif  // TVM_TARGET_VERSAL_AIE_TARGET_MACHINE_H_
