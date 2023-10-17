// Copyright© 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or Implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <mutex>

namespace amd {

namespace maize {

constexpr auto KERNEL_NAME = "DPU";
const unsigned MAX_INSTR_BUFSZ = 50331648;  // 48MB
const unsigned MAX_WEIGHTS_BUFSZ = std::getenv("MAIZE_MAX_WEIGHTS_BUFSZ")? atoi(std::getenv("MAIZE_MAX_WEIGHTS_BUFSZ")) : 150331648;  // 148MB
const bool SEPARATE_WEIGHTS_BO = true;
constexpr size_t ALIGN = 32 * 1024;

// Use to ensure buffers are at aligned offsets

#define ALIGNED_SIZE(size, align) (size % align) ? size + (align - (size % align)) : size;
constexpr int STATIC = 0;
constexpr int DYNAMIC = 1;
const unsigned DYNAMIC_REGION_OFFSET = 0x2d00000;

// One XrtContext (Singleton) will be shared by all Runners

class XrtContext {
 public:
  static XrtContext& GetInstance(const std::string* xclbin = nullptr);
  // Getters
  xrt::device& GetDevice();
  xrt::hw_context& GetContext(int ctx_idx);
  xrt::xclbin& GetXclbin(int ctx_idx);
  xrt::kernel& GetKernel(int ctx_idx);
  xrt::bo& GetInstrBo(int ctx_idx);
  xrt::bo CreateInstrSubBoSlice(int ctx_idx, size_t size, int region);
  xrt::bo& GetWeightsBo(int ctx_idx);
  std::tuple<xrt::bo, size_t> CreateWeightsSubBoSlice(int ctx_idx, size_t size);
  void GetOrCreateNewContext(int ctx_idx, const std::string* xclbin);

 private:
  XrtContext(const std::string* xclbin);  // XrtContext is a Singleton so we hide the constructor

 protected:
  const size_t dynamic_region_offset_;  // Offset of dynamic region within parent instruction bo
  xrt::device device_;
  std::vector<int> ctx_indices_;
  std::unordered_map<int, xrt::xclbin> xclbins_;
  std::unordered_map<int, xrt::xclbin> uuids_;
  std::unordered_map<int, xrt::hw_context> hw_contexts_;
  std::unordered_map<int, xrt::kernel> kernels_;
  std::unordered_map<int, xrt::bo> instr_bos_;
  std::unordered_map<int, std::unordered_map<size_t, xrt::bo>> instr_bo_static_region_offsets_;
  std::unordered_map<int, xrt::bo> weights_bos_;
  std::unordered_map<int, std::unordered_map<size_t, xrt::bo>> weights_bo_region_offsets_;
  std::mutex ctxIdx_mtx_;

};

}  // namespace maize

}  // namespace amd
