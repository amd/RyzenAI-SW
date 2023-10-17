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

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include <maize/bd_bundle.h>

namespace amd {
/**
 * @namespace maize
 *
 * @brief
 * maize is a namespace that provides public APIs and utility functions.
 *
 * @details
 * maize provides clients with the ability to create Runners that implement APIs such as execute, and wait.
 */
namespace maize {

/**
 * @class Runner
 *
 * @brief
 * Runner is used to provide clients with inference APIs.
 *
 * @details
 * Runner provides clients with APIs such as execute, execute_async, and wait.
 */
class Runner {
public:
  class Impl;

  enum Mode {
    AIESIM,
    XRT
  };

  struct Config {
    std::string name = "";
    std::string aie_control_config = "";
    std::string xclbin = "";
    std::string simdir = "./";
    std::string simparams = "";
    bool debug = false;
    int ctx_idx = 0;
  };

  typedef void * Job; 

  Runner();

  /**
   * Construct a Runner object from a json file containing instructions.
   *
   * @param instructions
   *  File path of instructions from which to construct this runner.
   * @param config
   *  Constant reference to amd::maize::Config. This is used for defining the runner configuration.
   * @param mode
   *  The runner mode defines what backend to use. Currently we support XRT backend, and aie simulator
   */
  Runner(const std::string &instructions, const Config &config, Mode mode = Runner::XRT);

  /**
   * Construct a Runner object from a vector of bd_bundle instructions.
   *
   * @param instructions
   *  Vector of bd_bundles. These define what AIE Operations need to be performed.
   * @param config
   *  Constant reference to amd::maize::Config. This is used for defining the runner configuration.
   * @param mode
   *  The runner mode defines what backend to use. Currently we support XRT backend, and aie simulator
   */
  Runner(const std::vector<bd_bundle> &instructions, const Config &config, Mode mode = Runner::XRT);

  // <ptr, bytessize, offset, is_weight>
  Job ExecuteAsync(std::vector<std::tuple<void *, size_t, size_t, bool>> &buffers);

  // <ptr, bytessize, offset, is_weight>
  void Execute(std::vector<std::tuple<void *, size_t, size_t, bool>> &buffers, int timeout = 0);

  // <ptr, bytessize, offset, is_weight>
  void Execute(const std::vector<bd_bundle> &instructions, std::vector<std::tuple<void *, size_t, size_t, bool>> &buffers, int timeout = 0);

  void Wait(Job job, int timeout = 0);
  
private:
  std::shared_ptr<Impl> handle_;
}; // class Runner

} // namespace maize
} // namespace amd
