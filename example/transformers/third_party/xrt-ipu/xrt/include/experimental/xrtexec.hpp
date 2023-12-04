/**
 * Copyright (C) 2019 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

////////////////////////////////////////////////////////////////
// This is experimental code, subject to disappear without warning
////////////////////////////////////////////////////////////////

#ifndef _XRT_XRTEXEC_H_
#define _XRT_XRTEXEC_H_
#include <memory>
#include "ert.h"
#include "xrt.h"

struct xrt_device;

namespace xrtcpp {

using index_type = uint32_t;
using value_type = uint32_t;
using addr_type = uint32_t;

/**
 * Open a context that allows the current process to use @cuidx
 *
 * A context on a CU can only be opened once for a given process
 * and should be released when process is done with the CU
 *
 * Throws on error
 */
void
acquire_cu_context(xrt_device* device, value_type cuidx);

/**
 * Release a context previously acquired
 *
 * Throws on error
 */
void
release_cu_context(xrt_device* device, value_type cuidx);

/**
 * Get underlying device handle from an xrt device
 *
 * This allows swithcing from OCL APIs to shim layer XRT APIs
 */
xclDeviceHandle
get_device_handle(const xrt_device* device);

namespace exec {

/**
 * class command : abstraction for commands executed by XRT
 */
class command
{
  struct impl;
protected:
  std::shared_ptr<impl> m_impl;

  command()
    : m_impl(0)
  {}

  command(xrt_device* dev, ert_cmd_opcode opcode);

public:
  /**
   * Execute a command
   *
   * Throws on error
   */
  void
  execute();

  void
  wait();

  bool
  completed() const;

  ert_cmd_state
  state() const;

};

/**
 * class exec_cu_command : concrete class for ERT_START_CU
 *
 * The CU register map is programmed by writing specific locations
 * in the command.   The register
 */
class exec_cu_command : public command
{
public:
  exec_cu_command(xrt_device* dev);

  /**
   * Add cu to command package pair to the command
   *
   * @cuidx: index of cu to execute
   * @value: the value to write to @addr
   *
   * Supports max 32 CUs indexed [0..31]
   */
  void
  add_cu(value_type cuidx);

  /**
   * Add command payload at index
   *
   * @index: register map index.  The index is offset from beginning 
   *   of CU register map such that index 0 is for AP ctrl.
   * @value: value to write
   */
  void
  add(index_type idx, value_type value);

  /**
   * Clear current CUs and payload if any
   *
   * This allows for reuse of this command, all entries
   * currently written will be 0'ed out as part of clear.
   */
  void
  clear();
};

/**
 * class exec_write_command : concrete class for ERT_EXEC_WRITE
 *
 * The exec write command allows XRT to values to specific
 * addresses exposed over AXI-lite.  The command value pairs
 * are written to offset associated with selected CU.  After
 * writing, the CU itself is started.
 */
class exec_write_command : public command
{
public:
  exec_write_command(xrt_device* dev);

  /**
   * Add cu to command package pair to the command
   *
   * @cuidx: index of cu to execute
   * @value: the value to write to @addr
   */
  void
  add_cu(value_type cuidx);

  /**
   * Add context identifier for this command.
   *
   * A context is value between [0..maxctx[ identifying the kernel
   * compute unit context that should be used when executing the
   * command.
   *
   * Specifying a context targeting a kernel that does not support
   * contexts is valid but has effect on execution.
   *
   * @ctx: context identifier
   */
  void
  add_ctx(uint32_t ctx);

  /**
   * Add {addr,value} pair to the command
   *
   * @addr: the address that will be written with @value
   * @value: the value to write to @addr
   */
  void
  add(addr_type addr, value_type value);

  /**
   * Clear current CUs and {addr,value} pairs if any
   */
  void
  clear();
};

}} //exec, xrtcpp
#endif
