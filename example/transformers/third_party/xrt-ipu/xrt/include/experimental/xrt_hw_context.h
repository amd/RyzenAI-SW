// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
#ifndef XRT_HW_CONTEXT_H_
#define XRT_HW_CONTEXT_H_

#include "xrt/detail/config.h"
#include "xrt/detail/pimpl.h"

#include "xrt/xrt_device.h"
#include "xrt/xrt_uuid.h"

#include "xcl_hwctx.h"

#ifdef __cplusplus

#include <map>

namespace xrt {

/**
 * class hw_context -- manage hw resources
 *
 * A hardware context associates an xclbin with hardware
 * resources.  Prior to creating a context, the xclbin
 * must be registered with the device (`xrt::device::register_xclbin`)
 *
 */
class hw_context_impl;
class hw_context : public detail::pimpl<hw_context_impl>
{
public:

  /**
   * Experimental specification of QoS requirements
   *
   * Free formed key-value entry.
   *
   * Supported keys are:
   *  - gops                   // giga operations per second
   *  - fps                    // frames per second
   *  - dma_bandwidth          // gigabytes per second
   *  - latency                // ??
   *  - frame_execution_time   // ??
   *  - priority               // ??
   *
   * Currently ignore for legacy platforms
   */
  using qos_type = std::map<std::string, uint32_t>;

  /**
   * @enum access_mode - legacy access mode
   *
   * @var exclusive
   *  Create a context for exclusive access to shareable resources.
   *  Legacy compute unit access control.
   * @var shared
   *  Create a context for shared access to shareable resources
   *  Legacy compute unit access control.
   *
   * Access mode is mutually exclusive with qos
   */
  enum class access_mode : uint8_t {
    exclusive = 0,
    shared = 1
  };

public:
  /**
   * hw_context() - Constructor for empty context
   */
  hw_context() = default;

  /**
   * hw_context() - Constructor with QoS control
   *
   * @param device
   *  Device where context is created
   * @param xclbin_id
   *  UUID of xclbin that should be assigned to HW resources
   * @qos
   *  Quality of service request that should be fulfilled by the context
   */
  XRT_API_EXPORT
  hw_context(const xrt::device& device, const xrt::uuid& xclbin_id, const qos_type& qos);

  /**
   * hw_context() - Construct with specific access control
   *
   * @param device
   *  Device where context is created
   * @param xclbin_id
   *  UUID of xclbin that should be assigned to HW resources
   * @param mode
   *  Access control for the context
   */
  XRT_API_EXPORT
  hw_context(const xrt::device& device, const xrt::uuid& xclbin_id, access_mode mode);

  ///@cond
  // Undocumented construction w/o specifying qos
  // Subject to change in default qos value
  hw_context(const xrt::device& device, const xrt::uuid& xclbin_id)
    : hw_context{device, xclbin_id, access_mode::shared}
  {}
  /// @endcond

  /**
   * get_device() - Device from which context was created
   */
  XRT_API_EXPORT
  xrt::device
  get_device() const;

  /**
   * get_xclbin_uuid() - UUID of xclbin from which context was created
   */
  XRT_API_EXPORT
  xrt::uuid
  get_xclbin_uuid() const;

  /**
   * get_xclbin() - Retrieve underlying xclbin matching the UUID
   */
  XRT_API_EXPORT
  xrt::xclbin
  get_xclbin() const;

  /**
   * get_mode() - Get the context access mode
   */
  XRT_API_EXPORT
  access_mode
  get_mode() const;

  /**
   * get_memory_group_id() - Get the memory group id for this context
   *
   * @return 
   *  The memory group id to be used when allocating buffers
   *  (see xrt::bo) that are not used as kernel arguments.
   *
   * The group_id can be used when allocating a buffer specific to
   * this hw context.  If buffers are allocated as kernel arguments,
   * then the group id should be obtained specific to the kernel
   * argument for which the buffer is used (see xrt::kernel::group_id())
   */
  XRT_API_EXPORT
  uint32_t
  get_memory_group_id() const;

  ///@cond
  // Undocument experimental API to change the QoS of a hardware context
  // Subject to change or removal
  XRT_API_EXPORT
      void
      update_qos(const qos_type& qos);
  ///@endcond

public:
  /// @cond
  // Undocumented internal access to low level context handle
  // Subject to change without warning
  XRT_API_EXPORT
  explicit operator xcl_hwctx_handle () const;
  /// @endcond
};

} // namespace xrt

#else
# error xrt_hwcontext is only implemented for C++
#endif // __cplusplus

#endif
