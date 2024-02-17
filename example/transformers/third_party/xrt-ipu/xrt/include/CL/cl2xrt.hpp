/**
 * Copyright (C) 2021 Xilinx, Inc
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
#ifndef XRT_CL2XRT_HPP
#define XRT_CL2XRT_HPP

#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#include <CL/cl.h>

#if defined(_WIN32)
# ifdef XOCL_SOURCE
#  define XOCL_EXPORT __declspec(dllexport)
# else
#  define XOCL_EXPORT __declspec(dllimport)
# endif
#else
# define XOCL_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
// C++ extensions that allow host applications to move from OpenCL
// objects to corresponding XRT native C++ objects
namespace xrt { namespace opencl {

/**
 * get_xrt_device() - Retrieve underlying xrt::device object
 *
 * @device: OpenCL device ID
 * Return:  xrt::device object associated with the OpenCL device
 */
XOCL_EXPORT
xrt::device
get_xrt_device(cl_device_id device);

/**
 * get_xrt_bo() - Retrieve underlying xrt::bo object
 *
 * @device: OpenCL device ID
 * @mem:    OpenCL memory object to convert to xrt::bo
 * Return:  xrt::bo object associated with the OpenCL buffer
 *
 * OpenCL memory objects are created in a cl_context and are
 * not uniquely associated with one single device.  It is
 * possible the cl_mem buffer has not been associated with a 
 * device and if so, the returned xrt::bo object is empty.
 */
XOCL_EXPORT
xrt::bo
get_xrt_bo(cl_device_id device, cl_mem mem);

/**
 * get_xrt_kernel() - Retrieve underlying xrt::kernel object
 *
 * @device: OpenCL device ID
 * @kernel: OpenCL kernel object to convert to xrt::kernel
 * Return:  xrt::kernel object associated with the OpenCL kernel
 *
 * OpenCL kernel objects are created in a cl_context and are not
 * uniquely associated with a single cl_device.  However, a
 * xrt::kernel objects is created for each device in the context in
 * which the cl_kernel objects was created.  This function returns the
 * xrt::kernel object for specified device and kernel pair.
 */
XOCL_EXPORT
xrt::kernel
get_xrt_kernel(cl_device_id device, cl_kernel kernel);

/**
 * get_xrt_run() - Retrieve underlying xrt::run object
 *
 * @device: OpenCL device ID
 * @kernel: OpenCL kernel object to convert to xrt::run
 * Return:  xrt::run object associated with the OpenCL kernel
 *
 * OpenCL kernel object is associated with an xrt::kernel object for
 * each device in the context in which the cl_kernel was created.
 * This function returns the xrt::run object corresponding to the
 * xrt::kernel object that in turn is associated with the cl_device
 * and cl_kernel pair.
 *
 * The returned run object reflects any scalar argument that were set
 * on the cl_kernel object, but does not reflect the global memory
 * objects as these are not mapped to a device until the enqueue
 * operation.
 *
 * The returned run object is cloned and detached from the cl_kernel
 * meaning that any changes to the run object are not reflected
 * in the cl_kernel object.
 */
XOCL_EXPORT
xrt::run
get_xrt_run(cl_device_id device, cl_kernel kernel);

}} // opencl, xrt

#endif // __cplusplus

#endif
