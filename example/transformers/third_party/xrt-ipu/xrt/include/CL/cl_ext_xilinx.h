/*******************************************************************************
 * Copyright (c) 2008-2015 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
 * KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
 * SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
 *    https://www.khronos.org/registry/
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 ******************************************************************************/

/*
 * Copyright (C) 2018-2020, Xilinx Inc - All rights reserved
 */

#ifndef __CL_EXT_XILINX_H
#define __CL_EXT_XILINX_H

// Xilinx implements deprecated APIs
// Examples use deprecated APIs
// Turn off deprecation warnings
#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

// Do *not* include cl_ext.h from this directory
#ifndef _WIN32
# include_next <CL/cl_ext.h>
#else
# pragma warning( push )
# pragma warning( disable : 4201 )
# include <../include/CL/cl_ext.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**************************
* Xilinx vendor extensions*
**************************/

/**
 * struct cl_mem_ext_ptr: Xilinx specific memory extensions
 *
 * Control bank allocation of buffer object.
 *
 * @flags:    Legacy bank flag when @kernel is nullptr
 * @argidx:   Same as @flags, argument index associated with valid @kernel
 * @host_ptr: Host pointer when buffer is created with CL_MEM_USE_HOST_PTR
 * @obj:      Same as @obj
 * @param:    Same as @kernel
 * @kernel:   Kernel associated with @argidx
 *
 * The default legacy layout has overloaded use of flags and param, which
 * is redefined/aliased in interpreted layouts.
 *
 * Two usages are supported:
 *  (1) Specify bank assignment for buffer with XCL_MEM mask. Optionally
 *      use host_ptr for reusing host side buffer per CL_MEM_USE_HOST_PTR.
 *  (2) Specify that buffer is for argument @argidx of kernel @kernel.
 *      Optionally use host_ptr as for (1).
 *
 * To use cl_mem_ext_ptr_t, simply pass the struct object as the host_ptr
 * clCreateBuffer and make sure cl_mem_flags specifies CL_MEM_EXT_PTR_XILINX
 */
typedef struct cl_mem_ext_ptr_t {
  union {
    struct { // legacy layout
      unsigned int flags;   // Top 8 bits reserved for XCL_MEM_EXT flags
      void *obj;
      void *param;
    };
    struct { // interpreted legcy bank assignment
      unsigned int banks;   // Top 8 bits reserved for XCL_MEM_EXT flags
      void *host_ptr;
      void *unused1;        // nullptr required
    };
    struct { // interpreted kernel arg assignment
      unsigned int argidx;  // Top 8 bits reserved for XCL_MEM_EXT flags
      void *host_ptr_;      // use as host_ptr
      cl_kernel kernel;
    };
  };
} cl_mem_ext_ptr_t;

/* Make clCreateBuffer to interpret host_ptr argument as cl_mem_ext_ptr_t */
#define CL_MEM_EXT_PTR_XILINX                       (1 << 31)

#define CL_XILINX_UNIMPLEMENTED  -20

#define CL_MEM_REGISTER_MAP                         (1 << 27)
#ifdef PMD_OCL
# define CL_REGISTER_MAP CL_MEM_REGISTER_MAP
#endif

/* Additional cl_device_partition_property */
#define CL_DEVICE_PARTITION_BY_CONNECTIVITY         (1 << 31)

#ifdef CL_VERSION_1_0
extern cl_int
clSetCommandQueueProperty(cl_command_queue command_queue,
                          cl_command_queue_properties properties,
                          cl_bool enable,
                          cl_command_queue_properties *old_properties);
#endif

/**
 * Aquire the device address associated with a cl_mem buffer on
 * a specific device.
 *
 * CL_INVALID_MEM_OBJECT: if mem is not a valid buffer object,
 *                        or if mem is not associated with device.
 * CL_INVALID_DEVICE    : if device is not a valid device.
 * CL_INVALID_VALUE     : if address is nullptr
 * CL_INVALID_VALUE     : if sz is different from sizeof(uintptr_tr)
 */
extern CL_API_ENTRY cl_int CL_API_CALL
xclGetMemObjDeviceAddress(cl_mem mem,
                          cl_device_id device,
                          size_t sz,
                          void* address);

/**
 * Acquire FD associated with a cl_mem buffer from an exporting device.
 *
 * CL_INVALID_MEM_OBJECT: if mem is not a valid buffer object,
 *                        or if mem is not associated with any device,
 *                        or if unable to obtain FD from exporting device
 * CL_INVALID_VALUE     : if fd is nullptr
 */
extern CL_API_ENTRY cl_int CL_API_CALL
xclGetMemObjectFd(cl_mem mem,
                  int* fd); /* returned fd */


/**
 * Acquire cl_mem buffer object in this context on the importing device associated
 * with a FD from an exporting device.
 *
 * CL_INVALID_MEM_OBJECT: if unable to obtain MemObject handle from exporting device
 * CL_INVALID_DEVICE    : if device is not a valid device,
 *                        if device is not in this context.
 * CL_INVALID_VALUE     : if fd is nullptr,
 *                        if context is nullptr,
 *                        if mem address of variable for cl_mem pointer is nullptr.
 */
extern CL_API_ENTRY cl_int CL_API_CALL
xclGetMemObjectFromFd(cl_context context,
                      cl_device_id deviceid,
                      cl_mem_flags flags,
                      int fd,
                      cl_mem* mem);

// incorrectly placed in cl.h
typedef struct _cl_image_filler_xilinx {
    cl_uint                 t0;
    cl_uint                 t1;
    cl_uint                 t2;
    cl_uint                 t3;
} cl_image_fillier_xilinx;

/*
 * Low level access to XRT device for use with xrt++
 */
struct xrt_device;
extern CL_API_ENTRY struct xrt_device*
xclGetXrtDevice(cl_device_id device,
                cl_int* errcode);

/**
 * Return information about the compute units of a kernel
 *
 * @kernel
 *   Kernel object being queried for compute unit.
 * @cuid
 *   Compute unit id within @kernel object [0..numcus[
 *   The CU id must be less that number of CUs as retrieved per
 *   CL_KERNEL_COMPUTE_UNIT_COUNT with clGetKernelInfo.
 * @param_name
 *   Information to query (see list below)
 * @param_value_size
 *   Number of bytes of memory in @param_value.
 *   Size must >= size of return type.
 * @param_value
 *   Pointer to memory where result is returned.
 *   Ignored if NULL.
 * @param_value_size_ret
 *   Actual size in bytes of data copied to @param_value.
 *   Ignored if NULL.
 *
 * @XCL_COMPUTE_UNIT_NAME
 * @type: char[]
 * @return: name of compute unit
 *
 * @XCL_COMPUTE_UNIT_INDEX:
 * @type: cl_uint
 * @return: XRT scheduler index of compute unit
 *
 * @XCL_COMPUTE_UNIT_BASE_ADDRESS:
 * @type: size_t
 * @return: Base address of compute unit
 *
 * @XCL_COMPUTE_UNIT_CONNECTIONS:
 * @type: cl_ulong
 * @return: Memory connection for each compute unit argument.
 *  Number of arguments are retrieved per CL_KERNEL_NUM_ARGS
 *  with clGetKernelInfo
 */
typedef cl_uint xcl_compute_unit_info;
extern CL_API_ENTRY cl_int CL_API_CALL
xclGetComputeUnitInfo(cl_kernel             kernel,
                      cl_uint               cu_id,
                      xcl_compute_unit_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret );

#define XCL_COMPUTE_UNIT_NAME         0x1320 // name of CU
#define XCL_COMPUTE_UNIT_INDEX        0x1321 // scheduler index of CU
#define XCL_COMPUTE_UNIT_CONNECTIONS  0x1322 // connectivity
#define XCL_COMPUTE_UNIT_BASE_ADDRESS 0x1323 // base address

/*
  Host Accessible Program Scope Globals
*/

//cl_mem_flags bitfield
//accepted by the <flags> parameter of clCreateBuffer


//valid flags in cl_mem_ext_ptr_t
#define XCL_MEM_DDR_BANK0               (1<<0)
#define XCL_MEM_DDR_BANK1               (1<<1)
#define XCL_MEM_DDR_BANK2               (1<<2)
#define XCL_MEM_DDR_BANK3               (1<<3)

//-- 8 reserved bits of flags in cl_mem_ext_ptr_t
#define XCL_MEM_LEGACY                  0x0
#define XCL_MEM_TOPOLOGY                (1<<31)
#define XCL_MEM_EXT_P2P_BUFFER          (1<<30)
#define XCL_MEM_EXT_HOST_ONLY           (1<<29)

/**
 * clGetKernelInfo() - kernel information
 *
 * @CL_KERNEL_COMPUTE_UNIT_COUNT
 * @type: cl_uint
 * @return: Number of compute units associated with this kernel object
 *
 * @CL_KERNEL_INSTANCE_BASE_ADDRESS
 * @type: size_t[]
 * @return: The base address of the compute units of this kernel object
 */
#define CL_KERNEL_COMPUTE_UNIT_COUNT    0x1300
#define CL_KERNEL_INSTANCE_BASE_ADDRESS 0x1301

/**
 * clGetKernelArgInfo() - kernel argument information
 *
 * @CL_KERNEL_ARG_OFFSET
 * @type: size_t
 * Return: Address offset for specified kernel argument.
 * The returned offset is relative to the base address of
 * a compute unit associated with the kernel.
 */
#define CL_KERNEL_ARG_OFFSET            0x1320

/**
 * clGetMemObjectInfo() - Memory object information
 *
 * @CL_MEM_BANK
 * @type: int
 * Return: The memory index associated with this global
 * memory object.
 */
#define CL_MEM_BANK                     0x1109

/**
 * clGetProgramBuildInfo() - (CR962714)
 *
 * @CL_PROGRAM_TARGET_TYPE
 * @type: cl_program_target_type (see below)
 * Return: The target type for this program
 */
#define CL_PROGRAM_TARGET_TYPE          0x1110

/**
 * clGetDeviceInfo()
 *
 * @CL_DEVICE_PCIE_BDF
 * @type: char[]
 * Return: The BDF if this device is a PCIe
 *
 * @CL_DEVICE_HANDLE
 * @type: void*
 * Return: The underlying device handle for use with low
 * level XRT APIs (xrt.h)
 * 
 * @CL_DEVICE_NODMA
 * @type: cl_bool
 * Return: True if underlying device is NODMA
 *
 * @CL_DEVICE_KDMA_COUNT
 * @type: cl_uint
 * Return: Number of kernel DMA blocks supported by device
 */
#define CL_DEVICE_PCIE_BDF              0x1120  // BUS/DEVICE/FUNCTION
#define CL_DEVICE_HANDLE                0x1121  // XRT device handle
#define CL_DEVICE_NODMA                 0x1122  // NODMA device check
#define CL_DEVICE_KDMA_COUNT            0x1123  // KDMA blocks

// valid target types (CR962714)
typedef cl_uint cl_program_target_type;
#define CL_PROGRAM_TARGET_TYPE_NONE     0x0
#define CL_PROGRAM_TARGET_TYPE_HW       0x1
#define CL_PROGRAM_TARGET_TYPE_SW_EMU   0x2
#define CL_PROGRAM_TARGET_TYPE_HW_EMU   0x4

////////////////////////////////////////////////////////////////
// DEPRECATED UNUSABLE STREAMING APIs
// Decl is required for internal test code (xcl2.hpp)
// Function cannot be used as they are not defined.
extern void clCreateStream();
extern void clReleaseStream();
extern void clWriteStream();
extern void clReadStream();
extern void clCreateStreamBuffer();
extern void clReleaseStreamBuffer();
extern void clPollStreams();
////////////////////////////////////////////////////////////////  

#ifdef __cplusplus
}
#endif

#ifdef _WIN32
# pragma warning( pop )
#endif

#endif
