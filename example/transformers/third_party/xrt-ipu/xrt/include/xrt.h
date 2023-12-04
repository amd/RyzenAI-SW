/*
 * Copyright (C) 2015-2022, Xilinx Inc - All rights reserved
 * Copyright (C) 2022 Advanced Micro Devices, Inc. - All rights reserved
 * Xilinx Runtime (XRT) APIs
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

#ifndef _XCL_XRT_CORE_H_
#define _XCL_XRT_CORE_H_

#ifdef __cplusplus
#include <cstdlib>
#include <cstdint>
#else
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#endif

#if defined(_WIN32)
#ifdef XCL_DRIVER_DLL_EXPORT
#define XCL_DRIVER_DLLESPEC __declspec(dllexport)
#else
#define XCL_DRIVER_DLLESPEC __declspec(dllimport)
#endif
#else
#define XCL_DRIVER_DLLESPEC __attribute__((visibility("default")))
#endif

#if defined (_WIN32)
#define NOMINMAX
#include <windows.h>
#include "windows/types.h"
#endif

#include "xclbin.h"
#include "xclerr.h"
#include "xclhal2_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * DOC: Xilinx Runtime (XRT) Library Interface Definitions
 *
 * Header file *xrt.h* defines data structures and function signatures
 * exported by Xilinx Runtime (XRT) Library. XRT is part of software
 * stack which is integrated into Xilinx reference platform.
 */

/**
 * typedef xclDeviceHandle - opaque device handle
 *
 * A device handle of xclDeviceHandle kind is obtained by opening a
 * device. Clients pass this device handle to refer to the opened
 * device in all future interaction with XRT.
 */
typedef void * xclDeviceHandle;
#define XRT_NULL_HANDLE NULL

/*
 * typedef xclBufferHandle - opaque buffer handle
 *
 * A buffer handle of xclBufferHandle kind is obtained by allocating
 * buffer objects. The buffer handle is used by XRT APIs that operate
 * on on buffer objects.
 */
#ifdef _WIN32
typedef void * xclBufferHandle;
# define NULLBO	INVALID_HANDLE_VALUE
#else
typedef unsigned int xclBufferHandle;
# define NULLBO	0xffffffff
#endif
#define XRT_NULL_BO NULLBO

/*
 * typedef xclBufferExportHandle
 *
 * Implementation specific type representing an exported buffer handle
 * that can be passed between processes.
 */
#ifdef _WIN32
typedef void* xclBufferExportHandle;  // TBD
#define NULLBOEXPORT INVALID_HANDLE_VALUE
#else
typedef int32_t xclBufferExportHandle;
#define NULLBOEXPORT -1
#endif
#define XRT_NULL_BO_EXPORT NULLBOEXPORT

struct axlf;

/**
 * Structure used to obtain various bits of information from the device.
 */

struct xclDeviceInfo2 {
  unsigned int mMagic; // = 0X586C0C6C; XL OpenCL X->58(ASCII), L->6C(ASCII), O->0 C->C L->6C(ASCII);
  char mName[256];
  unsigned short mHALMajorVersion;
  unsigned short mHALMinorVersion;
  unsigned short mVendorId;
  unsigned short mDeviceId;
  unsigned short mSubsystemId;
  unsigned short mSubsystemVendorId;
  unsigned short mDeviceVersion;
  size_t mDDRSize;                    // Size of DDR memory
  size_t mDataAlignment;              // Minimum data alignment requirement for host buffers
  size_t mDDRFreeSize;                // Total unused/available DDR memory
  size_t mMinTransferSize;            // Minimum DMA buffer size
  unsigned short mDDRBankCount;
  unsigned short mOCLFrequency[4];
  unsigned short mPCIeLinkWidth;
  unsigned short mPCIeLinkSpeed;
  unsigned short mDMAThreads;
  unsigned short mOnChipTemp;
  unsigned short mFanTemp;
  unsigned short mVInt;
  unsigned short mVAux;
  unsigned short mVBram;
  float mCurrent;
  unsigned short mNumClocks;
  unsigned short mFanSpeed;
  bool mMigCalib;
  unsigned long long mXMCVersion;
  unsigned long long mMBVersion;
  unsigned short m12VPex;
  unsigned short m12VAux;
  unsigned long long mPexCurr;
  unsigned long long mAuxCurr;
  unsigned short mFanRpm;
  unsigned short mDimmTemp[4];
  unsigned short mSE98Temp[4];
  unsigned short m3v3Pex;
  unsigned short m3v3Aux;
  unsigned short mDDRVppBottom;
  unsigned short mDDRVppTop;
  unsigned short mSys5v5;
  unsigned short m1v2Top;
  unsigned short m1v8Top;
  unsigned short m0v85;
  unsigned short mMgt0v9;
  unsigned short m12vSW;
  unsigned short mMgtVtt;
  unsigned short m1v2Bottom;
  unsigned long long mDriverVersion;
  unsigned int mPciSlot;
  bool mIsXPR;
  unsigned long long mTimeStamp;
  char mFpga[256];
  unsigned short mPCIeLinkWidthMax;
  unsigned short mPCIeLinkSpeedMax;
  unsigned short mVccIntVol;
  unsigned short mVccIntCurr;
  unsigned short mNumCDMA;
};

/**
 *  Unused, keep for backwards compatibility
 */
enum xclBOKind {
    XCL_BO_SHARED_VIRTUAL = 0,
    XCL_BO_SHARED_PHYSICAL,
    XCL_BO_MIRRORED_VIRTUAL,
    XCL_BO_DEVICE_RAM,
    XCL_BO_DEVICE_BRAM,
    XCL_BO_DEVICE_PREALLOCATED_BRAM,
};

enum xclBOSyncDirection {
    XCL_BO_SYNC_BO_TO_DEVICE = 0,
    XCL_BO_SYNC_BO_FROM_DEVICE,
    XCL_BO_SYNC_BO_GMIO_TO_AIE,
    XCL_BO_SYNC_BO_AIE_TO_GMIO,
};

/**
 * Define address spaces on the device AXI bus. The enums are used in
 * xclRead() and xclWrite() to pass relative offsets.
 */

enum xclAddressSpace {
    XCL_ADDR_SPACE_DEVICE_FLAT = 0,     // Absolute address space
    XCL_ADDR_SPACE_DEVICE_RAM = 1,      // Address space for the DDR memory
    XCL_ADDR_KERNEL_CTRL = 2,           // Address space for the OCL Region control port
    XCL_ADDR_SPACE_DEVICE_PERFMON = 3,  // Address space for the Performance monitors
    XCL_ADDR_SPACE_DEVICE_REG     = 4,  // Address space for device registers.
    XCL_ADDR_SPACE_DEVICE_CHECKER = 5,  // Address space for protocol checker
    XCL_ADDR_SPACE_MAX = 8
};


/**
 * Defines log message severity levels for messages sent to log file
 * with xclLogMsg cmd
 */

enum xrtLogMsgLevel {
     XRT_EMERGENCY = 0,
     XRT_ALERT = 1,
     XRT_CRITICAL = 2,
     XRT_ERROR = 3,
     XRT_WARNING = 4,
     XRT_NOTICE = 5,
     XRT_INFO = 6,
     XRT_DEBUG = 7
};

/**
 * Defines verbosity levels which are passed to xclOpen during device
 * creation time
 */

enum xclVerbosityLevel {
    XCL_QUIET = 0,
    XCL_INFO = 1,
    XCL_WARN = 2,
    XCL_ERROR = 3
};

enum xclResetKind {
    XCL_RESET_KERNEL, // not implemented through xocl user pf
    XCL_RESET_FULL,   // not implemented through xocl user pf
    XCL_USER_RESET
};

#define XCL_DEVICE_USAGE_COUNT 8
struct xclDeviceUsage {
    size_t h2c[XCL_DEVICE_USAGE_COUNT];
    size_t c2h[XCL_DEVICE_USAGE_COUNT];
    size_t ddrMemUsed[XCL_DEVICE_USAGE_COUNT];
    unsigned int ddrBOAllocated[XCL_DEVICE_USAGE_COUNT];
    unsigned int totalContexts;
    uint64_t xclbinId[4];
    unsigned int dma_channel_cnt;
    unsigned int mm_channel_cnt;
    uint64_t memSize[XCL_DEVICE_USAGE_COUNT];
};

struct xclBOProperties {
    uint32_t handle;
    uint32_t flags;
    uint64_t size;
    uint64_t paddr;
    int reserved; // not implemented
};

/**
 * DOC: XRT Device Management APIs
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

/**
 * xclProbe() - Enumerate devices found in the system
 *
 * Return: count of devices found
 */
XCL_DRIVER_DLLESPEC
unsigned int
xclProbe();

/**
 * xclOpen() - Open a device and obtain its handle.
 *
 * @deviceIndex:   Slot number of device 0 for first device, 1 for the second device...
 * @logFileName:   Unused, logging is controlled via xrt.ini
 * @level:         Unused, verbosity is controlled via xrt.ini
 *
 * Return:         Device handle
 */
XCL_DRIVER_DLLESPEC
xclDeviceHandle
xclOpen(unsigned int deviceIndex, const char* unused1, enum xclVerbosityLevel unused2);

/**
 * xclClose() - Close an opened device
 *
 * @handle:        Device handle
 */
XCL_DRIVER_DLLESPEC
void
xclClose(xclDeviceHandle handle);

/**
 * xclGetDeviceInfo2() - Obtain various bits of information from the device
 *
 * @handle:        Device handle
 * @info:          Information record
 * Return:         0 on success or appropriate error number
 */
XCL_DRIVER_DLLESPEC
int
xclGetDeviceInfo2(xclDeviceHandle handle, struct xclDeviceInfo2 *info);

/**
 * xclGetUsageInfo() - Obtain usage information from the device
 *
 * @handle:        Device handle
 * @info:          Information record
 * Return:         0 on success or appropriate error number
 */
XCL_DRIVER_DLLESPEC
int
xclGetUsageInfo(xclDeviceHandle handle, struct xclDeviceUsage *info);

/**
 * xclGetErrorStatus() - Obtain error information from the device
 *
 * @handle:        Device handle
 * @info:          Information record
 * Return:         0 on success or appropriate error number
 */
XCL_DRIVER_DLLESPEC
int
xclGetErrorStatus(xclDeviceHandle handle, struct xclErrorStatus *info);

/**
 * xclLoadXclBin() - Download FPGA image (xclbin) to the device
 *
 * @handle:        Device handle
 * @buffer:        Pointer to device image (xclbin) in memory
 * Return:         0 on success or appropriate error number
 *
 * Download FPGA image (AXLF) to the device. The PR bitstream is
 * encapsulated inside xclbin as a section.  The xclbin may also
 * contain other sections, which are suitably handled by the driver.
 *
 * This API also downloads OVERLAY (dtbo) section (Edge only).
 */
XCL_DRIVER_DLLESPEC
int
xclLoadXclBin(xclDeviceHandle handle, const struct axlf *buffer);

/**
 * xclGetSectionInfo() - Get Information from sysfs about the downloaded xclbin sections
 *
 * @handle:        Device handle
 * @info:          Pointer to preallocated memory which will store the return value.
 * @size:          Pointer to preallocated memory which will store the return size.
 * @kind:          axlf_section_kind for which info is being queried
 * @index:         The (sub)section index for the "kind" type.
 * Return:         0 on success or appropriate error number
 *
 * Get the section information from sysfs. The index corrresponds to the (section) entry
 * of the axlf_section_kind data being queried. The info and the size contain the return
 * binary value of the subsection and its size.
 */
XCL_DRIVER_DLLESPEC
int
xclGetSectionInfo(xclDeviceHandle handle, void* info, size_t *size,
                  enum axlf_section_kind kind, int index);

/**
 * xclReClock2() - Configure PR region frequncies
 *
 * @handle:        Device handle
 * @region:        PR region (always 0)
 * @targetFreqMHz: Array of target frequencies in order for the Clock Wizards driving
 *                 the PR region
 * Return:         0 on success or appropriate error number
 */
XCL_DRIVER_DLLESPEC
int
xclReClock2(xclDeviceHandle handle, unsigned short region,
            const unsigned short *targetFreqMHz);

/**
 * xclOpenContext() - Create shared/exclusive context on compute units
 *
 * @handle:        Device handle
 * @xclbinId:      UUID of the xclbin image running on the device
 * @ipIndex:       IP index
 * @shared:        Shared access or exclusive access
 * Return:         0 on success or appropriate error number
 *
 * The context is necessary before submitting execution jobs using
 * xclExecBuf(). Contexts may be exclusive or shared. Allocation of
 * exclusive contexts on a hardware IP would succeed only if another
 * client has not already setup up a context on that hardware IP.
 * Shared contexts can be concurrently allocated by many
 * processes on the same compute units.
 */
XCL_DRIVER_DLLESPEC
int
xclOpenContext(xclDeviceHandle handle, const xuid_t xclbinId, unsigned int ipIndex,
               bool shared);

/**
 * xclCloseContext() - Close previously opened context
 *
 * @handle:        Device handle
 * @xclbinId:      UUID of the xclbin image running on the device
 * @ipIndex:       ipIndex
 * Return:         0 on success or appropriate error number
 *
 * Close a previously allocated shared/exclusive context for a hardware IP.
 */
XCL_DRIVER_DLLESPEC
int
xclCloseContext(xclDeviceHandle handle, const xuid_t xclbinId, unsigned int ipIndex);

/*
 * Get the version number. 1 => Hal1 ; 2 => Hal2
 */
XCL_DRIVER_DLLESPEC
unsigned int
xclVersion();

/* End XRT Device Management APIs */


/**
 * xclLogMsg() - Send message to log file as per settings in ini file.
 *
 * @handle:        Device handle
 * @level:         Severity level of the msg
 * @tag:           Tag supplied by the client, like "OCL", "XMA", etc.
 * @format:        Format of Msg string to write to log file
 * @...:           All other arguments as per the format
 *
 * Return:         0 on success or appropriate error number
 */
XCL_DRIVER_DLLESPEC
int
xclLogMsg(xclDeviceHandle handle, enum xrtLogMsgLevel level, const char* tag,
          const char* format, ...);

/**
 * DOC: XRT Buffer Management APIs
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Buffer management APIs are used for managing device memory and migrating buffers
 * between host and device memory
 */

/**
 * xclAllocBO() - Allocate a BO of requested size with appropriate flags
 *
 * @handle:        Device handle
 * @size:          Size of buffer
 * @unused:        This argument is ignored
 * @flags:         Specify bank information, etc
 * Return:         BO handle
 */
XCL_DRIVER_DLLESPEC
xclBufferHandle
xclAllocBO(xclDeviceHandle handle, size_t size,int unused, unsigned int flags);

/**
 * xclAllocUserPtrBO() - Allocate a BO using userptr provided by the user
 *
 * @handle:        Device handle
 * @userptr:       Pointer to 4K aligned user memory
 * @size:          Size of buffer
 * @flags:         Specify bank information, etc
 * Return:         BO handle
 */
XCL_DRIVER_DLLESPEC
xclBufferHandle
xclAllocUserPtrBO(xclDeviceHandle handle,void *userptr, size_t size,
                  unsigned int flags);

/**
 * xclFreeBO() - Free a previously allocated BO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 */
XCL_DRIVER_DLLESPEC
void
xclFreeBO(xclDeviceHandle handle, xclBufferHandle boHandle);

/**
 * xclWriteBO() - Copy-in user data to host backing storage of BO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @src:           Source data pointer
 * @size:          Size of data to copy
 * @seek:          Offset within the BO
 * Return:         0 on success or appropriate error number
 *
 * Copy host buffer contents to previously allocated device
 * memory. ``seek`` specifies how many bytes to skip at the beginning
 * of the BO before copying-in ``size`` bytes of host buffer.
 */
XCL_DRIVER_DLLESPEC
size_t
xclWriteBO(xclDeviceHandle handle, xclBufferHandle boHandle,
           const void *src, size_t size, size_t seek);

/**
 * xclReadBO() - Copy-out user data from host backing storage of BO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @dst:           Destination data pointer
 * @size:          Size of data to copy
 * @skip:          Offset within the BO
 * Return:         0 on success or appropriate error number
 *
 * Copy contents of previously allocated device memory to host
 * buffer. ``skip`` specifies how many bytes to skip from the
 * beginning of the BO before copying-out ``size`` bytes of device
 * buffer.
 */
XCL_DRIVER_DLLESPEC
size_t
xclReadBO(xclDeviceHandle handle, xclBufferHandle boHandle,
          void *dst, size_t size, size_t skip);

/**
 * xclMapBO() - Memory map BO into user's address space
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @write:         READ only or READ/WRITE mapping
 * Return:         Memory mapped buffer
 *
 * Map the contents of the buffer object into host memory
 * To unmap the buffer call xclUnmapBO().
 */
XCL_DRIVER_DLLESPEC
void*
xclMapBO(xclDeviceHandle handle, xclBufferHandle boHandle, bool write);

/**
 * xclUnmapBO() - Unmap a BO that was previously mapped with xclMapBO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @addr:          The mapped void * pointer returned from xclMapBO()
 */
XCL_DRIVER_DLLESPEC
int
xclUnmapBO(xclDeviceHandle handle, xclBufferHandle boHandle, void* addr);

/**
 * xclSyncBO() - Synchronize buffer contents in requested direction
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @dir:           To device or from device
 * @size:          Size of data to synchronize
 * @offset:        Offset within the BO
 * Return:         0 on success or standard errno
 *
 * Synchronize the buffer contents between host and device. Depending
 * on the memory model this may require DMA to/from device or CPU
 * cache flushing/invalidation
 */
XCL_DRIVER_DLLESPEC
int
xclSyncBO(xclDeviceHandle handle, xclBufferHandle boHandle,
          enum xclBOSyncDirection dir, size_t size, size_t offset);
/**
 * xclCopyBO() - Copy device buffer contents to another buffer
 *
 * @handle:        Device handle
 * @dstBoHandle:   Destination BO handle
 * @srcBoHandle:   Source BO handle
 * @size:          Size of data to synchronize
 * @dst_offset:    dst  Offset within the BO
 * @src_offset:    src  Offset within the BO
 * Return:         0 on success or standard errno
 *
 * Copy from source buffer contents to destination buffer, can be
 * device to device or device to host.  Always perform WRITE to
 * achieve better performance, destination buffer can be on device or
 * host require DMA from device
 */
XCL_DRIVER_DLLESPEC
int
xclCopyBO(xclDeviceHandle handle, xclBufferHandle dstBoHandle,
          xclBufferHandle srcBoHandle, size_t size, size_t dst_offset,
          size_t src_offset);

/**
 * xclExportBO() - Obtain DMA-BUF file descriptor for a BO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle which needs to be exported
 * Return:         File handle to the BO or standard errno
 *
 * Export a BO for import into another device or Linux subsystem which
 * accepts DMA-BUF fd This operation is backed by Linux DMA-BUF
 * framework.  The file handle must be explicitly closed when no
 * longer needed.
 */
XCL_DRIVER_DLLESPEC
xclBufferExportHandle
xclExportBO(xclDeviceHandle handle, xclBufferHandle boHandle);

/**
 * xclImportBO() - Obtain BO handle for a BO represented by DMA-BUF file descriptor
 *
 * @handle:        Device handle
 * @fd:            File handle to foreign BO owned by another device which needs to be imported
 * @flags:         Unused
 * Return:         BO handle of the imported BO
 *
 * Import a BO exported by another device.     *
 * This operation is backed by Linux DMA-BUF framework
 */
XCL_DRIVER_DLLESPEC
xclBufferHandle
xclImportBO(xclDeviceHandle handle, xclBufferExportHandle fd, unsigned int flags);

/**
 * xclGetBOProperties() - Obtain xclBOProperties struct for a BO
 *
 * @handle:        Device handle
 * @boHandle:      BO handle
 * @properties:    BO properties struct pointer
 * Return:         0 on success
 *
 * This is the prefered method for obtaining BO property information.
 */
XCL_DRIVER_DLLESPEC
int
xclGetBOProperties(xclDeviceHandle handle, xclBufferHandle boHandle,
                   struct xclBOProperties *properties);

/**
 * xclIPName2Index() - Obtain IP index by IP name
 *
 * @handle:        Device handle
 * @ipName:        IP name. usually "<kernel name>:<instance name>"
 * Return:         IP index or appropriate error number
 *
 * xclIPName2Index() should be used to obtain unique index of IP as understood by other XRT APIs
 * like xclOpenContext().
 *
 */
XCL_DRIVER_DLLESPEC
int
xclIPName2Index(xclDeviceHandle handle, const char *ipName);

/* End XRT Buffer Management APIs */

/**
 * DOC: XRT Unmanaged DMA APIs
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Unmanaged DMA APIs are for exclusive use by the debuggers and
 * tools. The APIs allow clients to read/write from/to absolute device
 * address. No checks are performed if a buffer was allocated before
 * at the specified location or if the address is valid. Users who
 * want to take over the full memory managemnt of the device memory
 * should instead use XRT sub-buffer infrastructure for fine grained
 * placement of buffers as it also provides better DMA performancce
 * and richer set of features like mmap/munmap, etc.
 *
 * The unmanaged APIs are planned to be be deprecated in future.
 */

/**
 * xclUnmgdPread() - Perform unmanaged device memory read operation
 *
 * @handle:        Device handle
 * @flags:         Unused
 * @buf:           Destination data pointer
 * @size:          Size of data to copy
 * @offset:        Absolute offset inside device
 * Return:         0 on success or appropriate error number
 *
 * This API may be used to perform DMA operation from absolute
 * location specified. As stated before this API is for use by
 * debuggers and profilers. Do not use it in your application.
 */
XCL_DRIVER_DLLESPEC
ssize_t
xclUnmgdPread(xclDeviceHandle handle, unsigned int flags, void *buf,
              size_t size, uint64_t offset);

/**
 * xclUnmgdPwrite() - Perform unmanaged device memory read operation
 *
 * @handle:        Device handle
 * @flags:         Unused
 * @buf:           Source data pointer
 * @size:          Size of data to copy
 * @offset:        Absolute offset inside device
 * Return:         0 on success or appropriate error number
 *
 * This API may be used to perform DMA operation to an absolute
 * location specified. As stated before this API is for use by
 * debuggers and profilers. Do not use it in your application.
 */
XCL_DRIVER_DLLESPEC
ssize_t
xclUnmgdPwrite(xclDeviceHandle handle, unsigned int flags, const void *buf,
               size_t size, uint64_t offset);

/* End XRT Unmanaged DMA APIs */

/*
 * DOC: XRT Register read/write APIs
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * These functions are used to read and write peripherals sitting on
 * the address map.  Note that the offset is wrt the address
 * space. The register map and address map of execution units can be
 * obtained from xclbin. Note that these APIs are multi-threading/
 * multi-process safe and no checks are performed on the read/write
 * requests.  OpenCL runtime does **not** use these APIs but instead
 * uses execution management APIs defined below.
 */

/* XRT Register read/write APIs */

/*
 * TODO:
 * Define the following APIs
 *
 * 1. Host accessible pipe APIs: pread/pwrite
 * 2. Accelerator status, start, stop APIs
 * 3. Context creation APIs to support multiple clients
 * 4. Multiple OCL Region support
 * 5. DPDK style buffer management and device polling
 *
 */

/**
 * DOC: XRT Compute Unit Execution Management APIs
 *
 * These APIs are under development. These functions will be used to
 * start compute units and wait for them to finish.
 */

/**
 * xclExecBuf() - Submit an execution request to the embedded (or software) scheduler
 *
 * @handle:        Device handle
 * @cmdBO:         BO handle containing command packet
 * Return:         0 or standard error number
 *
 * Submit an exec buffer for execution. The exec buffer layout is
 * defined by struct ert_packet which is defined in file *ert.h*. The
 * BO should been allocated with DRM_XOCL_BO_EXECBUF flag.
 */
XCL_DRIVER_DLLESPEC
int
xclExecBuf(xclDeviceHandle handle, xclBufferHandle cmdBO);

/**
 * xclExecWait() - Wait for one or more execution events on the device
 *
 * @handle:                  Device handle
 * @timeoutMilliSec:         How long to wait for
 * Return:                   Same code as poll system call
 *
 * Wait for notification from the hardware. The function essentially
 * calls "poll" system call on the driver file handle. The return
 * value has same semantics as poll system call.  If return value is >
 * 0 caller should check the status of submitted exec buffers Note
 * that if you perform wait for the same handle from multiple threads,
 * you may lose wakeup for some of them. So, use different handle in
 * different threads.
 */
XCL_DRIVER_DLLESPEC
int
xclExecWait(xclDeviceHandle handle, int timeoutMilliSec);

/* End XRT Compute Unit Execution Management APIs */


XCL_DRIVER_DLLESPEC
const struct axlf_section_header*
wrap_get_axlf_section(const struct axlf* top, enum axlf_section_kind kind);

#ifdef __cplusplus
}
#endif

#include "deprecated/xrt.h"

#endif
