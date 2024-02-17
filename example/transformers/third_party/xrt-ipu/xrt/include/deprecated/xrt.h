/*
 * Copyright (C) 2020, Xilinx Inc - All rights reserved
 * Copyright (C) 2022, Advanced Micro Devices - All rights reserved
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

#ifndef _XCL_XRT_DEPRECATED_H_
#define _XCL_XRT_DEPRECATED_H_

/* This header file is included from include.xrt.h, it is not
 * Not a stand-alone header file */

// Including the deprecated xcl_app_debug.h file for the xclDebugReadIPStatus
// function enum parameter definition.
#include "deprecated/xcl_app_debug.h"

#ifdef __GNUC__
# define XRT_DEPRECATED __attribute__ ((deprecated))
#else
# define XRT_DEPRECATED
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Use xbutil to reset device */
XRT_DEPRECATED
XCL_DRIVER_DLLESPEC
int
xclResetDevice(xclDeviceHandle handle, enum xclResetKind kind);

/* Not supported */
XRT_DEPRECATED
XCL_DRIVER_DLLESPEC
int
xclLockDevice(xclDeviceHandle handle);

/* Not supported */
XRT_DEPRECATED
XCL_DRIVER_DLLESPEC
int
xclUnlockDevice(xclDeviceHandle handle);

/* Use xbmgmt to flash device */
XRT_DEPRECATED
XCL_DRIVER_DLLESPEC
int
xclUpgradeFirmware(xclDeviceHandle handle, const char *fileName);

/* Use xbmgmt to flash device */
XRT_DEPRECATED
XCL_DRIVER_DLLESPEC
int
xclUpgradeFirmware2(xclDeviceHandle handle, const char *file1, const char* file2);

/* Use xbmgmt to flash device */
XRT_DEPRECATED
XCL_DRIVER_DLLESPEC
int
xclUpgradeFirmwareXSpi(xclDeviceHandle handle, const char *fileName, int index);

/* Not supported */
XRT_DEPRECATED
XCL_DRIVER_DLLESPEC
int
xclBootFPGA(xclDeviceHandle handle);

/* Not supported */
XRT_DEPRECATED
XCL_DRIVER_DLLESPEC
int
xclRemoveAndScanFPGA();

/* Use xclGetBOProperties */
XRT_DEPRECATED
static inline size_t
xclGetBOSize(xclDeviceHandle handle, xclBufferHandle boHandle)
{
    struct xclBOProperties p;
    return !xclGetBOProperties(handle, boHandle, &p) ? (size_t)p.size : (size_t)-1;
}

/* Use xclGetBOProperties */
XRT_DEPRECATED
static inline uint64_t
xclGetDeviceAddr(xclDeviceHandle handle, xclBufferHandle boHandle)
{
    struct xclBOProperties p;
    return !xclGetBOProperties(handle, boHandle, &p) ? p.paddr : (uint64_t)-1;
}

/* Use xclRegWrite */
XRT_DEPRECATED
XCL_DRIVER_DLLESPEC
size_t
xclWrite(xclDeviceHandle handle, enum xclAddressSpace space, uint64_t offset,
         const void *hostBuf, size_t size);

/* Use xclRegRead */
XRT_DEPRECATED
XCL_DRIVER_DLLESPEC
size_t
xclRead(xclDeviceHandle handle, enum xclAddressSpace space, uint64_t offset,
        void *hostbuf, size_t size);

/* Not supported */
XRT_DEPRECATED
XCL_DRIVER_DLLESPEC
int
xclRegisterInterruptNotify(xclDeviceHandle handle, unsigned int userInterrupt,
                           int fd);

/**
 * DOC: XRT Stream Queue APIs
 *
 * NOTE: ALL STREAMING APIs ARE DEPRECATED!!!! THESE WILL BE REMOVED IN
 * A FUTURE RELEASE. PLEASE PORT YOUR APPLICATION TO USE SLAVE BRIDGE
 * (ALSO KNOWN AS HOST MEMORY) FOR EQUIVALENT FUNCTIONALITY.
 *
 */

enum xclStreamContextFlags {
	/* Enum for xclQueueContext.flags */
	XRT_QUEUE_FLAG_POLLING		= (1 << 2),
};

struct xclQueueContext {
    uint32_t	type;	   /* stream or packet Queue, read or write Queue*/
    uint32_t	state;	   /* initialized, running */
    uint64_t	route;	   /* route id from xclbin */
    uint64_t	flow;	   /* flow id from xclbin */
    uint32_t	qsize;	   /* number of descriptors */
    uint32_t	desc_size; /* this might imply max inline msg size */
    uint64_t	flags;	   /* isr en, wb en, etc */
};

struct xclReqBuffer {
    union {
	char*    buf;    // ptr or,
	uint64_t va;	 // offset
    };
    uint64_t  len;
    uint64_t  buf_hdl;   // NULL when first field is buffer pointer
};

enum xclQueueRequestKind {
    XCL_QUEUE_WRITE = 0,
    XCL_QUEUE_READ  = 1,
    //More, in-line etc.
};

enum xclQueueRequestFlag {
    XCL_QUEUE_REQ_EOT			= 1 << 0,
    XCL_QUEUE_REQ_CDH			= 1 << 1,
    XCL_QUEUE_REQ_NONBLOCKING		= 1 << 2,
    XCL_QUEUE_REQ_SILENT		= 1 << 3, /* not supp. not generate event for non-blocking req */
};

struct xclQueueRequest {
    enum xclQueueRequestKind op_code;
    struct xclReqBuffer*       bufs;
    uint32_t	        buf_num;
    char*               cdh;
    uint32_t	        cdh_len;
    uint32_t		flag;
    void*		priv_data;
    uint32_t            timeout;
};

struct xclReqCompletion {
    char			resv[64]; /* reserved for meta data */
    void			*priv_data;
    size_t			nbytes;
    int				err_code;
};

/* End XRT Stream Queue APIs */

/* Not supported */
XRT_DEPRECATED
XCL_DRIVER_DLLESPEC
int
xclExecBufWithWaitList(xclDeviceHandle handle, xclBufferHandle cmdBO,
                       size_t num_bo_in_wait_list, xclBufferHandle *bo_wait_list);

/* Not supported */
XRT_DEPRECATED
XCL_DRIVER_DLLESPEC
size_t
xclDebugReadIPStatus(xclDeviceHandle handle, enum xclDebugReadType type,
                     void* debugResults);

/* 
 * This function is for internal use. We don't want outside user to use it.
 * Once the internal project move to XRT APIs. Then we can create an internal function
 * and remove this one.
 */
 /*
  * @handle: Device handle
  * @ipIndex: IP index
  * @start: the start offset of the read-only register range
  * @size: the size of the read-only register range
  * Return: 0 on success or appropriate error number
  *
  * This function is to set the read-only register range on a CU. It will be system-wide impact.
  * This is used when open a CU in shared context. It allows multiple users to call xclRegRead()
  * to access CU without impact KDS/ERT scheduling. It is not able to change the range after
  * the first xclRegRead().
  * This function returns error when called in an exclusive context.
  */
XCL_DRIVER_DLLESPEC
int
xclIPSetReadRange(xclDeviceHandle handle, uint32_t ipIndex, uint32_t start, uint32_t size);

#ifdef __cplusplus
}
#endif

#endif
