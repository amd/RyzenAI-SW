"""
 Copyright (C) 2019-2021 Xilinx, Inc

 ctypes based Python binding for XRT

 Licensed under the Apache License, Version 2.0 (the "License"). You may
 not use this file except in compliance with the License. A copy of the
 License is located at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 License for the specific language governing permissions and limitations
 under the License.
"""

import os
import sys
import errno
import ctypes
import warnings
from numbers import Integral

from xclbin_binding import *
from ert_binding import *

libcore = ctypes.CDLL(os.environ['XILINX_XRT'] + "/lib/libxrt_core.so", mode=ctypes.RTLD_GLOBAL)
libcoreutil = ctypes.CDLL(os.environ['XILINX_XRT'] + "/lib/libxrt_coreutil.so", mode=ctypes.RTLD_GLOBAL)

xclDeviceHandle = ctypes.c_void_p
xrtDeviceHandle = ctypes.c_void_p
xrtKernelHandle = ctypes.c_void_p
xrtRunHandle = ctypes.c_void_p
xrtKernelRunHandle = ctypes.c_void_p
xrtBufferHandle = ctypes.c_void_p
xrtBufferFlags = ctypes.c_ulonglong
xrtMemoryGroup = ctypes.c_uint

class xclDeviceInfo2(ctypes.Structure):
    # "_fields_" is a required keyword
    _fields_ = [
     ("mMagic", ctypes.c_uint),
     ("mName", ctypes.c_char*256),
     ("mHALMajorVersion", ctypes.c_ushort),
     ("mHALMinorVersion", ctypes.c_ushort),
     ("mVendorId", ctypes.c_ushort),
     ("mDeviceId", ctypes.c_ushort),
     ("mSubsystemId", ctypes.c_ushort),
     ("mSubsystemVendorId", ctypes.c_ushort),
     ("mDeviceVersion", ctypes.c_ushort),
     ("mDDRSize", ctypes.c_size_t),
     ("mDataAlignment", ctypes.c_size_t),
     ("mDDRFreeSize", ctypes.c_size_t),
     ("mMinTransferSize", ctypes.c_size_t),
     ("mDDRBankCount", ctypes.c_ushort),
     ("mOCLFrequency", ctypes.c_ushort*4),
     ("mPCIeLinkWidth", ctypes.c_ushort),
     ("mPCIeLinkSpeed", ctypes.c_ushort),
     ("mDMAThreads", ctypes.c_ushort),
     ("mOnChipTemp", ctypes.c_short),
     ("mFanTemp", ctypes.c_short),
     ("mVInt", ctypes.c_ushort),
     ("mVAux", ctypes.c_ushort),
     ("mVBram", ctypes.c_ushort),
     ("mCurrent", ctypes.c_float),
     ("mNumClocks", ctypes.c_ushort),
     ("mFanSpeed", ctypes.c_ushort),
     ("mMigCalib", ctypes.c_bool),
     ("mXMCVersion", ctypes.c_ulonglong),
     ("mMBVersion", ctypes.c_ulonglong),
     ("m12VPex", ctypes.c_short),
     ("m12VAux", ctypes.c_short),
     ("mPexCurr", ctypes.c_ulonglong),
     ("mAuxCurr", ctypes.c_ulonglong),
     ("mFanRpm", ctypes.c_ushort),
     ("mDimmTemp", ctypes.c_ushort*4),
     ("mSE98Temp", ctypes.c_ushort*4),
     ("m3v3Pex", ctypes.c_ushort),
     ("m3v3Aux", ctypes.c_ushort),
     ("mDDRVppBottom",ctypes.c_ushort),
     ("mDDRVppTop", ctypes.c_ushort),
     ("mSys5v5", ctypes.c_ushort),
     ("m1v2Top", ctypes.c_ushort),
     ("m1v8Top", ctypes.c_ushort),
     ("m0v85", ctypes.c_ushort),
     ("mMgt0v9", ctypes.c_ushort),
     ("m12vSW", ctypes.c_ushort),
     ("mMgtVtt", ctypes.c_ushort),
     ("m1v2Bottom", ctypes.c_ushort),
     ("mDriverVersion, ", ctypes.c_ulonglong),
     ("mPciSlot", ctypes.c_uint),
     ("mIsXPR", ctypes.c_bool),
     ("mTimeStamp", ctypes.c_ulonglong),
     ("mFpga", ctypes.c_char*256),
     ("mPCIeLinkWidthMax", ctypes.c_ushort),
     ("mPCIeLinkSpeedMax", ctypes.c_ushort),
     ("mVccIntVol", ctypes.c_ushort),
     ("mVccIntCurr", ctypes.c_ushort),
     ("mNumCDMA", ctypes.c_ushort)
    ]

# Unused, keep for backwards compatibility
class xclBOKind:
    XCL_BO_SHARED_VIRTUAL           = 0
    XCL_BO_SHARED_PHYSICAL          = 1
    XCL_BO_MIRRORED_VIRTUAL         = 2
    XCL_BO_DEVICE_RAM               = 3
    XCL_BO_DEVICE_BRAM              = 4
    XCL_BO_DEVICE_PREALLOCATED_BRAM = 5

class xclBOSyncDirection:
    XCL_BO_SYNC_BO_TO_DEVICE   = 0
    XCL_BO_SYNC_BO_FROM_DEVICE = 1

class xclAddressSpace:
    XCL_ADDR_SPACE_DEVICE_FLAT    = 0  # Absolute address space
    XCL_ADDR_SPACE_DEVICE_RAM     = 1  # Address space for the DDR memory
    XCL_ADDR_KERNEL_CTRL          = 2  # Address space for the OCL Region control port
    XCL_ADDR_SPACE_DEVICE_PERFMON = 3  # Address space for the Performance monitors
    XCL_ADDR_SPACE_DEVICE_CHECKER = 5  # Address space for protocol checker
    XCL_ADDR_SPACE_MAX = 8

# Defines log message severity levels for messages sent to log file with xclLogMsg cmd
class xrtLogMsgLevel:
    XRT_EMERGENCY = 0
    XRT_ALERT     = 1
    XRT_CRITICAL  = 2
    XRT_ERROR     = 3
    XRT_WARNING   = 4
    XRT_NOTICE    = 5
    XRT_INFO      = 6
    XRT_DEBUG     = 7

# Defines log message severity levels for messages sent to log file with xclLogMsg cmd
class xclVerbosityLevel:
    XCL_QUIET = 0
    XCL_INFO  = 1
    XCL_WARN  = 2
    XCL_ERROR = 3


class xclDeviceUsage (ctypes.Structure):
    _fields_ = [
     ("h2c", ctypes.c_size_t*8),
     ("c2h", ctypes.c_size_t*8),
     ("ddeMemUsed", ctypes.c_size_t*8),
     ("ddrBOAllocated", ctypes.c_uint *8),
     ("totalContents", ctypes.c_uint),
     ("xclbinId", ctypes.c_ulonglong),
     ("dma_channel_cnt", ctypes.c_uint),
     ("mm_channel_cnt", ctypes.c_uint),
     ("memSize", ctypes.c_ulonglong*8)
    ]

class xclBOProperties (ctypes.Structure):
    _fields_ = [
     ("handle", ctypes.c_uint),
     ("flags" , ctypes.c_uint),
     ("size", ctypes.c_ulonglong),
     ("paddr", ctypes.c_ulonglong),
     ("reserved", ctypes.c_uint), # not implemented
    ]

def _valueOrError(res):
    """
    Validate return code from XRT C library and raise an exception if necessary
    """
    # check if result is some kind of integer
    # can't do a direct comparison with an int since some
    # functions return long and Python3 dropped support for long
    if isinstance(res, Integral):
        if res < 0:
            raise OSError(abs(res), os.strerror(abs(res)))
    # check if result type is a pointer. Python3 doesn't support pointer and int comparison
    elif isinstance(res.contents, ctypes.c_void_p) and res is None:
        raise OSError(errno.ENODEV, os.strerror(errno.ENODEV))
    return res

def _xclDeprecation(message):
    warnings.warn(message + ", please use xrt prefixed APIs or pybind11 based object oriented APIs", DeprecationWarning)

def _xclStreamDeprecation(message):
    warnings.warn(message + ", streaming APIs are being deprecated. Please migrate to Host Memory (previously known as Slave Bridge) based design", DeprecationWarning)

def xclProbe():
    """
    xclProbe() - Enumerate devices found in the system
    :return: count of devices found
    """
    return libcore.xclProbe()

def xclVersion():
    """
    :return: the version number. 1 => Hal1 ; 2 => Hal2
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    return libcore.xclVersion()

def xclOpen(deviceIndex, logFileName, level):
    """
    xclOpen(): Open a device and obtain its handle

    :param deviceIndex: (unsigned int) Slot number of device 0 for first device, 1 for the second device...
    :param logFileName: (const char pointer) Log file to use for optional logging
    :param level: (int) Severity level of messages to log
    :return: device handle
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclOpen.restype = ctypes.POINTER(xclDeviceHandle)
    libcore.xclOpen.argtypes = [ctypes.c_uint, ctypes.c_char_p, ctypes.c_int]
    return _valueOrError(libcore.xclOpen(deviceIndex, logFileName, level))


def xclClose(handle):
    """
    xclClose(): Close an opened device

    :param handle: (xclDeviceHandle) device handle
    :return: None
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclClose.restype = None
    libcore.xclClose.argtype = xclDeviceHandle
    libcore.xclClose(handle)


def xclGetDeviceInfo2 (handle, info):
    """
    xclGetDeviceInfo2() - Obtain various bits of information from the device

    :param handle: (xclDeviceHandle) device handle
    :param info: (xclDeviceInfo pointer) Information record
    :return: 0 on success or appropriate error number
    """

    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclGetDeviceInfo2.restype = ctypes.c_int
    libcore.xclGetDeviceInfo2.argtypes = [xclDeviceHandle, ctypes.POINTER(xclDeviceInfo2)]
    return _valueOrError(libcore.xclGetDeviceInfo2(handle, info))


def xclGetUsageInfo (handle, info):
    """
    xclGetUsageInfo() - Obtain usage information from the device
    :param handle: Device handle
    :param info: Information record
    :return: 0 on success or appropriate error number
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclGetUsageInfo.restype = ctypes.c_int
    libcore.xclGetUsageInfo.argtypes = [xclDeviceHandle, ctypes.POINTER(xclDeviceInfo2)]
    return _valueOrError(libcore.xclGetUsageInfo(handle, info))


def xclGetErrorStatus(handle, info):
    """
    xclGetErrorStatus() - Obtain error information from the device
    :param handle: Device handle
    :param info: Information record
    :return: 0 on success or appropriate error number
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclGetErrorStatus.restype = ctypes.c_int
    libcore.xclGetErrorStatus.argtypes = [xclDeviceHandle, ctypes.POINTER(xclDeviceInfo2)]
    return _valueOrError(libcore.xclGetErrorStatus(handle, info))


def xclLoadXclBin(handle, buf):
    """
    Download FPGA image (xclbin) to the device

    :param handle: (xclDeviceHandle) device handle
    :param buf: (void pointer) Pointer to device image (xclbin) in memory
    :return: 0 on success or appropriate error number

    Download FPGA image (AXLF) to the device. The PR bitstream is encapsulated inside
    xclbin as a section. xclbin may also contains other sections which are suitably
    handled by the driver
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclLoadXclBin.restype = ctypes.c_int
    libcore.xclLoadXclBin.argtypes = [xclDeviceHandle, ctypes.c_void_p]
    return _valueOrError(libcore.xclLoadXclBin(handle, buf))


def xclGetSectionInfo(handle, info, size, kind, index):
    """
    xclGetSectionInfo() - Get Information from sysfs about the downloaded xclbin sections
    :param handle: Device handle
    :param info: Pointer to preallocated memory which will store the return value.
    :param size: Pointer to preallocated memory which will store the return size.
    :param kind: axlf_section_kind for which info is being queried
    :param index: The (sub)section index for the "kind" type.
    :return: 0 on success or appropriate error number
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclGetSectionInfo.restype = ctypes.c_int
    libcore.xclGetSectionInfo.argtypes = [xclDeviceHandle, ctypes.POINTER(xclDeviceInfo2),
                                       ctypes.POINTER(ctypes.sizeof(xclDeviceInfo2)),
                                       ctypes.c_int, ctypes.c_int]
    return _valueOrError(libcore.xclGetSectionInfo(handle, info, size, kind, index))


def xclReClock2(handle, region, targetFreqMHz):
    """
    xclReClock2() - Configure PR region frequencies
    :param handle: Device handle
    :param region: PR region (always 0)
    :param targetFreqMHz: Array of target frequencies in order for the Clock Wizards driving the PR region
    :return: 0 on success or appropriate error number
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclReClock2.restype = ctypes.c_int
    libcore.xclReClock2.argtypes = [xclDeviceHandle, ctypes.c_uint, ctypes.c_uint]
    return _valueOrError(libcore.xclReClock2(handle, region, targetFreqMHz))


def xclLockDevice(handle):
    """
    The function is NOP; it exists for backward compatiblity.
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    return 0

def xclUnlockDevice(handle):
    """
    The function is NOP; it exists for backward compatiblity.
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    return 0

def xclOpenContext(handle, xclbinId, ipIndex, shared):
    """
    xclOpenContext() - Create shared/exclusive context on compute units
    :param handle: Device handle
    :param xclbinId: UUID of the xclbin image running on the device
    :param ipIndex: IP/CU index in the IP LAYOUT array
    :param shared: Shared access or exclusive access
    :return: 0 on success or appropriate error number

    The context is necessary before submitting execution jobs using xclExecBuf(). Contexts may be
    exclusive or shared. Allocation of exclusive contexts on a compute unit would succeed
    only if another client has not already setup up a context on that compute unit. Shared
    contexts can be concurrently allocated by many processes on the same compute units.
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclOpenContext.restype = ctypes.c_int
    libcore.xclOpenContext.argtypes = [xclDeviceHandle, ctypes.c_char_p, ctypes.c_uint, ctypes.c_bool]
    return _valueOrError(libcore.xclOpenContext(handle, xclbinId.bytes, ipIndex, shared))


def xclCloseContext(handle, xclbinId, ipIndex):
    """
    xclCloseContext() - Close previously opened context
    :param handle: Device handle
    :param xclbinId: UUID of the xclbin image running on the device
    :param ipIndex: IP/CU index in the IP LAYOUT array
    :return: 0 on success or appropriate error number

    Close a previously allocated shared/exclusive context for a compute unit.
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclCloseContext.restype = ctypes.c_int
    libcore.xclCloseContext.argtypes = [xclDeviceHandle, ctypes.c_char_p, ctypes.c_uint]
    return _valueOrError(libcore.xclCloseContext(handle, xclbinId.bytes, ipIndex))


def xclLogMsg(handle, level, tag, form, *args):
    """
    Send message to log file as per settings in ini file.

    :param handle: (xclDeviceHandle) device handle
    :param level: (xrtLogMsgLevel) Severity level of the msg
    :param tag: (const char*) Tag supplied by the client, like "OCL", "XMA", etc.
    :param format: (const char *) Format of Msg string to write to log file
    :param ...: All other arguments as per the format
    :return: 0 on success or appropriate error number
    """
    libcore.xclLogMsg.restype = ctypes.c_int
    libcore.xclLogMsg.argtypes = [xclDeviceHandle, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
    return _valueOrError(libcore.xclLogMsg(handle, level, tag, form, *args))


def xclAllocBO(handle, size, unused, flags):
    """
    Allocate a BO of requested size with appropriate flags

    :param handle: (xclDeviceHandle) device handle
    :param size: (size_t) Size of buffer
    :param unused: (int) unused parameter present for legacy reasons
    :param flags: (unsigned int) Specify bank information, etc
    :return: BO handle
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclAllocBO.restype = ctypes.c_uint
    libcore.xclAllocBO.argtypes = [xclDeviceHandle, ctypes.c_size_t, ctypes.c_int, ctypes.c_uint]
    return _valueOrError(libcore.xclAllocBO(handle, size, unused, flags))


def xclAllocUserPtrBO(handle, userptr, size, flags):
    """
    Allocate a BO using userptr provided by the user
    :param handle: Device handle
    :param userptr: Pointer to 4K aligned user memory
    :param size: Size of buffer
    :param flags: Specify bank information, etc
    :return: BO handle
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclAllocUserPtrBO.restype = ctypes.c_uint
    libcore.xclAllocUserPtrBO.argtypes = [xclDeviceHandle, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    return _valueOrError(libcore.xclAllocUserPtrBO(handle, userptr, size, flags))


def xclFreeBO(handle, boHandle):
    """
    Free a previously allocated BO

    :param handle: device handle
    :param boHandle: BO handle
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclFreeBO.restype = None
    libcore.xclFreeBO.argtypes = [xclDeviceHandle, ctypes.c_uint]
    libcore.xclFreeBO(handle, boHandle)


def xclWriteBO(handle, boHandle, src, size, seek):
    """
    Copy-in user data to host backing storage of BO
    :param handle: Device handle
    :param boHandle: BO handle
    :param src: Source data pointer
    :param size: Size of data to copy
    :param seek: Offset within the BO
    :return: 0 on success or appropriate error number
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclWriteBO.restype = ctypes.c_int
    libcore.xclWriteBO.argtypes = [xclDeviceHandle, ctypes.c_uint, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    return _valueOrError(libcore.xclWriteBO(handle, boHandle, src, size, seek))


def xclReadBO(handle, boHandle, dst, size, skip):
    """
    Copy-out user data from host backing storage of BO
    :param handle: Device handle
    :param boHandle: BO handle
    :param dst: Destination data pointer
    :param size: Size of data to copy
    :param skip: Offset within the BO
    :return: 0 on success or appropriate error number
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclReadBO.restype = ctypes.c_int
    libcore.xclReadBO.argtypes = [xclDeviceHandle, ctypes.c_uint, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    return _valueOrError(libcore.xclReadBO(handle, boHandle, dst, size, skip))


def xclMapBO(handle, boHandle, write):
    """
    Memory map BO into user's address space

    :param handle: (xclDeviceHandle) device handle
    :param boHandle: (unsigned int) BO handle
    :param write: (boolean) READ only or READ/WRITE mapping
    :return: (pointer) Memory mapped buffer

    Map the contents of the buffer object into host memory
    To unmap the buffer call xclUnmapBO()

    Return type void pointer doesn't get correctly binded in ctypes
    To map the buffer, explicitly specify the type and size of data
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclMapBO.restype = ctypes.POINTER(ctypes.c_char)
    libcore.xclMapBO.argtypes = [xclDeviceHandle, ctypes.c_uint, ctypes.c_bool]
    ptr = libcore.xclMapBO(handle, boHandle, write)
    return ptr

def xclUnmapBO(handle, boHandle, addr):
    """
    Unmap a previously mapped BO from user's address space

    :param handle: (xclDeviceHandle) device handle
    :param boHandle: (unsigned int) BO handle
    :param addr: (pointer) buffer pointer

    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclUnmapBO.restype = ctypes.c_int
    libcore.xclUnmapBO.argtypes = [xclDeviceHandle, ctypes.c_uint, ctypes.c_void_p]
    return _valueOrError(libcore.xclUnmapBO(handle, boHandle, addr))


def xclSyncBO(handle, boHandle, direction, size, offset):
    """
    Synchronize buffer contents in requested direction

    :param handle: (xclDeviceHandle) device handle
    :param boHandle: (unsigned int) BO handle
    :param direction: (xclBOSyncDirection) To device or from device
    :param size: (size_t) Size of data to synchronize
    :param offset: (size_t) Offset within the BO
    :return: 0 on success or standard errno
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclSyncBO.restype = ctypes.c_int
    libcore.xclSyncBO.argtypes = [xclDeviceHandle, ctypes.c_uint, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t]
    return _valueOrError(libcore.xclSyncBO(handle, boHandle, direction, size, offset))


def xclCopyBO(handle, dstBoHandle, srcBoHandle, size, dst_offset, src_offset):
    """
    Copy device buffer contents to another buffer
    :param handle: Device handle
    :param dstBoHandle: Destination BO handle
    :param srcBoHandle: Source BO handle
    :param size: Size of data to synchronize
    :param dst_offset: dst  Offset within the BO
    :param src_offset: src  Offset within the BO
    :return: 0 on success or standard errno
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclCopyBO.restype = ctypes.c_int
    libcore.xclCopyBO.argtypes = [xclDeviceHandle, ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t, ctypes.c_size_t,
                               ctypes.c_uint]
    return _valueOrError(xclCopyBO(handle, dstBoHandle, srcBoHandle, size, dst_offset, src_offset))


def xclExportBO(handle, boHandle):
    """
    Obtain DMA-BUF file descriptor for a BO
    :param handle: Device handle
    :param boHandle: BO handle which needs to be exported
    :return: File handle to the BO or standard errno
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclExportBO.restype = ctypes.c_int
    libcore.xclExportBO.argtypes = [xclDeviceHandle, ctypes.c_uint]
    return _valueOrError(libcore.xclExportBO(handle, boHandle))


def xclImportBO(handle, fd, flags):
    """
    Obtain BO handle for a BO represented by DMA-BUF file descriptor
    :param handle: Device handle
    :param fd: File handle to foreign BO owned by another device which needs to be imported
    :param flags: Unused
    :return: BO handle of the imported BO

    Import a BO exported by another device.
    This operation is backed by Linux DMA-BUF framework
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclImportBO.restype = ctypes.c_int
    libcore.xclImportBO.argtypes = [xclDeviceHandle, ctypes.c_int, ctypes.c_uint]
    libcore.xclImportBO(handle, fd, flags)

def xclGetBOProperties(handle, boHandle, properties):
    """
    Obtain xclBOProperties struct for a BO

    :param handle: (xclDeviceHandle) device handle
    :param boHandle: (unsigned int) BO handle
    :param properties: BO properties struct pointer
    :return: 0 on success
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclGetBOProperties.restype = ctypes.c_int
    libcore.xclGetBOProperties.argtypes = [xclDeviceHandle, ctypes.c_uint, ctypes.POINTER(xclBOProperties)]
    return _valueOrError(libcore.xclGetBOProperties(handle, boHandle, properties))


def xclUnmgdPread(handle, flags, buf, size, offeset):
    """
    Perform unmanaged device memory read operation
    :param handle: Device handle
    :param flags: Unused
    :param buf: Destination data pointer
    :param size: Size of data to copy
    :param offeset: Absolute offset inside device
    :return: size of bytes read or appropriate error number

    This API may be used to perform DMA operation from absolute location specified. Users
    may use this if they want to perform their own device memory management -- not using the buffer
    object (BO) framework defined before.
    """
    libcore.xclUnmgdPread.restype = ctypes.c_size_t
    libcore.xclUnmgdPread.argtypes = [xclDeviceHandle, ctypes.c_uint, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
    return libcore.xclUnmgdPread(handle, flags, buf, size, offeset)

def xclUnmgdPwrite(handle, flags, buf, size, offset):
    """
    Perform unmanaged device memory write operation
    :param handle: Device handle
    :param flags: Unused
    :param buf: Destination data pointer
    :param size: Size of data to copy
    :param offeset: Absolute offset inside device
    :return: size of bytes read or appropriate error number

    This API may be used to perform DMA operation from absolute location specified. Users
    may use this if they want to perform their own device memory management -- not using the buffer
    object (BO) framework defined before.
    """
    libcore.xclUnmgdPwrite.restype = ctypes.c_size_t
    libcore.xclUnmgdPwrite.argtypes = [xclDeviceHandle, ctypes.c_uint, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
    return libcore.xclUnmgdPwrite(handle, flags, buf, size, offset)

def xclWrite(handle, space, offset, hostBuf, size):
    """
    Perform register write operation
    :param handle:  Device handle
    :param space: Address space
    :param offset: Offset in the address space
    :param hostBuf: Source data pointer
    :param size: Size of data to copy
    :return: size of bytes written or appropriate error number

    This API may be used to write to device registers exposed on PCIe BAR. Offset is relative to the
    the address space. A device may have many address spaces.
    This API will be deprecated in future. Please use this API only for IP bringup/debugging. For
    execution management please use XRT Compute Unit Execution Management APIs defined below
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclWrite.restype = ctypes.c_size_t
    libcore.xclWrite.argtypes = [xclDeviceHandle, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    return libcore.xclWrite(handle, space, offset, hostBuf, size)

def xclRead(handle, space, offset, hostBuf, size):
    """
    Perform register write operation
    :param handle:  Device handle
    :param space: Address space
    :param offset: Offset in the address space
    :param hostBuf: Destination data pointer
    :param size: Size of data to copy
    :return: size of bytes written or appropriate error number

    This API may be used to write to device registers exposed on PCIe BAR. Offset is relative to the
    the address space. A device may have many address spaces.
    This API will be deprecated in future. Please use this API only for IP bringup/debugging. For
    execution management please use XRT Compute Unit Execution Management APIs defined below
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclRead.restype = ctypes.c_size_t
    libcore.xclRead.argtypes = [xclDeviceHandle, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    return libcore.xclRead(handle, space, offset, hostBuf, size)

def xclExecBuf(handle, cmdBO):
    """
    xclExecBuf() - Submit an execution request to the embedded (or software) scheduler
    :param handle: Device handle
    :param cmdBO: BO handle containing command packet
    :return: 0 or standard error number

    Submit an exec buffer for execution. The exec buffer layout is defined by struct ert_packet
    which is defined in file *ert.h*. The BO should been allocated with DRM_XOCL_BO_EXECBUF flag.
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclExecBuf.restype = ctypes.c_int
    libcore.xclExecBuf.argtypes = [xclDeviceHandle, ctypes.c_uint]
    return _valueOrError(libcore.xclExecBuf(handle, cmdBO))

def xclExecBufWithWaitList(handle, cmdBO, num_bo_in_wait_list, bo_wait_list):
    """
    Submit an execution request to the embedded (or software) scheduler
    :param handle: Device handle
    :param cmdBO:BO handle containing command packet
    :param num_bo_in_wait_list: Number of BO handles in wait list
    :param bo_wait_list: BO handles that must complete execution before cmdBO is started
    :return:0 or standard error number

    Submit an exec buffer for execution. The BO handles in the wait
    list must complete execution before cmdBO is started.  The BO
    handles in the wait list must have beeen submitted prior to this
    call to xclExecBufWithWaitList.
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclExecBufWithWaitList.restype = ctypes.c_int
    libcore.xclExecBufWithWaitList.argtypes = [xclDeviceHandle, ctypes.c_uint, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint)]
    return _valueOrError(libcore.xclExecBufWithWaitList(handle, cmdBO, num_bo_in_wait_list, bo_wait_list))

def xclExecWait(handle, timeoutMilliSec):
    """
    xclExecWait() - Wait for one or more execution events on the device
    :param handle: Device handle
    :param timeoutMilliSec: How long to wait for
    :return:  Same code as poll system call

    Wait for notification from the hardware. The function essentially calls "poll" system
    call on the driver file handle. The return value has same semantics as poll system call.
    If return value is > 0 caller should check the status of submitted exec buffers
    """
    _xclDeprecation(sys._getframe().f_code.co_name)
    libcore.xclExecWait.restype = ctypes.c_int
    libcore.xclExecWait.argtypes = [xclDeviceHandle, ctypes.c_int]
    return libcore.xclExecWait(handle, timeoutMilliSec)

def xclRegRead(handle, cu_index, offset, datap):
    """
    Read register in register space of a CU
    :param handle: Device handle
    :param cu_index: CU index
    :param offset: Offset in the register space
    :param datap: Pointer to where result will be saved
    :return: 0 or appropriate error number
    """
    libcore.xclRegRead.restype = ctypes.c_int
    libcore.xclRegRead.argtypes = [xclDeviceHandle, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
    return _valueOrError(libcore.xclRegRead(handle, cu_index, offset, datap))


def xclRegWrite(handle, cu_index, offset, data):
    """
    Write register in register space of a CU
    :param handle: Device handle
    :param cu_index: CU index
    :param offset: Offset in the register space
    :param data: Pointer to where result will be saved
    :return: 0 or appropriate error number
    """
    libcore.xclRegRead.restype = ctypes.c_int
    libcore.xclRegRead.argtypes = [xclDeviceHandle, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
    return _valueOrError(libcore.xclRegWrite(handle, cu_index, offset, data))


def xclDebugReadIPStatus(handle, type, debugResults):
    """

    :param handle:
    :param type:
    :param debugResults:
    :return:
    """
    libcore.xclDebugReadIPStatus.restype = ctypes.c_size_t
    libcore.xclDebugReadIPStatus.argtypes = [xclDeviceHandle, ctypes.c_int, ctypes.c_void_p]
    return libcore.xclDebugReadIPStatus(handle, type, debugResults)

def xrtDeviceOpen(deviceIndex):
    """
    xrtDeviceOpen(): Open a device and obtain its xrt device handle

    :param deviceIndex: (unsigned int) Slot number of device 0 for first device, 1 for the second device...
    """
    libcoreutil.xrtDeviceOpen.restype = ctypes.POINTER(xrtDeviceHandle)
    libcoreutil.xrtDeviceOpen.argTypes = [ctypes.c_uint]
    return _valueOrError(libcoreutil.xrtDeviceOpen(deviceIndex))

def xrtDeviceOpenByBDF(bdf):
    """
    xrtDeviceOpenByBDF(): Open a device and obtain its xrt device handle

    :param bdf: (const char *bdf) PCIe BDF identifying the device to open
    """
    libcoreutil.xrtDeviceOpenByBDF.restype = ctypes.POINTER(xrtDeviceHandle)
    libcoreutil.xrtDeviceOpenByBDF.argTypes = [ctypes.c_char_p]
    return _valueOrError(libcoreutil.xrtDeviceOpenByBDF(bdf))

def xrtDeviceClose(handle):
    """
    xrtDeviceClose(): Close an device handle opened with xrtDeviceOpen()

    :param handle: XRT device handle
    :return: 0 on success or appropriate error number
    """
    libcoreutil.xrtDeviceClose.restype = ctypes.c_int
    libcoreutil.xrtDeviceClose.argTypes = [xrtDeviceHandle]
    return _valueOrError(libcoreutil.xrtDeviceClose(handle))

def xrtDeviceLoadXclbin(handle, buf):
    """
    xrtDeviceLoadXclbin(): Load an xclbin image

    :param handle: XRT device handle
    :param buf: Pointer to complete axlf in memory image
    :return: 0 on success or appropriate error number
    """
    libcoreutil.xrtDeviceLoadXclbin.restype = ctypes.c_int
    libcoreutil.xrtDeviceLoadXclbin.argTypes = [xrtDeviceHandle, ctypes.c_void_p]
    return _valueOrError(libcoreutil.xrtDeviceLoadXclbin(handle, buf))


def xrtDeviceToXclDevice(handle):
    """
    Convert xrt device handle to xcl device handle for shim level APIs

    :param handle: XRT device handle
    :return: XCL device handle
    """
    libcoreutil.xrtDeviceToXclDevice.restype = ctypes.POINTER(xclDeviceHandle)
    libcoreutil.xrtDeviceToXclDevice.argtypes = [xrtDeviceHandle]
    return _valueOrError(libcoreutil.xrtDeviceToXclDevice(handle))

def xrtPLKernelOpen(handle, xclbinId, name):
    """
    Open a PL kernel and obtain its handle
    :param handle: Device handle
    :param xclbinId: UUID of the xclbin image running on the device
    :param name: Name of PL kernel
    :return: Kernel handle which must be closed with xrtKernelClose()
    """
    libcoreutil.xrtPLKernelOpen.restype = ctypes.POINTER(xrtKernelHandle)
    libcoreutil.xrtPLKernelOpen.argtypes = [xclDeviceHandle, ctypes.c_char_p, ctypes.c_char_p]
    if isinstance(name, str):
        nm = str.encode(name)
    else:
        nm = name
    return _valueOrError(libcoreutil.xrtPLKernelOpen(handle, xclbinId.bytes, nm))


def xrtKernelClose(khandle):
    """
    Close an opened kernel
    :param khandle: Kernel handle obtained in xrtPLKernelOpen()
    :return: 0 or raises an OSError exception
    """
    libcoreutil.xrtKernelClose.restype = ctypes.c_int
    libcoreutil.xrtKernelClose.argtypes = [xrtKernelHandle]
    res = libcoreutil.xrtKernelClose(khandle)
    if (res):
        res = errno.EINVAL
    return _valueOrError(-res);


def xrtKernelGetFunc(*args):
    """
    Obtain the callback function as defined below to start a kernel execution:
        First argument is kernel handle obtained in xrtPLKernelOpen()
        Followed by variable number of kernel arguments
        Returns run handle which must be closed with xrtRunClose()
    After obtain the callback function, it can be called as:
        func(kernel_handle, arg0, arg1, ...)

    :param args: variable number of kernel argument ctypes for this kernel
                 (kernel handle is not included)
    :return: python function that can be called with specific kernel arguments
             to start the kernel specified by the kernel handle
    """
    types = [xrtKernelHandle]
    for argtype in args:
        types.append(argtype)
    proto = ctypes.CFUNCTYPE(xrtKernelRunHandle, *types)
    func = proto(("xrtKernelRun", libcoreutil))
    return func


def xrtRunWait(rhandle):
    """
    Wait for a kernel execution to finish
    :param rhandle: kernel run handle obtained in xrtKernelRun()
    :return: ert_cmd_state code
    """
    libcoreutil.xrtRunWait.restype = ctypes.c_int
    libcoreutil.xrtRunWait.argtypes = [xrtRunHandle]
    return libcoreutil.xrtRunWait(rhandle)


def xrtRunClose(rhandle):
    """
    Close a run handle
    :param rhandle: kernel run handle obtained in xrtKernelRun()
    :return: 0 or throw OSError with error code
    """
    libcoreutil.xrtRunClose.restype = ctypes.c_int
    libcoreutil.xrtRunClose.argtypes = [xrtRunHandle]
    res = libcoreutil.xrtRunClose(rhandle)
    if (res):
        res = errno.EINVAL
    return _valueOrError(-res)


def xclIPName2Index(handle, name):
    """
    Obtain index of a kernel given its name.
    :param handle: Device handle
    :param name: Name of PL kernel
    :return: index of Kernel

    The index is used in APIs like xclOpenContext(), etc.
    """
    libcore.xclIPName2Index.restype = ctypes.c_int
    libcore.xclIPName2Index.argtypes = [xclDeviceHandle, ctypes.c_char_p]
    return _valueOrError(libcore.xclIPName2Index(handle, name))


def xrtBOAllocUserPtr(handle, userptr, size, flags, grp):
    """
    Allocate a BO using userptr provided by the user.
    :param handle: Device handle
    :param userptr: Pointer to 4K aligned user memory
    :param size: Size of buffer
    :param flags: Specify bank information, etc
    :param grp: Specify memory bank group ID
    :return: xrtBufferHandle on success or NULL
    """
    libcoreutil.xrtBOAllocUserPtr.restype = xrtBufferHandle
    libcoreutil.xrtBOAllocUserPtr.argtypes = [xclDeviceHandle, ctypes.c_void_p, ctypes.c_size_t, xrtBufferFlags, xrtMemoryGroup]
    return libcoreutil.xrtBOAllocUserPtr(handle, userptr, size, flags, grp)


def xrtBOAlloc(handle, size, flags, grp):
    """
    Allocate a BO of requested size with appropriate flags.
    :param handle: Device handle
    :param size: Size of buffer
    :param flags: Specify bank information, etc
    :param grp: Specify memory bank group ID
    :return: xrtBufferHandle on success or NULL
    """
    libcoreutil.xrtBOAlloc.restype = xrtBufferHandle
    libcoreutil.xrtBOAlloc.argtypes = [xclDeviceHandle, ctypes.c_size_t, xrtBufferFlags, xrtMemoryGroup]
    return libcoreutil.xrtBOAlloc(handle, size, flags, grp)


def xrtBOSubAlloc(parent, size, offset):
    """
    Allocate a sub buffer from a parent buffer
    :param parent: Parent buffer handle
    :param size: Size of sub buffer
    :param offset: Offset into parent buffer
    :return: xrtBufferHandle on success or NULL
    """
    libcoreutil.xrtBOSubAlloc.restype = xrtBufferHandle
    libcoreutil.xrtBOSubAlloc.argtypes = [xclDeviceHandle, ctypes.c_size_t, ctypes.c_size_t]
    return libcoreutil.xrtBOSubAlloc(parent, size, offset)


def xrtBOFree(bufhandle):
    """
    Free a previously allocated BO
    :param bufhandle: Buffer handle
    :return: 0 on success, or err code on error
    """
    libcoreutil.xrtBOFree.restype = ctypes.c_int
    libcoreutil.xrtBOFree.argtypes = [xrtBufferHandle]
    return _valueOrError(libcoreutil.xrtBOFree(bufhandle))


def xrtBOSync(bufhandle, direction, size, offset):
    """
    Synchronize buffer contents in requested direction
    :param bufhandle: Buffer handle
    :param direction: To device or from device
    :param size: Size of data to synchronize
    :param offset: Offset within the BO
    :return: 0 on success, or err code on error

    Synchronize the buffer contents between host and device. Depending
    on the memory model this may require DMA to/from device or CPU
    cache flushing/invalidation.
    """
    libcoreutil.xrtBOSync.restype = ctypes.c_int
    libcoreutil.xrtBOSync.argtypes = [xrtBufferHandle, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t]
    return _valueOrError(libcoreutil.xrtBOSync(bufhandle, direction, size, offset))


def xrtBOMap(bufhandle):
    """
    Memory map BO into user's address space
    :param bufhandle: Buffer handle
    :return: Memory mapped buffer, or NULL on error

    Map the contents of the buffer object into host memory
    To unmap the buffer call xclUnmapBO().
    """
    libcoreutil.xrtBOMap.restype = ctypes.c_void_p
    libcoreutil.xrtBOMap.argtypes = [xrtBufferHandle]
    return libcoreutil.xrtBOMap(bufhandle)


def xrtBOWrite(bufhandle, src, size, seek):
    """
    Copy-in user data to host backing storage of BO
    :param bufhandle: Buffer handle
    :param src: Source data pointer
    :param size: Size of data to copy
    :param seek: Offset within the BO
    :return: 0 on success, or err code on error

    Copy host buffer contents to previously allocated device
    memory. ``seek`` specifies how many bytes to skip at the beginning
    of the BO before copying-in ``size`` bytes of host buffer.
    """
    libcoreutil.xrtBOWrite.restype = ctypes.c_int
    libcoreutil.xrtBOWrite.argtypes = [xrtBufferHandle, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    return _valueOrError(libcoreutil.xrtBOWrite(bufhandle, src, size, seek))


def xrtBORead(bufhandle, dst, size, skip):
    """
    Copy-out user data from host backing storage of BO
    :param bufhandle: Buffer handle
    :param src: Destination data pointer
    :param size: Size of data to copy
    :param skip: Offset within the BO
    :return: 0 on success, or err code on error

    Copy contents of previously allocated device memory to host buffer.
    ``skip`` specifies how many bytes to skip from the beginning of the BO
    before copying-out ``size`` bytes of device buffer.
    """
    libcoreutil.xrtBOWrite.restype = ctypes.c_int
    libcoreutil.xrtBOWrite.argtypes = [xrtBufferHandle, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    return _valueOrError(libcoreutil.xrtBORead(bufhandle, dst, size, skip))


def xrtKernelArgGroupId(khandle, argno):
    """
    Acquire bank group id for kernel argument
    :param khandle: Handle to kernel previously opened with xrtKernelOpen
    :param argno: Index of kernel argument
    :Return: Group id or negative error code on error

    A valid group id is a non-negative integer.  The group id is required
    when constructing a buffer object.

    The kernel argument group id is ambigious if kernel has multiple kernel
    with different connectivity for specified argument.  In this case the
    API returns error.
    """
    libcoreutil.xrtKernelArgGroupId.restype = ctypes.c_int
    libcoreutil.xrtKernelArgGroupId.argtypes = [xrtKernelHandle, ctypes.c_int]
    return _valueOrError(libcoreutil.xrtKernelArgGroupId(khandle, argno))
