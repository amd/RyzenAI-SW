/**
 * Copyright (C) 2016-2017 Xilinx, Inc
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

#ifndef __XCL_MACROS__
#define __XCL_MACROS__

#ifdef __cplusplus
#include <cstdlib>
#include <cstdint>
#else
#include <stdlib.h>
#include <stdint.h>
#endif

#define xclOpen_n 0
#define xclClose_n 1
#define xclGetDeviceInfo_n 2
#define xclLoadBitstream_n 3
#define xclAllocDeviceBuffer_n 4
#define xclFreeDeviceBuffer_n 5
#define xclCopyBufferHost2Device_n 6
#define xclCopyBufferDevice2Host_n 7
#define xclWriteAddrSpaceDeviceRam_n 8
#define xclWriteAddrKernelCtrl_n 9
#define xclReadAddrSpaceDeviceRam_n 10
#define xclReadAddrKernelCtrl_n 11
#define xclUpgradeFirmware_n 12
#define xclBootFPGA_n 13
#define xclPerfMonReadCounters_n 14
#define xclPerfMonGetTraceCount_n 15
#define xclPerfMonReadTrace_n 16
#define xclGetDeviceTimestamp_n 17
#define xclReadBusStatus_n 18
#define xclGetDebugMessages_n 19
#define xclSetEnvironment_n 20
#define xclWriteHostEvent_n 21
#define xclCopyBO_n 22
#define xclCreateQueue_n 23
#define xclWriteQueue_n 24
#define xclReadQueue_n 25
#define xclDestroyQueue_n 26
#define xclImportBO_n 27
#define xclSetupInstance_n 28
#define xclPollCompletion_n 29
#define xclPollQueue_n 30
#define xclSetQueueOpt_n 31

#define xclPerfMonReadCounters_Streaming_n 32
#define xclPerfMonReadTrace_Streaming_n 33

#define xclQdma2HostReadMem_n 34
#define xclQdma2HostWriteMem_n 35
#define xclQdma2HostInterrupt_n 36
#define xclDmaVersion_n 37

#define xclGraphInit_n 38
#define xclGraphRun_n 39
#define xclGraphWait_n 40
#define xclGraphEnd_n 41
#define xclGraphUpdateRTP_n 42
#define xclGraphReadRTP_n 43
#define xclSyncBOAIENB_n 44
#define xclGMIOWait_n 45
#define xclLoadXclbinContent_n 46
#define xclGraphTimedWait_n 47
#define xclGraphTimedEnd_n 48
#define xclGraphResume_n 49

#define xclCopyBOFromFd_n 50
#define xclRegWrite_n 51
#define xclRegRead_n 52
#define swemuDriverVersion_n 53

#endif
