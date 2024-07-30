/******************************************************************************
* Copyright (C) 2021 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_ipu.c
* @{
*
* This file contains the data structures and routines for low level IO
* operations for ipu backend.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Tejus   05/27/2021 Initial creation.
* </pre>
*
******************************************************************************/
/***************************** Include Files *********************************/
#include <stdint.h>
#include <stdlib.h>

#ifdef __AIEIPU__

#include "com_helper.h"

#endif /* __AIEIPU__ */

#include "xaie_helper.h"
#include "xaie_io.h"
#include "xaie_io_common.h"

#ifdef __AIEIPU__
/****************************** Type Definitions *****************************/
typedef struct {
	u64 BaseAddr;
} XAie_IpuIO;

/************************** Variable Definitions *****************************/
static XAie_IpuIO IpuIO;

#define XAIE_IPU_MASKPOLL_COUNT 100

/************************** Function Definitions *****************************/
/*****************************************************************************/
/**
*
* This is the memory IO function to free the global IO instance
*
* @param	IOInst: IO Instance pointer.
*
* @return	None.
*
* @note		The global IO instance is a singleton and freed when
* the reference count reaches a zero. Internal only.
*
*******************************************************************************/
static AieRC XAie_IpuIO_Finish(void *IOInst)
{
	(void)IOInst;

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This is the memory IO function to initialize the global IO instance
*
* @param	DevInst: Device instance pointer.
*
* @return	XAIE_OK on success. Error code on failure.
*
* @note		Internal only.
*
*******************************************************************************/
static AieRC XAie_IpuIO_Init(XAie_DevInst *DevInst)
{
	XAie_IpuIO *IOInst = &IpuIO;

	IOInst->BaseAddr = DevInst->BaseAddr;
	DevInst->IOInst = IOInst;

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This is the memory IO function to write 32bit data to the specified address.
*
* @param	IOInst: IO instance pointer
* @param	RegOff: Register offset to read from.
* @param	Data: 32-bit data to be written.
*
* @return	None.
*
* @note		Internal only.
*
*******************************************************************************/
static AieRC XAie_IpuIO_Write32(void *IOInst, u64 RegOff, u32 Value)
{
	XAie_IpuIO *IpuIOInst = (XAie_IpuIO *)IOInst;

	reg_write32(IpuIOInst->BaseAddr + RegOff, Value);

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This is the memory IO function to read 32bit data from the specified address.
*
* @param	IOInst: IO instance pointer
* @param	RegOff: Register offset to read from.
* @param	Data: Pointer to store the 32 bit value
*
* @return	XAIE_OK on success.
*
* @note		Internal only.
*
*******************************************************************************/
static AieRC XAie_IpuIO_Read32(void *IOInst, u64 RegOff, u32 *Data)
{
	XAie_IpuIO *IpuIOInst = (XAie_IpuIO *)IOInst;

	*Data = reg_read32(IpuIOInst->BaseAddr + RegOff);

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This is the memory IO function to write masked 32bit data to the specified
* address.
*
* @param	IOInst: IO instance pointer
* @param	RegOff: Register offset to read from.
* @param	Mask: Mask to be applied to Data.
* @param	Value: 32-bit data to be written.
*
* @return	None.
*
* @note		Internal only.
*
*******************************************************************************/
static AieRC XAie_IpuIO_MaskWrite32(void *IOInst, u64 RegOff, u32 Mask,
		u32 Value)
{
	XAie_IpuIO *IpuIOInst = (XAie_IpuIO *)IOInst;

	reg_maskwrite32(IpuIOInst->BaseAddr + RegOff, Mask, Value);

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This is the memory IO function to mask poll an address for a value.
*
* @param	IOInst: IO instance pointer
* @param	RegOff: Register offset to read from.
* @param	Mask: Mask to be applied to Data.
* @param	Value: 32-bit value to poll for
* @param	TimeOutUs: Timeout in micro seconds.
*
* @return	XAIE_OK or XAIE_ERR.
*
* @note		Internal only.
*
*******************************************************************************/
static AieRC XAie_IpuIO_MaskPoll(void *IOInst, u64 RegOff, u32 Mask,
		u32 Value, u32 TimeOutUs)
{
	AieRC Ret = XAIE_ERR;
	u32 Count, RegVal;
	(void)TimeOutUs;

	/* current rtos implementation doesn't support sleep */
	Count = XAIE_IPU_MASKPOLL_COUNT;

	while (Count > 0U) {
		XAie_IpuIO_Read32(IOInst, RegOff, &RegVal);
		if((RegVal & Mask) == Value) {
			return XAIE_OK;
		}
		com_usleep(0U);
		Count--;
	}

	/* Check for the break from timed-out loop */
	XAie_IpuIO_Read32(IOInst, RegOff, &RegVal);
	if((RegVal & Mask) == Value) {
		Ret = XAIE_OK;
	}

	return Ret;
}

/*****************************************************************************/
/**
*
* This is the memory IO function to write a block of data to aie.
*
* @param	IOInst: IO instance pointer
* @param	RegOff: Register offset to read from.
* @param	Data: Pointer to the data buffer.
* @param	Size: Number of 32-bit words.
*
* @return	None.
*
* @note		Internal only.
*
*******************************************************************************/
static AieRC XAie_IpuIO_BlockWrite32(void *IOInst, u64 RegOff, const u32 *Data,
		u32 Size)
{
	for(u32 i = 0U; i < Size; i++) {
		XAie_IpuIO_Write32(IOInst, RegOff + i * 4U, *Data);
		Data++;
	}

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This is the memory IO function to initialize a chunk of aie address space with
* a specified value.
*
* @param	IOInst: IO instance pointer
* @param	RegOff: Register offset to read from.
* @param	Data: Data to initialize a chunk of aie address space..
* @param	Size: Number of 32-bit words.
*
* @return	None.
*
* @note		Internal only.
*
*******************************************************************************/
static AieRC XAie_IpuIO_BlockSet32(void *IOInst, u64 RegOff, u32 Data,
		u32 Size)
{
	for(u32 i = 0U; i < Size; i++)
		XAie_IpuIO_Write32(IOInst, RegOff+ i * 4U, Data);

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This is the function to run backend operations
*
* @param	IOInst: IO instance pointer
* @param	DevInst: AI engine partition device instance
* @param	Op: Backend operation code
* @param	Arg: Backend operation argument
*
* @return	XAIE_OK for success and error code for failure.
*
* @note		Internal only.
*
*******************************************************************************/
static AieRC XAie_IpuIO_RunOp(void *IOInst, XAie_DevInst *DevInst,
		XAie_BackendOpCode Op, void *Arg)
{
	(void)DevInst;
	switch(Op) {
		case XAIE_BACKEND_OP_CONFIG_SHIMDMABD:
		{
			XAie_ShimDmaBdArgs *BdArgs =
				(XAie_ShimDmaBdArgs *)Arg;

			XAie_IpuIO_BlockWrite32(IOInst, BdArgs->Addr,
				BdArgs->BdWords, BdArgs->NumBdWords);
			break;
		}
		default:
			XAIE_ERROR("IPU backend doesn't support operation %d\n",
					Op);
			return XAIE_FEATURE_NOT_SUPPORTED;
	}

	return XAIE_OK;
}

#else

static AieRC XAie_IpuIO_Finish(void *IOInst)
{
	/* no-op */
	(void)IOInst;
	return XAIE_OK;
}

static AieRC XAie_IpuIO_Init(XAie_DevInst *DevInst)
{
	/* no-op */
	(void)DevInst;
	XAIE_ERROR("Driver is not compiled with baremetal "
			"backend (__AIEBAREMETAL__)\n");
	return XAIE_INVALID_BACKEND;
}

static AieRC XAie_IpuIO_Write32(void *IOInst, u64 RegOff, u32 Value)
{
	/* no-op */
	(void)IOInst;
	(void)RegOff;
	(void)Value;

	return XAIE_ERR;
}

static AieRC XAie_IpuIO_Read32(void *IOInst, u64 RegOff, u32 *Data)
{
	/* no-op */
	(void)IOInst;
	(void)RegOff;
	(void)Data;
	return XAIE_ERR;
}

static AieRC XAie_IpuIO_MaskWrite32(void *IOInst, u64 RegOff, u32 Mask,
		u32 Value)
{
	/* no-op */
	(void)IOInst;
	(void)RegOff;
	(void)Mask;
	(void)Value;

	return XAIE_ERR;
}

static AieRC XAie_IpuIO_MaskPoll(void *IOInst, u64 RegOff, u32 Mask,
		u32 Value, u32 TimeOutUs)
{
	/* no-op */
	(void)IOInst;
	(void)RegOff;
	(void)Mask;
	(void)Value;
	(void)TimeOutUs;
	return XAIE_ERR;
}

static AieRC XAie_IpuIO_BlockWrite32(void *IOInst, u64 RegOff, const u32 *Data,
		u32 Size)
{
	/* no-op */
	(void)IOInst;
	(void)RegOff;
	(void)Data;
	(void)Size;

	return XAIE_ERR;
}

static AieRC XAie_IpuIO_BlockSet32(void *IOInst, u64 RegOff, u32 Data,
		u32 Size)
{
	/* no-op */
	(void)IOInst;
	(void)RegOff;
	(void)Data;
	(void)Size;

	return XAIE_ERR;
}

static AieRC XAie_IpuIO_RunOp(void *IOInst, XAie_DevInst *DevInst,
		XAie_BackendOpCode Op, void *Arg)
{
	(void)IOInst;
	(void)DevInst;
	(void)Op;
	(void)Arg;
	return XAIE_FEATURE_NOT_SUPPORTED;
}

#endif /* __AIEIPU__ */

static XAie_MemInst* XAie_IpuMemAllocate(XAie_DevInst *DevInst, u64 Size,
		XAie_MemCacheProp Cache)
{
	(void)DevInst;
	(void)Size;
	(void)Cache;
	return NULL;
}

static AieRC XAie_IpuMemFree(XAie_MemInst *MemInst)
{
	(void)MemInst;
	return XAIE_ERR;
}

static AieRC XAie_IpuMemSyncForCPU(XAie_MemInst *MemInst)
{
	(void)MemInst;
	return XAIE_ERR;
}

static AieRC XAie_IpuMemSyncForDev(XAie_MemInst *MemInst)
{
	(void)MemInst;
	return XAIE_ERR;
}

static AieRC XAie_IpuMemAttach(XAie_MemInst *MemInst, u64 MemHandle)
{
	(void)MemInst;
	(void)MemHandle;
	return XAIE_ERR;
}

static AieRC XAie_IpuMemDetach(XAie_MemInst *MemInst)
{
	(void)MemInst;
	return XAIE_ERR;
}

static AieRC XAie_IpuIO_CmdWrite(void *IOInst, u8 Col, u8 Row, u8 Command,
		u32 CmdWd0, u32 CmdWd1, const char *CmdStr)
{
	/* no-op */
	(void)IOInst;
	(void)Col;
	(void)Row;
	(void)Command;
	(void)CmdWd0;
	(void)CmdWd1;
	(void)CmdStr;

	return XAIE_ERR;
}

const XAie_Backend IpuBackend =
{
	.Type = XAIE_IO_BACKEND_BAREMETAL,
	.Ops.Init = XAie_IpuIO_Init,
	.Ops.Finish = XAie_IpuIO_Finish,
	.Ops.Write32 = XAie_IpuIO_Write32,
	.Ops.Read32 = XAie_IpuIO_Read32,
	.Ops.MaskWrite32 = XAie_IpuIO_MaskWrite32,
	.Ops.MaskPoll = XAie_IpuIO_MaskPoll,
	.Ops.BlockWrite32 = XAie_IpuIO_BlockWrite32,
	.Ops.BlockSet32 = XAie_IpuIO_BlockSet32,
	.Ops.CmdWrite = XAie_IpuIO_CmdWrite,
	.Ops.RunOp = XAie_IpuIO_RunOp,
	.Ops.MemAllocate = XAie_IpuMemAllocate,
	.Ops.MemFree = XAie_IpuMemFree,
	.Ops.MemSyncForCPU = XAie_IpuMemSyncForCPU,
	.Ops.MemSyncForDev = XAie_IpuMemSyncForDev,
	.Ops.MemAttach = XAie_IpuMemAttach,
	.Ops.MemDetach = XAie_IpuMemDetach,
	.Ops.GetTid = XAie_IODummyGetTid,
	.Ops.SubmitTxn = NULL,
};

/** @} */
