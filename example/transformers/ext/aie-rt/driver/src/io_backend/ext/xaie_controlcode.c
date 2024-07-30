/******************************************************************************
* Copyright (C) 2023 AMD.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie_cdo.c
* @{
*
* This file contains the data structures and routines for low level IO
* operations for controlcode backend.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who        Date     Changes
* ----- ------     -------- -----------------------------------------------------
* 1.0   Sankarji   03/08/2023 Initial creation.
* </pre>
*
******************************************************************************/
/***************************** Include Files *********************************/
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xaie_helper.h"
#include "xaie_io.h"
#include "xaie_io_common.h"
#include "xaie_io_privilege.h"
#include "xaie_npi.h"
#include "isa_stubs.h"

#ifdef __AIECONTROLCODE__

#define TEMP_ASM_FILE1    ".temp_data1.txt"
#define TEMP_ASM_FILE2    ".temp_data2.txt"
#define TEMP_ASM_FILE3    ".temp_data3.txt"

//#define UC_DMA_DATASZ					4
//#define DATA_SECTION_ALIGNMENT          16

/*#define START_JOB_OPSZ					8
#define END_JOB_OPSZ					4
#define EOF_OPSZ						4
#define UC_DMA_WRITE_DES_SYNC_OPSZ		4
#define MASK_WRITE32_OPSZ				16
#define WRITE_32_DATA_OPSZ				12
#define UC_DMA_BD_OPSZ					16
#define UC_DMA_DATASZ					4
#define DATA_SECTION_ALIGNMENT          16
#define HEADER_SIZE						16*/

/************************** Constant Definitions *****************************/

/****************************** Type Definitions *****************************/
typedef struct {
	u64 BaseAddr;
	u64 NpiBaseAddr;
	char ControlCodeFileName[128];
	FILE *ControlCodefp;
	FILE *ControlCodedatafp;
	FILE *ControlCodedata2fp;
	FILE *ControlCodedata3fp;
	u32  UcbdLabelNum;
	u32  UcbdDataNum;
	u32  UcDmaDataNum;
	u32  UcJobNum;
	u32  UcJobSize;
	u32  UcJobTextSize;
	u32 PageSize;
	u8 CombineCommands;
} XAie_ControlCodeIO;

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
static AieRC XAie_ControlCodeIO_Finish(void *IOInst)
{
	if(IOInst) {
		free(IOInst);
	}

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
static AieRC XAie_ControlCodeIO_Init(XAie_DevInst *DevInst)
{
	XAie_ControlCodeIO *IOInst;

	IOInst = (XAie_ControlCodeIO *)malloc(sizeof(*IOInst));
	if(IOInst == NULL) {
		XAIE_ERROR("Memory allocation failed\n");
		return XAIE_ERR;
	}

	IOInst->BaseAddr = DevInst->BaseAddr;
	IOInst->NpiBaseAddr = XAIE_NPI_BASEADDR;
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
* @note		None.
* @note		Internal only.
*
*******************************************************************************/
static AieRC XAie_ControlCodeIO_Write32(void *IOInst, u64 RegOff, u32 Value)
{
	XAie_ControlCodeIO  *ControlCodeInst = (XAie_ControlCodeIO *)IOInst;
	u32 DataAligner = (DATA_SECTION_ALIGNMENT -
		(ControlCodeInst->UcJobTextSize % DATA_SECTION_ALIGNMENT));

	if (ControlCodeInst->ControlCodefp != NULL) {

		if((ControlCodeInst->UcJobSize + ISA_OPSIZE_UC_DMA_WRITE_DES_SYNC +
			UC_DMA_BD_SIZE + UC_DMA_WORD_LEN + DataAligner) >= ControlCodeInst->PageSize) {
				fprintf(ControlCodeInst->ControlCodefp, "END_JOB\n\n");
				fprintf(ControlCodeInst->ControlCodefp, "START_JOB %d\n",
				ControlCodeInst->UcJobNum);
				ControlCodeInst->UcJobTextSize = PAGE_HEADER_SIZE + ISA_OPSIZE_START_JOB + ISA_OPSIZE_END_JOB + ISA_OPSIZE_EOF;
				ControlCodeInst->UcJobSize = ControlCodeInst->UcJobTextSize;
				ControlCodeInst->UcJobNum++;
				ControlCodeInst->CombineCommands = 0;
		}

		if(ControlCodeInst->CombineCommands) {
				fseek(ControlCodeInst->ControlCodedatafp, -3, SEEK_CUR);
				fprintf(ControlCodeInst->ControlCodedatafp, " 1\n");
			}
		else {
			fprintf(ControlCodeInst->ControlCodefp,
					"UC_DMA_WRITE_DES_SYNC\t @UCBD_label_%d\n",
					ControlCodeInst->UcbdLabelNum);
			fprintf(ControlCodeInst->ControlCodedatafp, "UCBD_label_%d:\n",
					ControlCodeInst->UcbdLabelNum);
			ControlCodeInst->CombineCommands = 1;
			ControlCodeInst->UcbdLabelNum++;
			ControlCodeInst->UcJobTextSize += ISA_OPSIZE_UC_DMA_WRITE_DES_SYNC;
			ControlCodeInst->UcJobSize += ISA_OPSIZE_UC_DMA_WRITE_DES_SYNC;
		}
		fprintf(ControlCodeInst->ControlCodedatafp,
				"\t UC_DMA_BD\t 0, 0x%lx, @WRITE_data_%d, 1, 0, 0\n",
				RegOff,  ControlCodeInst->UcbdDataNum);
		ControlCodeInst->UcJobSize += UC_DMA_BD_SIZE;
		fprintf(ControlCodeInst->ControlCodedata2fp, "WRITE_data_%d:\n",
				ControlCodeInst->UcbdDataNum);
		fprintf(ControlCodeInst->ControlCodedata2fp, "\t.long 0x%08x\n", Value);
		ControlCodeInst->UcbdDataNum++;
		ControlCodeInst->UcJobSize += UC_DMA_WORD_LEN;
	}

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
* @note		None.
* @note		Internal only.
*
*******************************************************************************/
static AieRC XAie_ControlCodeIO_Read32(void *IOInst, u64 RegOff, u32 *Data)
{
	/* no-op */
	(void)IOInst;
	(void)RegOff;
	*Data = 0U;
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
* @note		None.
* @note		Internal only.
*
*******************************************************************************/
static AieRC XAie_ControlCodeIO_MaskWrite32(void *IOInst, u64 RegOff, u32 Mask,
		u32 Value)
{
	XAie_ControlCodeIO  *ControlCodeInst = (XAie_ControlCodeIO *)IOInst;
	u32 DataAligner = (DATA_SECTION_ALIGNMENT -
		(ControlCodeInst->UcJobTextSize % DATA_SECTION_ALIGNMENT));


	if (ControlCodeInst->ControlCodefp != NULL) {
		if((ControlCodeInst->UcJobSize + ISA_OPSIZE_MASK_WRITE_32 +
			DataAligner) >= ControlCodeInst->PageSize) {
			fprintf(ControlCodeInst->ControlCodefp, "END_JOB\n\n");
			fprintf(ControlCodeInst->ControlCodefp, "START_JOB %d\n",
					ControlCodeInst->UcJobNum);
			ControlCodeInst->UcJobTextSize = PAGE_HEADER_SIZE + ISA_OPSIZE_START_JOB + ISA_OPSIZE_END_JOB + ISA_OPSIZE_EOF;
			ControlCodeInst->UcJobSize = ControlCodeInst->UcJobTextSize;
			ControlCodeInst->UcJobNum++;
		}

		fprintf(ControlCodeInst->ControlCodefp, "MASK_WRITE_32\t 0x%lx, 0x%x, 0x%x\n",
				RegOff, Mask, Value );
		ControlCodeInst->CombineCommands = 0;
		ControlCodeInst->UcJobSize += ISA_OPSIZE_MASK_WRITE_32;
		ControlCodeInst->UcJobTextSize += ISA_OPSIZE_MASK_WRITE_32;
	}
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
* @note		None.
* @note		Internal only.
*
*******************************************************************************/
static AieRC XAie_ControlCodeIO_MaskPoll(void *IOInst, u64 RegOff, u32 Mask, u32 Value,
		u32 TimeOutUs)
{

	XAie_ControlCodeIO  *ControlCodeInst = (XAie_ControlCodeIO *)IOInst;
	u32 DataAligner = (DATA_SECTION_ALIGNMENT -
		(ControlCodeInst->UcJobTextSize % DATA_SECTION_ALIGNMENT));


	if (ControlCodeInst->ControlCodefp != NULL) {
		if((ControlCodeInst->UcJobSize + ISA_OPSIZE_MASK_POLL_32 +
			DataAligner) >= ControlCodeInst->PageSize) {
			fprintf(ControlCodeInst->ControlCodefp, "END_JOB\n\n");
			fprintf(ControlCodeInst->ControlCodefp, "START_JOB %d\n",
					ControlCodeInst->UcJobNum);
			ControlCodeInst->UcJobTextSize = PAGE_HEADER_SIZE + ISA_OPSIZE_START_JOB + ISA_OPSIZE_END_JOB + ISA_OPSIZE_EOF;
			ControlCodeInst->UcJobSize = ControlCodeInst->UcJobTextSize;
			ControlCodeInst->UcJobNum++;
		}

		fprintf(ControlCodeInst->ControlCodefp, "MASK_POLL_32\t 0x%lx, 0x%x, 0x%x\n",
				RegOff, Mask, Value );
		ControlCodeInst->CombineCommands = 0;
		ControlCodeInst->UcJobSize += ISA_OPSIZE_MASK_POLL_32;
		ControlCodeInst->UcJobTextSize += ISA_OPSIZE_MASK_POLL_32;
	}
	return XAIE_OK;
}

//u32 uc_bd_num = 0;
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
* @note		None.
* @note		Internal only.
*
*******************************************************************************/
static AieRC XAie_ControlCodeIO_BlockWrite32(void *IOInst, u64 RegOff, const u32 *Data,
		u32 Size)
{
	u32 CompletedSize = 0;
	u32 IterationSize;
	u64 AdjustedOff = 0;

	XAie_ControlCodeIO  *ControlCodeInst = (XAie_ControlCodeIO *)IOInst;
	u32 DataAligner = (DATA_SECTION_ALIGNMENT -
		(ControlCodeInst->UcJobTextSize % DATA_SECTION_ALIGNMENT));



	CompletedSize = 0;
	while (Size > CompletedSize) {
		if (ControlCodeInst->ControlCodefp != NULL) {

			if((ControlCodeInst->UcJobSize + ISA_OPSIZE_UC_DMA_WRITE_DES_SYNC +
				UC_DMA_BD_SIZE + UC_DMA_WORD_LEN + DataAligner) >= ControlCodeInst->PageSize) {
				fprintf(ControlCodeInst->ControlCodefp, "END_JOB\n\n");
				fprintf(ControlCodeInst->ControlCodefp, "START_JOB %d\n",
						ControlCodeInst->UcJobNum);
				ControlCodeInst->UcJobTextSize = PAGE_HEADER_SIZE + ISA_OPSIZE_START_JOB + ISA_OPSIZE_END_JOB + ISA_OPSIZE_EOF;
				ControlCodeInst->UcJobSize = ControlCodeInst->UcJobTextSize;
				ControlCodeInst->UcJobNum++;
				ControlCodeInst->CombineCommands = 0;
			}

			if(ControlCodeInst->CombineCommands) {
				fseek(ControlCodeInst->ControlCodedatafp, -3, SEEK_CUR);
				fprintf(ControlCodeInst->ControlCodedatafp, " 1\n");
			}
			else {
				fprintf(ControlCodeInst->ControlCodefp,
						"UC_DMA_WRITE_DES_SYNC\t @UCBD_label_%d\n",
						ControlCodeInst->UcbdLabelNum);
				fprintf(ControlCodeInst->ControlCodedatafp, "UCBD_label_%d:\n",
						ControlCodeInst->UcbdLabelNum);
				ControlCodeInst->CombineCommands = 1;
				ControlCodeInst->UcbdLabelNum++;
				ControlCodeInst->UcJobSize += ISA_OPSIZE_UC_DMA_WRITE_DES_SYNC;
				ControlCodeInst->UcJobTextSize += ISA_OPSIZE_UC_DMA_WRITE_DES_SYNC;
			}

			DataAligner = (DATA_SECTION_ALIGNMENT -
				(ControlCodeInst->UcJobTextSize % DATA_SECTION_ALIGNMENT));
			fprintf(ControlCodeInst->ControlCodedata3fp, "DMAWRITE_data_%d:\n",
					ControlCodeInst->UcDmaDataNum);
			ControlCodeInst->UcJobSize += UC_DMA_BD_SIZE;
			for(IterationSize = 0; (IterationSize + CompletedSize) < Size && (ControlCodeInst->UcJobSize + UC_DMA_WORD_LEN + DataAligner) <= ControlCodeInst->PageSize; IterationSize++) {
				fprintf(ControlCodeInst->ControlCodedata3fp, "\t.long 0x%08x\n", *(Data+IterationSize));
				ControlCodeInst->UcJobSize += UC_DMA_WORD_LEN;
			}

			fprintf(ControlCodeInst->ControlCodedatafp,
					"\t UC_DMA_BD\t 0, 0x%lx, @DMAWRITE_data_%d, 0x%x, 0, 0\n",
					RegOff,  ControlCodeInst->UcDmaDataNum, IterationSize);
			AdjustedOff += (IterationSize * UC_DMA_WORD_LEN);
			CompletedSize += IterationSize;
			ControlCodeInst->UcDmaDataNum++;
		}
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
* @note		None.
* @note		Internal only.
*
*******************************************************************************/
static AieRC XAie_ControlCodeIO_BlockSet32(void *IOInst, u64 RegOff, u32 Data, u32 Size)
{
	u32 CompletedSize = 0;
	u32 IterationSize;
	u64 AdjustedOff = 0;

	XAie_ControlCodeIO  *ControlCodeInst = (XAie_ControlCodeIO *)IOInst;
	u32 DataAligner = (DATA_SECTION_ALIGNMENT -
		(ControlCodeInst->UcJobTextSize % DATA_SECTION_ALIGNMENT));

	CompletedSize = 0;
	while (Size > CompletedSize) {
		if (ControlCodeInst->ControlCodefp != NULL) {

			if((ControlCodeInst->UcJobSize + ISA_OPSIZE_UC_DMA_WRITE_DES_SYNC +
				UC_DMA_BD_SIZE + UC_DMA_WORD_LEN + DataAligner) >= ControlCodeInst->PageSize) {
				fprintf(ControlCodeInst->ControlCodefp, "END_JOB\n\n");
				fprintf(ControlCodeInst->ControlCodefp, "START_JOB %d\n",
						ControlCodeInst->UcJobNum);
				ControlCodeInst->UcJobTextSize = PAGE_HEADER_SIZE + ISA_OPSIZE_START_JOB + ISA_OPSIZE_END_JOB + ISA_OPSIZE_EOF;
				ControlCodeInst->UcJobSize = ControlCodeInst->UcJobTextSize;
				ControlCodeInst->UcJobNum++;
				ControlCodeInst->CombineCommands = 0;
			}

			if(ControlCodeInst->CombineCommands) {
				fseek(ControlCodeInst->ControlCodedatafp, -3, SEEK_CUR);
				fprintf(ControlCodeInst->ControlCodedatafp, " 1\n");
			}
			else {
				fprintf(ControlCodeInst->ControlCodefp,
						"UC_DMA_WRITE_DES_SYNC\t @UCBD_label_%d\n",
						ControlCodeInst->UcbdLabelNum);
				fprintf(ControlCodeInst->ControlCodedatafp, "UCBD_label_%d:\n",
						ControlCodeInst->UcbdLabelNum);
				ControlCodeInst->CombineCommands = 1;
				ControlCodeInst->UcbdLabelNum++;
				ControlCodeInst->UcJobSize += ISA_OPSIZE_UC_DMA_WRITE_DES_SYNC;
				ControlCodeInst->UcJobTextSize += ISA_OPSIZE_UC_DMA_WRITE_DES_SYNC;
			}

			DataAligner = (DATA_SECTION_ALIGNMENT -
				(ControlCodeInst->UcJobTextSize % DATA_SECTION_ALIGNMENT));
			fprintf(ControlCodeInst->ControlCodedata3fp, "DMAWRITE_data_%d:\n",
					ControlCodeInst->UcDmaDataNum);
			ControlCodeInst->UcJobSize += UC_DMA_BD_SIZE;
			for(IterationSize = 0; (IterationSize + CompletedSize) < Size &&
				(ControlCodeInst->UcJobSize + UC_DMA_WORD_LEN + DataAligner) <= ControlCodeInst->PageSize; IterationSize++) {
				fprintf(ControlCodeInst->ControlCodedata3fp, "\t.long 0x%08x\n", Data);
				ControlCodeInst->UcJobSize += UC_DMA_WORD_LEN;
			}

			fprintf(ControlCodeInst->ControlCodedatafp,
					"\t UC_DMA_BD\t 0, 0x%lx, @DMAWRITE_data_%d, %d, 0, 0\n\n",
					( RegOff + AdjustedOff),
					ControlCodeInst->UcDmaDataNum, IterationSize);
			AdjustedOff += (IterationSize * UC_DMA_WORD_LEN);
			CompletedSize += IterationSize;
			ControlCodeInst->UcDmaDataNum++;
		}
	}

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This is the function to write 32 bit value to NPI register address.
*
* @param	IOInst: IO instance pointer
* @param	RegOff: NPI register offset
* @param	RegVal: Value to write to register
*
* @return	None.
*
* @note		None.
* @note		Internal only.
*
*******************************************************************************/
static void _XAie_ControlCodeIO_NpiWrite32(void *IOInst, u32 RegOff, u32 RegVal)
{
	(void)IOInst;
	(void)RegOff;
	(void)RegVal;

	return;
}

/*****************************************************************************/
/**
*
* This is the memory IO function to mask poll a NPI address for a value.
*
* @param	IOInst: IO instance pointer
* @param	RegOff: Register offset to read from.
* @param	Mask: Mask to be applied to Data.
* @param	Value: 32-bit value to poll for
* @param	TimeOutUs: Timeout in micro seconds.
*
* @return	XAIE_OK or XAIE_ERR.
*
* @note		None.
* @note		Internal only.
*
*******************************************************************************/
static AieRC _XAie_ControlCodeIO_NpiMaskPoll(void *IOInst, u64 RegOff, u32 Mask,
		u32 Value, u32 TimeOutUs)
{
	(void)IOInst;
	(void)RegOff;
	(void)Mask;
	(void)Value;
	(void)TimeOutUs;

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
static AieRC XAie_ControlCodeIO_RunOp(void *IOInst, XAie_DevInst *DevInst,
		     XAie_BackendOpCode Op, void *Arg)
{
	AieRC RC = XAIE_OK;
	(void)IOInst;

	switch(Op) {
		case XAIE_BACKEND_OP_NPIWR32:
		{
			XAie_BackendNpiWrReq *Req = Arg;

			_XAie_ControlCodeIO_NpiWrite32(IOInst, Req->NpiRegOff,
					Req->Val);
			break;
		}
		case XAIE_BACKEND_OP_NPIMASKPOLL32:
		{
			XAie_BackendNpiMaskPollReq *Req = Arg;

			return _XAie_ControlCodeIO_NpiMaskPoll(IOInst, Req->NpiRegOff,
					Req->Mask, Req->Val, Req->TimeOutUs);
		}
		case XAIE_BACKEND_OP_ASSERT_SHIMRST:
		{
			u8 RstEnable = (u8)((uintptr_t)Arg & 0xFF);

			_XAie_NpiSetShimReset(DevInst, RstEnable);
			break;
		}
		case XAIE_BACKEND_OP_SET_PROTREG:
		{
			RC = _XAie_NpiSetProtectedRegEnable(DevInst, Arg);
			break;
		}
		case XAIE_BACKEND_OP_CONFIG_SHIMDMABD:
		{
			XAie_ShimDmaBdArgs *BdArgs =
				(XAie_ShimDmaBdArgs *)Arg;

			XAie_ControlCodeIO_BlockWrite32(IOInst, BdArgs->Addr,
				BdArgs->BdWords, BdArgs->NumBdWords);
			break;
		}
		case XAIE_BACKEND_OP_REQUEST_TILES:
			return _XAie_PrivilegeRequestTiles(DevInst,
					(XAie_BackendTilesArray *)Arg);
		case XAIE_BACKEND_OP_REQUEST_RESOURCE:
			return _XAie_RequestRscCommon(DevInst, Arg);
		case XAIE_BACKEND_OP_RELEASE_RESOURCE:
			return _XAie_ReleaseRscCommon(Arg);
		case XAIE_BACKEND_OP_FREE_RESOURCE:
			return _XAie_FreeRscCommon(Arg);
		case XAIE_BACKEND_OP_REQUEST_ALLOCATED_RESOURCE:
			return _XAie_RequestAllocatedRscCommon(DevInst, Arg);
		case XAIE_BACKEND_OP_PARTITION_INITIALIZE:
			return _XAie_PrivilegeInitPart(DevInst,
					(XAie_PartInitOpts *)Arg);
		case XAIE_BACKEND_OP_PARTITION_TEARDOWN:
			return _XAie_PrivilegeTeardownPart(DevInst);
		case XAIE_BACKEND_OP_GET_RSC_STAT:
			return _XAie_GetRscStatCommon(DevInst, Arg);
		case XAIE_BACKEND_OP_UPDATE_NPI_ADDR:
		{
			XAie_ControlCodeIO *ControlCodeIOInst = (XAie_ControlCodeIO *)IOInst;
			ControlCodeIOInst->NpiBaseAddr = *((u64 *)Arg);
			break;
		}
		case XAIE_BACKEND_OP_CONFIG_MEM_INTRLVNG:
			return _XAie_PrivilegeConfigMemInterleavingLoc(DevInst,
					(XAie_BackendTilesEnableArray *)Arg);
		default:
			XAIE_ERROR("CDO backend doesn't support operation"
					" %u.\n", Op);
			RC = XAIE_FEATURE_NOT_SUPPORTED;
			break;
	}

	return RC;
}

/*****************************************************************************/
/**
*
* This is the function used to start capturing the control code
* This is usefull only in control code backend. In other backend calling this
* makes no sense.
*
*  @return    1 on failure.
*
*
******************************************************************************/
AieRC XAie_OpenControlCodeFile(XAie_DevInst *DevInst, const char *FileName, u32 JobSize) {


	XAie_ControlCodeIO  *ControlCodeInst = (XAie_ControlCodeIO *)DevInst->IOInst;
	ControlCodeInst->UcbdLabelNum = 0;
	ControlCodeInst->UcbdDataNum = 0;
	ControlCodeInst->UcDmaDataNum = 0;
	ControlCodeInst->UcJobNum = 0;
	ControlCodeInst->UcJobSize = 0;
	ControlCodeInst->UcJobTextSize = 0;
	strcpy(ControlCodeInst->ControlCodeFileName, FileName);
	ControlCodeInst->ControlCodefp      = fopen(FileName, "w");
	ControlCodeInst->ControlCodedatafp  = fopen(TEMP_ASM_FILE1, "w");
	ControlCodeInst->ControlCodedata2fp = fopen(TEMP_ASM_FILE2, "w");
	ControlCodeInst->ControlCodedata3fp = fopen(TEMP_ASM_FILE3, "w");

	//ControlCodeInst->PageSize = JobSize;
	ControlCodeInst->PageSize = 8192;
	ControlCodeInst->CombineCommands = 0;

    if (ControlCodeInst->ControlCodefp == NULL ||
		ControlCodeInst->ControlCodedatafp == NULL ||
		ControlCodeInst->ControlCodedata2fp == NULL ||
		ControlCodeInst->ControlCodedata3fp == NULL) {

		if(ControlCodeInst->ControlCodefp) {
			fclose(ControlCodeInst->ControlCodefp);
		}
		if (ControlCodeInst->ControlCodedatafp) {
			fclose(ControlCodeInst->ControlCodedatafp);
		}
		if (ControlCodeInst->ControlCodedata2fp) {
			fclose(ControlCodeInst->ControlCodedata2fp);
		}
		if (ControlCodeInst->ControlCodedata3fp) {
			fclose(ControlCodeInst->ControlCodedata3fp);
		}
	    //printf("File could not be opened, fopen Error: %s\n", strerror(errno));
        return XAIE_ERR;
    }
    printf("Generating: %s\n", FileName);
	fprintf(ControlCodeInst->ControlCodefp, ";\n");
	fprintf(ControlCodeInst->ControlCodefp, ";text\n");
	fprintf(ControlCodeInst->ControlCodefp, ";\n");
	fprintf(ControlCodeInst->ControlCodefp, "START_JOB %d\n",
			ControlCodeInst->UcJobNum);
	ControlCodeInst->UcJobTextSize += PAGE_HEADER_SIZE + ISA_OPSIZE_START_JOB + ISA_OPSIZE_END_JOB + ISA_OPSIZE_EOF;
	ControlCodeInst->UcJobSize += PAGE_HEADER_SIZE + ISA_OPSIZE_START_JOB + ISA_OPSIZE_END_JOB + ISA_OPSIZE_EOF;
	ControlCodeInst->UcJobNum++;

	fprintf(ControlCodeInst->ControlCodedatafp, ";\n");
	fprintf(ControlCodeInst->ControlCodedatafp, ";data\n");
	fprintf(ControlCodeInst->ControlCodedatafp, ";\n");
	fprintf(ControlCodeInst->ControlCodedatafp, ".align    16\n");
	fprintf(ControlCodeInst->ControlCodedata2fp, ".align    4\n");
	return XAIE_OK;
}

/*****************************************************************************/
/**
* This function ends the current job and starts the new job
* @param DevInst AI engine device instance pointer
*
* @return  0 on success.
*
******************************************************************************/
AieRC XAie_StartNextJob(XAie_DevInst *DevInst)
{
	XAie_ControlCodeIO  *ControlCodeInst = (XAie_ControlCodeIO *)DevInst->IOInst;

	if (ControlCodeInst->ControlCodefp != NULL) {

		fprintf(ControlCodeInst->ControlCodefp, "END_JOB\n\n");
		fprintf(ControlCodeInst->ControlCodefp, "START_JOB %d\n",
				ControlCodeInst->UcJobNum);
				ControlCodeInst->UcJobTextSize = PAGE_HEADER_SIZE + ISA_OPSIZE_START_JOB + ISA_OPSIZE_END_JOB + ISA_OPSIZE_EOF;
				ControlCodeInst->UcJobSize = ControlCodeInst->UcJobTextSize;
				ControlCodeInst->UcJobNum++;
				ControlCodeInst->CombineCommands = 0;

				return XAIE_OK;
	}

	return XAIE_ERR;
}

/*****************************************************************************/
/**
* Merges the given text file.
* @param	SrcFile: Source File name.
* @param	DesFile: Destination File name.
*
* @note		Internal API only.
*
******************************************************************************/
static void _XAie_MegreFiles(char *SrcFile, char *DesFile) {
	FILE *SrcFp = NULL;
	FILE *DesFp = NULL;
	char TempBuf;

	SrcFp = fopen(SrcFile, "r+");
	DesFp = fopen(DesFile, "a+");


	if (!SrcFp || !DesFp) {
	printf("File open failed\n");
        return;
    }

	while ((TempBuf = fgetc(SrcFp)) != EOF) {
		fputc(TempBuf, DesFp);
	}

	fclose(SrcFp);
	fclose(DesFp);
	SrcFp = NULL;
	DesFp = NULL;

}

/*****************************************************************************/
/**
* This function used to stop the control code capture.
* This also merges the temp files and updates the control code file.
*
******************************************************************************/
void XAie_CloseControlCodeFile(XAie_DevInst *DevInst) {
	XAie_ControlCodeIO  *ControlCodeInst = (XAie_ControlCodeIO *)DevInst->IOInst;

	fprintf(ControlCodeInst->ControlCodefp, "END_JOB\n\n");
	fprintf(ControlCodeInst->ControlCodefp, "EOF\n\n");

	fclose(ControlCodeInst->ControlCodefp);
	fclose(ControlCodeInst->ControlCodedatafp);
	fclose(ControlCodeInst->ControlCodedata2fp);
	fclose(ControlCodeInst->ControlCodedata3fp);

	_XAie_MegreFiles(TEMP_ASM_FILE1, ControlCodeInst->ControlCodeFileName);
	_XAie_MegreFiles(TEMP_ASM_FILE2, ControlCodeInst->ControlCodeFileName);
	_XAie_MegreFiles(TEMP_ASM_FILE3, ControlCodeInst->ControlCodeFileName);


	remove(TEMP_ASM_FILE1);
	remove(TEMP_ASM_FILE2);
	remove(TEMP_ASM_FILE3);
}

#else

static AieRC XAie_ControlCodeIO_Finish(void *IOInst)
{
	/* no-op */
	(void)IOInst;
	return XAIE_OK;
}

static AieRC XAie_ControlCodeIO_Init(XAie_DevInst *DevInst)
{
	/* no-op */
	(void)DevInst;
	XAIE_ERROR("Driver is not compiled with cdo generation "
			"backend (__AIECONTROLCODE__)\n");
	return XAIE_INVALID_BACKEND;
}

static AieRC XAie_ControlCodeIO_Write32(void *IOInst, u64 RegOff, u32 Value)
{
	/* no-op */
	(void)IOInst;
	(void)RegOff;
	(void)Value;

	return XAIE_ERR;
}

static AieRC XAie_ControlCodeIO_Read32(void *IOInst, u64 RegOff, u32 *Data)
{
	/* no-op */
	(void)IOInst;
	(void)RegOff;
	(void)Data;
	return 0;
}

static AieRC XAie_ControlCodeIO_MaskWrite32(void *IOInst, u64 RegOff, u32 Mask,
		u32 Value)
{
	/* no-op */
	(void)IOInst;
	(void)RegOff;
	(void)Mask;
	(void)Value;

	return XAIE_ERR;
}

static AieRC XAie_ControlCodeIO_MaskPoll(void *IOInst, u64 RegOff, u32 Mask, u32 Value,
		u32 TimeOutUs)
{
	/* no-op */
	(void)IOInst;
	(void)RegOff;
	(void)Mask;
	(void)Value;
	(void)TimeOutUs;

	return XAIE_ERR;
}

static AieRC XAie_ControlCodeIO_BlockWrite32(void *IOInst, u64 RegOff, const u32 *Data,
		u32 Size)
{
	/* no-op */
	(void)IOInst;
	(void)RegOff;
	(void)Data;
	(void)Size;

	return XAIE_ERR;
}

static AieRC XAie_ControlCodeIO_BlockSet32(void *IOInst, u64 RegOff, u32 Data, u32 Size)
{
	/* no-op */
	(void)IOInst;
	(void)RegOff;
	(void)Data;
	(void)Size;

	return XAIE_ERR;
}

static AieRC XAie_ControlCodeIO_RunOp(void *IOInst, XAie_DevInst *DevInst,
		     XAie_BackendOpCode Op, void *Arg)
{
	(void)IOInst;
	(void)DevInst;
	(void)Op;
	(void)Arg;
	return XAIE_FEATURE_NOT_SUPPORTED;
}

#endif /* __AIECONTROLCODE__ */

static AieRC XAie_ControlCodeIO_CmdWrite(void *IOInst, u8 Col, u8 Row, u8 Command,
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

	return XAIE_OK;
}

static XAie_MemInst* XAie_ControlCodeMemAllocate(XAie_DevInst *DevInst, u64 Size,
		XAie_MemCacheProp Cache)
{
	(void)DevInst;
	(void)Size;
	(void)Cache;
	return NULL;
}

static AieRC XAie_ControlCodeMemFree(XAie_MemInst *MemInst)
{
	(void)MemInst;
	return XAIE_ERR;
}

static AieRC XAie_ControlCodeMemSyncForCPU(XAie_MemInst *MemInst)
{
	(void)MemInst;
	return XAIE_ERR;
}

static AieRC XAie_ControlCodeMemSyncForDev(XAie_MemInst *MemInst)
{
	(void)MemInst;
	return XAIE_ERR;
}

static AieRC XAie_ControlCodeMemAttach(XAie_MemInst *MemInst, u64 MemHandle)
{
	(void)MemInst;
	(void)MemHandle;
	return XAIE_ERR;
}

static AieRC XAie_ControlCodeMemDetach(XAie_MemInst *MemInst)
{
	(void)MemInst;
	return XAIE_ERR;
}

const XAie_Backend ControlCodeBackend =
{
	.Type = XAIE_IO_BACKEND_CONTROLCODE,
	.Ops.Init = XAie_ControlCodeIO_Init,
	.Ops.Finish = XAie_ControlCodeIO_Finish,
	.Ops.Write32 = XAie_ControlCodeIO_Write32,
	.Ops.Read32 = XAie_ControlCodeIO_Read32,
	.Ops.MaskWrite32 = XAie_ControlCodeIO_MaskWrite32,
	.Ops.MaskPoll = XAie_ControlCodeIO_MaskPoll,
	.Ops.BlockWrite32 = XAie_ControlCodeIO_BlockWrite32,
	.Ops.BlockSet32 = XAie_ControlCodeIO_BlockSet32,
	.Ops.CmdWrite = XAie_ControlCodeIO_CmdWrite,
	.Ops.RunOp = XAie_ControlCodeIO_RunOp,
	.Ops.MemAllocate = XAie_ControlCodeMemAllocate,
	.Ops.MemFree = XAie_ControlCodeMemFree,
	.Ops.MemSyncForCPU = XAie_ControlCodeMemSyncForCPU,
	.Ops.MemSyncForDev = XAie_ControlCodeMemSyncForDev,
	.Ops.MemAttach = XAie_ControlCodeMemAttach,
	.Ops.MemDetach = XAie_ControlCodeMemDetach,
	.Ops.GetTid = XAie_IODummyGetTid,
	.Ops.SubmitTxn = NULL,
};

/** @} */
