/******************************************************************************
* Copyright (C) 2022 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

/*****************************************************************************/
/**
* @file xaie_error_interrupt_test.c
* @{
*
* This file contains the baremetal test application of error interrupt for AIE.
*
* This application mocks all possible fatal errors in all AIE modules, and
* implements a sample routine to backtrack those.
*
* Note: The ai-engine-driver must be compiled with lite mode enabled. This could
*	be done by setting CFLAGS in driver makefile as follows,
*	EXTRA_CFLAGS=-DXAIE_FEATURE_LITE
*	XAIE_DEV_SINGLE_GEN=XAIE_DEV_GEN_AIE
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date        Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Nishad  01/28/2022  Initial creation
* </pre>
*
******************************************************************************/

/***************************** Include Files *********************************/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <xaiengine.h>
#include <xil_exception.h>
#include <xil_printf.h>
#include <xscugic.h>
#include <xstatus.h>

#include "platform.h"

/************************** Constant Definitions *****************************/
#define HW_GEN				XAIE_DEV_GEN_AIE
#define XAIE_NUM_ROWS			9
#define XAIE_NUM_COLS			50

#define XAIE_BASE_ADDR			0x20000000000
#define XAIE_COL_SHIFT			23
#define XAIE_ROW_SHIFT			18
#define XAIE_SHIM_ROW			0
#define XAIE_MEM_TILE_ROW_START		0
#define XAIE_MEM_TILE_NUM_ROWS		0
#define XAIE_AIE_TILE_ROW_START		1
#define XAIE_AIE_TILE_NUM_ROWS		8

#define INTC_DEVICE_ID			XPAR_SCUGIC_0_DEVICE_ID
#define AIE_IRQ_VECT_ID_0		180U
#define AIE_IRQ_VECT_ID_1		181U
#define AIE_IRQ_VECT_ID_2		182U

#define MOD_ID_TO_STR(Id)		AIEModule[(Id)]

#define XAie_Print(Format, Args...)					\
	do {								\
		printf("%s: %s():%d: "Format, "[INFO]", __func__,	\
				__LINE__,##Args);			\
	} while(0)

const char *AIEModule[] = {
	"memory",
	"core",
	"pl",
};

const XAie_Events NoCTileErrors[] = {
	XAIE_EVENT_AXI_MM_SLAVE_TILE_ERROR_PL,
	XAIE_EVENT_CONTROL_PKT_ERROR_PL,
	XAIE_EVENT_AXI_MM_DECODE_NSU_ERROR_PL,
	XAIE_EVENT_AXI_MM_SLAVE_NSU_ERROR_PL,
	XAIE_EVENT_AXI_MM_UNSUPPORTED_TRAFFIC_PL,
	XAIE_EVENT_AXI_MM_UNSECURE_ACCESS_IN_SECURE_MODE_PL,
	XAIE_EVENT_AXI_MM_BYTE_STROBE_ERROR_PL,
	XAIE_EVENT_DMA_S2MM_0_ERROR_PL,
	XAIE_EVENT_DMA_S2MM_1_ERROR_PL,
	XAIE_EVENT_DMA_MM2S_0_ERROR_PL,
	XAIE_EVENT_DMA_MM2S_1_ERROR_PL,
};

const XAie_Events PLTileErrors[] = {
	XAIE_EVENT_AXI_MM_SLAVE_TILE_ERROR_PL,
	XAIE_EVENT_CONTROL_PKT_ERROR_PL,
};

const XAie_Events CoreModErrors[] = {
	XAIE_EVENT_TLAST_IN_WSS_WORDS_0_2_CORE,
	XAIE_EVENT_PM_REG_ACCESS_FAILURE_CORE,
	XAIE_EVENT_STREAM_PKT_PARITY_ERROR_CORE,
	XAIE_EVENT_CONTROL_PKT_ERROR_CORE,
	XAIE_EVENT_AXI_MM_SLAVE_ERROR_CORE,
	XAIE_EVENT_INSTR_DECOMPRSN_ERROR_CORE,
	XAIE_EVENT_DM_ADDRESS_OUT_OF_RANGE_CORE,
	XAIE_EVENT_PM_ECC_ERROR_SCRUB_2BIT_CORE,
	XAIE_EVENT_PM_ECC_ERROR_2BIT_CORE,
	XAIE_EVENT_PM_ADDRESS_OUT_OF_RANGE_CORE,
	XAIE_EVENT_DM_ACCESS_TO_UNAVAILABLE_CORE,
	XAIE_EVENT_LOCK_ACCESS_TO_UNAVAILABLE_CORE,
};

const XAie_Events MemModErrors[] = {
	XAIE_EVENT_DM_ECC_ERROR_SCRUB_2BIT_MEM,
	XAIE_EVENT_DM_ECC_ERROR_2BIT_MEM,
	XAIE_EVENT_DM_PARITY_ERROR_BANK_2_MEM,
	XAIE_EVENT_DM_PARITY_ERROR_BANK_3_MEM,
	XAIE_EVENT_DM_PARITY_ERROR_BANK_4_MEM,
	XAIE_EVENT_DM_PARITY_ERROR_BANK_5_MEM,
	XAIE_EVENT_DM_PARITY_ERROR_BANK_6_MEM,
	XAIE_EVENT_DM_PARITY_ERROR_BANK_7_MEM,
	XAIE_EVENT_DMA_S2MM_0_ERROR_MEM,
	XAIE_EVENT_DMA_S2MM_1_ERROR_MEM,
	XAIE_EVENT_DMA_MM2S_0_ERROR_MEM,
	XAIE_EVENT_DMA_MM2S_1_ERROR_MEM,
};

XScuGic xInterruptController;
XAie_ErrorPayload *Buffer;
u32 ErrorsMocked;
u32 ErrorsBacktracked;

/* Buffer size to backtrack 10 errors in one shot */
ssize_t Size = 10 * sizeof(XAie_ErrorPayload);

typedef struct {
	XAie_DevInst *DevInst;
	u8 ProcIrqId;
	u8 IrqId;
} XAie_AppPayload;

/************************** Function Definitions *****************************/
/*****************************************************************************/
/**
*
* This API mocks all possible fatal errors in all shim tiles.
*
* @param	DevInst: Pointer to driver instance.
*
* @return	XAIE_OK on success, and error code on failure.
*
* @note		None.
*
******************************************************************************/
AieRC XAie_MockShimErrors(XAie_DevInst *DevInst)
{
	AieRC RC;

	for(u32 Col = 0; Col < XAIE_NUM_COLS; Col++) {
		XAie_LocType Loc = XAie_TileLoc(Col, XAIE_SHIM_ROW);

		/*
		 * AIE shim row follows PL-NOC-NOC-PL pattern. Below calculation
		 * yields true value only for shim NoC tiles.
		 */
		if ((Col % 4) / 2) {
			for (u32 E = 0;
				 E < sizeof(NoCTileErrors) / sizeof(NoCTileErrors[0]);
				 E++) {
				RC = XAie_EventGenerate(DevInst, Loc,
							XAIE_PL_MOD,
							NoCTileErrors[E]);
				if (RC != XAIE_OK)
					return RC;

				ErrorsMocked++;
			}
		} else {
			for (u32 E = 0;
				 E < sizeof(PLTileErrors) / sizeof(PLTileErrors[0]);
				 E++) {
				RC = XAie_EventGenerate(DevInst, Loc,
							XAIE_PL_MOD,
							PLTileErrors[E]);
				if (RC != XAIE_OK)
					return RC;

				ErrorsMocked++;
			}
		}
	}

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This API mocks all possible fatal errors in all memory modules.
*
* @param	DevInst: Pointer to driver instance.
*
* @return	XAIE_OK on success, and error code on failure.
*
* @note		None.
*
******************************************************************************/
AieRC XAie_MockMemErrors(XAie_DevInst *DevInst)
{
	AieRC RC;

	for (u8 Row = XAIE_AIE_TILE_ROW_START;
		Row < (XAIE_AIE_TILE_ROW_START + XAIE_AIE_TILE_NUM_ROWS);
		Row++) {
		for(u32 Col = 0; Col < XAIE_NUM_COLS; Col++) {
			XAie_LocType Loc = XAie_TileLoc(Col, Row);

			for (u32 E = 0;
				 E < sizeof(MemModErrors) / sizeof(MemModErrors[0]);
				 E++) {
				RC = XAie_EventGenerate(DevInst, Loc,
							XAIE_MEM_MOD,
							MemModErrors[E]);
				if (RC != XAIE_OK)
					return RC;

				ErrorsMocked++;
			}
		}
	}

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* This API mocks all possible fatal errors in all core modules.
*
* @param	DevInst: Pointer to driver instance.
*
* @return	XAIE_OK on success, and error code on failure.
*
* @note		None.
*
******************************************************************************/
AieRC XAie_MockCoreErrors(XAie_DevInst *DevInst)
{
	AieRC RC;

	for (u8 Row = XAIE_AIE_TILE_ROW_START;
		Row < (XAIE_AIE_TILE_ROW_START + XAIE_AIE_TILE_NUM_ROWS);
		Row++) {
		for(u32 Col = 0; Col < XAIE_NUM_COLS; Col++) {
			XAie_LocType Loc = XAie_TileLoc(Col, Row);

			for (u32 E = 0;
				 E < sizeof(CoreModErrors) / sizeof(CoreModErrors[0]);
				 E++) {
				RC = XAie_EventGenerate(DevInst, Loc,
							XAIE_CORE_MOD,
							CoreModErrors[E]);
				if (RC != XAIE_OK)
					return RC;

				ErrorsMocked++;
			}
		}
	}

	return XAIE_OK;
}

/*****************************************************************************/
/**
*
* Error interrupt handler that needs to be invoked in ISR.
*
* @param	Data: Void data pointer. Driver instance is passed to backtrack
*		      source of an error interrupt to a module in AIE.
*
* @return	None.
*
* @note		None.
*
*****************************************************************************/
void XAie_ErrorIsr(void *Data)
{
	AieRC RC;
	XAie_AppPayload *Payload = (XAie_AppPayload *) Data;
	XAie_DevInst *DevInst = Payload->DevInst;
	XAie_Range Cols;
	u8 IrqId = Payload->IrqId;

	XScuGic_Disable(&xInterruptController, IrqId);

	XAie_DisableErrorInterrupts(IrqId);

	XScuGic_Enable(&xInterruptController, IrqId);

	/* Initialize error metadata */
	XAie_ErrorMetadataInit(MData, Buffer, Size);

	XAie_MapIrqIdToCols(IrqId, &Cols);

	XAie_ErrorSetBacktrackRange(&MData, Cols);

	/*
	 * Loop until all errors are successfully backtracked. This could also
	 * be done as part of the bottom halve.
	 */
	do {
		RC = XAie_BacktrackErrorInterrupts(DevInst, &MData);
		for (u32 Count = 0; Count < MData.ErrorCount; Count++) {
			XAie_Print("%d: [IRQ %d] Error event %3d asserted in %s module at tile location (%2d,%2d)\n",
				   Count, IrqId, Buffer[Count].EventId,
				   MOD_ID_TO_STR(Buffer[Count].Module),
				   Buffer[Count].Loc.Col,
				   Buffer[Count].Loc.Row);
		}

		ErrorsBacktracked += MData.ErrorCount;

		if (!MData.IsNextInfoValid)
			continue;

		/*
		 * Insufficient error payload buffer. Consider increasing size
		 * of error payload buffer to reduce backtracking latency.
		 */
		XAie_Print("Backtracking discontinued in %s module at tile location (%2d,%2d)\n",
			   MOD_ID_TO_STR(MData.NextModule), MData.NextTile.Col,
			   MData.NextTile.Row);
	} while (RC != XAIE_OK);
}

/*****************************************************************************/
/**
*
* This API registers AIE error interrupt handler with the platform. Here
* registration is being done for ARM Cortex Core #0.
*
* @param	DevInst: Pointer to driver instance.
*
* @return	0 on success, and -1 on failure.
*
* @note		None.
*
*****************************************************************************/
u32 XAie_IrqInit(XAie_AppPayload *Payload)
{
	int Ret = 0;

	/* The configuration parameters of the interrupt controller */
	XScuGic_Config *IntcConfig;

	Xil_ExceptionDisable();

	/* Initialize the interrupt controller driver */
	IntcConfig = XScuGic_LookupConfig(INTC_DEVICE_ID);
	if (NULL == IntcConfig)
		return -1;

	Ret = XScuGic_CfgInitialize(&xInterruptController, IntcConfig,
				    IntcConfig->CpuBaseAddress);
	if (Ret != XST_SUCCESS)
		return -1;

	/*
	 * Register the interrupt handler to the hardware interrupt handling
	 * logic in the ARM processor.
	 */
	Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_IRQ_INT,
			(Xil_ExceptionHandler) XScuGic_InterruptHandler,
			&xInterruptController);

	Xil_ExceptionEnable();

	/* Connect Interrupt */
	XScuGic_Connect(&xInterruptController, Payload->ProcIrqId,
			(Xil_InterruptHandler) XAie_ErrorIsr, (void *) Payload);

	XScuGic_Enable(&xInterruptController, Payload->ProcIrqId);

	return Ret;
}

/*****************************************************************************/
/**
*
* Entry point for AIE error interrupt test.
*
* @param	None.
*
* @return	0 on success, and -1 on failure.
*
* @note		None.
*
*****************************************************************************/
int main()
{
	AieRC RC;

	init_platform();

	XAie_Print("AIE error interrupt test\n");

	/* Allocate buffer to query the location of an error interrupt in AIE */
	Buffer = (XAie_ErrorPayload *) malloc(Size);
	if (Buffer == NULL) {
		XAie_Print("Memory allocation failed.\n");
		return -1;
	}

	XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR,
			 XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
			 XAIE_NUM_COLS, XAIE_NUM_ROWS, XAIE_SHIM_ROW,
			 XAIE_MEM_TILE_ROW_START, XAIE_MEM_TILE_NUM_ROWS,
			 XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS);

	XAie_InstDeclare(DevInst, &ConfigPtr);

	XAie_AppPayload Payload[] = {
		{
			.DevInst = &DevInst,
			.ProcIrqId = AIE_IRQ_VECT_ID_0,
			.IrqId = 0,
		},
		{
			.DevInst = &DevInst,
			.ProcIrqId = AIE_IRQ_VECT_ID_1,
			.IrqId = 1,
		},
		{
			.DevInst = &DevInst,
			.ProcIrqId = AIE_IRQ_VECT_ID_2,
			.IrqId = 2,
		},
	};

	RC = XAie_SetupPartitionConfig(&DevInst, XAIE_BASE_ADDR, 0, 50);
	if(RC != XAIE_OK) {
		XAie_Print("Failed configure partition.\n");
		return -1;
	}

	RC = XAie_CfgInitialize(&DevInst, &ConfigPtr);
	if(RC != XAIE_OK) {
		XAie_Print("Driver initialization failed.\n");
		return -1;
	}

	XAie_IrqInit(&Payload[0]);
	XAie_IrqInit(&Payload[1]);
	XAie_IrqInit(&Payload[2]);

	RC = XAie_PartitionInitialize(&DevInst, NULL);
	if(RC != XAIE_OK) {
		XAie_Print("Partition initialization failed.\n");
		return -1;
	}

	RC = XAie_ErrorHandlingInit(&DevInst);
	if(RC != XAIE_OK) {
		XAie_Print("Broadcast error network setup failed.\n");
		return -1;
	}

	XAie_MockCoreErrors(&DevInst);
	XAie_MockMemErrors(&DevInst);

	/*
	 * Mock errors more than it can fit in the allocated buffer in a single
	 * backtrack routine.
	 */
	XScuGic_Disable(&xInterruptController, AIE_IRQ_VECT_ID_0);
	XScuGic_Disable(&xInterruptController, AIE_IRQ_VECT_ID_1);
	XScuGic_Disable(&xInterruptController, AIE_IRQ_VECT_ID_2);
	XAie_MockShimErrors(&DevInst);
	XScuGic_Enable(&xInterruptController, AIE_IRQ_VECT_ID_0);
	XScuGic_Enable(&xInterruptController, AIE_IRQ_VECT_ID_1);
	XScuGic_Enable(&xInterruptController, AIE_IRQ_VECT_ID_2);

	do {
		XAie_Print("Total errors mocked:\t%d\n", ErrorsMocked);
		XAie_Print("Total errors backtracked:\t%d\n",
			   ErrorsBacktracked);
		sleep(1);
	} while(ErrorsMocked != ErrorsBacktracked);

	free(Buffer);

	XAie_PartitionTeardown(&DevInst);
	XAie_Finish(&DevInst);

	XAie_Print("AIE error interrupt test successful.\n");

	cleanup_platform();

	return 0;
}

/** @} */
