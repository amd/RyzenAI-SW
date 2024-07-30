/******************************************************************************
* Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "CppUTest/TestHarness.h"
#include <hw_config.h>

#if AIE_GEN == 1

static XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR,
		XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
		XAIE_NUM_COLS, XAIE_NUM_ROWS, XAIE_SHIM_ROW,
		XAIE_MEM_TILE_ROW_START, XAIE_MEM_TILE_NUM_ROWS,
		XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS);

static XAie_InstDeclare(DevInst, &ConfigPtr);

TEST_GROUP(DmaApis_Aie)
{
	void setup()
	{
		AieRC RC;

		RC = XAie_CfgInitialize(&(DevInst), &ConfigPtr);
		CHECK_EQUAL(XAIE_OK, RC);

		RC = XAie_PartitionInitialize(&(DevInst), NULL);
		CHECK_EQUAL(XAIE_OK, RC);
	}
	void teardown()
	{
		AieRC RC;

		RC = XAie_PartitionTeardown(&(DevInst));
		CHECK_EQUAL(XAIE_OK, RC);

		XAie_Finish(&DevInst);
	}
};

TEST(DmaApis_Aie, BdApis)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, TileLoc);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetLock(&DmaDesc, XAie_LockInit(5, 1),
			XAie_LockInit(5, 0));
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetPkt(&DmaDesc, XAie_PacketInit(10, 5));
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetDoubleBuffer(&DmaDesc, 0x6000, XAie_LockInit(5, 1),
			XAie_LockInit(5, 0));
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaConfigFifoMode(&DmaDesc, XAIE_DMA_FIFO_COUNTER_1);
	CHECK_EQUAL(RC, XAIE_OK);
}

TEST(DmaApis_Aie, MultiDimApis)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;

	XAie_LocType TileLoc = XAie_TileLoc(5, 4);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, TileLoc);
	CHECK_EQUAL(RC, XAIE_OK);

	XAie_DmaTensor Tensor;
	XAie_DmaDimDesc Dimensions[2];

	Dimensions[0].AieDimDesc.Offset = 1;
	Dimensions[0].AieDimDesc.Incr = 1;
	Dimensions[0].AieDimDesc.Wrap = 32;
	Dimensions[1].AieDimDesc.Offset = 2;
	Dimensions[1].AieDimDesc.Incr = 2;
	Dimensions[1].AieDimDesc.Wrap = 20;

	Tensor.NumDim = 2U;
	Tensor.Dim = Dimensions;

	RC = XAie_DmaSetMultiDimAddr(&DmaDesc, &Tensor, 0x1800, 40);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaWriteBd(&DevInst, &DmaDesc, TileLoc, 8);
	CHECK_EQUAL(RC, XAIE_OK);
}

TEST(DmaApis_Aie, ShimDmaApis)
{
	AieRC RC = XAIE_OK;
	XAie_DmaDesc DmaDesc;

	XAie_LocType TileLoc = XAie_TileLoc(2, 0);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, TileLoc);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetLock(&DmaDesc, XAie_LockInit(5, 1),
			XAie_LockInit(5, 0));
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetPkt(&DmaDesc, XAie_PacketInit(10, 5));
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaConfigFifoMode(&DmaDesc, XAIE_DMA_FIFO_COUNTER_1);
	CHECK_EQUAL(RC, XAIE_FEATURE_NOT_SUPPORTED);

	RC = XAie_DmaWriteBd(&DevInst, &DmaDesc, TileLoc, 7);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelPauseStream(&DevInst, TileLoc, 1, DMA_S2MM,
			XAIE_ENABLE);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelPauseMem(&DevInst, TileLoc, 1, DMA_S2MM,
			XAIE_ENABLE);
	CHECK_EQUAL(RC, XAIE_OK);
}

TEST(DmaApis_Aie, InvalidArgs)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_DmaDescInit(NULL, &DmaDesc, TileLoc);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaSetDoubleBuffer(NULL, 0x6000, XAie_LockInit(5, 1),
			XAie_LockInit(5, 0));
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaConfigFifoMode(NULL, XAIE_DMA_FIFO_COUNTER_1);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);
}
#endif /* AIE_GEN == 1 */
