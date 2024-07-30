/******************************************************************************
* Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "CppUTest/TestHarness.h"
#include <hw_config.h>

#if AIE_GEN >= 2
static XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR,
		XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
		XAIE_NUM_COLS, XAIE_NUM_ROWS, XAIE_SHIM_ROW,
		XAIE_MEM_TILE_ROW_START, XAIE_MEM_TILE_NUM_ROWS,
		XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS);

static XAie_InstDeclare(DevInst, &ConfigPtr);

TEST_GROUP(DmaApis_AieMl)
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

TEST(DmaApis_AieMl, BdApis)
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

	RC = XAie_DmaSetOutofOrderBdId(&DmaDesc, 15);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaEnableCompression(&DmaDesc);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetDoubleBuffer(&DmaDesc, 0x6000, XAie_LockInit(5, 1),
			XAie_LockInit(5, 0));
	CHECK_EQUAL(RC, XAIE_FEATURE_NOT_SUPPORTED);

	RC = XAie_DmaConfigFifoMode(&DmaDesc, XAIE_DMA_FIFO_COUNTER_1);
	CHECK_EQUAL(RC, XAIE_FEATURE_NOT_SUPPORTED);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, XAie_TileLoc(8, 1));
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetLock(&DmaDesc, XAie_LockInit(5, 1),
			XAie_LockInit(5, 0));
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaWriteBd(&DevInst, &DmaDesc, XAie_TileLoc(8, 1), 7);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaUpdateBdLen(&DevInst, XAie_TileLoc(8, 1), 256, 7);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaUpdateBdAddr(&DevInst, XAie_TileLoc(8, 1), 0x7400, 7);
	CHECK_EQUAL(RC, XAIE_OK);
}

TEST(DmaApis_AieMl, MultiDimApis)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;

	XAie_LocType TileLoc = XAie_TileLoc(5, 4);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, TileLoc);
	CHECK_EQUAL(RC, XAIE_OK);

	XAie_DmaTensor Tensor;
	XAie_DmaDimDesc Dimensions[2];

	Dimensions[0].AieMlDimDesc.StepSize = 3;
	Dimensions[0].AieMlDimDesc.Wrap = 12;
	Dimensions[1].AieMlDimDesc.StepSize = 4;
	Dimensions[1].AieMlDimDesc.Wrap = 32;

	Tensor.NumDim = 2U;
	Tensor.Dim = Dimensions;

	RC = XAie_DmaSetMultiDimAddr(&DmaDesc, &Tensor, 0x1800, 40);
	CHECK_EQUAL(RC, XAIE_OK);

	Dimensions[0].AieMlDimDesc.StepSize = 0x1FFFF;
	RC = XAie_DmaSetMultiDimAddr(&DmaDesc, &Tensor, 0x1800, 40);
	CHECK_EQUAL(RC, XAIE_ERR);

	RC = XAie_DmaSetBdIteration(&DmaDesc, 15, 8, 9);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetBdIteration(&DmaDesc, 0x1FFFF, 8, 9);
	CHECK_EQUAL(RC, XAIE_ERR);

	RC = XAie_DmaWriteBd(&DevInst, &DmaDesc, TileLoc, 8);
	CHECK_EQUAL(RC, XAIE_OK);
}

TEST(DmaApis_AieMl, MemTileApis)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;

	XAie_LocType TileLoc = XAie_TileLoc(8, 1);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, TileLoc);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetLock(&DmaDesc, XAie_LockInit(5, 1),
			XAie_LockInit(5, 0));
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetPkt(&DmaDesc, XAie_PacketInit(10, 5));
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetOutofOrderBdId(&DmaDesc, 15);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaEnableCompression(&DmaDesc);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaWriteBd(&DevInst, &DmaDesc, XAie_TileLoc(8, 1), 7);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelPushBdToQueue(&DevInst, TileLoc, 2, DMA_MM2S, 2);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelPushBdToQueue(&DevInst, TileLoc, 1, DMA_MM2S, 28);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelSetStartQueue(&DevInst, TileLoc, 2, DMA_S2MM, 2,
			1, 0);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelSetStartQueue(&DevInst, TileLoc, 3, DMA_S2MM, 35,
			1, 0);
	CHECK_EQUAL(RC, XAIE_OK);
}

TEST(DmaApis_AieMl, ShimDmaApis)
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

	RC = XAie_DmaSetOutofOrderBdId(&DmaDesc, 15);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetAddrLen(&DmaDesc, 0x4008, 128);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetAddrLen(&DmaDesc, 0x4004, 128);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetAddrLen(&DmaDesc, 0x4002, 128);
	CHECK_EQUAL(RC, XAIE_INVALID_ADDRESS);

	RC = XAie_DmaSetAddrLen(&DmaDesc, 0x4001, 128);
	CHECK_EQUAL(RC, XAIE_INVALID_ADDRESS);

	XAie_DmaTensor Tensor;
	XAie_DmaDimDesc Dimensions[2];

	Dimensions[0].AieMlDimDesc.StepSize = 3;
	Dimensions[0].AieMlDimDesc.Wrap = 12;
	Dimensions[1].AieMlDimDesc.StepSize = 4;
	Dimensions[1].AieMlDimDesc.Wrap = 32;

	Tensor.NumDim = 2U;
	Tensor.Dim = Dimensions;

	RC = XAie_DmaSetMultiDimAddr(&DmaDesc, &Tensor, 0x1800, 40);
	CHECK_EQUAL(RC, XAIE_OK);

	Dimensions[0].AieMlDimDesc.StepSize = 0x100000;
	RC = XAie_DmaSetMultiDimAddr(&DmaDesc, &Tensor, 0x1800, 40);
	CHECK_EQUAL(RC, XAIE_ERR);

	RC = XAie_DmaSetBdIteration(&DmaDesc, 0x100000, 8, 9);
	CHECK_EQUAL(RC, XAIE_ERR);

	RC = XAie_DmaSetMultiDimAddr(&DmaDesc, &Tensor, 0x4002, 128);
	CHECK_EQUAL(RC, XAIE_INVALID_ADDRESS);

	RC = XAie_DmaSetMultiDimAddr(&DmaDesc, &Tensor, 0x10001, 128);
	CHECK_EQUAL(RC, XAIE_INVALID_ADDRESS);
}

TEST(DmaApis_AieMl, ChannelApis)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_DmaChannelSetStartQueue(&DevInst, TileLoc, 0, DMA_MM2S, 7, 3,
			XAIE_ENABLE);
	CHECK_EQUAL(RC, XAIE_OK);

	XAie_DmaChannelDesc NewDmaChannelDesc;
	RC = XAie_DmaChannelDescInit(&DevInst, &NewDmaChannelDesc, TileLoc);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelEnCompression(&NewDmaChannelDesc, 1);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelEnOutofOrder(&NewDmaChannelDesc, 1);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelSetControllerId(&NewDmaChannelDesc, 1);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelSetFoTMode(&NewDmaChannelDesc, DMA_FoT_NO_COUNTS);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaWriteChannel(&DevInst, &NewDmaChannelDesc, TileLoc, 0,
			DMA_MM2S);
	CHECK_EQUAL(RC, XAIE_OK);

	TileLoc = XAie_TileLoc(8, 1);
	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, TileLoc);
	RC = XAie_DmaSetZeroPadding(&DmaDesc, 1, DMA_ZERO_PADDING_AFTER, 3);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetZeroPadding(&DmaDesc, 2, DMA_ZERO_PADDING_AFTER, 3);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetZeroPadding(&DmaDesc, 0, DMA_ZERO_PADDING_AFTER, 3);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetZeroPadding(&DmaDesc, 1, DMA_ZERO_PADDING_BEFORE, 3);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetZeroPadding(&DmaDesc, 2, DMA_ZERO_PADDING_BEFORE, 3);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetZeroPadding(&DmaDesc, 0, DMA_ZERO_PADDING_BEFORE, 3);
	CHECK_EQUAL(RC, XAIE_OK);

	XAie_PadDesc PadDesc[3];
	XAie_DmaPadTensor PadTensor;

	PadDesc[0].After = 3;
	PadDesc[0].Before = 2;
	PadDesc[1].After = 3;
	PadDesc[1].Before = 2;
	PadDesc[2].After = 3;
	PadDesc[2].Before = 2;

	PadTensor.NumDim = 2;
	PadTensor.PadDesc = PadDesc;
	RC = XAie_DmaSetPadding(&DmaDesc, &PadTensor);
	CHECK_EQUAL(RC, XAIE_OK);
}

TEST(DmaApis_AieMl, MemTileZeroPad)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;

	XAie_LocType TileLoc = XAie_TileLoc(8, 1);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, TileLoc);
	CHECK_EQUAL(RC, XAIE_OK);

	XAie_DmaTensor Tensor;
	XAie_DmaDimDesc Dimensions[3];

	Dimensions[0].AieMlDimDesc.StepSize = 1;
	Dimensions[0].AieMlDimDesc.Wrap = 5;
	Dimensions[1].AieMlDimDesc.StepSize = 5;
	Dimensions[1].AieMlDimDesc.Wrap = 10;
	Dimensions[2].AieMlDimDesc.StepSize = 6;
	Dimensions[2].AieMlDimDesc.Wrap = 1;
	Tensor.NumDim = 3U;
	Tensor.Dim = Dimensions;

	RC = XAie_DmaSetMultiDimAddr(&DmaDesc, &Tensor, 0x1800, 40);
	CHECK_EQUAL(RC, XAIE_OK);

	XAie_PadDesc PadDesc[3];
	XAie_DmaPadTensor PadTensor;

	PadDesc[0].After = 3;
	PadDesc[0].Before = 2;
	PadDesc[1].After = 3;
	PadDesc[1].Before = 2;
	PadDesc[2].After = 3;
	PadDesc[2].Before = 0;
	PadTensor.NumDim = 3;
	PadTensor.PadDesc = PadDesc;

	RC = XAie_DmaSetPadding(&DmaDesc, &PadTensor);
	CHECK_EQUAL(RC, XAIE_OK);

#if AIE_GEN == 4
	RC = XAie_DmaSetPadValue(&DevInst, TileLoc, 3, 0xDEADBEEF);
	CHECK_EQUAL(RC, XAIE_OK);
#endif

#if 0
	RC = XAie_DmaWriteBd(&DevInst, &DmaDesc, TileLoc, 8);
	CHECK_EQUAL(RC, XAIE_INVALID_DMA_DESC);
#endif
}

TEST(DmaApis_AieMl, InvalidArgs)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_DmaDescInit(NULL, &DmaDesc, TileLoc);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaSetOutofOrderBdId(NULL, 15);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaEnableCompression(NULL);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelPushBdToQueue(&DevInst, XAie_TileLoc(8, 1), 1,
			DMA_MM2S, 2);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelPushBdToQueue(&DevInst, XAie_TileLoc(8, 1), 2,
			DMA_MM2S, 25);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelSetStartQueue(NULL, TileLoc, 0, DMA_MM2S, 7, 3,
			XAIE_ENABLE);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelDescInit(&DevInst, NULL, TileLoc);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	XAie_DmaChannelDesc NewDmaChannelDesc;
	RC = XAie_DmaChannelDescInit(NULL, &NewDmaChannelDesc, TileLoc);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelEnCompression(NULL, 1);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelEnOutofOrder(NULL, 1);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelSetControllerId(NULL, 1);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelSetFoTMode(NULL, DMA_FoT_NO_COUNTS);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaWriteChannel(NULL, NULL, TileLoc, 0,
			DMA_MM2S);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaSetZeroPadding(NULL, 1, DMA_ZERO_PADDING_AFTER, 3);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	TileLoc = XAie_TileLoc(8, 1);
	RC = XAie_DmaSetZeroPadding(NULL, 1, DMA_ZERO_PADDING_AFTER, 3);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);
}

#endif /* AIE_GEN == 2 || 3 || 4 */
