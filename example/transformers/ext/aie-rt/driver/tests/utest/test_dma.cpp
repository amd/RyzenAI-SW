/******************************************************************************
* Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "CppUTest/TestHarness.h"
#include <hw_config.h>

static XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR,
		XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
		XAIE_NUM_COLS, XAIE_NUM_ROWS, XAIE_SHIM_ROW,
		XAIE_MEM_TILE_ROW_START, XAIE_MEM_TILE_NUM_ROWS,
		XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS);

static XAie_InstDeclare(DevInst, &ConfigPtr);

TEST_GROUP(DmaApis)
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

TEST(DmaApis, AxiBurstLenCheck)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;
	XAie_LocType loc = XAie_TileLoc(3, 0);

	printf("AxiBurstLenCheck test:\n");

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, loc);
	CHECK_EQUAL(RC, XAIE_OK);

	CHECK(DmaDesc.DmaMod->AxiBurstLenCheck != NULL);

	RC = XAie_DmaSetAxi(&DmaDesc, 16U, 4U, 0U, 0U, 0U);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetAxi(&DmaDesc, 16U, 8U, 0U, 0U, 0U);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetAxi(&DmaDesc, 16U, 16U, 0U, 0U, 0U);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetAxi(&DmaDesc, 16U, 32U, 0U, 0U, 0U);
	switch (DevInst.DevProp.DevGen) {
	case XAIE_DEV_GEN_AIE2P:
	case XAIE_DEV_GEN_AIE2PS:
	case XAIE_DEV_GEN_AIE2P_STRIX_A0:
	case XAIE_DEV_GEN_AIE2P_STRIX_B0:
		CHECK_EQUAL(RC, XAIE_OK);
		break;
	default:
		CHECK_EQUAL(RC, XAIE_INVALID_BURST_LENGTH);
	}
	RC = XAie_DmaSetAxi(&DmaDesc, 16U, 3U, 0U, 0U, 0U);
	CHECK_EQUAL(RC, XAIE_INVALID_BURST_LENGTH);
};

TEST(DmaApis, BdApis)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;
	u32 Len;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, TileLoc);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetLock(&DmaDesc, XAie_LockInit(5, 1),
			XAie_LockInit(5, 0));
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetPkt(&DmaDesc, XAie_PacketInit(10, 5));
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetAddrLen(&DmaDesc, 0x4000, 128);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetNextBd(&DmaDesc, 5, XAIE_ENABLE);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaEnableBd(&DmaDesc);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetAxi(&DmaDesc, 16U, 4U, 0U, 0U, 0U);
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);

	RC = XAie_DmaWriteBd(&DevInst, &DmaDesc, TileLoc, 7);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaGetBdLen(&DevInst, TileLoc, &Len, 7);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaUpdateBdLen(&DevInst, TileLoc, 256, 7);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaUpdateBdAddr(&DevInst, TileLoc, 0x6000, 7);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, XAie_TileLoc(2, 0));
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaWriteBd(&DevInst, &DmaDesc, XAie_TileLoc(2, 0), 7);
	CHECK_EQUAL(RC, XAIE_OK);
}

TEST(DmaApis, ChannelApis)
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

	RC = XAie_DmaSetAddrLen(&DmaDesc, 0x4000, 128);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetNextBd(&DmaDesc, 5, XAIE_ENABLE);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaEnableBd(&DmaDesc);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaWriteBd(&DevInst, &DmaDesc, TileLoc, 7);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelResetAll(&DevInst, TileLoc, DMA_CHANNEL_RESET);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelResetAll(&DevInst, TileLoc, DMA_CHANNEL_UNRESET);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelReset(&DevInst, TileLoc, 0U, DMA_MM2S,
			DMA_CHANNEL_RESET);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelReset(&DevInst, TileLoc, 0U, DMA_MM2S,
			DMA_CHANNEL_UNRESET);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelPushBdToQueue(&DevInst, TileLoc, 1, DMA_MM2S, 5);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelEnable(&DevInst, TileLoc, 1, DMA_MM2S);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelDisable(&DevInst, TileLoc, 1, DMA_MM2S);
	CHECK_EQUAL(RC, XAIE_OK);

	u8 QueueSize;
	RC = XAie_DmaGetMaxQueueSize(&DevInst, TileLoc, &QueueSize);
	CHECK_EQUAL(RC, XAIE_OK);
}

TEST(DmaApis, ShimDmaApis)
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

	RC = XAie_DmaSetAddrLen(&DmaDesc, 0x4000, 128);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetAxi(&DmaDesc, 0, 4, 0, 0, 0);
	CHECK_EQUAL(RC, XAIE_OK);
	
	RC = XAie_DmaWriteBd(&DevInst, &DmaDesc, TileLoc, 7);
	CHECK_EQUAL(RC, XAIE_OK);

	u32 Len;
	RC = XAie_DmaGetBdLen(&DevInst, TileLoc, &Len,7);
	CHECK_EQUAL(RC, XAIE_OK);

#if AIE_GEN >= 2
	RC = XAie_DmaUpdateBdLen(&DevInst, TileLoc, 256, 7);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaUpdateBdAddr(&DevInst, TileLoc, 0x6000, 7);
	CHECK_EQUAL(RC, XAIE_OK);
#endif
}

TEST_GROUP(DmaStatusApis)
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

TEST(DmaStatusApis, TileDmaStatusApis)
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

	RC = XAie_DmaSetAddrLen(&DmaDesc, 0x4000, 128);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaDisableBd(&DmaDesc);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaEnableBd(&DmaDesc);
	CHECK_EQUAL(RC, XAIE_OK);

	u8 Bd = 5;
	RC = XAie_DmaWriteBd(&DevInst, &DmaDesc, TileLoc, Bd);
	CHECK_EQUAL(RC, XAIE_OK);

	/* MM2S Channel 1 */
	RC = XAie_DmaChannelPushBdToQueue(&DevInst, TileLoc, 1, DMA_MM2S, Bd);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelEnable(&DevInst, TileLoc, 1, DMA_MM2S);
	CHECK_EQUAL(RC, XAIE_OK);

	u8 BdCount = 0;
	RC = XAie_DmaGetPendingBdCount(&DevInst, TileLoc, 1, DMA_MM2S,
			&BdCount);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaWaitForDone(&DevInst, TileLoc, 1, DMA_MM2S, 0);
	CHECK_EQUAL(RC, XAIE_ERR);

	/* MM2S Channel 0 */
	RC = XAie_DmaChannelPushBdToQueue(&DevInst, TileLoc, 0, DMA_MM2S, Bd);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelEnable(&DevInst, TileLoc, 0, DMA_MM2S);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaGetPendingBdCount(&DevInst, TileLoc, 0, DMA_MM2S,
			&BdCount);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaWaitForDone(&DevInst, TileLoc, 0, DMA_MM2S, 0);
	CHECK_EQUAL(RC, XAIE_ERR);

	/* S2MM Channel 1 */
	RC = XAie_DmaChannelPushBdToQueue(&DevInst, TileLoc, 1, DMA_S2MM, Bd);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelEnable(&DevInst, TileLoc, 1, DMA_S2MM);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaGetPendingBdCount(&DevInst, TileLoc, 1, DMA_S2MM,
			&BdCount);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaWaitForDone(&DevInst, TileLoc, 1, DMA_S2MM, 0);
	CHECK_EQUAL(RC, XAIE_ERR);

	/* S2MM Channel 0 */
	RC = XAie_DmaChannelPushBdToQueue(&DevInst, TileLoc, 0, DMA_S2MM, Bd);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelEnable(&DevInst, TileLoc, 0, DMA_S2MM);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaGetPendingBdCount(&DevInst, TileLoc, 0, DMA_S2MM,
			&BdCount);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaWaitForDone(&DevInst, TileLoc, 0, DMA_S2MM, 0);
	CHECK_EQUAL(RC, XAIE_ERR);
}

TEST(DmaStatusApis, ShimDmaStatusApis)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;

	XAie_LocType TileLoc = XAie_TileLoc(2, 0);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, TileLoc);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetLock(&DmaDesc, XAie_LockInit(5, 1),
			XAie_LockInit(5, 0));
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetPkt(&DmaDesc, XAie_PacketInit(10, 5));
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaSetAddrLen(&DmaDesc, 0x4000, 128);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaEnableBd(&DmaDesc);
	CHECK_EQUAL(RC, XAIE_OK);
	
	u8 Bd = 5;
	RC = XAie_DmaWriteBd(&DevInst, &DmaDesc, TileLoc, Bd);
	CHECK_EQUAL(RC, XAIE_OK);

	/* MM2S Channel 1 */
	RC = XAie_DmaChannelPushBdToQueue(&DevInst, TileLoc, 1, DMA_MM2S, Bd);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelEnable(&DevInst, TileLoc, 1, DMA_MM2S);
	CHECK_EQUAL(RC, XAIE_OK);

	u8 BdCount = 0;
	RC = XAie_DmaGetPendingBdCount(&DevInst, TileLoc, 1, DMA_MM2S,
			&BdCount);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaWaitForDone(&DevInst, TileLoc, 1, DMA_MM2S, 0);
	CHECK_EQUAL(RC, XAIE_OK);

	/* MM2S Channel 0 */
	RC = XAie_DmaChannelPushBdToQueue(&DevInst, TileLoc, 0, DMA_MM2S, Bd);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelEnable(&DevInst, TileLoc, 0, DMA_MM2S);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaGetPendingBdCount(&DevInst, TileLoc, 0, DMA_MM2S,
			&BdCount);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaWaitForDone(&DevInst, TileLoc, 0, DMA_MM2S, 0);
	CHECK_EQUAL(RC, XAIE_OK);

	/* S2MM Channel 1 */
	RC = XAie_DmaChannelPushBdToQueue(&DevInst, TileLoc, 1, DMA_S2MM, Bd);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelEnable(&DevInst, TileLoc, 1, DMA_S2MM);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaGetPendingBdCount(&DevInst, TileLoc, 1, DMA_S2MM,
			&BdCount);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaWaitForDone(&DevInst, TileLoc, 1, DMA_S2MM, 0);
	CHECK_EQUAL(RC, XAIE_OK);

	/* S2MM Channel 0 */
	RC = XAie_DmaChannelPushBdToQueue(&DevInst, TileLoc, 0, DMA_S2MM, Bd);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaChannelEnable(&DevInst, TileLoc, 0, DMA_S2MM);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaGetPendingBdCount(&DevInst, TileLoc, 0, DMA_S2MM,
			&BdCount);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_DmaWaitForDone(&DevInst, TileLoc, 0, DMA_S2MM, 0);
	CHECK_EQUAL(RC, XAIE_OK);
}

TEST_GROUP(DmaInvalidApis)
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

TEST(DmaInvalidApis, InvalidArgs)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_DmaDescInit(NULL, &DmaDesc, TileLoc);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaSetLock(NULL, XAie_LockInit(5, 1),
			XAie_LockInit(5, 0));
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaSetPkt(NULL, XAie_PacketInit(10, 5));
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaSetAddrLen(NULL, 0x4000, 128);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaSetNextBd(NULL, 5, XAIE_ENABLE);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaDisableBd(NULL);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaEnableBd(NULL);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaSetAxi(NULL, 16U, 4U, 0U, 0U, 0U);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaSetInterleaveEnable(NULL, 1, 5, 22);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaWriteBd(&DevInst, NULL, TileLoc, 7);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelResetAll(NULL, TileLoc, DMA_CHANNEL_RESET);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelReset(NULL, TileLoc, 0U, DMA_MM2S,
			DMA_CHANNEL_RESET);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelPauseStream(NULL, TileLoc, 1, DMA_S2MM,
			XAIE_ENABLE);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelPauseMem(NULL, TileLoc, 1, DMA_S2MM,
			XAIE_ENABLE);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelPushBdToQueue(NULL, TileLoc, 1, DMA_MM2S, 5);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelEnable(NULL, TileLoc, 1, DMA_MM2S);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	u32 Len;
	RC = XAie_DmaGetBdLen(NULL, TileLoc, &Len, 7);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaGetBdLen(&DevInst, TileLoc, NULL, 7);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	u8 BdCount = 0;
	RC = XAie_DmaGetPendingBdCount(NULL, TileLoc, 1, DMA_MM2S,
			&BdCount);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaWaitForDone(NULL, TileLoc, 1, DMA_MM2S, 0);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelReset(&DevInst, TileLoc, 0U, DMA_MAX,
			DMA_CHANNEL_RESET);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelPauseStream(&DevInst, TileLoc, 1, DMA_MAX,
			XAIE_ENABLE);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelPauseMem(&DevInst, TileLoc, 1, DMA_MAX,
			XAIE_ENABLE);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelPushBdToQueue(&DevInst, TileLoc, 1, DMA_MAX, 5);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaChannelEnable(&DevInst, TileLoc, 1, DMA_MAX);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaGetPendingBdCount(&DevInst, TileLoc, 1, DMA_MAX,
			&BdCount);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_DmaWaitForDone(&DevInst, TileLoc, 1, DMA_MAX, 0);
#if AIE_GEN == 1
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);
#endif
}

TEST(DmaInvalidApis, InvalidTile)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;

	XAie_LocType TileLoc = XAie_TileLoc(0, 0);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, TileLoc);
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, XAie_TileLoc(8, 3));

	RC = XAie_DmaWriteBd(&DevInst, &DmaDesc, TileLoc, 7);
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);

	RC = XAie_DmaChannelReset(&DevInst, TileLoc, 0U, DMA_MM2S,
			DMA_CHANNEL_RESET);
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);

	RC = XAie_DmaChannelResetAll(&DevInst, TileLoc, DMA_CHANNEL_RESET);
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);

	RC = XAie_DmaChannelPauseStream(&DevInst, TileLoc, 1, DMA_S2MM,
			XAIE_ENABLE);
#if AIE_GEN == 1
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);
#endif

	RC = XAie_DmaChannelPauseMem(&DevInst, TileLoc, 1, DMA_S2MM,
			XAIE_ENABLE);
#if AIE_GEN == 1
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);
#endif

	RC = XAie_DmaChannelPushBdToQueue(&DevInst, TileLoc, 1, DMA_MM2S, 5);
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);

	RC = XAie_DmaChannelEnable(&DevInst, TileLoc, 1, DMA_MM2S);
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);

	u32 Len;
	RC = XAie_DmaGetBdLen(&DevInst, TileLoc, &Len, 7);
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);

	u8 BdCount = 0;
	RC = XAie_DmaGetPendingBdCount(&DevInst, TileLoc, 1, DMA_MM2S,
			&BdCount);
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);

	RC = XAie_DmaWaitForDone(&DevInst, TileLoc, 1, DMA_MM2S, 0);
#if AIE_GEN == 1
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);
#endif
}

TEST(DmaInvalidApis, InvalidChannelNumber)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, TileLoc);

	RC = XAie_DmaChannelReset(&DevInst, TileLoc, 3, DMA_MM2S,
			DMA_CHANNEL_RESET);
	CHECK_EQUAL(RC, XAIE_INVALID_CHANNEL_NUM);

	RC = XAie_DmaChannelPauseStream(&DevInst, XAie_TileLoc(2, 0), 100,
			DMA_S2MM, XAIE_ENABLE);
#if AIE_GEN == 1
	CHECK_EQUAL(RC, XAIE_INVALID_CHANNEL_NUM);
#endif

	RC = XAie_DmaChannelPauseMem(&DevInst, XAie_TileLoc(2, 0), 3,
			DMA_S2MM, XAIE_ENABLE);
#if AIE_GEN == 1
	CHECK_EQUAL(RC, XAIE_INVALID_CHANNEL_NUM);
#endif

	RC = XAie_DmaChannelEnable(&DevInst, TileLoc, 3, DMA_MM2S);
	CHECK_EQUAL(RC, XAIE_INVALID_CHANNEL_NUM);

	u8 BdCount = 0;
	RC = XAie_DmaGetPendingBdCount(&DevInst, TileLoc, 3, DMA_MM2S,
			&BdCount);
	CHECK_EQUAL(RC, XAIE_INVALID_CHANNEL_NUM);

	RC = XAie_DmaWaitForDone(&DevInst, TileLoc, 3, DMA_MM2S, 0);
	CHECK_EQUAL(RC, XAIE_INVALID_CHANNEL_NUM);

	RC = XAie_DmaChannelPushBdToQueue(&DevInst, TileLoc, 255, DMA_MM2S, 5);
	CHECK_EQUAL(RC, XAIE_INVALID_CHANNEL_NUM);
}

TEST(DmaInvalidApis, OtherErrorCodes)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, TileLoc);

	RC = XAie_DmaSetLock(&DmaDesc, XAie_LockInit(20, 1),
			XAie_LockInit(5, 0));
	CHECK_EQUAL(RC, XAIE_INVALID_LOCK_ID);

#if AIE_GEN == 1
	RC = XAie_DmaSetDoubleBuffer(&DmaDesc, 0x6000, XAie_LockInit(20, 1),
			 XAie_LockInit(5, 0));
	CHECK_EQUAL(RC, XAIE_INVALID_LOCK_ID);

	RC = XAie_DmaSetDoubleBuffer(&DmaDesc, 0x9005430, XAie_LockInit(5, 1),
			XAie_LockInit(5, 0));
	CHECK_EQUAL(RC, XAIE_INVALID_ADDRESS);

	RC = XAie_DmaSetLock(&DmaDesc, XAie_LockInit(6, 1),
			XAie_LockInit(5, 0));
	CHECK_EQUAL(RC, XAIE_INVALID_LOCK_ID);
#endif

	RC = XAie_DmaSetAddrLen(&DmaDesc, 0x4025300, 128);
	CHECK_EQUAL(RC, XAIE_INVALID_ADDRESS);

	RC = XAie_DmaSetNextBd(&DmaDesc, 255, XAIE_ENABLE);
	CHECK_EQUAL(RC, XAIE_INVALID_BD_NUM);

	RC = XAie_DmaWriteBd(&DevInst, &DmaDesc, TileLoc, 255);
	CHECK_EQUAL(RC, XAIE_INVALID_BD_NUM);

#if AIE_GEN == 2
	XAie_DmaDesc DmaDescUninitialized;
	DmaDescUninitialized.IsReady = 0;
#endif

	RC = XAie_DmaChannelPushBdToQueue(&DevInst, TileLoc, 1, DMA_MM2S, 255);
	CHECK_EQUAL(RC, XAIE_INVALID_BD_NUM);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, XAie_TileLoc(2, 0));
	RC = XAie_DmaSetAxi(&DmaDesc, 16U, 5U, 0U, 0U, 0U);
	CHECK_EQUAL(RC, XAIE_INVALID_BURST_LENGTH);

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, XAie_TileLoc(1, 0));
	RC = XAie_DmaSetInterleaveEnable(&DmaDesc, 1, 5, 22);
	CHECK_EQUAL(RC, XAIE_FEATURE_NOT_SUPPORTED);

	u32 Len;
	RC = XAie_DmaGetBdLen(&DevInst, TileLoc, &Len, 255);
	CHECK_EQUAL(RC, XAIE_INVALID_BD_NUM);
}

TEST(DmaInvalidApis, MultiDimAddressErrors)
{
	AieRC RC;
	XAie_DmaDesc DmaDesc;

	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, XAie_TileLoc(8, 3));

	XAie_DmaTensor Tensor;
	XAie_DmaDimDesc Dimensions[2];

#if AIE_GEN == 1
	Dimensions[0].AieDimDesc.Offset = 1;
	Dimensions[0].AieDimDesc.Incr = 1;
	Dimensions[0].AieDimDesc.Wrap = 32;
	Dimensions[1].AieDimDesc.Offset = 2;
	Dimensions[1].AieDimDesc.Incr = 2;
	Dimensions[1].AieDimDesc.Wrap = 20;
#elif AIE_GEN >= 2
	Dimensions[0].AieMlDimDesc.StepSize = 3;
	Dimensions[0].AieMlDimDesc.Wrap = 12;
	Dimensions[1].AieMlDimDesc.StepSize = 4;
	Dimensions[1].AieMlDimDesc.Wrap = 32;
#endif

	Tensor.NumDim = 2U;
	Tensor.Dim = Dimensions;

	RC = XAie_DmaSetMultiDimAddr(NULL, NULL, 0x1800, 40);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);
#if AIE_GEN >= 2

	/*AIEML AIE Tile */
	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, XAie_TileLoc(5, 4));
	Tensor.NumDim = 4U;
	RC = XAie_DmaSetMultiDimAddr(&DmaDesc, &Tensor, 0x1800, 40);
	CHECK_EQUAL(RC, XAIE_FEATURE_NOT_SUPPORTED);

	/*AIEML Shim Tile*/
	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, XAie_TileLoc(1, 0));
	Tensor.NumDim = 4U;
	RC = XAie_DmaSetMultiDimAddr(&DmaDesc, &Tensor, 0x1800, 40);
	CHECK_EQUAL(RC, XAIE_FEATURE_NOT_SUPPORTED);

	/*AIEML Mem Tile*/
	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, XAie_TileLoc(8, 3));
	Tensor.NumDim = 5U;
	RC = XAie_DmaSetMultiDimAddr(&DmaDesc, &Tensor, 0x1800, 40);
	CHECK_EQUAL(RC, XAIE_FEATURE_NOT_SUPPORTED);

	Tensor.NumDim = 2U;
	Dimensions[1].AieMlDimDesc.StepSize = 0;
	RC = XAie_DmaSetMultiDimAddr(&DmaDesc, &Tensor, 0x1800, 40);
	CHECK_EQUAL(RC, XAIE_ERR);

#elif AIE_GEN == 1

	/*AIE Tile */
	RC = XAie_DmaDescInit(&DevInst, &DmaDesc, XAie_TileLoc(5, 4));
	Tensor.NumDim = 3U;
	RC = XAie_DmaSetMultiDimAddr(&DmaDesc, &Tensor, 0x1800, 40);
	CHECK_EQUAL(RC, XAIE_FEATURE_NOT_SUPPORTED);
#endif
}
