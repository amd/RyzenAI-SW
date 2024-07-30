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

#if AIE_GEN != 4
TEST_GROUP(InterruptApis)
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

TEST(InterruptApis, InterruptSuccesses)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(2, 0);

	
	RC = XAie_IntrCtrlL1Disable(&DevInst, TileLoc, XAIE_EVENT_SWITCH_A, 1);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_IntrCtrlL1BroadcastBlock(&DevInst, TileLoc,
			XAIE_EVENT_SWITCH_A, 5);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_IntrCtrlL1BroadcastUnblock(&DevInst, TileLoc,
			XAIE_EVENT_SWITCH_A, 5);
	CHECK_EQUAL(RC, XAIE_OK);

	RC = XAie_IntrCtrlL2Disable(&DevInst, TileLoc, 5);
	CHECK_EQUAL(RC, XAIE_OK);

/*
	* TODO: remove below macro guard based on device generation once
	* interrupt apis are validated for the device.
	*/
#if AIE_GEN == 1 || AIE_GEN == 2
	RC = XAie_ErrorHandlingInit(&DevInst);
	CHECK_EQUAL(RC, XAIE_OK);
#endif
}

TEST_GROUP(InvalidInterrupts){
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

TEST(InvalidInterrupts, InvalidDevInst){
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(2, 0);

	RC = XAie_IntrCtrlL1Enable(NULL, TileLoc, XAIE_EVENT_SWITCH_A, 1);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1Disable(NULL, TileLoc, XAIE_EVENT_SWITCH_A, 1);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1IrqSet(NULL, TileLoc, XAIE_EVENT_SWITCH_A, 1);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1Event(NULL, TileLoc, XAIE_EVENT_SWITCH_A, 1,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1BroadcastBlock(NULL, TileLoc,
			XAIE_EVENT_SWITCH_A, 5);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1BroadcastUnblock(NULL, TileLoc,
			 XAIE_EVENT_SWITCH_A, 5);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL2Enable(NULL, TileLoc, 5);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL2Disable(NULL, TileLoc, 5);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_ErrorHandlingInit(NULL);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);
}

TEST(InvalidInterrupts, InvalidTile){
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(XAIE_NUM_COLS + 2,
			XAIE_NUM_ROWS + 2);

	RC = XAie_IntrCtrlL1Enable(&DevInst, TileLoc, XAIE_EVENT_SWITCH_A, 1);
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);

	RC = XAie_IntrCtrlL1IrqSet(&DevInst, TileLoc, XAIE_EVENT_SWITCH_A, 1);
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);

	RC = XAie_IntrCtrlL1Event(&DevInst, TileLoc, XAIE_EVENT_SWITCH_A, 1,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);

	RC = XAie_IntrCtrlL1BroadcastBlock(&DevInst, TileLoc,
			XAIE_EVENT_SWITCH_A, 5);
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);

	RC = XAie_IntrCtrlL1BroadcastUnblock(&DevInst, TileLoc,
			XAIE_EVENT_SWITCH_A, 5);
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);

	TileLoc = XAie_TileLoc(3, 1);

	RC = XAie_IntrCtrlL2Enable(&DevInst, TileLoc, 5);
	CHECK_EQUAL(RC, XAIE_INVALID_TILE);
}

TEST(InvalidInterrupts, OtherErrors){
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(2, 0);

	RC = XAie_IntrCtrlL1Enable(&DevInst, TileLoc, XAIE_EVENT_SWITCH_A, 99);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1IrqSet(&DevInst, TileLoc, XAIE_EVENT_SWITCH_A, 99);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1Event(&DevInst, TileLoc, XAIE_EVENT_SWITCH_A, 99,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1BroadcastBlock(&DevInst, TileLoc,
			XAIE_EVENT_SWITCH_A, -2);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1BroadcastUnblock(&DevInst, TileLoc,
			XAIE_EVENT_SWITCH_A, -2);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL2Enable(&DevInst, TileLoc, -2);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1Event(&DevInst, TileLoc, XAIE_EVENT_SWITCH_A, 2,
			XAIE_EVENT_TRUE_CORE);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1Event(&DevInst, XAie_TileLoc(1, 0),
			XAIE_EVENT_SWITCH_A, 0,
			XAIE_EVENT_DMA_S2MM_0_START_BD_PL);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1Event(&DevInst, TileLoc, XAIE_EVENT_SWITCH_A, 1,
			XAIE_EVENT_GROUP_0_MEM);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	TileLoc = XAie_TileLoc(5, 4);
	RC = XAie_IntrCtrlL1BroadcastBlock(&DevInst, TileLoc,
			XAIE_EVENT_SWITCH_A, 5);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1BroadcastUnblock(&DevInst, TileLoc,
			XAIE_EVENT_SWITCH_A, 5);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1Enable(&DevInst, TileLoc, XAIE_EVENT_SWITCH_A, 1);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1IrqSet(&DevInst, TileLoc, XAIE_EVENT_SWITCH_A, 2);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1Event(&DevInst, TileLoc, XAIE_EVENT_SWITCH_A, 2,
			XAIE_EVENT_TRUE_MEM);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	DevInst.IsReady = 0;
	RC = XAie_IntrCtrlL1Enable(&DevInst, TileLoc, XAIE_EVENT_SWITCH_A, 1);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1IrqSet(&DevInst, TileLoc, XAIE_EVENT_SWITCH_A, 1);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1Event(&DevInst, TileLoc, XAIE_EVENT_SWITCH_A, 1,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1BroadcastBlock(&DevInst, TileLoc,
			XAIE_EVENT_SWITCH_A, 5);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL1BroadcastUnblock(&DevInst, TileLoc,
			XAIE_EVENT_SWITCH_A, 5);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_IntrCtrlL2Disable(&DevInst, TileLoc, 5);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	RC = XAie_ErrorHandlingInit(&DevInst);
	CHECK_EQUAL(RC, XAIE_INVALID_ARGS);

	DevInst.IsReady = 1;
}
#endif
