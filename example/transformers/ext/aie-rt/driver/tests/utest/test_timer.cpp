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

TEST_GROUP(TimerApis)
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

TEST(TimerApis, TimerPlTile)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(1, 0);

	RC = XAie_ResetTimer(&DevInst, TileLoc, XAIE_PL_MOD);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_SetTimerTrigEventVal(&DevInst, TileLoc, XAIE_PL_MOD, 20, 33);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_WaitCycles(&DevInst, TileLoc, XAIE_PL_MOD,(u64)0);
	CHECK_EQUAL(XAIE_OK, RC);

	u64 TimeVal;

	RC = XAie_ReadTimer(&DevInst, TileLoc, XAIE_PL_MOD, &TimeVal);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_SetTimerResetEvent(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_PL, XAIE_RESETENABLE);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(TimerApis, TimerCoreModAieTile)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(5, 4);

	RC = XAie_SetTimerTrigEventVal(&DevInst, TileLoc,
			XAIE_CORE_MOD, 20, 4);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_ResetTimer(&DevInst, TileLoc, XAIE_CORE_MOD);
	CHECK_EQUAL(XAIE_OK, RC);

	u64 TimeVal;

	RC = XAie_ReadTimer(&DevInst, TileLoc, XAIE_CORE_MOD, &TimeVal);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_SetTimerResetEvent(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_TRUE_CORE, XAIE_RESETENABLE);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(TimerApis, TimerMemModAieTile)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(4, 5);

	RC = XAie_SetTimerTrigEventVal(&DevInst, TileLoc,
			XAIE_MEM_MOD, 20, 33);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_ResetTimer(&DevInst, TileLoc, XAIE_MEM_MOD);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_SetTimerResetEvent(&DevInst, TileLoc, XAIE_MEM_MOD,
			XAIE_EVENT_TRUE_MEM, XAIE_RESETENABLE);
	CHECK_EQUAL(XAIE_OK, RC);
}

#if AIE_GEN > 1
TEST(TimerApis, TimerMemTile)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 1);

	RC = XAie_SetTimerTrigEventVal(&DevInst, TileLoc,
			XAIE_MEM_MOD, 20, 33);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_WaitCycles(&DevInst, TileLoc, XAIE_MEM_MOD,(u64)0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_ResetTimer(&DevInst, TileLoc, XAIE_MEM_MOD);
	CHECK_EQUAL(XAIE_OK, RC);
}
#endif

TEST_GROUP(TimerNegs)
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

TEST(TimerNegs, InvalidArgs)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_SetTimerTrigEventVal(&DevInst, TileLoc, XAIE_PL_MOD, 20, 33);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_SetTimerResetEvent(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_MEM, XAIE_RESETENABLE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_WaitCycles(&DevInst, TileLoc, XAIE_PL_MOD,(u64)0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_ResetTimer(&DevInst, TileLoc, XAIE_PL_MOD);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	u64 TimeVal;

	RC = XAie_ReadTimer(&DevInst, TileLoc, XAIE_PL_MOD, &TimeVal);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_SetTimerTrigEventVal(NULL, TileLoc, XAIE_CORE_MOD, 20, 33);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_ResetTimer(NULL, TileLoc, XAIE_MEM_MOD);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_SetTimerResetEvent(NULL, TileLoc, XAIE_MEM_MOD,
			XAIE_EVENT_TRUE_MEM, XAIE_RESETENABLE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_ReadTimer(NULL, TileLoc, XAIE_PL_MOD, &TimeVal);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_ReadTimer(&DevInst, TileLoc, XAIE_PL_MOD, NULL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_WaitCycles(NULL, TileLoc, XAIE_PL_MOD, (u64)0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_SetTimerResetEvent(&DevInst, TileLoc, XAIE_MEM_MOD,
			XAIE_EVENT_TRUE_PL, XAIE_RESETENABLE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_SetTimerResetEvent(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_TRUE_MEM, XAIE_RESETENABLE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	TileLoc = XAie_TileLoc(1, 0);

	RC = XAie_WaitCycles(&DevInst, TileLoc,
			XAIE_PL_MOD, (u64)281474976710656);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 0;
	RC = XAie_ResetTimer(&DevInst, TileLoc, XAIE_PL_MOD);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_SetTimerTrigEventVal(&DevInst, TileLoc, XAIE_PL_MOD, 20, 33);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_WaitCycles(&DevInst, TileLoc, XAIE_PL_MOD,(u64)0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_ReadTimer(&DevInst, TileLoc, XAIE_PL_MOD, &TimeVal);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_SetTimerResetEvent(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_PL, XAIE_RESETENABLE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 1;
}


TEST(TimerNegs, InvalidTile)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(0,	2*XAIE_NUM_ROWS);

	RC = XAie_SetTimerTrigEventVal(&DevInst, TileLoc, XAIE_PL_MOD, 20, 33);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_ResetTimer(&DevInst, TileLoc, XAIE_PL_MOD);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_WaitCycles(&DevInst, TileLoc, XAIE_PL_MOD,(u64)0);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	u64 TimeVal;

	RC = XAie_ReadTimer(&DevInst, TileLoc, XAIE_PL_MOD, &TimeVal);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_SetTimerResetEvent(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_TRUE_CORE, XAIE_RESETENABLE);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

}
