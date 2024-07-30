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

TEST_GROUP(EventApis_Aie)
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

TEST(EventApis_Aie, Errors)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(5, 4);

	RC = XAie_EventGenerate(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventComboConfig(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_COMBO0, XAIE_EVENT_COMBO_E1_AND_E2,
			XAIE_EVENT_COMBO_EVENT_0_CORE,
			XAIE_EVENT_GROUP_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcast(&DevInst, TileLoc, XAIE_CORE_MOD, 1,
			XAIE_EVENT_GROUP_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventComboConfig(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_COMBO0, XAIE_EVENT_COMBO_E1_AND_E2,
			XAIE_EVENT_GROUP_0_MEM,
			XAIE_EVENT_FP_OVERFLOW_CORE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);
}

#endif /* AIE_GEN == 1 */
