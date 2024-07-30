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

TEST_GROUP(EventApis)
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

TEST(EventApis, EventPlMod)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(1, 0);

	RC = XAie_EventGenerate(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventComboConfig(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_COMBO0, XAIE_EVENT_COMBO_E1_AND_E2,
			XAIE_EVENT_COMBO_EVENT_0_PL,
			XAIE_EVENT_COMBO_EVENT_1_PL);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventComboConfig(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_COMBO2, XAIE_EVENT_COMBO_E1_AND_E2,
			XAIE_EVENT_COMBO_EVENT_0_PL,
			XAIE_EVENT_COMBO_EVENT_1_PL);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventComboReset(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_COMBO0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventSelectStrmPort(&DevInst, TileLoc, 1,
			XAIE_STRMSW_SLAVE, FIFO, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventSelectStrmPort(&DevInst, TileLoc, 1,
			XAIE_STRMSW_MASTER, FIFO, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventSelectStrmPortReset(&DevInst, TileLoc, 2);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventBroadcast(&DevInst, TileLoc, XAIE_PL_MOD, 2,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventBroadcastReset(&DevInst, TileLoc, XAIE_PL_MOD, 2);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventBroadcastBlockDir(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventBroadcastBlockMapDir(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventBroadcastUnblockDir(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventGroupControl(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_GROUP_0_PL, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventGroupReset(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_GROUP_0_PL);
	CHECK_EQUAL(XAIE_OK, RC);

	u8 eventPointer;
	RC = XAie_EventLogicalToPhysicalConv(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_GROUP_0_PL, &eventPointer);
	CHECK_EQUAL(XAIE_OK, RC);

	XAie_Events EnumEvent;
	RC = XAie_EventPhysicalToLogicalConv(&DevInst, TileLoc, XAIE_PL_MOD,
			eventPointer, &EnumEvent);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(EventApis, EventCoreMod)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(5, 4);

	RC = XAie_EventGenerate(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_TRUE_CORE);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventComboReset(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_COMBO0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventSelectStrmPort(&DevInst, TileLoc, 1,
	XAIE_STRMSW_SLAVE, FIFO, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventSelectStrmPortReset(&DevInst, TileLoc, 2);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventBroadcast(&DevInst, TileLoc, XAIE_CORE_MOD, 2,
			XAIE_EVENT_TRUE_CORE);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventBroadcastReset(&DevInst, TileLoc, XAIE_CORE_MOD, 2);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventBroadcastBlockDir(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventBroadcastBlockMapDir(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventBroadcastUnblockDir(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventGroupControl(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventGroupReset(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_CORE);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventPCEnable(&DevInst, TileLoc, 1, 0x4000);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventPCDisable(&DevInst, TileLoc, 1);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventPCReset(&DevInst, TileLoc, 1);
	CHECK_EQUAL(XAIE_OK, RC);

	u8 eventPointer;
	RC = XAie_EventLogicalToPhysicalConv(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_CORE, &eventPointer);
	CHECK_EQUAL(XAIE_OK, RC);

	XAie_Events EnumEvent;
	RC = XAie_EventPhysicalToLogicalConv(&DevInst, TileLoc, XAIE_CORE_MOD,
			eventPointer, &EnumEvent);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(EventApis, EventMemMod)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 1);

	RC = XAie_EventComboReset(&DevInst, TileLoc, XAIE_MEM_MOD,
			XAIE_EVENT_COMBO0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventSelectStrmPortReset(&DevInst, TileLoc, 2);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventBroadcastReset(&DevInst, TileLoc, XAIE_MEM_MOD, 2);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventBroadcastReset(&DevInst, XAie_TileLoc(5, 4),
			XAIE_MEM_MOD, 2);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventBroadcastBlockDir(&DevInst, TileLoc, XAIE_MEM_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventBroadcastBlockMapDir(&DevInst, TileLoc, XAIE_MEM_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_EventBroadcastUnblockDir(&DevInst, TileLoc, XAIE_MEM_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST_GROUP(EventNegs)
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

TEST(EventNegs, InvalidDevInst)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_EventGenerate(NULL, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_TRUE_CORE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventComboConfig(NULL, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_COMBO0, XAIE_EVENT_COMBO_E1_AND_E2,
			XAIE_EVENT_COMBO_EVENT_0_CORE,
			XAIE_EVENT_COMBO_EVENT_1_CORE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventComboReset(NULL, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_COMBO0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventSelectStrmPort(NULL, TileLoc, 1,
			XAIE_STRMSW_SLAVE, FIFO, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventSelectStrmPort(NULL, TileLoc, 1,
			XAIE_STRMSW_MASTER, FIFO, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventSelectStrmPortReset(NULL, TileLoc, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcast(NULL, TileLoc, XAIE_CORE_MOD, 2,
			XAIE_EVENT_TRUE_CORE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastReset(NULL, TileLoc, XAIE_CORE_MOD, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastBlockDir(NULL, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastBlockMapDir(NULL, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastUnblockDir(NULL, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventGroupControl(NULL, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_CORE, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventGroupReset(NULL, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_CORE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventPCEnable(NULL, TileLoc, 1, 0x4000);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventPCDisable(NULL, TileLoc, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventPCReset(NULL, TileLoc, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	u8 eventPointer;
	RC = XAie_EventLogicalToPhysicalConv(NULL, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_CORE, &eventPointer);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	XAie_Events EnumEvent;
	RC = XAie_EventPhysicalToLogicalConv(NULL, TileLoc, XAIE_CORE_MOD,
			eventPointer, &EnumEvent);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);
}

TEST(EventNegs, InvalidTile)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(XAIE_NUM_COLS + 5,
			XAIE_NUM_ROWS + 5);

	RC = XAie_EventGenerate(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_TRUE_CORE);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_EventComboConfig(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_COMBO0, XAIE_EVENT_COMBO_E1_AND_E2,
			XAIE_EVENT_COMBO_EVENT_0_CORE,
			XAIE_EVENT_COMBO_EVENT_1_CORE);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_EventComboReset(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_COMBO0);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_EventSelectStrmPort(&DevInst, TileLoc, 1,
			XAIE_STRMSW_SLAVE, FIFO, 0);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_EventSelectStrmPort(&DevInst, TileLoc, 1,
			XAIE_STRMSW_MASTER, FIFO, 0);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_EventSelectStrmPortReset(&DevInst, TileLoc, 2);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_EventBroadcast(&DevInst, TileLoc, XAIE_CORE_MOD, 2,
			XAIE_EVENT_TRUE_CORE);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_EventBroadcastReset(&DevInst, TileLoc, XAIE_CORE_MOD, 2);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_EventBroadcastBlockDir(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_EventBroadcastBlockMapDir(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_EventBroadcastUnblockDir(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_EventGroupControl(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_CORE, 0);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_EventGroupReset(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_CORE);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_EventPCEnable(&DevInst, TileLoc, 1, 0x4000);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_EventPCDisable(&DevInst, TileLoc, 1);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_EventPCReset(&DevInst, TileLoc, 1);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	u8 eventPointer;
	RC = XAie_EventLogicalToPhysicalConv(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_CORE, &eventPointer);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	XAie_Events EnumEvent;
	RC = XAie_EventPhysicalToLogicalConv(&DevInst, TileLoc, XAIE_CORE_MOD,
			eventPointer, &EnumEvent);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);
}

TEST(EventNegs, InvalidModule){
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(2, 0);

	RC = XAie_EventGenerate(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_TRUE_CORE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventComboConfig(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_COMBO0, XAIE_EVENT_COMBO_E1_AND_E2,
			XAIE_EVENT_COMBO_EVENT_0_CORE,
			XAIE_EVENT_COMBO_EVENT_1_CORE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventComboReset(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_COMBO0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcast(&DevInst, TileLoc, XAIE_CORE_MOD, 2,
			XAIE_EVENT_TRUE_CORE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastReset(&DevInst, TileLoc, XAIE_CORE_MOD, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastBlockDir(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastBlockMapDir(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastUnblockDir(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventGroupControl(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_CORE, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventGroupReset(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_CORE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	u8 eventPointer;
	RC = XAie_EventLogicalToPhysicalConv(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_CORE, &eventPointer);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	XAie_Events EnumEvent;
	RC = XAie_EventPhysicalToLogicalConv(&DevInst, TileLoc, XAIE_CORE_MOD,
			eventPointer, &EnumEvent);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);
}

TEST(EventNegs, OtherErrors){
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(5, 4);

	RC = XAie_EventGenerate(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_TRUE_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventComboConfig(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_COMBO0, XAIE_EVENT_COMBO_E1_AND_E2,
			XAIE_EVENT_COMBO_EVENT_0_PL,
			XAIE_EVENT_COMBO_EVENT_1_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventSelectStrmPort(&DevInst, TileLoc, 1,
			XAIE_STRMSW_SLAVE, SS_PORT_TYPE_MAX, 0);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_EventSelectStrmPort(&DevInst, TileLoc, 10,
			XAIE_STRMSW_SLAVE, FIFO, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcast(&DevInst, TileLoc, XAIE_CORE_MOD, 100,
			XAIE_EVENT_TRUE_CORE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcast(&DevInst, TileLoc, XAIE_CORE_MOD, 1,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastBlockDir(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 100, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastBlockDir(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 1, 16);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastUnblockDir(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 1, 16);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastUnblockDir(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 100, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventGroupControl(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_TRUE_CORE, 5);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventPCEnable(&DevInst, XAie_TileLoc(2, 0), 255,
			0x4000);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	TileLoc = XAie_TileLoc(5, 4);
	RC = XAie_EventBroadcastBlockMapDir(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, -1, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastBlockMapDir(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_SWITCH_A, 1,16);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	TileLoc = XAie_TileLoc(1, 0);
	XAie_LocType CoreTileLoc = XAie_TileLoc(5, 4);
	XAie_LocType ShimNocTileLoc = XAie_TileLoc(3, 0);

	RC = XAie_EventSelectStrmPort(&DevInst, XAie_TileLoc(8, 1), 1,
			XAIE_STRMSW_SLAVE, FIFO, 0);
#if AIE_GEN >= 2
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);
#elif AIE_GEN == 1
	CHECK_EQUAL(XAIE_OK, RC);
#endif

	RC = XAie_EventGenerate(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_PERF_CNT_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventComboConfig(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_COMBO0, XAIE_EVENT_COMBO_E1_AND_E2,
			XAIE_EVENT_COMBO_EVENT_0_MEM,
			XAIE_EVENT_COMBO_EVENT_1_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventComboConfig(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_COMBO0, XAIE_EVENT_COMBO_E1_AND_E2,
			XAIE_EVENT_COMBO_EVENT_0_PL,
			XAIE_EVENT_COMBO_EVENT_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventComboConfig(&DevInst, CoreTileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_COMBO0, XAIE_EVENT_COMBO_E1_AND_E2,
			XAIE_EVENT_COMBO_EVENT_0_CORE,
			XAIE_EVENT_COMBO_EVENT_1_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventComboReset(&DevInst, TileLoc, (XAie_ModuleType)3,
			XAIE_EVENT_COMBO0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventSelectStrmPortReset(&DevInst, ShimNocTileLoc, 2);
	CHECK_EQUAL(XAIE_OK, RC);

	u8 eventPointer;
	RC = XAie_EventLogicalToPhysicalConv(&DevInst, CoreTileLoc,
			XAIE_CORE_MOD, XAIE_EVENT_NONE_PL, &eventPointer);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventLogicalToPhysicalConv(&DevInst, TileLoc,
			XAIE_PL_MOD, XAIE_EVENT_NONE_CORE, &eventPointer);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	XAie_Events EnumEvent;
	eventPointer = 180;
	RC = XAie_EventPhysicalToLogicalConv(&DevInst, CoreTileLoc,
			XAIE_CORE_MOD, eventPointer, &EnumEvent);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 0;
	RC = XAie_EventGenerate(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventComboConfig(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_COMBO0, XAIE_EVENT_COMBO_E1_AND_E2,
			XAIE_EVENT_COMBO_EVENT_0_PL,
			XAIE_EVENT_COMBO_EVENT_1_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventComboReset(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_COMBO0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventSelectStrmPort(&DevInst, TileLoc, 1,
			XAIE_STRMSW_SLAVE, FIFO, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventSelectStrmPortReset(&DevInst, TileLoc, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcast(&DevInst, TileLoc, XAIE_PL_MOD, 2,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastReset(&DevInst, TileLoc, XAIE_PL_MOD, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastBlockDir(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastBlockMapDir(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventBroadcastUnblockDir(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_SWITCH_A, 2, XAIE_EVENT_BROADCAST_WEST);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventGroupControl(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_GROUP_0_PL, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventGroupReset(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_GROUP_0_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventPCEnable(&DevInst, TileLoc, 1, 0x4000);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventPCDisable(&DevInst, TileLoc, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventPCReset(&DevInst, TileLoc, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventLogicalToPhysicalConv(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_CORE, &eventPointer);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_EventPhysicalToLogicalConv(&DevInst, TileLoc, XAIE_CORE_MOD,
			eventPointer, &EnumEvent);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 1;
}
