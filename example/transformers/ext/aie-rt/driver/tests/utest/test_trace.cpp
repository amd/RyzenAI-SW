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

TEST_GROUP(TraceConfigApis)
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

TEST(TraceConfigApis, TraceControlConfig)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_TraceControlConfig(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_STREAM_STALL_CORE, XAIE_EVENT_DISABLED_CORE,
			XAIE_TRACE_EVENT_TIME);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_TraceControlConfigReset(&DevInst, TileLoc, XAIE_CORE_MOD);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_TracePktConfigReset(&DevInst, TileLoc, XAIE_CORE_MOD);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_TraceEventReset(&DevInst, TileLoc, XAIE_CORE_MOD, 5);
	CHECK_EQUAL(XAIE_OK, RC);

	XAie_Events Events[5] = { XAIE_EVENT_STREAM_STALL_CORE,
		XAIE_EVENT_DISABLED_CORE,
		XAIE_EVENT_INSTR_LOAD_CORE,
		XAIE_EVENT_INSTR_STORE_CORE,
		XAIE_EVENT_INSTR_STREAM_GET_CORE };
	u8 SlotIds[5] = { 2, 5, 6, 1, 0 };

	RC = XAie_TraceEventList(&DevInst, TileLoc, XAIE_CORE_MOD, Events,
			SlotIds, 5);
	CHECK_EQUAL(XAIE_OK, RC);

#if AIE_GEN == 1
	RC = XAie_TraceControlConfig(&DevInst, XAie_TileLoc(2, 0), XAIE_PL_MOD,
			XAIE_EVENT_DMA_S2MM_1_START_BD_PL,
			XAIE_EVENT_DMA_S2MM_0_GO_TO_IDLE_PL,
			XAIE_TRACE_EVENT_TIME);
#elif AIE_GEN >= 2
	RC = XAie_TraceControlConfig(&DevInst, XAie_TileLoc(2, 0), XAIE_PL_MOD,
			XAIE_EVENT_DMA_S2MM_1_START_TASK_PL,
			XAIE_EVENT_DMA_S2MM_0_FINISHED_TASK_PL,
			XAIE_TRACE_EVENT_TIME);
#endif
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_TraceStartEvent(&DevInst, TileLoc,
			XAIE_CORE_MOD, XAIE_EVENT_TRUE_CORE);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_TraceStopEvent(&DevInst, TileLoc,
			XAIE_CORE_MOD, XAIE_EVENT_TRUE_CORE);
	CHECK_EQUAL(XAIE_OK, RC);

	XAie_Packet Pkt = XAie_PacketInit(1, 1);
	RC = XAie_TracePktConfig(&DevInst, TileLoc,
			XAIE_CORE_MOD, Pkt);
	CHECK_EQUAL(XAIE_OK, RC);

	XAie_TraceMode Mode = XAIE_TRACE_EVENT_TIME;
	RC = XAie_TraceModeConfig(&DevInst, TileLoc,
			XAIE_CORE_MOD, Mode);
	CHECK_EQUAL(XAIE_OK, RC);

	XAie_TraceState Status;
	RC = XAie_TraceGetState(&DevInst, TileLoc,
			XAIE_CORE_MOD, &Status);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_TraceGetMode(&DevInst, TileLoc,
			XAIE_CORE_MOD, &Mode);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(TraceConfigApis, TraceControlConfigPl)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(1, 0);

	RC = XAie_TraceControlConfig(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_GROUP_0_PL, XAIE_EVENT_COMBO_EVENT_0_PL,
			XAIE_TRACE_EVENT_TIME);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_TraceControlConfigReset(&DevInst, TileLoc, XAIE_PL_MOD);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_TracePktConfigReset(&DevInst, TileLoc, XAIE_PL_MOD);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_TraceEventReset(&DevInst, TileLoc, XAIE_PL_MOD, 5);
	CHECK_EQUAL(XAIE_OK, RC);

	XAie_Events Events[5] = { XAIE_EVENT_PORT_IDLE_0_PL,
			XAIE_EVENT_PORT_RUNNING_0_PL,
			XAIE_EVENT_PORT_STALLED_0_PL,
			XAIE_EVENT_PORT_TLAST_0_PL,
			XAIE_EVENT_GROUP_ERRORS_PL };
	u8 SlotIds[5] = { 2, 5, 6, 1, 0 };

	RC = XAie_TraceEventList(&DevInst, TileLoc, XAIE_PL_MOD, Events,
			SlotIds, 5);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_TraceControlConfig(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TIMER_SYNC_PL,
			XAIE_EVENT_TIMER_VALUE_REACHED_PL,
			XAIE_TRACE_EVENT_TIME);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_TraceStartEvent(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_TraceStopEvent(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_OK, RC);

	XAie_Packet Pkt = XAie_PacketInit(1,1);
	RC = XAie_TracePktConfig(&DevInst, TileLoc, XAIE_PL_MOD, Pkt);
	CHECK_EQUAL(XAIE_OK, RC);

	XAie_TraceMode Mode = XAIE_TRACE_EVENT_TIME;

	XAie_TraceState Status;
	RC = XAie_TraceGetState(&DevInst, TileLoc, XAIE_PL_MOD, &Status);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_TraceGetMode(&DevInst, TileLoc, XAIE_PL_MOD, &Mode);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST_GROUP(TraceConfigApisNeg)
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

TEST(TraceConfigApisNeg, InvalidArgs)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_TraceControlConfig(NULL, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_COMBO_EVENT_0_PL, XAIE_EVENT_GROUP_0_PL,
			XAIE_TRACE_EVENT_TIME);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceEvent(NULL, TileLoc, XAIE_PL_MOD, XAIE_EVENT_TRUE_PL, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS,RC);

	RC = XAie_TraceStartEvent(NULL, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS,RC);

	RC = XAie_TraceStopEvent(NULL, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	XAie_Packet Pkt = XAie_PacketInit(1, 1);
	RC = XAie_TracePktConfig(NULL, TileLoc, XAIE_PL_MOD, Pkt);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	XAie_TraceMode Mode = XAIE_TRACE_EVENT_TIME;
	RC = XAie_TraceModeConfig(NULL, TileLoc, XAIE_PL_MOD, Mode);
	CHECK_EQUAL(XAIE_INVALID_ARGS,RC);

	XAie_TraceState Status;
	RC = XAie_TraceGetState(NULL, TileLoc, XAIE_PL_MOD, &Status);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceGetState(&DevInst, TileLoc, XAIE_PL_MOD, NULL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceGetMode(NULL, TileLoc, XAIE_PL_MOD, &Mode);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceGetMode(&DevInst, TileLoc, XAIE_PL_MOD, NULL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceControlConfigReset(NULL, TileLoc, XAIE_PL_MOD);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TracePktConfigReset(NULL, TileLoc, XAIE_PL_MOD);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	XAie_Events Events[5] = { XAIE_EVENT_STREAM_STALL_CORE,
		XAIE_EVENT_DISABLED_CORE,
		XAIE_EVENT_INSTR_LOAD_CORE,
		XAIE_EVENT_INSTR_STORE_CORE,
		XAIE_EVENT_INSTR_STREAM_GET_CORE };
	u8 SlotIds[5] = { 2, 5, 6, 1, 0 };

	RC = XAie_TraceEventList(NULL, TileLoc, XAIE_CORE_MOD, Events,
			SlotIds, 5);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceEventList(&DevInst, TileLoc, XAIE_CORE_MOD, Events,
			NULL, 5);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceEventReset(NULL, TileLoc, XAIE_PL_MOD, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceControlConfig(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_COMBO_EVENT_0_PL, XAIE_EVENT_GROUP_0_PL,
			XAIE_TRACE_EVENT_TIME);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceEvent(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_PL, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceStartEvent(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceStopEvent(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TracePktConfig(&DevInst, TileLoc,
			XAIE_PL_MOD, Pkt);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceModeConfig(&DevInst, TileLoc, XAIE_PL_MOD, Mode);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceGetState(&DevInst, TileLoc, XAIE_PL_MOD, &Status);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceGetMode(&DevInst, TileLoc, XAIE_PL_MOD, &Mode);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceControlConfigReset(&DevInst, TileLoc, XAIE_PL_MOD);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TracePktConfigReset(&DevInst, TileLoc, XAIE_PL_MOD);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceEventReset(&DevInst, TileLoc, XAIE_PL_MOD, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	TileLoc = XAie_TileLoc(1, 0);
	RC = XAie_TraceEvent(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_GROUP_LOCK_MEM, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceEvent(&DevInst, XAie_TileLoc(5, 4), XAIE_CORE_MOD,
			XAIE_EVENT_NONE_PL, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceEvent(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_PL, 9);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceStartEvent(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_GROUP_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceStopEvent(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_GROUP_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceStartEvent(&DevInst, XAie_TileLoc(5, 4), XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceStopEvent(&DevInst, XAie_TileLoc(5, 4), XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	Pkt = XAie_PacketInit(32, 8);
	RC = XAie_TracePktConfig(&DevInst, TileLoc, XAIE_PL_MOD, Pkt);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	Pkt = XAie_PacketInit(3, 8);
	RC = XAie_TracePktConfig(&DevInst, TileLoc, XAIE_PL_MOD, Pkt);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceModeConfig(&DevInst, TileLoc, XAIE_PL_MOD, Mode);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceModeConfig(&DevInst, TileLoc, XAIE_PL_MOD,
			(XAie_TraceMode)3);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceEventList(&DevInst, TileLoc, XAIE_PL_MOD, NULL, NULL,
			0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceControlConfig(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_CORE, XAIE_EVENT_TIMER_SYNC_CORE,
			Mode);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceControlConfig(&DevInst, XAie_TileLoc(5, 4), XAIE_CORE_MOD,
			XAIE_EVENT_COMBO_EVENT_0_PL, XAIE_EVENT_GROUP_0_PL,
			Mode);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceControlConfig(&DevInst, XAie_TileLoc(8, 3), XAIE_MEM_MOD,
			XAIE_EVENT_TRUE_CORE, XAIE_EVENT_GROUP_0_PL,
			XAIE_TRACE_EVENT_TIME);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceControlConfig(&DevInst, XAie_TileLoc(8, 3), XAIE_MEM_MOD,
			XAIE_EVENT_TRUE_PL, XAIE_EVENT_GROUP_0_CORE,
			XAIE_TRACE_EVENT_TIME);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceControlConfig(&DevInst, XAie_TileLoc(8, 3), XAIE_MEM_MOD,
			XAIE_EVENT_GROUP_0_CORE, XAIE_EVENT_TRUE_MEM,
			XAIE_TRACE_EVENT_TIME);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceControlConfig(&DevInst, XAie_TileLoc(8, 3), XAIE_MEM_MOD,
			XAIE_EVENT_TRUE_PL, XAIE_EVENT_TRUE_MEM,
			XAIE_TRACE_EVENT_TIME);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceEvent(&DevInst,TileLoc,XAIE_PL_MOD,
			XAIE_EVENT_TRUE_CORE, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	TileLoc = XAie_TileLoc(5, 4);

	RC = XAie_TraceEvent(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_MEM, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceStartEvent(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceStopEvent(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_MEM);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceControlConfig(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_GROUP_0_MEM, XAIE_EVENT_DISABLED_CORE,
			XAIE_TRACE_EVENT_TIME);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceControlConfig(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_DISABLED_CORE, XAIE_EVENT_GROUP_0_MEM,
			XAIE_TRACE_EVENT_TIME);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	/*Dev Inst is not ready*/
	DevInst.IsReady = 0;
	RC = XAie_TraceControlConfig(&DevInst, TileLoc, XAIE_CORE_MOD,
			XAIE_EVENT_STREAM_STALL_CORE, XAIE_EVENT_DISABLED_CORE,
			XAIE_TRACE_EVENT_TIME);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceControlConfigReset(&DevInst, TileLoc, XAIE_CORE_MOD);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TracePktConfigReset(&DevInst, TileLoc, XAIE_CORE_MOD);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceEventReset(&DevInst, TileLoc, XAIE_CORE_MOD, 5);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceEventList(&DevInst, TileLoc, XAIE_CORE_MOD, Events,
			SlotIds, 5);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

#if AIE_GEN == 1
	RC = XAie_TraceControlConfig(&DevInst, XAie_TileLoc(2, 0), XAIE_PL_MOD,
			XAIE_EVENT_DMA_S2MM_1_START_BD_PL,
			XAIE_EVENT_DMA_S2MM_0_GO_TO_IDLE_PL,
			XAIE_TRACE_EVENT_TIME);
#elif AIE_GEN >= 2
	RC = XAie_TraceControlConfig(&DevInst, XAie_TileLoc(2, 0), XAIE_PL_MOD,
			XAIE_EVENT_DMA_S2MM_1_START_TASK_PL,
			XAIE_EVENT_DMA_S2MM_0_FINISHED_TASK_PL,
			XAIE_TRACE_EVENT_TIME);
#endif
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceStartEvent(&DevInst, TileLoc,
			XAIE_CORE_MOD, XAIE_EVENT_TRUE_CORE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceStopEvent(&DevInst, TileLoc,
			XAIE_CORE_MOD, XAIE_EVENT_TRUE_CORE);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	Pkt = XAie_PacketInit(1, 1);
	RC = XAie_TracePktConfig(&DevInst, TileLoc,
			XAIE_CORE_MOD, Pkt);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	Mode = XAIE_TRACE_EVENT_TIME;
	RC = XAie_TraceModeConfig(&DevInst, TileLoc,
			XAIE_CORE_MOD, Mode);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceGetState(&DevInst, TileLoc, XAIE_CORE_MOD, &Status);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_TraceGetMode(&DevInst, TileLoc, XAIE_CORE_MOD, &Mode);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 1;
}

TEST(TraceConfigApisNeg, InvalidTile)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(XAIE_NUM_COLS + 5,
			XAIE_NUM_ROWS + 5);

	RC = XAie_TraceControlConfig(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_COMBO_EVENT_0_PL, XAIE_EVENT_GROUP_0_PL,
			XAIE_TRACE_EVENT_TIME);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_TraceEvent(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_PL, 1);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_TraceStartEvent(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_TraceStopEvent(&DevInst, TileLoc, XAIE_PL_MOD,
			XAIE_EVENT_TRUE_PL);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	XAie_Packet Pkt = XAie_PacketInit(1, 1);
	RC = XAie_TracePktConfig(&DevInst, TileLoc, XAIE_PL_MOD, Pkt);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	XAie_TraceMode Mode = XAIE_TRACE_EVENT_TIME;
	RC = XAie_TraceModeConfig(&DevInst, TileLoc, XAIE_PL_MOD, Mode);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	XAie_TraceState Status;
	RC = XAie_TraceGetState(&DevInst, TileLoc, XAIE_PL_MOD, &Status);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_TraceGetMode(&DevInst, TileLoc, XAIE_PL_MOD, &Mode);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_TraceControlConfigReset(&DevInst, TileLoc, XAIE_PL_MOD);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_TracePktConfigReset(&DevInst, TileLoc, XAIE_PL_MOD);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_TraceEventReset(&DevInst, TileLoc, XAIE_PL_MOD, 1);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);
}
