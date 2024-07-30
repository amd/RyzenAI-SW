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

TEST_GROUP(StreamSwitches)
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

TEST(StreamSwitches, StrmSwitchBasicAieTiles)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, DMA, 0, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, DMA, 0, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, DMA, 1, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, DMA, 1, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, CTRL, 0, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, CTRL, 0, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, FIFO, 0, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, FIFO, 0, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, SOUTH, 0, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, SOUTH, 0, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, SOUTH, 1, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, SOUTH, 1, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, SOUTH, 2, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, SOUTH, 2, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, SOUTH, 3, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, SOUTH, 3, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, WEST, 2, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, WEST, 2, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, EAST, 2, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, EAST, 2, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, EAST, 2, DMA, 1);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, EAST, 2, DMA, 1);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, NORTH, 2, FIFO, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, NORTH, 2, FIFO, 0);
	CHECK_EQUAL(XAIE_OK, RC);

#if AIE_GEN == 1
	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, TRACE, 1, EAST, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, TRACE, 0, NORTH, 2);
	CHECK_EQUAL(XAIE_OK, RC);
#endif

	u8 PhyPortId;
	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_SLAVE, SOUTH, 5, &PhyPortId);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_SLAVE, NORTH, 1, &PhyPortId);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_SLAVE, TRACE, 1, &PhyPortId);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_MASTER, DMA, 0, &PhyPortId);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_MASTER, NORTH, 1, &PhyPortId);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_MASTER, EAST, 1, &PhyPortId);
	CHECK_EQUAL(XAIE_OK, RC);

	StrmSwPortType PortType;
	u8 PortNum;
	u8 MaxPhyPortId;

#if AIE_GEN == 1
	MaxPhyPortId = 26;
#else
	MaxPhyPortId = 24;
#endif
	for(u8 i = 0; i <= MaxPhyPortId; i++) {
		RC = XAie_StrmSwPhysicalToLogicalPort(&DevInst, TileLoc,
				XAIE_STRMSW_SLAVE, i, &PortType, &PortNum);
		CHECK_EQUAL(XAIE_OK, RC);
	}

#if AIE_GEN == 1
	MaxPhyPortId = 24;
#else
	MaxPhyPortId = 22;
#endif
	for(u8 i = 0; i <= MaxPhyPortId; i++) {
		RC = XAie_StrmSwPhysicalToLogicalPort(&DevInst, TileLoc,
				XAIE_STRMSW_MASTER, i, &PortType, &PortNum);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

#if AIE_GEN >= 2
TEST(StreamSwitches, StrmSwitchBasicMemTiles)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 1);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, DMA, 0, NORTH, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, DMA, 0, NORTH, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, NORTH, 2, DMA, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, NORTH, 2, DMA, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, SOUTH, 2, DMA, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, SOUTH, 2, DMA, 0);
	CHECK_EQUAL(XAIE_OK, RC);

#if AIE_GEN == 1
	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, DMA, 2, DMA, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, DMA, 2, DMA, 0);
	CHECK_EQUAL(XAIE_OK, RC);
#endif

	StrmSwPortType PortType;
	u8 PortNum;
	u8 MaxPhyPortId = 17; /* For mem tile slave */

	for(u8 i = 0; i <= MaxPhyPortId; i++) {
		RC = XAie_StrmSwPhysicalToLogicalPort(&DevInst, TileLoc,
				XAIE_STRMSW_SLAVE, i, &PortType, &PortNum);
		CHECK_EQUAL(XAIE_OK, RC);
	}

	MaxPhyPortId = 16; /* For mem tile master */
	for(u8 i = 0; i <= MaxPhyPortId; i++) {
		RC = XAie_StrmSwPhysicalToLogicalPort(&DevInst, TileLoc,
				XAIE_STRMSW_MASTER, i, &PortType, &PortNum);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}
#endif

TEST(StreamSwitches, StrmSwitchBasicPlNocTiles)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 0);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, EAST, 0, SOUTH, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, EAST, 0, SOUTH, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, NORTH, 2, EAST, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, NORTH, 2, EAST, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, SOUTH, 2, WEST, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, SOUTH, 2, WEST, 0);
	CHECK_EQUAL(XAIE_OK, RC);

#if AIE_GEN == 1
	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, NORTH, 2, NORTH, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, NORTH, 2, NORTH, 0);
	CHECK_EQUAL(XAIE_OK, RC);
#endif

	u8 PhyPortId;
	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_SLAVE, SOUTH, 5, &PhyPortId);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_SLAVE, NORTH, 1, &PhyPortId);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_SLAVE, EAST, 3, &PhyPortId);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_MASTER, WEST, 1, &PhyPortId);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_MASTER, NORTH, 1, &PhyPortId);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_MASTER, EAST, 1, &PhyPortId);
	CHECK_EQUAL(XAIE_OK, RC);

	StrmSwPortType PortType;
	u8 PortNum;
	u8 MaxPhyPortId;

#if AIE_GEN == 1
	MaxPhyPortId = 23;
#else
	MaxPhyPortId = 22;
#endif
	for(u8 i = 0; i <= MaxPhyPortId; i++) {
		RC = XAie_StrmSwPhysicalToLogicalPort(&DevInst, TileLoc,
				XAIE_STRMSW_SLAVE, i, &PortType, &PortNum);
		CHECK_EQUAL(XAIE_OK, RC);
	}

#if AIE_GEN == 1
	MaxPhyPortId = 22;
#else
	MaxPhyPortId = 21;
#endif
	for(u8 i = 0; i <= MaxPhyPortId; i++) {
		RC = XAie_StrmSwPhysicalToLogicalPort(&DevInst, TileLoc,
				XAIE_STRMSW_SLAVE, i, &PortType, &PortNum);
		CHECK_EQUAL(XAIE_OK, RC);
	}
}

TEST(StreamSwitches, StrmSwitchErrors)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, NORTH, 10, NORTH, 0);
	CHECK_EQUAL(!RC, XAIE_OK);

	RC = XAie_StrmConnCctDisable(&DevInst, TileLoc, NORTH, 2, NORTH, 15);
	CHECK_EQUAL(!RC, XAIE_OK);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, TRACE, 2, SOUTH, 1);
	CHECK_EQUAL(!RC, XAIE_OK);
}

/* Packet switch mode test cases */
TEST_GROUP(StreamSwitchPktSwitchMode)
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

TEST(StreamSwitchPktSwitchMode, AieTileBasic)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_StrmPktSwMstrPortEnable(&DevInst, TileLoc, SOUTH, 2,
			XAIE_SS_PKT_DROP_HEADER, 2, 0x3);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmPktSwSlavePortEnable(&DevInst, TileLoc, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmPktSwSlaveSlotDisable(&DevInst, TileLoc, CORE, 0, 3);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(StreamSwitchPktSwitchMode, AieTileDisables)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_StrmPktSwMstrPortDisable(&DevInst, TileLoc, SOUTH, 2);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmPktSwSlavePortDisable(&DevInst, TileLoc, CORE, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmPktSwSlaveSlotEnable(&DevInst, TileLoc, CORE, 0, 3,
			XAie_PacketInit(5, 10), 0x5, 0x3, 2);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(StreamSwitchPktSwitchMode, AieShimTileBasic)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8,0);

	RC = XAie_StrmPktSwMstrPortEnable(&DevInst, TileLoc, SOUTH, 2,
			XAIE_SS_PKT_DROP_HEADER, 2, 0x3);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmPktSwSlavePortEnable(&DevInst, TileLoc, EAST, 1);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmPktSwSlaveSlotEnable(&DevInst, TileLoc, EAST, 1, 3,
			XAie_PacketInit(5, 10), 0x5, 0x3, 2);
	CHECK_EQUAL(XAIE_OK, RC);

	u8 portValue;
	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_SLAVE, EAST, 2, &portValue);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_MASTER, EAST, 2, &portValue);
	CHECK_EQUAL(XAIE_OK, RC);

	StrmSwPortType portType = EAST;
	RC = XAie_StrmSwPhysicalToLogicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_SLAVE, 2, &portType, &portValue);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmSwPhysicalToLogicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_MASTER, 2, &portType, &portValue);
	CHECK_EQUAL(XAIE_OK, RC);
}

#if AIE_GEN >= 2
TEST(StreamSwitchPktSwitchMode, AieMemTileBasic)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 1);

	RC = XAie_StrmPktSwMstrPortEnable(&DevInst, TileLoc, SOUTH, 2,
			XAIE_SS_PKT_DROP_HEADER, 2, 0x3);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmPktSwSlavePortEnable(&DevInst, TileLoc, NORTH, 0);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmPktSwSlaveSlotEnable(&DevInst, TileLoc, NORTH, 0, 3,
			XAie_PacketInit(5, 10), 0x5, 0x3, 2);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(StreamSwitchPktSwitchMode, AieTileDeterministicMerge)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	/* iterate over position for arbitor 0 */
	for(u8 i = 0; i < 4; i++) {
		RC = XAie_StrmSwDeterministicMergeConfig(&DevInst, TileLoc, 0U,
				NORTH, 1, 5U, i);
		CHECK_EQUAL(XAIE_OK, RC);
	}

	/* iterate over position for arbitor 1U */
	for(u8 i = 0; i < 4; i++) {
		RC = XAie_StrmSwDeterministicMergeConfig(&DevInst, TileLoc, 1U,
				EAST, 2, 5U, i);
		CHECK_EQUAL(XAIE_OK, RC);
	}

	/* iterate over position for invalid arbitor */
	for(u8 i = 0; i < 4; i++) {
		RC = XAie_StrmSwDeterministicMergeConfig(&DevInst, TileLoc, 5U,
				NORTH, 1, 5U, i);
		CHECK_EQUAL(XAIE_INVALID_ARGS, RC);
	}

	RC = XAie_StrmSwDeterministicMergeEnable(&DevInst, TileLoc, 0U);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmSwDeterministicMergeEnable(&DevInst, TileLoc, 1U);
	CHECK_EQUAL(XAIE_OK, RC);

}

TEST(StreamSwitchPktSwitchMode, MemTileDeterministicMerge)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	TileLoc = XAie_TileLoc(8, 1);
	/* iterate over position for arbitor 0 */
	for(u8 i = 0; i < 4; i++) {
		RC = XAie_StrmSwDeterministicMergeConfig(&DevInst, TileLoc, 0U,
				NORTH, 1, 5U, i);
		CHECK_EQUAL(XAIE_OK, RC);
	}

	/* iterate over position for arbitor 1U */
	for(u8 i = 0; i < 4; i++) {
		RC = XAie_StrmSwDeterministicMergeConfig(&DevInst, TileLoc, 1U,
				SOUTH, 2, 5U, i);
		CHECK_EQUAL(XAIE_OK, RC);
	}

	/* iterate over position for invalid arbitor */
	for(u8 i = 0; i < 4; i++) {
		RC = XAie_StrmSwDeterministicMergeConfig(&DevInst, TileLoc, 5U,
				NORTH, 1, 5U, i);
		CHECK_EQUAL(XAIE_INVALID_ARGS, RC);
	}

	RC = XAie_StrmSwDeterministicMergeEnable(&DevInst, TileLoc, 0U);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmSwDeterministicMergeEnable(&DevInst, TileLoc, 1U);
	CHECK_EQUAL(XAIE_OK, RC);
}

TEST(StreamSwitchPktSwitchMode, ShimTileDeterministicMerge)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	TileLoc = XAie_TileLoc(8, 0);
	/* iterate over position for arbitor 0 */
	for(u8 i = 0; i < 4; i++) {
		RC = XAie_StrmSwDeterministicMergeConfig(&DevInst, TileLoc, 0U,
				NORTH, 1, 5U, i);
		CHECK_EQUAL(XAIE_OK, RC);
	}

	/* iterate over position for arbitor 1U */
	for(u8 i = 0; i < 4; i++) {
		RC = XAie_StrmSwDeterministicMergeConfig(&DevInst, TileLoc, 1U,
				EAST, 2, 5U, i);
		CHECK_EQUAL(XAIE_OK, RC);
	}

	/* iterate over position for invalid arbitor */
	for(u8 i = 0; i < 4; i++) {
		RC = XAie_StrmSwDeterministicMergeConfig(&DevInst, TileLoc, 5U,
				NORTH, 1, 5U, i);
		CHECK_EQUAL(XAIE_INVALID_ARGS, RC);
	}

	RC = XAie_StrmSwDeterministicMergeEnable(&DevInst, TileLoc, 0U);
	CHECK_EQUAL(XAIE_OK, RC);

	RC = XAie_StrmSwDeterministicMergeEnable(&DevInst, TileLoc, 1U);
	CHECK_EQUAL(XAIE_OK, RC);
}

#endif

TEST_GROUP(StreamSwitchNegs)
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

TEST(StreamSwitchNegs, InvalidArgs)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(8, 3);

	RC = XAie_StrmPktSwMstrPortEnable(NULL, TileLoc, SOUTH, 2,
			XAIE_SS_PKT_DROP_HEADER, 2, 0x3);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmPktSwMstrPortEnable(&DevInst, TileLoc,
			(StrmSwPortType)11, 2,
			XAIE_SS_PKT_DROP_HEADER, 2, 0x3);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_StrmPktSwMstrPortEnable(&DevInst, TileLoc, SOUTH, 2,
			XAIE_SS_PKT_DROP_HEADER, 99, 0x3);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmPktSwMstrPortEnable(&DevInst, TileLoc, SOUTH, 15,
			XAIE_SS_PKT_DROP_HEADER, 2, 0x3);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_StrmPktSwMstrPortEnable(&DevInst, TileLoc, SOUTH, 2,
			(XAie_StrmSwPktHeader)2, 2, 0x3);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, EAST, 15, DMA, 1);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, EAST, 1, DMA, 19);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, (StrmSwPortType)11, 1,
			DMA, 19);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_StrmConnCctEnable(NULL, TileLoc, EAST, 1, DMA, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmPktSwSlavePortEnable(&DevInst, TileLoc, EAST, 12);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_StrmPktSwSlavePortEnable(&DevInst, TileLoc,
			(StrmSwPortType)11, 12);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_StrmPktSwSlavePortEnable(NULL, TileLoc, EAST, 1);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmPktSwSlaveSlotEnable(NULL, TileLoc, CORE, 0, 3,
			XAie_PacketInit(5, 10), 0x5, 0x3, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmPktSwSlaveSlotEnable(&DevInst, TileLoc,
			(StrmSwPortType)11, 0, 3, XAie_PacketInit(5, 10), 0x5,
			0x3, 2);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_StrmPktSwSlaveSlotEnable(&DevInst, TileLoc, CORE, 0, 3,
			XAie_PacketInit(5, 10), 0x5, 0x3, 99);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	u8 portValue;
	RC = XAie_StrmSwLogicalToPhysicalPort(NULL, TileLoc,
			XAIE_STRMSW_MASTER, EAST, 2, &portValue);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_MASTER, EAST, 2, NULL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_MASTER, SS_PORT_TYPE_MAX, 2, &portValue);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			(XAie_StrmPortIntf)3, EAST, 2, &portValue);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	StrmSwPortType portType = EAST;
	RC = XAie_StrmSwPhysicalToLogicalPort(NULL, TileLoc,
			XAIE_STRMSW_SLAVE, 2, &portType, &portValue);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmSwPhysicalToLogicalPort(&DevInst, TileLoc,
			(XAie_StrmPortIntf)3, 2, &portType, &portValue);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_StrmSwPhysicalToLogicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_SLAVE, 27, &portType, &portValue);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);


	RC = XAie_StrmSwPhysicalToLogicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_SLAVE, 2, NULL, &portValue);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmSwPhysicalToLogicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_SLAVE, 2, &portType, NULL);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	TileLoc = XAie_TileLoc(8, 3);
	/*Below is for branch coverage*/
	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, DMA, 0,
			(StrmSwPortType)10, 0);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, TRACE, 1, TRACE, 0);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_StrmPktSwMstrPortEnable(&DevInst, TileLoc, SOUTH, 2,
			XAIE_SS_PKT_DROP_HEADER, 1, 17);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmPktSwSlaveSlotEnable(&DevInst, TileLoc, CORE, 0, 3,
			XAie_PacketInit(5, 10), 0x9, 0x11, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmPktSwSlaveSlotEnable(&DevInst, TileLoc, CORE, 0, 3,
			XAie_PacketInit(5, 10), 0xFF, 1, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmPktSwSlaveSlotEnable(&DevInst, TileLoc, CORE, 0, 3,
			XAie_PacketInit(32, 8), 0, 1, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmPktSwSlaveSlotEnable(&DevInst, TileLoc, CORE, 0, 5,
			XAie_PacketInit(5, 10), 0, 1, 2);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	RC = XAie_StrmPktSwSlaveSlotEnable(&DevInst, TileLoc, CORE, 16, 3,
			XAie_PacketInit(5, 10), 0, 1, 2);
	CHECK_EQUAL(XAIE_ERR_STREAM_PORT, RC);

	/*Invalid Dev Inst*/
	DevInst.IsReady = 0;

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, DMA, 0, CORE, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmPktSwSlavePortEnable(&DevInst, TileLoc, CORE, 0);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmPktSwMstrPortEnable(&DevInst, TileLoc, SOUTH, 2,
			XAIE_SS_PKT_DROP_HEADER, 2, 0x3);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmPktSwSlaveSlotEnable(&DevInst, TileLoc, CORE, 0, 3,
			XAie_PacketInit(5, 10), 0x5, 0x3, 2);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_MASTER, EAST, 2, &portValue);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	RC = XAie_StrmSwPhysicalToLogicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_SLAVE, 2, &portType, &portValue);
	CHECK_EQUAL(XAIE_INVALID_ARGS, RC);

	DevInst.IsReady = 1;
}

TEST(StreamSwitchNegs, InvalidTile)
{
	AieRC RC;

	XAie_LocType TileLoc = XAie_TileLoc(XAIE_NUM_COLS + 1,
			XAIE_NUM_ROWS +1);

	RC = XAie_StrmConnCctEnable(&DevInst, TileLoc, DMA, 0, CORE, 0);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_StrmPktSwSlavePortEnable(&DevInst, TileLoc, CORE, 0);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_StrmPktSwMstrPortEnable(&DevInst, TileLoc, SOUTH, 2,
			XAIE_SS_PKT_DROP_HEADER, 2, 0x3);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	RC = XAie_StrmPktSwSlaveSlotEnable(&DevInst, TileLoc, CORE, 0, 3,
			XAie_PacketInit(5, 10), 0x5, 0x3, 2);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	u8 portValue;
	RC = XAie_StrmSwLogicalToPhysicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_MASTER, EAST, 2, &portValue);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);

	StrmSwPortType portType = EAST;
	RC = XAie_StrmSwPhysicalToLogicalPort(&DevInst, TileLoc,
			XAIE_STRMSW_SLAVE, 2, &portType, &portValue);
	CHECK_EQUAL(XAIE_INVALID_TILE, RC);
}
