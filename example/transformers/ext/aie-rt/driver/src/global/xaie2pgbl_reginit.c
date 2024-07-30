/******************************************************************************
* Copyright (C) 2021 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/


/*****************************************************************************/
/**
* @file xaie2pgbl_reginit.c
* @{
*
* This file contains the instances of the register bit field definitions for the
* Core, Memory, NoC and PL module registers.
*
* <pre>
* MODIFICATION HISTORY:
*
* Ver   Who     Date     Changes
* ----- ------  -------- -----------------------------------------------------
* 1.0   Tejus   09/07/2021  Initial creation
* </pre>
*
******************************************************************************/

/***************************** Include Files *********************************/
#include "xaie_core_aieml.h"
#include "xaie_device_aie2p.h"
#include "xaie_device_aieml.h"
#include "xaie_dma_aieml.h"
#include "xaie_dma_aie2p.h"
#include "xaie_events_aie2p.h"
#include "xaie_interrupt_aieml.h"
#include "xaie_locks_aieml.h"
#include "xaie_reset_aieml.h"
#include "xaie_ss_aieml.h"
#include "xaie2pgbl_params.h"
#include "xaiegbl_regdef.h"

/************************** Constant Definitions *****************************/

/**************************** Type Definitions *******************************/

/**************************** Macro Definitions ******************************/
#ifdef _MSC_VER
#define XAIE2P_TILES_BITMAPSIZE	1U
#else
#define XAIE2P_TILES_BITMAPSIZE	0U
#endif

/************************** Variable Definitions *****************************/
/* bitmaps to capture modules being used by the application */
static u32 Aie2PTilesInUse[XAIE2P_TILES_BITMAPSIZE];
static u32 Aie2PMemInUse[XAIE2P_TILES_BITMAPSIZE];
static u32 Aie2PCoreInUse[XAIE2P_TILES_BITMAPSIZE];

#ifdef XAIE_FEATURE_CORE_ENABLE
/*
 * Global instance for Core module Core_Control register.
 */
static const  XAie_RegCoreCtrl Aie2PCoreCtrlReg =
{
	XAIE2PGBL_CORE_MODULE_CORE_CONTROL,
	{XAIE2PGBL_CORE_MODULE_CORE_CONTROL_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_CORE_CONTROL_ENABLE_MASK},
	{XAIE2PGBL_CORE_MODULE_CORE_CONTROL_RESET_LSB, XAIE2PGBL_CORE_MODULE_CORE_CONTROL_RESET_MASK}
};

/*
 * Global instance for Core module Core_Status register.
 */
static const  XAie_RegCoreSts Aie2PCoreStsReg =
{
	.RegOff = XAIE2PGBL_CORE_MODULE_CORE_STATUS,
	.Mask = XAIE2PGBL_CORE_MODULE_CORE_STATUS_CORE_PROCESSOR_BUS_STALL_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_CORE_DONE_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_ERROR_HALT_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_ECC_SCRUBBING_STALL_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_ECC_ERROR_STALL_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_DEBUG_HALT_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_CASCADE_STALL_MCD_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_CASCADE_STALL_SCD_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_STREAM_STALL_MS0_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_STREAM_STALL_SS0_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_STREAM_STALL_SS0_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_LOCK_STALL_E_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_LOCK_STALL_N_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_LOCK_STALL_W_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_LOCK_STALL_S_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_MEMORY_STALL_E_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_MEMORY_STALL_N_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_MEMORY_STALL_W_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_MEMORY_STALL_S_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_RESET_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_RESET_MASK |
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_ENABLE_MASK,
	.Done = {XAIE2PGBL_CORE_MODULE_CORE_STATUS_CORE_DONE_LSB,
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_CORE_DONE_MASK},
	.Rst = {XAIE2PGBL_CORE_MODULE_CORE_STATUS_RESET_LSB,
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_RESET_MASK},
	.En = {XAIE2PGBL_CORE_MODULE_CORE_STATUS_ENABLE_LSB,
		XAIE2PGBL_CORE_MODULE_CORE_STATUS_ENABLE_MASK}
};

/*
 * Global instance for Core module for core debug registers.
 */
static const XAie_RegCoreDebug Aie2PCoreDebugReg =
{
	.RegOff = XAIE2PGBL_CORE_MODULE_DEBUG_CONTROL0,
	.DebugCtrl1Offset = XAIE2PGBL_CORE_MODULE_DEBUG_CONTROL1,
	.DebugHaltCoreEvent1.Lsb = XAIE2PGBL_CORE_MODULE_DEBUG_CONTROL1_DEBUG_HALT_CORE_EVENT1_LSB,
	.DebugHaltCoreEvent1.Mask = XAIE2PGBL_CORE_MODULE_DEBUG_CONTROL1_DEBUG_HALT_CORE_EVENT1_MASK,
	.DebugHaltCoreEvent0.Lsb = XAIE2PGBL_CORE_MODULE_DEBUG_CONTROL1_DEBUG_HALT_CORE_EVENT0_LSB,
	.DebugHaltCoreEvent0.Mask = XAIE2PGBL_CORE_MODULE_DEBUG_CONTROL1_DEBUG_HALT_CORE_EVENT0_MASK,
	.DebugSStepCoreEvent.Lsb = XAIE2PGBL_CORE_MODULE_DEBUG_CONTROL1_DEBUG_SINGLESTEP_CORE_EVENT_LSB,
	.DebugSStepCoreEvent.Mask = XAIE2PGBL_CORE_MODULE_DEBUG_CONTROL1_DEBUG_SINGLESTEP_CORE_EVENT_MASK,
	.DebugResumeCoreEvent.Lsb = XAIE2PGBL_CORE_MODULE_DEBUG_CONTROL1_DEBUG_RESUME_CORE_EVENT_LSB,
	.DebugResumeCoreEvent.Mask = XAIE2PGBL_CORE_MODULE_DEBUG_CONTROL1_DEBUG_RESUME_CORE_EVENT_MASK,
	.DebugHalt.Lsb = XAIE2PGBL_CORE_MODULE_DEBUG_CONTROL0_DEBUG_HALT_BIT_LSB,
	.DebugHalt.Mask = XAIE2PGBL_CORE_MODULE_DEBUG_CONTROL0_DEBUG_HALT_BIT_MASK
};

static const XAie_RegCoreDebugStatus Aie2PCoreDebugStatus =
{
	.RegOff = XAIE2PGBL_CORE_MODULE_DEBUG_STATUS,
	.DbgEvent1Halt.Lsb = XAIE2PGBL_CORE_MODULE_DEBUG_STATUS_DEBUG_EVENT1_HALTED_LSB,
	.DbgEvent1Halt.Mask = XAIE2PGBL_CORE_MODULE_DEBUG_STATUS_DEBUG_EVENT1_HALTED_MASK,
	.DbgEvent0Halt.Lsb = XAIE2PGBL_CORE_MODULE_DEBUG_STATUS_DEBUG_EVENT0_HALTED_LSB,
	.DbgEvent0Halt.Mask = XAIE2PGBL_CORE_MODULE_DEBUG_STATUS_DEBUG_EVENT0_HALTED_MASK,
	.DbgStrmStallHalt.Lsb = XAIE2PGBL_CORE_MODULE_DEBUG_STATUS_STREAM_STALL_HALTED_LSB,
	.DbgStrmStallHalt.Mask = XAIE2PGBL_CORE_MODULE_DEBUG_STATUS_STREAM_STALL_HALTED_MASK,
	.DbgLockStallHalt.Lsb = XAIE2PGBL_CORE_MODULE_DEBUG_STATUS_LOCK_STALL_HALTED_LSB,
	.DbgLockStallHalt.Mask = XAIE2PGBL_CORE_MODULE_DEBUG_STATUS_LOCK_STALL_HALTED_MASK,
	.DbgMemStallHalt.Lsb = XAIE2PGBL_CORE_MODULE_DEBUG_STATUS_MEMORY_STALL_HALTED_LSB,
	.DbgMemStallHalt.Mask = XAIE2PGBL_CORE_MODULE_DEBUG_STATUS_MEMORY_STALL_HALTED_MASK,
	.DbgPCEventHalt.Lsb = XAIE2PGBL_CORE_MODULE_DEBUG_STATUS_PC_EVENT_HALTED_LSB,
	.DbgPCEventHalt.Mask = XAIE2PGBL_CORE_MODULE_DEBUG_STATUS_PC_EVENT_HALTED_MASK,
	.DbgHalt.Lsb = XAIE2PGBL_CORE_MODULE_DEBUG_STATUS_DEBUG_HALTED_LSB,
	.DbgHalt.Mask = XAIE2PGBL_CORE_MODULE_DEBUG_STATUS_DEBUG_HALTED_MASK,
};

/*
 * Global instance for core event registers in the core module.
 */
static const XAie_RegCoreEvents Aie2PCoreEventReg =
{
	.EnableEventOff = XAIE2PGBL_CORE_MODULE_ENABLE_EVENTS,
	.DisableEventOccurred.Lsb = XAIE2PGBL_CORE_MODULE_ENABLE_EVENTS_DISABLE_EVENT_OCCURRED_LSB,
	.DisableEventOccurred.Mask = XAIE2PGBL_CORE_MODULE_ENABLE_EVENTS_DISABLE_EVENT_OCCURRED_MASK,
	.DisableEvent.Lsb = XAIE2PGBL_CORE_MODULE_ENABLE_EVENTS_DISABLE_EVENT_LSB,
	.DisableEvent.Mask = XAIE2PGBL_CORE_MODULE_ENABLE_EVENTS_DISABLE_EVENT_MASK,
	.EnableEvent.Lsb = XAIE2PGBL_CORE_MODULE_ENABLE_EVENTS_ENABLE_EVENT_LSB,
	.EnableEvent.Mask = XAIE2PGBL_CORE_MODULE_ENABLE_EVENTS_ENABLE_EVENT_MASK,
};

/*
 * Global instance for core accumulator control register.
 */
static const XAie_RegCoreAccumCtrl Aie2PCoreAccumCtrlReg =
{
	.RegOff = XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL,
	.CascadeInput.Lsb = XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL_INPUT_LSB,
	.CascadeInput.Mask = XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL_INPUT_MASK,
	.CascadeOutput.Lsb = XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL_OUTPUT_LSB,
	.CascadeOutput.Mask = XAIE2PGBL_CORE_MODULE_ACCUMULATOR_CONTROL_OUTPUT_MASK,
};
#endif /* XAIE_FEATURE_CORE_ENABLE */

#ifdef XAIE_FEATURE_CORE_ENABLE
/* Register field attribute for core process bus control */
static const XAie_RegCoreProcBusCtrl Aie2PCoreProcBusCtrlReg =
{
	.RegOff = XAIE2PGBL_CORE_MODULE_CORE_PROCESSOR_BUS,
	.CtrlEn = {XAIE2PGBL_CORE_MODULE_CORE_PROCESSOR_BUS_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_CORE_PROCESSOR_BUS_ENABLE_MASK}
};

/* Core Module */
static const  XAie_CoreMod Aie2PCoreMod =
{
	.IsCheckerBoard = 0U,
	.ProgMemAddr = 0x0,
	.ProgMemSize = 16 * 1024,
	.DataMemAddr = 0x40000,
	.ProgMemHostOffset = XAIE2PGBL_CORE_MODULE_PROGRAM_MEMORY,
	.DataMemSize = 64 * 1024,		/* AIE2P Tile Memory is 64kB */
	.DataMemShift = 16,
	.EccEvntRegOff = XAIE2PGBL_CORE_MODULE_ECC_SCRUBBING_EVENT,
	.CorePCOff = XAIE2PGBL_CORE_MODULE_CORE_PC,
	.CoreSPOff = XAIE2PGBL_CORE_MODULE_CORE_SP,
	.CoreLROff = XAIE2PGBL_CORE_MODULE_CORE_LR,
	.CoreCtrl = &Aie2PCoreCtrlReg,
	.CoreDebugStatus = &Aie2PCoreDebugStatus,
	.CoreSts = &Aie2PCoreStsReg,
	.CoreDebug = &Aie2PCoreDebugReg,
	.CoreEvent = &Aie2PCoreEventReg,
	.CoreAccumCtrl = &Aie2PCoreAccumCtrlReg,
	.ProcBusCtrl = &Aie2PCoreProcBusCtrlReg,
	.ConfigureDone = &_XAieMl_CoreConfigureDone,
	.Enable = &_XAieMl_CoreEnable,
	.WaitForDone = &_XAieMl_CoreWaitForDone,
	.ReadDoneBit = &_XAieMl_CoreReadDoneBit,
	.GetCoreStatus = &_XAieMl_CoreGetStatus
};
#endif /* XAIE_FEATURE_CORE_ENABLE */

#ifdef XAIE_FEATURE_SS_ENABLE
/*
 * Array of all Tile Stream Switch Master Config registers
 * The data structure contains number of ports and the register offsets
 */
static const  XAie_StrmPort Aie2PTileStrmMstr[SS_PORT_TYPE_MAX] =
{
	{	/* Core */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0,
	},
	{	/* DMA */
		.NumPorts = 2,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_DMA0,
	},
	{	/* Ctrl */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_TILE_CTRL,
	},
	{	/* Fifo */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_FIFO0,
	},
	{	/* South */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_SOUTH0,
	},
	{	/* West */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_WEST0,
	},
	{	/* North */
		.NumPorts = 6,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_NORTH0,
	},
	{	/* East */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_EAST0,
	},
	{	/* Trace */
		.NumPorts = 0,
		.PortBaseAddr = 0
	}
};

/*
 * Array of all Tile Stream Switch Slave Config registers
 * The data structure contains number of ports and the register offsets
 */
static const  XAie_StrmPort Aie2PTileStrmSlv[SS_PORT_TYPE_MAX] =
{
	{	/* Core */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0,
	},
	{	/* DMA */
		.NumPorts = 2,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_DMA_0,
	},
	{	/* Ctrl */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_TILE_CTRL,
	},
	{	/* Fifo */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_FIFO_0,
	},
	{	/* South */
		.NumPorts = 6,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_SOUTH_0,
	},
	{	/* West */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_WEST_0,
	},
	{	/* North */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_NORTH_0,
	},
	{	/* East */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_EAST_0,
	},
	{	/* Trace */
		.NumPorts = 2,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_TRACE
	}
};

/*
 * Array of all Shim NOC/PL Stream Switch Master Config registers
 * The data structure contains number of ports and the register offsets
 */
static const  XAie_StrmPort Aie2PShimStrmMstr[SS_PORT_TYPE_MAX] =
{
	{	/* Core */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* DMA */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* Ctrl */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_MASTER_CONFIG_TILE_CTRL,
	},
	{	/* Fifo */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_MASTER_CONFIG_FIFO0,
	},
	{	/* South */
		.NumPorts = 6,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_MASTER_CONFIG_SOUTH0,
	},
	{	/* West */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_MASTER_CONFIG_WEST0,
	},
	{	/* North */
		.NumPorts = 6,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_MASTER_CONFIG_NORTH0,
	},
	{	/* East */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_MASTER_CONFIG_EAST0,
	},
	{	/* Trace */
		.NumPorts = 0,
		.PortBaseAddr = 0
	}
};

/*
 * Array of all Shim NOC/PL Stream Switch Slave Config registers
 * The data structure contains number of ports and the register offsets
 */
static const  XAie_StrmPort Aie2PShimStrmSlv[SS_PORT_TYPE_MAX] =
{
	{	/* Core */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* DMA */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* Ctrl */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_SLAVE_CONFIG_TILE_CTRL,
	},
	{	/* Fifo */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_SLAVE_CONFIG_FIFO_0,
	},
	{	/* South */
		.NumPorts = 8,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_SLAVE_CONFIG_SOUTH_0,
	},
	{	/* West */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_SLAVE_CONFIG_WEST_0,
	},
	{	/* North */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_SLAVE_CONFIG_NORTH_0,
	},
	{	/* East */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_SLAVE_CONFIG_EAST_0,
	},
	{	/* Trace */
		.NumPorts = 2,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_SLAVE_CONFIG_TRACE
	}
};

/*
 * Array of all Mem Tile Stream Switch Master Config registers
 * The data structure contains number of ports and the register offsets
 */
static const  XAie_StrmPort Aie2PMemTileStrmMstr[SS_PORT_TYPE_MAX] =
{
	{	/* Core */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* DMA */
		.NumPorts = 6,
		.PortBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_MASTER_CONFIG_DMA0,
	},
	{	/* Ctrl */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_MASTER_CONFIG_TILE_CTRL,
	},
	{	/* Fifo */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* South */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_MASTER_CONFIG_SOUTH0,
	},
	{	/* West */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* North */
		.NumPorts = 6,
		.PortBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_MASTER_CONFIG_NORTH0,
	},
	{	/* East */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* Trace */
		.NumPorts = 0,
		.PortBaseAddr = 0
	}
};

/*
 * Array of all Mem Tile Stream Switch Slave Config registers
 * The data structure contains number of ports and the register offsets
 */
static const  XAie_StrmPort Aie2PMemTileStrmSlv[SS_PORT_TYPE_MAX] =
{
	{	/* Core */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* DMA */
		.NumPorts = 6,
		.PortBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_DMA_0,
	},
	{	/* Ctrl */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_TILE_CTRL,
	},
	{	/* Fifo */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* South */
		.NumPorts = 6,
		.PortBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_SOUTH_0,
	},
	{	/* West */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* North */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_NORTH_0,
	},
	{	/* East */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* Trace */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_TRACE
	}
};

/*
 * Array of all Shim NOC/PL Stream Switch Slave Slot Config registers of AIE2P.
 * The data structure contains number of ports and the register base address.
 */
static const  XAie_StrmPort Aie2PShimStrmSlaveSlot[SS_PORT_TYPE_MAX] =
{
	{	/* Core */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* DMA */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* Ctrl */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_SLAVE_TILE_CTRL_SLOT0,
	},
	{	/* Fifo */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_SLAVE_FIFO_0_SLOT0,
	},
	{	/* South */
		.NumPorts = 8,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_SLAVE_SOUTH_0_SLOT0,
	},
	{	/* West */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_SLAVE_WEST_0_SLOT0,
	},
	{	/* North */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_SLAVE_NORTH_0_SLOT0,
	},
	{	/* East */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_SLAVE_EAST_0_SLOT0,
	},
	{	/* Trace */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_SLAVE_TRACE_SLOT0
	}
};

/*
 * Array of all AIE2P Tile Stream Switch Slave Slot Config registers.
 * The data structure contains number of ports and the register offsets
 */
static const  XAie_StrmPort Aie2PTileStrmSlaveSlot[SS_PORT_TYPE_MAX] =
{
	{	/* Core */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0,
	},
	{	/* DMA */
		.NumPorts = 2,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_DMA_0_SLOT0,
	},
	{	/* Ctrl */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_TILE_CTRL_SLOT0,
	},
	{	/* Fifo */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_FIFO_0_SLOT0,
	},
	{	/* South */
		.NumPorts = 6,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_SOUTH_0_SLOT0,
	},
	{	/* West */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_WEST_0_SLOT0,
	},
	{	/* North */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_NORTH_0_SLOT0,
	},
	{	/* East */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_EAST_0_SLOT0,
	},
	{	/* Trace */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_TRACE_SLOT0
	}
};

/*
 * Array of all AIE2P Mem Tile Stream Switch Slave Slot Config registers
 * The data structure contains number of ports and the register offsets
 */
static const  XAie_StrmPort Aie2PMemTileStrmSlaveSlot[SS_PORT_TYPE_MAX] =
{
	{	/* Core */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* DMA */
		.NumPorts = 6,
		.PortBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_SLAVE_DMA_0_SLOT0,
	},
	{	/* Ctrl */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_SLAVE_TILE_CTRL_SLOT0,
	},
	{	/* Fifo */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* South */
		.NumPorts = 6,
		.PortBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_SLAVE_SOUTH_0_SLOT0,
	},
	{	/* West */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* North */
		.NumPorts = 4,
		.PortBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_SLAVE_NORTH_0_SLOT0,
	},
	{	/* East */
		.NumPorts = 0,
		.PortBaseAddr = 0,
	},
	{	/* Trace */
		.NumPorts = 1,
		.PortBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_SLAVE_TRACE_SLOT0,
	}
};

static const XAie_StrmSwPortMap Aie2PTileStrmSwMasterPortMap[] =
{
	{
		/* PhyPort 0 */
		.PortType = CORE,
		.PortNum = 0,
	},
	{
		/* PhyPort 1 */
		.PortType = DMA,
		.PortNum = 0,
	},
	{
		/* PhyPort 2 */
		.PortType = DMA,
		.PortNum = 1,
	},
	{
		/* PhyPort 3 */
		.PortType = CTRL,
		.PortNum = 0,
	},
	{
		/* PhyPort 4 */
		.PortType = FIFO,
		.PortNum = 0,
	},
	{
		/* PhyPort 5 */
		.PortType = SOUTH,
		.PortNum = 0,
	},
	{
		/* PhyPort 6 */
		.PortType = SOUTH,
		.PortNum = 1,
	},
	{
		/* PhyPort 7 */
		.PortType = SOUTH,
		.PortNum = 2,
	},
	{
		/* PhyPort 8 */
		.PortType = SOUTH,
		.PortNum = 3,
	},
	{
		/* PhyPort 9 */
		.PortType = WEST,
		.PortNum = 0,
	},
	{
		/* PhyPort 10 */
		.PortType = WEST,
		.PortNum = 1,
	},
	{
		/* PhyPort 11 */
		.PortType = WEST,
		.PortNum = 2,
	},
	{
		/* PhyPort 12 */
		.PortType = WEST,
		.PortNum = 3,
	},
	{
		/* PhyPort 13 */
		.PortType = NORTH,
		.PortNum = 0,
	},
	{
		/* PhyPort 14 */
		.PortType = NORTH,
		.PortNum = 1,
	},
	{
		/* PhyPort 15 */
		.PortType = NORTH,
		.PortNum = 2,
	},
	{
		/* PhyPort 16 */
		.PortType = NORTH,
		.PortNum = 3,
	},
	{
		/* PhyPort 17 */
		.PortType = NORTH,
		.PortNum = 4,
	},
	{
		/* PhyPort 18 */
		.PortType = NORTH,
		.PortNum = 5,
	},
	{
		/* PhyPort 19 */
		.PortType = EAST,
		.PortNum = 0,
	},
	{
		/* PhyPort 20 */
		.PortType = EAST,
		.PortNum = 1,
	},
	{
		/* PhyPort 21 */
		.PortType = EAST,
		.PortNum = 2,
	},
	{
		/* PhyPort 22 */
		.PortType = EAST,
		.PortNum = 3,
	},
};

static const XAie_StrmSwPortMap Aie2PTileStrmSwSlavePortMap[] =
{
	{
		/* PhyPort 0 */
		.PortType = CORE,
		.PortNum = 0,
	},
	{
		/* PhyPort 1 */
		.PortType = DMA,
		.PortNum = 0,
	},
	{
		/* PhyPort 2 */
		.PortType = DMA,
		.PortNum = 1,
	},
	{
		/* PhyPort 3 */
		.PortType = CTRL,
		.PortNum = 0,
	},
	{
		/* PhyPort 4 */
		.PortType = FIFO,
		.PortNum = 0,
	},
	{
		/* PhyPort 5 */
		.PortType = SOUTH,
		.PortNum = 0,
	},
	{
		/* PhyPort 6 */
		.PortType = SOUTH,
		.PortNum = 1,
	},
	{
		/* PhyPort 7 */
		.PortType = SOUTH,
		.PortNum = 2,
	},
	{
		/* PhyPort 8 */
		.PortType = SOUTH,
		.PortNum = 3,
	},
	{
		/* PhyPort 9 */
		.PortType = SOUTH,
		.PortNum = 4,
	},
	{
		/* PhyPort 10 */
		.PortType = SOUTH,
		.PortNum = 5,
	},
	{
		/* PhyPort 11 */
		.PortType = WEST,
		.PortNum = 0,
	},
	{
		/* PhyPort 12 */
		.PortType = WEST,
		.PortNum = 1,
	},
	{
		/* PhyPort 13 */
		.PortType = WEST,
		.PortNum = 2,
	},
	{
		/* PhyPort 14 */
		.PortType = WEST,
		.PortNum = 3,
	},
	{
		/* PhyPort 15 */
		.PortType = NORTH,
		.PortNum = 0,
	},
	{
		/* PhyPort 16 */
		.PortType = NORTH,
		.PortNum = 1,
	},
	{
		/* PhyPort 17 */
		.PortType = NORTH,
		.PortNum = 2,
	},
	{
		/* PhyPort 18 */
		.PortType = NORTH,
		.PortNum = 3,
	},
	{
		/* PhyPort 19 */
		.PortType = EAST,
		.PortNum = 0,
	},
	{
		/* PhyPort 20 */
		.PortType = EAST,
		.PortNum = 1,
	},
	{
		/* PhyPort 21 */
		.PortType = EAST,
		.PortNum = 2,
	},
	{
		/* PhyPort 22 */
		.PortType = EAST,
		.PortNum = 3,
	},
	{
		/* PhyPort 23 */
		.PortType = TRACE,
		.PortNum = 0,
	},
	{
		/* PhyPort 24 */
		.PortType = TRACE,
		.PortNum = 1,
	},
};

static const XAie_StrmSwPortMap Aie2PShimStrmSwMasterPortMap[] =
{
	{
		/* PhyPort 0 */
		.PortType = CTRL,
		.PortNum = 0,
	},
	{
		/* PhyPort 1 */
		.PortType = FIFO,
		.PortNum = 0,
	},
	{
		/* PhyPort 2 */
		.PortType = SOUTH,
		.PortNum = 0,
	},
	{
		/* PhyPort 3 */
		.PortType = SOUTH,
		.PortNum = 1,
	},
	{
		/* PhyPort 4 */
		.PortType = SOUTH,
		.PortNum = 2,
	},
	{
		/* PhyPort 5 */
		.PortType = SOUTH,
		.PortNum = 3,
	},
	{
		/* PhyPort 6 */
		.PortType = SOUTH,
		.PortNum = 4,
	},
	{
		/* PhyPort 7 */
		.PortType = SOUTH,
		.PortNum = 5,
	},
	{
		/* PhyPort 8 */
		.PortType = WEST,
		.PortNum = 0,
	},
	{
		/* PhyPort 9 */
		.PortType = WEST,
		.PortNum = 1,
	},
	{
		/* PhyPort 10 */
		.PortType = WEST,
		.PortNum = 2,
	},
	{
		/* PhyPort 11 */
		.PortType = WEST,
		.PortNum = 3,
	},
	{
		/* PhyPort 12 */
		.PortType = NORTH,
		.PortNum = 0,
	},
	{
		/* PhyPort 13 */
		.PortType = NORTH,
		.PortNum = 1,
	},
	{
		/* PhyPort 14 */
		.PortType = NORTH,
		.PortNum = 2,
	},
	{
		/* PhyPort 15 */
		.PortType = NORTH,
		.PortNum = 3,
	},
	{
		/* PhyPort 16 */
		.PortType = NORTH,
		.PortNum = 4,
	},
	{
		/* PhyPort 17 */
		.PortType = NORTH,
		.PortNum = 5,
	},
	{
		/* PhyPort 18 */
		.PortType = EAST,
		.PortNum = 0,
	},
	{
		/* PhyPort 19 */
		.PortType = EAST,
		.PortNum = 1,
	},
	{
		/* PhyPort 20 */
		.PortType = EAST,
		.PortNum = 2,
	},
	{
		/* PhyPort 21 */
		.PortType = EAST,
		.PortNum = 3,
	},
};

static const XAie_StrmSwPortMap Aie2PShimStrmSwSlavePortMap[] =
{
	{
		/* PhyPort 0 */
		.PortType = CTRL,
		.PortNum = 0,
	},
	{
		/* PhyPort 1 */
		.PortType = FIFO,
		.PortNum = 0,
	},
	{
		/* PhyPort 2 */
		.PortType = SOUTH,
		.PortNum = 0,
	},
	{
		/* PhyPort 3 */
		.PortType = SOUTH,
		.PortNum = 1,
	},
	{
		/* PhyPort 4 */
		.PortType = SOUTH,
		.PortNum = 2,
	},
	{
		/* PhyPort 5 */
		.PortType = SOUTH,
		.PortNum = 3,
	},
	{
		/* PhyPort 6 */
		.PortType = SOUTH,
		.PortNum = 4,
	},
	{
		/* PhyPort 7 */
		.PortType = SOUTH,
		.PortNum = 5,
	},
	{
		/* PhyPort 8 */
		.PortType = SOUTH,
		.PortNum = 6,
	},
	{
		/* PhyPort 9 */
		.PortType = SOUTH,
		.PortNum = 7,
	},
	{
		/* PhyPort 10 */
		.PortType = WEST,
		.PortNum = 0,
	},
	{
		/* PhyPort 11 */
		.PortType = WEST,
		.PortNum = 1,
	},
	{
		/* PhyPort 12 */
		.PortType = WEST,
		.PortNum = 2,
	},
	{
		/* PhyPort 13 */
		.PortType = WEST,
		.PortNum = 3,
	},
	{
		/* PhyPort 14 */
		.PortType = NORTH,
		.PortNum = 0,
	},
	{
		/* PhyPort 15 */
		.PortType = NORTH,
		.PortNum = 1,
	},
	{
		/* PhyPort 16 */
		.PortType = NORTH,
		.PortNum = 2,
	},
	{
		/* PhyPort 17 */
		.PortType = NORTH,
		.PortNum = 3,
	},
	{
		/* PhyPort 18 */
		.PortType = EAST,
		.PortNum = 0,
	},
	{
		/* PhyPort 19 */
		.PortType = EAST,
		.PortNum = 1,
	},
	{
		/* PhyPort 20 */
		.PortType = EAST,
		.PortNum = 2,
	},
	{
		/* PhyPort 21 */
		.PortType = EAST,
		.PortNum = 3,
	},
	{
		/* PhyPort 22 */
		.PortType = TRACE,
		.PortNum = 0,
	},
};

static const XAie_StrmSwPortMap Aie2PMemTileStrmSwMasterPortMap[] =
{
	{
		/* PhyPort 0 */
		.PortType = DMA,
		.PortNum = 0,
	},
	{
		/* PhyPort 1 */
		.PortType = DMA,
		.PortNum = 1,
	},
	{
		/* PhyPort 2 */
		.PortType = DMA,
		.PortNum = 2,
	},
	{
		/* PhyPort 3 */
		.PortType = DMA,
		.PortNum = 3,
	},
	{
		/* PhyPort 4 */
		.PortType = DMA,
		.PortNum = 4,
	},
	{
		/* PhyPort 5 */
		.PortType = DMA,
		.PortNum = 5,
	},
	{
		/* PhyPort 6 */
		.PortType = CTRL,
		.PortNum = 0,
	},
	{
		/* PhyPort 7 */
		.PortType = SOUTH,
		.PortNum = 0,
	},
	{
		/* PhyPort 8 */
		.PortType = SOUTH,
		.PortNum = 1,
	},
	{
		/* PhyPort 9 */
		.PortType = SOUTH,
		.PortNum = 2,
	},
	{
		/* PhyPort 10 */
		.PortType = SOUTH,
		.PortNum = 3,
	},
	{
		/* PhyPort 11 */
		.PortType = NORTH,
		.PortNum = 0,
	},
	{
		/* PhyPort 12 */
		.PortType = NORTH,
		.PortNum = 1,
	},
	{
		/* PhyPort 13 */
		.PortType = NORTH,
		.PortNum = 2,
	},
	{
		/* PhyPort 14 */
		.PortType = NORTH,
		.PortNum = 3,
	},
	{
		/* PhyPort 15 */
		.PortType = NORTH,
		.PortNum = 4,
	},
	{
		/* PhyPort 16 */
		.PortType = NORTH,
		.PortNum = 5,
	},
};

static const XAie_StrmSwPortMap Aie2PMemTileStrmSwSlavePortMap[] =
{
	{
		/* PhyPort 0 */
		.PortType = DMA,
		.PortNum = 0,
	},
	{
		/* PhyPort 1 */
		.PortType = DMA,
		.PortNum = 1,
	},
	{
		/* PhyPort 2 */
		.PortType = DMA,
		.PortNum = 2,
	},
	{
		/* PhyPort 3 */
		.PortType = DMA,
		.PortNum = 3,
	},
	{
		/* PhyPort 4 */
		.PortType = DMA,
		.PortNum = 4,
	},
	{
		/* PhyPort 5 */
		.PortType = DMA,
		.PortNum = 5,
	},
	{
		/* PhyPort 6 */
		.PortType = CTRL,
		.PortNum = 0,
	},
	{
		/* PhyPort 7 */
		.PortType = SOUTH,
		.PortNum = 0,
	},
	{
		/* PhyPort 8 */
		.PortType = SOUTH,
		.PortNum = 1,
	},
	{
		/* PhyPort 9 */
		.PortType = SOUTH,
		.PortNum = 2,
	},
	{
		/* PhyPort 10 */
		.PortType = SOUTH,
		.PortNum = 3,
	},
	{
		/* PhyPort 11 */
		.PortType = NORTH,
		.PortNum = 0,
	},
	{
		/* PhyPort 12 */
		.PortType = NORTH,
		.PortNum = 1,
	},
	{
		/* PhyPort 13 */
		.PortType = NORTH,
		.PortNum = 2,
	},
	{
		/* PhyPort 14 */
		.PortType = NORTH,
		.PortNum = 3,
	},
	{
		/* PhyPort 15 */
		.PortType = NORTH,
		.PortNum = 4,
	},
	{
		/* PhyPort 16 */
		.PortType = NORTH,
		.PortNum = 5,
	},
	{
		/* PhyPort 17 */
		.PortType = TRACE,
		.PortNum = 0,
	},
};

/*
 * Data structure to capture stream switch deterministic merge properties for
 * AIE2P Tiles.
 */
static const XAie_StrmSwDetMerge Aie2PAieTileStrmSwDetMerge = {
	.NumArbitors = 2U,
	.NumPositions = 4U,
	.ArbConfigOffset = 0x10,
	.ConfigBase = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1,
	.EnableBase = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_CTRL,
	.SlvId0.Lsb = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_SLAVE_ID_0_LSB,
	.SlvId0.Mask = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_SLAVE_ID_0_MASK,
	.SlvId1.Lsb = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_SLAVE_ID_1_LSB,
	.SlvId1.Mask = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_SLAVE_ID_1_MASK,
	.PktCount0.Lsb = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_PACKET_COUNT_0_LSB,
	.PktCount0.Mask = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_PACKET_COUNT_0_MASK,
	.PktCount1.Lsb = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_PACKET_COUNT_1_LSB,
	.PktCount1.Mask = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_PACKET_COUNT_1_MASK,
	.Enable.Lsb = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_CTRL_ENABLE_LSB,
	.Enable.Mask = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_CTRL_ENABLE_MASK,
};

/*
 * Data structure to capture stream switch deterministic merge properties for
 * AIE2P Mem Tiles.
 */
static const XAie_StrmSwDetMerge Aie2PMemTileStrmSwDetMerge = {
	.NumArbitors = 2U,
	.NumPositions = 4U,
	.ArbConfigOffset = 0x10,
	.ConfigBase = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1,
	.EnableBase = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_CTRL,
	.SlvId0.Lsb = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_SLAVE_ID_0_LSB,
	.SlvId0.Mask = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_SLAVE_ID_0_MASK,
	.SlvId1.Lsb = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_SLAVE_ID_1_LSB,
	.SlvId1.Mask = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_SLAVE_ID_1_MASK,
	.PktCount0.Lsb = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_PACKET_COUNT_0_LSB,
	.PktCount0.Mask = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_PACKET_COUNT_0_MASK,
	.PktCount1.Lsb = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_PACKET_COUNT_1_LSB,
	.PktCount1.Mask = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_PACKET_COUNT_1_MASK,
	.Enable.Lsb = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_CTRL_ENABLE_LSB,
	.Enable.Mask = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_CTRL_ENABLE_MASK,
};

/*
 * Data structure to capture stream switch deterministic merge properties for
 * AIE2P SHIM PL Tiles.
 */
static const XAie_StrmSwDetMerge Aie2PShimTileStrmSwDetMerge = {
	.NumArbitors = 2U,
	.NumPositions = 4U,
	.ArbConfigOffset = 0x10,
	.ConfigBase = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1,
	.EnableBase = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_CTRL,
	.SlvId0.Lsb = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_SLAVE_ID_0_LSB,
	.SlvId0.Mask = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_SLAVE_ID_0_MASK,
	.SlvId1.Lsb = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_SLAVE_ID_1_LSB,
	.SlvId1.Mask = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_SLAVE_ID_1_MASK,
	.PktCount0.Lsb = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_PACKET_COUNT_0_LSB,
	.PktCount0.Mask = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_PACKET_COUNT_0_MASK,
	.PktCount1.Lsb = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_PACKET_COUNT_1_LSB,
	.PktCount1.Mask = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_SLAVE0_1_PACKET_COUNT_1_MASK,
	.Enable.Lsb = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_CTRL_ENABLE_LSB,
	.Enable.Mask = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_DETERMINISTIC_MERGE_ARB0_CTRL_ENABLE_MASK,
};

/*
 * Data structure to capture all stream configs for XAIEGBL_TILE_TYPE_AIETILE
 */
static const  XAie_StrmMod Aie2PTileStrmSw =
{
	.SlvConfigBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0,
	.MstrConfigBaseAddr = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0,
	.PortOffset = 0x4,
	.NumSlaveSlots = 4U,
	.SlotOffsetPerPort = 0x10,
	.SlotOffset = 0x4,
	.DetMergeFeature = XAIE_FEATURE_AVAILABLE,
	.MstrEn = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_MASTER_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_MASTER_ENABLE_MASK},
	.MstrPktEn = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_PACKET_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_PACKET_ENABLE_MASK},
	.DrpHdr = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_DROP_HEADER_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_DROP_HEADER_MASK},
	.Config = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_CONFIGURATION_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_CONFIGURATION_MASK},
	.SlvEn = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0_SLAVE_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0_SLAVE_ENABLE_MASK},
	.SlvPktEn = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0_PACKET_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0_PACKET_ENABLE_MASK},
	.SlotPktId = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ID_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ID_MASK},
	.SlotMask = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_MASK_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_MASK_MASK},
	.SlotEn = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ENABLE_MASK},
	.SlotMsel = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_MSEL_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_MSEL_MASK},
	.SlotArbitor = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ARBIT_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ARBIT_MASK},
	.MstrConfig = Aie2PTileStrmMstr,
	.SlvConfig = Aie2PTileStrmSlv,
	.SlvSlotConfig = Aie2PTileStrmSlaveSlot,
	.MaxMasterPhyPortId = 22U,
	.MaxSlavePhyPortId = 24U,
	.MasterPortMap = Aie2PTileStrmSwMasterPortMap,
	.SlavePortMap = Aie2PTileStrmSwSlavePortMap,
	.DetMerge = &Aie2PAieTileStrmSwDetMerge,
	.PortVerify = _XAieMl_AieTile_StrmSwCheckPortValidity,
};

/*
 * Data structure to capture all stream configs for XAIEGBL_TILE_TYPE_SHIMNOC/PL
 */
static const  XAie_StrmMod Aie2PShimStrmSw =
{
	.SlvConfigBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_SLAVE_CONFIG_TILE_CTRL,
	.MstrConfigBaseAddr = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_MASTER_CONFIG_TILE_CTRL,
	.PortOffset = 0x4,
	.NumSlaveSlots = 4U,
	.SlotOffsetPerPort = 0x10,
	.SlotOffset = 0x4,
	.DetMergeFeature = XAIE_FEATURE_AVAILABLE,
	.MstrEn = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_MASTER_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_MASTER_ENABLE_MASK},
	.MstrPktEn = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_PACKET_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_PACKET_ENABLE_MASK},
	.DrpHdr = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_DROP_HEADER_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_DROP_HEADER_MASK},
	.Config = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_CONFIGURATION_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_CONFIGURATION_MASK},
	.SlvEn = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0_SLAVE_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0_SLAVE_ENABLE_MASK},
	.SlvPktEn = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0_PACKET_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0_PACKET_ENABLE_MASK},
	.SlotPktId = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ID_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ID_MASK},
	.SlotMask = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_MASK_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_MASK_MASK},
	.SlotEn = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ENABLE_MASK},
	.SlotMsel = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_MSEL_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_MSEL_MASK},
	.SlotArbitor = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ARBIT_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ARBIT_MASK},
	.MstrConfig = Aie2PShimStrmMstr,
	.SlvConfig = Aie2PShimStrmSlv,
	.SlvSlotConfig = Aie2PShimStrmSlaveSlot,
	.MaxMasterPhyPortId = 21U,
	.MaxSlavePhyPortId = 22U,
	.MasterPortMap = Aie2PShimStrmSwMasterPortMap,
	.SlavePortMap = Aie2PShimStrmSwSlavePortMap,
	.DetMerge = &Aie2PShimTileStrmSwDetMerge,
	.PortVerify = _XAieMl_ShimTile_StrmSwCheckPortValidity,
};

/*
 * Data structure to capture all stream configs for XAIEGBL_TILE_TYPE_MEMTILE
 */
static const  XAie_StrmMod Aie2PMemTileStrmSw =
{
	.SlvConfigBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_DMA_0,
	.MstrConfigBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_MASTER_CONFIG_DMA0,
	.PortOffset = 0x4,
	.NumSlaveSlots = 4U,
	.SlotOffsetPerPort = 0x10,
	.SlotOffset = 0x4,
	.DetMergeFeature = XAIE_FEATURE_AVAILABLE,
	.MstrEn = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_MASTER_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_MASTER_ENABLE_MASK},
	.MstrPktEn = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_PACKET_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_PACKET_ENABLE_MASK},
	.DrpHdr = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_DROP_HEADER_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_DROP_HEADER_MASK},
	.Config = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_CONFIGURATION_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_AIE_CORE0_CONFIGURATION_MASK},
	.SlvEn = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0_SLAVE_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0_SLAVE_ENABLE_MASK},
	.SlvPktEn = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0_PACKET_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0_PACKET_ENABLE_MASK},
	.SlotPktId = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ID_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ID_MASK},
	.SlotMask = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_MASK_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_MASK_MASK},
	.SlotEn = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ENABLE_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ENABLE_MASK},
	.SlotMsel = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_MSEL_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_MSEL_MASK},
	.SlotArbitor = {XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ARBIT_LSB, XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_SLAVE_AIE_CORE0_SLOT0_ARBIT_MASK},
	.MstrConfig = Aie2PMemTileStrmMstr,
	.SlvConfig = Aie2PMemTileStrmSlv,
	.SlvSlotConfig = Aie2PMemTileStrmSlaveSlot,
	.MaxMasterPhyPortId = 16U,
	.MaxSlavePhyPortId = 17U,
	.MasterPortMap = Aie2PMemTileStrmSwMasterPortMap,
	.SlavePortMap = Aie2PMemTileStrmSwSlavePortMap,
	.DetMerge = &Aie2PMemTileStrmSwDetMerge,
	.PortVerify = _XAieMl_MemTile_StrmSwCheckPortValidity,
};
#endif /* XAIE_FEATURE_SS_ENABLE */

#ifdef XAIE_FEATURE_DMA_ENABLE
static const  XAie_DmaBdEnProp Aie2PMemTileDmaBdEnProp =
{
	.NxtBd.Idx = 1U,
	.NxtBd.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_1_NEXT_BD_LSB,
	.NxtBd.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_1_NEXT_BD_MASK,
	.UseNxtBd.Idx = 1U,
	.UseNxtBd.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_1_USE_NEXT_BD_LSB,
	.UseNxtBd.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_1_USE_NEXT_BD_MASK,
	.ValidBd.Idx = 7U,
	.ValidBd.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_7_VALID_BD_LSB,
	.ValidBd.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_7_VALID_BD_MASK,
	.OutofOrderBdId.Idx = 0U,
	.OutofOrderBdId.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_0_OUT_OF_ORDER_BD_ID_LSB,
	.OutofOrderBdId.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_0_OUT_OF_ORDER_BD_ID_MASK,
	.TlastSuppress.Idx = 2U,
	.TlastSuppress.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_2_TLAST_SUPPRESS_LSB,
	.TlastSuppress.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_2_TLAST_SUPPRESS_MASK,
};

static const  XAie_DmaBdPkt Aie2PMemTileDmaBdPktProp =
{
	.EnPkt.Idx = 0U,
	.EnPkt.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_0_ENABLE_PACKET_LSB,
	.EnPkt.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_0_ENABLE_PACKET_MASK,
	.PktId.Idx = 0U,
	.PktId.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_0_PACKET_ID_LSB,
	.PktId.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_0_PACKET_ID_MASK,
	.PktType.Idx = 0U,
	.PktType.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_0_PACKET_TYPE_LSB,
	.PktType.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_0_PACKET_TYPE_MASK,
};

static const  XAie_DmaBdLock Aie2PMemTileDmaLockProp =
{
	.AieMlDmaLock.LckRelVal.Idx = 7U,
	.AieMlDmaLock.LckRelVal.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_7_LOCK_REL_VALUE_LSB,
	.AieMlDmaLock.LckRelVal.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_7_LOCK_REL_VALUE_MASK,
	.AieMlDmaLock.LckRelId.Idx = 7U,
	.AieMlDmaLock.LckRelId.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_7_LOCK_REL_ID_LSB,
	.AieMlDmaLock.LckRelId.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_7_LOCK_REL_ID_MASK,
	.AieMlDmaLock.LckAcqEn.Idx = 7U,
	.AieMlDmaLock.LckAcqEn.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_7_LOCK_ACQ_ENABLE_LSB,
	.AieMlDmaLock.LckAcqEn.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_7_LOCK_ACQ_ENABLE_MASK,
	.AieMlDmaLock.LckAcqVal.Idx = 7U,
	.AieMlDmaLock.LckAcqVal.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_7_LOCK_ACQ_VALUE_LSB,
	.AieMlDmaLock.LckAcqVal.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_7_LOCK_ACQ_VALUE_MASK,
	.AieMlDmaLock.LckAcqId.Idx = 7U,
	.AieMlDmaLock.LckAcqId.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_7_LOCK_ACQ_ID_LSB,
	.AieMlDmaLock.LckAcqId.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_7_LOCK_ACQ_ID_MASK,
};

static const  XAie_DmaBdBuffer Aie2PMemTileBufferProp =
{
	.TileDmaBuff.BaseAddr.Idx = 1U,
	.TileDmaBuff.BaseAddr.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_1_BASE_ADDRESS_LSB,
	.TileDmaBuff.BaseAddr.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_1_BASE_ADDRESS_MASK,
};

static const XAie_DmaBdDoubleBuffer Aie2PMemTileDoubleBufferProp =
{
	.EnDoubleBuff = {0U},
	.BaseAddr_B = {0U},
	.FifoMode = {0U},
	.EnIntrleaved = {0U},
	.IntrleaveCnt = {0U},
	.BuffSelect = {0U},
};

static const  XAie_DmaBdMultiDimAddr Aie2PMemTileMultiDimProp =
{
	.AieMlMultiDimAddr.DmaDimProp[1U].StepSize.Idx = 3U,
	.AieMlMultiDimAddr.DmaDimProp[1U].StepSize.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_3_D1_STEPSIZE_LSB,
	.AieMlMultiDimAddr.DmaDimProp[1U].StepSize.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_3_D1_STEPSIZE_MASK,
	.AieMlMultiDimAddr.DmaDimProp[0U].StepSize.Idx = 2U,
	.AieMlMultiDimAddr.DmaDimProp[0U].StepSize.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_2_D0_STEPSIZE_LSB,
	.AieMlMultiDimAddr.DmaDimProp[0U].StepSize.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_2_D0_STEPSIZE_MASK,
	.AieMlMultiDimAddr.DmaDimProp[1U].Wrap.Idx = 3U,
	.AieMlMultiDimAddr.DmaDimProp[1U].Wrap.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_3_D1_WRAP_LSB,
	.AieMlMultiDimAddr.DmaDimProp[1U].Wrap.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_3_D1_WRAP_MASK,
	.AieMlMultiDimAddr.DmaDimProp[0U].Wrap.Idx = 2U,
	.AieMlMultiDimAddr.DmaDimProp[0U].Wrap.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_2_D0_WRAP_LSB,
	.AieMlMultiDimAddr.DmaDimProp[0U].Wrap.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_2_D0_WRAP_MASK,
	.AieMlMultiDimAddr.DmaDimProp[2U].StepSize.Idx = 4U,
	.AieMlMultiDimAddr.DmaDimProp[2U].StepSize.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_4_D2_STEPSIZE_LSB,
	.AieMlMultiDimAddr.DmaDimProp[2U].StepSize.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_4_D2_STEPSIZE_MASK,
	.AieMlMultiDimAddr.DmaDimProp[2U].Wrap.Idx = 4U,
	.AieMlMultiDimAddr.DmaDimProp[2U].Wrap.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_4_D2_WRAP_LSB,
	.AieMlMultiDimAddr.DmaDimProp[2U].Wrap.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_4_D2_WRAP_MASK,
	.AieMlMultiDimAddr.IterCurr.Idx = 6U,
	.AieMlMultiDimAddr.IterCurr.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_6_ITERATION_CURRENT_LSB,
	.AieMlMultiDimAddr.IterCurr.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_6_ITERATION_CURRENT_MASK,
	.AieMlMultiDimAddr.Iter.Wrap.Idx = 6U,
	.AieMlMultiDimAddr.Iter.Wrap.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_6_ITERATION_WRAP_LSB,
	.AieMlMultiDimAddr.Iter.Wrap.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_6_ITERATION_WRAP_MASK,
	.AieMlMultiDimAddr.Iter.StepSize.Idx = 6U,
	.AieMlMultiDimAddr.Iter.StepSize.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_6_ITERATION_STEPSIZE_LSB,
	.AieMlMultiDimAddr.Iter.StepSize.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_6_ITERATION_STEPSIZE_MASK,
	.AieMlMultiDimAddr.DmaDimProp[3U].StepSize.Idx = 5U,
	.AieMlMultiDimAddr.DmaDimProp[3U].StepSize.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_5_D3_STEPSIZE_LSB,
	.AieMlMultiDimAddr.DmaDimProp[3U].StepSize.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_5_D3_STEPSIZE_MASK,
};

static const  XAie_DmaBdPad Aie2PMemTilePadProp =
{
	.D0_PadBefore.Idx = 1U,
	.D0_PadBefore.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_1_D0_PAD_BEFORE_LSB,
	.D0_PadBefore.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_1_D0_PAD_BEFORE_MASK,
	.D1_PadBefore.Idx = 3U,
	.D1_PadBefore.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_3_D1_PAD_BEFORE_LSB,
	.D1_PadBefore.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_3_D1_PAD_BEFORE_MASK,
	.D2_PadBefore.Idx = 4U,
	.D2_PadBefore.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_4_D2_PAD_BEFORE_LSB,
	.D2_PadBefore.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_4_D2_PAD_BEFORE_MASK,
	.D0_PadAfter.Idx = 5U,
	.D0_PadAfter.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_5_D0_PAD_AFTER_LSB,
	.D0_PadAfter.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_5_D0_PAD_AFTER_MASK,
	.D1_PadAfter.Idx = 5U,
	.D1_PadAfter.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_5_D1_PAD_AFTER_LSB,
	.D1_PadAfter.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_5_D1_PAD_AFTER_MASK,
	.D2_PadAfter.Idx = 5U,
	.D2_PadAfter.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_5_D2_PAD_AFTER_LSB,
	.D2_PadAfter.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_5_D2_PAD_AFTER_MASK,
};

static const  XAie_DmaBdCompression Aie2PMemTileCompressionProp =
{
	.EnCompression.Idx = 4U,
	.EnCompression.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_4_ENABLE_COMPRESSION_LSB,
	.EnCompression.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_4_ENABLE_COMPRESSION_MASK,
};

/* Data structure to capture register offsets and masks for Mem Tile Dma */
static const  XAie_DmaBdProp Aie2PMemTileDmaProp =
{
	.AddrAlignMask = 0x3,
	.AddrAlignShift = 0x2,
	.AddrMax = 0x180000,
	.LenActualOffset = 0U,
	.StepSizeMax = (1U << 17) - 1U,
	.WrapMax = (1U << 10U) - 1U,
	.IterStepSizeMax = (1U << 17) - 1U,
	.IterWrapMax = (1U << 6U) - 1U,
	.IterCurrMax = (1U << 6) - 1U,
	.BufferLen.Idx = 0U,
	.BufferLen.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_0_BUFFER_LENGTH_LSB,
	.BufferLen.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_0_BUFFER_LENGTH_MASK,
	.Buffer = &Aie2PMemTileBufferProp,
	.DoubleBuffer = &Aie2PMemTileDoubleBufferProp,
	.Lock = &Aie2PMemTileDmaLockProp,
	.Pkt = &Aie2PMemTileDmaBdPktProp,
	.BdEn = &Aie2PMemTileDmaBdEnProp,
	.AddrMode = &Aie2PMemTileMultiDimProp,
	.Pad = &Aie2PMemTilePadProp,
	.Compression = &Aie2PMemTileCompressionProp,
	.SysProp = NULL
};

static const XAie_DmaChStatus Aie2PMemTileDmaChStatus =
{
	/* This database is common for mm2s and s2mm channels */
	.AieMlDmaChStatus.Status.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_STATUS_0_STATUS_LSB,
	.AieMlDmaChStatus.Status.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_STATUS_0_STATUS_MASK,
	.AieMlDmaChStatus.TaskQSize.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_STATUS_0_TASK_QUEUE_SIZE_LSB,
	.AieMlDmaChStatus.TaskQSize.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_STATUS_0_TASK_QUEUE_SIZE_MASK,
	.AieMlDmaChStatus.ChannelRunning.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_STATUS_0_CHANNEL_RUNNING_LSB,
	.AieMlDmaChStatus.ChannelRunning.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_STATUS_0_CHANNEL_RUNNING_MASK,
	.AieMlDmaChStatus.StalledLockAcq.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_STATUS_0_STALLED_LOCK_ACQ_LSB,
	.AieMlDmaChStatus.StalledLockAcq.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_STATUS_0_STALLED_LOCK_ACQ_MASK,
	.AieMlDmaChStatus.StalledLockRel.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_STATUS_0_STALLED_LOCK_REL_LSB,
	.AieMlDmaChStatus.StalledLockRel.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_STATUS_0_STALLED_LOCK_REL_MASK,
	.AieMlDmaChStatus.StalledStreamStarve.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_STATUS_0_STALLED_STREAM_STARVATION_LSB,
	.AieMlDmaChStatus.StalledStreamStarve.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_STATUS_0_STALLED_STREAM_STARVATION_MASK,
	.AieMlDmaChStatus.StalledTCT.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_STATUS_0_STALLED_TCT_OR_COUNT_FIFO_FULL_LSB,
	.AieMlDmaChStatus.StalledTCT.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_STATUS_0_STALLED_TCT_OR_COUNT_FIFO_FULL_MASK,
};

static const  XAie_DmaChProp Aie2PMemTileDmaChProp =
{
	.HasFoTMode = XAIE_FEATURE_AVAILABLE,
	.HasControllerId = XAIE_FEATURE_AVAILABLE,
	.HasEnCompression = XAIE_FEATURE_AVAILABLE,
	.HasEnOutOfOrder = XAIE_FEATURE_AVAILABLE,
	.MaxFoTMode = DMA_FoT_COUNTS_FROM_MM_REG,
	.MaxRepeatCount = 256U,
	.ControllerId.Idx = 0,
	.ControllerId.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_CTRL_CONTROLLER_ID_LSB,
	.ControllerId.Mask =XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_CTRL_CONTROLLER_ID_MASK ,
	.EnCompression.Idx = 0,
	.EnCompression.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_CTRL_DECOMPRESSION_ENABLE_LSB,
	.EnCompression.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_CTRL_DECOMPRESSION_ENABLE_MASK,
	.EnOutofOrder.Idx = 0,
	.EnOutofOrder.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_CTRL_ENABLE_OUT_OF_ORDER_LSB,
	.EnOutofOrder.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_CTRL_ENABLE_OUT_OF_ORDER_MASK,
	.FoTMode.Idx = 0,
	.FoTMode.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_CTRL_FOT_MODE_LSB,
	.FoTMode.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_CTRL_FOT_MODE_MASK ,
	.Reset.Idx = 0,
	.Reset.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_CTRL_RESET_LSB,
	.Reset.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_CTRL_RESET_MASK,
	.EnToken.Idx = 1,
	.EnToken.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_START_QUEUE_ENABLE_TOKEN_ISSUE_LSB,
	.EnToken.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_START_QUEUE_ENABLE_TOKEN_ISSUE_MASK,
	.RptCount.Idx = 1,
	.RptCount.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_START_QUEUE_REPEAT_COUNT_LSB,
	.RptCount.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_START_QUEUE_REPEAT_COUNT_MASK,
	.StartBd.Idx = 1,
	.StartBd.Lsb = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_START_QUEUE_START_BD_ID_LSB,
	.StartBd.Mask = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_START_QUEUE_START_BD_ID_MASK,
	.PauseStream = {0U},
	.PauseMem = {0U},
	.Enable = {0U},
	.StartQSizeMax = 4U,
	.DmaChStatus = &Aie2PMemTileDmaChStatus,
};

/* Mem Tile Dma Module */
static const  XAie_DmaMod Aie2PMemTileDmaMod =
{
	.BaseAddr = XAIE2PGBL_MEM_TILE_MODULE_DMA_BD0_0,
	.IdxOffset = 0x20,  /* This is the offset between each BD */
	.NumBds = 48,	   /* Number of BDs for AIE2P Tile DMA */
	.NumLocks = 192U,
	.NumAddrDim = 4U,
	.DoubleBuffering = XAIE_FEATURE_UNAVAILABLE,
	.Compression = XAIE_FEATURE_AVAILABLE,
	.Padding = XAIE_FEATURE_AVAILABLE,
	.OutofOrderBdId = XAIE_FEATURE_AVAILABLE,
	.InterleaveMode = XAIE_FEATURE_UNAVAILABLE,
	.FifoMode = XAIE_FEATURE_UNAVAILABLE,
	.EnTokenIssue = XAIE_FEATURE_AVAILABLE,
	.RepeatCount = XAIE_FEATURE_AVAILABLE,
	.TlastSuppress = XAIE_FEATURE_AVAILABLE,
	.StartQueueBase = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_START_QUEUE,
	.ChCtrlBase = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_0_CTRL,
	.NumChannels = 6,  /* number of s2mm/mm2s channels */
	.ChIdxOffset = 0x8,  /* This is the offset between each channel */
	.ChStatusBase = XAIE2PGBL_MEM_TILE_MODULE_DMA_S2MM_STATUS_0,
	.ChStatusOffset = 0x20,
	.PadValueBase = XAIE2PGBL_MEM_TILE_MODULE_DMA_MM2S_0_CONSTANT_PAD_VALUE,
	.BdProp = &Aie2PMemTileDmaProp,
	.ChProp = &Aie2PMemTileDmaChProp,
	.DmaBdInit = &_XAieMl_MemTileDmaInit,
	.SetLock = &_XAieMl_DmaSetLock,
	.SetIntrleave = NULL,
	.SetMultiDim = &_XAieMl_DmaSetMultiDim,
	.SetBdIter = &_XAieMl_DmaSetBdIteration,
	.WriteBd = &_XAieMl_MemTileDmaWriteBd,
	.ReadBd = &_XAieMl_MemTileDmaReadBd,
	.PendingBd = &_XAieMl_DmaGetPendingBdCount,
	.WaitforDone = &_XAieMl_DmaWaitForDone,
	.WaitforBdTaskQueue = &_XAieMl_DmaWaitForBdTaskQueue,
	.BdChValidity = &_XAieMl_MemTileDmaCheckBdChValidity,
	.UpdateBdLen = &_XAieMl_DmaUpdateBdLen,
	.UpdateBdAddr = &_XAieMl_DmaUpdateBdAddr,
	.GetChannelStatus = &_XAieMl_DmaGetChannelStatus,
	.AxiBurstLenCheck = NULL,
};

static const  XAie_DmaBdEnProp Aie2PTileDmaBdEnProp =
{
	.NxtBd.Idx = 5U,
	.NxtBd.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_NEXT_BD_LSB,
	.NxtBd.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_NEXT_BD_MASK,
	.UseNxtBd.Idx = 5U,
	.UseNxtBd.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_USE_NEXT_BD_LSB,
	.UseNxtBd.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_USE_NEXT_BD_MASK,
	.ValidBd.Idx = 5U,
	.ValidBd.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_VALID_BD_LSB,
	.ValidBd.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_VALID_BD_MASK,
	.OutofOrderBdId.Idx = 1U,
	.OutofOrderBdId.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_1_OUT_OF_ORDER_BD_ID_LSB,
	.OutofOrderBdId.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_1_OUT_OF_ORDER_BD_ID_MASK,
	.TlastSuppress.Idx = 5U,
	.TlastSuppress.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_TLAST_SUPPRESS_LSB,
	.TlastSuppress.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_TLAST_SUPPRESS_MASK,
};

static const  XAie_DmaBdPkt Aie2PTileDmaBdPktProp =
{
	.EnPkt.Idx = 1U,
	.EnPkt.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_1_ENABLE_PACKET_LSB,
	.EnPkt.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_1_ENABLE_PACKET_MASK,
	.PktId.Idx = 1U,
	.PktId.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_1_PACKET_ID_LSB,
	.PktId.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_1_PACKET_ID_MASK,
	.PktType.Idx = 1U,
	.PktType.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_1_PACKET_TYPE_LSB,
	.PktType.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_1_PACKET_TYPE_MASK,
};

static const  XAie_DmaBdLock Aie2PTileDmaLockProp =
{
	.AieMlDmaLock.LckRelVal.Idx = 5U,
	.AieMlDmaLock.LckRelVal.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_LOCK_REL_VALUE_LSB,
	.AieMlDmaLock.LckRelVal.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_LOCK_REL_VALUE_MASK,
	.AieMlDmaLock.LckRelId.Idx = 5U,
	.AieMlDmaLock.LckRelId.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_LOCK_REL_ID_LSB,
	.AieMlDmaLock.LckRelId.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_LOCK_REL_ID_MASK,
	.AieMlDmaLock.LckAcqEn.Idx = 5U,
	.AieMlDmaLock.LckAcqEn.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_LOCK_ACQ_ENABLE_LSB,
	.AieMlDmaLock.LckAcqEn.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_LOCK_ACQ_ENABLE_MASK,
	.AieMlDmaLock.LckAcqVal.Idx = 5U,
	.AieMlDmaLock.LckAcqVal.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_LOCK_ACQ_VALUE_LSB,
	.AieMlDmaLock.LckAcqVal.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_LOCK_ACQ_VALUE_MASK,
	.AieMlDmaLock.LckAcqId.Idx = 5U,
	.AieMlDmaLock.LckAcqId.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_LOCK_ACQ_ID_LSB,
	.AieMlDmaLock.LckAcqId.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_5_LOCK_ACQ_ID_MASK,
};

static const  XAie_DmaBdBuffer Aie2PTileDmaBufferProp =
{
	.TileDmaBuff.BaseAddr.Idx = 0U,
	.TileDmaBuff.BaseAddr.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_0_BASE_ADDRESS_LSB,
	.TileDmaBuff.BaseAddr.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_0_BASE_ADDRESS_MASK,
};

static const XAie_DmaBdDoubleBuffer Aie2PTileDmaDoubleBufferProp =
{
	.EnDoubleBuff = {0U},
	.BaseAddr_B = {0U},
	.FifoMode = {0U},
	.EnIntrleaved = {0U},
	.IntrleaveCnt = {0U},
	.BuffSelect = {0U},
};

static const  XAie_DmaBdMultiDimAddr Aie2PTileDmaMultiDimProp =
{
	.AieMlMultiDimAddr.DmaDimProp[0U].StepSize.Idx = 2U,
	.AieMlMultiDimAddr.DmaDimProp[0U].StepSize.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_2_D0_STEPSIZE_LSB,
	.AieMlMultiDimAddr.DmaDimProp[0U].StepSize.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_2_D0_STEPSIZE_MASK,
	.AieMlMultiDimAddr.DmaDimProp[0U].Wrap.Idx = 3U,
	.AieMlMultiDimAddr.DmaDimProp[0U].Wrap.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_3_D0_WRAP_LSB,
	.AieMlMultiDimAddr.DmaDimProp[0U].Wrap.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_3_D0_WRAP_MASK,
	.AieMlMultiDimAddr.DmaDimProp[1U].StepSize.Idx = 2U,
	.AieMlMultiDimAddr.DmaDimProp[1U].StepSize.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_2_D1_STEPSIZE_LSB,
	.AieMlMultiDimAddr.DmaDimProp[1U].StepSize.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_2_D1_STEPSIZE_MASK,
	.AieMlMultiDimAddr.DmaDimProp[1U].Wrap.Idx = 3U,
	.AieMlMultiDimAddr.DmaDimProp[1U].Wrap.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_3_D1_WRAP_LSB,
	.AieMlMultiDimAddr.DmaDimProp[1U].Wrap.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_3_D1_WRAP_MASK,
	.AieMlMultiDimAddr.DmaDimProp[2U].StepSize.Idx = 3U,
	.AieMlMultiDimAddr.DmaDimProp[2U].StepSize.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_3_D2_STEPSIZE_LSB,
	.AieMlMultiDimAddr.DmaDimProp[2U].StepSize.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_3_D2_STEPSIZE_MASK,
	.AieMlMultiDimAddr.IterCurr.Idx = 4U,
	.AieMlMultiDimAddr.IterCurr.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_4_ITERATION_CURRENT_LSB,
	.AieMlMultiDimAddr.IterCurr.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_4_ITERATION_CURRENT_MASK,
	.AieMlMultiDimAddr.Iter.Wrap.Idx = 4U,
	.AieMlMultiDimAddr.Iter.Wrap.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_4_ITERATION_WRAP_LSB,
	.AieMlMultiDimAddr.Iter.Wrap.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_4_ITERATION_WRAP_MASK,
	.AieMlMultiDimAddr.Iter.StepSize.Idx = 4U,
	.AieMlMultiDimAddr.Iter.StepSize.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_4_ITERATION_STEPSIZE_LSB,
	.AieMlMultiDimAddr.Iter.StepSize.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_4_ITERATION_STEPSIZE_MASK,
	.AieMlMultiDimAddr.DmaDimProp[2U].Wrap = {0U},
	.AieMlMultiDimAddr.DmaDimProp[3U].Wrap = {0U},
	.AieMlMultiDimAddr.DmaDimProp[3U].StepSize = {0U}
};

static const  XAie_DmaBdCompression Aie2PTileDmaCompressionProp =
{
	.EnCompression.Idx = 1U,
	.EnCompression.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_1_ENABLE_COMPRESSION_LSB,
	.EnCompression.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_1_ENABLE_COMPRESSION_MASK,
};

/* Data structure to capture register offsets and masks for Tile Dma */
static const  XAie_DmaBdProp Aie2PTileDmaProp =
{
	.AddrAlignMask = 0x3,
	.AddrAlignShift = 0x2,
	.AddrMax = 0x20000,
	.LenActualOffset = 0U,
	.StepSizeMax = (1U << 13) - 1U,
	.WrapMax = (1U << 8U) - 1U,
	.IterStepSizeMax = (1U << 13) - 1U,
	.IterWrapMax = (1U << 6U) - 1U,
	.IterCurrMax = (1U << 6) - 1U,
	.BufferLen.Idx = 0U,
	.BufferLen.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_0_BUFFER_LENGTH_LSB,
	.BufferLen.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_0_BUFFER_LENGTH_MASK,
	.Buffer = &Aie2PTileDmaBufferProp,
	.DoubleBuffer = &Aie2PTileDmaDoubleBufferProp,
	.Lock = &Aie2PTileDmaLockProp,
	.Pkt = &Aie2PTileDmaBdPktProp,
	.BdEn = &Aie2PTileDmaBdEnProp,
	.AddrMode = &Aie2PTileDmaMultiDimProp,
	.Pad = NULL,
	.Compression = &Aie2PTileDmaCompressionProp,
	.SysProp = NULL
};

static const XAie_DmaChStatus Aie2PTileDmaChStatus =
{
	/* This database is common for mm2s and s2mm channels */
	.AieMlDmaChStatus.Status.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_STATUS_0_STATUS_LSB,
	.AieMlDmaChStatus.Status.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_STATUS_0_STATUS_MASK,
	.AieMlDmaChStatus.TaskQSize.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_STATUS_0_TASK_QUEUE_SIZE_LSB,
	.AieMlDmaChStatus.TaskQSize.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_STATUS_0_TASK_QUEUE_SIZE_MASK,
	.AieMlDmaChStatus.ChannelRunning.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_STATUS_0_CHANNEL_RUNNING_LSB,
	.AieMlDmaChStatus.ChannelRunning.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_STATUS_0_CHANNEL_RUNNING_MASK,
	.AieMlDmaChStatus.StalledLockAcq.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_STATUS_0_STALLED_LOCK_ACQ_LSB,
	.AieMlDmaChStatus.StalledLockAcq.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_STATUS_0_STALLED_LOCK_ACQ_MASK,
	.AieMlDmaChStatus.StalledLockRel.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_STATUS_0_STALLED_LOCK_REL_LSB,
	.AieMlDmaChStatus.StalledLockRel.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_STATUS_0_STALLED_LOCK_REL_MASK,
	.AieMlDmaChStatus.StalledStreamStarve.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_STATUS_0_STALLED_STREAM_STARVATION_LSB,
	.AieMlDmaChStatus.StalledStreamStarve.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_STATUS_0_STALLED_STREAM_STARVATION_MASK,
	.AieMlDmaChStatus.StalledTCT.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_STATUS_0_STALLED_TCT_OR_COUNT_FIFO_FULL_LSB,
	.AieMlDmaChStatus.StalledTCT.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_STATUS_0_STALLED_TCT_OR_COUNT_FIFO_FULL_MASK,

};

/* Data structure to capture register offsets and masks for Mem Tile and
 * Tile Dma Channels
 */
static const  XAie_DmaChProp Aie2PDmaChProp =
{
	.HasFoTMode = XAIE_FEATURE_AVAILABLE,
	.HasControllerId = XAIE_FEATURE_AVAILABLE,
	.HasEnCompression = XAIE_FEATURE_AVAILABLE,
	.HasEnOutOfOrder = XAIE_FEATURE_AVAILABLE,
	.MaxFoTMode = DMA_FoT_COUNTS_FROM_MM_REG,
	.MaxRepeatCount = 256U,
	.ControllerId.Idx = 0,
	.ControllerId.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_CTRL_CONTROLLER_ID_LSB,
	.ControllerId.Mask =XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_CTRL_CONTROLLER_ID_MASK ,
	.EnCompression.Idx = 0,
	.EnCompression.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_CTRL_DECOMPRESSION_ENABLE_LSB,
	.EnCompression.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_CTRL_DECOMPRESSION_ENABLE_MASK,
	.EnOutofOrder.Idx = 0,
	.EnOutofOrder.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_CTRL_ENABLE_OUT_OF_ORDER_LSB,
	.EnOutofOrder.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_CTRL_ENABLE_OUT_OF_ORDER_MASK,
	.FoTMode.Idx = 0,
	.FoTMode.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_CTRL_FOT_MODE_LSB,
	.FoTMode.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_CTRL_FOT_MODE_MASK ,
	.Reset.Idx = 0,
	.Reset.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_CTRL_RESET_LSB,
	.Reset.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_CTRL_RESET_MASK,
	.EnToken.Idx = 1,
	.EnToken.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_START_QUEUE_ENABLE_TOKEN_ISSUE_LSB,
	.EnToken.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_START_QUEUE_ENABLE_TOKEN_ISSUE_MASK,
	.RptCount.Idx = 1,
	.RptCount.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_START_QUEUE_REPEAT_COUNT_LSB,
	.RptCount.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_START_QUEUE_REPEAT_COUNT_MASK,
	.StartBd.Idx = 1,
	.StartBd.Lsb = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_START_QUEUE_START_BD_ID_LSB,
	.StartBd.Mask = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_START_QUEUE_START_BD_ID_MASK,
	.PauseStream = {0U},
	.PauseMem = {0U},
	.Enable = {0U},
	.StartQSizeMax = 4U,
	.DmaChStatus = &Aie2PTileDmaChStatus,
};

/* Tile Dma Module */
static const  XAie_DmaMod Aie2PTileDmaMod =
{
	.BaseAddr = XAIE2PGBL_MEMORY_MODULE_DMA_BD0_0,
	.IdxOffset = 0x20,  	/* This is the offset between each BD */
	.NumBds = 16U,	   	/* Number of BDs for AIE2P Tile DMA */
	.NumLocks = 16U,
	.NumAddrDim = 3U,
	.DoubleBuffering = XAIE_FEATURE_UNAVAILABLE,
	.Compression = XAIE_FEATURE_AVAILABLE,
	.Padding = XAIE_FEATURE_UNAVAILABLE,
	.OutofOrderBdId = XAIE_FEATURE_AVAILABLE,
	.InterleaveMode = XAIE_FEATURE_UNAVAILABLE,
	.FifoMode = XAIE_FEATURE_UNAVAILABLE,
	.EnTokenIssue = XAIE_FEATURE_AVAILABLE,
	.RepeatCount = XAIE_FEATURE_AVAILABLE,
	.TlastSuppress = XAIE_FEATURE_AVAILABLE,
	.StartQueueBase = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_START_QUEUE,
	.ChCtrlBase = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_CTRL,
	.NumChannels = 2U,  /* Number of s2mm/mm2s channels */
	.ChIdxOffset = 0x8,  /* This is the offset between each channel */
	.ChStatusBase = XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_STATUS_0,
	.ChStatusOffset = 0x10,
	.PadValueBase = 0x0,
	.BdProp = &Aie2PTileDmaProp,
	.ChProp = &Aie2PDmaChProp,
	.DmaBdInit = &_XAieMl_TileDmaInit,
	.SetLock = &_XAieMl_DmaSetLock,
	.SetIntrleave = NULL,
	.SetMultiDim = &_XAieMl_DmaSetMultiDim,
	.SetBdIter = &_XAieMl_DmaSetBdIteration,
	.WriteBd = &_XAieMl_TileDmaWriteBd,
	.ReadBd = &_XAieMl_TileDmaReadBd,
	.PendingBd = &_XAieMl_DmaGetPendingBdCount,
	.WaitforDone = &_XAieMl_DmaWaitForDone,
	.WaitforBdTaskQueue = &_XAieMl_DmaWaitForBdTaskQueue,
	.BdChValidity = &_XAieMl_DmaCheckBdChValidity,
	.UpdateBdLen = &_XAieMl_DmaUpdateBdLen,
	.UpdateBdAddr = &_XAieMl_DmaUpdateBdAddr,
	.GetChannelStatus = &_XAieMl_DmaGetChannelStatus,
	.AxiBurstLenCheck = NULL,
};

static const  XAie_DmaBdEnProp Aie2PShimDmaBdEnProp =
{
	.NxtBd.Idx = 7U,
	.NxtBd.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_NEXT_BD_LSB,
	.NxtBd.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_NEXT_BD_MASK,
	.UseNxtBd.Idx = 7U,
	.UseNxtBd.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_USE_NEXT_BD_LSB,
	.UseNxtBd.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_USE_NEXT_BD_MASK,
	.ValidBd.Idx = 7U,
	.ValidBd.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_VALID_BD_LSB,
	.ValidBd.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_VALID_BD_MASK,
	.OutofOrderBdId.Idx = 2U,
	.OutofOrderBdId.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_2_OUT_OF_ORDER_BD_ID_LSB,
	.OutofOrderBdId.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_2_OUT_OF_ORDER_BD_ID_MASK,
	.TlastSuppress.Idx = 7U,
	.TlastSuppress.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_TLAST_SUPPRESS_LSB,
	.TlastSuppress.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_TLAST_SUPPRESS_MASK,
};

static const  XAie_DmaBdPkt Aie2PShimDmaBdPktProp =
{
	.EnPkt.Idx = 2U,
	.EnPkt.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_2_ENABLE_PACKET_LSB,
	.EnPkt.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_2_ENABLE_PACKET_MASK,
	.PktId.Idx = 2U,
	.PktId.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_2_PACKET_ID_LSB,
	.PktId.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_2_PACKET_ID_MASK,
	.PktType.Idx = 2U,
	.PktType.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_2_PACKET_TYPE_LSB,
	.PktType.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_2_PACKET_TYPE_MASK,
};

static const  XAie_DmaBdLock Aie2PShimDmaLockProp =
{
	.AieMlDmaLock.LckRelVal.Idx = 7U,
	.AieMlDmaLock.LckRelVal.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_LOCK_REL_VALUE_LSB,
	.AieMlDmaLock.LckRelVal.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_LOCK_REL_VALUE_MASK,
	.AieMlDmaLock.LckRelId.Idx = 7U,
	.AieMlDmaLock.LckRelId.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_LOCK_REL_ID_LSB,
	.AieMlDmaLock.LckRelId.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_LOCK_REL_ID_MASK,
	.AieMlDmaLock.LckAcqEn.Idx = 7U,
	.AieMlDmaLock.LckAcqEn.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_LOCK_ACQ_ENABLE_LSB,
	.AieMlDmaLock.LckAcqEn.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_LOCK_ACQ_ENABLE_MASK,
	.AieMlDmaLock.LckAcqVal.Idx = 7U,
	.AieMlDmaLock.LckAcqVal.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_LOCK_ACQ_VALUE_LSB,
	.AieMlDmaLock.LckAcqVal.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_LOCK_ACQ_VALUE_MASK,
	.AieMlDmaLock.LckAcqId.Idx = 7U,
	.AieMlDmaLock.LckAcqId.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_LOCK_ACQ_ID_LSB,
	.AieMlDmaLock.LckAcqId.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_7_LOCK_ACQ_ID_MASK,
};

static const  XAie_DmaBdBuffer Aie2PShimDmaBufferProp =
{
	.ShimDmaBuff.AddrLow.Idx = 1U,
	.ShimDmaBuff.AddrLow.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_1_BASE_ADDRESS_LOW_LSB,
	.ShimDmaBuff.AddrLow.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_1_BASE_ADDRESS_LOW_MASK,
	.ShimDmaBuff.AddrHigh.Idx = 2U,
	.ShimDmaBuff.AddrHigh.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_2_BASE_ADDRESS_HIGH_LSB,
	.ShimDmaBuff.AddrHigh.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_2_BASE_ADDRESS_HIGH_MASK,
};

static const  XAie_DmaBdDoubleBuffer Aie2PShimDmaDoubleBufferProp =
{
	.EnDoubleBuff = {0U},
	.BaseAddr_B = {0U},
	.FifoMode = {0U},
	.EnIntrleaved = {0U},
	.IntrleaveCnt = {0U},
	.BuffSelect = {0U},
};

static const  XAie_DmaBdMultiDimAddr Aie2PShimDmaMultiDimProp =
{
	.AieMlMultiDimAddr.DmaDimProp[0U].StepSize.Idx = 3U,
	.AieMlMultiDimAddr.DmaDimProp[0U].StepSize.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_3_D0_STEPSIZE_LSB,
	.AieMlMultiDimAddr.DmaDimProp[0U].StepSize.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_3_D0_STEPSIZE_MASK,
	.AieMlMultiDimAddr.DmaDimProp[0U].Wrap.Idx = 3U,
	.AieMlMultiDimAddr.DmaDimProp[0U].Wrap.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_3_D0_WRAP_LSB,
	.AieMlMultiDimAddr.DmaDimProp[0U].Wrap.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_3_D0_WRAP_MASK,
	.AieMlMultiDimAddr.DmaDimProp[1U].StepSize.Idx =3U ,
	.AieMlMultiDimAddr.DmaDimProp[1U].StepSize.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_4_D1_STEPSIZE_LSB,
	.AieMlMultiDimAddr.DmaDimProp[1U].StepSize.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_4_D1_STEPSIZE_MASK,
	.AieMlMultiDimAddr.DmaDimProp[1U].Wrap.Idx = 3U,
	.AieMlMultiDimAddr.DmaDimProp[1U].Wrap.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_4_D1_WRAP_LSB,
	.AieMlMultiDimAddr.DmaDimProp[1U].Wrap.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_4_D1_WRAP_MASK,
	.AieMlMultiDimAddr.DmaDimProp[2U].StepSize.Idx = 5U,
	.AieMlMultiDimAddr.DmaDimProp[2U].StepSize.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_5_D2_STEPSIZE_LSB,
	.AieMlMultiDimAddr.DmaDimProp[2U].StepSize.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_5_D2_STEPSIZE_MASK,
	.AieMlMultiDimAddr.IterCurr.Idx = 6U,
	.AieMlMultiDimAddr.IterCurr.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_6_ITERATION_CURRENT_LSB,
	.AieMlMultiDimAddr.IterCurr.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_6_ITERATION_CURRENT_MASK,
	.AieMlMultiDimAddr.Iter.Wrap.Idx = 6U,
	.AieMlMultiDimAddr.Iter.Wrap.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_6_ITERATION_WRAP_LSB,
	.AieMlMultiDimAddr.Iter.Wrap.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_6_ITERATION_WRAP_MASK,
	.AieMlMultiDimAddr.Iter.StepSize.Idx = 6U,
	.AieMlMultiDimAddr.Iter.StepSize.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_6_ITERATION_STEPSIZE_LSB,
	.AieMlMultiDimAddr.Iter.StepSize.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_6_ITERATION_STEPSIZE_MASK,
	.AieMlMultiDimAddr.DmaDimProp[2U].Wrap = {0U},
	.AieMlMultiDimAddr.DmaDimProp[3U].Wrap = {0U},
	.AieMlMultiDimAddr.DmaDimProp[3U].StepSize = {0U}
};

static const  XAie_DmaSysProp Aie2PShimDmaSysProp =
{
	.SMID.Idx = 5U,
	.SMID.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_5_SMID_LSB,
	.SMID.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_5_SMID_MASK,
	.BurstLen.Idx = 4U,
	.BurstLen.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_4_BURST_LENGTH_LSB,
	.BurstLen.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_4_BURST_LENGTH_MASK,
	.AxQos.Idx = 5U,
	.AxQos.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_5_AXQOS_LSB,
	.AxQos.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_5_AXQOS_MASK,
	.SecureAccess.Idx = 3U,
	.SecureAccess.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_3_SECURE_ACCESS_LSB,
	.SecureAccess.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_3_SECURE_ACCESS_MASK,
	.AxCache.Idx = 5U,
	.AxCache.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_5_AXCACHE_LSB,
	.AxCache.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_5_AXCACHE_MASK,
};

/* Data structure to capture register offsets and masks for Tile Dma */
static const  XAie_DmaBdProp Aie2PShimDmaProp =
{
	.AddrAlignMask = 0x3,
	.AddrAlignShift = 0U,
	.AddrMax = 0x1000000000000,
	.LenActualOffset = 0U,
	.StepSizeMax = (1U << 20) - 1U,
	.WrapMax = (1U << 10U) - 1U,
	.IterStepSizeMax = (1U << 20) - 1U,
	.IterWrapMax = (1U << 6U) - 1U,
	.IterCurrMax = (1U << 6) - 1U,
	.BufferLen.Idx = 0U,
	.BufferLen.Lsb = XAIE2PGBL_NOC_MODULE_DMA_BD0_0_BUFFER_LENGTH_LSB,
	.BufferLen.Mask = XAIE2PGBL_NOC_MODULE_DMA_BD0_0_BUFFER_LENGTH_MASK,
	.Buffer = &Aie2PShimDmaBufferProp,
	.DoubleBuffer = &Aie2PShimDmaDoubleBufferProp,
	.Lock = &Aie2PShimDmaLockProp,
	.Pkt = &Aie2PShimDmaBdPktProp,
	.BdEn = &Aie2PShimDmaBdEnProp,
	.AddrMode = &Aie2PShimDmaMultiDimProp,
	.Pad = NULL,
	.Compression = NULL,
	.SysProp = &Aie2PShimDmaSysProp
};

static const XAie_DmaChStatus Aie2PShimDmaChStatus =
{
	/* This database is common for mm2s and s2mm channels */
	.AieMlDmaChStatus.Status.Lsb = XAIE2PGBL_NOC_MODULE_DMA_S2MM_STATUS_0_STATUS_LSB,
	.AieMlDmaChStatus.Status.Mask = XAIE2PGBL_NOC_MODULE_DMA_S2MM_STATUS_0_STATUS_MASK,
	.AieMlDmaChStatus.TaskQSize.Lsb = XAIE2PGBL_NOC_MODULE_DMA_S2MM_STATUS_0_TASK_QUEUE_SIZE_LSB,
	.AieMlDmaChStatus.TaskQSize.Mask = XAIE2PGBL_NOC_MODULE_DMA_S2MM_STATUS_0_TASK_QUEUE_SIZE_MASK,
	.AieMlDmaChStatus.ChannelRunning.Lsb = XAIE2PGBL_NOC_MODULE_DMA_S2MM_STATUS_0_CHANNEL_RUNNING_LSB,
	.AieMlDmaChStatus.ChannelRunning.Mask = XAIE2PGBL_NOC_MODULE_DMA_S2MM_STATUS_0_CHANNEL_RUNNING_MASK,
	.AieMlDmaChStatus.StalledLockAcq.Lsb = XAIE2PGBL_NOC_MODULE_DMA_S2MM_STATUS_0_STALLED_LOCK_ACQ_LSB,
	.AieMlDmaChStatus.StalledLockAcq.Mask = XAIE2PGBL_NOC_MODULE_DMA_S2MM_STATUS_0_STALLED_LOCK_ACQ_MASK,
	.AieMlDmaChStatus.StalledLockRel.Lsb = XAIE2PGBL_NOC_MODULE_DMA_S2MM_STATUS_0_STALLED_LOCK_REL_LSB,
	.AieMlDmaChStatus.StalledLockRel.Mask = XAIE2PGBL_NOC_MODULE_DMA_S2MM_STATUS_0_STALLED_LOCK_REL_MASK,
	.AieMlDmaChStatus.StalledStreamStarve.Lsb = XAIE2PGBL_NOC_MODULE_DMA_S2MM_STATUS_0_STALLED_STREAM_STARVATION_LSB,
	.AieMlDmaChStatus.StalledStreamStarve.Mask = XAIE2PGBL_NOC_MODULE_DMA_S2MM_STATUS_0_STALLED_STREAM_STARVATION_MASK,
	.AieMlDmaChStatus.StalledTCT.Lsb = XAIE2PGBL_NOC_MODULE_DMA_S2MM_STATUS_0_STALLED_TCT_OR_COUNT_FIFO_FULL_LSB,
	.AieMlDmaChStatus.StalledTCT.Mask = XAIE2PGBL_NOC_MODULE_DMA_S2MM_STATUS_0_STALLED_TCT_OR_COUNT_FIFO_FULL_MASK,
};

/* Data structure to capture register offsets and masks for Mem Tile and
 * Tile Dma Channels
 */
static const  XAie_DmaChProp Aie2PShimDmaChProp =
{
	.HasFoTMode = XAIE_FEATURE_AVAILABLE,
	.HasControllerId = XAIE_FEATURE_AVAILABLE,
	.HasEnCompression = XAIE_FEATURE_AVAILABLE,
	.HasEnOutOfOrder = XAIE_FEATURE_AVAILABLE,
	.MaxFoTMode = DMA_FoT_COUNTS_FROM_MM_REG,
	.MaxRepeatCount = 256U,
	.ControllerId.Idx = 0U,
	.ControllerId.Lsb = XAIE2PGBL_NOC_MODULE_DMA_S2MM_0_CTRL_CONTROLLER_ID_LSB ,
	.ControllerId.Mask = XAIE2PGBL_NOC_MODULE_DMA_S2MM_0_CTRL_CONTROLLER_ID_MASK ,
	.EnCompression.Idx = 0U,
	.EnCompression.Lsb = 0U,
	.EnCompression.Mask = 0U,
	.EnOutofOrder.Idx = 0U,
	.EnOutofOrder.Lsb = XAIE2PGBL_NOC_MODULE_DMA_S2MM_0_CTRL_ENABLE_OUT_OF_ORDER_LSB,
	.EnOutofOrder.Mask = XAIE2PGBL_NOC_MODULE_DMA_S2MM_0_CTRL_ENABLE_OUT_OF_ORDER_MASK,
	.FoTMode.Idx = 0,
	.FoTMode.Lsb = XAIE2PGBL_NOC_MODULE_DMA_S2MM_0_CTRL_FOT_MODE_LSB,
	.FoTMode.Mask = XAIE2PGBL_NOC_MODULE_DMA_S2MM_0_CTRL_FOT_MODE_MASK ,
	.Reset.Idx = 0U,
	.Reset.Lsb = 0U,
	.Reset.Mask = 0U,
	.EnToken.Idx = 1U,
	.EnToken.Lsb = XAIE2PGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE_ENABLE_TOKEN_ISSUE_LSB,
	.EnToken.Mask = XAIE2PGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE_ENABLE_TOKEN_ISSUE_MASK,
	.RptCount.Idx = 1U,
	.RptCount.Lsb = XAIE2PGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE_REPEAT_COUNT_LSB,
	.RptCount.Mask = XAIE2PGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE_REPEAT_COUNT_MASK,
	.StartBd.Idx = 1U,
	.StartBd.Lsb = XAIE2PGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE_START_BD_ID_LSB,
	.StartBd.Mask = XAIE2PGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE_START_BD_ID_MASK,
	.PauseStream = {0U},
	.PauseMem = {0U},
	.Enable = {0U},
	.StartQSizeMax = 4U,
	.DmaChStatus = &Aie2PShimDmaChStatus,
};

/* Tile Dma Module */
static const  XAie_DmaMod Aie2PShimDmaMod =
{
	.BaseAddr = XAIE2PGBL_NOC_MODULE_DMA_BD0_0,
	.IdxOffset = 0x20,  	/* This is the offset between each BD */
	.NumBds = 16U,	   	/* Number of BDs for AIE2P Tile DMA */
	.NumLocks = 16U,
	.NumAddrDim = 3U,
	.DoubleBuffering = XAIE_FEATURE_UNAVAILABLE,
	.Compression = XAIE_FEATURE_UNAVAILABLE,
	.Padding = XAIE_FEATURE_UNAVAILABLE,
	.OutofOrderBdId = XAIE_FEATURE_AVAILABLE,
	.InterleaveMode = XAIE_FEATURE_UNAVAILABLE,
	.FifoMode = XAIE_FEATURE_UNAVAILABLE,
	.EnTokenIssue = XAIE_FEATURE_AVAILABLE,
	.RepeatCount = XAIE_FEATURE_AVAILABLE,
	.TlastSuppress = XAIE_FEATURE_AVAILABLE,
	.StartQueueBase = XAIE2PGBL_NOC_MODULE_DMA_S2MM_0_TASK_QUEUE,
	.ChCtrlBase = XAIE2PGBL_NOC_MODULE_DMA_S2MM_0_CTRL,
	.NumChannels = 2U,  /* Number of s2mm/mm2s channels */
	.ChIdxOffset = 0x8,  /* This is the offset between each channel */
	.ChStatusBase = XAIE2PGBL_NOC_MODULE_DMA_S2MM_STATUS_0,
	.ChStatusOffset = 0x8,
	.PadValueBase = 0x0,
	.BdProp = &Aie2PShimDmaProp,
	.ChProp = &Aie2PShimDmaChProp,
	.DmaBdInit = &_XAieMl_ShimDmaInit,
	.SetLock = &_XAieMl_DmaSetLock,
	.SetIntrleave = NULL,
	.SetMultiDim = &_XAieMl_DmaSetMultiDim,
	.SetBdIter = &_XAieMl_DmaSetBdIteration,
	.WriteBd = &_XAieMl_ShimDmaWriteBd,
	.ReadBd = &_XAieMl_ShimDmaReadBd,
	.PendingBd = &_XAieMl_DmaGetPendingBdCount,
	.WaitforDone = &_XAieMl_DmaWaitForDone,
	.WaitforBdTaskQueue = &_XAieMl_DmaWaitForBdTaskQueue,
	.BdChValidity = &_XAieMl_DmaCheckBdChValidity,
	.UpdateBdLen = &_XAieMl_ShimDmaUpdateBdLen,
	.UpdateBdAddr = &_XAieMl_ShimDmaUpdateBdAddr,
	.GetChannelStatus = &_XAieMl_DmaGetChannelStatus,
	.AxiBurstLenCheck = &_XAie2P_AxiBurstLenCheck,
};
#endif /* XAIE_FEATURE_DMA_ENABLE */

#ifdef XAIE_FEATURE_DATAMEM_ENABLE
/* Data Memory Module for Tile data memory*/
static const  XAie_MemMod Aie2PTileMemMod =
{
	.Size = 0x10000,
	.MemAddr = XAIE2PGBL_MEMORY_MODULE_DATAMEMORY,
	.EccEvntRegOff = XAIE2PGBL_MEMORY_MODULE_ECC_SCRUBBING_EVENT,
};

/* Data Memory Module for Mem Tile data memory*/
static const  XAie_MemMod Aie2PMemTileMemMod =
{
	.Size = 0x80000,
	.MemAddr = XAIE2PGBL_MEM_TILE_MODULE_DATAMEMORY,
	.EccEvntRegOff = XAIE2PGBL_MEM_TILE_MODULE_ECC_SCRUBBING_EVENT,
};
#endif /* XAIE_FEATURE_DATAMEM_ENABLE */

#ifdef XAIE_FEATURE_PL_ENABLE
/* Register field attributes for PL interface down sizer for 32 and 64 bits */
static const  XAie_RegFldAttr Aie2PDownSzr32_64Bit[] =
{
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH0_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH0_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH1_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH1_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH2_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH2_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH3_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH3_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH4_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH4_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH5_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH5_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH6_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH6_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH7_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH7_MASK}
};

/* Register field attributes for PL interface down sizer for 128 bits */
static const  XAie_RegFldAttr Aie2PDownSzr128Bit[] =
{
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH0_SOUTH1_128_COMBINE_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH0_SOUTH1_128_COMBINE_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH2_SOUTH3_128_COMBINE_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH2_SOUTH3_128_COMBINE_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH4_SOUTH5_128_COMBINE_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH4_SOUTH5_128_COMBINE_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH6_SOUTH7_128_COMBINE_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG_SOUTH6_SOUTH7_128_COMBINE_MASK}
};

/* Register field attributes for PL interface up sizer */
static const  XAie_RegFldAttr Aie2PUpSzr32_64Bit[] =
{
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH0_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH0_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH1_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH1_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH2_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH2_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH3_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH3_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH4_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH4_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH5_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH5_MASK}
};

/* Register field attributes for PL interface up sizer for 128 bits */
static const  XAie_RegFldAttr Aie2PUpSzr128Bit[] =
{
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH0_SOUTH1_128_COMBINE_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH0_SOUTH1_128_COMBINE_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH2_SOUTH3_128_COMBINE_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH2_SOUTH3_128_COMBINE_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH4_SOUTH5_128_COMBINE_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG_SOUTH4_SOUTH5_128_COMBINE_MASK}
};

/* Register field attributes for PL interface down sizer bypass */
static const  XAie_RegFldAttr Aie2PDownSzrByPass[] =
{
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_BYPASS_SOUTH0_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_BYPASS_SOUTH0_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_BYPASS_SOUTH1_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_BYPASS_SOUTH1_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_BYPASS_SOUTH2_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_BYPASS_SOUTH2_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_BYPASS_SOUTH4_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_BYPASS_SOUTH4_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_BYPASS_SOUTH5_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_BYPASS_SOUTH5_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_BYPASS_SOUTH6_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_BYPASS_SOUTH6_MASK}
};

/* Register field attributes for PL interface down sizer enable */
static const  XAie_RegFldAttr Aie2PDownSzrEnable[] =
{
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE_SOUTH0_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE_SOUTH0_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE_SOUTH1_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE_SOUTH1_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE_SOUTH2_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE_SOUTH2_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE_SOUTH3_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE_SOUTH3_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE_SOUTH4_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE_SOUTH4_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE_SOUTH5_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE_SOUTH5_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE_SOUTH6_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE_SOUTH6_MASK},
	{XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE_SOUTH7_LSB, XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE_SOUTH7_MASK}
};

/* Register field attributes for SHIMNOC Mux configuration */
static const  XAie_RegFldAttr Aie2PShimMuxConfig[] =
{
	{XAIE2PGBL_NOC_MODULE_MUX_CONFIG_SOUTH2_LSB, XAIE2PGBL_NOC_MODULE_MUX_CONFIG_SOUTH2_MASK},
	{XAIE2PGBL_NOC_MODULE_MUX_CONFIG_SOUTH3_LSB, XAIE2PGBL_NOC_MODULE_MUX_CONFIG_SOUTH3_MASK},
	{XAIE2PGBL_NOC_MODULE_MUX_CONFIG_SOUTH6_LSB, XAIE2PGBL_NOC_MODULE_MUX_CONFIG_SOUTH6_MASK},
	{XAIE2PGBL_NOC_MODULE_MUX_CONFIG_SOUTH7_LSB, XAIE2PGBL_NOC_MODULE_MUX_CONFIG_SOUTH7_MASK},
};

/* Register field attributes for SHIMNOC DeMux configuration */
static const  XAie_RegFldAttr Aie2PShimDeMuxConfig[] =
{
	{XAIE2PGBL_NOC_MODULE_DEMUX_CONFIG_SOUTH2_LSB, XAIE2PGBL_NOC_MODULE_DEMUX_CONFIG_SOUTH2_MASK},
	{XAIE2PGBL_NOC_MODULE_DEMUX_CONFIG_SOUTH3_LSB, XAIE2PGBL_NOC_MODULE_DEMUX_CONFIG_SOUTH3_MASK},
	{XAIE2PGBL_NOC_MODULE_DEMUX_CONFIG_SOUTH4_LSB, XAIE2PGBL_NOC_MODULE_DEMUX_CONFIG_SOUTH4_MASK},
	{XAIE2PGBL_NOC_MODULE_DEMUX_CONFIG_SOUTH5_LSB, XAIE2PGBL_NOC_MODULE_DEMUX_CONFIG_SOUTH5_MASK}
};

#ifdef XAIE_FEATURE_PRIVILEGED_ENABLE
/* Register to set SHIM clock buffer control */
static const XAie_ShimClkBufCntr Aie2PShimClkBufCntr =
{
	.RegOff = 0xFFF20,
	.RstEnable = XAIE_DISABLE,
	.ClkBufEnable = {0, 0x1}
};

static const XAie_ShimRstMod Aie2PShimTileRst =
{
	.RegOff = 0,
	.RstCntr = {0},
	.RstShims = _XAieMl_RstShims,
};

/* Register feild attributes for Shim AXI MM config for NSU Errors */
static const XAie_ShimNocAxiMMConfig Aie2PShimNocAxiMMConfig =
{
	.RegOff = XAIE2PGBL_NOC_MODULE_ME_AXIMM_CONFIG,
	.NsuSlvErr = {XAIE2PGBL_NOC_MODULE_ME_AXIMM_CONFIG_SLVERR_BLOCK_LSB, XAIE2PGBL_NOC_MODULE_ME_AXIMM_CONFIG_SLVERR_BLOCK_MASK},
	.NsuDecErr = {XAIE2PGBL_NOC_MODULE_ME_AXIMM_CONFIG_DECERR_BLOCK_LSB, XAIE2PGBL_NOC_MODULE_ME_AXIMM_CONFIG_DECERR_BLOCK_MASK}
};
#endif /* XAIE_FEATURE_PRIVILEGED_ENABLE */

/* PL Interface module for SHIMPL Tiles */
static const  XAie_PlIfMod Aie2PPlIfMod =
{
	.UpSzrOff = XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG,
	.DownSzrOff = XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG,
	.DownSzrEnOff = XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE,
	.DownSzrByPassOff = XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_BYPASS,
	.ColRstOff = 0xFFF28,
	.NumUpSzrPorts = 0x6,
	.MaxByPassPortNum = 0x6,
	.NumDownSzrPorts = 0x8,
	.UpSzr32_64Bit = Aie2PUpSzr32_64Bit,
	.UpSzr128Bit = Aie2PUpSzr128Bit,
	.DownSzr32_64Bit = Aie2PDownSzr32_64Bit,
	.DownSzr128Bit = Aie2PDownSzr128Bit,
	.DownSzrEn = Aie2PDownSzrEnable,
	.DownSzrByPass = Aie2PDownSzrByPass,
	.ShimNocMuxOff = 0x0,
	.ShimNocDeMuxOff = 0x0,
	.ShimNocMux = NULL,
	.ShimNocDeMux = NULL,
	.ColRst = {0, 0x1},
#ifdef XAIE_FEATURE_PRIVILEGED_ENABLE
	.ClkBufCntr = &Aie2PShimClkBufCntr,
	.ShimTileRst = &Aie2PShimTileRst,
#else
	.ClkBufCntr = NULL,
	.ShimTileRst = NULL,
#endif /* XAIE_FEATURE_PRIVILEGED_ENABLE */
	.ShimNocAxiMM = NULL,
};

/* PL Interface module for SHIMNOC Tiles */
static const  XAie_PlIfMod Aie2PShimTilePlIfMod =
{
	.UpSzrOff = XAIE2PGBL_PL_MODULE_PL_INTERFACE_UPSIZER_CONFIG,
	.DownSzrOff = XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_CONFIG,
	.DownSzrEnOff = XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_ENABLE,
	.DownSzrByPassOff = XAIE2PGBL_PL_MODULE_PL_INTERFACE_DOWNSIZER_BYPASS,
	.ColRstOff = 0xFFF28,
	.NumUpSzrPorts = 0x6,
	.MaxByPassPortNum = 0x6,
	.NumDownSzrPorts = 0x8,
	.UpSzr32_64Bit = Aie2PUpSzr32_64Bit,
	.UpSzr128Bit = Aie2PUpSzr128Bit,
	.DownSzr32_64Bit = Aie2PDownSzr32_64Bit,
	.DownSzr128Bit = Aie2PDownSzr128Bit,
	.DownSzrEn = Aie2PDownSzrEnable,
	.DownSzrByPass = Aie2PDownSzrByPass,
	.ShimNocMuxOff = XAIE2PGBL_NOC_MODULE_MUX_CONFIG,
	.ShimNocDeMuxOff = XAIE2PGBL_NOC_MODULE_DEMUX_CONFIG,
	.ShimNocMux = Aie2PShimMuxConfig,
	.ShimNocDeMux = Aie2PShimDeMuxConfig,
	.ColRst = {0, 0x1},
#ifdef XAIE_FEATURE_PRIVILEGED_ENABLE
	.ClkBufCntr = &Aie2PShimClkBufCntr,
	.ShimTileRst = &Aie2PShimTileRst,
	.ShimNocAxiMM = &Aie2PShimNocAxiMMConfig,
#else
	.ClkBufCntr = NULL,
	.ShimTileRst = NULL,
	.ShimNocAxiMM = NULL,
#endif /* XAIE_FEATURE_PRIVILEGED_ENABLE */
};
#endif /* XAIE_FEATURE_PL_ENABLE */

#ifdef XAIE_FEATURE_LOCK_ENABLE
static const XAie_RegFldAttr Aie2PTileLockInit =
{
	.Lsb = XAIE2PGBL_MEMORY_MODULE_LOCK0_VALUE_LOCK_VALUE_LSB,
	.Mask = XAIE2PGBL_MEMORY_MODULE_LOCK0_VALUE_LOCK_VALUE_MASK,
};

/* Lock Module for AIE Tiles  */
static const  XAie_LockMod Aie2PTileLockMod =
{
	.BaseAddr = XAIE2PGBL_MEMORY_MODULE_LOCK_REQUEST,
	.NumLocks = 16U,
	.LockIdOff = 0x400,
	.RelAcqOff = 0x200,
	.LockValOff = 0x4,
	.LockValUpperBound = 63,
	.LockValLowerBound = -64,
	.LockSetValBase = XAIE2PGBL_MEMORY_MODULE_LOCK0_VALUE,
	.LockSetValOff = 0x10,
	.LockInit = &Aie2PTileLockInit,
	.Acquire = &_XAieMl_LockAcquire,
	.Release = &_XAieMl_LockRelease,
	.SetValue = &_XAieMl_LockSetValue,
	.GetValue = &_XAieMl_LockGetValue,
};

static const XAie_RegFldAttr Aie2PShimNocLockInit =
{
	.Lsb = XAIE2PGBL_NOC_MODULE_LOCK0_VALUE_LOCK_VALUE_LSB,
	.Mask = XAIE2PGBL_NOC_MODULE_LOCK0_VALUE_LOCK_VALUE_MASK,
};

/* Lock Module for SHIM NOC Tiles  */
static const  XAie_LockMod Aie2PShimNocLockMod =
{
	.BaseAddr = XAIE2PGBL_NOC_MODULE_LOCK_REQUEST,
	.NumLocks = 16U,
	.LockIdOff = 0x400,
	.RelAcqOff = 0x200,
	.LockValOff = 0x4,
	.LockValUpperBound = 63,
	.LockValLowerBound = -64,
	.LockSetValBase = XAIE2PGBL_NOC_MODULE_LOCK0_VALUE,
	.LockSetValOff = 0x10,
	.LockInit = &Aie2PShimNocLockInit,
	.Acquire = &_XAieMl_LockAcquire,
	.Release = &_XAieMl_LockRelease,
	.SetValue = &_XAieMl_LockSetValue,
	.GetValue = &_XAieMl_LockGetValue,
};

static const XAie_RegFldAttr Aie2PMemTileLockInit =
{
	.Lsb = XAIE2PGBL_MEM_TILE_MODULE_LOCK0_VALUE_LOCK_VALUE_LSB,
	.Mask = XAIE2PGBL_MEM_TILE_MODULE_LOCK0_VALUE_LOCK_VALUE_MASK,
};

/* Lock Module for Mem Tiles  */
static const  XAie_LockMod Aie2PMemTileLockMod =
{
	.BaseAddr = XAIE2PGBL_MEM_TILE_MODULE_LOCK_REQUEST,
	.NumLocks = 64U,
	.LockIdOff = 0x400,
	.RelAcqOff = 0x200,
	.LockValOff = 0x4,
	.LockValUpperBound = 63,
	.LockValLowerBound = -64,
	.LockSetValBase = XAIE2PGBL_MEM_TILE_MODULE_LOCK0_VALUE,
	.LockSetValOff = 0x10,
	.LockInit = &Aie2PMemTileLockInit,
	.Acquire = &_XAieMl_LockAcquire,
	.Release = &_XAieMl_LockRelease,
	.SetValue = &_XAieMl_LockSetValue,
	.GetValue = &_XAieMl_LockGetValue,
};
#endif /* XAIE_FEATURE_LOCK_ENABLE */

#ifdef XAIE_FEATURE_PERFCOUNT_ENABLE
/*
 * Data structure to capture registers & offsets for Core and memory Module of
 * performance counter.
 */
static const XAie_PerfMod Aie2PTilePerfCnt[] =
{
	{	.MaxCounterVal = 2U,
		.StartStopShift = 16U,
		.ResetShift = 8U,
		.PerfCounterOffsetAdd = 0X4,
		.PerfCtrlBaseAddr = XAIE2PGBL_MEMORY_MODULE_PERFORMANCE_CONTROL0,
		.PerfCtrlOffsetAdd = 0x4,
		.PerfCtrlResetBaseAddr = XAIE2PGBL_MEMORY_MODULE_PERFORMANCE_CONTROL1,
		.PerfCounterBaseAddr = XAIE2PGBL_MEMORY_MODULE_PERFORMANCE_COUNTER0,
		.PerfCounterEvtValBaseAddr = XAIE2PGBL_MEMORY_MODULE_PERFORMANCE_COUNTER0_EVENT_VALUE,
		{XAIE2PGBL_MEMORY_MODULE_PERFORMANCE_CONTROL0_CNT0_START_EVENT_LSB, XAIE2PGBL_MEMORY_MODULE_PERFORMANCE_CONTROL0_CNT0_START_EVENT_MASK},
		{XAIE2PGBL_MEMORY_MODULE_PERFORMANCE_CONTROL0_CNT0_STOP_EVENT_LSB, XAIE2PGBL_MEMORY_MODULE_PERFORMANCE_CONTROL0_CNT0_STOP_EVENT_MASK},
		{XAIE2PGBL_MEMORY_MODULE_PERFORMANCE_CONTROL1_CNT0_RESET_EVENT_LSB,XAIE2PGBL_MEMORY_MODULE_PERFORMANCE_CONTROL1_CNT0_RESET_EVENT_MASK},
	},
	{	.MaxCounterVal = 4U,
		.StartStopShift = 16U,
		.ResetShift = 8U,
		.PerfCounterOffsetAdd = 0X4,
		.PerfCtrlBaseAddr = XAIE2PGBL_CORE_MODULE_PERFORMANCE_CONTROL0,
		.PerfCtrlOffsetAdd = 0x4,
		.PerfCtrlResetBaseAddr = XAIE2PGBL_CORE_MODULE_PERFORMANCE_CONTROL2,
		.PerfCounterBaseAddr = XAIE2PGBL_CORE_MODULE_PERFORMANCE_COUNTER0,
		.PerfCounterEvtValBaseAddr = XAIE2PGBL_CORE_MODULE_PERFORMANCE_COUNTER0_EVENT_VALUE,
		{XAIE2PGBL_CORE_MODULE_PERFORMANCE_CONTROL0_CNT0_START_EVENT_LSB, XAIE2PGBL_CORE_MODULE_PERFORMANCE_CONTROL0_CNT0_START_EVENT_MASK},
		{XAIE2PGBL_CORE_MODULE_PERFORMANCE_CONTROL0_CNT0_STOP_EVENT_LSB, XAIE2PGBL_CORE_MODULE_PERFORMANCE_CONTROL0_CNT0_STOP_EVENT_MASK},
		{XAIE2PGBL_CORE_MODULE_PERFORMANCE_CONTROL2_CNT0_RESET_EVENT_LSB, XAIE2PGBL_CORE_MODULE_PERFORMANCE_CONTROL2_CNT0_RESET_EVENT_MASK},
	}
};

/*
 * Data structure to capture registers & offsets for PL Module of performance
 * counter.
 */
static const XAie_PerfMod Aie2PPlPerfCnt =
{
	.MaxCounterVal = 2U,
	.StartStopShift = 16U,
	.ResetShift = 8U,
	.PerfCounterOffsetAdd = 0x4,
	.PerfCtrlBaseAddr = XAIE2PGBL_PL_MODULE_PERFORMANCE_CTRL0,
	.PerfCtrlOffsetAdd = 0x0,
	.PerfCtrlResetBaseAddr = XAIE2PGBL_PL_MODULE_PERFORMANCE_CTRL1,
	.PerfCounterBaseAddr = XAIE2PGBL_PL_MODULE_PERFORMANCE_COUNTER0,
	.PerfCounterEvtValBaseAddr = XAIE2PGBL_PL_MODULE_PERFORMANCE_COUNTER0_EVENT_VALUE,
	{XAIE2PGBL_PL_MODULE_PERFORMANCE_CTRL0_CNT0_START_EVENT_LSB, XAIE2PGBL_PL_MODULE_PERFORMANCE_CTRL0_CNT0_START_EVENT_MASK},
	{XAIE2PGBL_PL_MODULE_PERFORMANCE_CTRL0_CNT0_STOP_EVENT_LSB, XAIE2PGBL_PL_MODULE_PERFORMANCE_CTRL0_CNT0_STOP_EVENT_MASK},
	{XAIE2PGBL_PL_MODULE_PERFORMANCE_CTRL1_CNT0_RESET_EVENT_LSB,XAIE2PGBL_PL_MODULE_PERFORMANCE_CTRL1_CNT0_RESET_EVENT_MASK},};

/*
 * Data structure to capture registers & offsets for Mem tile Module of
 * performance counter.
 */
static const XAie_PerfMod Aie2PMemTilePerfCnt =
{
	.MaxCounterVal = 4U,
	.StartStopShift = 16U,
	.ResetShift = 8U,
	.PerfCounterOffsetAdd = 0X4,
	.PerfCtrlBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_PERFORMANCE_CONTROL0,
	.PerfCtrlOffsetAdd = 0x4,
	.PerfCtrlResetBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_PERFORMANCE_CONTROL2,
	.PerfCounterBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_PERFORMANCE_COUNTER0,
	.PerfCounterEvtValBaseAddr = XAIE2PGBL_MEM_TILE_MODULE_PERFORMANCE_COUNTER0_EVENT_VALUE,
	{XAIE2PGBL_MEM_TILE_MODULE_PERFORMANCE_CONTROL0_CNT0_START_EVENT_LSB, XAIE2PGBL_MEM_TILE_MODULE_PERFORMANCE_CONTROL0_CNT0_START_EVENT_MASK},
	{XAIE2PGBL_MEM_TILE_MODULE_PERFORMANCE_CONTROL0_CNT0_STOP_EVENT_LSB, XAIE2PGBL_MEM_TILE_MODULE_PERFORMANCE_CONTROL0_CNT0_STOP_EVENT_MASK},
	{XAIE2PGBL_MEM_TILE_MODULE_PERFORMANCE_CONTROL2_CNT0_RESET_EVENT_LSB, XAIE2PGBL_MEM_TILE_MODULE_PERFORMANCE_CONTROL2_CNT0_RESET_EVENT_MASK},
};
#endif /* XAIE_FEATURE_PERFCOUNT_ENABLE */

#ifdef XAIE_FEATURE_EVENTS_ENABLE
/* Enum to event number mapping of all events of AIE2P Core Mod of aie tile */
static const u8 Aie2PCoreModEventMapping[] =
{
	XAIE2P_EVENTS_CORE_NONE,
	XAIE2P_EVENTS_CORE_TRUE,
	XAIE2P_EVENTS_CORE_GROUP_0,
	XAIE2P_EVENTS_CORE_TIMER_SYNC,
	XAIE2P_EVENTS_CORE_TIMER_VALUE_REACHED,
	XAIE2P_EVENTS_CORE_PERF_CNT_0,
	XAIE2P_EVENTS_CORE_PERF_CNT_1,
	XAIE2P_EVENTS_CORE_PERF_CNT_2,
	XAIE2P_EVENTS_CORE_PERF_CNT_3,
	XAIE2P_EVENTS_CORE_COMBO_EVENT_0,
	XAIE2P_EVENTS_CORE_COMBO_EVENT_1,
	XAIE2P_EVENTS_CORE_COMBO_EVENT_2,
	XAIE2P_EVENTS_CORE_COMBO_EVENT_3,
	XAIE2P_EVENTS_CORE_GROUP_PC_EVENT,
	XAIE2P_EVENTS_CORE_PC_0,
	XAIE2P_EVENTS_CORE_PC_1,
	XAIE2P_EVENTS_CORE_PC_2,
	XAIE2P_EVENTS_CORE_PC_3,
	XAIE2P_EVENTS_CORE_PC_RANGE_0_1,
	XAIE2P_EVENTS_CORE_PC_RANGE_2_3,
	XAIE2P_EVENTS_CORE_GROUP_STALL,
	XAIE2P_EVENTS_CORE_MEMORY_STALL,
	XAIE2P_EVENTS_CORE_STREAM_STALL,
	XAIE2P_EVENTS_CORE_CASCADE_STALL,
	XAIE2P_EVENTS_CORE_LOCK_STALL,
	XAIE2P_EVENTS_CORE_DEBUG_HALTED,
	XAIE2P_EVENTS_CORE_ACTIVE,
	XAIE2P_EVENTS_CORE_DISABLED,
	XAIE2P_EVENTS_CORE_ECC_ERROR_STALL,
	XAIE2P_EVENTS_CORE_ECC_SCRUBBING_STALL,
	XAIE2P_EVENTS_CORE_GROUP_PROGRAM_FLOW,
	XAIE2P_EVENTS_CORE_INSTR_EVENT_0,
	XAIE2P_EVENTS_CORE_INSTR_EVENT_1,
	XAIE2P_EVENTS_CORE_INSTR_CALL,
	XAIE2P_EVENTS_CORE_INSTR_RETURN,
	XAIE2P_EVENTS_CORE_INSTR_VECTOR,
	XAIE2P_EVENTS_CORE_INSTR_LOAD,
	XAIE2P_EVENTS_CORE_INSTR_STORE,
	XAIE2P_EVENTS_CORE_INSTR_STREAM_GET,
	XAIE2P_EVENTS_CORE_INSTR_STREAM_PUT,
	XAIE2P_EVENTS_CORE_INSTR_CASCADE_GET,
	XAIE2P_EVENTS_CORE_INSTR_CASCADE_PUT,
	XAIE2P_EVENTS_CORE_INSTR_LOCK_ACQUIRE_REQ,
	XAIE2P_EVENTS_CORE_INSTR_LOCK_RELEASE_REQ,
	XAIE2P_EVENTS_CORE_GROUP_ERRORS_0,
	XAIE2P_EVENTS_CORE_GROUP_ERRORS_1,
	XAIE2P_EVENTS_CORE_SRS_OVERFLOW,
	XAIE2P_EVENTS_CORE_UPS_OVERFLOW,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_CORE_FP_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_CORE_PM_REG_ACCESS_FAILURE,
	XAIE2P_EVENTS_CORE_STREAM_PKT_PARITY_ERROR,
	XAIE2P_EVENTS_CORE_CONTROL_PKT_ERROR,
	XAIE2P_EVENTS_CORE_AXI_MM_SLAVE_ERROR,
	XAIE2P_EVENTS_CORE_INSTR_DECOMPRSN_ERROR,
	XAIE2P_EVENTS_CORE_DM_ADDRESS_OUT_OF_RANGE,
	XAIE2P_EVENTS_CORE_PM_ECC_ERROR_SCRUB_CORRECTED,
	XAIE2P_EVENTS_CORE_PM_ECC_ERROR_SCRUB_2BIT,
	XAIE2P_EVENTS_CORE_PM_ECC_ERROR_1BIT,
	XAIE2P_EVENTS_CORE_PM_ECC_ERROR_2BIT,
	XAIE2P_EVENTS_CORE_PM_ADDRESS_OUT_OF_RANGE,
	XAIE2P_EVENTS_CORE_DM_ACCESS_TO_UNAVAILABLE,
	XAIE2P_EVENTS_CORE_LOCK_ACCESS_TO_UNAVAILABLE,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_CORE_GROUP_STREAM_SWITCH,
	XAIE2P_EVENTS_CORE_PORT_IDLE_0,
	XAIE2P_EVENTS_CORE_PORT_RUNNING_0,
	XAIE2P_EVENTS_CORE_PORT_STALLED_0,
	XAIE2P_EVENTS_CORE_PORT_TLAST_0,
	XAIE2P_EVENTS_CORE_PORT_IDLE_1,
	XAIE2P_EVENTS_CORE_PORT_RUNNING_1,
	XAIE2P_EVENTS_CORE_PORT_STALLED_1,
	XAIE2P_EVENTS_CORE_PORT_TLAST_1,
	XAIE2P_EVENTS_CORE_PORT_IDLE_2,
	XAIE2P_EVENTS_CORE_PORT_RUNNING_2,
	XAIE2P_EVENTS_CORE_PORT_STALLED_2,
	XAIE2P_EVENTS_CORE_PORT_TLAST_2,
	XAIE2P_EVENTS_CORE_PORT_IDLE_3,
	XAIE2P_EVENTS_CORE_PORT_RUNNING_3,
	XAIE2P_EVENTS_CORE_PORT_STALLED_3,
	XAIE2P_EVENTS_CORE_PORT_TLAST_3,
	XAIE2P_EVENTS_CORE_PORT_IDLE_4,
	XAIE2P_EVENTS_CORE_PORT_RUNNING_4,
	XAIE2P_EVENTS_CORE_PORT_STALLED_4,
	XAIE2P_EVENTS_CORE_PORT_TLAST_4,
	XAIE2P_EVENTS_CORE_PORT_IDLE_5,
	XAIE2P_EVENTS_CORE_PORT_RUNNING_5,
	XAIE2P_EVENTS_CORE_PORT_STALLED_5,
	XAIE2P_EVENTS_CORE_PORT_TLAST_5,
	XAIE2P_EVENTS_CORE_PORT_IDLE_6,
	XAIE2P_EVENTS_CORE_PORT_RUNNING_6,
	XAIE2P_EVENTS_CORE_PORT_STALLED_6,
	XAIE2P_EVENTS_CORE_PORT_TLAST_6,
	XAIE2P_EVENTS_CORE_PORT_IDLE_7,
	XAIE2P_EVENTS_CORE_PORT_RUNNING_7,
	XAIE2P_EVENTS_CORE_PORT_STALLED_7,
	XAIE2P_EVENTS_CORE_PORT_TLAST_7,
	XAIE2P_EVENTS_CORE_GROUP_BROADCAST,
	XAIE2P_EVENTS_CORE_BROADCAST_0,
	XAIE2P_EVENTS_CORE_BROADCAST_1,
	XAIE2P_EVENTS_CORE_BROADCAST_2,
	XAIE2P_EVENTS_CORE_BROADCAST_3,
	XAIE2P_EVENTS_CORE_BROADCAST_4,
	XAIE2P_EVENTS_CORE_BROADCAST_5,
	XAIE2P_EVENTS_CORE_BROADCAST_6,
	XAIE2P_EVENTS_CORE_BROADCAST_7,
	XAIE2P_EVENTS_CORE_BROADCAST_8,
	XAIE2P_EVENTS_CORE_BROADCAST_9,
	XAIE2P_EVENTS_CORE_BROADCAST_10,
	XAIE2P_EVENTS_CORE_BROADCAST_11,
	XAIE2P_EVENTS_CORE_BROADCAST_12,
	XAIE2P_EVENTS_CORE_BROADCAST_13,
	XAIE2P_EVENTS_CORE_BROADCAST_14,
	XAIE2P_EVENTS_CORE_BROADCAST_15,
	XAIE2P_EVENTS_CORE_GROUP_USER_EVENT,
	XAIE2P_EVENTS_CORE_USER_EVENT_0,
	XAIE2P_EVENTS_CORE_USER_EVENT_1,
	XAIE2P_EVENTS_CORE_USER_EVENT_2,
	XAIE2P_EVENTS_CORE_USER_EVENT_3,
	XAIE2P_EVENTS_CORE_EDGE_DETECTION_EVENT_0,
	XAIE2P_EVENTS_CORE_EDGE_DETECTION_EVENT_1,
	XAIE2P_EVENTS_CORE_FP_HUGE,
	XAIE2P_EVENTS_CORE_INT_FP_0,
	XAIE2P_EVENTS_CORE_FP_INF,
	XAIE2P_EVENTS_CORE_INSTR_WARNING,
	XAIE2P_EVENTS_CORE_INSTR_ERROR,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_CORE_STREAM_SWITCH_PORT_PARITY_ERROR,
	XAIE2P_EVENTS_CORE_PROCESSOR_BUS_ERROR,
	XAIE2P_EVENTS_CORE_SPARSITY_OVERFLOW,
};

/* Enum to event number mapping of all events of AIE2P Mem Mod of aie tile */
static const u8 Aie2PMemModEventMapping[] =
{
	XAIE2P_EVENTS_MEM_NONE,
	XAIE2P_EVENTS_MEM_TRUE,
	XAIE2P_EVENTS_MEM_GROUP_0,
	XAIE2P_EVENTS_MEM_TIMER_SYNC,
	XAIE2P_EVENTS_MEM_TIMER_VALUE_REACHED,
	XAIE2P_EVENTS_MEM_PERF_CNT_0,
	XAIE2P_EVENTS_MEM_PERF_CNT_1,
	XAIE2P_EVENTS_MEM_COMBO_EVENT_0,
	XAIE2P_EVENTS_MEM_COMBO_EVENT_1,
	XAIE2P_EVENTS_MEM_COMBO_EVENT_2,
	XAIE2P_EVENTS_MEM_COMBO_EVENT_3,
	XAIE2P_EVENTS_MEM_GROUP_WATCHPOINT,
	XAIE2P_EVENTS_MEM_WATCHPOINT_0,
	XAIE2P_EVENTS_MEM_WATCHPOINT_1,
	XAIE2P_EVENTS_MEM_GROUP_DMA_ACTIVITY,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_MEM_DMA_S2MM_0_FINISHED_BD,
	XAIE2P_EVENTS_MEM_DMA_S2MM_1_FINISHED_BD,
	XAIE2P_EVENTS_MEM_DMA_MM2S_0_FINISHED_BD,
	XAIE2P_EVENTS_MEM_DMA_MM2S_1_FINISHED_BD,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_MEM_GROUP_LOCK,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_MEM_LOCK_0_REL,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_MEM_LOCK_1_REL,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_MEM_LOCK_2_REL,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_MEM_LOCK_3_REL,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_MEM_LOCK_4_REL,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_MEM_LOCK_5_REL,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_MEM_LOCK_6_REL,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_MEM_LOCK_7_REL,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_MEM_GROUP_MEMORY_CONFLICT,
	XAIE2P_EVENTS_MEM_CONFLICT_DM_BANK_0,
	XAIE2P_EVENTS_MEM_CONFLICT_DM_BANK_1,
	XAIE2P_EVENTS_MEM_CONFLICT_DM_BANK_2,
	XAIE2P_EVENTS_MEM_CONFLICT_DM_BANK_3,
	XAIE2P_EVENTS_MEM_CONFLICT_DM_BANK_4,
	XAIE2P_EVENTS_MEM_CONFLICT_DM_BANK_5,
	XAIE2P_EVENTS_MEM_CONFLICT_DM_BANK_6,
	XAIE2P_EVENTS_MEM_CONFLICT_DM_BANK_7,
	XAIE2P_EVENTS_MEM_GROUP_ERRORS,
	XAIE2P_EVENTS_MEM_DM_ECC_ERROR_SCRUB_CORRECTED,
	XAIE2P_EVENTS_MEM_DM_ECC_ERROR_SCRUB_2BIT,
	XAIE2P_EVENTS_MEM_DM_ECC_ERROR_1BIT,
	XAIE2P_EVENTS_MEM_DM_ECC_ERROR_2BIT,
	XAIE2P_EVENTS_MEM_DM_PARITY_ERROR_BANK_2,
	XAIE2P_EVENTS_MEM_DM_PARITY_ERROR_BANK_3,
	XAIE2P_EVENTS_MEM_DM_PARITY_ERROR_BANK_4,
	XAIE2P_EVENTS_MEM_DM_PARITY_ERROR_BANK_5,
	XAIE2P_EVENTS_MEM_DM_PARITY_ERROR_BANK_6,
	XAIE2P_EVENTS_MEM_DM_PARITY_ERROR_BANK_7,
	XAIE2P_EVENTS_MEM_DMA_S2MM_0_ERROR,
	XAIE2P_EVENTS_MEM_DMA_S2MM_1_ERROR,
	XAIE2P_EVENTS_MEM_DMA_MM2S_0_ERROR,
	XAIE2P_EVENTS_MEM_DMA_MM2S_1_ERROR,
	XAIE2P_EVENTS_MEM_GROUP_BROADCAST,
	XAIE2P_EVENTS_MEM_BROADCAST_0,
	XAIE2P_EVENTS_MEM_BROADCAST_1,
	XAIE2P_EVENTS_MEM_BROADCAST_2,
	XAIE2P_EVENTS_MEM_BROADCAST_3,
	XAIE2P_EVENTS_MEM_BROADCAST_4,
	XAIE2P_EVENTS_MEM_BROADCAST_5,
	XAIE2P_EVENTS_MEM_BROADCAST_6,
	XAIE2P_EVENTS_MEM_BROADCAST_7,
	XAIE2P_EVENTS_MEM_BROADCAST_8,
	XAIE2P_EVENTS_MEM_BROADCAST_9,
	XAIE2P_EVENTS_MEM_BROADCAST_10,
	XAIE2P_EVENTS_MEM_BROADCAST_11,
	XAIE2P_EVENTS_MEM_BROADCAST_12,
	XAIE2P_EVENTS_MEM_BROADCAST_13,
	XAIE2P_EVENTS_MEM_BROADCAST_14,
	XAIE2P_EVENTS_MEM_BROADCAST_15,
	XAIE2P_EVENTS_MEM_GROUP_USER_EVENT,
	XAIE2P_EVENTS_MEM_USER_EVENT_0,
	XAIE2P_EVENTS_MEM_USER_EVENT_1,
	XAIE2P_EVENTS_MEM_USER_EVENT_2,
	XAIE2P_EVENTS_MEM_USER_EVENT_3,
	XAIE2P_EVENTS_MEM_EDGE_DETECTION_EVENT_0,
	XAIE2P_EVENTS_MEM_EDGE_DETECTION_EVENT_1,
	XAIE2P_EVENTS_MEM_DMA_S2MM_0_START_TASK,
	XAIE2P_EVENTS_MEM_DMA_S2MM_1_START_TASK,
	XAIE2P_EVENTS_MEM_DMA_MM2S_0_START_TASK,
	XAIE2P_EVENTS_MEM_DMA_MM2S_1_START_TASK,
	XAIE2P_EVENTS_MEM_DMA_S2MM_0_FINISHED_TASK,
	XAIE2P_EVENTS_MEM_DMA_S2MM_1_FINISHED_TASK,
	XAIE2P_EVENTS_MEM_DMA_MM2S_0_FINISHED_TASK,
	XAIE2P_EVENTS_MEM_DMA_MM2S_1_FINISHED_TASK,
	XAIE2P_EVENTS_MEM_DMA_S2MM_0_STALLED_LOCK,
	XAIE2P_EVENTS_MEM_DMA_S2MM_1_STALLED_LOCK,
	XAIE2P_EVENTS_MEM_DMA_MM2S_0_STALLED_LOCK,
	XAIE2P_EVENTS_MEM_DMA_MM2S_1_STALLED_LOCK,
	XAIE2P_EVENTS_MEM_DMA_S2MM_0_STREAM_STARVATION,
	XAIE2P_EVENTS_MEM_DMA_S2MM_1_STREAM_STARVATION,
	XAIE2P_EVENTS_MEM_DMA_MM2S_0_STREAM_BACKPRESSURE,
	XAIE2P_EVENTS_MEM_DMA_MM2S_1_STREAM_BACKPRESSURE,
	XAIE2P_EVENTS_MEM_DMA_S2MM_0_MEMORY_BACKPRESSURE,
	XAIE2P_EVENTS_MEM_DMA_S2MM_1_MEMORY_BACKPRESSURE,
	XAIE2P_EVENTS_MEM_DMA_MM2S_0_MEMORY_STARVATION,
	XAIE2P_EVENTS_MEM_DMA_MM2S_1_MEMORY_STARVATION,
	XAIE2P_EVENTS_MEM_LOCK_SEL0_ACQ_EQ,
	XAIE2P_EVENTS_MEM_LOCK_SEL0_ACQ_GE,
	XAIE2P_EVENTS_MEM_LOCK_SEL0_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_MEM_LOCK_SEL1_ACQ_EQ,
	XAIE2P_EVENTS_MEM_LOCK_SEL1_ACQ_GE,
	XAIE2P_EVENTS_MEM_LOCK_SEL1_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_MEM_LOCK_SEL2_ACQ_EQ,
	XAIE2P_EVENTS_MEM_LOCK_SEL2_ACQ_GE,
	XAIE2P_EVENTS_MEM_LOCK_SEL2_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_MEM_LOCK_SEL3_ACQ_EQ,
	XAIE2P_EVENTS_MEM_LOCK_SEL3_ACQ_GE,
	XAIE2P_EVENTS_MEM_LOCK_SEL3_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_MEM_LOCK_SEL4_ACQ_EQ,
	XAIE2P_EVENTS_MEM_LOCK_SEL4_ACQ_GE,
	XAIE2P_EVENTS_MEM_LOCK_SEL4_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_MEM_LOCK_SEL5_ACQ_EQ,
	XAIE2P_EVENTS_MEM_LOCK_SEL5_ACQ_GE,
	XAIE2P_EVENTS_MEM_LOCK_SEL5_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_MEM_LOCK_SEL6_ACQ_EQ,
	XAIE2P_EVENTS_MEM_LOCK_SEL6_ACQ_GE,
	XAIE2P_EVENTS_MEM_LOCK_SEL6_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_MEM_LOCK_SEL7_ACQ_EQ,
	XAIE2P_EVENTS_MEM_LOCK_SEL7_ACQ_GE,
	XAIE2P_EVENTS_MEM_LOCK_SEL7_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_MEM_LOCK_ERROR,
	XAIE2P_EVENTS_MEM_DMA_TASK_TOKEN_STALL,
};

/* Enum to event number mapping of all events of AIE2P NOC tile */
static const u8 Aie2PNocModEventMapping[] =
{
	XAIE2P_EVENTS_PL_NONE,
	XAIE2P_EVENTS_PL_TRUE,
	XAIE2P_EVENTS_PL_GROUP_0,
	XAIE2P_EVENTS_PL_TIMER_SYNC,
	XAIE2P_EVENTS_PL_TIMER_VALUE_REACHED,
	XAIE2P_EVENTS_PL_PERF_CNT_0,
	XAIE2P_EVENTS_PL_PERF_CNT_1,
	XAIE2P_EVENTS_PL_COMBO_EVENT_0,
	XAIE2P_EVENTS_PL_COMBO_EVENT_1,
	XAIE2P_EVENTS_PL_COMBO_EVENT_2,
	XAIE2P_EVENTS_PL_COMBO_EVENT_3,
	XAIE2P_EVENTS_PL_GROUP_DMA_ACTIVITY,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_DMA_S2MM_0_FINISHED_BD,
	XAIE2P_EVENTS_PL_DMA_S2MM_1_FINISHED_BD,
	XAIE2P_EVENTS_PL_DMA_MM2S_0_FINISHED_BD,
	XAIE2P_EVENTS_PL_DMA_MM2S_1_FINISHED_BD,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_GROUP_LOCK,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_LOCK_0_REL,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_LOCK_1_REL,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_LOCK_2_REL,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_LOCK_3_REL,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_LOCK_4_REL,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_LOCK_5_REL,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_GROUP_ERRORS,
	XAIE2P_EVENTS_PL_AXI_MM_SLAVE_ERROR,
	XAIE2P_EVENTS_PL_CONTROL_PKT_ERROR,
	XAIE2P_EVENTS_PL_AXI_MM_DECODE_NSU_ERROR,
	XAIE2P_EVENTS_PL_AXI_MM_SLAVE_NSU_ERROR,
	XAIE2P_EVENTS_PL_AXI_MM_UNSUPPORTED_TRAFFIC,
	XAIE2P_EVENTS_PL_AXI_MM_UNSECURE_ACCESS_IN_SECURE_MODE,
	XAIE2P_EVENTS_PL_AXI_MM_BYTE_STROBE_ERROR,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_GROUP_STREAM_SWITCH,
	XAIE2P_EVENTS_PL_PORT_IDLE_0,
	XAIE2P_EVENTS_PL_PORT_RUNNING_0,
	XAIE2P_EVENTS_PL_PORT_STALLED_0,
	XAIE2P_EVENTS_PL_PORT_TLAST_0,
	XAIE2P_EVENTS_PL_PORT_IDLE_1,
	XAIE2P_EVENTS_PL_PORT_RUNNING_1,
	XAIE2P_EVENTS_PL_PORT_STALLED_1,
	XAIE2P_EVENTS_PL_PORT_TLAST_1,
	XAIE2P_EVENTS_PL_PORT_IDLE_2,
	XAIE2P_EVENTS_PL_PORT_RUNNING_2,
	XAIE2P_EVENTS_PL_PORT_STALLED_2,
	XAIE2P_EVENTS_PL_PORT_TLAST_2,
	XAIE2P_EVENTS_PL_PORT_IDLE_3,
	XAIE2P_EVENTS_PL_PORT_RUNNING_3,
	XAIE2P_EVENTS_PL_PORT_STALLED_3,
	XAIE2P_EVENTS_PL_PORT_TLAST_3,
	XAIE2P_EVENTS_PL_PORT_IDLE_4,
	XAIE2P_EVENTS_PL_PORT_RUNNING_4,
	XAIE2P_EVENTS_PL_PORT_STALLED_4,
	XAIE2P_EVENTS_PL_PORT_TLAST_4,
	XAIE2P_EVENTS_PL_PORT_IDLE_5,
	XAIE2P_EVENTS_PL_PORT_RUNNING_5,
	XAIE2P_EVENTS_PL_PORT_STALLED_5,
	XAIE2P_EVENTS_PL_PORT_TLAST_5,
	XAIE2P_EVENTS_PL_PORT_IDLE_6,
	XAIE2P_EVENTS_PL_PORT_RUNNING_6,
	XAIE2P_EVENTS_PL_PORT_STALLED_6,
	XAIE2P_EVENTS_PL_PORT_TLAST_6,
	XAIE2P_EVENTS_PL_PORT_IDLE_7,
	XAIE2P_EVENTS_PL_PORT_RUNNING_7,
	XAIE2P_EVENTS_PL_PORT_STALLED_7,
	XAIE2P_EVENTS_PL_PORT_TLAST_7,
	XAIE2P_EVENTS_PL_GROUP_BROADCAST_A,
	XAIE2P_EVENTS_PL_BROADCAST_A_0,
	XAIE2P_EVENTS_PL_BROADCAST_A_1,
	XAIE2P_EVENTS_PL_BROADCAST_A_2,
	XAIE2P_EVENTS_PL_BROADCAST_A_3,
	XAIE2P_EVENTS_PL_BROADCAST_A_4,
	XAIE2P_EVENTS_PL_BROADCAST_A_5,
	XAIE2P_EVENTS_PL_BROADCAST_A_6,
	XAIE2P_EVENTS_PL_BROADCAST_A_7,
	XAIE2P_EVENTS_PL_BROADCAST_A_8,
	XAIE2P_EVENTS_PL_BROADCAST_A_9,
	XAIE2P_EVENTS_PL_BROADCAST_A_10,
	XAIE2P_EVENTS_PL_BROADCAST_A_11,
	XAIE2P_EVENTS_PL_BROADCAST_A_12,
	XAIE2P_EVENTS_PL_BROADCAST_A_13,
	XAIE2P_EVENTS_PL_BROADCAST_A_14,
	XAIE2P_EVENTS_PL_BROADCAST_A_15,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_USER_EVENT_0,
	XAIE2P_EVENTS_PL_USER_EVENT_1,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_EDGE_DETECTION_EVENT_0,
	XAIE2P_EVENTS_PL_EDGE_DETECTION_EVENT_1,
	XAIE2P_EVENTS_PL_DMA_S2MM_0_START_TASK,
	XAIE2P_EVENTS_PL_DMA_S2MM_1_START_TASK,
	XAIE2P_EVENTS_PL_DMA_MM2S_0_START_TASK,
	XAIE2P_EVENTS_PL_DMA_MM2S_1_START_TASK,
	XAIE2P_EVENTS_PL_DMA_S2MM_0_FINISHED_TASK,
	XAIE2P_EVENTS_PL_DMA_S2MM_1_FINISHED_TASK,
	XAIE2P_EVENTS_PL_DMA_MM2S_0_FINISHED_TASK,
	XAIE2P_EVENTS_PL_DMA_MM2S_1_FINISHED_TASK,
	XAIE2P_EVENTS_PL_DMA_S2MM_0_STALLED_LOCK,
	XAIE2P_EVENTS_PL_DMA_S2MM_1_STALLED_LOCK,
	XAIE2P_EVENTS_PL_DMA_MM2S_0_STALLED_LOCK,
	XAIE2P_EVENTS_PL_DMA_MM2S_1_STALLED_LOCK,
	XAIE2P_EVENTS_PL_DMA_S2MM_0_STREAM_STARVATION,
	XAIE2P_EVENTS_PL_DMA_S2MM_1_STREAM_STARVATION,
	XAIE2P_EVENTS_PL_DMA_MM2S_0_STREAM_BACKPRESSURE,
	XAIE2P_EVENTS_PL_DMA_MM2S_1_STREAM_BACKPRESSURE,
	XAIE2P_EVENTS_PL_DMA_S2MM_0_MEMORY_BACKPRESSURE,
	XAIE2P_EVENTS_PL_DMA_S2MM_1_MEMORY_BACKPRESSURE,
	XAIE2P_EVENTS_PL_DMA_MM2S_0_MEMORY_STARVATION,
	XAIE2P_EVENTS_PL_DMA_MM2S_1_MEMORY_STARVATION,
	XAIE2P_EVENTS_PL_LOCK_0_ACQ_EQ,
	XAIE2P_EVENTS_PL_LOCK_0_ACQ_GE,
	XAIE2P_EVENTS_PL_LOCK_0_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_PL_LOCK_1_ACQ_EQ,
	XAIE2P_EVENTS_PL_LOCK_1_ACQ_GE,
	XAIE2P_EVENTS_PL_LOCK_1_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_PL_LOCK_2_ACQ_EQ,
	XAIE2P_EVENTS_PL_LOCK_2_ACQ_GE,
	XAIE2P_EVENTS_PL_LOCK_2_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_PL_LOCK_3_ACQ_EQ,
	XAIE2P_EVENTS_PL_LOCK_3_ACQ_GE,
	XAIE2P_EVENTS_PL_LOCK_3_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_PL_LOCK_4_ACQ_EQ,
	XAIE2P_EVENTS_PL_LOCK_4_ACQ_GE,
	XAIE2P_EVENTS_PL_LOCK_4_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_PL_LOCK_5_ACQ_EQ,
	XAIE2P_EVENTS_PL_LOCK_5_ACQ_GE,
	XAIE2P_EVENTS_PL_LOCK_5_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_PL_STREAM_SWITCH_PARITY_ERROR,
	XAIE2P_EVENTS_PL_DMA_S2MM_ERROR,
	XAIE2P_EVENTS_PL_DMA_MM2S_ERROR,
	XAIE2P_EVENTS_PL_LOCK_ERROR,
	XAIE2P_EVENTS_PL_DMA_TASK_TOKEN_STALL,
};

/* Enum to event number mapping of all events of AIE2P PL Module */
static const u8 Aie2PPlModEventMapping[] =
{
	XAIE2P_EVENTS_PL_NONE,
	XAIE2P_EVENTS_PL_TRUE,
	XAIE2P_EVENTS_PL_GROUP_0,
	XAIE2P_EVENTS_PL_TIMER_SYNC,
	XAIE2P_EVENTS_PL_TIMER_VALUE_REACHED,
	XAIE2P_EVENTS_PL_PERF_CNT_0,
	XAIE2P_EVENTS_PL_PERF_CNT_1,
	XAIE2P_EVENTS_PL_COMBO_EVENT_0,
	XAIE2P_EVENTS_PL_COMBO_EVENT_1,
	XAIE2P_EVENTS_PL_COMBO_EVENT_2,
	XAIE2P_EVENTS_PL_COMBO_EVENT_3,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_GROUP_ERRORS,
	XAIE2P_EVENTS_PL_AXI_MM_SLAVE_ERROR,
	XAIE2P_EVENTS_PL_CONTROL_PKT_ERROR,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_GROUP_STREAM_SWITCH,
	XAIE2P_EVENTS_PL_PORT_IDLE_0,
	XAIE2P_EVENTS_PL_PORT_RUNNING_0,
	XAIE2P_EVENTS_PL_PORT_STALLED_0,
	XAIE2P_EVENTS_PL_PORT_TLAST_0,
	XAIE2P_EVENTS_PL_PORT_IDLE_1,
	XAIE2P_EVENTS_PL_PORT_RUNNING_1,
	XAIE2P_EVENTS_PL_PORT_STALLED_1,
	XAIE2P_EVENTS_PL_PORT_TLAST_1,
	XAIE2P_EVENTS_PL_PORT_IDLE_2,
	XAIE2P_EVENTS_PL_PORT_RUNNING_2,
	XAIE2P_EVENTS_PL_PORT_STALLED_2,
	XAIE2P_EVENTS_PL_PORT_TLAST_2,
	XAIE2P_EVENTS_PL_PORT_IDLE_3,
	XAIE2P_EVENTS_PL_PORT_RUNNING_3,
	XAIE2P_EVENTS_PL_PORT_STALLED_3,
	XAIE2P_EVENTS_PL_PORT_TLAST_3,
	XAIE2P_EVENTS_PL_PORT_IDLE_4,
	XAIE2P_EVENTS_PL_PORT_RUNNING_4,
	XAIE2P_EVENTS_PL_PORT_STALLED_4,
	XAIE2P_EVENTS_PL_PORT_TLAST_4,
	XAIE2P_EVENTS_PL_PORT_IDLE_5,
	XAIE2P_EVENTS_PL_PORT_RUNNING_5,
	XAIE2P_EVENTS_PL_PORT_STALLED_5,
	XAIE2P_EVENTS_PL_PORT_TLAST_5,
	XAIE2P_EVENTS_PL_PORT_IDLE_6,
	XAIE2P_EVENTS_PL_PORT_RUNNING_6,
	XAIE2P_EVENTS_PL_PORT_STALLED_6,
	XAIE2P_EVENTS_PL_PORT_TLAST_6,
	XAIE2P_EVENTS_PL_PORT_IDLE_7,
	XAIE2P_EVENTS_PL_PORT_RUNNING_7,
	XAIE2P_EVENTS_PL_PORT_STALLED_7,
	XAIE2P_EVENTS_PL_PORT_TLAST_7,
	XAIE2P_EVENTS_PL_GROUP_BROADCAST_A,
	XAIE2P_EVENTS_PL_BROADCAST_A_0,
	XAIE2P_EVENTS_PL_BROADCAST_A_1,
	XAIE2P_EVENTS_PL_BROADCAST_A_2,
	XAIE2P_EVENTS_PL_BROADCAST_A_3,
	XAIE2P_EVENTS_PL_BROADCAST_A_4,
	XAIE2P_EVENTS_PL_BROADCAST_A_5,
	XAIE2P_EVENTS_PL_BROADCAST_A_6,
	XAIE2P_EVENTS_PL_BROADCAST_A_7,
	XAIE2P_EVENTS_PL_BROADCAST_A_8,
	XAIE2P_EVENTS_PL_BROADCAST_A_9,
	XAIE2P_EVENTS_PL_BROADCAST_A_10,
	XAIE2P_EVENTS_PL_BROADCAST_A_11,
	XAIE2P_EVENTS_PL_BROADCAST_A_12,
	XAIE2P_EVENTS_PL_BROADCAST_A_13,
	XAIE2P_EVENTS_PL_BROADCAST_A_14,
	XAIE2P_EVENTS_PL_BROADCAST_A_15,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_USER_EVENT_0,
	XAIE2P_EVENTS_PL_USER_EVENT_1,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_EDGE_DETECTION_EVENT_0,
	XAIE2P_EVENTS_PL_EDGE_DETECTION_EVENT_1,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE2P_EVENTS_PL_STREAM_SWITCH_PARITY_ERROR,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
	XAIE_EVENT_INVALID,
};

/* Enum to event number mapping of all events of AIE2P Mem Tile Module */
static const u8 Aie2PMemTileModEventMapping[] =
{
	XAIE2P_EVENTS_MEM_TILE_NONE,
	XAIE2P_EVENTS_MEM_TILE_TRUE,
	XAIE2P_EVENTS_MEM_TILE_GROUP_0,
	XAIE2P_EVENTS_MEM_TILE_TIMER_SYNC,
	XAIE2P_EVENTS_MEM_TILE_TIMER_VALUE_REACHED,
	XAIE2P_EVENTS_MEM_TILE_PERF_CNT0_EVENT,
	XAIE2P_EVENTS_MEM_TILE_PERF_CNT1_EVENT,
	XAIE2P_EVENTS_MEM_TILE_PERF_CNT2_EVENT,
	XAIE2P_EVENTS_MEM_TILE_PERF_CNT3_EVENT,
	XAIE2P_EVENTS_MEM_TILE_COMBO_EVENT_0,
	XAIE2P_EVENTS_MEM_TILE_COMBO_EVENT_1,
	XAIE2P_EVENTS_MEM_TILE_COMBO_EVENT_2,
	XAIE2P_EVENTS_MEM_TILE_COMBO_EVENT_3,
	XAIE2P_EVENTS_MEM_TILE_EDGE_DETECTION_EVENT_0,
	XAIE2P_EVENTS_MEM_TILE_EDGE_DETECTION_EVENT_1,
	XAIE2P_EVENTS_MEM_TILE_GROUP_WATCHPOINT,
	XAIE2P_EVENTS_MEM_TILE_WATCHPOINT_0,
	XAIE2P_EVENTS_MEM_TILE_WATCHPOINT_1,
	XAIE2P_EVENTS_MEM_TILE_WATCHPOINT_2,
	XAIE2P_EVENTS_MEM_TILE_WATCHPOINT_3,
	XAIE2P_EVENTS_MEM_TILE_GROUP_DMA_ACTIVITY,
	XAIE2P_EVENTS_MEM_TILE_DMA_S2MM_SEL0_START_TASK,
	XAIE2P_EVENTS_MEM_TILE_DMA_S2MM_SEL1_START_TASK,
	XAIE2P_EVENTS_MEM_TILE_DMA_MM2S_SEL0_START_TASK,
	XAIE2P_EVENTS_MEM_TILE_DMA_MM2S_SEL1_START_TASK,
	XAIE2P_EVENTS_MEM_TILE_DMA_S2MM_SEL0_FINISHED_BD,
	XAIE2P_EVENTS_MEM_TILE_DMA_S2MM_SEL1_FINISHED_BD,
	XAIE2P_EVENTS_MEM_TILE_DMA_MM2S_SEL0_FINISHED_BD,
	XAIE2P_EVENTS_MEM_TILE_DMA_MM2S_SEL1_FINISHED_BD,
	XAIE2P_EVENTS_MEM_TILE_DMA_S2MM_SEL0_FINISHED_TASK,
	XAIE2P_EVENTS_MEM_TILE_DMA_S2MM_SEL1_FINISHED_TASK,
	XAIE2P_EVENTS_MEM_TILE_DMA_MM2S_SEL0_FINISHED_TASK,
	XAIE2P_EVENTS_MEM_TILE_DMA_MM2S_SEL1_FINISHED_TASK,
	XAIE2P_EVENTS_MEM_TILE_DMA_S2MM_SEL0_STALLED_LOCK,
	XAIE2P_EVENTS_MEM_TILE_DMA_S2MM_SEL1_STALLED_LOCK,
	XAIE2P_EVENTS_MEM_TILE_DMA_MM2S_SEL0_STALLED_LOCK,
	XAIE2P_EVENTS_MEM_TILE_DMA_MM2S_SEL1_STALLED_LOCK,
	XAIE2P_EVENTS_MEM_TILE_DMA_S2MM_SEL0_STREAM_STARVATION,
	XAIE2P_EVENTS_MEM_TILE_DMA_S2MM_SEL1_STREAM_STARVATION,
	XAIE2P_EVENTS_MEM_TILE_DMA_MM2S_SEL0_STREAM_BACKPRESSURE,
	XAIE2P_EVENTS_MEM_TILE_DMA_MM2S_SEL1_STREAM_BACKPRESSURE,
	XAIE2P_EVENTS_MEM_TILE_DMA_S2MM_SEL0_MEMORY_BACKPRESSURE,
	XAIE2P_EVENTS_MEM_TILE_DMA_S2MM_SEL1_MEMORY_BACKPRESSURE,
	XAIE2P_EVENTS_MEM_TILE_DMA_MM2S_SEL0_MEMORY_STARVATION,
	XAIE2P_EVENTS_MEM_TILE_DMA_MM2S_SEL1_MEMORY_STARVATION,
	XAIE2P_EVENTS_MEM_TILE_GROUP_LOCK,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL0_ACQ_EQ,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL0_ACQ_GE,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL0_REL,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL0_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL1_ACQ_EQ,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL1_ACQ_GE,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL1_REL,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL1_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL2_ACQ_EQ,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL2_ACQ_GE,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL2_REL,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL2_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL3_ACQ_EQ,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL3_ACQ_GE,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL3_REL,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL3_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL4_ACQ_EQ,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL4_ACQ_GE,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL4_REL,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL4_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL5_ACQ_EQ,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL5_ACQ_GE,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL5_REL,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL5_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL6_ACQ_EQ,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL6_ACQ_GE,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL6_REL,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL6_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL7_ACQ_EQ,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL7_ACQ_GE,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL7_REL,
	XAIE2P_EVENTS_MEM_TILE_LOCK_SEL7_EQUAL_TO_VALUE,
	XAIE2P_EVENTS_MEM_TILE_GROUP_STREAM_SWITCH,
	XAIE2P_EVENTS_MEM_TILE_PORT_IDLE_0,
	XAIE2P_EVENTS_MEM_TILE_PORT_RUNNING_0,
	XAIE2P_EVENTS_MEM_TILE_PORT_STALLED_0,
	XAIE2P_EVENTS_MEM_TILE_PORT_TLAST_0,
	XAIE2P_EVENTS_MEM_TILE_PORT_IDLE_1,
	XAIE2P_EVENTS_MEM_TILE_PORT_RUNNING_1,
	XAIE2P_EVENTS_MEM_TILE_PORT_STALLED_1,
	XAIE2P_EVENTS_MEM_TILE_PORT_TLAST_1,
	XAIE2P_EVENTS_MEM_TILE_PORT_IDLE_2,
	XAIE2P_EVENTS_MEM_TILE_PORT_RUNNING_2,
	XAIE2P_EVENTS_MEM_TILE_PORT_STALLED_2,
	XAIE2P_EVENTS_MEM_TILE_PORT_TLAST_2,
	XAIE2P_EVENTS_MEM_TILE_PORT_IDLE_3,
	XAIE2P_EVENTS_MEM_TILE_PORT_RUNNING_3,
	XAIE2P_EVENTS_MEM_TILE_PORT_STALLED_3,
	XAIE2P_EVENTS_MEM_TILE_PORT_TLAST_3,
	XAIE2P_EVENTS_MEM_TILE_PORT_IDLE_4,
	XAIE2P_EVENTS_MEM_TILE_PORT_RUNNING_4,
	XAIE2P_EVENTS_MEM_TILE_PORT_STALLED_4,
	XAIE2P_EVENTS_MEM_TILE_PORT_TLAST_4,
	XAIE2P_EVENTS_MEM_TILE_PORT_IDLE_5,
	XAIE2P_EVENTS_MEM_TILE_PORT_RUNNING_5,
	XAIE2P_EVENTS_MEM_TILE_PORT_STALLED_5,
	XAIE2P_EVENTS_MEM_TILE_PORT_TLAST_5,
	XAIE2P_EVENTS_MEM_TILE_PORT_IDLE_6,
	XAIE2P_EVENTS_MEM_TILE_PORT_RUNNING_6,
	XAIE2P_EVENTS_MEM_TILE_PORT_STALLED_6,
	XAIE2P_EVENTS_MEM_TILE_PORT_TLAST_6,
	XAIE2P_EVENTS_MEM_TILE_PORT_IDLE_7,
	XAIE2P_EVENTS_MEM_TILE_PORT_RUNNING_7,
	XAIE2P_EVENTS_MEM_TILE_PORT_STALLED_7,
	XAIE2P_EVENTS_MEM_TILE_PORT_TLAST_7,
	XAIE2P_EVENTS_MEM_TILE_GROUP_MEMORY_CONFLICT,
	XAIE2P_EVENTS_MEM_TILE_CONFLICT_DM_BANK_0,
	XAIE2P_EVENTS_MEM_TILE_CONFLICT_DM_BANK_1,
	XAIE2P_EVENTS_MEM_TILE_CONFLICT_DM_BANK_2,
	XAIE2P_EVENTS_MEM_TILE_CONFLICT_DM_BANK_3,
	XAIE2P_EVENTS_MEM_TILE_CONFLICT_DM_BANK_4,
	XAIE2P_EVENTS_MEM_TILE_CONFLICT_DM_BANK_5,
	XAIE2P_EVENTS_MEM_TILE_CONFLICT_DM_BANK_6,
	XAIE2P_EVENTS_MEM_TILE_CONFLICT_DM_BANK_7,
	XAIE2P_EVENTS_MEM_TILE_CONFLICT_DM_BANK_8,
	XAIE2P_EVENTS_MEM_TILE_CONFLICT_DM_BANK_9,
	XAIE2P_EVENTS_MEM_TILE_CONFLICT_DM_BANK_10,
	XAIE2P_EVENTS_MEM_TILE_CONFLICT_DM_BANK_11,
	XAIE2P_EVENTS_MEM_TILE_CONFLICT_DM_BANK_12,
	XAIE2P_EVENTS_MEM_TILE_CONFLICT_DM_BANK_13,
	XAIE2P_EVENTS_MEM_TILE_CONFLICT_DM_BANK_14,
	XAIE2P_EVENTS_MEM_TILE_CONFLICT_DM_BANK_15,
	XAIE2P_EVENTS_MEM_TILE_GROUP_ERRORS,
	XAIE2P_EVENTS_MEM_TILE_DM_ECC_ERROR_SCRUB_CORRECTED,
	XAIE2P_EVENTS_MEM_TILE_DM_ECC_ERROR_SCRUB_2BIT,
	XAIE2P_EVENTS_MEM_TILE_DM_ECC_ERROR_1BIT,
	XAIE2P_EVENTS_MEM_TILE_DM_ECC_ERROR_2BIT,
	XAIE2P_EVENTS_MEM_TILE_DMA_S2MM_ERROR,
	XAIE2P_EVENTS_MEM_TILE_DMA_MM2S_ERROR,
	XAIE2P_EVENTS_MEM_TILE_STREAM_SWITCH_PARITY_ERROR,
	XAIE2P_EVENTS_MEM_TILE_STREAM_PKT_ERROR,
	XAIE2P_EVENTS_MEM_TILE_CONTROL_PKT_ERROR,
	XAIE2P_EVENTS_MEM_TILE_AXI_MM_SLAVE_ERROR,
	XAIE2P_EVENTS_MEM_TILE_LOCK_ERROR,
	XAIE2P_EVENTS_MEM_TILE_DMA_TASK_TOKEN_STALL,
	XAIE2P_EVENTS_MEM_TILE_GROUP_BROADCAST,
	XAIE2P_EVENTS_MEM_TILE_BROADCAST_0,
	XAIE2P_EVENTS_MEM_TILE_BROADCAST_1,
	XAIE2P_EVENTS_MEM_TILE_BROADCAST_2,
	XAIE2P_EVENTS_MEM_TILE_BROADCAST_3,
	XAIE2P_EVENTS_MEM_TILE_BROADCAST_4,
	XAIE2P_EVENTS_MEM_TILE_BROADCAST_5,
	XAIE2P_EVENTS_MEM_TILE_BROADCAST_6,
	XAIE2P_EVENTS_MEM_TILE_BROADCAST_7,
	XAIE2P_EVENTS_MEM_TILE_BROADCAST_8,
	XAIE2P_EVENTS_MEM_TILE_BROADCAST_9,
	XAIE2P_EVENTS_MEM_TILE_BROADCAST_10,
	XAIE2P_EVENTS_MEM_TILE_BROADCAST_11,
	XAIE2P_EVENTS_MEM_TILE_BROADCAST_12,
	XAIE2P_EVENTS_MEM_TILE_BROADCAST_13,
	XAIE2P_EVENTS_MEM_TILE_BROADCAST_14,
	XAIE2P_EVENTS_MEM_TILE_BROADCAST_15,
	XAIE2P_EVENTS_MEM_TILE_GROUP_USER_EVENT,
	XAIE2P_EVENTS_MEM_TILE_USER_EVENT_0,
	XAIE2P_EVENTS_MEM_TILE_USER_EVENT_1,
};

static const XAie_EventGroup Aie2PMemGroupEvent[] =
{
	{
		.GroupEvent = XAIE_EVENT_GROUP_0_MEM,
		.GroupOff = 0U,
		.GroupMask = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_0_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_0_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_WATCHPOINT_MEM,
		.GroupOff = 1U,
		.GroupMask = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_WATCHPOINT_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_WATCHPOINT_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_DMA_ACTIVITY_MEM,
		.GroupOff = 2U,
		.GroupMask = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_DMA_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_DMA_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_LOCK_MEM,
		.GroupOff = 3U,
		.GroupMask = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_LOCK_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_LOCK_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_MEMORY_CONFLICT_MEM,
		.GroupOff = 4U,
		.GroupMask = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_MEMORY_CONFLICT_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_MEMORY_CONFLICT_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_ERRORS_MEM,
		.GroupOff = 5U,
		.GroupMask = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_ERROR_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_ERROR_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_BROADCAST_MEM,
		.GroupOff = 6U,
		.GroupMask = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_BROADCAST_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_BROADCAST_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_USER_EVENT_MEM,
		.GroupOff = 7U,
		.GroupMask = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_USER_EVENT_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_USER_EVENT_ENABLE_MASK,
	},
};

static const XAie_EventGroup Aie2PCoreGroupEvent[] =
{
	{
		.GroupEvent = XAIE_EVENT_GROUP_0_CORE,
		.GroupOff = 0U,
		.GroupMask = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_0_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_0_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_PC_EVENT_CORE,
		.GroupOff = 1U,
		.GroupMask = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_PC_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_PC_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_CORE_STALL_CORE,
		.GroupOff = 2U,
		.GroupMask = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_CORE_STALL_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_CORE_STALL_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_CORE_PROGRAM_FLOW_CORE,
		.GroupOff = 3U,
		.GroupMask = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_CORE_PROGRAM_FLOW_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_CORE_PROGRAM_FLOW_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_ERRORS_0_CORE,
		.GroupOff = 4U,
		.GroupMask = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_ERRORS0_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_ERRORS0_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_ERRORS_1_CORE,
		.GroupOff = 5U,
		.GroupMask = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_ERRORS1_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_ERRORS1_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_STREAM_SWITCH_CORE,
		.GroupOff = 6U,
		.GroupMask = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_STREAM_SWITCH_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_STREAM_SWITCH_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_BROADCAST_CORE,
		.GroupOff = 7U,
		.GroupMask = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_BROADCAST_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_BROADCAST_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_USER_EVENT_CORE,
		.GroupOff = 8U,
		.GroupMask = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_USER_EVENT_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_USER_EVENT_ENABLE_MASK,
	},
};

static const XAie_EventGroup Aie2PPlGroupEvent[] =
{
	{
		.GroupEvent = XAIE_EVENT_GROUP_0_PL,
		.GroupOff = 0U,
		.GroupMask = XAIE2PGBL_PL_MODULE_EVENT_GROUP_0_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_PL_MODULE_EVENT_GROUP_0_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_DMA_ACTIVITY_PL,
		.GroupOff = 1U,
		.GroupMask = XAIE2PGBL_PL_MODULE_EVENT_GROUP_DMA_ACTIVITY_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_PL_MODULE_EVENT_GROUP_DMA_ACTIVITY_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_LOCK_PL,
		.GroupOff = 2U,
		.GroupMask = XAIE2PGBL_PL_MODULE_EVENT_GROUP_LOCK_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_PL_MODULE_EVENT_GROUP_LOCK_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_ERRORS_PL,
		.GroupOff = 3U,
		.GroupMask = XAIE2PGBL_PL_MODULE_EVENT_GROUP_ERRORS_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_PL_MODULE_EVENT_GROUP_ERRORS_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_STREAM_SWITCH_PL,
		.GroupOff = 4U,
		.GroupMask = XAIE2PGBL_PL_MODULE_EVENT_GROUP_STREAM_SWITCH_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_PL_MODULE_EVENT_GROUP_STREAM_SWITCH_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_BROADCAST_A_PL,
		.GroupOff = 5U,
		.GroupMask = XAIE2PGBL_PL_MODULE_EVENT_GROUP_BROADCAST_A_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_PL_MODULE_EVENT_GROUP_BROADCAST_A_ENABLE_MASK,
	},
};

static const XAie_EventGroup Aie2PMemTileGroupEvent[] = {
	{
		.GroupEvent = XAIE_EVENT_GROUP_0_MEM_TILE,
		.GroupOff = 0U,
		.GroupMask = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_0_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_0_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_WATCHPOINT_MEM_TILE,
		.GroupOff = 1U,
		.GroupMask = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_WATCHPOINT_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_WATCHPOINT_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_DMA_ACTIVITY_MEM_TILE,
		.GroupOff = 2U,
		.GroupMask = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_DMA_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_DMA_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_LOCK_MEM_TILE,
		.GroupOff = 3U,
		.GroupMask = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_LOCK_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_LOCK_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_STREAM_SWITCH_MEM_TILE,
		.GroupOff = 4U,
		.GroupMask = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_STREAM_SWITCH_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_STREAM_SWITCH_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_MEMORY_CONFLICT_MEM_TILE,
		.GroupOff = 5U,
		.GroupMask = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_MEMORY_CONFLICT_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_MEMORY_CONFLICT_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_ERRORS_MEM_TILE,
		.GroupOff = 6U,
		.GroupMask = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_ERROR_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_ERROR_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_BROADCAST_MEM_TILE,
		.GroupOff = 7U,
		.GroupMask = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_BROADCAST_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_BROADCAST_ENABLE_MASK,
	},
	{
		.GroupEvent = XAIE_EVENT_GROUP_USER_EVENT_MEM_TILE,
		.GroupOff = 8U,
		.GroupMask = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_USER_EVENT_ENABLE_MASK,
		.ResetValue = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_USER_EVENT_ENABLE_MASK,
	},
};

/* mapping of user events for core module */
static const XAie_EventMap Aie2PTileCoreModUserEventMap =
{
	.RscId = 0U,
	.Event = XAIE_EVENT_USER_EVENT_0_CORE,
};

/* mapping of user events for memory module */
static const XAie_EventMap Aie2PTileMemModUserEventStart =
{
	.RscId = 0U,
	.Event = XAIE_EVENT_USER_EVENT_0_MEM,
};

/* mapping of user events for mem tile memory module */
static const XAie_EventMap Aie2PMemTileMemModUserEventStart =
{
	.RscId = 0U,
	.Event = XAIE_EVENT_USER_EVENT_0_MEM_TILE,
};

/* mapping of user events for memory module */
static const XAie_EventMap Aie2PShimTilePlModUserEventStart =
{
	.RscId = 0U,
	.Event = XAIE_EVENT_USER_EVENT_0_PL,
};

static const XAie_EventMap Aie2PTileCoreModPCEventMap =
{
	.RscId = 0U,
	.Event = XAIE_EVENT_PC_0_CORE,
};

/* mapping of broadcast events for core module */
static const XAie_EventMap Aie2PTileCoreModBroadcastEventMap =
{
	.RscId = 0U,
	.Event = XAIE_EVENT_BROADCAST_0_CORE,
};

/* mapping of broadcast events for memory module */
static const XAie_EventMap Aie2PTileMemModBroadcastEventStart =
{
	.RscId = 0U,
	.Event = XAIE_EVENT_BROADCAST_0_MEM,
};

/* mapping of broadcast events for Pl module */
static const XAie_EventMap Aie2PShimTilePlModBroadcastEventStart =
{
	.RscId = 0U,
	.Event = XAIE_EVENT_BROADCAST_A_0_PL,
};

/* mapping of broadcast events for Mem tile mem module */
static const XAie_EventMap Aie2PMemTileMemModBroadcastEventStart =
{
	.RscId = 0U,
	.Event = XAIE_EVENT_BROADCAST_0_MEM_TILE,
};

/*
 * Data structure to capture core and memory module events properties.
 * For memory module default error group mask enables,
 *	DM_ECC_Error_Scrub_2bit,
 *	DM_ECC_Error_2bit,
 *	DM_Parity_Error_Bank_2,
 *	DM_Parity_Error_Bank_3,
 *	DM_Parity_Error_Bank_4,
 *	DM_Parity_Error_Bank_5,
 *	DM_Parity_Error_Bank_6,
 *	DM_Parity_Error_Bank_7,
 *	DMA_S2MM_0_Error,
 *	DMA_S2MM_1_Error,
 *	DMA_MM2S_0_Error,
 *	DMA_MM2S_1_Error,
 *	Lock_Error.
 * For core module default error group mask enables,
 *	PM_Reg_Access_Failure,
 *	Stream_Pkt_Parity_Error,
 *	Control_Pkt_Error,
 *	AXI_MM_Slave_Error,
 *	Instruction_Decompression_Error,
 *	DM_address_out_of_range,
 *	PM_ECC_Error_Scrub_2bit,
 *	PM_ECC_Error_2bit,
 *	PM_address_out_of_range,
 *	DM_access_to_Unavailable,
 *	Lock_Access_to_Unavailable,
 *	Decompression_underflow,
 *	Stream_Switch_Port_Parity_Error,
 *	Processor_Bus_Error.
 */
static const XAie_EvntMod Aie2PTileEvntMod[] =
{
	{
		.XAie_EventNumber = Aie2PMemModEventMapping,
		.EventMin = XAIE_EVENT_NONE_MEM,
		.EventMax = XAIE_EVENT_DMA_TASK_TOKEN_STALL_MEM,
		.ComboEventBase = XAIE_EVENT_COMBO_EVENT_0_MEM,
		.PerfCntEventBase = XAIE_EVENT_PERF_CNT_0_MEM,
		.UserEventBase = XAIE_EVENT_USER_EVENT_0_MEM,
		.PortIdleEventBase = XAIE_EVENT_INVALID,
		.GenEventRegOff = XAIE2PGBL_MEMORY_MODULE_EVENT_GENERATE,
		.GenEvent = {XAIE2PGBL_MEMORY_MODULE_EVENT_GENERATE_EVENT_LSB, XAIE2PGBL_MEMORY_MODULE_EVENT_GENERATE_EVENT_MASK},
		.ComboInputRegOff = XAIE2PGBL_MEMORY_MODULE_COMBO_EVENT_INPUTS,
		.ComboEventMask = XAIE2PGBL_MEMORY_MODULE_COMBO_EVENT_INPUTS_EVENTA_MASK,
		.ComboEventOff = 8U,
		.ComboCtrlRegOff = XAIE2PGBL_MEMORY_MODULE_COMBO_EVENT_CONTROL,
		.ComboConfigMask = XAIE2PGBL_MEMORY_MODULE_COMBO_EVENT_CONTROL_COMBO0_MASK,
		.ComboConfigOff = 8U,
		.BaseStrmPortSelectRegOff = XAIE_FEATURE_UNAVAILABLE,
		.NumStrmPortSelectIds = XAIE_FEATURE_UNAVAILABLE,
		.StrmPortSelectIdsPerReg = XAIE_FEATURE_UNAVAILABLE,
		.PortIdMask = XAIE_FEATURE_UNAVAILABLE,
		.PortIdOff = XAIE_FEATURE_UNAVAILABLE,
		.PortMstrSlvMask = XAIE_FEATURE_UNAVAILABLE,
		.PortMstrSlvOff = XAIE_FEATURE_UNAVAILABLE,
		.BaseDmaChannelSelectRegOff = XAIE_FEATURE_UNAVAILABLE,
		.NumDmaChannelSelectIds = XAIE_FEATURE_UNAVAILABLE,
		.DmaChannelIdOff = XAIE_FEATURE_UNAVAILABLE,
		.DmaChannelIdMask = XAIE_FEATURE_UNAVAILABLE,
		.DmaChannelMM2SOff = XAIE_FEATURE_UNAVAILABLE,
		.EdgeEventRegOff = XAIE2PGBL_MEMORY_MODULE_EDGE_DETECTION_EVENT_CONTROL,
		.EdgeDetectEvent = {XAIE2PGBL_MEMORY_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_EVENT_0_LSB, XAIE2PGBL_MEMORY_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_EVENT_0_MASK},
		.EdgeDetectTrigger = {XAIE2PGBL_MEMORY_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_0_TRIGGER_RISING_LSB, XAIE2PGBL_MEMORY_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_0_TRIGGER_FALLING_MASK | XAIE2PGBL_MEMORY_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_0_TRIGGER_RISING_MASK},
		.EdgeEventSelectIdOff = XAIE2PGBL_MEMORY_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_EVENT_1_LSB,
		.NumEdgeSelectIds = 2U,
		.BaseBroadcastRegOff = XAIE2PGBL_MEMORY_MODULE_EVENT_BROADCAST0,
		.NumBroadcastIds = 16U,
		.BaseBroadcastSwBlockRegOff = XAIE2PGBL_MEMORY_MODULE_EVENT_BROADCAST_BLOCK_SOUTH_SET,
		.BaseBroadcastSwUnblockRegOff = XAIE2PGBL_MEMORY_MODULE_EVENT_BROADCAST_BLOCK_SOUTH_CLR,
		.BroadcastSwOff = 0U,
		.BroadcastSwBlockOff = 16U,
		.BroadcastSwUnblockOff = 16U,
		.NumSwitches = 1U,
		.BaseGroupEventRegOff = XAIE2PGBL_MEMORY_MODULE_EVENT_GROUP_0_ENABLE,
		.NumGroupEvents = 8U,
		.DefaultGroupErrorMask = 0x7FFAU,
		.Group = Aie2PMemGroupEvent,
		.BasePCEventRegOff = XAIE_FEATURE_UNAVAILABLE,
		.NumPCEvents = XAIE_FEATURE_UNAVAILABLE,
		.PCAddr = {XAIE_FEATURE_UNAVAILABLE, XAIE_FEATURE_UNAVAILABLE},
		.PCValid = {XAIE_FEATURE_UNAVAILABLE, XAIE_FEATURE_UNAVAILABLE},
		.BaseStatusRegOff = XAIE2PGBL_MEMORY_MODULE_EVENT_STATUS0,
		.NumUserEvents = 4U,
		.UserEventMap = &Aie2PTileMemModUserEventStart,
		.PCEventMap = NULL,
		.BroadcastEventMap = &Aie2PTileMemModBroadcastEventStart,
		.ErrorHaltRegOff = XAIE_FEATURE_UNAVAILABLE,
	},
	{
		.XAie_EventNumber = Aie2PCoreModEventMapping,
		.EventMin = XAIE_EVENT_NONE_CORE,
		.EventMax = XAIE_EVENT_SPARSITY_OVERFLOW_CORE,
		.ComboEventBase = XAIE_EVENT_COMBO_EVENT_0_CORE,
		.PerfCntEventBase = XAIE_EVENT_PERF_CNT_0_CORE,
		.UserEventBase = XAIE_EVENT_USER_EVENT_0_CORE,
		.PortIdleEventBase = XAIE_EVENT_PORT_IDLE_0_CORE,
		.GenEventRegOff = XAIE2PGBL_CORE_MODULE_EVENT_GENERATE,
		.GenEvent = {XAIE2PGBL_CORE_MODULE_EVENT_GENERATE_EVENT_LSB, XAIE2PGBL_CORE_MODULE_EVENT_GENERATE_EVENT_MASK},
		.ComboInputRegOff = XAIE2PGBL_CORE_MODULE_COMBO_EVENT_INPUTS,
		.ComboEventMask = XAIE2PGBL_CORE_MODULE_COMBO_EVENT_INPUTS_EVENTA_MASK,
		.ComboEventOff = 8U,
		.ComboCtrlRegOff = XAIE2PGBL_CORE_MODULE_COMBO_EVENT_CONTROL,
		.ComboConfigMask = XAIE2PGBL_CORE_MODULE_COMBO_EVENT_CONTROL_COMBO0_MASK,
		.ComboConfigOff = 8U,
		.BaseStrmPortSelectRegOff = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_EVENT_PORT_SELECTION_0,
		.NumStrmPortSelectIds = 8U,
		.StrmPortSelectIdsPerReg = 4U,
		.PortIdMask = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_EVENT_PORT_SELECTION_0_PORT_0_ID_MASK,
		.PortIdOff = 8U,
		.PortMstrSlvMask = XAIE2PGBL_CORE_MODULE_STREAM_SWITCH_EVENT_PORT_SELECTION_0_PORT_0_MASTER_SLAVE_MASK,
		.PortMstrSlvOff = 5U,
		.BaseDmaChannelSelectRegOff = XAIE_FEATURE_UNAVAILABLE,
		.NumDmaChannelSelectIds = XAIE_FEATURE_UNAVAILABLE,
		.DmaChannelIdOff = XAIE_FEATURE_UNAVAILABLE,
		.DmaChannelIdMask = XAIE_FEATURE_UNAVAILABLE,
		.DmaChannelMM2SOff = XAIE_FEATURE_UNAVAILABLE,
		.EdgeEventRegOff = XAIE2PGBL_CORE_MODULE_EDGE_DETECTION_EVENT_CONTROL,
		.EdgeDetectEvent = {XAIE2PGBL_CORE_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_EVENT_0_LSB, XAIE2PGBL_CORE_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_EVENT_0_MASK},
		.EdgeDetectTrigger = {XAIE2PGBL_CORE_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_0_TRIGGER_RISING_LSB, XAIE2PGBL_CORE_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_0_TRIGGER_FALLING_MASK | XAIE2PGBL_CORE_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_0_TRIGGER_RISING_MASK},
		.EdgeEventSelectIdOff = XAIE2PGBL_CORE_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_EVENT_1_LSB,
		.NumEdgeSelectIds = 2U,
		.BaseBroadcastRegOff = XAIE2PGBL_CORE_MODULE_EVENT_BROADCAST0,
		.NumBroadcastIds = 16U,
		.BaseBroadcastSwBlockRegOff = XAIE2PGBL_CORE_MODULE_EVENT_BROADCAST_BLOCK_SOUTH_SET,
		.BaseBroadcastSwUnblockRegOff = XAIE2PGBL_CORE_MODULE_EVENT_BROADCAST_BLOCK_SOUTH_CLR,
		.BroadcastSwOff = 0U,
		.BroadcastSwBlockOff = 16U,
		.BroadcastSwUnblockOff = 16U,
		.NumSwitches = 1U,
		.BaseGroupEventRegOff = XAIE2PGBL_CORE_MODULE_EVENT_GROUP_0_ENABLE,
		.NumGroupEvents = 9U,
		.DefaultGroupErrorMask = 0x1CF5F80U,
		.Group = Aie2PCoreGroupEvent,
		.BasePCEventRegOff = XAIE2PGBL_CORE_MODULE_PC_EVENT0,
		.NumPCEvents = 4U,
		.PCAddr = {XAIE2PGBL_CORE_MODULE_PC_EVENT0_PC_ADDRESS_LSB, XAIE2PGBL_CORE_MODULE_PC_EVENT0_PC_ADDRESS_MASK},
		.PCValid = {XAIE2PGBL_CORE_MODULE_PC_EVENT0_VALID_LSB, XAIE2PGBL_CORE_MODULE_PC_EVENT0_VALID_MASK},
		.BaseStatusRegOff = XAIE2PGBL_CORE_MODULE_EVENT_STATUS0,
		.NumUserEvents = 4U,
		.UserEventMap = &Aie2PTileCoreModUserEventMap,
		.PCEventMap = &Aie2PTileCoreModPCEventMap,
		.BroadcastEventMap = &Aie2PTileCoreModBroadcastEventMap,
		.ErrorHaltRegOff = XAIE2PGBL_CORE_MODULE_ERROR_HALT_EVENT,
	}
};

/*
 * Data structure to capture NOC tile events properties.
 * For PL module default error group mask enables,
 *	AXI_MM_Slave_Tile_Error,
 *	Control_Pkt_Error,
 *	Stream_Switch_Parity_Error,
 *	AXI_MM_Decode_NSU_Error,
 *	AXI_MM_Slave_NSU_Error,
 *	AXI_MM_Unsupported_Traffic,
 *	AXI_MM_Unsecure_Access_in_Secure_Mode,
 *	AXI_MM_Byte_Strobe_Error,
 *	DMA_S2MM_Error,
 *	DMA_MM2S_Error,
 *	Lock_Error.
 */
static const XAie_EvntMod Aie2PNocEvntMod =
{
	.XAie_EventNumber = Aie2PNocModEventMapping,
	.EventMin = XAIE_EVENT_NONE_PL,
	.EventMax = XAIE_EVENT_DMA_TASK_TOKEN_STALL_PL,
	.ComboEventBase = XAIE_EVENT_COMBO_EVENT_0_PL,
	.PerfCntEventBase = XAIE_EVENT_PERF_CNT_0_PL,
	.UserEventBase = XAIE_EVENT_USER_EVENT_0_PL,
	.PortIdleEventBase = XAIE_EVENT_PORT_IDLE_0_PL,
	.GenEventRegOff = XAIE2PGBL_PL_MODULE_EVENT_GENERATE,
	.GenEvent = {XAIE2PGBL_PL_MODULE_EVENT_GENERATE_EVENT_LSB, XAIE2PGBL_PL_MODULE_EVENT_GENERATE_EVENT_MASK},
	.ComboInputRegOff = XAIE2PGBL_PL_MODULE_COMBO_EVENT_INPUTS,
	.ComboEventMask = XAIE2PGBL_PL_MODULE_COMBO_EVENT_INPUTS_EVENTA_MASK,
	.ComboEventOff = 8U,
	.ComboCtrlRegOff = XAIE2PGBL_PL_MODULE_COMBO_EVENT_CONTROL,
	.ComboConfigMask = XAIE2PGBL_PL_MODULE_COMBO_EVENT_CONTROL_COMBO0_MASK,
	.ComboConfigOff = 8U,
	.BaseStrmPortSelectRegOff = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_EVENT_PORT_SELECTION_0,
	.NumStrmPortSelectIds = 8U,
	.StrmPortSelectIdsPerReg = 4U,
	.PortIdMask = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_EVENT_PORT_SELECTION_0_PORT_0_ID_MASK,
	.PortIdOff = 8U,
	.PortMstrSlvMask = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_EVENT_PORT_SELECTION_0_PORT_0_MASTER_SLAVE_MASK,
	.PortMstrSlvOff = 5U,
	.BaseDmaChannelSelectRegOff = XAIE_FEATURE_UNAVAILABLE,
	.NumDmaChannelSelectIds = XAIE_FEATURE_UNAVAILABLE,
	.DmaChannelIdOff = XAIE_FEATURE_UNAVAILABLE,
	.DmaChannelIdMask = XAIE_FEATURE_UNAVAILABLE,
	.DmaChannelMM2SOff = XAIE_FEATURE_UNAVAILABLE,
	.EdgeEventRegOff = XAIE2PGBL_PL_MODULE_EDGE_DETECTION_EVENT_CONTROL,
	.EdgeDetectEvent = {XAIE2PGBL_PL_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_EVENT_0_LSB, XAIE2PGBL_PL_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_EVENT_0_MASK},
	.EdgeDetectTrigger = {XAIE2PGBL_PL_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_0_TRIGGER_RISING_LSB, XAIE2PGBL_PL_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_0_TRIGGER_FALLING_MASK | XAIE2PGBL_PL_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_0_TRIGGER_RISING_MASK},
	.EdgeEventSelectIdOff = XAIE2PGBL_PL_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_EVENT_1_LSB,
	.NumEdgeSelectIds = 2U,
	.BaseBroadcastRegOff = XAIE2PGBL_PL_MODULE_EVENT_BROADCAST0_A,
	.NumBroadcastIds = 16U,
	.BaseBroadcastSwBlockRegOff = XAIE2PGBL_PL_MODULE_EVENT_BROADCAST_A_BLOCK_SOUTH_SET,
	.BaseBroadcastSwUnblockRegOff = XAIE2PGBL_PL_MODULE_EVENT_BROADCAST_A_BLOCK_SOUTH_CLR,
	.BroadcastSwOff = 64U,
	.BroadcastSwBlockOff = 16U,
	.BroadcastSwUnblockOff = 16U,
	.NumSwitches = 2U,
	.BaseGroupEventRegOff = XAIE2PGBL_PL_MODULE_EVENT_GROUP_0_ENABLE,
	.NumGroupEvents = 6U,
	.DefaultGroupErrorMask = 0x7FFU,
	.Group = Aie2PPlGroupEvent,
	.BasePCEventRegOff = XAIE_FEATURE_UNAVAILABLE,
	.NumPCEvents = XAIE_FEATURE_UNAVAILABLE,
	.PCAddr = {XAIE_FEATURE_UNAVAILABLE, XAIE_FEATURE_UNAVAILABLE},
	.PCValid = {XAIE_FEATURE_UNAVAILABLE, XAIE_FEATURE_UNAVAILABLE},
	.BaseStatusRegOff = XAIE2PGBL_PL_MODULE_EVENT_STATUS0,
	.NumUserEvents = 4U,
	.UserEventMap = &Aie2PShimTilePlModUserEventStart,
	.PCEventMap = NULL,
	.BroadcastEventMap = &Aie2PShimTilePlModBroadcastEventStart,
	.ErrorHaltRegOff = XAIE_FEATURE_UNAVAILABLE,
};

/*
 * Data structure to capture PL module events properties.
 * For PL module default error group mask enables,
 *	AXI_MM_Slave_Tile_Error,
 *	Control_Pkt_Error,
 *	Stream_Switch_Parity_Error,
 *	AXI_MM_Decode_NSU_Error,
 *	AXI_MM_Slave_NSU_Error,
 *	AXI_MM_Unsupported_Traffic,
 *	AXI_MM_Unsecure_Access_in_Secure_Mode,
 *	AXI_MM_Byte_Strobe_Error,
 *	DMA_S2MM_Error,
 *	DMA_MM2S_Error,
 *	Lock_Error.
 */
static const XAie_EvntMod Aie2PPlEvntMod =
{
	.XAie_EventNumber = Aie2PPlModEventMapping,
	.EventMin = XAIE_EVENT_NONE_PL,
	.EventMax = XAIE_EVENT_DMA_TASK_TOKEN_STALL_PL,
	.ComboEventBase = XAIE_EVENT_COMBO_EVENT_0_PL,
	.PerfCntEventBase = XAIE_EVENT_PERF_CNT_0_PL,
	.UserEventBase = XAIE_EVENT_USER_EVENT_0_PL,
	.PortIdleEventBase = XAIE_EVENT_PORT_IDLE_0_PL,
	.GenEventRegOff = XAIE2PGBL_PL_MODULE_EVENT_GENERATE,
	.GenEvent = {XAIE2PGBL_PL_MODULE_EVENT_GENERATE_EVENT_LSB, XAIE2PGBL_PL_MODULE_EVENT_GENERATE_EVENT_MASK},
	.ComboInputRegOff = XAIE2PGBL_PL_MODULE_COMBO_EVENT_INPUTS,
	.ComboEventMask = XAIE2PGBL_PL_MODULE_COMBO_EVENT_INPUTS_EVENTA_MASK,
	.ComboEventOff = 8U,
	.ComboCtrlRegOff = XAIE2PGBL_PL_MODULE_COMBO_EVENT_CONTROL,
	.ComboConfigMask = XAIE2PGBL_PL_MODULE_COMBO_EVENT_CONTROL_COMBO0_MASK,
	.ComboConfigOff = 8U,
	.BaseStrmPortSelectRegOff = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_EVENT_PORT_SELECTION_0,
	.NumStrmPortSelectIds = 8U,
	.StrmPortSelectIdsPerReg = 4U,
	.PortIdMask = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_EVENT_PORT_SELECTION_0_PORT_0_ID_MASK,
	.PortIdOff = 8U,
	.PortMstrSlvMask = XAIE2PGBL_PL_MODULE_STREAM_SWITCH_EVENT_PORT_SELECTION_0_PORT_0_MASTER_SLAVE_MASK,
	.PortMstrSlvOff = 5U,
	.BaseDmaChannelSelectRegOff = XAIE_FEATURE_UNAVAILABLE,
	.NumDmaChannelSelectIds = XAIE_FEATURE_UNAVAILABLE,
	.DmaChannelIdOff = XAIE_FEATURE_UNAVAILABLE,
	.DmaChannelIdMask = XAIE_FEATURE_UNAVAILABLE,
	.DmaChannelMM2SOff = XAIE_FEATURE_UNAVAILABLE,
	.EdgeEventRegOff = XAIE2PGBL_PL_MODULE_EDGE_DETECTION_EVENT_CONTROL,
	.EdgeDetectEvent = {XAIE2PGBL_PL_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_EVENT_0_LSB, XAIE2PGBL_PL_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_EVENT_0_MASK},
	.EdgeDetectTrigger = {XAIE2PGBL_PL_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_0_TRIGGER_RISING_LSB, XAIE2PGBL_PL_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_0_TRIGGER_FALLING_MASK | XAIE2PGBL_PL_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_0_TRIGGER_RISING_MASK},
	.EdgeEventSelectIdOff = XAIE2PGBL_PL_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_EVENT_1_LSB,
	.NumEdgeSelectIds = 2U,
	.BaseBroadcastRegOff = XAIE2PGBL_PL_MODULE_EVENT_BROADCAST0_A,
	.NumBroadcastIds = 16U,
	.BaseBroadcastSwBlockRegOff = XAIE2PGBL_PL_MODULE_EVENT_BROADCAST_A_BLOCK_SOUTH_SET,
	.BaseBroadcastSwUnblockRegOff = XAIE2PGBL_PL_MODULE_EVENT_BROADCAST_A_BLOCK_SOUTH_CLR,
	.BroadcastSwOff = 64U,
	.BroadcastSwBlockOff = 16U,
	.BroadcastSwUnblockOff = 16U,
	.NumSwitches = 2U,
	.BaseGroupEventRegOff = XAIE2PGBL_PL_MODULE_EVENT_GROUP_0_ENABLE,
	.NumGroupEvents = 6U,
	.DefaultGroupErrorMask = 0x7FFU,
	.Group = Aie2PPlGroupEvent,
	.BasePCEventRegOff = XAIE_FEATURE_UNAVAILABLE,
	.NumPCEvents = XAIE_FEATURE_UNAVAILABLE,
	.PCAddr = {XAIE_FEATURE_UNAVAILABLE, XAIE_FEATURE_UNAVAILABLE},
	.PCValid = {XAIE_FEATURE_UNAVAILABLE, XAIE_FEATURE_UNAVAILABLE},
	.BaseStatusRegOff = XAIE2PGBL_PL_MODULE_EVENT_STATUS0,
	.NumUserEvents = 4U,
	.UserEventMap = &Aie2PShimTilePlModUserEventStart,
	.BroadcastEventMap = &Aie2PShimTilePlModBroadcastEventStart,
	.PCEventMap = NULL,
	.ErrorHaltRegOff = XAIE_FEATURE_UNAVAILABLE,
};

/*
 * Data structure to capture mem tile module events properties.
 * For mem tile default error group mask enables,
 *	DM_ECC_Error_Scrub_2bit,
 *	DM_ECC_Error_2bit,
 *	DMA_S2MM_Error,
 *	DMA_MM2S_Error,
 *	Stream_Switch_Parity_Error,
 *	Stream_Pkt_Parity_Error,
 *	Control_Pkt_Error,
 *	AXI-MM_Slave_Error,
 *	Lock_Error.
 */
static const XAie_EvntMod Aie2PMemTileEvntMod =
{
	.XAie_EventNumber = Aie2PMemTileModEventMapping,
	.EventMin = XAIE_EVENT_NONE_MEM_TILE,
	.EventMax = XAIE_EVENT_USER_EVENT_1_MEM_TILE,
	.ComboEventBase = XAIE_EVENT_COMBO_EVENT_0_MEM_TILE,
	.PerfCntEventBase = XAIE_EVENT_PERF_CNT0_EVENT_MEM_TILE,
	.UserEventBase = XAIE_EVENT_USER_EVENT_0_MEM_TILE,
	.PortIdleEventBase = XAIE_EVENT_PORT_IDLE_0_MEM_TILE,
	.GenEventRegOff = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GENERATE,
	.GenEvent = {XAIE2PGBL_MEM_TILE_MODULE_EVENT_GENERATE_EVENT_LSB, XAIE2PGBL_MEM_TILE_MODULE_EVENT_GENERATE_EVENT_MASK},
	.ComboInputRegOff = XAIE2PGBL_MEM_TILE_MODULE_COMBO_EVENT_INPUTS,
	.ComboEventMask = XAIE2PGBL_MEM_TILE_MODULE_COMBO_EVENT_INPUTS_EVENTA_MASK,
	.ComboEventOff = 8U,
	.ComboCtrlRegOff = XAIE2PGBL_MEM_TILE_MODULE_COMBO_EVENT_CONTROL,
	.ComboConfigMask = XAIE2PGBL_MEM_TILE_MODULE_COMBO_EVENT_CONTROL_COMBO0_MASK,
	.ComboConfigOff = 8U,
	.BaseStrmPortSelectRegOff = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_EVENT_PORT_SELECTION_0,
	.NumStrmPortSelectIds = 8U,
	.StrmPortSelectIdsPerReg = 4U,
	.PortIdMask = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_EVENT_PORT_SELECTION_0_PORT_0_ID_MASK,
	.PortIdOff = 8U,
	.PortMstrSlvMask = XAIE2PGBL_MEM_TILE_MODULE_STREAM_SWITCH_EVENT_PORT_SELECTION_0_PORT_0_MASTER_SLAVE_MASK,
	.PortMstrSlvOff = 5U,
	.BaseDmaChannelSelectRegOff = XAIE2PGBL_MEM_TILE_MODULE_DMA_EVENT_CHANNEL_SELECTION,
	.NumDmaChannelSelectIds = 2U,
	.DmaChannelIdOff = 8U,
	.DmaChannelIdMask = XAIE2PGBL_MEM_TILE_MODULE_DMA_EVENT_CHANNEL_SELECTION_S2MM_SEL0_CHANNEL_MASK,
	.DmaChannelMM2SOff = XAIE2PGBL_MEM_TILE_MODULE_DMA_EVENT_CHANNEL_SELECTION_MM2S_SEL0_CHANNEL_LSB,
	.EdgeEventRegOff = XAIE2PGBL_MEM_TILE_MODULE_EDGE_DETECTION_EVENT_CONTROL,
	.EdgeDetectEvent = {XAIE2PGBL_MEM_TILE_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_EVENT_0_LSB, XAIE2PGBL_MEM_TILE_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_EVENT_0_MASK},
	.EdgeDetectTrigger = {XAIE2PGBL_MEM_TILE_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_0_TRIGGER_RISING_LSB, XAIE2PGBL_MEM_TILE_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_0_TRIGGER_FALLING_MASK | XAIE2PGBL_MEM_TILE_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_0_TRIGGER_RISING_MASK},
	.EdgeEventSelectIdOff = XAIE2PGBL_MEM_TILE_MODULE_EDGE_DETECTION_EVENT_CONTROL_EDGE_DETECTION_EVENT_1_LSB,
	.NumEdgeSelectIds = 2U,
	.BaseBroadcastRegOff = XAIE2PGBL_MEM_TILE_MODULE_EVENT_BROADCAST0,
	.NumBroadcastIds = 16U,
	.BaseBroadcastSwBlockRegOff = XAIE2PGBL_MEM_TILE_MODULE_EVENT_BROADCAST_A_BLOCK_SOUTH_SET,
	.BaseBroadcastSwUnblockRegOff = XAIE2PGBL_MEM_TILE_MODULE_EVENT_BROADCAST_A_BLOCK_SOUTH_CLR,
	.BroadcastSwOff = 64U,
	.BroadcastSwBlockOff = 16U,
	.BroadcastSwUnblockOff = 16U,
	.NumSwitches = 2U,
	.BaseGroupEventRegOff = XAIE2PGBL_MEM_TILE_MODULE_EVENT_GROUP_0_ENABLE,
	.NumGroupEvents = 9U,
	.DefaultGroupErrorMask = 0x7FAU,
	.Group = Aie2PMemTileGroupEvent,
	.BasePCEventRegOff = XAIE_FEATURE_UNAVAILABLE,
	.NumPCEvents = XAIE_FEATURE_UNAVAILABLE,
	.PCAddr = {XAIE_FEATURE_UNAVAILABLE, XAIE_FEATURE_UNAVAILABLE},
	.PCValid = {XAIE_FEATURE_UNAVAILABLE, XAIE_FEATURE_UNAVAILABLE},
	.BaseStatusRegOff = XAIE2PGBL_MEM_TILE_MODULE_EVENT_STATUS0,
	.NumUserEvents = 2U,
	.UserEventMap = &Aie2PMemTileMemModUserEventStart,
	.PCEventMap = NULL,
	.BroadcastEventMap = &Aie2PMemTileMemModBroadcastEventStart,
	.ErrorHaltRegOff = XAIE_FEATURE_UNAVAILABLE,
};
#endif /* XAIE_FEATURE_EVENTS_ENABLE */

#ifdef XAIE_FEATURE_TIMER_ENABLE
static const XAie_TimerMod Aie2PTileTimerMod[] =
{
	 {
		.TrigEventLowValOff = XAIE2PGBL_MEMORY_MODULE_TIMER_TRIG_EVENT_LOW_VALUE,
		.TrigEventHighValOff = XAIE2PGBL_MEMORY_MODULE_TIMER_TRIG_EVENT_HIGH_VALUE,
		.LowOff = XAIE2PGBL_MEMORY_MODULE_TIMER_LOW,
		.HighOff = XAIE2PGBL_MEMORY_MODULE_TIMER_HIGH,
		.CtrlOff = XAIE2PGBL_MEMORY_MODULE_TIMER_CONTROL,
		{XAIE2PGBL_MEMORY_MODULE_TIMER_CONTROL_RESET_LSB, XAIE2PGBL_MEMORY_MODULE_TIMER_CONTROL_RESET_MASK},
		{XAIE2PGBL_MEMORY_MODULE_TIMER_CONTROL_RESET_EVENT_LSB, XAIE2PGBL_MEMORY_MODULE_TIMER_CONTROL_RESET_EVENT_MASK},
	},
	{
		.TrigEventLowValOff = XAIE2PGBL_CORE_MODULE_TIMER_TRIG_EVENT_LOW_VALUE,
		.TrigEventHighValOff = XAIE2PGBL_CORE_MODULE_TIMER_TRIG_EVENT_HIGH_VALUE,
		.LowOff = XAIE2PGBL_CORE_MODULE_TIMER_LOW,
		.HighOff = XAIE2PGBL_CORE_MODULE_TIMER_HIGH,
		.CtrlOff = XAIE2PGBL_CORE_MODULE_TIMER_CONTROL,
		{XAIE2PGBL_CORE_MODULE_TIMER_CONTROL_RESET_LSB, XAIE2PGBL_CORE_MODULE_TIMER_CONTROL_RESET_MASK},
		{XAIE2PGBL_CORE_MODULE_TIMER_CONTROL_RESET_EVENT_LSB, XAIE2PGBL_CORE_MODULE_TIMER_CONTROL_RESET_EVENT_MASK},
	}
};

static const XAie_TimerMod Aie2PPlTimerMod =
{
	.TrigEventLowValOff = XAIE2PGBL_PL_MODULE_TIMER_TRIG_EVENT_LOW_VALUE,
	.TrigEventHighValOff = XAIE2PGBL_PL_MODULE_TIMER_TRIG_EVENT_HIGH_VALUE,
	.LowOff = XAIE2PGBL_PL_MODULE_TIMER_LOW,
	.HighOff = XAIE2PGBL_PL_MODULE_TIMER_HIGH,
	.CtrlOff = XAIE2PGBL_PL_MODULE_TIMER_CONTROL,
	{XAIE2PGBL_PL_MODULE_TIMER_CONTROL_RESET_LSB, XAIE2PGBL_PL_MODULE_TIMER_CONTROL_RESET_MASK},
	{XAIE2PGBL_PL_MODULE_TIMER_CONTROL_RESET_EVENT_LSB, XAIE2PGBL_PL_MODULE_TIMER_CONTROL_RESET_EVENT_MASK}
};

static const XAie_TimerMod Aie2PMemTileTimerMod =
{
	.TrigEventLowValOff = XAIE2PGBL_MEM_TILE_MODULE_TIMER_TRIG_EVENT_LOW_VALUE ,
	.TrigEventHighValOff = XAIE2PGBL_MEM_TILE_MODULE_TIMER_TRIG_EVENT_HIGH_VALUE,
	.LowOff = XAIE2PGBL_MEM_TILE_MODULE_TIMER_LOW,
	.HighOff = XAIE2PGBL_MEM_TILE_MODULE_TIMER_HIGH,
	.CtrlOff = XAIE2PGBL_MEM_TILE_MODULE_TIMER_CONTROL,
	{XAIE2PGBL_MEM_TILE_MODULE_TIMER_CONTROL_RESET_LSB, XAIE2PGBL_MEM_TILE_MODULE_TIMER_CONTROL_RESET_MASK},
	{XAIE2PGBL_MEM_TILE_MODULE_TIMER_CONTROL_RESET_EVENT_LSB, XAIE2PGBL_MEM_TILE_MODULE_TIMER_CONTROL_RESET_EVENT_MASK}
};
#endif /* XAIE_FEATURE_TIMER_ENABLE */

#ifdef XAIE_FEATURE_TRACE_ENABLE
/*
 * Data structure to configure trace event register for XAIE_MEM_MOD module
 * type
 */
static const XAie_RegFldAttr Aie2PMemTraceEvent[] =
{
	{XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT0_TRACE_EVENT0_LSB, XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT0_TRACE_EVENT0_MASK},
	{XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT0_TRACE_EVENT1_LSB, XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT0_TRACE_EVENT1_MASK},
	{XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT0_TRACE_EVENT2_LSB, XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT0_TRACE_EVENT2_MASK},
	{XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT0_TRACE_EVENT3_LSB, XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT0_TRACE_EVENT3_MASK},
	{XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT1_TRACE_EVENT4_LSB, XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT1_TRACE_EVENT4_MASK},
	{XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT1_TRACE_EVENT5_LSB, XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT1_TRACE_EVENT5_MASK},
	{XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT1_TRACE_EVENT6_LSB, XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT1_TRACE_EVENT6_MASK},
	{XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT1_TRACE_EVENT7_LSB, XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT1_TRACE_EVENT7_MASK}
};

/*
 * Data structure to configure trace event register for XAIE_CORE_MOD module
 * type
 */
static const XAie_RegFldAttr Aie2PCoreTraceEvent[] =
{
	{XAIE2PGBL_CORE_MODULE_TRACE_EVENT0_TRACE_EVENT0_LSB, XAIE2PGBL_CORE_MODULE_TRACE_EVENT0_TRACE_EVENT0_MASK},
	{XAIE2PGBL_CORE_MODULE_TRACE_EVENT0_TRACE_EVENT1_LSB, XAIE2PGBL_CORE_MODULE_TRACE_EVENT0_TRACE_EVENT1_MASK},
	{XAIE2PGBL_CORE_MODULE_TRACE_EVENT0_TRACE_EVENT2_LSB, XAIE2PGBL_CORE_MODULE_TRACE_EVENT0_TRACE_EVENT2_MASK},
	{XAIE2PGBL_CORE_MODULE_TRACE_EVENT0_TRACE_EVENT3_LSB, XAIE2PGBL_CORE_MODULE_TRACE_EVENT0_TRACE_EVENT3_MASK},
	{XAIE2PGBL_CORE_MODULE_TRACE_EVENT1_TRACE_EVENT4_LSB, XAIE2PGBL_CORE_MODULE_TRACE_EVENT1_TRACE_EVENT4_MASK},
	{XAIE2PGBL_CORE_MODULE_TRACE_EVENT1_TRACE_EVENT5_LSB, XAIE2PGBL_CORE_MODULE_TRACE_EVENT1_TRACE_EVENT5_MASK},
	{XAIE2PGBL_CORE_MODULE_TRACE_EVENT1_TRACE_EVENT6_LSB, XAIE2PGBL_CORE_MODULE_TRACE_EVENT1_TRACE_EVENT6_MASK},
	{XAIE2PGBL_CORE_MODULE_TRACE_EVENT1_TRACE_EVENT7_LSB, XAIE2PGBL_CORE_MODULE_TRACE_EVENT1_TRACE_EVENT7_MASK}
};

/*
 * Data structure to configure trace event register for XAIE_PL_MOD module
 * type
 */
static const XAie_RegFldAttr Aie2PPlTraceEvent[] =
{
	{XAIE2PGBL_PL_MODULE_TRACE_EVENT0_TRACE_EVENT0_LSB, XAIE2PGBL_PL_MODULE_TRACE_EVENT0_TRACE_EVENT0_MASK},
	{XAIE2PGBL_PL_MODULE_TRACE_EVENT0_TRACE_EVENT1_LSB, XAIE2PGBL_PL_MODULE_TRACE_EVENT0_TRACE_EVENT1_MASK},
	{XAIE2PGBL_PL_MODULE_TRACE_EVENT0_TRACE_EVENT2_LSB, XAIE2PGBL_PL_MODULE_TRACE_EVENT0_TRACE_EVENT2_MASK},
	{XAIE2PGBL_PL_MODULE_TRACE_EVENT0_TRACE_EVENT3_LSB, XAIE2PGBL_PL_MODULE_TRACE_EVENT0_TRACE_EVENT3_MASK},
	{XAIE2PGBL_PL_MODULE_TRACE_EVENT1_TRACE_EVENT4_LSB, XAIE2PGBL_PL_MODULE_TRACE_EVENT1_TRACE_EVENT4_MASK},
	{XAIE2PGBL_PL_MODULE_TRACE_EVENT1_TRACE_EVENT5_LSB, XAIE2PGBL_PL_MODULE_TRACE_EVENT1_TRACE_EVENT5_MASK},
	{XAIE2PGBL_PL_MODULE_TRACE_EVENT1_TRACE_EVENT6_LSB, XAIE2PGBL_PL_MODULE_TRACE_EVENT1_TRACE_EVENT6_MASK},
	{XAIE2PGBL_PL_MODULE_TRACE_EVENT1_TRACE_EVENT7_LSB, XAIE2PGBL_PL_MODULE_TRACE_EVENT1_TRACE_EVENT7_MASK}
};

/*
 * Data structure to configure trace event register for
 * XAIEGBL_TILE_TYPE_MEMTILE type
 */
static const XAie_RegFldAttr Aie2PMemTileTraceEvent[] =
{
	{XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT0_TRACE_EVENT0_LSB, XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT0_TRACE_EVENT0_MASK},
	{XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT0_TRACE_EVENT1_LSB, XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT0_TRACE_EVENT1_MASK},
	{XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT0_TRACE_EVENT2_LSB, XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT0_TRACE_EVENT2_MASK},
	{XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT0_TRACE_EVENT3_LSB, XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT0_TRACE_EVENT3_MASK},
	{XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT1_TRACE_EVENT4_LSB, XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT1_TRACE_EVENT4_MASK},
	{XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT1_TRACE_EVENT5_LSB, XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT1_TRACE_EVENT5_MASK},
	{XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT1_TRACE_EVENT6_LSB, XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT1_TRACE_EVENT6_MASK},
	{XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT1_TRACE_EVENT7_LSB, XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT1_TRACE_EVENT7_MASK}
};

/*
 * Data structure to configure trace module for XAIEGBL_TILE_TYPE_AIETILE tile
 * type
 */
static const XAie_TraceMod Aie2PTileTraceMod[] =
{
	{
		.CtrlRegOff = XAIE2PGBL_MEMORY_MODULE_TRACE_CONTROL0,
		.PktConfigRegOff = XAIE2PGBL_MEMORY_MODULE_TRACE_CONTROL1,
		.StatusRegOff = XAIE2PGBL_MEMORY_MODULE_TRACE_STATUS,
		.EventRegOffs = (u32 []){XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT0, XAIE2PGBL_MEMORY_MODULE_TRACE_EVENT1},
		.NumTraceSlotIds = 8U,
		.NumEventsPerSlot = 4U,
		.StopEvent = {XAIE2PGBL_MEMORY_MODULE_TRACE_CONTROL0_TRACE_STOP_EVENT_LSB, XAIE2PGBL_MEMORY_MODULE_TRACE_CONTROL0_TRACE_STOP_EVENT_MASK},
		.StartEvent = {XAIE2PGBL_MEMORY_MODULE_TRACE_CONTROL0_TRACE_START_EVENT_LSB, XAIE2PGBL_MEMORY_MODULE_TRACE_CONTROL0_TRACE_START_EVENT_MASK},
		.ModeConfig = {XAIE_FEATURE_UNAVAILABLE, XAIE_FEATURE_UNAVAILABLE},
		.PktType = {XAIE2PGBL_MEMORY_MODULE_TRACE_CONTROL1_PACKET_TYPE_LSB, XAIE2PGBL_MEMORY_MODULE_TRACE_CONTROL1_PACKET_TYPE_MASK},
		.PktId = {XAIE2PGBL_MEMORY_MODULE_TRACE_CONTROL1_ID_LSB, XAIE2PGBL_MEMORY_MODULE_TRACE_CONTROL1_ID_MASK},
		.State = {XAIE2PGBL_MEMORY_MODULE_TRACE_STATUS_STATE_LSB, XAIE2PGBL_MEMORY_MODULE_TRACE_STATUS_STATE_MASK},
		.ModeSts = {XAIE2PGBL_MEMORY_MODULE_TRACE_STATUS_MODE_LSB, XAIE2PGBL_MEMORY_MODULE_TRACE_STATUS_MODE_MASK},
		.Event = Aie2PMemTraceEvent
	},
	{
		.CtrlRegOff = XAIE2PGBL_CORE_MODULE_TRACE_CONTROL0,
		.PktConfigRegOff = XAIE2PGBL_CORE_MODULE_TRACE_CONTROL1,
		.StatusRegOff = XAIE2PGBL_CORE_MODULE_TRACE_STATUS,
		.EventRegOffs = (u32 []){XAIE2PGBL_CORE_MODULE_TRACE_EVENT0, XAIE2PGBL_CORE_MODULE_TRACE_EVENT1},
		.NumTraceSlotIds = 8U,
		.NumEventsPerSlot = 4U,
		.StopEvent = {XAIE2PGBL_CORE_MODULE_TRACE_CONTROL0_TRACE_STOP_EVENT_LSB, XAIE2PGBL_CORE_MODULE_TRACE_CONTROL0_TRACE_STOP_EVENT_MASK},
		.StartEvent = {XAIE2PGBL_CORE_MODULE_TRACE_CONTROL0_TRACE_START_EVENT_LSB, XAIE2PGBL_CORE_MODULE_TRACE_CONTROL0_TRACE_START_EVENT_MASK},
		.ModeConfig = {XAIE2PGBL_CORE_MODULE_TRACE_CONTROL0_MODE_LSB, XAIE2PGBL_CORE_MODULE_TRACE_CONTROL0_MODE_MASK},
		.PktType = {XAIE2PGBL_CORE_MODULE_TRACE_CONTROL1_PACKET_TYPE_LSB, XAIE2PGBL_CORE_MODULE_TRACE_CONTROL1_PACKET_TYPE_MASK},
		.PktId = {XAIE2PGBL_CORE_MODULE_TRACE_CONTROL1_ID_LSB, XAIE2PGBL_CORE_MODULE_TRACE_CONTROL1_ID_MASK},
		.State = {XAIE2PGBL_CORE_MODULE_TRACE_STATUS_STATE_LSB, XAIE2PGBL_CORE_MODULE_TRACE_STATUS_STATE_MASK},
		.ModeSts = {XAIE2PGBL_CORE_MODULE_TRACE_STATUS_MODE_LSB, XAIE2PGBL_CORE_MODULE_TRACE_STATUS_MODE_MASK},
		.Event = Aie2PCoreTraceEvent
	}
};

/*
 * Data structure to configure trace module for XAIEGBL_TILE_TYPE_SHIMNOC/PL
 * tile type
 */
static const XAie_TraceMod Aie2PPlTraceMod =
{
	.CtrlRegOff = XAIE2PGBL_PL_MODULE_TRACE_CONTROL0,
	.PktConfigRegOff = XAIE2PGBL_PL_MODULE_TRACE_CONTROL1,
	.StatusRegOff = XAIE2PGBL_PL_MODULE_TRACE_STATUS,
	.EventRegOffs = (u32 []){XAIE2PGBL_PL_MODULE_TRACE_EVENT0, XAIE2PGBL_PL_MODULE_TRACE_EVENT1},
	.NumTraceSlotIds = 8U,
	.NumEventsPerSlot = 4U,
	.StopEvent = {XAIE2PGBL_PL_MODULE_TRACE_CONTROL0_TRACE_STOP_EVENT_LSB, XAIE2PGBL_PL_MODULE_TRACE_CONTROL0_TRACE_STOP_EVENT_MASK},
	.StartEvent = {XAIE2PGBL_PL_MODULE_TRACE_CONTROL0_TRACE_START_EVENT_LSB, XAIE2PGBL_PL_MODULE_TRACE_CONTROL0_TRACE_START_EVENT_MASK},
	.ModeConfig = {XAIE_FEATURE_UNAVAILABLE, XAIE_FEATURE_UNAVAILABLE},
	.PktType = {XAIE2PGBL_PL_MODULE_TRACE_CONTROL1_PACKET_TYPE_LSB, XAIE2PGBL_PL_MODULE_TRACE_CONTROL1_PACKET_TYPE_MASK},
	.PktId = {XAIE2PGBL_PL_MODULE_TRACE_CONTROL1_ID_LSB, XAIE2PGBL_PL_MODULE_TRACE_CONTROL1_ID_MASK},
	.State = {XAIE2PGBL_PL_MODULE_TRACE_STATUS_STATE_LSB, XAIE2PGBL_PL_MODULE_TRACE_STATUS_STATE_MASK},
	.ModeSts = {XAIE2PGBL_PL_MODULE_TRACE_STATUS_MODE_LSB, XAIE2PGBL_PL_MODULE_TRACE_STATUS_MODE_MASK},
	.Event = Aie2PPlTraceEvent
};

/*
 * Data structure to configure trace module for XAIEGBL_TILE_TYPE_MEMTILE
 * tile type
 */
static const XAie_TraceMod Aie2PMemTileTraceMod =
{
	.CtrlRegOff = XAIE2PGBL_MEM_TILE_MODULE_TRACE_CONTROL0,
	.PktConfigRegOff = XAIE2PGBL_MEM_TILE_MODULE_TRACE_CONTROL1,
	.StatusRegOff = XAIE2PGBL_MEM_TILE_MODULE_TRACE_STATUS,
	.EventRegOffs = (u32 []){XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT0, XAIE2PGBL_MEM_TILE_MODULE_TRACE_EVENT1},
	.NumTraceSlotIds = 8U,
	.NumEventsPerSlot = 4U,
	.StopEvent = {XAIE2PGBL_MEM_TILE_MODULE_TRACE_CONTROL0_TRACE_STOP_EVENT_LSB, XAIE2PGBL_MEM_TILE_MODULE_TRACE_CONTROL0_TRACE_STOP_EVENT_MASK},
	.StartEvent = {XAIE2PGBL_MEM_TILE_MODULE_TRACE_CONTROL0_TRACE_START_EVENT_LSB, XAIE2PGBL_MEM_TILE_MODULE_TRACE_CONTROL0_TRACE_START_EVENT_MASK},
	.ModeConfig = {XAIE_FEATURE_UNAVAILABLE, XAIE_FEATURE_UNAVAILABLE},
	.PktType = {XAIE2PGBL_MEM_TILE_MODULE_TRACE_CONTROL1_PACKET_TYPE_LSB, XAIE2PGBL_MEM_TILE_MODULE_TRACE_CONTROL1_PACKET_TYPE_MASK},
	.PktId = {XAIE2PGBL_MEM_TILE_MODULE_TRACE_CONTROL1_ID_LSB, XAIE2PGBL_MEM_TILE_MODULE_TRACE_CONTROL1_ID_MASK},
	.State = {XAIE2PGBL_MEM_TILE_MODULE_TRACE_STATUS_STATE_LSB, XAIE2PGBL_MEM_TILE_MODULE_TRACE_STATUS_STATE_MASK},
	.ModeSts = {XAIE2PGBL_MEM_TILE_MODULE_TRACE_STATUS_MODE_LSB, XAIE2PGBL_MEM_TILE_MODULE_TRACE_STATUS_MODE_MASK},
	.Event = Aie2PMemTileTraceEvent
};
#endif /* XAIE_FEATURE_TRACE_ENABLE */

#ifdef XAIE_FEATURE_INTR_L1_ENABLE
/*
 * Data structure to configures first level interrupt controller for
 * XAIEGBL_TILE_TYPE_SHIMPL tile type
 */
static const XAie_L1IntrMod Aie2PPlL1IntrMod =
{
	.BaseEnableRegOff = XAIE2PGBL_PL_MODULE_INTERRUPT_CONTROLLER_1ST_LEVEL_ENABLE_A,
	.BaseDisableRegOff = XAIE2PGBL_PL_MODULE_INTERRUPT_CONTROLLER_1ST_LEVEL_DISABLE_A,
	.BaseIrqRegOff = XAIE2PGBL_PL_MODULE_INTERRUPT_CONTROLLER_1ST_LEVEL_IRQ_NO_A,
	.BaseIrqEventRegOff = XAIE2PGBL_PL_MODULE_INTERRUPT_CONTROLLER_1ST_LEVEL_IRQ_EVENT_A,
	.BaseIrqEventMask = XAIE2PGBL_PL_MODULE_INTERRUPT_CONTROLLER_1ST_LEVEL_IRQ_EVENT_A_IRQ_EVENT0_MASK,
	.BaseBroadcastBlockRegOff = XAIE2PGBL_PL_MODULE_INTERRUPT_CONTROLLER_1ST_LEVEL_BLOCK_NORTH_IN_A_SET,
	.BaseBroadcastUnblockRegOff = XAIE2PGBL_PL_MODULE_INTERRUPT_CONTROLLER_1ST_LEVEL_BLOCK_NORTH_IN_A_CLEAR,
	.SwOff = 0x30U,
	.NumIntrIds = 20U,
	.NumIrqEvents = 4U,
	.IrqEventOff = 8U,
	.NumBroadcastIds = 16U,
	.MaxErrorBcIdsRvd = 4U,
#ifdef XAIE_FEATURE_INTR_INIT_ENABLE
	.IntrCtrlL1IrqId = &_XAieMl_IntrCtrlL1IrqId,
#else
	.IntrCtrlL1IrqId = NULL,
#endif
};
#endif /* XAIE_FEATURE_INTR_L1_ENABLE */

#ifdef XAIE_FEATURE_INTR_L2_ENABLE
/*
 * Data structure to configures second level interrupt controller for
 * XAIEGBL_TILE_TYPE_SHIMNOC tile type
 */
static const XAie_L2IntrMod Aie2PNoCL2IntrMod =
{
	.EnableRegOff = XAIE2PGBL_NOC_MODULE_INTERRUPT_CONTROLLER_2ND_LEVEL_ENABLE,
	.DisableRegOff = XAIE2PGBL_NOC_MODULE_INTERRUPT_CONTROLLER_2ND_LEVEL_DISABLE,
	.IrqRegOff = XAIE2PGBL_NOC_MODULE_INTERRUPT_CONTROLLER_2ND_LEVEL_INTERRUPT,
	.NumBroadcastIds = 16U,
	.NumNoCIntr = 4U,
};
#endif /* XAIE_FEATURE_INTR_L2_ENABLE */

#ifdef XAIE_FEATURE_PRIVILEGED_ENABLE
/*
 * Data structure to configures tile control for
 * XAIEGBL_TILE_TYPE_AIETILE tile type
 */
static const XAie_TileCtrlMod Aie2PCoreTileCtrlMod =
{
	.TileCtrlRegOff = XAIE2PGBL_CORE_MODULE_TILE_CONTROL,
	.IsolateEast = {XAIE2PGBL_CORE_MODULE_TILE_CONTROL_ISOLATE_FROM_EAST_LSB, XAIE2PGBL_CORE_MODULE_TILE_CONTROL_ISOLATE_FROM_EAST_MASK},
	.IsolateNorth = {XAIE2PGBL_CORE_MODULE_TILE_CONTROL_ISOLATE_FROM_NORTH_LSB, XAIE2PGBL_CORE_MODULE_TILE_CONTROL_ISOLATE_FROM_NORTH_MASK},
	.IsolateWest = {XAIE2PGBL_CORE_MODULE_TILE_CONTROL_ISOLATE_FROM_WEST_LSB, XAIE2PGBL_CORE_MODULE_TILE_CONTROL_ISOLATE_FROM_WEST_MASK},
	.IsolateSouth = {XAIE2PGBL_CORE_MODULE_TILE_CONTROL_ISOLATE_FROM_SOUTH_LSB, XAIE2PGBL_CORE_MODULE_TILE_CONTROL_ISOLATE_FROM_SOUTH_MASK},
	.IsolateDefaultOn = XAIE_ENABLE,
};

/*
 * Data structure to configures tile control for
 * XAIEGBL_TILE_TYPE_MEMTILE tile type
 */
static const XAie_TileCtrlMod Aie2PMemTileCtrlMod =
{
	.TileCtrlRegOff = XAIE2PGBL_MEM_TILE_MODULE_TILE_CONTROL,
	.IsolateEast = {XAIE2PGBL_MEM_TILE_MODULE_TILE_CONTROL_ISOLATE_FROM_EAST_LSB, XAIE2PGBL_MEM_TILE_MODULE_TILE_CONTROL_ISOLATE_FROM_EAST_MASK},
	.IsolateNorth = {XAIE2PGBL_MEM_TILE_MODULE_TILE_CONTROL_ISOLATE_FROM_NORTH_LSB, XAIE2PGBL_MEM_TILE_MODULE_TILE_CONTROL_ISOLATE_FROM_NORTH_MASK},
	.IsolateWest = {XAIE2PGBL_MEM_TILE_MODULE_TILE_CONTROL_ISOLATE_FROM_WEST_LSB, XAIE2PGBL_MEM_TILE_MODULE_TILE_CONTROL_ISOLATE_FROM_WEST_MASK},
	.IsolateSouth = {XAIE2PGBL_MEM_TILE_MODULE_TILE_CONTROL_ISOLATE_FROM_SOUTH_LSB, XAIE2PGBL_MEM_TILE_MODULE_TILE_CONTROL_ISOLATE_FROM_SOUTH_MASK},
	.IsolateDefaultOn = XAIE_ENABLE,
};

/*
 * Data structure to configures tile control for
 * XAIEGBL_TILE_TYPE_SHIMPL/NOC tile type
 */
static const XAie_TileCtrlMod Aie2PShimTileCtrlMod =
{
	.TileCtrlRegOff = XAIE2PGBL_PL_MODULE_TILE_CONTROL,
	.IsolateEast = {XAIE2PGBL_PL_MODULE_TILE_CONTROL_ISOLATE_FROM_EAST_LSB, XAIE2PGBL_PL_MODULE_TILE_CONTROL_ISOLATE_FROM_EAST_MASK},
	.IsolateNorth = {XAIE2PGBL_PL_MODULE_TILE_CONTROL_ISOLATE_FROM_NORTH_LSB, XAIE2PGBL_PL_MODULE_TILE_CONTROL_ISOLATE_FROM_NORTH_MASK},
	.IsolateWest = {XAIE2PGBL_PL_MODULE_TILE_CONTROL_ISOLATE_FROM_WEST_LSB, XAIE2PGBL_PL_MODULE_TILE_CONTROL_ISOLATE_FROM_WEST_MASK},
	.IsolateSouth = {XAIE2PGBL_PL_MODULE_TILE_CONTROL_ISOLATE_FROM_SOUTH_LSB, XAIE2PGBL_PL_MODULE_TILE_CONTROL_ISOLATE_FROM_SOUTH_MASK},
	.IsolateDefaultOn = XAIE_ENABLE,
};

/*
 * Data structure to configures memory control for
 * XAIEGBL_TILE_TYPE_AIETILE tile type
 */
static const XAie_MemCtrlMod Aie2PTileMemCtrlMod[] =
{
	{
		.MemCtrlRegOff = XAIE2PGBL_MEMORY_MODULE_MEMORY_CONTROL,
		.MemZeroisation = {XAIE2PGBL_MEMORY_MODULE_MEMORY_CONTROL_MEMORY_ZEROISATION_LSB, XAIE2PGBL_MEMORY_MODULE_MEMORY_CONTROL_MEMORY_ZEROISATION_MASK},
	},
	{
		.MemCtrlRegOff = XAIE2PGBL_CORE_MODULE_MEMORY_CONTROL,
		.MemZeroisation = {XAIE2PGBL_CORE_MODULE_MEMORY_CONTROL_MEMORY_ZEROISATION_LSB, XAIE2PGBL_CORE_MODULE_MEMORY_CONTROL_MEMORY_ZEROISATION_MASK},
	},
};

/*
 * Data structure to configures memory control for
 * XAIEGBL_TILE_TYPE_MEMTILE tile type
 */
static const XAie_MemCtrlMod Aie2PMemTileMemCtrlMod =
{
	.MemCtrlRegOff = XAIE2PGBL_MEM_TILE_MODULE_MEMORY_CONTROL,
	.MemZeroisation = {XAIE2PGBL_MEM_TILE_MODULE_MEMORY_CONTROL_MEMORY_ZEROISATION_LSB, XAIE2PGBL_MEM_TILE_MODULE_MEMORY_CONTROL_MEMORY_ZEROISATION_MASK},
	.MemInterleaving = {XAIE2PGBL_MEM_TILE_MODULE_MEMORY_CONTROL_MEMORY_INTERLEAVING_LSB, XAIE2PGBL_MEM_TILE_MODULE_MEMORY_CONTROL_MEMORY_INTERLEAVING_MASK},
};
#endif /* XAIE_FEATURE_PRIVILEGED_ENABLE */

#ifdef XAIE_FEATURE_CORE_ENABLE
	#define AIE2PCOREMOD &Aie2PCoreMod
#else
	#define AIE2PCOREMOD NULL
#endif
#ifdef XAIE_FEATURE_SS_ENABLE
	#define AIE2PTILESTRMSW &Aie2PTileStrmSw
	#define AIE2PSHIMSTRMSW &Aie2PShimStrmSw
	#define AIE2PMEMTILESTRMSW &Aie2PMemTileStrmSw
#else
	#define AIE2PTILESTRMSW NULL
	#define AIE2PSHIMSTRMSW NULL
	#define AIE2PMEMTILESTRMSW NULL
#endif
#ifdef XAIE_FEATURE_DMA_ENABLE
	#define AIE2PTILEDMAMOD &Aie2PTileDmaMod
	#define AIE2PSHIMDMAMOD &Aie2PShimDmaMod
	#define AIE2PMEMTILEDMAMOD &Aie2PMemTileDmaMod
#else
	#define AIE2PTILEDMAMOD NULL
	#define AIE2PSHIMDMAMOD NULL
	#define AIE2PMEMTILEDMAMOD NULL
#endif
#ifdef XAIE_FEATURE_DATAMEM_ENABLE
	#define AIE2PTILEMEMMOD &Aie2PTileMemMod
	#define AIE2PMEMTILEMEMMOD &Aie2PMemTileMemMod
#else
	#define AIE2PTILEMEMMOD NULL
	#define AIE2PMEMTILEMEMMOD NULL
#endif
#ifdef XAIE_FEATURE_PL_ENABLE
	#define AIE2PSHIMTILEPLIFMOD &Aie2PShimTilePlIfMod
	#define AIE2PPLIFMOD &Aie2PPlIfMod
#else
	#define AIE2PSHIMTILEPLIFMOD NULL
	#define AIE2PPLIFMOD NULL
#endif
#ifdef XAIE_FEATURE_LOCK_ENABLE
	#define AIE2PTILELOCKMOD &Aie2PTileLockMod
	#define AIE2PSHIMNOCLOCKMOD &Aie2PShimNocLockMod
	#define AIE2PMEMTILELOCKMOD &Aie2PMemTileLockMod
#else
	#define AIE2PTILELOCKMOD NULL
	#define AIE2PSHIMNOCLOCKMOD NULL
	#define AIE2PMEMTILELOCKMOD NULL
#endif
#ifdef XAIE_FEATURE_PERFCOUNT_ENABLE
	#define AIE2PTILEPERFCNT Aie2PTilePerfCnt
	#define AIE2PPLPERFCNT &Aie2PPlPerfCnt
	#define AIE2PMEMTILEPERFCNT &Aie2PMemTilePerfCnt
#else
	#define AIE2PTILEPERFCNT NULL
	#define AIE2PPLPERFCNT NULL
	#define AIE2PMEMTILEPERFCNT NULL
#endif
#ifdef XAIE_FEATURE_EVENTS_ENABLE
	#define AIE2PTILEEVNTMOD Aie2PTileEvntMod
	#define AIE2PNOCEVNTMOD &Aie2PNocEvntMod
	#define AIE2PPLEVNTMOD &Aie2PPlEvntMod
	#define AIE2PMEMTILEEVNTMOD &Aie2PMemTileEvntMod
#else
	#define AIE2PTILEEVNTMOD NULL
	#define AIE2PNOCEVNTMOD NULL
	#define AIE2PPLEVNTMOD NULL
	#define AIE2PMEMTILEEVNTMOD NULL
#endif
#ifdef XAIE_FEATURE_TIMER_ENABLE
	#define AIE2PTILETIMERMOD Aie2PTileTimerMod
	#define AIE2PPLTIMERMOD &Aie2PPlTimerMod
	#define AIE2PMEMTILETIMERMOD &Aie2PMemTileTimerMod
#else
	#define AIE2PTILETIMERMOD NULL
	#define AIE2PPLTIMERMOD NULL
	#define AIE2PMEMTILETIMERMOD NULL
#endif
#ifdef XAIE_FEATURE_TRACE_ENABLE
	#define AIE2PTILETRACEMOD Aie2PTileTraceMod
	#define AIE2PPLTRACEMOD &Aie2PPlTraceMod
	#define AIE2PMEMTILETRACEMOD &Aie2PMemTileTraceMod
#else
	#define AIE2PTILETRACEMOD NULL
	#define AIE2PPLTRACEMOD NULL
	#define AIE2PMEMTILETRACEMOD NULL
#endif
#ifdef XAIE_FEATURE_INTR_L1_ENABLE
	#define AIEMLPLL1INTRMOD &Aie2PPlL1IntrMod
#else
	#define AIEMLPLL1INTRMOD NULL
#endif
#ifdef XAIE_FEATURE_INTR_L2_ENABLE
	#define AIEMLNOCL2INTRMOD &Aie2PNoCL2IntrMod
#else
	#define AIEMLNOCL2INTRMOD NULL
#endif
#ifdef XAIE_FEATURE_PRIVILEGED_ENABLE
	#define AIEMLCORETILECTRLMOD &Aie2PCoreTileCtrlMod
	#define AIEMLTILEMEMCTRLMOD Aie2PTileMemCtrlMod
	#define AIEMLSHIMTILECTRLMOD &Aie2PShimTileCtrlMod
	#define AIEMLMEMTILECTRLMOD &Aie2PMemTileCtrlMod
	#define AIEMLMEMTILEMEMCTRLMOD &Aie2PMemTileMemCtrlMod
#else
	#define AIEMLCORETILECTRLMOD NULL
	#define AIEMLTILEMEMCTRLMOD NULL
	#define AIEMLSHIMTILECTRLMOD NULL
	#define AIEMLMEMTILECTRLMOD NULL
	#define AIEMLMEMTILEMEMCTRLMOD NULL
#endif

/*
 * AIE2P Module
 * This data structure captures all the modules for each tile type.
 * Depending on the tile type, this data strcuture can be used to access all
 * hardware properties of individual modules.
 */
XAie_TileMod Aie2PMod[] =
{
	{
		/*
		 * AIE2P Tile Module indexed using XAIEGBL_TILE_TYPE_AIETILE
		 */
		.NumModules = 2U,
		.CoreMod = AIE2PCOREMOD,
		.StrmSw  = AIE2PTILESTRMSW,
		.DmaMod  = AIE2PTILEDMAMOD,
		.MemMod  = AIE2PTILEMEMMOD,
		.PlIfMod = NULL,
		.LockMod = AIE2PTILELOCKMOD,
		.PerfMod = AIE2PTILEPERFCNT,
		.EvntMod = AIE2PTILEEVNTMOD,
		.TimerMod = AIE2PTILETIMERMOD,
		.TraceMod = AIE2PTILETRACEMOD,
		.L1IntrMod = NULL,
		.L2IntrMod = NULL,
		.TileCtrlMod = AIEMLCORETILECTRLMOD,
		.MemCtrlMod = AIEMLTILEMEMCTRLMOD,
	},
	{
		/*
		 * AIE2P Shim Noc Module indexed using XAIEGBL_TILE_TYPE_SHIMNOC
		 */
		.NumModules = 1U,
		.CoreMod = NULL,
		.StrmSw  = AIE2PSHIMSTRMSW,
		.DmaMod  = AIE2PSHIMDMAMOD,
		.MemMod  = NULL,
		.PlIfMod = AIE2PSHIMTILEPLIFMOD,
		.LockMod = AIE2PSHIMNOCLOCKMOD,
		.PerfMod = AIE2PPLPERFCNT,
		.EvntMod = AIE2PNOCEVNTMOD,
		.TimerMod = AIE2PPLTIMERMOD,
		.TraceMod = AIE2PPLTRACEMOD,
		.L1IntrMod = AIEMLPLL1INTRMOD,
		.L2IntrMod = AIEMLNOCL2INTRMOD,
		.TileCtrlMod = AIEMLSHIMTILECTRLMOD,
		.MemCtrlMod = NULL,
	},
	{
		/*
		 * AIE2P Shim PL Module indexed using XAIEGBL_TILE_TYPE_SHIMPL
		 */
		.NumModules = 1U,
		.CoreMod = NULL,
		.StrmSw  = AIE2PSHIMSTRMSW,
		.DmaMod  = NULL,
		.MemMod  = NULL,
		.PlIfMod = AIE2PPLIFMOD,
		.LockMod = NULL,
		.PerfMod = AIE2PPLPERFCNT,
		.EvntMod = AIE2PPLEVNTMOD,
		.TimerMod = AIE2PPLTIMERMOD,
		.TraceMod = AIE2PPLTRACEMOD,
		.L1IntrMod = AIEMLPLL1INTRMOD,
		.L2IntrMod = NULL,
		.TileCtrlMod = AIEMLSHIMTILECTRLMOD,
		.MemCtrlMod = NULL,
	},
	{
		/*
		 * AIE2P MemTile Module indexed using XAIEGBL_TILE_TYPE_MEMTILE
		 */
		.NumModules = 1U,
		.CoreMod = NULL,
		.StrmSw  = AIE2PMEMTILESTRMSW,
		.DmaMod  = AIE2PMEMTILEDMAMOD,
		.MemMod  = AIE2PMEMTILEMEMMOD,
		.PlIfMod = NULL,
		.LockMod = AIE2PMEMTILELOCKMOD,
		.PerfMod = AIE2PMEMTILEPERFCNT,
		.EvntMod = AIE2PMEMTILEEVNTMOD,
		.TimerMod = AIE2PMEMTILETIMERMOD,
		.TraceMod = AIE2PMEMTILETRACEMOD,
		.L1IntrMod = NULL,
		.L2IntrMod = NULL,
		.TileCtrlMod = AIEMLMEMTILECTRLMOD,
		.MemCtrlMod = AIEMLMEMTILEMEMCTRLMOD,
	},
};

/* Device level operations for aieml */
XAie_DeviceOps Aie2PDevOps =
{
	.IsCheckerBoard = 0U,
	.TilesInUse = Aie2PTilesInUse,
	.MemInUse = Aie2PMemInUse,
	.CoreInUse = Aie2PCoreInUse,
	.GetTTypefromLoc = &_XAie2p_GetTTypefromLoc,
#ifdef XAIE_FEATURE_PRIVILEGED_ENABLE
	.SetPartColShimReset = &_XAieMl_SetPartColShimReset,
	.SetPartColClockAfterRst = &_XAieMl_SetPartColClockAfterRst,
	.SetPartIsolationAfterRst = &_XAieMl_SetPartIsolationAfterRst,
	.PartMemZeroInit = &_XAieMl_PartMemZeroInit,
	.RequestTiles = &_XAieMl_RequestTiles,
#else
	.SetPartColShimReset = NULL,
	.SetPartColClockAfterRst = NULL,
	.SetPartIsolationAfterRst = NULL,
	.PartMemZeroInit = NULL,
	.RequestTiles = NULL,
#endif
};

/** @} */
