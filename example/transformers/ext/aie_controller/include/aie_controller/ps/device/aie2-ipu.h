/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __AIE2_IPU_H__
#define __AIE2_IPU_H__

#define HW_GEN                   XAIE_DEV_GEN_AIE2IPU
#define XAIE_NUM_ROWS            6
#define XAIE_NUM_COLS            5
#ifdef __AIE2IPU_BASE_ADDR_CDO__
#define XAIE_BASE_ADDR           0x40000000
#else
#define XAIE_BASE_ADDR           0x20000000000
#endif
#define XAIE_COL_SHIFT           25
#define XAIE_ROW_SHIFT           20
#define XAIE_SHIM_ROW            0
#define XAIE_MEM_TILE_ROW_START  1
#define XAIE_MEM_TILE_NUM_ROWS   1
#define XAIE_AIE_TILE_ROW_START  2
#define XAIE_AIE_TILE_NUM_ROWS   4
#define FOR_WRITE                0
#define FOR_READ                 1

#endif