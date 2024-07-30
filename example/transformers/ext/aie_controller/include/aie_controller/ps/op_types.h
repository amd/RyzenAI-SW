/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __OP_TYPES_H__
#define __OP_TYPES_H__

#ifdef __cplusplus
extern "C" {
#include "xaiengine.h"
}
#else
#include "xaiengine.h"
#endif

#include "op_defs.h"
#include "op_base.h"

#define DEBUG_STR_MAX_LEN 512

typedef struct {
    op_base b;
} transaction_op_t;

typedef struct{
    op_base b;
    XAie_LocType tileLoc;
    short channelNum;
    XAie_DmaDirection dma_direction;
} wait_op_t;

typedef struct {
    op_base b;
    XAie_LocType tileLoc;
    short channelNum;
    XAie_DmaDirection dma_direction;
    u8 pendingBDThres;
} pendingBDCount_op_t;

typedef struct {
    op_base b;
    char msg[DEBUG_STR_MAX_LEN];
} print_op_t;

typedef struct {
    uint32_t word;
    uint32_t config;
} tct_op_t;

typedef struct {
    op_base b;
    u32 action;
    u64 regaddr; // regaddr to patch
    u64 argidx;  // kernel arg idx to get value to write at regaddr
    u64 argplus; // value to add to what's passed @ argidx (e.g., offset to shim addr)
} patch_op_t;

/*
 * Structs for reading registers
 */
typedef struct {
    uint64_t address;
} register_data_t;

typedef struct {
    uint32_t count;
    register_data_t data[1]; // variable size
} read_register_op_t;

/*
 * Struct for record timer identifier
 */
typedef struct {
    uint32_t id;
} record_timer_op_t;

#endif 
