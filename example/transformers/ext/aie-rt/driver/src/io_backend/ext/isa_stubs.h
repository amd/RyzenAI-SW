/******************************************************************************
* Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#ifndef _ISA_STUBS_H_
#define _ISA_STUBS_H_

#include <stdint.h>

// macro define

//uc dma bd length
#define UC_DMA_BD_SIZE 0x10
//task page size
#define TASK_PAGE_SIZE 0x2000
//task page header length
#define PAGE_HEADER_SIZE 0x10
//alignment of data section in a page
#define DATA_SECTION_ALIGNMENT 0x10
//minimum length of an ucdma
#define UC_DMA_WORD_LEN 0x04

// Op codes

#define ISA_OPCODE_START_JOB 0x00
#define ISA_OPCODE_UC_DMA_WRITE_DES 0x01
#define ISA_OPCODE_WAIT_UC_DMA 0x02
#define ISA_OPCODE_MASK_WRITE_32 0x03
#define ISA_OPCODE_WRITE_32 0x05
#define ISA_OPCODE_WAIT_TCTS 0x06
#define ISA_OPCODE_END_JOB 0x07
#define ISA_OPCODE_YIELD 0x08
#define ISA_OPCODE_UC_DMA_WRITE_DES_SYNC 0x09
#define ISA_OPCODE_WRITE_32_D 0x0b
#define ISA_OPCODE_READ_32 0x0c
#define ISA_OPCODE_READ_32_D 0x0d
#define ISA_OPCODE_APPLY_OFFSET_57 0x0e
#define ISA_OPCODE_ADD 0x0f
#define ISA_OPCODE_MOV 0x10
#define ISA_OPCODE_LOCAL_BARRIER 0x11
#define ISA_OPCODE_REMOTE_BARRIER 0x12
#define ISA_OPCODE_EOF 0xff
#define ISA_OPCODE_POLL_32 0x13
#define ISA_OPCODE_MASK_POLL_32 0x14
#define ISA_OPCODE_TRACE 0x15
#define ISA_OPCODE_NOP 0x16


// Operation sizes

#define ISA_OPSIZE_START_JOB 0x08
#define ISA_OPSIZE_UC_DMA_WRITE_DES 0x08
#define ISA_OPSIZE_WAIT_UC_DMA 0x04
#define ISA_OPSIZE_MASK_WRITE_32 0x10
#define ISA_OPSIZE_WRITE_32 0x0c
#define ISA_OPSIZE_WAIT_TCTS 0x08
#define ISA_OPSIZE_END_JOB 0x04
#define ISA_OPSIZE_YIELD 0x04
#define ISA_OPSIZE_UC_DMA_WRITE_DES_SYNC 0x04
#define ISA_OPSIZE_WRITE_32_D 0x0c
#define ISA_OPSIZE_READ_32 0x08
#define ISA_OPSIZE_READ_32_D 0x04
#define ISA_OPSIZE_APPLY_OFFSET_57 0x08
#define ISA_OPSIZE_ADD 0x08
#define ISA_OPSIZE_MOV 0x08
#define ISA_OPSIZE_LOCAL_BARRIER 0x04
#define ISA_OPSIZE_REMOTE_BARRIER 0x08
#define ISA_OPSIZE_EOF 0x04
#define ISA_OPSIZE_POLL_32 0x0c
#define ISA_OPSIZE_MASK_POLL_32 0x10
#define ISA_OPSIZE_TRACE 0x04
#define ISA_OPSIZE_NOP 0x04

#ifdef CERT_FW
// Operation implementation forward declarations

static unsigned int control_op_start_job(const uint8_t *_pc, uint8_t job_id, uint16_t size);
static unsigned int control_op_uc_dma_write_des(const uint8_t *_pc, uint8_t wait_handle_reg, uint16_t descriptor_ptr);
static unsigned int control_op_wait_uc_dma(const uint8_t *_pc, uint8_t wait_handle_reg);
static unsigned int control_op_mask_write_32(const uint8_t *_pc, uint32_t address, uint32_t mask, uint32_t value);
static unsigned int control_op_write_32(const uint8_t *_pc, uint32_t address, uint32_t value);
static unsigned int control_op_wait_tcts(const uint8_t *_pc, uint16_t tile_id, uint8_t actor_id, uint8_t target_tcts);
static unsigned int control_op_end_job(const uint8_t *_pc);
static unsigned int control_op_yield(const uint8_t *_pc);
static unsigned int control_op_uc_dma_write_des_sync(const uint8_t *_pc, uint16_t descriptor_ptr);
static unsigned int control_op_write_32_d(const uint8_t *_pc, uint8_t flags, uint32_t address, uint32_t value);
static unsigned int control_op_read_32(const uint8_t *_pc, uint8_t value_reg, uint32_t address);
static unsigned int control_op_read_32_d(const uint8_t *_pc, uint8_t address_reg, uint8_t value_reg);
static unsigned int control_op_apply_offset_57(const uint8_t *_pc, uint16_t table_ptr, uint16_t num_entries, uint8_t offset_high_reg, uint8_t offset_low_reg);
static unsigned int control_op_add(const uint8_t *_pc, uint8_t dest_reg, uint32_t value);
static unsigned int control_op_mov(const uint8_t *_pc, uint8_t dest_reg, uint32_t value);
static unsigned int control_op_local_barrier(const uint8_t *_pc, uint8_t local_barrier_id, uint8_t num_participants);
static unsigned int control_op_remote_barrier(const uint8_t *_pc, uint8_t remote_barrier_id, uint32_t party_mask);
static unsigned int control_op_eof(const uint8_t *_pc);
static unsigned int control_op_poll_32(const uint8_t *_pc, uint32_t address, uint32_t value);
static unsigned int control_op_mask_poll_32(const uint8_t *_pc, uint32_t address, uint32_t mask, uint32_t value);
static unsigned int control_op_trace(const uint8_t *_pc, uint16_t info);
static unsigned int control_op_nop(const uint8_t *_pc);


// Dispatchers

static inline unsigned int control_dispatch_start_job(const uint8_t *pc)
{
  return control_op_start_job(
    pc,
    /* job_id (const) */ *(uint8_t *)(&pc[2]),
    /* size (jobsize) */ *(uint16_t *)(&pc[4])
  );
}

static inline unsigned int control_dispatch_uc_dma_write_des(const uint8_t *pc)
{
  return control_op_uc_dma_write_des(
    pc,
    /* wait_handle (register) */ *(uint8_t *)(&pc[2]),
    /* descriptor_ptr (const) */ *(uint16_t *)(&pc[4])
  );
}

static inline unsigned int control_dispatch_wait_uc_dma(const uint8_t *pc)
{
  return control_op_wait_uc_dma(
    pc,
    /* wait_handle (register) */ *(uint8_t *)(&pc[2])
  );
}

static inline unsigned int control_dispatch_mask_write_32(const uint8_t *pc)
{
  return control_op_mask_write_32(
    pc,
    /* address (const) */ *(uint32_t *)(&pc[4]),
    /* mask (const) */ *(uint32_t *)(&pc[8]),
    /* value (const) */ *(uint32_t *)(&pc[12])
  );
}

static inline unsigned int control_dispatch_write_32(const uint8_t *pc)
{
  return control_op_write_32(
    pc,
    /* address (const) */ *(uint32_t *)(&pc[4]),
    /* value (const) */ *(uint32_t *)(&pc[8])
  );
}

static inline unsigned int control_dispatch_wait_tcts(const uint8_t *pc)
{
  return control_op_wait_tcts(
    pc,
    /* tile_id (const) */ *(uint16_t *)(&pc[2]),
    /* actor_id (const) */ *(uint8_t *)(&pc[4]),
    /* target_tcts (const) */ *(uint8_t *)(&pc[6])
  );
}

static inline unsigned int control_dispatch_end_job(const uint8_t *pc)
{
  return control_op_end_job(
    pc
  );
}

static inline unsigned int control_dispatch_yield(const uint8_t *pc)
{
  return control_op_yield(
    pc
  );
}

static inline unsigned int control_dispatch_uc_dma_write_des_sync(const uint8_t *pc)
{
  return control_op_uc_dma_write_des_sync(
    pc,
    /* descriptor_ptr (const) */ *(uint16_t *)(&pc[2])
  );
}

static inline unsigned int control_dispatch_write_32_d(const uint8_t *pc)
{
  return control_op_write_32_d(
    pc,
    /* flags (const) */ *(uint8_t *)(&pc[2]),
    /* address (const) */ *(uint32_t *)(&pc[4]),
    /* value (const) */ *(uint32_t *)(&pc[8])
  );
}

static inline unsigned int control_dispatch_read_32(const uint8_t *pc)
{
  return control_op_read_32(
    pc,
    /* value (register) */ *(uint8_t *)(&pc[2]),
    /* address (const) */ *(uint32_t *)(&pc[4])
  );
}

static inline unsigned int control_dispatch_read_32_d(const uint8_t *pc)
{
  return control_op_read_32_d(
    pc,
    /* address (register) */ *(uint8_t *)(&pc[2]),
    /* value (register) */ *(uint8_t *)(&pc[3])
  );
}

static inline unsigned int control_dispatch_apply_offset_57(const uint8_t *pc)
{
  return control_op_apply_offset_57(
    pc,
    /* table_ptr (const) */ *(uint16_t *)(&pc[2]),
    /* num_entries (const) */ *(uint16_t *)(&pc[4]),
    /* offset_high (register) */ *(uint8_t *)(&pc[6]),
    /* offset_low (register) */ *(uint8_t *)(&pc[7])
  );
}

static inline unsigned int control_dispatch_add(const uint8_t *pc)
{
  return control_op_add(
    pc,
    /* dest (register) */ *(uint8_t *)(&pc[2]),
    /* value (const) */ *(uint32_t *)(&pc[4])
  );
}

static inline unsigned int control_dispatch_mov(const uint8_t *pc)
{
  return control_op_mov(
    pc,
    /* dest (register) */ *(uint8_t *)(&pc[2]),
    /* value (const) */ *(uint32_t *)(&pc[4])
  );
}

static inline unsigned int control_dispatch_local_barrier(const uint8_t *pc)
{
  return control_op_local_barrier(
    pc,
    /* local_barrier_id (barrier) */ *(uint8_t *)(&pc[2]),
    /* num_participants (const) */ *(uint8_t *)(&pc[3])
  );
}

static inline unsigned int control_dispatch_remote_barrier(const uint8_t *pc)
{
  return control_op_remote_barrier(
    pc,
    /* remote_barrier_id (barrier) */ *(uint8_t *)(&pc[2]),
    /* party_mask (const) */ *(uint32_t *)(&pc[4])
  );
}

static inline unsigned int control_dispatch_eof(const uint8_t *pc)
{
  return control_op_eof(
    pc
  );
}

static inline unsigned int control_dispatch_poll_32(const uint8_t *pc)
{
  return control_op_poll_32(
    pc,
    /* address (const) */ *(uint32_t *)(&pc[4]),
    /* value (const) */ *(uint32_t *)(&pc[8])
  );
}

static inline unsigned int control_dispatch_mask_poll_32(const uint8_t *pc)
{
  return control_op_mask_poll_32(
    pc,
    /* address (const) */ *(uint32_t *)(&pc[4]),
    /* mask (const) */ *(uint32_t *)(&pc[8]),
    /* value (const) */ *(uint32_t *)(&pc[12])
  );
}

static inline unsigned int control_dispatch_trace(const uint8_t *pc)
{
  return control_op_trace(
    pc,
    /* info (const) */ *(uint16_t *)(&pc[2])
  );
}

static inline unsigned int control_dispatch_nop(const uint8_t *pc)
{
  return control_op_nop(
    pc
  );
}


// Case statements for regular operations

#define DISPATCH_REGULAR_OPS \
  case ISA_OPCODE_UC_DMA_WRITE_DES: pc += control_dispatch_uc_dma_write_des(pc); break; \
  case ISA_OPCODE_WAIT_UC_DMA: pc += control_dispatch_wait_uc_dma(pc); break; \
  case ISA_OPCODE_MASK_WRITE_32: pc += control_dispatch_mask_write_32(pc); break; \
  case ISA_OPCODE_WRITE_32: pc += control_dispatch_write_32(pc); break; \
  case ISA_OPCODE_WAIT_TCTS: pc += control_dispatch_wait_tcts(pc); break; \
  case ISA_OPCODE_YIELD: pc += control_dispatch_yield(pc); break; \
  case ISA_OPCODE_UC_DMA_WRITE_DES_SYNC: pc += control_dispatch_uc_dma_write_des_sync(pc); break; \
  case ISA_OPCODE_WRITE_32_D: pc += control_dispatch_write_32_d(pc); break; \
  case ISA_OPCODE_READ_32: pc += control_dispatch_read_32(pc); break; \
  case ISA_OPCODE_READ_32_D: pc += control_dispatch_read_32_d(pc); break; \
  case ISA_OPCODE_APPLY_OFFSET_57: pc += control_dispatch_apply_offset_57(pc); break; \
  case ISA_OPCODE_ADD: pc += control_dispatch_add(pc); break; \
  case ISA_OPCODE_MOV: pc += control_dispatch_mov(pc); break; \
  case ISA_OPCODE_LOCAL_BARRIER: pc += control_dispatch_local_barrier(pc); break; \
  case ISA_OPCODE_REMOTE_BARRIER: pc += control_dispatch_remote_barrier(pc); break; \
  case ISA_OPCODE_POLL_32: pc += control_dispatch_poll_32(pc); break; \
  case ISA_OPCODE_MASK_POLL_32: pc += control_dispatch_mask_poll_32(pc); break; \
  case ISA_OPCODE_TRACE: pc += control_dispatch_trace(pc); break; \
  case ISA_OPCODE_NOP: pc += control_dispatch_nop(pc); break;

#endif // CERT_FW

#endif
