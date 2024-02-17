/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef SUPER_INSTR_H
#define SUPER_INSTR_H

#include "ml_params.h"
#include <stdint.h>
namespace ryzenai {
// NOTE: DO NOT RE-ORDER THE FIELDS OF THESE STRUCTS.
//       These fields are ordered to match the instruction
//       encoding defined by the super kernel and wrapper.

struct GemmData {
  MLKernelParams params;
};

struct GemmInstr {
  int16_t instr_size;
  int16_t instr_repeat;
  uint32_t core_mask;
  uint32_t dyn_out_size;
  uint32_t subinstr_size;
  uint32_t opcode_config;
  GemmData data;
};

struct GemmPadding {
  int32_t __unused[2];
};

struct GemmSeq {
  // header
  int32_t total_size;
  int32_t __unused;
  int16_t repeat_count;
  int16_t repeat_size;

  // instructions
  GemmInstr instr[3];

  // padding for multiple of four word alignment
  GemmPadding padding;
};

// The total size of the super kernel instruction sequence is the number
// of four byte words minus 2, since the first four bytes
// encode the total size, and the second four bytes are discarded.
// This is very confusing, since the bytes encoding
// the total size are not included in the total size, but the
// bytes encoding size are included for all other sizes in the instruction.
static const int SEQ_BYTES = sizeof(GemmSeq);
static const int SEQ_WORDS = SEQ_BYTES / 4;
static const int TOTAL_SIZE = SEQ_WORDS - 2;

// The instruction size includes 5 words (size/repeat, core mask, dynamic output
// size, subinstruction size, opcode/config) plus the varaible length data
// field. The subinstruction only has the subinstr size and opcode/config
// followed by the variable length data field, and the subinstr size is encoded
// in bytes instead of words.
static const int INSTR_SIZE = sizeof(GemmInstr) / 4;
static const int SUBINSTR_SIZE = 4 * (INSTR_SIZE - 3);
static const int PADDING_BYTES = sizeof(GemmPadding);
static const int PADDING_WORDS = sizeof(GemmPadding) / 4;

// The total size of the super kernel sequence in four byte words must
// be a multiple of four, due to the unrolling implemented in the superkernel.
// Otherwise, some bytes of the instruction will be missing.
static_assert(SEQ_BYTES % 4 == 0);
static_assert(TOTAL_SIZE % 4 == 0);

// Gemm instruction must be an integer number of
// four byte words due to packet serialization.
static_assert(sizeof(GemmInstr) % 4 == 0);

// Padding at the end of the sequence must be an integer
// number of four byte words.
static_assert(sizeof(GemmPadding) % 4 == 0);

// Since the instruction sequence will be copied by the superkernel
// into an array of bytes, it must be trivially copyable,
// otherwise bizarre bugs may insue. This check is included
// here to make sure this property is maintained.
static_assert(std::is_trivially_copyable_v<GemmSeq>);

// initialize instr_ddr to encode a multiple subvolume GEMM operation
// using the super kernel instruction format
static inline void
init_gemm_instr_ddr(int8_t *instr_ddr, int M, int K, int N,
                    // Subvolume dimensions
                    int Msubv, int Ksubv = 128, int Nsubv = 64,
                    // Block granularity (default is for OLOH kernel type)
                    int Mgran = 8, int Kgran = 8, int Ngran = 16,
                    // AIE array size
                    int aie_rows = 4, int aie_cols = 4) {
  // Generate kernel parameters
  MLKernelParams params;
  params.update_params(Msubv, Ksubv, Nsubv, Mgran, Kgran, Ngran);
  params.ctrl.parts.out_64 = 1;
  // Round M up to nearest multiple of Msubv
  // NOTE: this is included to account for special cases where
  //       the input to a core is zero-padded by the BDs
  M = ((M + (Msubv - 1)) / Msubv) * Msubv;

  // Compute repeat counts as follows:
  //     OUTER_REPEAT
  //     --> (N / (aie_cols * Nsubv)) iterations are required to
  //         compute the full N dimension
  //     --> (M / Msubv) iterations of the full N dimension are required to
  //         compute the full M dimension
  //     INNER_REPEAT
  //     --> One iteration initializes the TDM buffers
  //     --> (K / (aie_rows * Ksubv)) - 2 iterations accumulate the TDM buffers
  //     --> One iteration generates the final output subvolume
  const int OUTER_REPEAT = (M / Msubv) * (N / (aie_cols * Nsubv));
  const int INNER_REPEAT = (K / (aie_rows * Ksubv)) - 2;

  GemmSeq *seq = (GemmSeq *)instr_ddr;

  // header
  seq->total_size = TOTAL_SIZE;
  seq->repeat_count = OUTER_REPEAT;
  seq->repeat_size = TOTAL_SIZE;
  seq->padding = {0};
  seq->__unused = 0;

  // config bits are ordered as follows:
  //      acq_B acq_A wait_out init

  // instruction #1: initialize TDMs
  //      config: 1 1 0 1 == 0xD
  seq->instr[0].instr_size = INSTR_SIZE;
  seq->instr[0].instr_repeat = 1;
  seq->instr[0].core_mask = 0x003c000f;
  seq->instr[0].dyn_out_size = 128;
  seq->instr[0].subinstr_size = SUBINSTR_SIZE;
  seq->instr[0].opcode_config = 0x02010d01;
  seq->instr[0].data.params = params;

  // instruction #2: accumulate TDMs
  //      config: 1 1 0 0 == 0xC
  seq->instr[1].instr_size = INSTR_SIZE;
  seq->instr[1].instr_repeat = INNER_REPEAT;
  seq->instr[1].core_mask = 0x003c000f;
  seq->instr[1].dyn_out_size = 128;
  seq->instr[1].subinstr_size = SUBINSTR_SIZE;
  seq->instr[1].opcode_config = 0x02010c01;
  seq->instr[1].data.params = params;

  // instruction #3: generate output
  //      config: 1 1 1 0 == 0xE
  seq->instr[2].instr_size = INSTR_SIZE + PADDING_WORDS;
  seq->instr[2].instr_repeat = 1;
  seq->instr[2].core_mask = 0x003c000f;
  seq->instr[2].dyn_out_size = 128;
  seq->instr[2].subinstr_size = SUBINSTR_SIZE + PADDING_BYTES;
  seq->instr[2].opcode_config = 0x02010e01;
  seq->instr[2].data.params = params;
}
} // namespace ryzenai
#endif // SUPER_INSTR_H