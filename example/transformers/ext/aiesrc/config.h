#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>

// Actual GeMM subvolume size computed
static int const M_SUBV = 8;
static int const K_SUBV = 32;
static int const N_SUBV = 128;
static int const GRP_SIZE = 32;

// Max GeMM subvolume size the core will support
static int const MAX_M_SUBV = 8;
static int const MAX_K_SUBV = 128;
static int const MAX_N_SUBV = 128;
static int const MAX_GRP_SIZE = 128;

static_assert(MAX_K_SUBV == MAX_GRP_SIZE, "Invalid subvolume and group size combination!");
static_assert(K_SUBV == GRP_SIZE, "Invalid subvolume and group size combination!");
static_assert(MAX_K_SUBV >= K_SUBV, "computed K subvol dim is larger than the supported subvol dim!");
static_assert(MAX_GRP_SIZE >= GRP_SIZE, "computed grp_size is larger than the supported grp_size!");

// AIE array parameters
static int const NUM_ROWS = 4;
static int const NUM_COLS = 4;
static int const GMIO_BURST_LENGTH = 256;
static int const GMIO_BANDWIDTH = 8;

// Max supported subvol dimension Core buffer allocations
static int const ALIGN = 64;
static int const CORE_IN1_SIZE = MAX_M_SUBV * MAX_K_SUBV * 2; // bfloat16 input
static int const CORE_IN2_SIZE = (((MAX_K_SUBV * MAX_N_SUBV / 2
                                    + MAX_K_SUBV * MAX_N_SUBV / MAX_GRP_SIZE / 2
                                    + MAX_K_SUBV * MAX_N_SUBV / MAX_GRP_SIZE * 2) + ALIGN - 1) / ALIGN) * ALIGN;
static int const CORE_OUT_SIZE = MAX_M_SUBV * MAX_N_SUBV * 4; // fp32 output
static int const CORE_WGT_SIZE = MAX_K_SUBV * MAX_N_SUBV * 2; // bfloat16 weights
static int const CORE_ACC_SIZE = MAX_M_SUBV * MAX_N_SUBV * 4; // fp32 accumulation

// Actual core subvol dimension
static int const DATA_IN1_SIZE = M_SUBV * K_SUBV * 2; // bfloat16 input
static int const DATA_IN2_SIZE = (((K_SUBV * N_SUBV / 2
                                + K_SUBV * N_SUBV / GRP_SIZE / 2
                                + K_SUBV * N_SUBV / GRP_SIZE * 2) + ALIGN - 1) / ALIGN) * ALIGN;
static int const DATA_OUT_SIZE = M_SUBV * N_SUBV * 4; // fp32 output
static int const DATA_WGT_SIZE = K_SUBV * N_SUBV * 2; // bfloat16 weights
static int const DATA_ACC_SIZE = M_SUBV * N_SUBV * 4; // fp32 accumulation

static int const CORE_STACK_SIZE = 4096;
static int const CORE_BANK_SIZE = 16384;
// Bank #1
static_assert(CORE_WGT_SIZE/2 <= CORE_BANK_SIZE, "Bank #1 allocation failed!");
static int const CORE_WGT1_ADDR     = (0 * CORE_BANK_SIZE);

// Bank #2
static_assert(CORE_WGT_SIZE/2 <= CORE_BANK_SIZE, "Bank #2 allocation failed!");
static int const CORE_WGT2_ADDR     = (1 * CORE_BANK_SIZE);

// Bank #3
static_assert(CORE_IN1_SIZE + CORE_IN2_SIZE + CORE_ACC_SIZE <= CORE_BANK_SIZE, "Bank #3 allocation failed!");
static int const CORE_IN1_PING_ADDR = (2 * CORE_BANK_SIZE);
static int const CORE_IN2_PONG_ADDR = (2 * CORE_BANK_SIZE) + CORE_IN1_SIZE;
static int const CORE_OUT_PING_ADDR = (2 * CORE_BANK_SIZE + CORE_IN1_SIZE + CORE_IN2_SIZE);


// Bank #4
static_assert(CORE_IN1_SIZE + CORE_IN2_SIZE + CORE_STACK_SIZE <= CORE_BANK_SIZE, "Bank #3 allocation failed!");
static int const CORE_IN1_PONG_ADDR = (3 * CORE_BANK_SIZE);
static int const CORE_IN2_PING_ADDR = (3 * CORE_BANK_SIZE) + CORE_IN1_SIZE;
static int const CORE_STACK_ADDR    = (3 * CORE_BANK_SIZE) + CORE_IN1_SIZE + CORE_IN2_SIZE;
//static int const CORE_OUT_PING_ADDR = (3 * CORE_BANK_SIZE);

#endif // CONFIG_H
