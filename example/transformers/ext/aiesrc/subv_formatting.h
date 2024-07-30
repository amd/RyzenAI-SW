#ifndef SUBV_FORMATTING_H
#define SUBV_FORMATTING_H

#include "config.h"

struct BiasSubv
{
    uint16_t coefs[N_SUBV];
};

struct WgtSubv
{
    uint8_t zeros[K_SUBV * N_SUBV / (2 * GRP_SIZE)];
    uint16_t scales[K_SUBV * N_SUBV / GRP_SIZE];
    uint8_t quants[K_SUBV * N_SUBV / 2];
};

union CoreSubv
{
    BiasSubv bias;
    WgtSubv wgts;
    uint8_t padding[DATA_IN2_SIZE];
};

static_assert(sizeof(CoreSubv) == DATA_IN2_SIZE, "Invalid core subvolume format!");

struct ParamSubv
{
    int32_t outer_loop;
    int32_t inner_loop;
    int32_t kernel_rows;
    int32_t group_size;
    int32_t sign;
};

#endif // SUBV_FORMATTING_H
