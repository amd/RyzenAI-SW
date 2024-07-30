#include "config.h"

// See below link for an explanation on how this works
// https://docs.xilinx.com/r/en-US/ug1079-ai-engine-kernel-coding/Shared-Graph-Scoped-Tables
int8_t buf_wgt1[CORE_WGT_SIZE / 2] = {0};
int8_t buf_wgt2[CORE_WGT_SIZE / 2] = {0};
//int8_t buf_acc[CORE_ACC_SIZE] = {0};
