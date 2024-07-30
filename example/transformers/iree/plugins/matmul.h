#ifndef _IREE_PLUGINS_MATMUL_H_
#define _IREE_PLUGINS_MATMUL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>

// FP32 matrix multiplication called by system_plugin.c.
// - *output = matmul(input_a, input_b)
// Prerequisite: It is the caller's responsibility to reserve input/output
// buffers of the following dimension.
// - dim(input_a) = (row_dim, inner_dim)
// - dim(input_b) = (inner_dim, col_dim)
// - dim(output) = (row_dim, col_dim)
// NOTE This function is thread-safe.
void matmul_f32_impl(const float *input_a, const float *input_b, float *output,
                     size_t row_dim, size_t inner_dim, size_t col_dim);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _IREE_PLUGINS_MATMUL_H_
