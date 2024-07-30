#ifndef KERNELS_H
#define KERNELS_H

#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#include "config.h"
#include "subv_formatting.h"


void gemm_wrapper(
    adf::input_async_buffer<int8_t>& buf_in1,
    adf::input_async_buffer<int8_t>& buf_in2,
    adf::output_async_buffer<int8_t>& buf_out);

#endif // KERNELS_H
