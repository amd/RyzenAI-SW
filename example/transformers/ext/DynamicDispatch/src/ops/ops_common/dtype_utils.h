/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef DTYPE_UTILS
#define DTYPE_UTILS

//
// Helper functions to pack/unpack bfloat16 and int4
// data types, which aren't natively supported by the CPU
//

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <stdint.h>

#include <algorithm>
#include <array>

namespace ryzenai {

/*
* converts float to bfloat16 by rounding the LSB to nearest even
@param x is a floating point value
@return bfloat16 value in uint16_t variable
*/
static inline uint16_t float_to_bfloat16(float x) {
  uint32_t i;
  uint8_t *src = (uint8_t *)&x;
  uint8_t *tmp = (uint8_t *)&i;
  // copy float to uint32_t
  std::memcpy(tmp, src, sizeof(float));
  // round to nearest even
  uint32_t lsb = (i >> 16) & 0x1;
  uint32_t bias = 0x7fff + lsb;
  i += bias;
  // extract upper half of input
  uint16_t y = uint16_t(i >> 16);
  return y;
}

#ifdef __GNUC__
#include <cpuid.h>
#else
#include <intrin.h>
#endif

static __inline void cpuidex_wrapper(int __cpuid_info[4], int __leaf,
                                     int __subleaf) {
#ifdef __GNUC__
  __cpuid_count(__leaf, __subleaf, __cpuid_info[0], __cpuid_info[1],
                __cpuid_info[2], __cpuid_info[3]);
#else
  __cpuidex(__cpuid_info, __leaf, __subleaf);
#endif
}

static bool check_avx512_and_bf16_support() {

  int cpuid[4];

  cpuidex_wrapper(cpuid, 0, 0);
  if (cpuid[0] >= 7) {
    cpuidex_wrapper(cpuid, 7, 0);
    bool avx512vl_supported =
        (cpuid[1] & ((int)1 << 31)) != 0; // AVX-512-VL: EBX[31]
    cpuidex_wrapper(cpuid, 7, 1);
    bool avx512bf16_supported =
        (cpuid[0] & (1 << 5)) != 0; // AVX-512 BF16: EAX[5]
    return avx512vl_supported && avx512bf16_supported;
  }

  return false;
}

#include <immintrin.h>

static inline void float_buffer_to_bfloat16(const float *in,
                                            std::size_t num_elements,
                                            std::uint16_t *out,
                                            const bool use_avx) {
  if (!use_avx) {
    std::transform(in, in + num_elements, out, float_to_bfloat16);
  } else {
// This code is not compiling on the linux build machine in DD
#ifdef _WIN32
    static_assert(sizeof(float) == 4);

    constexpr std::size_t VECTOR_SIZE_BYTES = (512 / 8);
    constexpr std::size_t FLOAT_SIZE_BYTES = sizeof(float);

    static_assert(VECTOR_SIZE_BYTES % FLOAT_SIZE_BYTES == 0);
    constexpr std::size_t FLOATS_PER_VECTOR =
        VECTOR_SIZE_BYTES / FLOAT_SIZE_BYTES;

    const std::size_t num_iter = num_elements / FLOATS_PER_VECTOR;
    const std::size_t remainder = num_elements - num_iter * FLOATS_PER_VECTOR;

    for (std::size_t i = 0; i < num_iter; ++i) {
      __m512 float_vec = _mm512_loadu_ps(in);
      __m256bh bf16_vec = _mm512_cvtneps2bf16(float_vec);
      _mm256_storeu_epi16(out, bf16_vec);

      in += FLOATS_PER_VECTOR;
      out += FLOATS_PER_VECTOR;
    }

    for (std::size_t i = 0; i < remainder; ++i) {
      *out++ = float_to_bfloat16(*in++);
    }
#else
    throw;
#endif
  }
}

/*
 * converts bfloat16 value to float value by zeropadding the last 16 bits
 * @param x is a bfloat16 value in uint16_t var
 * @return float output
 */
static float bfloat16_to_float(uint16_t x) {
  float y = 0.0;
  uint8_t *src = (uint8_t *)&x;
  uint8_t *dst = (uint8_t *)&y;
  std::memcpy(&dst[2], src, sizeof(uint16_t));
  return y;
}

/*
 * Simulate rounding bfloat16 to nearest even
 * @param x is a bfloat16 input in float variable
 * @return bfloat16 value with nearest even rounding in float type
 */
static float bfloat16_rnd_even(float x) {
  return bfloat16_to_float(float_to_bfloat16(x));
}

/*
 * generate a random number of bfloat16 dtype
 * @param non
 * @return bfloat16 data in uint16_t dtype
 */
static uint16_t rand_bfloat16(float range = 1.0) {
  float x = range * (2.0 * (rand() / (float)RAND_MAX) - 1.0);
  return float_to_bfloat16(x);
}

/*
 * pack two int4 values (with 0s in MSB) into an int8 variable
 * @param x is the first int4 value in int8 dtype
 * @param y is the second int4 value in int8 dtype
 * @return packed int8 value
 */
static uint8_t pack_v2int4(int x, int y) {
  assert(-8 <= x && x <= 7);
  assert(-8 <= y && y <= 7);
  return (x & 0xF) | ((y & 0xF) << 4);
}

/*
 * pack two uint4 values (with 0s in MSB) into an uint8 variable
 * @param x is the first uint4 value in int8 dtype
 * @param y is the second uint4 value in int8 dtype
 * @return packed uint8 value
 */
static uint8_t pack_v2uint4(int x, int y) {
  assert(0 <= x && x <= 15);
  assert(0 <= y && y <= 15);
  return (x & 0xF) | ((y & 0xF) << 4);
}

struct v2int {
  int x;
  int y;
};

/*
 * unpack an int8 variable into 2 int4 variables
 * @param a is uint8_t variable
 * @return v2int object with 2 int4 elements
 */
static v2int unpack_v2int4(uint8_t a) {
  v2int v;
  // Extract nibbles
  v.x = (a & 0x0F);
  v.y = (a & 0xF0) >> 4;
  // Convert to signed two's complement
  v.x = (v.x % 8) - ((v.x / 8) * 8);
  v.y = (v.y % 8) - ((v.y / 8) * 8);
  return v;
}

/*
 * unpack an int8 variable into 2 uint4 variables
 * @param a is uint8_t variable
 * @return v2int object with 2 uint4 elements
 */
static v2int unpack_v2uint4(uint8_t a) {
  v2int v;
  // Extract nibbles
  v.x = (a & 0x0F);
  v.y = (a & 0xF0) >> 4;
  return v;
}

/*
 * random number generator for int4 dtype
 */
static int rand_int4(int data_range = 8) {
  return (rand() % (2 * data_range)) - data_range;
}

/*
 * random number generator for uint4 dtype
 */
static int rand_uint4(int data_range = 16) { return (rand() % data_range); }

} // namespace ryzenai
#endif // DTYPE_UTILS
