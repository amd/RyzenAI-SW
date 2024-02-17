#ifndef DTYPE_UTILS
#define DTYPE_UTILS

#include <stdlib.h>

//
// Helper functions to pack/unpack bfloat16 and int4
// data types, which aren't natively supported by the CPU
//

#include <assert.h>
#include <stdint.h>

namespace ryzenai {

/*
* converts float to bfloat16 by rounding the LSB to nearest even
@param x is a floating point value
@return bfloat16 value in uint16_t variable
*/
uint16_t float_to_bfloat16(float x) {
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

/*
 * converts bfloat16 value to float value by zeropadding the last 16 bits
 * @param x is a bfloat16 value in uint16_t var
 * @return float output
 */
float bfloat16_to_float(uint16_t x) {
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
float bfloat16_rnd_even(float x) {
  return bfloat16_to_float(float_to_bfloat16(x));
}

/*
 * generate a random number of bfloat16 dtype
 * @param non
 * @return bfloat16 data in uint16_t dtype
 */
uint16_t rand_bfloat16(float range = 1.0) {
  float x = range * (2.0 * (rand() / (float)RAND_MAX) - 1.0);
  return float_to_bfloat16(x);
}

/*
 * pack two int4 values (with 0s in MSB) into an int8 variable
 * @param x is the first int4 value in int8 dtype
 * @param y is the second int4 value in int8 dtype
 * @return packed int8 value
 */
uint8_t pack_v2int4(int x, int y) {
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
uint8_t pack_v2uint4(int x, int y) {
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
v2int unpack_v2int4(uint8_t a) {
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
v2int unpack_v2uint4(uint8_t a) {
  v2int v;
  // Extract nibbles
  v.x = (a & 0x0F);
  v.y = (a & 0xF0) >> 4;
  return v;
}

/*
 * random number generator for int4 dtype
 */
int rand_int4(int data_range = 8) {
  return (rand() % (2 * data_range)) - data_range;
}

/*
 * random number generator for uint4 dtype
 */
int rand_uint4(int data_range = 16) { return (rand() % data_range); }

} // namespace ryzenai
#endif // DTYPE_UTILS
