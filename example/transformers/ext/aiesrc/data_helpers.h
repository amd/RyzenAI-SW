#ifndef DATA_HELPERS
#define DATA_HELPERS

#include <stdlib.h>

//
// Helper functions to pack/unpack bfloat16 and int4
// data types, which aren't natively supported by the CPU
//

#include <stdint.h>
#include <assert.h>

uint16_t float_to_bfloat16(float x)
{
    uint32_t i;
    uint8_t* src = (uint8_t*) &x;
    uint8_t* tmp = (uint8_t*) &i;
    // copy float to uint32_t
    tmp[0] = src[0];
    tmp[1] = src[1];
    tmp[2] = src[2];
    tmp[3] = src[3];
    // round to nearest even
    uint32_t lsb = (i >> 16) & 0x1;
    uint32_t bias = 0x7fff + lsb;
    i += bias;
    // extract upper half of input
    uint16_t y = uint16_t(i >> 16);
    return y;
}

float bfloat16_to_float(uint16_t x)
{
    float y = 0.0;
    uint8_t* src = (uint8_t*) &x;
    uint8_t* dst = (uint8_t*) &y;
    dst[2] = src[0];
    dst[3] = src[1];
    return y;
}

float bfloat16_rnd_even(float x)
{
    // Simulate rounding bfloat16 to nearest even
    return bfloat16_to_float(float_to_bfloat16(x));
}

uint16_t rand_bfloat16()
{
    float x = 2.0 * (rand() / (float) RAND_MAX) - 1.0;
    return float_to_bfloat16(x);
}

uint8_t pack_v2int4(int x, int y)
{
    assert(-8 <= x && x <= 7);
    assert(-8 <= y && y <= 7);
    return (x & 0xF) | ((y & 0xF) << 4);
}

struct v2int {
    int x;
    int y;
};

v2int unpack_v2int4(uint8_t a)
{
    v2int v;
    // Extract nibbles
    v.x = (a & 0x0F);
    v.y = (a & 0xF0) >> 4;
    // Convert to signed two's complement
    v.x = (v.x % 8) - ((v.x / 8) * 8);
    v.y = (v.y % 8) - ((v.y / 8) * 8);
    return v;
}

int rand_int4()
{
    return (rand() % 16) - 8;
}

#endif // DATA_HELPERS
