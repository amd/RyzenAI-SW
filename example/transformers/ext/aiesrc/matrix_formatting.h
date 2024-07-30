#ifndef MATRIX_FORMATTING_H
#define MATRIX_FORMATTING_H

#include <assert.h>

#include "config.h"
#include "subv_formatting.h"
#include "data_helpers.h"

static inline int zig_zag_index(int row, int col, int num_rows, int zz)
{
    return (row * zz) + (col % zz) + ((col / zz) * (num_rows * zz));
}

static inline int row_major_index(int row, int col, int num_cols)
{
    return (row * num_cols) + col;
}

struct QuantMatrix
{
    static const int BIAS_START_ROW = 0;
    static const int WGT_START_ROW = 1;
    // Subvolume have a zig-zag of NUM_ROWS so that each core in a column
    // get a unicast input. This is effectively "W4 indexing" but at a
    // subvolume level.
    static const int SUBV_ZZ = NUM_ROWS;

    CoreSubv* data;
    int data_size;
    int num_rows;
    int num_cols;
    int subv_rows;
    int subv_cols;

    QuantMatrix(int num_rows, int num_cols)
    {
        this->num_rows = num_rows;
        this->num_cols = num_cols;
        subv_rows = (num_rows / K_SUBV) + WGT_START_ROW;
        subv_cols = (num_cols / N_SUBV);
        data = nullptr;
        data_size = subv_rows * subv_cols * sizeof(CoreSubv);

        assert(subv_cols % SUBV_ZZ == 0);
    }

    uint16_t& bias(int col)
    {
        assert(col < num_cols);
        int i0 = subv_index(BIAS_START_ROW, (col / N_SUBV));
        int i1 = (col % N_SUBV);
        return data[i0].bias.coefs[i1];
    }

    uint8_t& quant(int row, int col)
    {
        assert(row < num_rows);
        assert(col < num_cols);
        int const d = 2;
        int i0 = subv_index(WGT_START_ROW + (row / K_SUBV), (col / N_SUBV));
        int i1 = row_major_index((row % K_SUBV), (col % N_SUBV) / d, (N_SUBV / d));
        return data[i0].wgts.quants[i1];
    }

    uint8_t& zero(int row, int col)
    {
        assert(row < num_rows);
        assert(col < num_cols);
        int const rd = GRP_SIZE;
        int const cd = 2;
        int i0 = subv_index(WGT_START_ROW + (row / K_SUBV), (col / N_SUBV));
        int i1 = row_major_index((row % K_SUBV) / rd, (col % N_SUBV) / cd, (N_SUBV / cd));
        return data[i0].wgts.zeros[i1];
    }

    uint16_t& scale(int row, int col)
    {
        assert(row < num_rows);
        assert(col < num_cols);
        int const rd = GRP_SIZE;
        int i0 = subv_index(WGT_START_ROW + (row / K_SUBV), (col / N_SUBV));
        int i1 = row_major_index((row % K_SUBV) / rd, (col % N_SUBV), N_SUBV);
        return data[i0].wgts.scales[i1];
    }

    int subv_index(int row, int col)
    {
        assert(row < subv_rows);
        assert(col < subv_cols);
        return zig_zag_index(row, col, subv_rows, SUBV_ZZ);
    }

    // Dequantize weight to float for golden calculation
    float weight(int row, int col)
    {
        uint8_t iq = quant(row, col);
        v2int vq = unpack_v2int4(iq);
        int q = (col % 2 == 0) ? vq.x : vq.y;

        uint8_t iz = zero(row, col);
        v2int vz = unpack_v2int4(iz);
        int z = (col % 2 == 0) ? vz.x : vz.y;

        uint16_t is = scale(row, col);
        float s = bfloat16_to_float(is);

        float w = (q - z) * s;

        return bfloat16_rnd_even(w);
    }
};

template<typename T>
struct ActMatrix
{
    T* data;
    int data_size;
    int num_rows;
    int num_cols;
    int tile_rows;
    int tile_cols;

    ActMatrix(int num_rows, int num_cols, int tile_rows, int tile_cols)
    {
        this->num_rows = num_rows;
        this->num_cols = num_cols;
        this->tile_rows = tile_rows;
        this->tile_cols = tile_cols;
        data = nullptr;
        data_size = num_rows * num_cols * sizeof(T);
    }

    T& act(int row, int col)
    {
        assert(row < num_rows);
        assert(col < num_cols);
        int tile_size = tile_rows * tile_cols;
        int block_cols = num_cols / tile_cols;
        int i1 = row_major_index(row % tile_rows, col % tile_cols, tile_cols);
        int i2 = row_major_index(row / tile_rows, col / tile_cols, block_cols);
        int idx = i1 + (i2 * tile_size);
        assert(idx < num_rows * num_cols);
        return data[idx];
    }
};

#endif // MATRIX_FORMATTING_H
