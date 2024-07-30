/*
 * Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef MATRIX_FORMATTING_H
#define MATRIX_FORMATTING_H

#include <assert.h>

#include "dtype_utils.h"

namespace ryzenai {

// AIE array parameters
static int const NUM_ROWS = 4;
static int const NUM_COLS = 4;
// Core buffer allocations
static int const ALIGN = 64;
static int const M_SUBV = 8;
static int const K_SUBV = 128;
static int const N_SUBV = 128;

/*get the size of wgt buffer*/
template <int Ksubv = 128, int Nsubv = 128, int Grpsize = 128>
constexpr int get_in2_size() {
  return (((Ksubv * Nsubv / 2 + Ksubv * Nsubv / Grpsize / 2 +
            Ksubv * Nsubv / Grpsize * 2) +
           ALIGN - 1) /
          ALIGN) *
         ALIGN;
}

/* subvol used to transfer bias to AIE*/
template <int Nsubv = 128> struct BiasSubv {
  uint16_t coefs[Nsubv];
};

/* subvol used to transfer quantized weights, zeros, scales to AIE*/
template <int Ksubv = 128, int Nsubv = 128, int Grpsize = 128> struct WgtSubv {
  uint8_t zeros[Ksubv * Nsubv / (2 * Grpsize)];
  uint16_t scales[Ksubv * Nsubv / Grpsize];
  uint8_t quants[Ksubv * Nsubv / 2];
};

/*Union of params/wgt subv*/
template <int Ksubv = 128, int Nsubv = 128, int Grpsize = 128> union CoreSubv {
  BiasSubv<Nsubv> bias;
  WgtSubv<Ksubv, Nsubv, Grpsize> wgts;
  uint8_t padding[get_in2_size<Ksubv, Nsubv, Grpsize>()];
};

/* mladf subvol used to transfer quantized weights, zeros, scales to AIE*/
template <int Ksubv = 128, int Nsubv = 128, int Grpsize = 128>
struct mladfWgtSubv {
  uint8_t quants[Ksubv * Nsubv / 2];
  uint8_t zeros[((Ksubv * Nsubv / (2 * Grpsize)) + 63) / 64 * 64];
  uint16_t scales[Ksubv * Nsubv / Grpsize];
};

/*Union of params/wgt mladf subv*/
template <int Ksubv = 128, int Nsubv = 128, int Grpsize = 128>
union mladfCoreSubv {
  BiasSubv<Nsubv> bias;
  mladfWgtSubv<Ksubv, Nsubv, Grpsize> wgts;
  uint8_t padding[get_in2_size<Ksubv, Nsubv, Grpsize>()];
};

/*subvol to send layer params*/
struct ParamSubv {
  int32_t outer_loop;
  int32_t inner_loop;
  int32_t kernel_rows;
  int32_t group_size;
  int32_t sign;

  void gen_params(int M, int K, int N, int group_size, int sign) {
    int Ksubv;
    if (group_size < 128)
      Ksubv = 32;
    else
      Ksubv = 128;
    int outer_m_loop = (M > M_SUBV) ? M / M_SUBV : 1;
    this->outer_loop = outer_m_loop * N / (NUM_ROWS * NUM_COLS * N_SUBV);
    this->inner_loop = K / Ksubv;
    this->kernel_rows = (M > M_SUBV) ? M_SUBV : M;
    this->group_size = group_size;
    this->sign = sign;
  };
};

/*
 * get the index in the formatted linear buffer
 * @param row is the row index in original tensor
 * @param col is the column index in original tensor
 * @param num_rows is number of rows in a subvolume
 * @param zz is the number of rows in zigzag blocks
 */
static inline int zig_zag_index(int row, int col, int num_rows, int zz) {
  return (row * zz) + (col % zz) + ((col / zz) * (num_rows * zz));
}

/*
 * compute the row major index
 * @param row is the row index in original tensor
 * @param col is the column index in original tensor
 * @num_cols is the total no. of columns in each row
 */
static inline int row_major_index(int row, int col, int num_cols) {
  return (row * num_cols) + col;
}

/*
 * compute the WH_w8 index
 * @param row is the row index in original tensor
 * @param col is the column index in original tensor
 * @num_rows is the total no. of Ksubv in each row
 */
static inline int wh_w8_index(int row, int col, int num_rows) {
  uint32_t a0 = col / 4;
  uint32_t a1 = col % 4;
  uint32_t idx = a1 + row * 4 + a0 * num_rows * 4;
  return idx;
}

/*
 * This struct has functions to compute the index of weights, scales and zeros
 * in the formatted 2D tensor from given indices in the original 2D tensor
 */
template <int Msubv = 8, int Ksubv = 128, int Nsubv = 128, int Grpsize = 128>
struct QuantMatrix {
  static const int BIAS_START_ROW = 0;
  static const int WGT_START_ROW = 1;
  // Subvolume have a zig-zag of NUM_ROWS so that each core in a column
  // get a unicast input. This is effectively "W4 indexing" but at a
  // subvolume level.
  static const int SUBV_ZZ = NUM_ROWS;
  // static_assert(sizeof(CoreSubv) == get_in2_size<Ksubv, Nsubv, Grpsize>(),
  //             "Invalid core subvolume format!");
  CoreSubv<Ksubv, Nsubv, Grpsize> *data;
  int data_size;
  int num_rows;
  int num_cols;
  int subv_rows;
  int subv_cols;

  QuantMatrix(int num_rows, int num_cols) {
    this->num_rows = num_rows;
    this->num_cols = num_cols;
    subv_rows = (num_rows / Ksubv) + WGT_START_ROW;
    subv_cols = (num_cols / Nsubv);
    data = nullptr;
    data_size = subv_rows * subv_cols * sizeof(CoreSubv<Ksubv, Nsubv, Grpsize>);
    assert(subv_cols % SUBV_ZZ == 0);
  }

  /*
   * compute the index for the bias in the formatted tensor based on the col
   * index
   * @param col is the column index of the bias in the original tensor
   */
  uint16_t &bias(int col) {
    assert(col < num_cols);
    int i0 = subv_index(BIAS_START_ROW, (col / Nsubv));
    int i1 = (col % Nsubv);
    return data[i0].bias.coefs[i1];
  }

  /*
   * compute the index of weights in formatted tensor
   * @param row is row index in original tensor
   * @param col is column index in original tensor
   */
  uint8_t &quant(int row, int col) {
    assert(row < num_rows);
    assert(col < num_cols);
    int const d = 2;
    int i0 = subv_index(WGT_START_ROW + (row / Ksubv), (col / Nsubv));
    int i1 = row_major_index((row % Ksubv), (col % Nsubv) / d, (Nsubv / d));
    return data[i0].wgts.quants[i1];
  }

  /*
   * compute the index of zeros in formatted tensor
   * @param row is row index in original tensor
   * @param col is column index in original tensor
   */
  uint8_t &zero(int row, int col) {
    assert(row < num_rows);
    assert(col < num_cols);
    int const rd = Grpsize;
    int const cd = 2;
    int i0 = subv_index(WGT_START_ROW + (row / Ksubv), (col / Nsubv));
    int i1 =
        row_major_index((row % Ksubv) / rd, (col % Nsubv) / cd, (Nsubv / cd));
    return data[i0].wgts.zeros[i1];
  }

  /*
   * compute the index of scale in formatted tensor
   * @param row is row index in original tensor
   * @param col is column index in original tensor
   */
  uint16_t &scale(int row, int col) {
    assert(row < num_rows);
    assert(col < num_cols);
    int const rd = Grpsize;
    int i0 = subv_index(WGT_START_ROW + (row / Ksubv), (col / Nsubv));
    int i1 = row_major_index((row % Ksubv) / rd, (col % Nsubv), Nsubv);
    return data[i0].wgts.scales[i1];
  }

  int subv_index(int row, int col) {
    assert(row < subv_rows);
    assert(col < subv_cols);
    return zig_zag_index(row, col, subv_rows, SUBV_ZZ);
  }

  /*
   * Dequantize weight to float for golden calculation
   * This is used only in the DI validation
   * @param row is the index of weight in original tensor
   * @param col is the index of column in the original tensor
   * @return the bfloat16 value of weight in float variable
   */
  float weight(int row, int col) {
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

/*
 * This struct works for mladf kernels, it has functions to compute the index of
 * weights, scales and zeros in the formatted 2D tensor from given indices in
 * the original 2D tensor
 */
template <int Msubv = 16, int Ksubv = 128, int Nsubv = 64, int Grpsize = 128>
struct mladfQuantMatrix {
  static const int BIAS_START_ROW = 0;
  static const int WGT_START_ROW = 1;
  // Subvolume have a zig-zag of NUM_ROWS so that each core in a column
  // get a unicast input. This is effectively "W4 indexing" but at a
  // subvolume level.
  static const int SUBV_ZZ = NUM_ROWS;
  mladfCoreSubv<Ksubv, Nsubv, Grpsize> *data;
  int data_size;
  int num_rows;
  int num_cols;
  int subv_rows;
  int subv_cols;
  int Blksize;

  mladfQuantMatrix(int num_rows, int num_cols, int blk_size) {
    this->num_rows = num_rows;
    this->num_cols = num_cols;
    this->Blksize = blk_size;
    subv_rows = (num_rows / Ksubv) + WGT_START_ROW;
    subv_cols = (num_cols / Nsubv);
    data = nullptr;
    data_size =
        subv_rows * subv_cols * sizeof(mladfCoreSubv<Ksubv, Nsubv, Grpsize>);
    assert(subv_cols % SUBV_ZZ == 0);
  }

  /*
   * compute the index for the bias in the formatted tensor based on the col
   * index
   * @param col is the column index of the bias in the original tensor
   */
  uint16_t &bias(int col) {
    assert(col < num_cols);
    int x0 = col / Nsubv;
    int y0 = 0;
    int x1 = x0 / Blksize;
    int x2 = x0 % Blksize;
    int i0 = x2 + y0 * Blksize + x1 * subv_rows * Blksize;
    int i1 = (col % Nsubv);
    return data[i0].bias.coefs[i1];
  }

  /*
   * compute the index of weights in formatted tensor
   * @param row is row index in original tensor
   * @param col is column index in original tensor
   */
  uint8_t &quant(int row, int col) {
    assert(row < num_rows);
    assert(col < num_cols);
    int const d = 2;
    int x0 = col / Nsubv;
    int y0 = row / Ksubv + 1;
    int x1 = x0 / Blksize;
    int x2 = x0 % Blksize;
    int i0 = x2 + y0 * Blksize + x1 * subv_rows * Blksize;
    int i1 = wh_w8_index((row % Ksubv), (col % Nsubv) / d, Ksubv);
    return data[i0].wgts.quants[i1];
  }

  /*
   * compute the index of zeros in formatted tensor
   * @param row is row index in original tensor
   * @param col is column index in original tensor
   */
  uint8_t &zero(int row, int col) {
    assert(row < num_rows);
    assert(col < num_cols);
    int const rd = Grpsize;
    int const cd = 2;
    int x0 = col / Nsubv;
    int y0 = row / Ksubv + 1;
    int x1 = x0 / Blksize;
    int x2 = x0 % Blksize;
    int i0 = x2 + y0 * Blksize + x1 * subv_rows * Blksize;
    int i1 =
        row_major_index((row % Ksubv) / rd, (col % Nsubv) / cd, (Nsubv / cd));
    return data[i0].wgts.zeros[i1];
  }

  /*
   * compute the index of scale in formatted tensor
   * @param row is row index in original tensor
   * @param col is column index in original tensor
   */
  uint16_t &scale(int row, int col) {
    assert(row < num_rows);
    assert(col < num_cols);
    int const rd = Grpsize;
    int x0 = col / Nsubv;
    int y0 = row / Ksubv + 1;
    int x1 = x0 / Blksize;
    int x2 = x0 % Blksize;
    int i0 = x2 + y0 * Blksize + x1 * subv_rows * Blksize;
    int i1 = row_major_index((row % Ksubv) / rd, (col % Nsubv), Nsubv);
    return data[i0].wgts.scales[i1];
  }

  int subv_index(int row, int col) {
    assert(row < subv_rows);
    assert(col < subv_cols);
    return zig_zag_index(row, col, subv_rows, SUBV_ZZ);
  }

  /*
   * Dequantize weight to float for golden calculation
   * This is used only in the DI validation
   * @param row is the index of weight in original tensor
   * @param col is the index of column in the original tensor
   * @return the bfloat16 value of weight in float variable
   */
  float weight(int row, int col) {
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

/*
 * This struct has functions to compute the indices of formatted activation
 * tensor from the given indices of the original tensor
 */
template <typename T> struct ActMatrix {
  T *data;
  int data_size;
  int num_rows;
  int num_cols;
  int tile_rows;
  int tile_cols;

  ActMatrix(int num_rows, int num_cols, int tile_rows, int tile_cols) {
    this->num_rows = num_rows;
    this->num_cols = num_cols;
    this->tile_rows = tile_rows;
    this->tile_cols = tile_cols;
    data = nullptr;
    data_size = num_rows * num_cols * sizeof(T);
  }

  /*
   * Compute the index of activation in the formatted tensor
   * @param row is the row index of the activation in original tensor
   * @param col is the column index of the activation in original tensor
   */
  T &act(int row, int col) {
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
} // namespace ryzenai
#endif // MATRIX_FORMATTING_H
