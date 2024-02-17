#ifndef MATRIX_FORMATTING_H
#define MATRIX_FORMATTING_H

#include <assert.h>

#include "dtype_utils.h"

namespace ryzenai {
// GeMM subvolume size
static int const M_SUBV = 32;
static int const K_SUBV = 32;
static int const N_SUBV = 128;

static int const GRP_SIZE = 32;

// AIE array parameters
static int const NUM_ROWS = 4;
static int const NUM_COLS = 4;
// Core buffer allocations
static int const ALIGN = 64;
static int const CORE_IN1_SIZE = M_SUBV * K_SUBV * 2; // bfloat16 input
static int const CORE_IN2_SIZE =
    (((K_SUBV * N_SUBV / 2 + K_SUBV * N_SUBV / GRP_SIZE / 2 +
       K_SUBV * N_SUBV / GRP_SIZE * 2) +
      ALIGN - 1) /
     ALIGN) *
    ALIGN;
static int const CORE_OUT_SIZE = M_SUBV * N_SUBV * 4; // fp32 output
static int const CORE_WGT_SIZE = K_SUBV * N_SUBV * 2; // bfloat16 weights
static int const CORE_ACC_SIZE = M_SUBV * N_SUBV * 4; // fp32 accumulation

struct BiasSubv {
  uint16_t coefs[N_SUBV];
};

struct WgtSubv {
  uint8_t quants[K_SUBV * N_SUBV / 2];
  uint8_t zeros[K_SUBV * N_SUBV / (2 * GRP_SIZE)];
  uint16_t scales[K_SUBV * N_SUBV / GRP_SIZE];
};

union CoreSubv {
  BiasSubv bias;
  WgtSubv wgts;
  uint8_t padding[CORE_IN2_SIZE];
};

static_assert(sizeof(CoreSubv) == CORE_IN2_SIZE,
              "Invalid core subvolume format!");

struct ParamSubv {
  int32_t outer_loop;
  int32_t inner_loop;
  int32_t kernel_rows;

  void gen_params(int M, int K, int N) {
    this->outer_loop = N / (NUM_ROWS * NUM_COLS * N_SUBV);
    this->inner_loop = K / K_SUBV;
    this->kernel_rows = M;
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
 * This struct has functions to compute the index of weights, scales and zeros
 * in the formatted 2D tensor from given indices in the original 2D tensor
 */
struct QuantMatrix {
  static const int BIAS_START_ROW = 0;
  static const int WGT_START_ROW = 1;
  // Subvolume have a zig-zag of NUM_ROWS so that each core in a column
  // get a unicast input. This is effectively "W4 indexing" but at a
  // subvolume level.
  static const int SUBV_ZZ = NUM_ROWS;

  CoreSubv *data;
  int data_size;
  int num_rows;
  int num_cols;
  int subv_rows;
  int subv_cols;

  QuantMatrix(int num_rows, int num_cols) {
    this->num_rows = num_rows;
    this->num_cols = num_cols;
    subv_rows = (num_rows / K_SUBV) + WGT_START_ROW;
    subv_cols = (num_cols / N_SUBV);
    data = nullptr;
    data_size = subv_rows * subv_cols * sizeof(CoreSubv);

    assert(subv_cols % SUBV_ZZ == 0);
  }

  /*
   * compute the index for the bias in the formatted tensor based on the col
   * index
   * @param col is the column index of the bias in the original tensor
   */
  uint16_t &bias(int col) {
    assert(col < num_cols);
    int i0 = subv_index(BIAS_START_ROW, (col / N_SUBV));
    int i1 = (col % N_SUBV);
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
    int i0 = subv_index(WGT_START_ROW + (row / K_SUBV), (col / N_SUBV));
    int i1 = row_major_index((row % K_SUBV), (col % N_SUBV) / d, (N_SUBV / d));
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
    int const rd = GRP_SIZE;
    int const cd = 2;
    int i0 = subv_index(WGT_START_ROW + (row / K_SUBV), (col / N_SUBV));
    int i1 = row_major_index((row % K_SUBV) / rd, (col % N_SUBV) / cd, (N_SUBV / cd));
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
    int const rd = GRP_SIZE;
    int i0 = subv_index(WGT_START_ROW + (row / K_SUBV), (col / N_SUBV));
    int i1 = row_major_index((row % K_SUBV) / rd, (col % N_SUBV), N_SUBV);
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
