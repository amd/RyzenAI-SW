#pragma once

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>

int constexpr qry_subv_rows = 32;
int constexpr qry_subv_rows_PSH = 16;
int constexpr qry_subv_cols = 96;
int constexpr key_subv_rows = 64;
int constexpr key_subv_rows_PSJ = 16; // for M = 128
int constexpr key_subv_cols = 96;
int constexpr val_subv_rows = 64;
int constexpr val_subv_cols = 64;
int constexpr out_subv_rows = 32;
int constexpr out_subv_cols = 64;

int constexpr mha_win_st_pad = 64;
int constexpr mha_win_val_subv_cols = 32;

int constexpr mha_psq2_st_pad = 80; // 77 padded to 80
int constexpr mha_psq2_val_subv_cols = 64;

int constexpr mha_psr_sq = 16;
int constexpr mha_psr_st_pad = 80; // 77 padded to 80
int constexpr mha_psr_val_subv_cols = 64;

int constexpr mha_channel_sh = 49;
int constexpr mha_channel_val_subv_cols = 32;

int constexpr num_heads = 12;
int constexpr gprb_rows = 96;
int constexpr gprb_cols = 8;

int constexpr num_qdq_nodes = 6;
int constexpr QDQparam_size = 16;

int constexpr GPRB_buf_size = 1024;

// for gprb_vec64
int constexpr gprb_c0_scalar_idx = 8;
int constexpr qk_qdq_c0_scalar_idx = 9;
int constexpr smv_qdq_c0_scalar_idx = 10;

// general qdq idx
int constexpr qdq_c0_idx = 0;
int constexpr qdq_c1_idx = 2;
int constexpr qdq_c2_idx = 3;
int constexpr qdq_c3_idx = 4;
int constexpr qdq_Mv_idx = 5;
int constexpr qdq_Nv_idx = 6;
int constexpr qdq_SQb_idx = 7;
int constexpr qdq_Sout_idx = 8;
int constexpr qdq_Stdm_idx = 9;
// for PSF and PSJ, set this be 1; for PSH, if uint8_q, set it be 0, if
// uint16_q, set it be 1
int constexpr qdq_isint16_idx = 10;

// gprb_vec32
int constexpr gprb_act_scale_idx = 10;
int constexpr gprb_act_zero_idx = 11;
int constexpr gprb_wgt_scale_idx = 12;
int constexpr gprb_wgt_zero_idx = 13;
int constexpr gprb_model_a_idx = 14;
int constexpr gprb_model_b_idx = 26;
int constexpr gprb_model_c_idx = 27;
int constexpr gprb_isint16_idx = 28;

int constexpr mha_isint16_idx = 2;

namespace mhagprb_matrix {
struct bfloat16_t {
  uint16_t value;
};

template <typename Tq0, typename Tzp, int gprb_rows, int gprb_cols,
          int num_heads>
struct GprbParams {
  uint8_t proj_mat[gprb_rows * gprb_cols];
  Tq0 qdq_bias[gprb_cols];
  int64_t c0;
  int32_t c1;
  int32_t c2;
  int32_t c3;
  int32_t M;
  int32_t N;
  int32_t shift_Qb;
  int32_t shift_Qout;
  int32_t res;
  bfloat16_t model_a[num_heads];
  bfloat16_t model_b;
  bfloat16_t model_c;
  bfloat16_t act_scale;
  bfloat16_t wgt_scale;
  Tzp act_zero_point;
  Tzp wgt_zero_point;
  int32_t isint16;
};

inline int row_major_index(int row, int col, int num_rows, int num_cols) {
  assert(row < num_rows);
  assert(col < num_cols);
  return (row * num_cols) + col;
}

inline int col_major_index(int row, int col, int num_rows, int num_cols) {
  assert(row < num_rows);
  assert(col < num_cols);
  return (col * num_rows) + row;
}

inline int w8_index(int row, int col, int num_rows, int num_cols) {
  assert(row < num_rows);
  assert(col < num_cols);
  int constexpr zz = 8;
  return (row * zz) + (col % zz) + ((col / zz) * (num_rows * zz));
}

template <typename T> struct RowMajorMatrix {
  int const num_rows;
  int const num_cols;
  T *const data;

  RowMajorMatrix(int num_rows, int num_cols, void *data)
      : num_rows(num_rows), num_cols(num_cols), data(static_cast<T *>(data)) {}

  T &at(int row, int col) {
    assert(row < num_rows);
    assert(col < num_cols);
    int idx = row_major_index(row, col, num_rows, num_cols);
    assert(idx < num_rows * num_cols);
    return data[idx];
  }

  static int size(int num_rows, int num_cols) {
    return num_rows * num_cols * sizeof(T);
  }
};

template <typename T, int subv_rows, int subv_cols> struct ScaleTensor {
  int const num_heads;
  int const num_rows;
  int const num_cols;
  T *const data;

  ScaleTensor(int num_heads, int num_rows, int num_cols, void *data)
      : num_heads(num_heads), num_rows(num_rows), num_cols(num_cols),
        data(static_cast<T *>(data)) {}

  T &at(int head, int row, int col) {
    assert(head < num_heads);
    assert(row < num_rows);
    assert(col < num_cols);
    int constexpr subv_size = subv_rows * subv_cols;
    int const head_size = num_rows * num_cols;
    int const r = row % subv_rows;
    int const c = col % subv_cols;
    int const i = w8_index(r, c, subv_rows, subv_cols);
    int const rr = row / subv_rows;
    int const cc = col / subv_cols;
    int const ii =
        col_major_index(rr, cc, (num_rows / subv_rows), (num_cols / subv_cols));
    int const idx = i + (ii * subv_size) + (head * head_size);
    assert(idx < num_heads * head_size);
    return data[idx];
  }

  static int size(int num_heads, int num_rows, int num_cols) {
    return num_heads * num_rows * num_cols * sizeof(T);
  }
};

inline uint32_t float_to_uint(float f) {
  uint32_t i = 0;
  char *ptr_f = reinterpret_cast<char *>(&f);
  char *ptr_i = reinterpret_cast<char *>(&i);
  ptr_i[0] = ptr_f[0];
  ptr_i[1] = ptr_f[1];
  ptr_i[2] = ptr_f[2];
  ptr_i[3] = ptr_f[3];
  return i;
}

inline float uint_to_float(uint32_t i) {
  float f = 0;
  char *ptr_f = reinterpret_cast<char *>(&f);
  char *ptr_i = reinterpret_cast<char *>(&i);
  ptr_f[0] = ptr_i[0];
  ptr_f[1] = ptr_i[1];
  ptr_f[2] = ptr_i[2];
  ptr_f[3] = ptr_i[3];
  return f;
}

inline bfloat16_t float_to_bfloat16(float fp) {
  uint32_t bits = float_to_uint(fp);
  uint32_t lsb = (bits >> 16) & 0x1;
  uint32_t bias = 0x7FFF + lsb;
  uint32_t rnd = bits + bias;
  return bfloat16_t{uint16_t(rnd >> 16)};
}

inline float bfloat16_to_float(bfloat16_t bf) {
  return uint_to_float(uint32_t(bf.value) << 16);
}

template <typename T> struct MhaParams {
  // static int const attn_dim = 512;
  // static int const num_cores = 8;
  // static int const mask_cols = attn_dim / num_cores;
  static int const num_heads = 12;
  static int const gprb_rows = 96;
  static int const gprb_cols = 8;
  // NOTE: We pad num_scalars to 24 for memory alignment
  static int const num_scalars = 56;
  static int const core_elems =
      (gprb_rows * gprb_cols) + gprb_cols + num_scalars;

  T *const data;

  MhaParams(void *data) : data(static_cast<T *>(data)) {}

  T &gprb_mat(int row, int col) {
    assert(row < gprb_rows);
    assert(col < gprb_cols);
    int const idx = w8_index(row, col, gprb_rows, gprb_cols);
    assert(idx < core_elems);
    return data[idx];
  }

  T &gprb_vec(int col) {
    assert(col < gprb_cols);
    int const idx = (gprb_rows * gprb_cols) + col;
    assert(idx < core_elems);
    return data[idx];
  }

  T &gprb_a(int head_idx) {
    int const idx = (gprb_rows * gprb_cols) + gprb_cols + 0;
    assert(idx < core_elems);
    return data[idx + head_idx];
  }

  T &gprb_b() {
    int const idx = (gprb_rows * gprb_cols) + gprb_cols + num_heads;
    assert(idx < core_elems);
    return data[idx];
  }

  T &gprb_c() {
    int const idx = (gprb_rows * gprb_cols) + gprb_cols + num_heads + 1;
    assert(idx < core_elems);
    return data[idx];
  }

  static int size() {
    int act_size = core_elems * sizeof(T); // int8
    if constexpr (std::is_same_v<T, int16_t>) {
      act_size = (core_elems - 56 + 8) * sizeof(T);
    }
    return act_size;
  }
};

template <typename T, int key_subv_rows, int key_subv_cols, int val_subv_rows,
          int val_subv_cols>
struct ActKVMatrix {
  int const key_rows;
  int const key_cols;
  int const val_rows;
  int const val_cols;
  T *const data;

  ActKVMatrix(int key_rows, int key_cols, int val_rows, int val_cols,
              void *data)
      : key_rows(key_rows), key_cols(key_cols), val_rows(val_rows),
        val_cols(val_cols), data(static_cast<T *>(data)) {
    assert(key_rows % key_subv_rows == 0);
    assert(key_cols % key_subv_cols == 0);
    assert(val_rows % val_subv_rows == 0);
    assert(val_cols % val_subv_cols == 0);
  }

  T &atK(int row, int col) {
    int const idx = row_major_index(row, col, key_rows, key_cols);
    assert(idx < (key_rows * key_cols) + (val_rows * val_cols));
    return data[idx];
  }

  T &atV(int row, int col) {
    int const idx =
        (key_rows * key_cols) + row_major_index(row, col, val_rows, val_cols);
    assert(idx < (key_rows * key_cols) + (val_rows * val_cols));
    return data[idx];
  }

  static int size(int key_rows, int key_cols, int val_rows, int val_cols) {
    return (key_rows * key_cols * sizeof(T)) +
           (val_rows * val_cols * sizeof(T));
  }
};

template <typename T, int subv_rows, int subv_cols> struct ActQMatrix {
  int const num_rows;
  int const num_cols;
  T *const data;

  ActQMatrix(int num_rows, int num_cols, void *data)
      : num_rows(num_rows), num_cols(num_cols), data(static_cast<T *>(data)) {
    assert(num_rows % subv_rows == 0);
    assert(num_cols % subv_cols == 0);
  }

  T &at(int row, int col) {
    int const idx = row_major_index(row, col, num_rows, num_cols);
    assert(idx < num_rows * num_cols);
    return data[idx];
  }

  T &at(int idx) { return data[idx]; }

  static int size(int num_rows, int num_cols) {
    return num_rows * num_cols * sizeof(T);
  }
};

template <typename T, int subv_rows, int subv_cols> struct OutMatrix {
  int const num_rows;
  int const num_cols;
  T *const data;

  OutMatrix(int num_rows, int num_cols, void *data)
      : num_rows(num_rows), num_cols(num_cols), data(static_cast<T *>(data)) {
    assert(num_rows % subv_rows == 0);
    assert(num_cols % subv_cols == 0);
  }

  T &at(int row, int col) {
    int const idx = row_major_index(row, col, num_rows, num_cols);
    assert(idx < num_rows * num_cols);
    return data[idx];
  }

  static int size(int num_rows, int num_cols) {
    return num_rows * num_cols * sizeof(T);
  }
};

template <typename T> void init_random(T mat, int min = -128, int max = 128) {
  for (int i = 0; i < mat.num_rows; ++i) {
    for (int j = 0; j < mat.num_cols; ++j) {
      mat.at(i, j) = (rand() % (max - min)) + min;
    }
  }
}

template <typename T>
void init_random_bfloat16(T mat, int min = -128, int max = 128) {
  for (int i = 0; i < mat.num_rows; ++i) {
    for (int j = 0; j < mat.num_cols; ++j) {
      mat.at(i, j) =
          float_to_bfloat16(float((rand() % (max - min)) + min)).value;
    }
  }
}

template <typename T>
void init_random_KV(T mat, int min = -128, int max = 128) {
  for (int i = 0; i < mat.key_rows; ++i) {
    for (int j = 0; j < mat.key_cols; ++j) {
      mat.atK(i, j) = (rand() % (max - min)) + min;
    }
  }
  for (int i = 0; i < mat.val_rows; ++i) {
    for (int j = 0; j < mat.val_cols; ++j) {
      mat.atV(i, j) = (rand() % (max - min)) + min;
    }
  }
}

template <typename T>
void init_random_scale_tensor(T mat, int min = -128, int max = 128) {
  for (int h = 0; h < mat.num_heads; ++h) {
    for (int i = 0; i < mat.num_rows; ++i) {
      for (int j = 0; j < mat.num_cols; ++j) {
        mat.at(h, i, j) = (rand() % (max - min)) + min;
      }
    }
  }
}

template <typename T>
void init_random_mha_params(MhaParams<T> prm, int min, int max) {

  for (int row = 0; row < MhaParams<T>::gprb_rows; ++row) {
    for (int col = 0; col < MhaParams<T>::gprb_cols; ++col) {
      // int16_t val = (rand() % (max - min)) + min;
      prm.gprb_mat(row, col) = (rand() % (max - min)) + min; // val;
    }
  }

  for (int col = 0; col < MhaParams<T>::gprb_cols; ++col) {
    // int16_t val = (rand() % (max - min)) + min;
    prm.gprb_vec(col) = (rand() % (max - min)) + min;
  }

  prm.gprb_a() = 8;
  prm.gprb_b() = 1;
  prm.gprb_c() = 2;
}

template <typename T = uint8_t> struct gemm_qdq_param {
  // q params
  int64_t C0;
  int32_t C1;
  int32_t C2;
  int32_t C3;
  T sqb;
  T sout;

  // dq params
  T zero_point;
  float scale;
};

template <typename T> T saturate(T val, T min, T max) {
  return std::min(std::max(val, min), max);
}

inline uint8_t srs_to_int8(uint32_t x, int shift) {
  return static_cast<uint8_t>(
      saturate<uint32_t>(std::round(x >> shift), -128, 127));
}

inline uint8_t srs_to_uint8(uint64_t x, int shift) {
  return static_cast<uint8_t>(
      saturate<uint8_t>(std::round(x >> shift), 0, 255));
}

template <typename T1, typename T2, typename T3, typename T4, typename qdqT>
void qdq_asym_golden(T1 A, T2 B, T3 X,
                     // uint32_t C0, uint32_t C2, uint32_t C1, uint32_t C3,
                     // uint8_t sqb, uint8_t sout,
                     gemm_qdq_param<qdqT> &param, T4 Y, bool B_is_transposed) {
  std::vector<uint32_t> ifmsum;
  for (int r = 0; r < A.num_rows; ++r) {
    uint32_t acc = 0;
    for (int c = 0; c < A.num_cols; ++c) {
      acc += A.at(r, c);
    }
    ifmsum.push_back(acc);
  }

  if (!B_is_transposed) {
    std::vector<uint32_t> wgtsum;
    for (int c = 0; c < B.num_cols; ++c) {
      uint32_t acc = 0;
      for (int r = 0; r < B.num_rows; ++r) {
        acc += B.at(r, c);
      }
      wgtsum.push_back(acc);
    }
    for (int c = 0; c < X.num_cols; ++c) {
      for (int r = 0; r < X.num_rows; ++r) {
        Y.at(r, c) = srs_to_uint8(
            (int64_t)X.at(r, c) * (int64_t)param.C2 +
                (int64_t)param.C1 * (int64_t)ifmsum[r] +
                (((int64_t)wgtsum[c] * (int64_t)param.C0 + (int64_t)param.C3) >>
                 param.sqb),
            param.sout);
      }
    }
  } else {
    // uint32_t wgtsum[B.num_rows];
    std::vector<uint32_t> wgtsum;
    for (int r = 0; r < B.num_rows; ++r) {
      uint32_t acc = 0;
      for (int c = 0; c < B.num_cols; ++c) {
        acc += B.at(r, c);
      }
      wgtsum.push_back(acc);
    }
    for (int c = 0; c < X.num_cols; ++c) {
      for (int r = 0; r < X.num_rows; ++r) {
        Y.at(r, c) = srs_to_uint8(
            (int64_t)X.at(r, c) * (int64_t)param.C2 +
                (int64_t)param.C1 * (int64_t)ifmsum[r] +
                (((int64_t)wgtsum[c] * (int64_t)param.C0 + (int64_t)param.C3) >>
                 param.sqb),
            param.sout);
      }
    }
  }
}

template <typename Ta, typename Tb>
int check_result_mha(Ta cpu_Y, Tb aie_Y, float max_pct_diff = 0.0,
                     bool enable_logging = true) {
  int err_count = 0;
  int max_err = 0;
  float sum_pct_diff = 0.0;
  float L2_norm = 0;
  for (int r = 0; r < cpu_Y.num_rows; ++r) {
    for (int c = 0; c < cpu_Y.num_cols; ++c) {
      int diff = std::abs(cpu_Y.at(r, c) - aie_Y.at(r, c));
      L2_norm += ((float)diff * (float)diff);
      float abs_ref = (float)std::abs(cpu_Y.at(r, c));
      float denominator = (abs_ref == 0.0f) ? abs_ref + 0.00001 : abs_ref;
      float pct_diff = 100.0 * (diff / denominator);
      // float pct_diff = 100.0 * (diff / (float) std::abs(cpu_Y.at(r, c)));
      // bool is_fail = (pct_diff > max_pct_diff) &&
      // (std::abs((cpu_Y.at(r,c)>>8) - (aie_Y.at(r,c)>>8)) > 2);
      bool is_fail = (pct_diff > max_pct_diff) &&
                     (std::abs((cpu_Y.at(r, c)) - (aie_Y.at(r, c))) > 2);

      sum_pct_diff += pct_diff;
      if (is_fail) {
        err_count += 1;
      }
      if (is_fail && enable_logging) {
        std::cout << "Y[" << r << ", " << c << "]: "
                  << "Expected: " << (int)(cpu_Y.at(r, c)) << ", "
                  << "Received: " << (int)(aie_Y.at(r, c)) << ", "
                  << "Pct Diff: " << pct_diff << "%\n";
      }
      max_err = (diff > max_err) ? diff : max_err;
    }
  }
  L2_norm = sqrt(L2_norm);
  std::cout << "L2_norm is " << L2_norm << std::endl;
  float avg_pct_diff = sum_pct_diff / (cpu_Y.num_rows * cpu_Y.num_cols);
  // std::cout << "Average Relative Error = " << avg_pct_diff << "%\n";
  std::cout << "Error Count = " << err_count << "\n";
  std::cout << "Max error = " << max_err << "\n";
  return err_count;
}

template <typename Ta, typename Tb>
float check_result(Ta cpu_Y, Tb aie_Y, float max_pct_diff = 0.0,
                   bool enable_logging = true, int valid_row = 512) {
  int err_count = 0;
  float sum_pct_diff = 0.0;
  float L2_norm = 0;
  int max_diff = 0, max_diff_cpuY = 0, max_diff_aieY = 0;
  int max_diff_h[12] = {0};
  for (int r = 0; r < valid_row; ++r) {

    for (int c = 0; c < cpu_Y.num_cols; ++c) {
      int diff = std::abs(cpu_Y.at(r, c) - aie_Y.at(r, c));
      L2_norm += ((float)diff * (float)diff);
      if (diff > max_diff) {
        max_diff = diff;
        max_diff_cpuY = cpu_Y.at(r, c);
        max_diff_aieY = aie_Y.at(r, c);
      }
      int idx_h = c / 64;
      if (diff > max_diff_h[idx_h]) {
        max_diff_h[idx_h] = diff;
      }
      float abs_ref = (float)std::abs(cpu_Y.at(r, c));
      float denominator = (abs_ref == 0.0f) ? abs_ref + 0.001 : abs_ref;
      float pct_diff = 100.0 * (diff / denominator);
      bool is_fail = (pct_diff > max_pct_diff) &&
                     (std::abs(cpu_Y.at(r, c) - aie_Y.at(r, c)) > 2);

      sum_pct_diff += pct_diff;
      if (is_fail) {
        err_count += 1;
      }
      if ((idx_h == 5) && enable_logging) {
        std::cout << "Y[" << r << ", " << c << "]: "
                  << "Expected: " << (int)(cpu_Y.at(r, c)) << ", "
                  << "Received: " << (int)(aie_Y.at(r, c)) << ", "
                  << "Diff: " << (int)diff << "\n";
      }
    }
  }
  for (int i = 0; i < 12; i++) {
    std::cout << i << ": " << max_diff_h[i] << std::endl;
  }
  L2_norm = sqrt(L2_norm);
  std::cout << "L2_norm is " << L2_norm << std::endl;
  float avg_pct_diff = sum_pct_diff / (cpu_Y.num_rows * cpu_Y.num_cols);
  std::cout << "Average Relative Error = " << avg_pct_diff << "%\n";
  std::cout << "Maximum Absolute difference = " << max_diff << ", "
            << "Expected: " << max_diff_cpuY << ", "
            << "Received: " << max_diff_aieY << "\n";
  std::cout << "Error Count = " << err_count << "\n";
  return avg_pct_diff;
}
} // namespace mhagprb_matrix
