#pragma once

#include <assert.h>
#include <cmath>
#include <iostream>
#include <stdlib.h>

namespace matmulgelu_matrix {
int constexpr Msubv = 64;
int constexpr Msubv_16 = 32;
int constexpr Ksubv = 128;
int constexpr Nsubv = 64;

int constexpr QDQparam_size = 16;
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

// gelu qdq idx
int constexpr gelu_dq_zp_idx = 0;
int constexpr gelu_dq_scale_idx = 1;
int constexpr gelu_q_zp_idx = 2;
int constexpr gelu_q_scale_idx = 3;
int constexpr gelu_isint16_idx = 4;

struct bfloat16_t {
  uint16_t value;
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

inline int h4_index(int row, int col, int num_rows, int num_cols) {
  int constexpr zz = 4;
  return (col * zz) + (row % zz) + ((row / zz) * (zz * num_cols));
}

template <typename T, int subv_rows, int subv_cols> struct ActMatrix {
  int const num_rows;
  int const num_cols;
  T *const data;

  ActMatrix(int num_rows, int num_cols, void *data)
      : num_rows(num_rows), num_cols(num_cols), data(static_cast<T *>(data)) {
    // assert(num_rows % subv_rows == 0);
    // assert(num_cols % subv_cols == 0);
  }

  T &at(int row, int col) {
    assert(row < num_rows);
    assert(col < num_cols);
    int const idx = row * num_cols + col;
    assert(idx < num_rows * num_cols);
    return data[idx];
  }

  static int size(int num_rows, int num_cols) {
    return num_rows * num_cols * sizeof(T);
  }
};

template <typename T, int subv_rows, int subv_cols> struct WgtMatrix {
  int const num_rows;
  int const num_cols;
  T *const data;

  WgtMatrix(int num_rows, int num_cols, void *data)
      : num_rows(num_rows), num_cols(num_cols), data(static_cast<T *>(data)) {
    assert(num_rows % subv_rows == 0);
    assert(num_cols % subv_cols == 0);
  }

  T &at(int row, int col) {
    assert(row < num_rows);
    assert(col < num_cols);
    int constexpr subv_size = subv_rows * subv_cols;
    int const r = row % subv_rows;
    int const c = col % subv_cols;
    int const i = w8_index(r, c, subv_rows, subv_cols);
    int const rr = row / subv_rows;
    int const cc = col / subv_cols;
    int const ii =
        col_major_index(rr, cc, (num_rows / subv_rows), (num_cols / subv_cols));
    int const idx = i + (ii * subv_size);
    assert(idx < num_rows * num_cols);
    return data[idx];
  }

  static int size(int num_rows, int num_cols) {
    return num_rows * num_cols * sizeof(T);
  }
};

template <typename T, int subv_rows, int subv_cols, int aie_rows = 4,
          int aie_cols = 2>
struct BiaVector {
  int const num_rows;
  int const num_cols;
  T *const data;

  BiaVector(int num_cols, void *data)
      : num_rows(1), num_cols(num_cols), data(static_cast<T *>(data)) {}

  T &at(int row, int col) {
    int const idx = col + row * num_cols;
    return data[idx];
  }

  static int size(int num_cols) { return num_cols * sizeof(T); }
};

template <typename T, int subv_rows, int subv_cols, int aie_rows = 4,
          int aie_cols = 2>
struct OutMatrix {
  int const num_rows;
  int const num_cols;
  T *const data;

  OutMatrix(int num_rows, int num_cols, void *data)
      : num_rows(num_rows), num_cols(num_cols), data(static_cast<T *>(data)) {
    // assert(num_rows % (subv_rows * aie_rows) == 0);
    //  assert(num_cols % (subv_cols * aie_cols) == 0);
  }

  T &at(int row, int col) {
    assert(row < num_rows);
    assert(col < num_cols);
    int const idx = row * num_cols + col;
    assert(idx < num_rows * num_cols);
    return data[idx];
  }

  static int size(int num_rows, int num_cols) {
    return num_rows * num_cols * sizeof(T);
  }
};

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

template <typename T, int subv_rows, int subv_cols> struct BiasVector {
  int const num_rows;
  int const num_cols;
  T *const data;

  BiasVector(int num_cols, void *data)
      : num_rows(1), num_cols(num_cols), data(static_cast<T *>(data)) {}

  T &at(int row, int col) {
    int const idx = col + row * num_cols;
    return data[idx];
  }

  static int size(int num_cols) { return num_cols * sizeof(T); }
};

template <typename T> void init_random(T mat, int min = -128, int max = 128) {
  for (int i = 0; i < mat.num_rows; ++i) {
    for (int j = 0; j < mat.num_cols; ++j) {
      mat.at(i, j) = (rand() % (max - min)) + min;
    }
  }
}

template <typename T1, typename T2, typename T3>
void dequant(T1 mat, T2 Out, float s, T3 z) {
  for (int i = 0; i < mat.num_rows; ++i) {
    for (int j = 0; j < mat.num_cols; ++j) {
      Out.at(i, j) = (float)(mat.at(i, j) - z) * s;
    }
  }
}

template <typename T> T saturate(T val, T min, T max) {
  return std::min(std::max(val, min), max);
}

template <typename T1, typename T2, typename T3>
void quant(T1 mat, T2 Out, float s, T3 z, std::string Ytype) {
  bfloat16_t temp;
  if (Ytype == "uint8") {
    for (int i = 0; i < mat.num_rows; ++i) {
      for (int j = 0; j < mat.num_cols; ++j) {
        temp.value = (uint16_t)mat.at(i, j);
        Out.at(i, j) = static_cast<uint8_t>(saturate<int32_t>(
            (std::round(bfloat16_to_float(temp) / s) + z), 0, 255));
      }
    }
  } else {
    for (int i = 0; i < mat.num_rows; ++i) {
      for (int j = 0; j < mat.num_cols; ++j) {
        temp.value = (uint16_t)mat.at(i, j);
        Out.at(i, j) = static_cast<uint16_t>(saturate<int32_t>(
            (std::round(bfloat16_to_float(temp) / s) + z), 0, 65535));
      }
    }
  }
}

inline int32_t srs_to_int32(int64_t x) {
  if (x > INT32_MAX) {
    x = INT32_MAX;
  } else if (x < INT32_MIN) {
    x = INT32_MIN;
  }
  return static_cast<int32_t>(x);
}

inline int16_t srs_to_int16(int64_t x) {
  if (x > INT16_MAX) {
    x = INT16_MAX;
  } else if (x < INT16_MIN) {
    x = INT16_MIN;
  }
  return static_cast<int16_t>(x);
}

inline float gelu_golden(float in) {
  // float exp_x = std::exp(-in);
  // float sg  = in/(1+exp_x);
  // auto inf = static_cast<float>(in);
  float xr2 = in / (std::sqrt(2));
  float t = std::erf(xr2);
  float g = in * 0.5 * (1.0 + t);

  return g;
}

inline int8_t srs_to_int8(int64_t x, int shift, bool sign = true) {
  if (sign)
    return static_cast<int8_t>(
        saturate<int32_t>(((x >> (shift - 1)) + 1) >> 1, -128, 127));
  else {
    int64_t inp_floor = (x >> shift);
    int64_t inp_frac = x - (inp_floor << shift);
    if (inp_frac == (1 << (shift - 1))) {
      if (inp_floor % 2) { // odd
        return static_cast<uint8_t>(saturate<int32_t>(inp_floor + 1, 0, 255));
      } else {
        return static_cast<uint8_t>(saturate<int32_t>(inp_floor, 0, 255));
      }
    } else {
      return static_cast<uint8_t>(
          saturate<int32_t>(((x >> (shift - 1)) + 1) >> 1, 0, 255));
    }
  }
}

inline int16_t srs_to_int16(int64_t x, int shift, bool sign = true) {
  if (sign)
    return static_cast<int16_t>(
        saturate<int32_t>(((x >> (shift - 1)) + 1) >> 1, INT16_MIN, INT16_MAX));
  else {
    int64_t inp_floor = (x >> shift);
    int64_t inp_frac = x - (inp_floor << shift);
    if (inp_frac == (1 << (shift - 1))) {
      if (inp_floor % 2) { // odd
        return static_cast<uint16_t>(
            saturate<uint32_t>(inp_floor + 1, 0, UINT16_MAX));
      } else {
        return static_cast<uint16_t>(
            saturate<uint32_t>(inp_floor, 0, UINT16_MAX));
      }
    } else {
      return static_cast<uint16_t>(
          saturate<uint32_t>(((x >> (shift - 1)) + 1) >> 1, 0, UINT16_MAX));
    }
  }
}

inline int32_t srs_to_int32(int64_t x, int shift, bool sign = true) {
  if (sign)
    return static_cast<int32_t>(
        saturate<int32_t>(((x >> (shift - 1)) + 1) >> 1, INT32_MIN, INT32_MAX));
  else {
    int64_t inp_floor = (x >> shift);
    int64_t inp_frac = x - (inp_floor << shift);
    if (inp_frac == (1 << (shift - 1))) {
      if (inp_floor % 2) { // odd
        return static_cast<uint32_t>(
            saturate<uint32_t>(inp_floor + 1, 0, UINT32_MAX));
      } else {
        return static_cast<uint32_t>(
            saturate<uint32_t>(inp_floor, 0, UINT32_MAX));
      }
    } else {
      return static_cast<uint32_t>(
          saturate<uint32_t>(((x >> (shift - 1)) + 1) >> 1, 0, UINT32_MAX));
    }
  }
}

inline int64_t sls_to_int64(int64_t x, int shift, bool sign = true) {
  // NOTE: No rounding when upshifted
  if (sign)
    return static_cast<int64_t>(x << shift);
  else {
    return static_cast<uint64_t>(x << shift);
  }
}

template <typename Ta, typename Toi, typename Tout>
void qdq_golden(Ta A, Toi X, int32_t C2, int32_t C1, int64_t *C0, uint8_t sqb,
                uint8_t sout, Tout Y, std::string Ytype) {
  // uint32_t ifmsum[A.num_rows];
  std::vector<int64_t> ifmsum;
  for (int r = 0; r < A.num_rows; ++r) {
    int64_t acc = 0;
    for (int c = 0; c < A.num_cols; ++c) {
      acc += A.at(r, c);
      // printf("r:%d c:%d, A.at(r,c) = %d, ifmsum[r] = %d\n", r, c,A.at(r,c),
      // ifmsum[r]);
    }
    ifmsum.push_back(acc);
  }

  for (int c = 0; c < X.num_cols; ++c) {
    for (int r = 0; r < X.num_rows; ++r) {
      if (Ytype == "uint8") {
        Y.at(r, c) = srs_to_int8((int64_t)X.at(r, c) * (int64_t)C2 +
                                     (int64_t)C1 * (int64_t)ifmsum[r] +
                                     ((int64_t)C0[c] << sqb),
                                 sout, false);
      } else { // uint16
        Y.at(r, c) = srs_to_int16((int64_t)X.at(r, c) * (int64_t)C2 +
                                      (int64_t)C1 * (int64_t)ifmsum[r] +
                                      ((int64_t)C0[c] << sqb),
                                  sout, false);
      }
      // if ((r == 0 && c == 0) || (r == 0 && c == 1)) {
      //    printf("r:%d, c:%d, X=%d, C2=%d, C1=%d, ifmsum[r]=%d\n", r, c,
      //    (int64_t)X.at(r, c), (int64_t)C2, (int64_t)C1, (int64_t)ifmsum[r]);
      //    printf("r:%d, c:%d, C0[c]:%lld, sqb:%d, sout:%d, X*+: %lld,
      //    shifted:%d\n", r, c, (int64_t)C0[c], sqb, sout, (int64_t)X.at(r, c)
      //    *
      //        (int64_t)C2 + (int64_t)C1 * (int64_t)ifmsum[r] + ((int64_t)C0[c]
      //        << sqb), (int64_t)((int64_t)X.at(r, c)
      //    * (int64_t)C2 + (int64_t)C1 * (int64_t)ifmsum[r] + ((int64_t)C0[c]
      //    << sqb)) >> sout);
      // }
    }
  }
}

template <typename T1, typename T2, typename T3, typename T4>
void qdq_asym_golden(T1 A, T2 B, T3 X, int32_t C0, int32_t C2, int32_t C1,
                     int32_t C3, uint8_t sqb, uint8_t sout, T4 Y) {
  // int32_t ifmsum[A.num_rows];
  std::vector<int32_t> ifmsum;
  for (int r = 0; r < A.num_rows; ++r) {
    int32_t acc = 0;
    for (int c = 0; c < A.num_cols; ++c) {
      acc += A.at(r, c);
      // printf("r:%d c:%d, A.at(r,c) = %d, ifmsum[r] = %d\n", r, c,A.at(r,c),
      // ifmsum[r]);
    }
    ifmsum.push_back(acc);
  }
  // int32_t wgtsum[B.num_cols];
  std::vector<int32_t> wgtsum;
  for (int c = 0; c < B.num_cols; ++c) {
    int32_t acc = 0;
    for (int r = 0; r < B.num_rows; ++r) {
      acc += B.at(r, c);
    }
    // printf("c:%d, wgtsum[r] = %d\n", c, wgtsum[c]);
    wgtsum.push_back(acc);
  }

  for (int c = 0; c < X.num_cols; ++c) {
    for (int r = 0; r < X.num_rows; ++r) {
      Y.at(r, c) = srs_to_int8(X.at(r, c) * C2 + C1 * ifmsum[r] +
                                   ((wgtsum[c] * C0 + C3) >> sqb),
                               sout);
      // printf("r:%d, c:%d, (int64_t)X.at(r, c)=%d, (int64_t)C1 *
      // (int64_t)ifmsum[r] =%d, ((int64_t)C0[c] << sqb)=%d, C2=%d,
      // (int64_t)ifmsum[r]=%d\n", r,c, (int64_t)X.at(r, c), (int64_t)C1 *
      // (int64_t)ifmsum[r]  , ((int64_t)C0[c] << sqb), C2, (int64_t)ifmsum[r]);
      // printf("r:%d, c:%d, X:%d, C1:%d, C0[c]:%d, sqb:%d, sout:%d, X*+: %d,
      // shifted:%d\n", r,c,X.at(r, c),C1,C0[c],sqb,sout,X.at(r, c) * C1 +
      // (C0[r] << sqb), (X.at(r, c) * C1 + (C0[r] << sqb))>>sout);
    }
  }
}

template <typename Tx, typename Tw, typename Ty>
void cpu_matmul(Tx X, Tw W, Ty Y, std::string Ytype) {
  for (int r = 0; r < Y.num_rows; ++r) {
    for (int c = 0; c < Y.num_cols; ++c) {
      uint32_t acc = 0;
      for (int k = 0; k < X.num_cols; ++k) {
        acc += X.at(r, k) * W.at(k, c);
      }
      if (Ytype == "int16") {
        Y.at(r, c) = srs_to_int16(acc, true);
      } else if (Ytype == "int8") {
        Y.at(r, c) = srs_to_int8(acc, 12, true);
      } else if (Ytype == "int32") {
        Y.at(r, c) = acc;
      }
    }
  }
}

template <typename Ta, typename Tw, typename To>
void cpu_matmul(Ta X, Tw W, To Y, int shift, int Msubv, int Ksubv, int Nsubv) {
  for (int r = 0; r < Y.num_rows; ++r) {
    for (int c = 0; c < Y.num_cols; ++c) {
      int64_t acc = 0;
      for (int k_shard = 0; k_shard < (X.num_cols) / Ksubv; ++k_shard) {
        /*
         *  Presently the AIE GEMM kernel does upshift before loading the acc
         *  Followed by down shift when writing back the acc to TDM
         *  This is done at the subvol boundary (32 x 128 x 64)
         *  For this reason, the inner dim is broken down to shards of 128
         *  and encapsulated by up/down shift as seen below
         */
        acc = sls_to_int64(acc, shift, false);
        for (int k = 0; k < Ksubv; ++k) {
          acc += (X.at(r, ((k_shard * Ksubv) + k)) *
                  W.at(((k_shard * Ksubv) + k), c));
        }
        acc = srs_to_int32(acc, shift, false);
      }
      Y.at(r, c) = acc;
    }
  }
}
template <typename T> int check_result(T cpu_Y, T aie_Y) {
  int fail = 0;
  int err_count = 0;
  int max_diff = 0;
  float L2_norm = 0;
  for (int r = 0; r < aie_Y.num_rows; ++r) {
    for (int c = 0; c < aie_Y.num_cols; ++c) {
      int32_t diff = std::abs(cpu_Y.at(r, c) - aie_Y.at(r, c));
      L2_norm += ((float)diff * (float)diff);
      if (diff > max_diff)
        max_diff = diff;
      if (diff > 1) {
        // std::cout << "ERROR: Y[" << r << ", " << c << "]: "
        //          << "Expected: " << int(cpu_Y.at(r, c)) << ", "
        //          << "Received: " << int(aie_Y.at(r, c)) << ", "
        //          << "Diff: " << int(diff) << "\n";
        fail = 1;
        err_count++;
      }
    }
  }
  L2_norm = sqrt(L2_norm);
  std::cout << "max_diff is " << max_diff << std::endl;
  std::cout << "L2_norm is " << L2_norm << std::endl;
  return err_count;
}

} // namespace matmulgelu_matrix
