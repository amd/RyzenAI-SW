#pragma once

#include <assert.h>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <stdlib.h>

namespace matmul_a16w8_mladf_matrix {
int constexpr Msubv = 32;
int constexpr Msubv_16 = 16;
int constexpr Ksubv = 128;
int constexpr Nsubv = 16;

int constexpr QDQparam_size = 2;
// general qdq idx
int constexpr qdq_c1_idx = 0;
int constexpr qdq_c2_idx = 1;

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

// inline int16_t srs_to_int16(int64_t x, int shift, bool sign = true) {
//   if (sign)
//     return static_cast<int16_t>(
//         saturate<int32_t>(((x >> (shift - 1)) + 1) >> 1, INT16_MIN,
//         INT16_MAX));
//   else {
//     int64_t inp_floor = (x >> shift);
//     int64_t inp_frac = x - (inp_floor << shift);
//     if (inp_frac == (1 << (shift - 1))) {
//       if (inp_floor % 2) { // odd
//         return static_cast<uint16_t>(
//             saturate<uint32_t>(inp_floor + 1, 0, UINT16_MAX));
//       } else {
//         return static_cast<uint16_t>(
//             saturate<uint32_t>(inp_floor, 0, UINT16_MAX));
//       }
//     } else {
//       return static_cast<uint16_t>(
//           saturate<uint32_t>(((x >> (shift - 1)) + 1) >> 1, 0, UINT16_MAX));
//     }
//   }
// }

// inline int32_t srs_to_int32(int64_t x, int shift, bool sign = true) {
//   if (sign)
//     return static_cast<int32_t>(
//         saturate<int32_t>(((x >> (shift - 1)) + 1) >> 1, INT32_MIN,
//         INT32_MAX));
//   else {
//     int64_t inp_floor = (x >> shift);
//     int64_t inp_frac = x - (inp_floor << shift);
//     if (inp_frac == (1 << (shift - 1))) {
//       if (inp_floor % 2) { // odd
//         return static_cast<uint32_t>(
//             saturate<uint32_t>(inp_floor + 1, 0, UINT32_MAX));
//       } else {
//         return static_cast<uint32_t>(
//             saturate<uint32_t>(inp_floor, 0, UINT32_MAX));
//       }
//     } else {
//       return static_cast<uint32_t>(
//           saturate<uint32_t>(((x >> (shift - 1)) + 1) >> 1, 0, UINT32_MAX));
//     }
//   }
// }

inline int64_t sls_to_int64(int64_t x, int shift, bool sign = true) {
  // NOTE: No rounding when upshifted
  if (sign)
    return static_cast<int64_t>(x << shift);
  else {
    return static_cast<uint64_t>(x << shift);
  }
}

inline uint32_t srs_to_int32_0(int64_t x, int shift) {
  double y = static_cast<double>(x) / std::pow(2, shift);
  y = std::round(y);
  return static_cast<uint32_t>(saturate<uint32_t>(y, 0, UINT32_MAX));
}

// template <typename Tx, typename Tw, typename Ty>
// void cpu_matmul(Tx X, Tw W, Ty Y, std::string Ytype) {
//   for (int r = 0; r < Y.num_rows; ++r) {
//     for (int c = 0; c < Y.num_cols; ++c) {
//       uint32_t acc = 0;
//       for (int k = 0; k < X.num_cols; ++k) {
//         acc += X.at(r, k) * W.at(k, c);
//       }
//       if (Ytype == "int16") {
//         Y.at(r, c) = srs_to_int16(acc, true);
//       } else if (Ytype == "int8") {
//         Y.at(r, c) = srs_to_int8(acc, 12, true);
//       } else if (Ytype == "int32") {
//         Y.at(r, c) = acc;
//       }
//     }
//   }
// }

inline uint16_t srs_to_int16(int64_t x, int shift) {
  std::fesetround(FE_TONEAREST);
  double y = static_cast<double>(x) / std::pow(2, shift);
  y = std::nearbyint(y);
  return static_cast<uint16_t>(saturate<uint16_t>(y, 0, UINT16_MAX));
}

inline uint32_t srs_to_int32(int64_t x, int shift) {
  std::fesetround(FE_TONEAREST);
  double y = static_cast<double>(x) / std::pow(2, shift);
  y = std::nearbyint(y);
  return static_cast<uint32_t>(saturate<uint32_t>(y, 0, UINT32_MAX));
}

template <typename Tx, typename Tw, typename Ty>
void cpu_qdq_matmul(Tx X, Tw W, Ty Y, std::vector<int64_t> C0, uint32_t C1,
                    uint32_t C2, int32_t shift_gemm_out, int32_t shift_qdq_out,
                    std::string Ytype) {
  for (int r = 0; r < Y.num_rows; ++r) {
    for (int c = 0; c < Y.num_cols; ++c) {
      int64_t acc = 0;
      uint32_t a_sum = 0;
      for (int k = 0; k < X.num_cols; ++k) {
        acc += X.at(r, k) * W.at(k, c);
        a_sum += X.at(r, k);
      }
      acc = srs_to_int32(acc, shift_gemm_out);
      acc = C2 * acc + a_sum * C1 + C0[c];

      if (Ytype == "uint16") {
        Y.at(r, c) = srs_to_int16(acc, shift_qdq_out);
      } else {
        Y.at(r, c) = acc;
      }
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
        if (err_count < 128)
          std::cout << "ERROR: Y[" << r << ", " << c << "]: "
                    << "Expected: " << int(cpu_Y.at(r, c)) << ", "
                    << "Received: " << int(aie_Y.at(r, c)) << ", "
                    << "Diff: " << int(diff) << "\n";
        fail = 1;
        err_count++;
      }
    }
  }
  L2_norm = sqrt(L2_norm);
  std::cout << "num_rows=" << aie_Y.num_rows << " num_cols=" << aie_Y.num_cols
            << std::endl;
  std::cout << "max_diff is " << max_diff << std::endl;
  std::cout << "L2_norm is " << L2_norm << std::endl;
  return err_count;
}

} // namespace matmul_a16w8_mladf_matrix
