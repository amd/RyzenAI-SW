#pragma once

#ifndef LRN_MATRIX_HPP
#define LRN_MATRIX_HPP

#include <assert.h>
#include <float.h>
#include <iostream>
#include <math.h>
#include <numeric>
#include <stdlib.h>

namespace lrn_matrix {
static float Ep_lrn = 0.000009999999747378752;
int constexpr QDQparam_size = 16;
int constexpr QDQparam_info_size = 2;
int constexpr Mwgt = 8;
// for PSF and PSJ, set this be 1; for PSH, if uint8_q, set it be 0, if
// uint16_q, set it be 1
int constexpr lrn_isint16_idx = 2;
int constexpr lrn_qdq_ifm_zp_idx = 4;
union Float32Bits {
  uint32_t u;
  float f;
};

static const uint32_t kF32BfMantiBitDiff = 16;

static float bfloat2float(uint16_t bfloatBits) {
  Float32Bits floatBits;
  floatBits.u = static_cast<uint32_t>(bfloatBits) << kF32BfMantiBitDiff;
  return floatBits.f;
}

static int16_t float2bfloat(float floatValue) {
  if (std::isnan(floatValue))
    return std::signbit(floatValue) ? 0xFFC0 : 0x7FC0;

  Float32Bits floatBits;
  floatBits.f = floatValue;
  uint16_t bfloatBits;

  // Least significant bit of resulting bfloat.
  uint32_t lsb = (floatBits.u >> kF32BfMantiBitDiff) & 1;
  uint32_t roundingBias = 0x7fff + lsb;
  floatBits.u += roundingBias;
  bfloatBits = static_cast<int16_t>(floatBits.u >> kF32BfMantiBitDiff);
  return bfloatBits;
}

template <typename T>
void init_random_bias(T mat, int min = -128, int max = 128) {
  for (int i = 0; i < mat.num_cols; ++i) {
    mat.gamma(i) = float2bfloat(float((rand() % (max - min)) + min));
    mat.beta(i) = float2bfloat(float((rand() % (max - min)) + min));
  }
}

template <typename T> void init_random(T mat, int min = -128, int max = 128) {
  for (int i = 0; i < mat.num_rows; ++i) {
    for (int j = 0; j < mat.num_cols; ++j) {
      mat.at(i, j) = (rand() % (max - min)) + min;
    }
  }
}

template <typename T>
void initialize_random_bfloat16(std::vector<T> &vec, size_t size,
                                float data_min, float data_max) {
  for (int i = 0; i < size; i++) {
    vec[i] = float2bfloat(
        float((rand() / (RAND_MAX / (data_max - data_min))) + data_min));
  }
}

inline int row_major_index(int row, int col, int num_rows, int num_cols) {
  assert(row < num_rows);
  assert(col < num_cols);
  return (row * num_cols) + col;
}

inline int w8_index(int row, int col, int num_rows, int num_cols) {
  assert(row < num_rows);
  assert(col < num_cols);
  int constexpr zz = 8;
  return (row * zz) + (col % zz) + ((col / zz) * (num_rows * zz));
}

template <typename T, int subv_rows, int subv_cols> struct ActMatrix {
  int const num_rows;
  int const num_cols;
  T *const data;

  ActMatrix(int num_rows, int num_cols, void *data)
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
        row_major_index(rr, cc, (num_rows / subv_rows), (num_cols / subv_cols));
    int const idx = i + (ii * subv_size);
    assert(idx < num_rows * num_cols);
    return data[idx];
  }

  static int size(int num_rows, int num_cols) {
    return num_rows * num_cols * sizeof(T);
  }
};

template <typename T, int subv_cols> struct BiasVector {
  int const num_cols;
  T *const data;

  BiasVector(int num_cols, void *data)
      : num_cols(num_cols), data(static_cast<T *>(data)) {
    assert(num_cols % subv_cols == 0);
  }

  T &gamma(int col) {
    assert(col < num_cols);
    int idx = col;
    assert(idx < 2 * num_cols);
    return data[idx];
  }

  T &beta(int col) {
    assert(col < num_cols);
    int idx = col + num_cols;
    assert(idx < 2 * num_cols);
    return data[idx];
  }

  static int size(int num_cols) { return 2 * num_cols * sizeof(T); }
};

template <typename T> float average(std::vector<T> const &v) {
  if (v.empty()) {
    return 0;
  }

  auto const count = static_cast<float>(v.size());
  float sum = 0;
  for (int i = 0; i < count; i++) {
    // std::cout<<v[i]<<" ";
    sum += v[i];
  }
  // std::cout<<"\n";
  return sum / count;
}

template <typename T> float variance(std::vector<T> const &v, float mean) {
  if (v.empty()) {
    return 0;
  }
  auto const count = static_cast<float>(v.size());
  float sum_s = 0;
  for (int i = 0; i < count; i++) {
    sum_s += (v[i] - mean) * (v[i] - mean);
  }

  return sum_s / count;
}

template <typename in_el_type = int8_t, typename out_el_type = int8_t>
void compute_lrn(std::vector<std::vector<in_el_type>> &In,
                 std::vector<int16_t> G, std::vector<int16_t> B,
                 std::vector<std::vector<out_el_type>> &Out) {
  int num_rows = In.size();
  int num_cols = In[0].size();
  int G_cols = G.size();
  int B_cols = B.size();

  assert(num_rows > 0);
  assert(num_cols > 0);
  assert(num_cols == G_cols);
  assert(num_cols == B_cols);

  for (int r = 0; r < num_rows; r++) {
    std::vector<int16_t> inp_row;
    for (in_el_type i : In[r]) {
      inp_row.push_back(static_cast<int16_t>(i));
    }

    // compute mean and var for each row
    float mean = average<int16_t>(inp_row);
    float var = variance<int16_t>(inp_row, mean);
    // std::cout<<"mean "<<mean<<" var "<<var<<"\n";
    // std::cout<< G[r] << " "<< B[r]<<"\n";
    //  compute op1 and op2
    for (int c = 0; c < num_cols; c++) {
      float op1 = 1 / sqrt(var + Ep_lrn);
      float op2 = (mean * op1);
      // if (r == 0) {
      //     std::cout << "op1 " << op1 * G[c] << " op2 " << G[c] * op2 + B[c]
      //     << "\n"; std::cout << "G " << G[c] << " B " << B[c] << "\n";
      // }

      Out[r].push_back(
          static_cast<out_el_type>(In[r][c] * op1 * G[c] - G[c] * op2 + B[c]));
      // std::cout << Out[r][c] << " ";
    }
  }
}

template <typename in_el_type = int16_t, typename out_el_type = int8_t>
void compute_lrn_bfloat16(std::vector<std::vector<in_el_type>> &In,
                          std::vector<float> G, std::vector<float> B,
                          std::vector<std::vector<out_el_type>> &Out) {
  int num_rows = In.size();
  int num_cols = In[0].size();
  int G_cols = G.size();
  int B_cols = B.size();

  assert(num_rows > 0);
  assert(num_cols > 0);

  for (int r = 0; r < num_rows; r++) {

    std::vector<float> inp_row;
    for (in_el_type i : In[r]) {
      inp_row.push_back(bfloat2float(i));
    }

    // compute mean and var for each row
    float mean = average<float>(inp_row);
    float var = variance<float>(inp_row, mean);
    //  compute op1 and op2
    for (int c = 0; c < num_cols; c++) {
      float op1 = 1 / sqrt(var + Ep_lrn);
      float op2 = (mean * op1);

      Out[r].push_back(((inp_row[c] * op1 * G[c] - G[c] * op2 + B[c])));
    }
  }
}

template <typename in_el_type = int16_t, typename out_el_type = int8_t>
void compute_gpn_bfloat16(std::vector<std::vector<in_el_type>> &In,
                          std::vector<float> G, std::vector<float> B,
                          std::vector<std::vector<out_el_type>> &Out) {
  int num_rows = In.size();
  int num_cols = In[0].size();
  int G_cols = G.size();
  int B_cols = B.size();
  int a_idx;
  int cols_per_group = num_cols / 32;
  assert(num_rows > 0);
  assert(num_cols > 0);

  for (int ng = 0; ng < 32; ng++) {

    std::vector<float> inp_grp;
    for (int ii = 0; ii < num_rows; ii++) {
      for (int c = ng * cols_per_group; c < (ng + 1) * cols_per_group; c++) {
        inp_grp.push_back(bfloat2float(In[ii][c]));
      }
    }

    float mean = average<float>(inp_grp);
    float var = variance<float>(inp_grp, mean);

    for (int ii = 0; ii < num_rows; ii++) {
      for (int c = ng * cols_per_group; c < (ng + 1) * cols_per_group; c++) {
        float op1 = 1 / sqrt(var + Ep_lrn);
        float op2 = (mean * op1);
        a_idx = c;
        Out[ii].push_back(bfloat2float(In[ii][c]) * op1 * G[a_idx] -
                          G[a_idx] * op2 + B[a_idx]);
      }
    }
  }
}

template <typename T> T saturate(T val, T min, T max) {
  return std::min(std::max(val, min), max);
}

inline uint8_t srs_to_uint8(int32_t x) {
  return static_cast<uint8_t>(saturate<int32_t>(std::round(x), 0, 255));
}

static uint16_t srs_to_uint16(int32_t x) {
  return static_cast<uint16_t>(saturate<int32_t>(std::round(x), 0, UINT16_MAX));
}

template <typename Tout>
void q_bfloat2uint8(std::vector<std::vector<float>> &In, int16_t sc_out,
                    uint8_t zp_out, Tout Out) {
  float sc = bfloat2float(sc_out);
  int num_rows = In.size();
  int num_cols = In[0].size();
  for (int r = 0; r < num_rows; r++) {
    std::vector<float> inp_row;
    for (float i : In[r]) {
      inp_row.push_back(i);
    }
    for (int c = 0; c < num_cols; c++) {
      int32_t temp = static_cast<int>(inp_row[c] / sc + zp_out);
      Out.at(r, c) = srs_to_uint8(temp);
      // if (r == 0 && c == 0) {
      //     std::cout << inp_row[c] / sc + zp_out << std::endl;
      //     std::cout << temp << std::endl;
      //     std::cout << (int)Out.at(r, c) << std::endl;
      // }
    }
  }
}

template <typename Tout>
void q_bfloat2uint16(std::vector<std::vector<float>> &In, int16_t sc_out,
                     uint16_t zp_out, Tout Out) {
  float sc = bfloat2float(sc_out);
  int num_rows = In.size();
  int num_cols = In[0].size();
  for (int r = 0; r < num_rows; r++) {
    std::vector<float> inp_row;
    for (float i : In[r]) {
      inp_row.push_back(i);
    }
    for (int c = 0; c < num_cols; c++) {
      int32_t temp = static_cast<int>(inp_row[c] / sc + zp_out);
      Out.at(r, c) = srs_to_uint16(temp);
    }
  }
}

template <typename T1, typename T2>
void dequant_to_bfloat(std::vector<T1> &in_vec, std::vector<T2> &out_vec,
                       int zero_point, float scale) {
  auto num_cols = in_vec.size();
  for (int i = 0; i < num_cols; ++i) {
    out_vec[i] = float2bfloat((in_vec[i] - zero_point) * scale);
  }
}
} // namespace lrn_matrix
#endif // MATRIX_HPP
