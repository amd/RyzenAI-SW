#include "mhagprb_matrix.hpp"
#include <math.h>

using namespace mhagprb_matrix;

template <typename Tzp, int gprb_rows, int gprb_cols, int num_heads>
void init_gprb_params(
    GprbParams<int64_t, Tzp, gprb_rows, gprb_cols, num_heads> &prm) {
  prm.c0 = 10;
  prm.c1 = 1;
  prm.c2 = 1;
  prm.c3 = 0;
  prm.M = 32;
  prm.N = gprb_cols;
  prm.shift_Qb = 2;
  prm.shift_Qout = 8;
  prm.res = 0;
  for (int i = 0; i < gprb_rows * gprb_cols; ++i) {
    prm.proj_mat[i] = rand() % 16;
  }
  for (int i = 0; i < gprb_cols; ++i) {
    prm.qdq_bias[i] = 1;
  }
  for (int h = 0; h < num_heads; ++h) {
    prm.model_a[h] = float_to_bfloat16(4.0);
  }
  prm.model_b = float_to_bfloat16(3.0);
  prm.model_c = float_to_bfloat16(2.0);
  prm.act_scale = float_to_bfloat16(0.0125);
  prm.wgt_scale = float_to_bfloat16(0.5);
  prm.act_zero_point = 28;
  prm.wgt_zero_point = 8;
}
template <typename T> T srs(int64_t acc, int shift) {
  T val;
  if constexpr (std::is_same_v<T, int8_t>) {
    acc = int32_t(acc / float(1 << shift));
    if (acc > INT8_MAX) {
      val = INT8_MAX;
    } else if (acc < INT8_MIN) {
      val = INT8_MIN;
    } else {
      val = acc;
    }
  } else {
    acc = int32_t(acc / float(1 << shift));
    if (acc > UINT8_MAX) {
      val = UINT8_MAX;
    } else if (acc < 0) {
      val = 0;
    } else {
      val = acc;
    }
  }
  return val;
}

template <typename T>
uint8_t quant(int32_t acc, int32_t ifmsum, int32_t c1, int64_t c0,
              int32_t shift_Qb, int32_t c2, int32_t shift_Qout) {
  return srs<T>(((int64_t)acc * c2) + ((int64_t)ifmsum * c1) + (c0 << shift_Qb),
                shift_Qout);
}

float dequant_to_float(int i, int zp, bfloat16_t s) {
  return bfloat16_to_float(s) * (i - zp);
}

template <typename Tact, typename WgT, typename T_bias, int gprb_rows,
          int gprb_cols, int num_heads>
void calculate_gprb(
    RowMajorMatrix<Tact> query, RowMajorMatrix<WgT> wgt,
    GprbParams<int64_t, Tact, gprb_rows, gprb_cols, num_heads> &gprb,
    RowMajorMatrix<T_bias> bias, int head_idx, bool verbose = false) {
  for (int row = 0; row < query.num_rows; ++row) {

    int32_t acc[gprb_cols];

    // Matmul
    for (int col = 0; col < gprb_cols; ++col) {
      acc[col] = 0;
      for (int k = 0; k < gprb_rows; ++k) {
        acc[col] += query.at(row, k) * gprb.proj_mat[(k * gprb_cols) + col];
      }
    }

    uint32_t ifmsum = 0;
    for (int c = 0; c < query.num_cols; ++c) {
      ifmsum += query.at(row, c);
      // printf("r:%d c:%d, A.at(r,c) = %d, ifmsum[r] = %d\n", r, c,A.at(r,c),
      // ifmsum[r]);
    }

    // Reshape + Reduce
    float r0 = 0;
    float r1 = 0;
    for (int col = 0; col < gprb_cols; col += 2) {
      Tact q0 = quant<Tact>(acc[col + 0], ifmsum, gprb.c1, gprb.qdq_bias[col],
                            gprb.shift_Qb, gprb.c2, gprb.shift_Qout);
      Tact q1 = quant<Tact>(acc[col + 1], ifmsum, gprb.c1, gprb.qdq_bias[col],
                            gprb.shift_Qb, gprb.c2, gprb.shift_Qout);
      float dq0 = dequant_to_float(q0, gprb.act_zero_point, gprb.act_scale);
      float dq1 = dequant_to_float(q1, gprb.act_zero_point, gprb.act_scale);
      r0 += dq0;
      r1 += dq1;
      if (verbose) {
        std::cout << "acc0 = " << acc[col + 0] << "\n";
        std::cout << "acc1 = " << acc[col + 1] << "\n";
        std::cout << "q0 = " << static_cast<uint64_t>(q0) << "\n";
        std::cout << "q1 = " << static_cast<uint64_t>(q1) << "\n";
        std::cout << "dq0 = " << dq0 << "\n";
        std::cout << "dq1 = " << dq1 << "\n";
      }
    }
    if (verbose) {
      std::cout << "r0 = " << r0 << "\n";
      std::cout << "r1 = " << r1 << "\n";
    }

    // Sigmoid
    float s0 = 1.0 / (1.0 + powf(2.0, -r0));
    float s1 = 1.0 / (1.0 + powf(2.0, -r1));
    if (verbose) {
      std::cout << "s0 = " << s0 << "\n";
      std::cout << "s1 = " << s1 << "\n";
    }

    // Vector Ops
    float x0 = s0 * bfloat16_to_float(gprb.model_a[head_idx]);
    float x1 = x0 - bfloat16_to_float(gprb.model_b);
    float x2 = x1 * s1;
    float x3 = x2 + bfloat16_to_float(gprb.model_c);
    float result = x3;
    if (verbose) {
      std::cout << "x0 = " << x0 << "\n";
      std::cout << "x1 = " << x1 << "\n";
      std::cout << "x2 = " << x2 << "\n";
      std::cout << "x3 = " << x3 << "\n";
    }

    // Broadcast Mul
    for (int col = 0; col < wgt.num_cols; ++col) {
      float dq = dequant_to_float(wgt.at(row, col), gprb.wgt_zero_point,
                                  gprb.wgt_scale);
      bias.at(row, col) = dq * result;
    }
  }
}
