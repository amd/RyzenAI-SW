#pragma once

#include <array>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <stdint.h>
#include <type_traits>

namespace iconv_matrix {
int constexpr C_IN_SPLIT_DWC = 32;
int constexpr C_OUT_SPLIT_DWC = 32;
int constexpr C_IN_SPLIT_CONV = 16;
int constexpr C_OUT_SPLIT_CONV = 16;
int constexpr C_IN_SPLIT_CONV_PSR = 16;
int constexpr C_OUT_SPLIT_CONV_PSR = 8;
int constexpr C_IN_SPLIT_CONV7 = 8;
int constexpr C_OUT_SPLIT_CONV7 = 16;
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
int constexpr qdq_wgt_zp_idx = 10;
int constexpr qdq_ifm_zp_idx = 11;

// Cis and Cos
using SUBV_T = std::array<int, 2>;
constexpr std::array<std::pair<int, SUBV_T>, 4> subv_conv3x3_4x4 = {
    {{0, {16, 16}}, {1, {32, 16}}, {2, {40, 16}}, {3, {64, 16}}}};

inline constexpr SUBV_T get_subv(const int &key) {
  for (size_t i = 0; i < subv_conv3x3_4x4.size(); ++i) {
    if (subv_conv3x3_4x4[i].first == key) {
      return subv_conv3x3_4x4[i].second; // Key found
    }
  }
  return SUBV_T(); // Key not found
}

inline int search_subv_mode(int Xi) {
  int out_mode = 0;
  if (Xi == 64) {
    out_mode = 0;
  } else if (Xi == 32) {
    out_mode = 1;
  } else if (Xi == 16) {
    out_mode = 2;
  } else if (Xi == 8) {
    out_mode = 3;
  }
  return (out_mode);
}

static int round_up_to_multiple(int x, int m) { return ((x + m - 1) / m) * m; }

template <typename T> struct ActTensor {
  int const C;
  int const Y;
  int const X;
  T *const data;

  ActTensor(int C, int Y, int X, void *data)
      : C(C), Y(Y), X(X), data(static_cast<T *>(data)) {}

  T &at(int c, int y, int x) {
    assert(c < C);
    assert(y < Y);
    assert(x < X);
    int idx = (y * X * C) + (x * C) + c;
    assert(idx < C * Y * X);
    return data[idx];
  }

  void print(char const *msg = nullptr) {
    if (msg != nullptr) {
      std::cout << msg;
    }
    for (int c = 0; c < C; ++c) {
      for (int y = 0; y < Y; ++y) {
        for (int x = 0; x < X; ++x) {
          if (std::is_integral<T>::value) {
            std::cout << static_cast<int64_t>(at(c, y, x)) << " ";
          } else {
            std::cout << at(c, y, x) << " ";
          }
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
  }

  void init_random(int64_t min = 0, int64_t max = 4) {
    for (int c = 0; c < C; ++c) {
      for (int y = 0; y < Y; ++y) {
        for (int x = 0; x < X; ++x) {
          if (std::is_integral<T>::value) {
            at(c, y, x) = (rand() % (max - min)) + min;
          } else {
            at(c, y, x) = ((max - min) * (rand() / float(RAND_MAX))) + min;
          }
        }
      }
    }
  }

  static int size(int C, int Y, int X) { return C * Y * X * sizeof(T); }
};

template <typename T, int Cos, int Cis> struct ConvWgtTensor;

template <int Cos, int Cis> struct ConvWgtTensor<uint8_t, Cos, Cis> {
  using Tw = uint8_t;
  // static int constexpr Cos = 16;
  // static int constexpr Cis = 16;
  static int constexpr subv_align_bytes = 64;
  static int constexpr subv_qdq_c0_size = Cos * sizeof(int64_t);
  static int constexpr subv_qdq_c1_size = sizeof(int32_t);
  static int constexpr subv_qdq_c2_size = sizeof(int32_t);
  static int constexpr subv_shift_tdm_size = sizeof(int32_t);
  static int constexpr subv_shift_res_size = sizeof(int32_t);
  static int constexpr subv_zp_wgt_size = sizeof(int32_t);
  static int constexpr subv_qdq_size = subv_qdq_c0_size + subv_qdq_c1_size +
                                       subv_qdq_c2_size + subv_shift_tdm_size +
                                       subv_shift_res_size + subv_zp_wgt_size;

  int const Co;
  int const Ci;
  int const Ky;
  int const Kx;
  int const subv_wgt_size;
  int const subv_size;
  char *const data;

  ConvWgtTensor(int Co, int Ci, int Ky, int Kx, void *data)
      : Co(Co), Ci(Ci), Ky(Ky), Kx(Kx),
        subv_wgt_size(round_up_to_multiple(Cos * Cis * Ky * Kx * sizeof(Tw),
                                           subv_align_bytes)),
        subv_size(round_up_to_multiple(subv_wgt_size + subv_qdq_size,
                                       subv_align_bytes)),
        data(static_cast<char *>(data)) {
    assert(Co % 8 == 0);
    assert(Ci % 8 == 0);
    assert(Co >= Cos);
    assert(Ci >= Cis);
  }

  char *subv_ptr(int co, int ci) {
    //
    // Indexing equation determined by tiling, with the following order.
    // Read this list from right-to-left to determine inner-to-outermost
    // traversal order.
    //
    // Co Ci
    //
    int offset = subv_size * (((co / Cos) * (Ci / Cis)) + (ci / Cis));
    return data + offset;
  }

  Tw &at(int co, int ci, int ky, int kx) {
    assert(co < Co);
    assert(ci < Ci);
    assert(ky < Ky);
    assert(kx < Kx);
    //
    // Indexing equation detemined by kernel, with the following order.
    // Read this list from right-to-left to determine inner-to-outermost
    // traversal order.
    //
    // Co:Cos Ci:Cis Ky Kx Ci:8 Co:8
    //
    int subv_idx = (((co % Cos) / 8) * Cis * Ky * Kx * 8) +
                   (((ci % Cis) / 8) * Ky * Kx * 8 * 8) + (ky * Kx * 8 * 8) +
                   (kx * 8 * 8) + ((ci % 8) * 8) + (co % 8);
    Tw *ptr = reinterpret_cast<Tw *>(subv_ptr(co, ci));
    return ptr[subv_idx];
  }

  void set_qdq_c0(int co, int64_t coeff0) {
    for (int ci = 0; ci < Ci; ci += Cis) {
      int64_t *qdq_c0 =
          reinterpret_cast<int64_t *>(subv_ptr(co, ci) + subv_wgt_size);
      qdq_c0[co % Cos] = coeff0;
    }
  }

  void set_qdq_c1(int32_t coeff1) {
    for (int co = 0; co < Co; co += Cos) {
      for (int ci = 0; ci < Ci; ci += Cis) {
        int32_t *qdq_c1 = reinterpret_cast<int32_t *>(
            subv_ptr(co, ci) + subv_wgt_size + subv_qdq_c0_size);
        *qdq_c1 = coeff1;
      }
    }
  }

  void set_qdq_c2(int32_t coeff2) {
    for (int co = 0; co < Co; co += Cos) {
      for (int ci = 0; ci < Ci; ci += Cis) {
        int32_t *qdq_c2 =
            reinterpret_cast<int32_t *>(subv_ptr(co, ci) + subv_wgt_size +
                                        subv_qdq_c0_size + subv_qdq_c1_size);
        *qdq_c2 = coeff2;
      }
    }
  }

  void set_shift_tdm(int32_t shift) {
    for (int co = 0; co < Co; co += Cos) {
      for (int ci = 0; ci < Ci; ci += Cis) {
        int32_t *shift_tdm = reinterpret_cast<int32_t *>(
            subv_ptr(co, ci) + subv_wgt_size + subv_qdq_c0_size +
            subv_qdq_c1_size + subv_qdq_c2_size);
        *shift_tdm = shift;
      }
    }
  }

  void set_shift_res(int32_t shift) {
    for (int co = 0; co < Co; co += Cos) {
      for (int ci = 0; ci < Ci; ci += Cis) {
        int32_t *shift_res = reinterpret_cast<int32_t *>(
            subv_ptr(co, ci) + subv_wgt_size + subv_qdq_c0_size +
            subv_qdq_c1_size + subv_qdq_c2_size + subv_shift_tdm_size);
        *shift_res = shift;
      }
    }
  }

  void set_zp_wgt(int32_t zp) {
    for (int co = 0; co < Co; co += Cos) {
      for (int ci = 0; ci < Ci; ci += Cis) {
        int32_t *zp_wgt = reinterpret_cast<int32_t *>(
            subv_ptr(co, ci) + subv_wgt_size + subv_qdq_c0_size +
            subv_qdq_c1_size + subv_qdq_c2_size + subv_shift_tdm_size +
            subv_shift_res_size);
        *zp_wgt = zp;
      }
    }
  }

  void print(char const *msg = nullptr) {
    if (msg != nullptr) {
      std::cout << msg;
    }
    for (int co = 0; co < Co; ++co) {
      for (int ci = 0; ci < Ci; ++ci) {
        for (int ky = 0; ky < Ky; ++ky) {
          for (int kx = 0; kx < Kx; ++kx) {
            std::cout << static_cast<int64_t>(at(co, ci, ky, kx)) << " ";
          }
          std::cout << "\n";
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
  }

  void init_random(int64_t min = 0, int64_t max = 8) {
    for (int co = 0; co < Co; ++co) {
      for (int ci = 0; ci < Ci; ++ci) {
        for (int ky = 0; ky < Ky; ++ky) {
          for (int kx = 0; kx < Kx; ++kx) {
            at(co, ci, ky, kx) = (rand() % (max - min)) + min;
          }
        }
      }
    }
  }

  static int size(int Co, int Ci, int Ky, int Kx) {
    int num_subv = (Co / Cos) * (Ci / Cis);
    int subv_wgt_size = round_up_to_multiple(Cos * Cis * Ky * Kx * sizeof(Tw),
                                             subv_align_bytes);
    int subv_size =
        round_up_to_multiple(subv_wgt_size + subv_qdq_size, subv_align_bytes);
    return num_subv * subv_size;
  }
};

template <typename T, int Cs> struct DwcWgtTensor;

template <int Cs> struct DwcWgtTensor<uint8_t, Cs> {
  using Tw = uint8_t;
  // static int constexpr Cs = 32;
  static int constexpr subv_align_bytes = 64;
  static int constexpr subv_qdq_c0_size = Cs * sizeof(int64_t);
  static int constexpr subv_qdq_c1_size = sizeof(int32_t);
  static int constexpr subv_qdq_c2_size = sizeof(int32_t);
  static int constexpr subv_shift_tdm_size = sizeof(int32_t);
  static int constexpr subv_shift_res_size = sizeof(int32_t);
  static int constexpr subv_zp_wgt_size = sizeof(int32_t);
  static int constexpr subv_qdq_size = subv_qdq_c0_size + subv_qdq_c1_size +
                                       subv_qdq_c2_size + subv_shift_tdm_size +
                                       subv_shift_res_size + subv_zp_wgt_size;

  int const C;
  int const Ky;
  int const Kx;
  int const subv_wgt_size;
  int const subv_size;
  char *const data;

  DwcWgtTensor(int C, int Ky, int Kx, void *data)
      : C(C), Ky(Ky), Kx(Kx), subv_wgt_size(round_up_to_multiple(
                                  Cs * Ky * Kx * sizeof(Tw), subv_align_bytes)),
        subv_size(round_up_to_multiple(subv_wgt_size + subv_qdq_size,
                                       subv_align_bytes)),
        data(static_cast<char *>(data)) {
    assert(C >= Cs);
  }

  char *subv_ptr(int c) {
    int offset = subv_size * (c / Cs);
    return data + offset;
  }

  Tw &at(int c, int ky, int kx) {
    assert(c < C);
    assert(ky < Ky);
    assert(kx < Kx);
    //
    // Indexing equation determined by the kernel, with the following order.
    // Read this list from right-to-left to determine inner-to-outermost
    // traversal order.
    //
    // C:Cs Ky Kx C:8
    //
    int subv_idx =
        (((c % Cs) / 8) * Ky * Kx * 8) + (ky * Kx * 8) + (kx * 8) + (c % 8);
    Tw *ptr = reinterpret_cast<Tw *>(subv_ptr(c));
    return ptr[subv_idx];
  }

  void set_qdq_c0(int c, int64_t coeff0) {
    int64_t *qdq_c0 = reinterpret_cast<int64_t *>(subv_ptr(c) + subv_wgt_size);
    qdq_c0[c % Cs] = coeff0;
  }

  void set_qdq_c1(int32_t coeff1) {
    for (int c = 0; c < C; c += Cs) {
      int32_t *qdq_c1 = reinterpret_cast<int32_t *>(
          subv_ptr(c) + subv_wgt_size + subv_qdq_c0_size);
      *qdq_c1 = coeff1;
    }
  }

  void set_qdq_c2(int32_t coeff2) {
    for (int c = 0; c < C; c += Cs) {
      int32_t *qdq_c2 = reinterpret_cast<int32_t *>(
          subv_ptr(c) + subv_wgt_size + subv_qdq_c0_size + subv_qdq_c1_size);
      *qdq_c2 = coeff2;
    }
  }

  void set_shift_tdm(int32_t shift) {
    for (int c = 0; c < C; c += Cs) {
      int32_t *shift_tdm = reinterpret_cast<int32_t *>(
          subv_ptr(c) + subv_wgt_size + subv_qdq_c0_size + subv_qdq_c1_size +
          subv_qdq_c2_size);
      *shift_tdm = shift;
    }
  }

  void set_shift_res(int32_t shift) {
    for (int c = 0; c < C; c += Cs) {
      int32_t *shift_res = reinterpret_cast<int32_t *>(
          subv_ptr(c) + subv_wgt_size + subv_qdq_c0_size + subv_qdq_c1_size +
          subv_qdq_c2_size + subv_shift_tdm_size);
      *shift_res = shift;
    }
  }

  void set_zp_wgt(int32_t zp) {
    for (int c = 0; c < C; c += Cs) {
      int32_t *zp_wgt = reinterpret_cast<int32_t *>(
          subv_ptr(c) + subv_wgt_size + subv_qdq_c0_size + subv_qdq_c1_size +
          subv_qdq_c2_size + subv_shift_tdm_size + subv_shift_res_size);
      *zp_wgt = zp;
    }
  }

  void print(char const *msg = nullptr) {
    if (msg != nullptr) {
      std::cout << msg;
    }
    for (int c = 0; c < C; ++c) {
      for (int ky = 0; ky < Ky; ++ky) {
        for (int kx = 0; kx < Kx; ++kx) {
          std::cout << static_cast<int64_t>(at(c, ky, kx)) << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
  }

  void init_random(int64_t min = 0, int64_t max = 8) {
    for (int c = 0; c < C; ++c) {
      for (int ky = 0; ky < Ky; ++ky) {
        for (int kx = 0; kx < Kx; ++kx) {
          at(c, ky, kx) = (rand() % (max - min)) + min;
        }
      }
    }
  }

  static int size(int C, int Ky, int Kx) {
    int num_subv = C / Cs;
    int subv_wgt_size =
        round_up_to_multiple(Cs * Ky * Kx * sizeof(Tw), subv_align_bytes);
    int subv_size =
        round_up_to_multiple(subv_wgt_size + subv_qdq_size, subv_align_bytes);
    return num_subv * subv_size;
  }
};

// Computes the Ci dimension of a folded weight matrix
inline int fold_channel_in_dim(int Ci, int fold_factor, int Ci_gran) {
  int Ci_p = ((Ci + Ci_gran - 1) / Ci_gran) * Ci_gran;
  int Ci_f = Ci_p * fold_factor;
  return Ci_f;
}

// Computes the Kx dimension of a folded weight matrix
inline int fold_kernel_x_dim(int Kx, int fold_factor) {
  int Kx_f = (Kx + fold_factor - 1) / fold_factor;
  return Kx_f;
}

inline int fold_spatial_x_dim(int Xo, int Kx, int Sx, int fold_factor,
                              int Xi_gran) {
  int Kx_f = fold_kernel_x_dim(Kx, fold_factor);
  int Sx_f = Sx / fold_factor;
  int x = ((Xo - 1) * Sx_f) + Kx_f;
  return round_up_to_multiple(x, Xi_gran);
}

inline int fold_spatial_y_dim(int Yi, int pad) {
  int Yi_f = Yi + 2 * pad;
  return Yi_f;
}

inline int fold_stride_x_dim(int Sx, int fold_factor) {
  assert(Sx % fold_factor == 0);
  int Sx_f = Sx / fold_factor;
  return Sx_f;
}

template <typename Tw, typename T>
void format_dwc_wgt(Tw W, T *wgt_data, int64_t *qdq, int C1, int C2, int Stdm,
                    int Sout, int Zp) {
  for (int c = 0; c < W.C; ++c) {
    for (int y = 0; y < W.Ky; ++y) {
      for (int x = 0; x < W.Kx; ++x) {
        W.at(c, y, x) = wgt_data[c * W.Ky * W.Kx + y * W.Kx + x];
      }
    }
  }

  for (int c = 0; c < W.C; ++c) {
    W.set_qdq_c0(c, qdq[c]);
  }
  W.set_qdq_c1(C1);
  W.set_qdq_c2(C2);
  W.set_shift_tdm(Stdm);
  W.set_shift_res(Sout);
  W.set_zp_wgt(Zp);
}

template <typename Tw, typename T>
void format_conv_wgt(Tw W, T *wgt_data, int CO, int CI, int64_t *qdq, int C1,
                     int C2, int Stdm, int Sout, int Zp) {
  for (int o = 0; o < CO; ++o) {
    for (int i = 0; i < CI; ++i) {
      for (int y = 0; y < W.Ky; ++y) {
        for (int x = 0; x < W.Kx; ++x) {
          W.at(o, i, y, x) =
              wgt_data[o * CI * W.Ky * W.Kx + i * W.Ky * W.Kx + y * W.Kx + x];
        }
      }
    }
    for (int i = CI; i < W.Ci; ++i) {
      for (int y = 0; y < W.Ky; ++y) {
        for (int x = 0; x < W.Kx; ++x) {
          W.at(o, i, y, x) = (T)Zp;
        }
      }
    }
  }

  for (int c = 0; c < CO; ++c) {
    W.set_qdq_c0(c, qdq[c]);
  }
  W.set_qdq_c1(C1);
  W.set_qdq_c2(C2);
  W.set_shift_tdm(Stdm);
  W.set_shift_res(Sout);
  W.set_zp_wgt(Zp);
}

// This function will fold pixels from Kx into the Cin dimension.
// The final filter width will be ceil(Kx / fold_factor, 1) and the
// final filter depth will be ceil(Ci, Ci_gran) * fold_factor. Each fold
// will be appended to the end of the input channel dimension of the filter.
//
// The input data is assumed to be formatted as Co Ci Ky Kx where
// the innermost traversal is read from right-to-left.
//
// Proceeding down the input channel dimension, we will first traverse
// all pixels from the first fold, then the second fold, ..., etc.
// Any extra trailing pixels are padded with wgt_zp. Padding will
// be inserted at the end of each fold to round each fold up to a
// multiple of Ci_gran.
//
// The filter Kx columns for each fold are interleaved so that
// the first column belongs to the first fold, second column
// belongs to the second fold, etc. then this wraps on the
// number of folds.
template <typename T, int Cos, int Cis>
void fold_conv_wgt(T *wgt_data, T wgt_zp, int Co, int Ci, int Ky, int Kx,
                   int fold_factor, int Ci_gran,
                   ConvWgtTensor<T, Cos, Cis> wgt_fold) {
  int Ci_p = ((Ci + Ci_gran - 1) / Ci_gran) * Ci_gran;
  int Ci_f = fold_channel_in_dim(Ci, fold_factor, Ci_gran);
  int Kx_f = fold_kernel_x_dim(Kx, fold_factor);

  assert(wgt_fold.Co == Co);
  assert(wgt_fold.Ci == Ci_f);
  assert(wgt_fold.Ky == Ky);
  assert(wgt_fold.Kx == Kx_f);

  for (int o = 0; o < Co; ++o) {
    for (int f = 0; f < fold_factor; ++f) {
      for (int i = 0; i < Ci_p; ++i) {
        for (int y = 0; y < Ky; ++y) {
          for (int x = 0; x < Kx_f; ++x) {
            // NOTE: Here src_i or src_x may be out of bounds if we attempt
            // to fold more pixels than exist in the weight matrix. This is
            // when the zero-point padding will occur.
            int dst_i = i + (f * Ci_p);
            int src_i = i;
            int dst_x = x;
            int src_x = (x * fold_factor) + f;
            int src_idx =
                (o * Ci * Ky * Kx) + (src_i * Ky * Kx) + (y * Kx) + (src_x);
            T val = ((src_i < Ci) && (src_x < Kx)) ? wgt_data[src_idx] : wgt_zp;
            wgt_fold.at(o, dst_i, y, dst_x) = val;
          }
        }
      }
    }
  }
}

// This function will fold pixels from X dimension into
// the C dimension, and round up the X/C dimensions according
// to the provided granularities. The same padding is applied
// in all dimensions, with additional trailing padding in the X and C
// dimensions to fill gaps where dimensions are rounded up to
// meet the granularity requirement.
//
// The input data is assumed to be formatted as Ci Yi Xi where
// the innermost traversal is read from right-to-left.
template <typename T>
void fold_conv_ifm(T const *ifm_data, T ifm_zp, int Ci, int Yi, int Xi, int Xo,
                   int Kx, int Sx, int pad, int fold_factor, int Ci_gran,
                   int Xi_gran, ActTensor<T> ifm_fold) {
  int Ci_fold = fold_channel_in_dim(Ci, fold_factor, Ci_gran);
  int Yi_fold = fold_spatial_y_dim(Yi, pad);
  int Xi_fold = fold_spatial_x_dim(Xo, Kx, Sx, fold_factor, Xi_gran);
  assert(ifm_fold.C == Ci_fold);
  assert(ifm_fold.Y == Yi_fold);
  assert(ifm_fold.X == Xi_fold);

  // NOTE: Since ActTensor is YXC ordered, folding pixels from the
  // X dimension into the C dimension requires no additional formatting.
  // We just need to insert the padding values in the YXC dimensions.

  int Ci_pad = Ci_fold / fold_factor;
  int Yi_pad = Yi_fold;
  int Xi_pad = Xi_fold * fold_factor;
  ActTensor<T> ifm_reshape(Ci_pad, Yi_pad, Xi_pad, ifm_fold.data);
  for (int c = 0; c < Ci_pad; ++c) {
    for (int y = 0; y < Yi_pad; ++y) {
      for (int x = 0; x < Xi_pad; ++x) {
        if (c < Ci && pad <= y && y < Yi + pad && pad <= x && x < Xi + pad) {
          int idx = (c * Yi * Xi) + ((y - pad) * Xi) + ((x - pad));
          assert(idx < Ci * Yi * Xi);
          ifm_reshape.at(c, y, x) = ifm_data[idx];
        } else {
          ifm_reshape.at(c, y, x) = ifm_zp;
        }
      }
    }
  }
}

template <typename T>
void format_conv_ifm(T const *ifm_data, T ifm_zp, int Ci,
                     ActTensor<T> ifm_fold) {
  memset((void *)ifm_fold.data, ifm_zp,
         (ifm_fold.C * ifm_fold.X * ifm_fold.Y * sizeof(T)));
  for (int y = 0; y < ifm_fold.Y; y++) {
    for (int x = 0; x < ifm_fold.X; x++) {
      for (int c = 0; c < Ci; c++) {
        ifm_fold.at(c, y, x) =
            ifm_data[c * ifm_fold.X * ifm_fold.Y + y * ifm_fold.X + x];
        // ifm_fold.at(c, y, x) = ifm_data[y * ifm_fold.X * Ci + x * Ci + c];
      }
    }
  }
}

template <typename T>
void format_conv_ofm(T *ofm_data, int Co, ActTensor<T> Out) {
  for (int y = 0; y < Out.Y; y++) {
    for (int x = 0; x < Out.X; x++) {
      for (int c = 0; c < Co; c++) {
        ofm_data[c * Out.X * Out.Y + y * Out.X + x] = Out.at(c, y, x);
      }
    }
  }
}

template <typename T> T srs(int64_t acc, int shift) {
  acc = acc >> shift;
  T val = 0;
  if (std::is_same<T, int8_t>::value) {
    val = (acc > INT8_MAX) ? INT8_MAX : (acc < INT8_MIN) ? INT8_MIN : acc;
  } else if (std::is_same<T, uint8_t>::value) {
    val = (acc > UINT8_MAX) ? UINT8_MAX : (acc < 0) ? 0 : acc;
  } else if (std::is_same<T, int16_t>::value) {
    val = (acc > INT16_MAX) ? INT16_MAX : (acc < INT16_MIN) ? INT16_MIN : acc;
  } else if (std::is_same<T, uint16_t>::value) {
    val = (acc > UINT16_MAX) ? UINT16_MAX : (acc < 0) ? 0 : acc;
  } else if (std::is_same<T, int32_t>::value) {
    val = (acc > INT32_MAX) ? INT32_MAX : (acc < INT32_MIN) ? INT32_MIN : acc;
  } else if (std::is_same<T, uint32_t>::value) {
    val = (acc > UINT32_MAX) ? UINT32_MAX : (acc < 0) ? 0 : acc;
  } else {
    val = acc;
  }
  return val;
}

template <typename Tin, typename Tw, typename To, typename din, typename dw>
void cpu_conv_2d(Tin ifm, Tw wgt, To ofm, int Sy, int Sx, int P, int shift) {
  for (int co = 0; co < ofm.C; ++co) {
    for (int yi = -P, yo = 0; yo < ofm.Y; yi += Sy, ++yo) {
      for (int xi = -P, xo = 0; xo < ofm.X; xi += Sx, ++xo) {
        int64_t acc = 0;
        for (int ci = 0; ci < ifm.C; ++ci) {
          for (int ky = 0; ky < wgt.Ky; ++ky) {
            for (int kx = 0; kx < wgt.Kx; ++kx) {
              int y = yi + ky;
              int x = xi + kx;
              din a = (0 <= y && y < ifm.Y && 0 <= x && x < ifm.X)
                          ? ifm.at(ci, y, x)
                          : 0;
              dw w = wgt.at(co, ci, ky, kx);
              acc += a * w;
            }
          }
        }
        ofm.at(co, yo, xo) = srs<din>(acc, shift);
      }
    }
  }
}

template <typename Ta, typename Tw, typename To, typename din, typename dw>
void cpu_conv_dw(Ta ifm, Tw wgt, To ofm, int Sy, int Sx, int P, int shift) {
  assert(ifm.C == ofm.C);
  for (int c = 0; c < ifm.C; ++c) {
    for (int yi = -P, yo = 0; yo < ofm.Y; yi += Sy, ++yo) {
      for (int xi = -P, xo = 0; xo < ofm.X; xi += Sx, ++xo) {
        int64_t acc = 0;
        for (int ky = 0; ky < wgt.Ky; ++ky) {
          for (int kx = 0; kx < wgt.Kx; ++kx) {
            int y = yi + ky;
            int x = xi + kx;
            din a = (0 <= y && y < ifm.Y && 0 <= x && x < ifm.X)
                        ? ifm.at(c, y, x)
                        : 0;
            dw w = wgt.at(c, ky, kx);
            acc += a * w;
          }
        }
        ofm.at(c, yo, xo) = srs<din>(acc, shift);
      }
    }
  }
}

template <typename T>
int check_result(T expected, T received, int CO_valid, int epsilon = 0) {
  assert(expected.C == received.C);
  assert(expected.Y == received.Y);
  assert(expected.X == received.X);

  int err_count = 0;
  int max_diff = 0;
  float L2_norm = 0;
  for (int c = 0; c < CO_valid; ++c) {
    for (int y = 0; y < expected.Y; ++y) {
      for (int x = 0; x < expected.X; ++x) {
        int32_t diff = std::abs(expected.at(c, y, x) - received.at(c, y, x));
        L2_norm += ((float)diff * (float)diff);
        if (diff > max_diff)
          max_diff = diff;
        if (diff > epsilon) {
          err_count += 1;
          std::cout << "ERROR: [" << c << ", " << y << ", " << x << "]: "
                    << "Expected: " << expected.at(c, y, x) << ", "
                    << "Received: " << received.at(c, y, x) << ", "
                    << "Diff: " << int(diff) << "\n";
        }
      }
    }
  }

  L2_norm = std::sqrt(L2_norm);
  std::cout << "max_diff is " << max_diff << std::endl;
  std::cout << "L2_norm is " << L2_norm << std::endl;

  return err_count;
}

} // namespace iconv_matrix
