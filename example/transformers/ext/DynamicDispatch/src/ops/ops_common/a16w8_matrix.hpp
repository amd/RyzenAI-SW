#ifndef A16W8_MATRIX_HPP
#define A16W8_MATRIX_HPP

#include <assert.h>
#include <iostream>
#include <stdlib.h>

static int constexpr Msubv = 32;
static int constexpr Ksubv = 128;
static int constexpr Nsubv = 64;

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
    assert(num_rows % subv_rows == 0);
    assert(num_cols % subv_cols == 0);
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
    assert(num_rows % (subv_rows * aie_rows) == 0);
    // assert(num_cols % (subv_cols * aie_cols) == 0);
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

template <typename T>
static void init_random(T mat, int min = -128, int max = 128) {
  for (int i = 0; i < mat.num_rows; ++i) {
    for (int j = 0; j < mat.num_cols; ++j) {
      mat.at(i, j) = (rand() % (max - min)) + min;
    }
  }
}

template <typename T>
static void print_matrix(T mat, const char *msg = nullptr) {
  if (msg != nullptr) {
    std::cout << msg << "\n";
  }
  for (int i = 0; i < mat.num_rows; ++i) {
    for (int j = 0; j < mat.num_cols; ++j) {
      std::cout << static_cast<int64_t>(mat.at(i, j)) << " ";
    }
    std::cout << "\n";
  }
}

static int32_t srs_to_int32(int64_t x) {
  if (x > INT32_MAX) {
    x = INT32_MAX;
  } else if (x < INT32_MIN) {
    x = INT32_MIN;
  }
  return static_cast<int32_t>(x);
}

static int16_t srs_to_int16(int64_t x) {
  if (x > INT16_MAX) {
    x = INT16_MAX;
  } else if (x < INT16_MIN) {
    x = INT16_MIN;
  }
  return static_cast<int16_t>(x);
}

static float gelu_golden(int16_t in) {
  // float exp_x = std::exp(-in);
  // float sg  = in/(1+exp_x);
  auto inf = static_cast<float>(in);
  float xr2 = inf / (std::sqrt(2));
  float t = std::erf(xr2);
  float g = inf * 0.5 * (1.0 + t);

  return g;
}

template <int Msubv, int Ksubv, int Nsubv>
static void cpu_matmul(ActMatrix<int16_t, Msubv, Ksubv> X,
                       WgtMatrix<int8_t, Ksubv, Nsubv> W,
                       OutMatrix<int16_t, Msubv, Nsubv> Y, bool bias = false,
                       bool gelu = false) {
  for (int r = 0; r < Y.num_rows; ++r) {
    for (int c = 0; c < Y.num_cols; ++c) {
      int64_t acc = 0;
      for (int k = 0; k < X.num_cols; ++k) {
        acc += X.at(r, k) * W.at(k, c);
      }
      if (gelu) {
        Y.at(r, c) = gelu_golden(srs_to_int16(acc));
      } else {
        Y.at(r, c) = srs_to_int16(acc);
      }
    }
  }
}

template <int Msubv, int Ksubv, int Nsubv>
static void cpu_matmul_int32(ActMatrix<int16_t, Msubv, Ksubv> X,
                             WgtMatrix<int8_t, Ksubv, Nsubv> W,
                             OutMatrix<int32_t, Msubv, Nsubv> Y) {
  for (int r = 0; r < Y.num_rows; ++r) {
    for (int c = 0; c < Y.num_cols; ++c) {
      int64_t acc = 0;
      for (int k = 0; k < X.num_cols; ++k) {
        acc += X.at(r, k) * W.at(k, c);
      }
      Y.at(r, c) = srs_to_int32(acc);
    }
  }
}

template <typename T> static int check_result(T cpu_Y, T aie_Y) {
  int fail = 0;
  int err_count = 0;
  for (int r = 0; r < aie_Y.num_rows; ++r) {
    for (int c = 0; c < aie_Y.num_cols; ++c) {
      int32_t diff = cpu_Y.at(r, c) - aie_Y.at(r, c);
      if (0) {
        std::cout << "ERROR: Y[" << r << ", " << c << "]: "
                  << "Expected: " << cpu_Y.at(r, c) << ", "
                  << "Received: " << aie_Y.at(r, c) << "\n";
        fail = 1;
        err_count++;
      }
    }
  }
  return err_count;
}

template <typename T>
int check_add_result(std::vector<T> cpu_Y, std::vector<T> aie_Y,
                     std::tuple<int, int> tensor_shape) {
  auto num_rows = std::get<0>(tensor_shape);
  auto num_cols = std::get<1>(tensor_shape);

  int fail = 0;
  int err_count = 0;
  for (int r = 0; r < num_rows; ++r) {
    for (int c = 0; c < num_cols; ++c) {
      int32_t diff = cpu_Y.at(r * num_cols + c) - aie_Y.at(r * num_cols + c);
      if (diff > 1) {
        // std::cout << "ERROR: Y[" << r << ", " << c << "]: "
        //           << "Expected: " << cpu_Y.at(r*num_cols + c) << ", "
        //           << "Received: " << aie_Y.at(r*num_cols + c) << "\n";
        fail = 1;
        err_count++;
      }
    }
  }
  return err_count;
}

#endif // MATRIX_HPP
