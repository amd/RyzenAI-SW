#include "matmul.h"

#include <memory>
#include <mutex>

#include "dynamic_quantlinear.hpp"

namespace {

using Gemm = ::ryzenai::dynamicquantlinear<float, float>;

std::mutex gMatmulF32ImplMutex;

// Pure CPU implementation only for the dev/debugging purpose.
void matmul_f32_cpu_impl(const float *input_a, const float *input_b,
                         float *output, size_t row_dim, size_t inner_dim,
                         size_t col_dim) {
  // - shape(input_a) = (row_dim, inner_dim)
  // - shape(input_b) = (inner_dim, col_dim)
  // - shape(output) = (row_dim, col_dim)
  for (int64_t row = 0; row < row_dim; ++row) {
    for (int64_t col = 0; col < col_dim; ++col) {
      float *output_elem = output + (row * col_dim + col);
      for (int64_t inner = 0; inner < inner_dim; ++inner) {
        // input_a[row][inner] * input_b[inner][col]
        *output_elem +=
            input_a[row * inner_dim + inner] * input_b[inner * col_dim + col];
      }
    }
  }
}

// Convenience container for all GEMM-related input arguments and the output
// buffer pointer.
struct Params {
  Params(const float *input_a, const float *input_b, float *output,
         size_t row_dim, size_t inner_dim, size_t col_dim)
      : input_a(input_a), input_b(input_b), output(output), row_dim(row_dim),
        inner_dim(inner_dim), col_dim(col_dim) {}

  const float *input_a = nullptr;
  const float *input_b = nullptr;
  float *output = nullptr;
  size_t row_dim = 0, inner_dim = 0, col_dim = 0;
};

// Container for the gemm instances and the corresponding dimensions. The fields
// other than the shared_ptr are only for verification.
struct GemmContainer {
  GemmContainer(std::unique_ptr<Gemm> gemm, int k, int n, float first_elem)
      : gemm(std::move(gemm)), k(k), n(n), first_elem(first_elem) {}

  std::shared_ptr<Gemm> gemm;
  int k = 0, n = 0;
  float first_elem = 0;
};

using GemmTable = std::unordered_map<const float *, GemmContainer>;

// If found, returns the existing dynamicquantlinear<> instance in the static
// container indexed by Params::input_b. Otherwise, creates a new one and
// returns it.
// NOTE This function is NOT thread-safe.
GemmContainer FindOrMakeGemm(const Params &p) {
  // The static container of dynamicquantlinear<> instances created so far.
  static GemmTable gemm_table;

  static const std::string aie_kernel_dll =
      Utils::get_dll_path() + "libGemmQnnAie_4x2048_2048x2048.dll";
  auto gemm_iter = gemm_table.find(p.input_b);
  if (gemm_iter == gemm_table.end()) {
    const float scale_weights =
        (Utils::abs_max<float>(p.input_b, p.inner_dim * p.col_dim)) / 127;
    const float requantize_out_scale = 1.0f;
    auto gemm = std::make_unique<Gemm>(
        aie_kernel_dll, /*x_shape=*/std::tuple<int, int>{4, 2048},
        /*y_shape=*/std::tuple<int, int>{2048, 2048}, scale_weights,
        requantize_out_scale,
        /*nworkers=*/1);
    gemm_iter =
        gemm_table
            .emplace(p.input_b, GemmContainer(std::move(gemm), p.inner_dim,
                                              p.col_dim, p.input_b[0]))
            .first;
    gemm_iter->second.gemm->initialize_weights_data(
        const_cast<float *>(p.input_b),
        /*wt_shape=*/std::tuple<int, int>{p.inner_dim, p.col_dim});
  } else {
    // Verify that the dimensions for the stored gemm matches the incoming
    // dimensions.
    const GemmContainer &prev_gemm = gemm_iter->second;
    if (prev_gemm.k != p.inner_dim || prev_gemm.n != p.col_dim ||
        prev_gemm.first_elem != p.input_b[0]) {
      throw std::runtime_error(
          "Cached gemm dimensions do not match the requested ones.");
    }
  }
  return gemm_iter->second;
}

#ifdef IREE_PLUGINS_MATMUL_DEBUG_LOG
void CompareOutput(const Params &p) {
  std::vector<float> expected_output(p.row_dim * p.col_dim, 0);
  matmul_f32_cpu_impl(p.input_a, p.input_b, expected_output.data(), p.row_dim,
                      p.inner_dim, p.col_dim);

  printf("Expected: %s\n",
         PrintMatrix(expected_output.data(),
                     std::tuple<int32_t, int32_t>{p.row_dim, p.col_dim},
                     /*print_limit=*/5)
             .c_str());
  printf("Actual: %s\n",
         PrintMatrix(p.output,
                     std::tuple<int32_t, int32_t>{p.row_dim, p.col_dim},
                     /*print_limit=*/5)
             .c_str());
  printf("m = %llu; k = %llu; n = %llu\n", p.row_dim, p.inner_dim, p.col_dim);
  const std::vector<float> input_a(p.input_a,
                                   p.input_a + (p.row_dim * p.inner_dim));
  const std::vector<float> input_b(p.input_b,
                                   p.input_b + (p.inner_dim * p.col_dim));
  const std::vector<float> output(p.output, p.output + (p.row_dim * p.col_dim));
  printf("input_a: %s\n", ToString(input_a).c_str());
  printf("input_b: %s\n", ToString(input_b).c_str());
  printf("expected output : %s\n", ToString(expected_output).c_str());
  printf("actual output : %s\n", ToString(output).c_str());
}
#endif

} // namespace

void matmul_f32_impl(const float *input_a, const float *input_b, float *output,
                     size_t row_dim, size_t inner_dim, size_t col_dim) {
#ifdef IREE_PLUGINS_EMULATE_MATMUL
  matmul_f32_cpu_impl(input_a, input_b, output, row_dim, inner_dim, col_dim);
#else
  const Params params(input_a, input_b, output, row_dim, inner_dim, col_dim);
  {
    std::lock_guard<std::mutex> lock(gMatmulF32ImplMutex);
    FindOrMakeGemm(params).gemm->execute_aie(
        const_cast<float *>(input_a),
        /*a_shape=*/std::tuple<int, int>{row_dim, inner_dim}, output);
  }
#ifdef IREE_PLUGINS_MATMUL_DEBUG_LOG
  CompareOutput(params);
#endif
#endif
}
