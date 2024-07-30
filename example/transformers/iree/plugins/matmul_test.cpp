#include "plugins/matmul.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

using ::testing::FloatNear;
using ::testing::Pointwise;

// Basic case with no scaling.
TEST(MatmulTest, Basic) {
  const int m = 2;
  const int n = 2;
  const int k = 2;

  const std::vector<float> a = {1, 2, 3, 4};
  const std::vector<float> b = {2, 3, 4, 5};
  std::vector<float> c(4, 0);

  matmul_f32_impl(a.data(), b.data(), c.data(), m, n, k);

  EXPECT_THAT(c, Pointwise(FloatNear(/*max_abs_error=*/0.2), {10, 13, 22, 29}));
}

// Weights are small fractions but greater activations produce the same outputs.
TEST(MatmulTest, SmallWeights) {
  const int m = 2;
  const int n = 2;
  const int k = 2;

  const std::vector<float> a = {100, 200, 300, 400};
  const std::vector<float> b = {.02, .03, .04, .05};
  std::vector<float> c(4, 0);

  matmul_f32_impl(a.data(), b.data(), c.data(), m, n, k);

  EXPECT_THAT(c, Pointwise(FloatNear(/*max_abs_error=*/0.2), {10, 13, 22, 29}));
}

// Weights are small and the outputs are also small.
TEST(MatmulTest, SmallWeightsAndSmallOutputs) {
  const int m = 2;
  const int n = 2;
  const int k = 2;

  const std::vector<float> a = {1, 2, 3, 4};
  const std::vector<float> b = {.02, .03, .04, .05};
  std::vector<float> c(4, 0);

  matmul_f32_impl(a.data(), b.data(), c.data(), m, n, k);

  EXPECT_THAT(
      c, Pointwise(FloatNear(/*max_abs_error=*/0.002), {.1, .13, .22, .29}));
}

// Activations are small fractions but greater weights produce the same outputs.
TEST(MatmulTest, SmallActivations) {
  const int m = 2;
  const int n = 2;
  const int k = 2;

  const std::vector<float> a = {.01, .02, .03, .04};
  const std::vector<float> b = {200, 300, 400, 500};
  std::vector<float> c(4, 0);

  matmul_f32_impl(a.data(), b.data(), c.data(), m, n, k);

  EXPECT_THAT(c, Pointwise(FloatNear(/*max_abs_error=*/0.2), {10, 13, 22, 29}));
}

// Activations are small and the outputs are also small.
TEST(MatmulTest, SmallActivationsAndSmallOutputs) {
  const int m = 2;
  const int n = 2;
  const int k = 2;

  const std::vector<float> a = {.01, .02, .03, .04};
  const std::vector<float> b = {2, 3, 4, 5};
  std::vector<float> c(4, 0);

  matmul_f32_impl(a.data(), b.data(), c.data(), m, n, k);

  EXPECT_THAT(
      c, Pointwise(FloatNear(/*max_abs_error=*/0.002), {.1, .13, .22, .29}));
}

TEST(MatmulTest, SmallActivationsAndSmallWeights) {
  const int m = 2;
  const int n = 2;
  const int k = 2;

  const std::vector<float> a = {.01, .02, .03, .04};
  const std::vector<float> b = {.02, .03, .04, .05};
  std::vector<float> c(4, 0);

  matmul_f32_impl(a.data(), b.data(), c.data(), m, n, k);

  EXPECT_THAT(c, Pointwise(FloatNear(/*max_abs_error=*/0.00002),
                           {.001, .0013, .0022, .0029}));
}

} // namespace
