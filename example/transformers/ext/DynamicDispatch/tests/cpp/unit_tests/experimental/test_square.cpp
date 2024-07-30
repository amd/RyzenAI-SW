#include <gtest/gtest.h>
#include <iostream>

#include <ops/experimental/square.hpp>

#include "test_common.hpp"

#define NUM_ELEMS 32

template <typename InT, typename OutT> int32_t test_square() {

  std::vector<InT> input(NUM_ELEMS, 0);
  std::vector<OutT> output(NUM_ELEMS, 0);
  std::vector<OutT> golden(NUM_ELEMS, 0);

  for (int i = 0; i < NUM_ELEMS; i++) {
    input.at(i) = i;
    golden.at(i) = input.at(i) * input.at(i);
  }

  ryzenai::square square_ = ryzenai::square<InT, OutT>(true);
  std::vector<Tensor> input_tensors;
  input_tensors = {{input.data(), {32}, "int32"}};
  std::vector<Tensor> output_tensors;
  output_tensors = {{output.data(), {32}, "int32"}};
  std::vector<Tensor> const_tensors;
  square_.initialize_const_params(const_tensors);
  square_.execute(input_tensors, output_tensors);

  int32_t err_count = 0;
  for (int i = 0; i < NUM_ELEMS; i++) {
    std::cout << "Golden: " << golden.at(i) << " , Actual: " << output.at(i)
              << std::endl;
    if (golden.at(i) != output.at(i)) {
      err_count++;
    }
  }

  return err_count;
}

TEST(EXPTL_square, square1) {
  int32_t err_count = test_square<int32_t, int32_t>();
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
