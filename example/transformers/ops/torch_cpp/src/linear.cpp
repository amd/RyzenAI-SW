#include "../include/linear.hpp"

torch::Tensor cpu::linear::mmul(torch::Tensor x, torch::Tensor weights) {
  auto res = torch::mm(x, weights);
  return res;
}
