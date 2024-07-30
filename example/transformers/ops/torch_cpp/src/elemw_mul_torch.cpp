#include "../include/elemw_mul_torch.hpp"

#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

#include "utils.h"

// AIE Driver headers
#include "xaiengine.h"

// Headers to create Txn binary
#include "op_buf.hpp"
#include "op_types.h"

// dpu kernel metadata
#include "dpu_kernel_metadata.hpp"

aie::elemw_mul_torch::elemw_mul_torch(size_t size) {}
aie::elemw_mul_torch::~elemw_mul_torch() {}

template <typename InOutT = uint16_t>
void run_elem_mul(InOutT *aInput, InOutT *bInput, InOutT *aie_out, int M, int N,
                  bool debug = false, const std::string &a_dtype = "bfloat16",
                  const std::string &b_dtype = "bfloat16",
                  const std::string &c_dtype = "bfloat16") {

  assert(a_dtype == "bfloat16" && "Currently only supporting bfloat16, "
                                  "bfloat16 -> eltwise_mul -> bfloat16");
  assert(a_dtype == b_dtype &&
         "Currently only supporting homogeneous input and output data types");
  assert(a_dtype == c_dtype &&
         "Currently only supporting homogeneous input and output data types");
  size_t Ms = static_cast<size_t>(M);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ns};
  std::vector<size_t> b_shape = {Ms, Ns};
  std::vector<size_t> aie_out_shape = {Ms, Ns};
  ryzenai::elw_mul mulKernel =
      ryzenai::elw_mul<InOutT, InOutT, InOutT>(a_dtype, true);
  mulKernel.debug(debug);
  std::vector<Tensor> input_Tensors = {{aInput, a_shape, a_dtype},
                                       {bInput, b_shape, b_dtype}};
  std::vector<Tensor> output_Tensor = {{aie_out, aie_out_shape, c_dtype}};
  mulKernel.execute(input_Tensors, output_Tensor);
}

torch::Tensor aie::elemw_mul_torch::execute(torch::Tensor x, torch::Tensor y) {
  int K = 1;
  int M = 1;
  int N = 11008;
  torch::Tensor z;
  if (x.dim() == 2) {
    M = x.sizes()[0];
    N = x.sizes()[1];
    z = torch::empty({M, N}).to(torch::kBFloat16);
  }
  if (x.dim() == 3) {
    K = x.sizes()[0];
    M = x.sizes()[1];
    N = x.sizes()[2];
    z = torch::empty({K, M, N}).to(torch::kBFloat16);
  }
  auto xCasted = static_cast<uint16_t *>(x.data_ptr());
  auto yCasted = static_cast<uint16_t *>(y.data_ptr());
  auto zCasted = static_cast<uint16_t *>(z.data_ptr());

  if (std::string((Utils::get_env_var("DEVICE"))) == "stx") {
    run_elem_mul<uint16_t>(xCasted, yCasted, zCasted, K * M, N, false,
                           "bfloat16", "bfloat16", "bfloat16");
  }
  return z;
}
