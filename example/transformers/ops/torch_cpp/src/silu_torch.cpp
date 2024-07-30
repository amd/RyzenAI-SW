#include "../include/silu_torch.hpp"

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

aie::silu_torch::silu_torch(size_t size) { siluKernel.debug(false); }
aie::silu_torch::~silu_torch() {}

template <typename InOutT = uint16_t>
void run_silu(InOutT *aInput, InOutT *aie_out, int M, int N, bool debug = false,
              const std::string &a_dtype = "bfloat16",
              const std::string &c_dtype = "bfloat16") {
  assert(a_dtype == "bfloat16" &&
         "Currently only supporting bfloat16 -> silu -> bfloat16");
  assert(a_dtype == c_dtype &&
         "Currently only supporting homogeneous input and output data types");
  size_t Ms = static_cast<size_t>(M);
  size_t Ns = static_cast<size_t>(N);
  std::vector<size_t> a_shape = {Ms, Ns};
  std::vector<size_t> aie_out_shape = {Ms, Ns};
  ryzenai::silu siluKernel = ryzenai::silu<InOutT, InOutT>(a_dtype, true);
  siluKernel.debug(debug);
  std::vector<Tensor> input_Tensor = {{aInput, a_shape, a_dtype}};
  std::vector<Tensor> output_Tensor = {{aie_out, aie_out_shape, c_dtype}};
  siluKernel.execute(input_Tensor, output_Tensor);
}

torch::Tensor aie::silu_torch::execute(torch::Tensor x) {

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

  auto zCasted = static_cast<uint16_t *>(z.data_ptr());
  if (std::string((Utils::get_env_var("DEVICE"))) == "stx") {
    run_silu<uint16_t>(xCasted, zCasted, K * M, N, false, "bfloat16",
                       "bfloat16");
  }
  return z;
}
