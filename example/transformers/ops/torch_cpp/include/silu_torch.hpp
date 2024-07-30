#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#if __has_include(<ryzenai/dynamic_dispatch/ops/silu/silu.hpp>)
#include <ryzenai/dynamic_dispatch/ops/silu/silu.hpp>
#else
#include <ops/silu/silu.hpp>
#endif

namespace aie {
class silu_torch {
  ryzenai::silu<uint16_t, uint16_t> siluKernel =
      ryzenai::silu<uint16_t, uint16_t>("bfloat16", true);

public:
  std::string xclbinFileName;
  silu_torch(size_t size);
  ~silu_torch();
  torch::Tensor execute(torch::Tensor x);
};
} // namespace aie
