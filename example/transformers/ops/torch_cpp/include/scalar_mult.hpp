#include <ATen/Functions.h>
#include <torch/extension.h>
#include <torch/torch.h>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace aie {
class scalar_mult {
private:
  xrt::device device_;
  xrt::xclbin xclbin_;
  xrt::kernel kernel_;
  xrt::hw_context context_;
  xrt::bo instr_bo_;
  xrt::bo input_;
  xrt::bo output_;
  void generate_txn(uint64_t src, uint64_t dest, uint32_t size);
  void initialize_device(std::string xcl);

public:
  std::string xclbinFileName;
  scalar_mult(size_t size);
  ~scalar_mult();

  torch::Tensor execute(torch::Tensor x);
};
} // namespace aie
