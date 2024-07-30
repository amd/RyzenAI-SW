#include "./include/bmm_torch.hpp"
#include "./include/elemw_add_torch.hpp"
#include "./include/elemw_mul_torch.hpp"
#include "./include/gemm_torch.hpp"
#include "./include/linear.hpp"
#include "./include/mha.hpp"
#include "./include/mha_npu_torch.hpp"
#include "./include/mlp_npu_torch.hpp"
#include "./include/qlinear.hpp"
#include "./include/rmsnorm_torch.hpp"
#include "./include/rope_torch.hpp"
#include "./include/scalar_mult.hpp"
#include "./include/silu_torch.hpp"
#include "./include/softmax_torch.hpp"
#include "./include/torch_linear.hpp"

PYBIND11_MODULE(_ryzenai_torch_cpp, m) {
  py::class_<cpu::qlinear>(m, "cpu_qlinear")
      .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor>())
      .def("qmmul", &cpu::qlinear::qmmul, "CPU QLinear forward");

  py::class_<cpu::linear>(m, "cpu_linear")
      .def(py::init<>())
      .def("mmul", &cpu::linear::mmul, "CPU Linear forward");

  py::class_<cpu::mha>(m, "cpu_mha")
      .def(py::init<>())
      .def("mha_tensorized", &cpu::mha::mha_tensorized, "CPU MHA tensorized")
      .def("mha_multithread", &cpu::mha::mha_multithread, "CPU MHA multithread")
      .def("mha_flat", &cpu::mha::mha_flat,
           "CPU MHA flat attn, less memory, depthwise fusion")
      .def("mha_flat_multithread", &cpu::mha::mha_flat_multithread,
           "CPU MHA flat attn, less memory, depthwise fusion & multithread")
      .def("mha_top", &cpu::mha::mha_top, "CPU MHA top");

  py::class_<aie::scalar_mult>(m, "aie_scalar_mult")
      .def(py::init<size_t>())
      .def("execute", &aie::scalar_mult::execute, "AIE int32_scalar execute");

  py::class_<aie::torch_linear>(m, "aie_linear_bf16")
      .def(py::init<const std::tuple<int, int> &,
                    const std::tuple<int, int> &>())
      .def("initialize_weights", &aie::torch_linear::initialize_weights,
           "Initialize weights for linear layer")
      .def("execute", &aie::torch_linear::execute, "AIE bfloat16 matmul");

  py::class_<aie::elemw_mul_torch>(m, "aie_elemw_mul_torch")
      .def(py::init<size_t>())
      .def("execute", &aie::elemw_mul_torch::execute, "AIE bfloat16 execute");

  py::class_<aie::silu_torch>(m, "aie_silu_torch")
      .def(py::init<size_t>())
      .def("execute", &aie::silu_torch::execute, "AIE bfloat16 execute");

  py::class_<aie::bmm_torch>(m, "aie_bmm_torch")
      .def(py::init<bool>())
      .def("execute", &aie::bmm_torch::execute, "AIE bfloat16 execute");

  py::class_<aie::elemw_add_torch>(m, "aie_elemw_add_torch")
      .def(py::init<>())
      .def("execute", &aie::elemw_add_torch::execute, "AIE bfloat16 execute");

  py::class_<aie::softmax_torch>(m, "aie_softmax_torch")
      .def(py::init<>())
      .def("execute", &aie::softmax_torch::execute, "AIE bfloat16 execute");

  py::class_<aie::mha_npu_torch>(m, "aie_mha_npu_torch")
      .def(py::init<>())
      .def("execute", &aie::mha_npu_torch::execute, "AIE bfloat16 execute");

  py::class_<aie::mlp_npu_torch>(m, "aie_mlp_npu_torch")
      .def(py::init<>())
      .def("initialize_weights", &aie::mlp_npu_torch::initialize_weights,
           "Initialize weights for mlp")
      .def("execute", &aie::mlp_npu_torch::execute, "AIE bfloat16 mlp");

  py::class_<aie::rope_torch>(m, "aie_rope_torch")
      .def(py::init<>())
      .def("execute", &aie::rope_torch::execute, "AIE bfloat16 execute");

  py::class_<aie::rmsnorm_torch>(m, "aie_rmsnorm_torch")
      .def(py::init<>())
      .def("execute", &aie::rmsnorm_torch::execute, "AIE bfloat16 execute");

  py::class_<aie::gemm_torch>(m, "aie_gemm_torch")
      .def(py::init<int, int, int, bool>())
      .def("initialize_params", &aie::gemm_torch::initialize_params,
           "Set weights, scales, bias, zeros for this layer")
      .def("execute", &aie::gemm_torch::execute, "AIE bfloat16 execute");
}
