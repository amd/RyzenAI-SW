#include <filesystem>
#include <iomanip>

#include <ops/op_interface.hpp>
#include <utils/tfuncs.hpp>
#include <utils/utils.hpp>

ryzenai::dynamic_dispatch::instruction_registry OpInterface::instr_reg_;
std::string OpInterface::dod_base_dir{};

void OpInterface::set_dod_base_dir(const std::string &dir) {
  //   DOD_THROW_IF(!dir.empty() && !std::filesystem::exists(dir),
  //                OpsFusion::dod_format("Dir {} doesn't exist",
  //                std::quoted(dir)));

  if (dod_base_dir.empty()) {
    RYZENAI_LOG_TRACE(
        OpsFusion::dod_format("Setting DoD base dir to {}", std::quoted(dir)));
  } else {
    RYZENAI_LOG_TRACE(OpsFusion::dod_format(
        "[WARNING] Overwriting DoD base dir from {} to {}",
        std::quoted(dod_base_dir), std::quoted(dir)));
  }

  dod_base_dir = dir;
}

std::string OpInterface::get_dod_base_dir() {
  bool is_dod_base_dir_set = !dod_base_dir.empty();
  if (is_dod_base_dir_set) {
    return dod_base_dir;
  }

  RYZENAI_LOG_TRACE(
      "DoD base dir is not set. Checking the DOD_ROOT env variable.");
  std::string dod_root_env = Utils::get_env_var("DOD_ROOT");
  if (dod_root_env.empty()) {
    DOD_THROW("DoD base dir is not set. Use OpInterface::set_dod_base_dir(dir) "
              "API or set DOD_ROOT env variable.");
  }

  return dod_root_env;
}

std::string convert_argtype_to_string(OpArgMap::OpArgType arg_type) {

  std::string arg;
  switch (arg_type) {
  case OpArgMap::OpArgType::INPUT:
    arg = "in";
    break;
  case OpArgMap::OpArgType::OUTPUT:
    arg = "out";
    break;
  case OpArgMap::OpArgType::SCRATCH_PAD:
    arg = "scratch";
    break;
  case OpArgMap::OpArgType::CONST_INPUT:
    arg = "const";
    break;
  case OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT:
    arg = "super_instr";
    break;
  default:
    DOD_THROW("Invalide arg_type conversion to string");
    break;
  }

  return arg;
}

std::string cvt_to_string(const OpArgMap &arg) {
  return OpsFusion::dod_format(
      "argtype:{}, xrt_id:{}, onnx_id:{}, offset:{}, size:{}",
      convert_argtype_to_string(arg.arg_type), arg.xrt_arg_idx,
      arg.onnx_arg_idx, arg.offset, arg.size);
}

std::string cvt_to_string(const std::vector<OpArgMap> &argmap) {
  std::ostringstream oss;
  size_t idx = 0;
  for (const auto &arg : argmap) {
    oss << OpsFusion::dod_format("{} - {}", idx, cvt_to_string(arg))
        << std::endl;
    idx++;
  }
  return oss.str();
}
