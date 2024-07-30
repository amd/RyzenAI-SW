#pragma once

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <op_fuser/fuse_types.hpp>
#include <utils/instruction_registry.hpp>
#include <utils/tfuncs.hpp>
#include <xrt_context/xrt_context.hpp>

#include <xrt/xrt_bo.h>

struct Tensor {
  void *data{nullptr};
  std::vector<size_t> shape;
  std::string dtype;
};

struct OpArgMap {
  enum OpArgType {
    INPUT,
    OUTPUT,
    SCRATCH_PAD,
    CONST_INPUT,
    CONST_KERNEL_PARAM_INPUT,
  };
  OpArgType arg_type;
  size_t xrt_arg_idx;
  size_t onnx_arg_idx;
  size_t offset;
  size_t size; // in bytes
  size_t padding_offset = 0;
};

class OpInterface {
public:
  OpInterface() {}
  OpInterface(const std::vector<std::string> in_dtypes,
              const std::vector<std::string> out_dtypes){};
  virtual ~OpInterface() = default;

  virtual void
  initialize_inputs(const std::vector<Tensor> &inputs,
                    const std::map<std::string, std::any> &attr = {}) {}

  virtual void
  initialize_const_params(void *dest, const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {}) {}

  virtual void
  initialize_const_params(const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {}) {}

  virtual const std::vector<uint8_t>
  get_transaction_bin(std::vector<Tensor> &input, std::vector<Tensor> &output,
                      const std::map<std::string, std::any> &attr = {}) {
    return {};
  }

  virtual const std::vector<uint8_t>
  get_super_kernel_params(std::vector<Tensor> &input,
                          std::vector<Tensor> &output,
                          const std::map<std::string, std::any> &attr = {}) {
    return {};
  };

  virtual std::vector<OpArgMap>
  get_buffer_reqs(std::vector<Tensor> &input, std::vector<Tensor> &output,
                  const std::map<std::string, std::any> &attr = {}) {
    return {};
  }

  virtual const std::map<std::string, std::any> &get_attr() const {
    static const std::map<std::string, std::any> empty_map;
    return empty_map;
  }

  virtual void execute(std::vector<Tensor> &input,
                       std::vector<Tensor> &output) {}

  virtual void execute(std::vector<xrt::bo> &input,
                       std::vector<xrt::bo> &output) {}

  static void set_dod_base_dir(const std::string &dir);

  static std::string get_dod_base_dir();

protected:
  std::shared_ptr<ryzenai::dynamic_dispatch::xrt_context> xrt_ctx_;
  static ryzenai::dynamic_dispatch::instruction_registry instr_reg_;
  static std::string dod_base_dir;
};

std::string convert_argtype_to_string(OpArgMap::OpArgType arg_type);
std::string cvt_to_string(const OpArgMap &arg);
std::string cvt_to_string(const std::vector<OpArgMap> &argmap);

// Utility to invoke OpInterface methods with verbose error checks.
template <typename Func, typename... Args>
static auto dd_invoke_op_method(const std::string &func_name,
                                const char *srcfile, size_t line_no,
                                const OpsFusion::Metadata::OpInfo &op_info,
                                Func &&func, OpInterface *op, Args &&...args) {
  OpsFusion::LifeTracer lt(
      OpsFusion::dod_format("Invoking {}() for op:{}, op_type:{}", func_name,
                            op_info.name, op_info.type));

  try {
    return func(op, std::forward<Args>(args)...);
  } catch (std::exception &e) {
    throw std::runtime_error(OpsFusion::dod_format(
        "[{}:{}] Invoking {}() failed !!\nDetails:\n  Op "
        "Name: {}\n  Op Type: {}\n  Error: {}",
        srcfile, line_no, func_name, op_info.name, op_info.type, e.what()));
  } catch (...) {
    throw std::runtime_error(OpsFusion::dod_format(
        "[{}:{}] Invoking {}() failed !!\nDetails:\n  Op "
        "Name: {}\n  Op Type: {}\n  Error: Unknown Exception",
        srcfile, line_no, func_name, op_info.name, op_info.type));
  }
}

// Invoke OpInterface method with verbose error check
#define DD_INVOKE_OPMETHOD(method_name, op_object, op_info, ...)               \
  dd_invoke_op_method("OpInterface::" #method_name, __FILE__, __LINE__,        \
                      op_info, std::mem_fn(&OpInterface::method_name),         \
                      op_object, __VA_ARGS__)

// Invoke OpInterface method with verbose error check
#define DD_INVOKE_OVERLOADED_OPMETHOD(method_name, signature, op_object,       \
                                      op_info, ...)                            \
  dd_invoke_op_method("OpInterface::" #method_name, __FILE__, __LINE__,        \
                      op_info,                                                 \
                      std::mem_fn<signature>(&OpInterface::method_name),       \
                      op_object, __VA_ARGS__)
