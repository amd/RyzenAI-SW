#pragma once
#include <memory>

#include "op_interface.hpp"
#include <op_fuser/fuse_types.hpp>
#include <stdexcept>

using namespace std::literals::string_literals;

namespace OpsFusion {

class OpBuilder {
public:
  OpBuilder() = default;
  virtual ~OpBuilder() = default;

  // TODO : What is the right info to be passed to the builder ?
  static std::unique_ptr<OpInterface>
  create(const std::string &op_name, const Metadata::OpInfo &op_info,
         const std::map<std::string, Metadata::OffsetInfo> &tensor_map);

  static bool is_supported(const std::string &op_type,
                           const std::vector<std::string> &types,
                           const std::map<std::string, std::any> &attr);
};

} // namespace OpsFusion
