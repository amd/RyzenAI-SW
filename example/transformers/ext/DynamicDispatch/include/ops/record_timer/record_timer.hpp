#pragma once

#include <ops/op_interface.hpp>

namespace ryzenai {

class record_timer : public OpInterface {
public:
  record_timer();
  const std::vector<uint8_t>
  get_transaction_bin(std::vector<Tensor> &input, std::vector<Tensor> &output,
                      const std::map<std::string, std::any> &attr) override;
};

} // namespace ryzenai
