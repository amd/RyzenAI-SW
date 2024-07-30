#pragma once

#include <cstdint>
#include <vector>

namespace ryzenai {
std::vector<uint8_t>
prepend_mtile_const_pad_txn(const std::vector<uint8_t> &base_txn,
                            const uint32_t pad_value, uint8_t num_channels,
                            uint8_t num_cols);

} // namespace ryzenai
