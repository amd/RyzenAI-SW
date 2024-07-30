#pragma once

#include <cstdint>

namespace ryzenai {
namespace xcom {

struct ddr_buffer_info_s {
  std::int64_t ifm_addr;
  std::int64_t ifm_size;
  std::int64_t param_addr;
  std::int64_t param_size;
  std::int64_t ofm_addr;
  std::int64_t ofm_size;
  std::int64_t inter_addr;
  std::int64_t inter_size;
  std::int64_t mc_code_addr;
  std::int64_t mc_code_size;
  std::int64_t pad_control_packet;
};

static_assert(sizeof(ddr_buffer_info_s) == 11 * sizeof(std::int64_t));

} // namespace xcom
} // namespace ryzenai
