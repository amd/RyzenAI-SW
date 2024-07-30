#pragma once

#include "ops/op_interface.hpp"
#include "ops/ops_common.hpp"
#include <utils/txn_container.hpp>

namespace ryzenai {
template <typename InT, typename OutT> class maxpool : public OpInterface {
private:
  int zp_;

  /* Sandip TBD : Max Kernel parameters should be defined here and should be
   * checked in code */

  /* actual input matrix */
  int64_t inputShape_[3];
  /* actual output matrix */
  int64_t outputShape_[3];

  //  static instruction_registry instr_reg_;
  static std::once_flag instr_reg_flag_;

  int ifmDtypeSize_;
  /* size for weights dtype*/
  int weightDtypeSize_;
  /* size for output activation dtype*/
  int ofmDtypeSize_;
  /* variables to store profile data */
  int64_t run_aie_time_;

  static std::once_flag logger_flag_;
  uint64_t maxpool_id_;
  static uint64_t maxpool_count;
  /* debug flag */
  bool debug_ = false;

  /*xclbin and mc_code selection variables*/
  std::string ifmDtype_;
  std::string ofmDtype_;

public:
  maxpool(const std::string &a_dtype, const std::string &c_dtype,
          const std::map<std::string, std::any> &attr = {});
  std::vector<OpArgMap>
  get_buffer_reqs(std::vector<Tensor> &input, std::vector<Tensor> &output,
                  const std::map<std::string, std::any> &attr = {}) override;
  void set_params(const std::string &modelName);
};
} // namespace ryzenai
