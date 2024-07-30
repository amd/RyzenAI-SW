#pragma once

#include "ops/op_interface.hpp"
#include "ops/ops_common.hpp"
#include <utils/txn_container.hpp>

namespace ryzenai {
template <typename InT, typename OutT> class concateOps : public OpInterface {
private:
  std::map<std::string, std::vector<matrix_shapes>> default_shapes_;
  int64_t graphId_;
  int64_t inChannels_;
  int64_t outChannels_;

  //  static instruction_registry instr_reg_;
  static std::once_flag instr_reg_flag_;
  /* XRT BO for tiled activation matrix */
  xrt::bo ifmBo_;
  /* XRT BO for tiled weight matrix */
  xrt::bo constBo_;
  /* XRT BO for tiled output matrix */
  xrt::bo ofmBo_;
  /* size for input activation dtype*/
  xrt::bo scratchBo_;
  /* size for scratch pad buffer*/

  /* variables to store profile data */
  int64_t run_aie_time_;

  static std::once_flag logger_flag_;
  uint64_t concatenate_id_;
  static uint64_t concatenate_count;
  /* debug flag */
  bool debug_ = false;

  /*xclbin and mc_code selection variables*/
  std::string ifmDtype_;
  std::string weightDtype_;
  std::string ofmDtype_;
  std::string txn_fname_prefix_;
  std::string param_fname_prefix_;

  std::vector<std::unique_ptr<OpInterface>> op_interfaces_;
  /* Add the CreateConvOperator function declaration */
  void CreateConvOperator(const std::map<std::string, std::any> &attrs);
  void CreateMaxpoolOperator(const std::map<std::string, std::any> &attrs);

  void setup_instr_registry();
  std::string get_instr_key(std::string prefix, int64_t graphId,
                            int64_t inChannels, int64_t outChannels);
  void WriteToFile(void *src, uint64_t length);

public:
  concateOps(
      const int graphId, const int inChannels, const int outChannels,
      const std::vector<std::map<std::string, std::any>> &attributes = {});
  void set_params(const std::string &modelName, bool debugFlag);
  std::vector<OpArgMap>
  get_buffer_reqs(std::vector<Tensor> &input, std::vector<Tensor> &output,
                  const std::map<std::string, std::any> &attr = {});
  void
  initialize_const_params(void *dest, const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {});
  void
  initialize_const_params(const std::vector<Tensor> &const_params,
                          const std::map<std::string, std::any> &attr = {});
  void execute(const std::vector<Tensor> &input, std::vector<Tensor> &output);
  const std::vector<uint8_t>
  get_transaction_bin(std::vector<Tensor> &input, std::vector<Tensor> &output,
                      const std::map<std::string, std::any> &attr = {});
};

} // namespace ryzenai
