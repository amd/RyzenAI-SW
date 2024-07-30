#include <nlohmann/json.hpp>
#include <ops/act_act_matmul_qdq/act_act_matmul_qdq.hpp>
#include <ops/bmm/bmm.hpp>
#include <ops/concat/concat.hpp>
#include <ops/concateOps/concateOps.hpp>
#include <ops/conv/conv.hpp>
#include <ops/conv2matmul/conv2matmul.hpp>
#include <ops/elwadd/elwadd.hpp>
#include <ops/elwmul/elwmul.hpp>
#include <ops/elwmul_qdq/elwmul_qdq.hpp>
#include <ops/experimental/cube.hpp>
#include <ops/experimental/square.hpp>
#include <ops/gap/gap.hpp>
#include <ops/gelu/gelu.cpp>
#include <ops/groupnorm/groupnorm.hpp>
#include <ops/iconv/iconv.hpp>
#include <ops/layernorm/layernorm.hpp>
#include <ops/lstm/lstm.hpp>
#include <ops/maskedsoftmax/maskedsoftmax.hpp>
#include <ops/matmul/matmul.hpp>
#include <ops/matmul_a16a16_mladf/matmul_a16a16_mladf.hpp>
#include <ops/matmul_a16w8_mladf/matmul_a16w8_mladf.hpp>
#include <ops/matmulbias/matmulbias.hpp>
#include <ops/matmulgeluadd/matmulgeluadd.hpp>
#include <ops/matvecadd/matvecadd.hpp>
#include <ops/mha/mha.hpp>
#include <ops/mhachannel/mhachannel.hpp>
#include <ops/mhagprb/mhagprb.hpp>
#include <ops/mhapsr/mhapsr.hpp>
#include <ops/mhawindow/mhawindow.hpp>
#include <ops/mladfadd/mladfadd.hpp>
#include <ops/mladfelwadd/mladfelwadd.hpp>
#include <ops/mladfelwmul/mladfelwmul.hpp>
#include <ops/mladfmatmulbias/mladfmatmulbias.hpp>
#include <ops/mladfmharope/mladfmharope.hpp>
#include <ops/mladfrmsnorm/mladfrmsnorm.hpp>
#include <ops/mladfsoftmax/mladfsoftmax.hpp>
#include <ops/nni_resize/nni_resize.hpp>
#include <ops/op_builder.hpp>
#include <ops/pm_load/pm_load.hpp>
#include <ops/record_timer/record_timer.hpp>
#include <ops/silu/silu.hpp>
#include <ops/silu_qdq/silu_qdq.hpp>
#include <ops/slice/slice.hpp>
#include <ops/softmax_qdq/softmax_qdq.hpp>
#include <ops/transpose/transpose.hpp>
#include <ops/xcom/conv/conv.hpp>

using json = nlohmann::json;

#include <utils/utils.hpp>

std::any json_to_any(const json &j);

std::map<std::string, std::any> json_to_map(const json &j) {
  std::map<std::string, std::any> result;

  for (auto it = j.begin(); it != j.end(); ++it) {
    result[it.key()] = json_to_any(it.value());
  }

  return result;
}

std::any json_to_any(const json &j) {
  if (j.is_object()) {
    return json_to_map(j);
  } else if (j.is_array()) {
    std::vector<std::any> array;
    for (const auto &item : j) {
      array.push_back(json_to_any(item));
    }
    return array;
  } else if (j.is_string()) {
    return j.get<std::string>();
  } else if (j.is_boolean()) {
    return j.get<bool>();
  } else if (j.is_number_integer()) {
    return j.get<int>();
  } else if (j.is_number_unsigned()) {
    return j.get<unsigned int>();
  } else if (j.is_number_float()) {
    return j.get<double>();
  } else {
    return {};
  }
}

namespace OpsFusion {

/// @brief Extract datatypes of all arguments of the op from Nodes' attributes.
/// This is valid only for DD-vaip-cpp flow.
static std::vector<std::string>
extract_dtypes_from_attrs(const OpsFusion::Metadata::OpInfo &op_info) {
  const auto &in_arg_dtypes = std::any_cast<const std::vector<std::string> &>(
      MAP_AT(op_info.attr, "in_dtypes"));
  const auto &out_arg_dtypes = std::any_cast<const std::vector<std::string> &>(
      MAP_AT(op_info.attr, "out_dtypes"));

  std::vector<std::string> arg_dtypes;
  arg_dtypes.insert(arg_dtypes.end(), in_arg_dtypes.begin(),
                    in_arg_dtypes.end());
  arg_dtypes.insert(arg_dtypes.end(), out_arg_dtypes.begin(),
                    out_arg_dtypes.end());

  return arg_dtypes;
}

/// @brief Extract datatypes of all arguments of the op from Nodes' tensors.
/// This is valid only for DD-vaip-Python flow.
static std::vector<std::string> extract_dtypes_from_tensors(
    const OpsFusion::Metadata::OpInfo &op_info,
    const std::map<std::string, Metadata::OffsetInfo> &tensor_map) {
  std::vector<std::string> arg_dtypes;
  for (const auto &arg : op_info.args) {
    arg_dtypes.push_back(MAP_AT(tensor_map, arg).dtype);
  }
  return arg_dtypes;
}

/// @brief Wrapper to extract datatypes of all arguments of the op
static std::vector<std::string> extract_arg_dtypes(
    const OpsFusion::Metadata::OpInfo &op_info,
    const std::map<std::string, Metadata::OffsetInfo> &tensor_map) {
  std::vector<std::string> arg_dtypes;
  if (op_info.attr.find("in_dtypes") != op_info.attr.end() &&
      op_info.attr.find("out_dtypes") != op_info.attr.end()) {
    arg_dtypes = extract_dtypes_from_attrs(op_info);
  } else {
    arg_dtypes = extract_dtypes_from_tensors(op_info, tensor_map);
  }
  return arg_dtypes;
}

std::unique_ptr<OpInterface>
create_impl(const std::string &op_type, const std::vector<std::string> &types,
            const std::map<std::string, std::any> &attr) {
  if (op_type == "MatMul" || op_type == "QMatMul") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 4);
    if ((a_type == "uint8") && (b_type == "uint8") && (c_type == "uint8")) {
      return std::make_unique<ryzenai::matmul<uint8_t, uint8_t, uint8_t>>(
          a_type, b_type, c_type, false, attr);
    } else if ((a_type == "uint16") && (b_type == "uint8") &&
               (c_type == "uint16")) {
      return std::make_unique<ryzenai::matmul<uint16_t, uint8_t, uint16_t>>(
          a_type, b_type, c_type, false, attr);
    } else {
      throw std::runtime_error(
          "Provided Datatypes are not supported by current Matmul Impl.");
    }
  } else if (op_type == "QMatMulDynamic") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 3);
    if ((a_type == "uint16") && (b_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<
          ryzenai::act_act_matmul<uint16_t, uint16_t, uint16_t>>(
          a_type, b_type, c_type, false, attr);
    } else {
      throw std::runtime_error("Provided Datatypes are not supported by "
                               "current QMatMulDynamic Impl.");
    }
  } else if (op_type == "QMulSoftmax") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::softmax_qdq<int16_t, int16_t, uint16_t>>(
          a_type, "uint16", c_type, false, attr);
    } else {
      throw std::runtime_error("Provided Datatypes are not supported by "
                               "current QMulSoftMax Impl.");
    }
  } else if (op_type == "QBroadcastAdd") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 3);
    if ((a_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<
          ryzenai::matvec_add<uint16_t, uint16_t, uint16_t>>(
          a_type, "uint16", c_type, false, attr);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current BroadcastAdd Impl.");
    }
  } else if (op_type == "MladfMatMul") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    // b_type is overriden to uint4. Do not used b_type
    (void)b_type;
    const auto &c_type = ARRAY_AT(types, 0);
    if ((a_type == "bfloat16")) {
      return std::make_unique<
          ryzenai::mladfmatmulbias<int16_t, int8_t, int16_t, int16_t>>(
          a_type, "uint4", c_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current MladfMatmul Impl.");
    }
  } else if (op_type == "LRN" || op_type == "LayerNorm" ||
             op_type == "QLayerNorm") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 4);
    if ((a_type == "bfloat16") && (b_type == "uint16") && (c_type == "uint8")) {
      return std::make_unique<ryzenai::layernorm<int16_t, int16_t, uint8_t>>(
          a_type, b_type, c_type, false, attr);
    } else if ((a_type == "bfloat16") && (b_type == "uint16") &&
               (c_type == "uint16")) {
      return std::make_unique<ryzenai::layernorm<int16_t, int16_t, uint16_t>>(
          a_type, b_type, c_type, false, attr);

    } else if ((a_type == "uint16") && (b_type == "uint16") &&
               (c_type == "uint16")) {
      return std::make_unique<ryzenai::layernorm<int16_t, int16_t, uint16_t>>(
          "bfloat16", b_type, c_type, false, attr);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current LRN Impl.");
    }
  } else if (op_type == "QGroupNorm") {
    const auto &a_type = "bfloat16";
    const auto &c_type = "bfloat16";
    if ((a_type == "bfloat16") && (c_type == "bfloat16")) {
      return std::make_unique<ryzenai::groupnorm<int16_t, int16_t, uint16_t>>(
          a_type, "bfloat16", c_type, false, attr);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current QGroupNorm Impl.");
    }
  } else if (op_type == "MatMulAdd" || op_type == "QMatMulAdd") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 4);
    if ((a_type == "uint8") && (b_type == "uint8") && (c_type == "uint8")) {
      return std::make_unique<ryzenai::matmul<uint8_t, uint8_t, uint8_t>>(
          a_type, b_type, c_type, false, attr);
    } else if ((a_type == "uint16") && (b_type == "uint8") &&
               (c_type == "uint16")) {
      return std::make_unique<ryzenai::matmul<uint16_t, uint8_t, uint16_t>>(
          a_type, b_type, c_type, false, attr);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current MatmulAdd Impl.");
    }
  } else if (op_type == "QELWEMUL_qdq") {
    const auto &a_type = "bfloat16"; // ARRAY_AT(types, 0);
    const auto &b_type = "uint16";   // ARRAY_AT(types, 1);
    const auto &c_type = "uint16";   // ARRAY_AT(types, 3);
    if ((a_type == "bfloat16") && (b_type == "uint16") &&
        (c_type == "uint16")) {
      return std::make_unique<
          ryzenai::elwmul_qdq<uint16_t, uint16_t, uint16_t>>(
          a_type, b_type, c_type, false, attr);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current QELWEMUL_qdq Impl.");
    }
  } else if (op_type == "QResize") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::nni_resize<uint16_t, uint16_t>>(
          a_type, c_type, false, attr);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current QResize Impl.");
    }
  } else if (op_type == "ADD" || op_type == "Add" || op_type == "DQAdd" ||
             op_type == "QEltWiseAdd") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 3);
    if ((a_type == "uint8") && (c_type == "bfloat16")) {
      return std::make_unique<ryzenai::elw_add<uint8_t, uint8_t, uint16_t>>(
          a_type, "uint8", c_type, false, attr);
    } else if ((a_type == "uint16") && (c_type == "bfloat16")) {
      return std::make_unique<ryzenai::elw_add<uint16_t, uint16_t, uint16_t>>(
          a_type, "uint16", c_type, false, attr);
    } else if ((a_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::elw_add<uint16_t, uint16_t, uint16_t>>(
          a_type, "uint16", c_type, false, attr);
    } else if ((a_type == "bfloat16") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::elw_add<uint16_t, uint16_t, uint16_t>>(
          a_type, "uint16", c_type, false, attr);
    } else if ((a_type == "bfloat16") && (c_type == "bfloat16")) {
      return std::make_unique<ryzenai::elw_add<uint16_t, uint16_t, uint16_t>>(
          a_type, "uint16", c_type, false, attr);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current ADD Impl.");
    }
  } else if (op_type == "MHAGRPB" || op_type == "QMHAGRPB") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 7);
    const auto &c_type = ARRAY_AT(types, 9);
    if ((a_type == "uint8") && (b_type == "uint8") && (c_type == "uint8")) {
      return std::make_unique<ryzenai::mhagrpb<uint8_t, uint8_t, uint8_t>>(
          a_type, b_type, c_type, false);
    } else if ((a_type == "uint16") && (b_type == "uint8") &&
               (c_type == "uint16")) {
      return std::make_unique<ryzenai::mhagrpb<uint16_t, uint8_t, uint16_t>>(
          a_type, b_type, c_type, false);
    } else if ((a_type == "uint16") && (b_type == "uint16") &&
               (c_type == "uint16")) {
      return std::make_unique<ryzenai::mhagrpb<uint16_t, uint16_t, uint16_t>>(
          a_type, b_type, c_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current MHA GRPB Impl.");
    }
  } else if (op_type == "QMHAWINDOW") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::mhawindow<uint16_t, uint8_t, uint16_t>>(
          a_type, "uint8", c_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current QMHAWINDOW Impl.");
    }
  } else if (op_type == "QMHA") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::mha<uint16_t, uint8_t, uint16_t>>(
          a_type, "uint8", c_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current QMHA Impl.");
    }
  } else if (op_type == "QReshapeTranspose") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::transpose<uint16_t, int8_t, uint16_t>>(
          a_type, "uint8", c_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current ReshapeTranspose Impl.");
    }
  } else if (op_type == "QSlice") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::slice<uint16_t, uint16_t>>(
          a_type, c_type, false, attr);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current QSlice Impl.");
    }
  } else if (op_type == "QConcat") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::concat<uint16_t, uint16_t>>(
          a_type, c_type, false, attr);
    } else {
      throw std::runtime_error("Provided Datatypes are not supported by "
                               "current QConcat Impl.");
    }
  } else if (op_type == "QGlobalAvgPool") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::gap<uint16_t, uint16_t>>(a_type, c_type,
                                                                false, attr);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current QGlobalAvgPool Impl.");
    }
  } else if (op_type == "QMHACHANNEL") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::mhachannel<uint16_t, uint8_t, uint16_t>>(
          a_type, "uint8", c_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported  by current QMHACHANNEL Impl.");
    }
  } else if (op_type == "MatMulAddGelu" || op_type == "QMatMulAddGelu") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 5);
    if ((a_type == "uint8") && (b_type == "uint8") && (c_type == "uint8")) {
      return std::make_unique<
          ryzenai::matmulgeluadd<uint8_t, uint8_t, uint8_t>>(a_type, b_type,
                                                             c_type, false);
    } else if ((a_type == "uint16") && (b_type == "uint8") &&
               (c_type == "uint16")) {
      return std::make_unique<
          ryzenai::matmulgeluadd<uint16_t, uint8_t, uint16_t>>(a_type, b_type,
                                                               c_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current MatmulAddGelu Impl.");
    }
  } else if (op_type == "square") {
    return std::make_unique<ryzenai::square<int32_t, int32_t>>(false);
  } else if (op_type == "cube") {
    return std::make_unique<ryzenai::cube<int32_t, int32_t>>(false);
  } else if (op_type == "PM_LOAD") {
    return std::make_unique<ryzenai::pm_load>(false);
  } else if (op_type == "SILU") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 1);
    if ((a_type == "bfloat16") && (c_type == "bfloat16")) {
      return std::make_unique<ryzenai::silu<uint16_t, uint16_t>>(a_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current Silu Impl.");
    }
  } else if (op_type == "QSilu") {

    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 2);
    const auto &b_type = "uint16";
    if ((a_type == "bfloat16") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::silu_qdq<uint16_t, uint16_t, uint16_t>>(
          a_type, b_type, c_type, false, attr);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current Silu Impl.");
    }
  } else if (op_type == "QGelu") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "uint16") && (c_type == "bfloat16")) {
      return std::make_unique<ryzenai::gelu<uint16_t, uint16_t, uint16_t>>(
          a_type, "uint16", c_type, false, attr);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current ADD Impl.");
    }
  } else if (op_type == "BMM") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "bfloat16") && (b_type == "bfloat16") &&
        (c_type == "bfloat16")) {
      return std::make_unique<ryzenai::bmm<uint16_t, uint16_t, uint16_t>>(
          a_type, b_type, c_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current BMM Impl.");
    }
  } else if (op_type == "ELWMUL") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "bfloat16") && (b_type == "bfloat16") &&
        (c_type == "bfloat16")) {
      return std::make_unique<ryzenai::elw_mul<uint16_t, uint16_t, uint16_t>>(
          a_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current Elwmul Impl.");
    }
  } else if (op_type == "MLADFADD") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "bfloat16") && (b_type == "bfloat16") &&
        (c_type == "bfloat16")) {
      return std::make_unique<ryzenai::mladf_add<uint16_t, uint16_t, uint16_t>>(
          a_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current MLADFADD Impl.");
    }
  } else if (op_type == "MASKEDSOFTMAX") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "bfloat16") && (b_type == "bfloat16") &&
        (c_type == "bfloat16")) {
      return std::make_unique<
          ryzenai::masked_softmax<uint16_t, uint16_t, uint16_t>>(a_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current Elwmul Impl.");
    }
  } else if (op_type == "MLADFMHAROPE") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "bfloat16") && (b_type == "bfloat16") &&
        (c_type == "bfloat16")) {
      return std::make_unique<ryzenai::mha_rope<uint16_t, uint16_t, uint16_t>>(
          a_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current MHA Rope Impl.");
    }
  } else if (op_type == "MLADFRMSNORM") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "bfloat16") && (b_type == "bfloat16") &&
        (c_type == "bfloat16")) {
      return std::make_unique<ryzenai::rms_norm<uint16_t, uint16_t, uint16_t>>(
          a_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current RMS Norm Impl.");
    }
  } else if (op_type == "MLADFMATMULA16A16") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 3);
    if ((a_type == "uint16") && (b_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<
          ryzenai::matmul_a16a16_mladf<uint16_t, uint16_t, uint16_t>>(
          a_type, b_type, c_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current MLADFMATMULA16A16 Impl.");
    }
  } else if (op_type == "MLADFMATMULA16W8") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 5);
    if ((a_type == "uint16") && (b_type == "uint8") && (c_type == "uint16")) {
      return std::make_unique<
          ryzenai::matmul_a16w8_mladf<uint16_t, uint8_t, uint16_t>>(
          a_type, b_type, c_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current MLADFMATMULA16W8 Impl.");
    }
  } else if (op_type == "RECORD_TIMER") {
    return std::make_unique<ryzenai::record_timer>();
  } else if (op_type == "QConv") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 2);
    if ((a_type == "uint16") && (b_type == "uint8") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::conv<uint16_t, uint8_t, uint16_t>>(
          a_type, b_type, c_type, false, attr);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported  by current QConv Impl.");
    }
  } else if (op_type == "QConcateOPs") {
    const auto &a_type = types.at(0);
    const auto &b_type = types.at(1);
    const auto &c_type = types.at(2);

    std::string json_attr;

    if (attr.count("list_attrs") &&
        attr.at("list_attrs").type() == typeid(std::vector<std::string>)) {
      const auto &attrs_vec =
          std::any_cast<const std::vector<string> &>(attr.at("list_attrs"));
      json_attr = attrs_vec[0];
    }

    std::vector<std::map<std::string, std::any>> attrVec;

    // Set default to 320
    int graphId = 320;
    int inChannels = 8;
    int outChannels = 16;

    json data;
    try {
      data = json::parse(json_attr, nullptr, true);
    } catch (std::exception &e) {
      std::cout << e.what() << std::endl;
      throw runtime_error("Failed to parse JSON");
    }

    // Fill the attributes
    for (auto &elem : data) {
      std::map<std::string, std::any> attr_map;

      for (auto it = elem.begin(); it != elem.end(); ++it) {
        if (it.key() == "opType" || it.key() == "opIfmDtype" ||
            it.key() == "opWtsDtype" || it.key() == "opOfmDtype") {
          attr_map[it.key()] = json_to_any(it.value());
        } else if (it.key() == "group" || it.key() == "zero_point" ||
                   it.key() == "width") {
          auto value = json_to_any(it.value());
          std::vector<int> x;
          x.push_back(std::any_cast<int>(value));
          attr_map[it.key()] = x;
        } else if (it.key() == "graphID") {
          auto value = json_to_any(it.value());
          graphId = std::any_cast<int>(value);
        } else if (it.key() == "inChannels") {
          auto value = json_to_any(it.value());
          inChannels = std::any_cast<int>(value);
        } else if (it.key() == "outChannels") {
          auto value = json_to_any(it.value());
          outChannels = std::any_cast<int>(value);
        } else {
          auto value = json_to_any(it.value());
          std::vector<int> x;
          for (const auto &elem : std::any_cast<std::vector<std::any>>(value)) {
            if (elem.type() == typeid(int)) {
              x.push_back(std::any_cast<int>(elem));
            }
          }
          attr_map[it.key()] = x;
        }
      }
      attrVec.push_back(attr_map);
    }

    if ((a_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::concateOps<uint16_t, uint16_t>>(
          graphId, inChannels, outChannels, attrVec);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported  by current QConcatOPs Impl.");
    }
  } else if (op_type == "IConv") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 4);
    if ((a_type == "uint16") && (b_type == "uint8") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::iconv<uint16_t, uint8_t, uint16_t>>(
          a_type, b_type, c_type, false, attr);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current IConv Impl.");
    }
  } else if (op_type == "xcom-conv2d") {
    // order depends on onnx node
    constexpr std::uint32_t ACT_INDEX = 0;
    constexpr std::uint32_t WEIGHT_INDEX = 3;
    constexpr std::uint32_t BIAS_INDEX = 6;
    constexpr std::uint32_t OUT_INDEX = 11;
    const auto &a_type = ARRAY_AT(types, ACT_INDEX);
    const auto &b_type = ARRAY_AT(types, WEIGHT_INDEX);
    const auto &bias_type = ARRAY_AT(types, BIAS_INDEX);
    const auto &c_type = ARRAY_AT(types, OUT_INDEX);
    bool load_xrt = false;
    if ((a_type == "int8") && (b_type == "int8") && (c_type == "int8")) {
      return std::make_unique<ryzenai::xcom::conv2d<
          std::int8_t, std::int8_t, std::int8_t, std::int8_t, false>>(
          a_type, b_type, bias_type, c_type, load_xrt);
    } else if ((a_type == "uint16") && (b_type == "uint8") &&
               (bias_type == "int32") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::xcom::conv2d<
          std::uint16_t, std::uint8_t, std::int32_t, std::uint16_t, false>>(
          a_type, b_type, bias_type, c_type, load_xrt);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported  by current XCOM::CONV2d Impl.");
    }
  } else if (op_type == "PSRMHA") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 4);
    if ((a_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::mhapsr<uint16_t, uint8_t, uint16_t>>(
          a_type, "uint8", c_type, false, attr);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current PSRMHA Impl.");
    }
  } else if (op_type == "QConv2MatMul") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 4);
    if ((a_type == "uint16") && (b_type == "uint8") && (c_type == "uint16")) {
      return std::make_unique<
          ryzenai::conv2matmul<uint16_t, uint8_t, uint16_t>>(
          a_type, b_type, c_type, false, attr);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current QConv2MatMul Impl.");
    }
  } else if (op_type == "Mladfsoftmax") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 6);
    if ((a_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<
          ryzenai::mladf_softmax<uint16_t, uint8_t, uint16_t>>(a_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current Mladfsoftmax Impl.");
    }
  } else if (op_type == "QLstm") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &b_type = ARRAY_AT(types, 1);
    const auto &c_type = ARRAY_AT(types, 7);
    if ((a_type == "uint16") && (b_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<ryzenai::lstm<uint16_t, uint16_t, uint16_t>>(
          a_type, b_type, c_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current Mladfsoftmax Impl.");
    }
  } else if (op_type == "Mladfelwadd") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 3);
    if ((a_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<
          ryzenai::ml_adf_elw_add<uint16_t, uint16_t, uint16_t>>(
          a_type, "uint16", c_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current ADD Impl.");
    }
  } else if (op_type == "Mladfelwmul") {
    const auto &a_type = ARRAY_AT(types, 0);
    const auto &c_type = ARRAY_AT(types, 3);
    if ((a_type == "uint16") && (c_type == "uint16")) {
      return std::make_unique<
          ryzenai::ml_adf_elw_mul<uint16_t, uint16_t, uint16_t>>(
          a_type, "uint16", c_type, false);
    } else {
      throw std::runtime_error(
          "Datatypes are not supported by current ADD Impl.");
    }
  } else {
    throw std::runtime_error("No implementation for op_type : "s + op_type);
  }
}

// TODO : What is the right info to be passed to the builder ?
std::unique_ptr<OpInterface> OpBuilder::create(
    const std::string &op_name, const Metadata::OpInfo &op_info,
    const std::map<std::string, Metadata::OffsetInfo> &tensor_map) {
  std::vector<std::string> arg_dtypes;
  try {
    arg_dtypes = extract_arg_dtypes(op_info, tensor_map);
    return create_impl(op_info.type, arg_dtypes, op_info.attr);
  } catch (std::exception &e) {
    std::ostringstream oss;
    for (int i = 0; i < arg_dtypes.size(); ++i) {
      oss << i << ":" << arg_dtypes.at(i) << " ";
    }
    DOD_THROW(dod_format("OpBuilder::create() failed.\n"
                         "Details:\n"
                         "  OpName: {}\n"
                         "  OpType: {}\n"
                         "  Provided arg dtypes: {}\n"
                         "  Error: {}",
                         op_name, op_info.type, oss.str(), e.what()));

    // return nullptr to fix compiler warning. Control should not reach here.
    return nullptr;
  }
}

bool OpBuilder::is_supported(const std::string &op_type,
                             const std::vector<std::string> &types,
                             const std::map<std::string, std::any> &attr) {
  try {
    create_impl(op_type, types, attr);
  } catch (const std::exception &) {
    return false;
  }
  return true;
}

} // namespace OpsFusion
