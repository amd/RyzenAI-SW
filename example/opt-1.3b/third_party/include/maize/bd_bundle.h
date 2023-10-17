#pragma once

#include <nlohmann/json.hpp>

#include <adf/adf_types.hpp>

namespace aiectrl {

inline void to_json(nlohmann::json& j, const dma_buffer_descriptor &bd) {
    j = nlohmann::json{
      {"address", bd.address},
      {"length", bd.length},
      {"stepsize", bd.stepsize},
      {"wrap", bd.wrap},
      {"padding", bd.padding},
      {"enable_packet", bd.enable_packet},
      {"packet_id", bd.packet_id},
      {"out_of_order_bd_id", bd.out_of_order_bd_id},
      {"tlast_suppress", bd.tlast_suppress},
      {"iteration_stepsize", bd.iteration_stepsize},
      {"iteration_wrap", bd.iteration_wrap},
      {"iteration_current", bd.iteration_current},
      {"enable_compression", bd.enable_compression},
      {"lock_acq_enable", bd.lock_acq_enable},
      {"lock_acq_value", bd.lock_acq_value},
      {"lock_acq_id", bd.lock_acq_id},
      {"lock_rel_value", bd.lock_rel_value},
      {"lock_rel_id", bd.lock_rel_id},
      {"use_next_bd", bd.use_next_bd},
      {"next_bd", bd.next_bd}
    };
}

inline void from_json(const nlohmann::json& j, dma_buffer_descriptor &bd) {
    j.at("address").get_to(bd.address);
    j.at("length").get_to(bd.length);
    j.at("stepsize").get_to(bd.stepsize);
    j.at("wrap").get_to(bd.wrap);
    j.at("padding").get_to(bd.padding);
    j.at("enable_packet").get_to(bd.enable_packet);
    j.at("packet_id").get_to(bd.packet_id);
    j.at("out_of_order_bd_id").get_to(bd.out_of_order_bd_id);
    j.at("tlast_suppress").get_to(bd.tlast_suppress);
    j.at("iteration_stepsize").get_to(bd.iteration_stepsize);
    j.at("iteration_wrap").get_to(bd.iteration_wrap);
    j.at("iteration_current").get_to(bd.iteration_current);
    j.at("enable_compression").get_to(bd.enable_compression);
    j.at("lock_acq_enable").get_to(bd.lock_acq_enable);
    j.at("lock_acq_value").get_to(bd.lock_acq_value);
    j.at("lock_acq_id").get_to(bd.lock_acq_id);
    j.at("lock_rel_value").get_to(bd.lock_rel_value);
    j.at("lock_rel_id").get_to(bd.lock_rel_id);
    j.at("use_next_bd").get_to(bd.use_next_bd);
    j.at("next_bd").get_to(bd.next_bd);
}

}

namespace amd { namespace maize {

struct bd_bundle {
    int32_t type = 0; // 0 is gm // 1 is memtile // 2 is gmio_wait // 3 is memtile_wait // 4 is write lock
    int32_t direction = 0; // 0 is gm2aie // 1 is aie2gm
    int32_t port_id = 0;
    int32_t buffer_id = 0;
    int32_t ext_buffer_id = -1;
    int32_t repeat_count = 1;
    int32_t enable_tct = 0;
    int32_t column = -1;
    int32_t row = -1;
    std::string port_name = ""; // Must use either port_id or port name
    std::vector<uint32_t> bd_ids;
    std::vector<aiectrl::dma_buffer_descriptor> bds;
};

inline void to_json(nlohmann::json& j, const bd_bundle &b) {
    j = nlohmann::json{
      {"type", b.type},
      {"direction", b.direction},
      {"port_id", b.port_id},
      {"buffer_id", b.buffer_id},
      {"ext_buffer_id", b.ext_buffer_id},
      {"repeat_count", b.repeat_count},
      {"enable_tct", b.enable_tct},
      {"column", b.column},
      {"row", b.row},
      {"port_name", b.port_name},
      {"buffer_descriptor_ids", b.bd_ids},
      {"buffer_descriptors", b.bds}
    }; 
}

inline void from_json(const nlohmann::json& j, bd_bundle &b) {
    j.at("type").get_to(b.type);
    j.at("direction").get_to(b.direction);
    j.at("port_id").get_to(b.port_id);
    j.at("buffer_id").get_to(b.buffer_id);
    j.at("ext_buffer_id").get_to(b.ext_buffer_id);
    j.at("repeat_count").get_to(b.repeat_count);
    j.at("enable_tct").get_to(b.enable_tct);
    j.at("column").get_to(b.column);
    j.at("row").get_to(b.row);
    j.at("port_name").get_to(b.port_name);
    j.at("buffer_descriptor_ids").get_to(b.bd_ids);
    j.at("buffer_descriptors").get_to(b.bds);
}

} // namespace maize
} // namespace amd
