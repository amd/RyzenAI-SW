
#include <memory>
#include <unordered_map>
#include <xrt_context/xrt_context.hpp>

std::unordered_map<std::string,
                   std::shared_ptr<ryzenai::dynamic_dispatch::xrt_context>>
    ryzenai::dynamic_dispatch::xrt_context::ctx_map_;

std::mutex ryzenai::dynamic_dispatch::xrt_context::xrt_ctx_mutex_;
