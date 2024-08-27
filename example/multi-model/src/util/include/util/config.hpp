#pragma once
#include "check.hpp"
#include "nlohmann_json.hpp"
using Config = nlohmann::json;

template <class T>
bool check_type(const Config& config, const std::string& key);

template <>
bool check_type<int32_t>(const Config& config, const std::string& key) {
  return config[key].is_number_integer();
}
template <>
bool check_type<int64_t>(const Config& config, const std::string& key) {
  return config[key].is_number_integer();
}
template <>
bool check_type<float>(const Config& config, const std::string& key) {
  return config[key].is_number_float();
}
template <>
bool check_type<size_t>(const Config& config, const std::string& key) {
  return config[key].is_number_unsigned();
}
template <>
bool check_type<std::string>(const Config& config, const std::string& key) {
  return config[key].is_string();
}
template <>
bool check_type<Config>(const Config& config, const std::string& key) {
  return config[key].is_object();
}

template <>
bool check_type<bool>(const Config& config, const std::string& key) {
  return config[key].is_boolean();
}

bool check_array(const Config& config) { return config.is_array(); }

#define CONFIG_GET(config, type, name, key)       \
  CHECK_WITH_INFO(config.contains(key), key)      \
  CHECK_WITH_INFO(check_type<type>(config, key),  \
                  std::string(key) + ":" + #type) \
  type name = config[key].get<type>();

#define CONFIG_GET_ARRAY(config, type, name, key) \
  CHECK_WITH_INFO(config.contains(key), key)      \
  type name = config[key].get<type>();            \
  CHECK_WITH_INFO(check_array(name), key);