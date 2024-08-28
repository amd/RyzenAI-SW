#pragma once
#include <filesystem>
#include <string>
bool is_file(const std::string& file_path) {
  return std::filesystem::is_regular_file(std::filesystem::path{file_path});
}
bool is_directory(const std::string& file_path) {
  return std::filesystem::is_directory(std::filesystem::path{file_path});
}
bool exist(const std::string& file_path) {
  return std::filesystem::exists(std::filesystem::path{file_path});
}
bool check_extension(const std::string& file_path, const std::string& ext) {
  return std::filesystem::path{file_path}.extension().string() == ext;
}
std::string absolute(const std::string& file_path) {
  return std::filesystem::absolute(std::filesystem::path{file_path}).string();
}
