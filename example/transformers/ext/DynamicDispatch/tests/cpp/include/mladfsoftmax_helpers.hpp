#pragma once

#include "test_common.hpp"

namespace mladfsoftmax_helpers {

template <typename T>
void read_bin_to_vector(const std::string &file_path, std::vector<T> &vec) {
  std::ifstream ifs(file_path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Failed to open the file.");
  }

  // Get the file size
  ifs.seekg(0, std::ios::end);
  std::streamsize file_size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  // Ensure the vector has the correct size
  std::streamsize element_num = file_size / sizeof(T);
  if (vec.size() != static_cast<size_t>(element_num)) {
    throw std::runtime_error(
        "The vector size does not match the number of elements in the file.");
  }

  // Read the data into the vector
  if (!ifs.read(reinterpret_cast<char *>(vec.data()), file_size)) {
    throw std::runtime_error("Failed to read the data into the vector.");
  }
}

} // namespace mladfsoftmax_helpers
