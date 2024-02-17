/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __BUFFER_OPS_H__
#define __BUFFER_OPS_H__

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

namespace ryzenai {
// Copy values from text files into buff, expecting values are ascii encoded hex
void init_hex_buf(int *buff, size_t bytesize, std::string &filename) {
  std::ifstream ifs(filename);

  if (!ifs.is_open()) {
    std::cout << "Failure opening file " + filename + " for reading!!"
              << std::endl;
    abort();
  }

  std::string line;
  while (getline(ifs, line)) {
    if (line.at(0) != '#') {
      unsigned int temp = 0;
      std::stringstream ss(line);
      ss >> std::hex >> temp;
      *(buff++) = temp;
    }
  }
}

void init_hex_buf(xrt::bo &bo, size_t bytesize, std::string &filename) {
  init_hex_buf(bo.map<int *>(), bytesize, filename);
}

size_t get_instr_size(std::string &fname) {
  std::ifstream myfile(fname);
  size_t i = 0;
  if (myfile.is_open()) {
    std::string line;
    while (getline(myfile, line)) {
      if (line.at(0) != '#') {
        i++;
      }
    }
    myfile.close();
  }
  return i;
}
} // namespace ryzenai
#endif // __BUFFER_OPS_H__
