#ifndef __SFMX_UTIL_H__
#define __SFMX_UTIL_H__

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class SfmxUtil {
public:
  SfmxUtil();
  ~SfmxUtil();

  bool compare_out(uint8_t *out_subv, uint8_t *out_gemm, int num_rows,
                   int num_cols, int multiplier);
  float Sixteen_bits_ToFloat(int number, int idx);
  bool get_in_from_file(int8_t *in_subv, std::string in_fn,
                        bool is_param = false, bool file_dump = false);
};

#include "sfmx_util.cpp"

#endif
