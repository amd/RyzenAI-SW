
#ifndef __LRN_UTIL_H__
#define __LRN_UTIL_H__

#include <fstream>
#include <iostream>
#include <string>

class LrnUtil {
public:
  LrnUtil();
  void Float_Bits_to_INT8(int8_t *store, float a);
  bool get_input_from_file_bf16(int8_t *in_subv, std::string file);
  // bool get_parameter_input_bf16(int8_t* params_ptr, std::vector<std::string>
  // param_files);
  bool get_input_from_file_int8(int8_t *input, std::string file);
  float get_maximum_difference_BF16(int8_t *output, int8_t *output_reference,
                                    int number_of_values);
  int number_of_rounding_errors(int8_t *output, int8_t *output_reference,
                                int number_of_values);
  bool within_delta(int8_t *output, int8_t *reference, int number_of_values);
  bool write_to_file_bf16(int8_t *output_subv, std::string file,
                          int num_elements);
  bool write_to_file_int8(int8_t *output_subv, std::string file,
                          int num_elements);
};

#include "lrn_util.cpp"

#endif
