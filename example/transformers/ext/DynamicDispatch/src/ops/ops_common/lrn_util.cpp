#include "lrn_util.h"
#include "sfmx_util.h"
#include <fstream>

LrnUtil::LrnUtil() {}

void LrnUtil::Float_Bits_to_INT8(int8_t *store, float a) {

  int bits = *((int *)&a);
  bits = bits >> 16;

  int8_t first_half = bits;
  int8_t second_half = bits >> 8;

  store[0] = first_half;
  store[1] = second_half;
}

bool LrnUtil::get_input_from_file_bf16(int8_t *in_subv, std::string file) {
  bool write = 0;
  std::ifstream input_file(file);
  std::cout << file << std::endl;
  int idx_offset = 0;

  if (input_file.is_open()) {
    write = 1;
    std::string line;

    int idx_offset = 0;

    while (std::getline(input_file, line)) {

      float curr = std::stod(line);

      LrnUtil::Float_Bits_to_INT8(in_subv + idx_offset, curr);

      idx_offset += 2;
    }
  }

  input_file.close();

  return write;
}

// bool LrnUtil::get_parameter_input_bf16(int8_t * param_ptr,
// std::vector<std::string> param_files)
// {
//     bool write1 = 0;
//     bool write2 = 0;
//     std::ifstream  gamma_file(param_files[0]);
//     std::ifstream  beta_file(param_files[1]);

//     int idx_offset = 0;
//     if (gamma_file.is_open())
//     {
//         write1 = 1;
//         std::string line;

//         while (std::getline(gamma_file, line))
//         {

//             float curr = std::stod(line);

//             LrnUtil::Float_Bits_to_INT8(param_ptr + idx_offset, curr);

//             idx_offset += 2;
//         }
//     }

//     if (beta_file.is_open())
//     {
//         write2 = 1;
//         std::string line;

//         while (std::getline(beta_file, line))
//         {
//             float curr = std::stod(line);

//             LrnUtil::Float_Bits_to_INT8(param_ptr + idx_offset, curr);
//             idx_offset += 2;

//         }
//     }

//     gamma_file.close();
//     beta_file.close();

//     return write1 && write2;

// }

bool LrnUtil::get_input_from_file_int8(int8_t *input, std::string file) {
  bool write = 0;
  std::ifstream input_file(file);

  int idx_offset = 0;

  if (input_file.is_open()) {
    write = 1;
    std::string line;

    int idx_offset = 0;

    while (std::getline(input_file, line)) {

      int8_t curr = (int8_t)std::stoi(line);

      input[idx_offset] = curr;
      idx_offset++;
    }
  }

  input_file.close();

  return write;
}

float LrnUtil::get_maximum_difference_BF16(int8_t *output,
                                           int8_t *output_reference,
                                           int number_of_values) {
  SfmxUtil SoftMaxUtil;
  int16_t *sixteen_bit_ptr_output = (int16_t *)output;
  int16_t *sixteen_bit_ptr_reference = (int16_t *)output_reference;
  float maximum_difference = 0;

  for (int i = 0; i < number_of_values; ++i) {
    float curr_output =
        SoftMaxUtil.Sixteen_bits_ToFloat(sixteen_bit_ptr_output[i]);
    float curr_reference =
        SoftMaxUtil.Sixteen_bits_ToFloat(sixteen_bit_ptr_reference[i]);
    // printf("Expected %f, Received %f \n", curr_reference, curr_output);

    if (abs(curr_output - curr_reference) > maximum_difference) {

      maximum_difference = abs(curr_output - curr_reference);
    }
  }

  return maximum_difference;
}

int LrnUtil::number_of_rounding_errors(int8_t *output, int8_t *reference,
                                       int number_of_values) {
  int errors = 0;
  for (int i = 0; i < number_of_values; ++i) {
    if (output[i] != reference[i]) {
      errors++;
    }
  }

  return errors;
}

bool LrnUtil::within_delta(int8_t *output, int8_t *reference,
                           int number_of_values) {
  bool pass = true;
  for (int i = 0; i < number_of_values; ++i) {
    if (abs(output[i] - reference[i]) > 1) {
      pass = false;
    }
  }

  return pass;
}

bool LrnUtil::write_to_file_bf16(int8_t *in_subv, std::string file,
                                 int num_elements) {
  SfmxUtil SoftMaxUtil;
  std::ofstream output_file(file);
  bool written = false;

  int16_t *sixteen_bit_ptr_output = (int16_t *)in_subv;

  if (output_file.is_open()) {
    written = true;
    for (int i = 0; i < num_elements; ++i) {
      float calculated_value =
          SoftMaxUtil.Sixteen_bits_ToFloat(sixteen_bit_ptr_output[i]);
      output_file << calculated_value << std::endl;
    }
  }

  return written;
}

bool LrnUtil::write_to_file_int8(int8_t *in_subv, std::string file,
                                 int num_elements) {
  SfmxUtil SoftMaxUtil;

  std::ofstream output_file(file);

  bool written = false;

  if (output_file.is_open()) {
    written = true;
    for (int i = 0; i < num_elements; ++i) {
      int curr_val = in_subv[i];
      output_file << curr_val << std::endl;
    }
  }
  return written;
}
