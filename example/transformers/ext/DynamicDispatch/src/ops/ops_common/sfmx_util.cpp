#include <cmath>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

SfmxUtil::SfmxUtil() {}

float SfmxUtil::Sixteen_bits_ToFloat(int number, int idx = 0) {
  int num;
  if (idx == 1)
    num = (number & 0xFFFF0000) >> 16;
  else
    num = (number & 0x0000FFFF);

  bool negative = ((num & 0x00008000) != 0);
  float sign = (negative) ? -1.0f : 1.0f;

  int exponents = (num & 0x00007F80) >> 7;
  float power2 = std::pow(2.0f, exponents - 127);

  float factor = 1.0f;
  for (int bit = 0; bit < 7; bit++) {
    int mask = (1 << bit);
    factor += (num & mask) ? std::pow(2.0f, bit - 7) : 0.0f;
  }

  return sign * power2 * factor;
}

bool SfmxUtil::get_in_from_file(int8_t *xin, std::string in_filename,
                                bool is_param, bool file_dump) {
  std::ifstream in_file;
  in_file.open(in_filename.c_str(), std::ifstream::in);

  // float sum_exp_neg[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  // 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; float sum_exp_pos[16] =
  // {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  // 0.0f, 0.0f, 0.0f, 0.0f}; float sum_two_neg[16] = {0.0f, 0.0f, 0.0f, 0.0f,
  // 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  // float sum_two_pos[16] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  // 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  // float sum_exp_pos[16] =
  // {4.23927e+14, 2.9201e+18, 1.77133e+13, 7.5053e+12, 6.13493e+14, 1.7788e+20, 4.45581e+14,
  // 7.09484e+20, 1.59189e+15, 7.01741e+20, 8.66296e+16, 1.44364e+17, 2.50982e+16,
  // 1.86922e+11, 1.65019e+19, 5.39671e+17}; float sum_two_pos[16] =
  // {1.62207e+10, 6.60853e+12, 1.57453e+09, 9.12583e+08, 1.96211e+10, 1.25234e+14,
  // 1.79609e+10, 2.9409e+14, 3.51257e+10, 2.81941e+14, 5.528e+11, 8.13018e+11, 2.43422e+11,
  // 7.59128e+07, 2.09261e+13, 2.44896e+12};

  // for(int i = 0; i < 16; i++)
  //     printf("%5.6g ", sum_two_pos[i]);

  // std::ofstream sum_exp_pos_file;//("sum_exp_pos.bin",std::ios_base::binary);
  // std::ofstream sum_two_pos_file;//("sum_two_pos.bin",std::ios_base::binary);

  // FILE *fp = fopen("sfmx_error.txt", "a");
  // FILE *fexp = fopen("sfmx_exp_pos.bin", "a+b");
  // FILE *ftwo = fopen("sfmx_two_pos.bin", "a+b");
  size_t r1; //= fwrite(a, sizeof a[0], SIZE, f1);
  size_t r2; //= fwrite(a, sizeof a[0], SIZE, f1);

  // sum_exp_pos_file.open("sum_exp_pos.bin", std::ios::out | std::ios::binary);
  // sum_two_pos_file.open("sum_two_pos.bin", std::ios::out | std::ios::binary);

  if (in_file.is_open()) {
    std::cout << "Opened filename " << in_filename << std::endl;
    int count = 0;
    std::string line;
    int elem_idx;
    int line_count;

    int w_idx;
    int block_idx, inBlk_idx;
    int cidx_inblk;
    int cidx;

    float two_x_approx, exp_x_mathval;
    float two_x = 0.0f;
    while (in_file) {
      std::getline(in_file, line);
      std::istringstream ss(line);
      int num; // 1, num2;
      // int idx = 0; // which number a in line of file

      if (is_param) {
        while (ss >> num) {
          *(xin + count) = (int8_t)(num);
          // DEBUG("Input :%d \n",*(xin + count));
          count++;
        }
      } else {
        while (ss >> num) {
          float x = Sixteen_bits_ToFloat(num, 0);

          elem_idx = (count / 2);
          line_count = elem_idx / 8;
          w_idx = line_count % 16;
          inBlk_idx = elem_idx % (8 * 16);
          cidx_inblk = inBlk_idx % 8;
          block_idx = elem_idx / (8 * 16);
          cidx = 8 * block_idx + cidx_inblk;

          // two_x_approx  = std::pow(2.0f, x) / sum_two_pos[w_idx];
          // if(file_dump)
          //     r1 = fwrite( &two_x_approx , 1, sizeof(float), ftwo ) ;

          if (w_idx == 0 && file_dump)
            two_x += std::pow(2.0f, x);

          *(xin + count) = (uint8_t)(num & 0x000000FF);
          // printf("Buffer read :%d \n",*(xin + count));
          count++;
          *(xin + count) = (uint8_t)((num & 0x0000FF00) >> 8);
          // printf("Buffer read :%d \n",*(xin + count));
          count++;

          //----------------------------------------------------------------------------

          ss >> num;
          x = Sixteen_bits_ToFloat(num, 0);

          elem_idx = (count / 2);
          line_count = elem_idx / 8;
          w_idx = line_count % 16;
          inBlk_idx = elem_idx % (8 * 16);
          cidx_inblk = inBlk_idx % 8;
          block_idx = elem_idx / (8 * 16);
          cidx = 8 * block_idx + cidx_inblk;

          if (w_idx == 0 && file_dump)
            two_x += std::pow(2.0f, x);

          // two_x_approx  = std::pow(2.0f, x) / sum_two_pos[w_idx];
          // if(file_dump)
          //     r1 = fwrite( &two_x_approx , 1, sizeof(float), ftwo ) ;

          // std::cout<<num<<" "<<std::endl;
          *(xin + count) = (uint8_t)(num & 0x000000FF);
          // printf("Buffer read :%d \n",*(xin + count));
          count++;
          *(xin + count) = (uint8_t)((num & 0x0000FF00) >> 8);
          // printf("Buffer read :%d \n",*(xin + count));
          count++;
        }
        // if((count/2) % 8 == 0)
        //     printf("\n");
      }
      // std::cout<<std::endl;
    }
    printf("Sum of 2^x: %5.6g\n", two_x);
    printf("Total read bytes: %d\n", count);

    printf("\n Total sum_exp_pos:\n");

  } else {
    printf("Error: input FP not open \n");
    exit(0);
  }
  return true;
}

bool SfmxUtil::compare_out(
    uint8_t *out_subv, // entry point to 2D buffer
    uint8_t *out_refv, // entry point to 2D buffer
    int num_rows,      // number of rows     (number of elements in a column)
    int num_cols,      // number of columns  (number of elements in a row   )
    int multiplier)    // number of bytes per element
{
  bool result = true;
  for (int i = 0; i < (num_rows * num_cols * multiplier); i += multiplier) {
    // FAILED if LSB ()
    uint32_t out32b =
        ((uint8_t)(out_subv[i + 1]) << 8) + (uint8_t)(out_subv[i]);
    uint32_t ref32b =
        ((uint8_t)(out_refv[i + 1]) << 8) + (uint8_t)(out_refv[i]);

    int out_negative = ((out32b & 0x00008000) >> 15);
    int ref_negative = ((ref32b & 0x00008000) >> 15);

    int out_exponents = (out32b & 0x00007F80) >> 7;
    int ref_exponents = (ref32b & 0x00007F80) >> 7;

    int out_mantissa = (out32b & 0x0000007F);
    int ref_mantissa = (ref32b & 0x0000007F);

    float out_fp32_val = Sixteen_bits_ToFloat(int(out32b));
    float ref_fp32_val = Sixteen_bits_ToFloat(int(ref32b));
    float abs_diff = abs(out_fp32_val - ref_fp32_val);

    if ((out_mantissa != ref_mantissa) || (out_exponents != ref_exponents) ||
        (out_negative != ref_negative)) {
      // printf("sign: (%d, %d), expo: (%4d, %4d), mantissa: (%4d, %4d), index:
      // %5d, DIFF\n", out_negative, ref_negative, out_exponents, ref_exponents,
      // out_mantissa, ref_mantissa, i);

      // if(abs_diff > 0.001f)
      printf("Output Val: %5.5g,  Expected Val: %5.5g, index: %5d  DIFF: %5.5g "
             "!! ",
             out_fp32_val, ref_fp32_val, i, abs_diff);

      if ((abs_diff > 0.01f) ||
          abs(out_fp32_val - ref_fp32_val) / fmax(out_fp32_val, ref_fp32_val) >
              0.015625f) // 1/64
      {
        printf("NO GOOD : %5.6g %d\n",
               abs(out_fp32_val - ref_fp32_val) /
                   fmax(out_fp32_val, ref_fp32_val),
               out_mantissa); // printf("failed as diff is too big!\n");
        result &= false;
      }
      /*else if(abs(out_fp32_val-ref_fp32_val) / fmax(out_fp32_val,
      ref_fp32_val) < 0.015625f)
      {
          printf(" OK \n");
      }*/
      else {
        printf(" OK \n");
        // printf("NO GOOD : %5.6g %d %d\n", abs(out_fp32_val-ref_fp32_val) /
        // fmax(out_fp32_val, ref_fp32_val), out_mantissa, ref_mantissa);
      }

      // if( (0.0f > out_fp32_val) || (out_fp32_val > 1.0f))
      //{
      //     //printf("failed as output value out of [0, 1] range! %5.6g\n",
      //     out_fp32_val); result &= false;
      // }
      // if( (0.0f > ref_fp32_val) || (ref_fp32_val > 1.0f))
      //{
      //     //printf("failed as reference value out of [0, 1] range!  %5.6g\n",
      //     ref_fp32_val); result &= false;
      // }
    } else {
      // if(abs_diff != 0.0f)
      {
        // printf("sign: (%d, %d), expo: (%4d, %4d), mantissa: (%4d, %4d),
        // index: %5d, NODIFF!\n", out_negative, ref_negative, out_exponents,
        // ref_exponents, out_mantissa, ref_mantissa, i);
        printf("Output Val: %5.5g, Expected Val: %5.5g, index: %5d, NODIFF    "
               "OK\n",
               out_fp32_val, ref_fp32_val, i);
      }
    }
  }
  return result;
}

SfmxUtil::~SfmxUtil() {}
