/*
 * Copyright � 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#include "../src/ops/ops_common/iconv_matrix.hpp"
#include "enable_perf.hpp"
#include "ops/ops_common/help_file.hpp"
#include <ops/iconv/iconv.hpp>

#include "test_common.hpp"
#define RANDOM_DATA
using namespace iconv_matrix;

template <typename InT = int8_t, typename WgT = int8_t, typename OuT = int16_t>
int test_iconv(int CI, int YI, int XI, int CO, int YO, int XO, int KY, int KX,
               int strideH, int strideW, bool debug = false,
               const std::string &a_dtype = "int16",
               const std::string &b_dtype = "int8",
               const std::string &c_dtype = "int32",
               const std::string &model_name = "PSF") {
  int err_count = 0;

  size_t CIs = static_cast<size_t>(CI);
  size_t YIs = static_cast<size_t>(YI);
  size_t XIs = static_cast<size_t>(XI);
  size_t COs = static_cast<size_t>(CO);
  size_t YOs = static_cast<size_t>(YO);
  size_t XOs = static_cast<size_t>(XO);
  size_t KYs = static_cast<size_t>(KY);
  size_t KXs = static_cast<size_t>(KX);

  std::vector<size_t> a_shape = {1, CIs, YIs, XIs};   // activate
  std::vector<size_t> b_shape = {COs, CIs, KYs, KXs}; // weight
  std::vector<size_t> qdq_shape = {COs};
  std::vector<size_t> qdq_params_shape = {QDQparam_size};
  std::vector<size_t> aie_out_shape = {1, COs, YOs, XOs};

  int YI_act = YI, XI_act = XI, YO_act = YO, XO_act = XO;
  if (YI == 1) { // PSI case
    a_shape = {YIs, XIs, CIs};
    YIs = static_cast<size_t>(sqrt(XI));
    XIs = YIs;
    YI_act = sqrt(XI_act);
    XI_act = YI_act;
  }

  if (YO == 1) { // PSI case
    aie_out_shape = {YOs, XOs, COs};
    YOs = static_cast<size_t>(sqrt(XO));
    XOs = YOs;
    YO_act = sqrt(XO_act);
    XO_act = YO_act;
  }

  if (model_name == "PSI" && strideH == 1) { // DWC
    b_shape = {COs, 1, KYs, KXs};
  }

  std::vector<InT> a(CI * YI * XI);
  std::vector<WgT> b_s1(CO * KY * KX);
  std::vector<WgT> b_s2(CO * CI * KY * KX);
  std::vector<int64_t> qdq(CO, 0); // c0
  std::vector<int32_t> qdq_params(QDQparam_size);
  std::vector<OuT> cpu_out_qdq(CO * YO * XO);
  std::vector<OuT> aie_out(CO * YO * XO);

  ActTensor<InT> X(CI, YI_act, XI_act, a.data());
  ActTensor<OuT> cpu_Y_qdq(CO, YO_act, XO_act, cpu_out_qdq.data());
  ActTensor<OuT> aie_Y(CO, YO_act, XO_act, aie_out.data());
#ifdef RANDOM_DATA
  srand(0xABCD);
  X.init_random();

  uint32_t C1 = 0;
  uint32_t C2 = 1;
  uint8_t SQb = 0;
  uint8_t Sout = 0;
  uint8_t Stdm = 0;
  WgT wgt_zp = 0;
  InT ifm_zp = 0;

  int64_t c0 = 0;
  *(int64_t *)(&qdq_params[qdq_c0_idx]) = c0;
  qdq_params[qdq_c1_idx] = C1;
  qdq_params[qdq_c2_idx] = C2;
  qdq_params[qdq_c3_idx] = 0;
  // qdq_params[qdq_Mv_idx] = Msubv;
  // qdq_params[qdq_Nv_idx] = Nsubv;
  qdq_params[qdq_SQb_idx] = SQb;
  qdq_params[qdq_Sout_idx] = Sout;
  qdq_params[qdq_Stdm_idx] = Stdm;
  qdq_params[qdq_wgt_zp_idx] = wgt_zp;
  qdq_params[qdq_ifm_zp_idx] = ifm_zp;

  int CO_padded = CO;
  int CI_padded = CI;

  if (model_name == "PSI" && strideH == 1) { // DWC
    initialize_random<WgT>(b_s1, CO * KY * KX, 4, 0);
    std::vector<WgT> cpu_b(DwcWgtTensor<WgT, C_IN_SPLIT_DWC>::size(CO, KY, KX));
    DwcWgtTensor<WgT, C_IN_SPLIT_DWC> W(CO, KY, KX, cpu_b.data());

    format_dwc_wgt(W, b_s1.data(), qdq.data(), C1, C2, Stdm, Sout, wgt_zp);

    int stridex = 1;
    int stridey = 1;
    int pad = 1;
    int shift = 0;
    cpu_conv_dw<ActTensor<InT>, DwcWgtTensor<WgT, C_IN_SPLIT_DWC>,
                ActTensor<OuT>, InT, WgT>(X, W, cpu_Y_qdq, stridey, stridex,
                                          pad, shift);
  } else if (strideH == 1 || strideH == 2) { // CONV
    initialize_random<WgT>(b_s2, CO * CI * KY * KX, 8, 0);
    if (model_name == "4x4PSR") { // PSR 4x4 design
      if (CI == 4) {
        CI_padded = 16;
      }
      if (CO == 4) {
        CO_padded = 64;
      }
      auto split_mode = search_subv_mode(XI);
      int stridex = strideW;
      int stridey = strideH;
      int pad = 1;
      int shift = 0;
      if (split_mode == 0) {
        constexpr SUBV_T subv = get_subv(0);
        int constexpr Cis = subv[0];
        int constexpr Cos = subv[1];
        std::vector<WgT> cpu_b(
            ConvWgtTensor<WgT, Cos, Cis>::size(CO_padded, CI_padded, KY, KX));
        ConvWgtTensor<WgT, Cos, Cis> W(CO_padded, CI_padded, KY, KX,
                                       cpu_b.data());

        format_conv_wgt(W, b_s2.data(), CO, CI, qdq.data(), C1, C2, Stdm, Sout,
                        wgt_zp);

        cpu_conv_2d<ActTensor<InT>, ConvWgtTensor<WgT, Cos, Cis>,
                    ActTensor<OuT>, InT, WgT>(X, W, cpu_Y_qdq, stridey, stridex,
                                              pad, shift);
      } else if (split_mode == 1) {
        constexpr SUBV_T subv = get_subv(1);
        int constexpr Cis = subv[0];
        int constexpr Cos = subv[1];
        std::vector<WgT> cpu_b(
            ConvWgtTensor<WgT, Cos, Cis>::size(CO_padded, CI_padded, KY, KX));
        ConvWgtTensor<WgT, Cos, Cis> W(CO_padded, CI_padded, KY, KX,
                                       cpu_b.data());

        format_conv_wgt(W, b_s2.data(), CO, CI, qdq.data(), C1, C2, Stdm, Sout,
                        wgt_zp);

        cpu_conv_2d<ActTensor<InT>, ConvWgtTensor<WgT, Cos, Cis>,
                    ActTensor<OuT>, InT, WgT>(X, W, cpu_Y_qdq, stridey, stridex,
                                              pad, shift);
      } else if (split_mode == 2) {
        constexpr SUBV_T subv = get_subv(2);
        int constexpr Cis = subv[0];
        int constexpr Cos = subv[1];
        std::vector<WgT> cpu_b(
            ConvWgtTensor<WgT, Cos, Cis>::size(CO_padded, CI_padded, KY, KX));
        ConvWgtTensor<WgT, Cos, Cis> W(CO_padded, CI_padded, KY, KX,
                                       cpu_b.data());

        format_conv_wgt(W, b_s2.data(), CO, CI, qdq.data(), C1, C2, Stdm, Sout,
                        wgt_zp);

        cpu_conv_2d<ActTensor<InT>, ConvWgtTensor<WgT, Cos, Cis>,
                    ActTensor<OuT>, InT, WgT>(X, W, cpu_Y_qdq, stridey, stridex,
                                              pad, shift);
      } else if (split_mode == 3) {
        constexpr SUBV_T subv = get_subv(3);
        int constexpr Cis = subv[0];
        int constexpr Cos = subv[1];
        std::vector<WgT> cpu_b(
            ConvWgtTensor<WgT, Cos, Cis>::size(CO_padded, CI_padded, KY, KX));
        ConvWgtTensor<WgT, Cos, Cis> W(CO_padded, CI_padded, KY, KX,
                                       cpu_b.data());

        format_conv_wgt(W, b_s2.data(), CO, CI, qdq.data(), C1, C2, Stdm, Sout,
                        wgt_zp);

        cpu_conv_2d<ActTensor<InT>, ConvWgtTensor<WgT, Cos, Cis>,
                    ActTensor<OuT>, InT, WgT>(X, W, cpu_Y_qdq, stridey, stridex,
                                              pad, shift);
      } else {
        std::cout << "ERROR: Unsupported split mode" << std::endl;
      }
    } else {
      if (strideH == 1) { // PSR 3x3 stride1
        if (XI == 8) {
          std::vector<WgT> cpu_b(
              ConvWgtTensor<WgT, C_OUT_SPLIT_CONV, C_IN_SPLIT_CONV>::size(
                  CO, CI, KY, KX));
          ConvWgtTensor<WgT, C_OUT_SPLIT_CONV, C_IN_SPLIT_CONV> W(
              CO, CI, KY, KX, cpu_b.data());
          format_conv_wgt(W, b_s2.data(), CO, CI, qdq.data(), C1, C2, Stdm,
                          Sout, wgt_zp);
          int stridex = 1;
          int stridey = 1;
          int pad = 1;
          int shift = 0;
          cpu_conv_2d<ActTensor<InT>,
                      ConvWgtTensor<WgT, C_OUT_SPLIT_CONV, C_IN_SPLIT_CONV>,
                      ActTensor<OuT>, InT, WgT>(X, W, cpu_Y_qdq, stridey,
                                                stridex, pad, shift);
        } else {
          if (CO == 4) {
            CO_padded = 32;
          }
          if (CI == 4) {
            CI_padded = 16;
          }

          std::vector<WgT> cpu_b(
              ConvWgtTensor<WgT, C_OUT_SPLIT_CONV_PSR,
                            C_IN_SPLIT_CONV_PSR>::size(CO_padded, CI_padded, KY,
                                                       KX));
          ConvWgtTensor<WgT, C_OUT_SPLIT_CONV_PSR, C_IN_SPLIT_CONV_PSR> W(
              CO_padded, CI_padded, KY, KX, cpu_b.data());

          format_conv_wgt(W, b_s2.data(), CO, CI, qdq.data(), C1, C2, Stdm,
                          Sout, wgt_zp);

          int stridex = 1;
          int stridey = 1;
          int pad = 1;
          int shift = 0;
          cpu_conv_2d<
              ActTensor<InT>,
              ConvWgtTensor<WgT, C_OUT_SPLIT_CONV_PSR, C_IN_SPLIT_CONV_PSR>,
              ActTensor<OuT>, InT, WgT>(X, W, cpu_Y_qdq, stridey, stridex, pad,
                                        shift);
        }
      } else { // 3x3 stride2

        std::vector<WgT> cpu_b(
            ConvWgtTensor<WgT, C_OUT_SPLIT_CONV, C_IN_SPLIT_CONV>::size(
                CO, CI, KY, KX));
        ConvWgtTensor<WgT, C_OUT_SPLIT_CONV, C_IN_SPLIT_CONV> W(CO, CI, KY, KX,
                                                                cpu_b.data());
        format_conv_wgt(W, b_s2.data(), CO, CI, qdq.data(), C1, C2, Stdm, Sout,
                        wgt_zp);
        int stridex = 2;
        int stridey = 2;
        int pad = 1;
        int shift = 0;
        cpu_conv_2d<ActTensor<InT>,
                    ConvWgtTensor<WgT, C_OUT_SPLIT_CONV, C_IN_SPLIT_CONV>,
                    ActTensor<OuT>, InT, WgT>(X, W, cpu_Y_qdq, stridey, stridex,
                                              pad, shift);
      }
    }
  } else { // CONV7
    initialize_random<WgT>(b_s2, CO * CI * KY * KX, 8, 0);

    std::vector<WgT> cpu_b(
        ConvWgtTensor<WgT, C_OUT_SPLIT_CONV7, C_IN_SPLIT_CONV7>::size(CO, 8, 7,
                                                                      4));
    ConvWgtTensor<WgT, C_OUT_SPLIT_CONV7, C_IN_SPLIT_CONV7> W(CO, 8, 7, 4,
                                                              cpu_b.data());

    int fold_factor = 2;
    int Ci_gran = 4;

    fold_conv_wgt<WgT>(b_s2.data(), wgt_zp, CO, CI, KY, KX, fold_factor,
                       Ci_gran, W);

    for (int c = 0; c < W.Co; ++c) {
      W.set_qdq_c0(c, qdq[c]);
    }
    W.set_qdq_c1(C1);
    W.set_qdq_c2(C2);
    W.set_shift_tdm(Stdm);
    W.set_shift_res(Sout);
    W.set_zp_wgt(wgt_zp);
    int stridex = 4;
    int stridey = 2;
    int pad = 0;
    int shift = 0;
    cpu_conv_2d<ActTensor<InT>,
                ConvWgtTensor<WgT, C_OUT_SPLIT_CONV7, C_IN_SPLIT_CONV7>,
                ActTensor<OuT>, InT, WgT>(X, W, cpu_Y_qdq, stridey, stridex,
                                          pad, shift);
  }
#else
  std::string mode;
  if (strideH == 1 &&
      (CI == 128 || CI == 256 || CI == 512 || CI == 1024)) { // DWC
    mode = "dwc";
  } else { // CONV
    mode = "conv";
  }
  std::string ifm_shape = std::to_string(CI) + "x" + std::to_string(YI_act) +
                          "x" + std::to_string(XI_act);
  std::string ofm_shape = std::to_string(CO) + "x" + std::to_string(YO_act) +
                          "x" + std::to_string(XO_act);

  // std::string data_folder = OpInterface::get_dod_base_dir() +
  //                           "//bin_files//psi_model_files//" + mode + "_" +
  //                           ifm_shape + "_" + ofm_shape + "_data//";
  std::string data_folder = OpInterface::get_dod_base_dir() +
                            "//bin_files//PSR_conv_data//" + mode + "_" +
                            ifm_shape + "_" + ofm_shape + "//data//";
  std::string ifm_filename = data_folder + "ifm.txt";
  std::string wgt_filename = data_folder + "wgt.txt";
  std::string qdq_c0_filename = data_folder + "c0.txt";
  std::string qdq_c1_filename = data_folder + "c1.txt";
  std::string qdq_c2_filename = data_folder + "c2.txt";
  std::string shift_tdm_filename = data_folder + "shift_tdm.txt";
  std::string shift_res_filename = data_folder + "shift_res.txt";
  std::string zp_wgt_filename = data_folder + "zp_wgt.txt";
  std::string ofm_filename = data_folder + "ofm.txt";
  std::string zp_ifm_filename = data_folder + "zp_ifm.txt";

  if (strideH == 4) {
    int Ci_no_fold = 3;
    int Yi_no_fold = 224;
    int Xi_no_fold = 224;

    int Ky_no_fold = 7;
    int Kx_no_fold = 7;

    int Sx_no_fold = 4;
    int pad_no_fold = 3;

    int fold_factor = 2;
    int Ci_gran = 4;
    int Xi_gran = 4;

    int ifm_elements = Ci_no_fold * Yi_no_fold * Xi_no_fold;
    int wgt_elements = CO * Ci_no_fold * Ky_no_fold * Kx_no_fold;
    // auto ifm_data = static_cast<InT*>(malloc(ifm_elements * sizeof(InT)));

    std::vector<uint32_t> aint(Ci_no_fold * Yi_no_fold * Xi_no_fold);
    read_data_file<uint32_t>(ifm_filename, (uint32_t *)aint.data());
    for (int r = 0; r < Ci_no_fold * Yi_no_fold * Xi_no_fold; r++) {
      a[r] = (InT)(aint[r]);
    }

    std::vector<uint32_t> bint(CO * Ci_no_fold * Ky_no_fold * Kx_no_fold);
    read_data_file<uint32_t>(wgt_filename, (uint32_t *)bint.data());
    for (int r = 0; r < CO * Ci_no_fold * Ky_no_fold * Kx_no_fold; r++) {
      b_s2[r] = (WgT)(bint[r]);
    }
  } else {
    std::vector<uint32_t> aint(CI * YI * XI);
    read_data_file<uint32_t>(ifm_filename, (uint32_t *)aint.data());
    int r = 0;
    if (CI == 4) { // first layer in PSR
      for (int r = 0; r < CI * YI * XI; r++) {
        a[r] = (InT)(aint[r]);
      }
    } else {
      for (int c = 0; c < X.C; ++c) {
        for (int y = 0; y < X.Y; ++y) {
          for (int x = 0; x < X.X; ++x) {
            X.at(c, y, x) = (InT)(aint[r++]);
          }
        }
      }
    }

    if (mode == "dwc") { // DWC
      std::vector<uint32_t> bint(CO * KY * KX);
      read_data_file<uint32_t>(wgt_filename, (uint32_t *)bint.data());
      for (int r = 0; r < CO * KY * KX; r++) {
        b_s1[r] = (WgT)(bint[r]);
      }
    } else { // CONV
      std::vector<uint32_t> bint(CO * CI * KY * KX);
      read_data_file<uint32_t>(wgt_filename, (uint32_t *)bint.data());
      for (int r = 0; r < CO * CI * KY * KX; r++) {
        b_s2[r] = (WgT)(bint[r]);
      }
    }
  }

  std::vector<uint32_t> cint(CO * YO * XO);
  read_data_file<uint32_t>(ofm_filename, (uint32_t *)cint.data());
  int r = 0;
  for (int c = 0; c < cpu_Y_qdq.C; ++c) {
    for (int y = 0; y < cpu_Y_qdq.Y; ++y) {
      for (int x = 0; x < cpu_Y_qdq.X; ++x) {
        cpu_Y_qdq.at(c, y, x) = (InT)(cint[r++]);
      }
    }
  }
  // for (int r = 0; r < CO * YO * XO; r++) {
  //     cpu_out_qdq[r] = (OuT)(cint[r]);
  // }

  read_data_file<int64_t>(qdq_c0_filename, (int64_t *)qdq.data());

  int32_t C1;
  int32_t C2;
  uint8_t SQb = 0;
  uint32_t Sout;
  uint32_t Stdm;
  uint32_t Zp;
  uint32_t Zp_ifm;

  read_data_file<uint32_t>(qdq_c1_filename, (uint32_t *)&C1);
  read_data_file<uint32_t>(qdq_c2_filename, (uint32_t *)&C2);
  read_data_file<uint32_t>(shift_tdm_filename, (uint32_t *)&Stdm);
  read_data_file<uint32_t>(shift_res_filename, (uint32_t *)&Sout);
  read_data_file<uint32_t>(zp_wgt_filename, (uint32_t *)&Zp);
  read_data_file<uint32_t>(zp_ifm_filename, (uint32_t *)&Zp_ifm);

  int64_t c0 = 0;
  *(int64_t *)(&qdq_params[qdq_c0_idx]) = c0;
  qdq_params[qdq_c1_idx] = C1;
  qdq_params[qdq_c2_idx] = C2;
  qdq_params[qdq_c3_idx] = 0;

  qdq_params[qdq_SQb_idx] = SQb;
  qdq_params[qdq_Sout_idx] = Sout;
  qdq_params[qdq_Stdm_idx] = Stdm;
  qdq_params[qdq_wgt_zp_idx] = Zp;
  qdq_params[qdq_ifm_zp_idx] = Zp_ifm;
#endif
  std::map<std::string, std::any> attr;
  attr["strides"] = std::vector<int>{strideH, strideW};
  attr["input_shape"] = std::vector<int>{1, CI, YI_act, XI_act};
  attr["input_format"] = std::vector<string>{"NCHW"};
  if (model_name == "4x4PSR") {
    attr["design_param"] = std::vector<string>{"4x4"};
  }

  ryzenai::iconv iconv_ =
      ryzenai::iconv<InT, WgT, OuT>(a_dtype, b_dtype, c_dtype, false, attr);

  iconv_.debug(debug);
  std::vector<size_t> param_shape = {YIs, XIs, YOs, XOs};
  iconv_.set_params(model_name, param_shape);

  std::vector<Tensor> const_Tensor;
  if (strideH == 1 &&
      (CI == 128 || CI == 256 || CI == 512 || CI == 1024)) { // DWC
    const_Tensor = {{b_s1.data(), b_shape, b_dtype},
                    {qdq.data(), qdq_shape, "int64"},
                    {qdq_params.data(), qdq_params_shape, "int32"}};
  } else {
    const_Tensor = {{b_s2.data(), b_shape, b_dtype},
                    {qdq.data(), qdq_shape, "int64"},
                    {qdq_params.data(), qdq_params_shape, "int32"}};
  }

  iconv_.initialize_const_params(const_Tensor);

  std::vector<Tensor> input_Tensor;
  input_Tensor = {{a.data(), a_shape, a_dtype}};

  std::vector<Tensor> output_Tensor;
  output_Tensor = {{aie_out.data(), aie_out_shape, c_dtype}};

#ifdef UNIT_TEST_PERF
  LOG_THIS("CO = " << CO << ", YO = " << YO << ", XO = " << XO
                   << ", CI = " << CI << ", YI = " << YI << ", XI = " << XI);
  PROFILE_THIS(iconv_.execute(input_Tensor, output_Tensor));
#else
  iconv_.execute(input_Tensor, output_Tensor);
#endif
  err_count = check_result(cpu_Y_qdq, aie_Y, CO, 2000);

  return err_count;
}

// iCONV, int CI, int YI, int XI, int CO, int YO, int XO, int KY, int KX, int
// strideH, int strideW
TEST(PSI_iCONV_Testa16w8, Kernel1) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      512, 1, 196, 512, 1, 196, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_iCONV_Testa16w8, Kernel2) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      512, 1, 196, 1024, 1, 49, 3, 3, 2, 2, false, "uint16", "uint8", "uint16",
      "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_iCONV_Testa16w8, Kernel3) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      1024, 1, 49, 1024, 1, 49, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_iCONV_Testa16w8, Kernel4) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      256, 1, 784, 512, 1, 196, 3, 3, 2, 2, false, "uint16", "uint8", "uint16",
      "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_iCONV_Testa16w8, Kernel5) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      256, 1, 784, 256, 1, 784, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_iCONV_Testa16w8, Kernel6) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      128, 1, 3136, 256, 1, 784, 3, 3, 2, 2, false, "uint16", "uint8", "uint16",
      "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_iCONV_Testa16w8, Kernel7) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      128, 1, 3136, 128, 1, 3136, 3, 3, 1, 1, false, "uint16", "uint8",
      "uint16", "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSI_iCONV_Testa16w8, Kernel8) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      3, 224, 224, 128, 1, 3136, 7, 7, 4, 4, false, "uint16", "uint8", "uint16",
      "PSI");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel1) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      320, 64, 64, 320, 64, 64, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel2) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      320, 32, 32, 640, 32, 32, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel3) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      640, 32, 32, 640, 32, 32, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel4) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      640, 16, 16, 1280, 16, 16, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel5) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      1280, 16, 16, 1280, 16, 16, 3, 3, 1, 1, false, "uint16", "uint8",
      "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel6) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      1280, 8, 8, 1280, 8, 8, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel7) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      2560, 8, 8, 1280, 8, 8, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel8) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      2560, 16, 16, 1280, 16, 16, 3, 3, 1, 1, false, "uint16", "uint8",
      "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel9) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      1920, 16, 16, 1280, 16, 16, 3, 3, 1, 1, false, "uint16", "uint8",
      "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel10) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      1280, 32, 32, 1280, 32, 32, 3, 3, 1, 1, false, "uint16", "uint8",
      "uint16", "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel11) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      1920, 32, 32, 640, 32, 32, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel12) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      1280, 32, 32, 640, 32, 32, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel13) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      960, 32, 32, 640, 32, 32, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel14) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      640, 64, 64, 640, 64, 64, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel15) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      960, 64, 64, 320, 64, 64, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel16) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      640, 64, 64, 320, 64, 64, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel17) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      4, 64, 64, 320, 64, 64, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel18) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      320, 64, 64, 4, 64, 64, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel19) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      320, 64, 64, 320, 32, 32, 3, 3, 2, 2, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel20) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      640, 32, 32, 640, 16, 16, 3, 3, 2, 2, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(PSR_iCONV_Testa16w8, Kernel21) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      1280, 16, 16, 1280, 8, 8, 3, 3, 2, 2, false, "uint16", "uint8", "uint16",
      "PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

// PSR 4x4
TEST(C4PSR_iCONV_Testa16w8, Kernel1) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      320, 64, 64, 320, 64, 64, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel2) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      320, 32, 32, 640, 32, 32, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel3) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      640, 32, 32, 640, 32, 32, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel4) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      640, 16, 16, 1280, 16, 16, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel5) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      1280, 16, 16, 1280, 16, 16, 3, 3, 1, 1, false, "uint16", "uint8",
      "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel6) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      1280, 8, 8, 1280, 8, 8, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel7) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      2560, 8, 8, 1280, 8, 8, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel8) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      2560, 16, 16, 1280, 16, 16, 3, 3, 1, 1, false, "uint16", "uint8",
      "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel9) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      1920, 16, 16, 1280, 16, 16, 3, 3, 1, 1, false, "uint16", "uint8",
      "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel10) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      1280, 32, 32, 1280, 32, 32, 3, 3, 1, 1, false, "uint16", "uint8",
      "uint16", "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel11) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      1920, 32, 32, 640, 32, 32, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel12) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      1280, 32, 32, 640, 32, 32, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel13) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      960, 32, 32, 640, 32, 32, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel14) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      640, 64, 64, 640, 64, 64, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel15) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      960, 64, 64, 320, 64, 64, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel16) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      640, 64, 64, 320, 64, 64, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel17) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      4, 64, 64, 320, 64, 64, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel18) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      320, 64, 64, 4, 64, 64, 3, 3, 1, 1, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel19) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      320, 64, 64, 320, 32, 32, 3, 3, 2, 2, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel20) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      640, 32, 32, 640, 16, 16, 3, 3, 2, 2, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}

TEST(C4PSR_iCONV_Testa16w8, Kernel21) {
  int err_count = test_iconv<uint16_t, uint8_t, uint16_t>(
      1280, 16, 16, 1280, 8, 8, 3, 3, 2, 2, false, "uint16", "uint8", "uint16",
      "4x4PSR");
  EXPECT_TRUE(err_count == 0) << "Error Count = " << err_count;
}
