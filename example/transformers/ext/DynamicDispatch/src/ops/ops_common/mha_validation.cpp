#include "mhagprb_matrix.hpp"
#include <cmath>
#include <limits>

#define NUM_ROWS 4
#define NUM_COLS 2
#define ENABLE_BIAS_ADD 1
#define ENABLE_MASK_ADD 1 // only take affect if bias add is 1

namespace mhagprb_matrix {
static int get_HW_index(int y, int x, int H, int W) { return y * W + x; }

// get the 1d index with the "w8" aligned layout
static int get_HW8_index(int y, int x, int H, int W) {
  int blockId = x / 8;
  int blockSize = H * 8;
  int index_in_block = (y * 8 + (x % 8));
  return blockId * blockSize + index_in_block;
}

// Convert layout from row-major to 28 aligned
template <typename T>                  // T == int8 or int16 or int32
static void from_HW_to_HW8(T *HW_buf,  // src
                           T *HW8_buf, // dst
                           int H, int W) {
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      int wr_idx = get_HW8_index(y, x, H, W);
      int rd_idx = get_HW_index(y, x, H, W);
      HW8_buf[wr_idx] = HW_buf[rd_idx];
    }
  }
}

// Convert layout from W8 aligned to row-major
template <typename T>                  // T == int8 or int16 or int32
static void from_HW8_to_HW(T *HW8_buf, // src
                           T *HW_buf,  // dst
                           int H, int W) {
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      int wr_idx = get_HW_index(y, x, H, W);
      int rd_idx = get_HW8_index(y, x, H, W);
      HW_buf[wr_idx] = HW8_buf[rd_idx];
    }
  }
}

static void randomize(int16_t *buf, int numrows, int numcols) {
  int seed = 0x3785; // time(NULL);
  srand(seed);
  for (int y = 0; y < numrows; y++) {
    for (int x = 0; x < numcols; x++) {
      int idx = y * numcols + x;
      buf[idx] = (std::rand() % 32 - 16); //(std::rand() % 256 - 128);
    }
  }
}

template <typename T1, typename T2>
static void ref_mac(T1 *ifm, T2 *wgt, uint32_t *v_acc, int m_subv, int k_subv,
                    int n_subv, bool is_wgt_transposed = true) {
  int ifm_idx;
  int wgt_idx;
  uint32_t acc;
  for (int r = 0; r < m_subv; r++) {   // M
    for (int c = 0; c < n_subv; c++) { // N

      acc = 0;
      for (int k = 0; k < k_subv; k++) // K
      {
        ifm_idx = r * k_subv + k; // kth entry of the r-th row     : ifm [r, k]
        if (is_wgt_transposed)
          wgt_idx = k * n_subv + c; // kth entry of the c-th column  : wgtT[k,
                                    // c]
        else
          wgt_idx = c * k_subv + k; // kth entry of the c-th row     : wgt [c,
                                    // k], 1 row has k_subv elements
        acc += ifm[ifm_idx] * wgt[wgt_idx];
      }
      v_acc[r * n_subv + c] = acc;
    }
  }
}

/*
void int32_to_int16(int16_t* dst, int32_t* src, int height, int width)
{
        for(int y = 0; y < height; y++){
                for(int x = 0; x < width; x++){
                        int idx = y*width+x;
                        dst[idx] = (int16_t)(src[idx] & 0x0000FFFF);
                }
        }
}
*/

static float truncate_to_bf16(float in) {
  uint32_t *tmp = (uint32_t *)&in;
  *tmp = (*tmp) & 0xFFFF0000;
  return *((float *)tmp);
}

static void uint32_to_int8_to_fp32(float *dst, uint32_t *src, int height,
                                   int width, int shift) {
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {

      uint32_t number = src[r * width + c];
      float fp_num = truncate_to_bf16((float)number);

      dst[r * width + c] = fp_num / (float)(1 << shift);
    }
  }
}

static void fp32_to_int16(int16_t *dst, float *src, int height, int width) {
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {

      float fp_num = src[r * width + c]; // softmax output, range : [0.0, 1.0]

      // if(fp_num != 0.0f)
      //	printf("fp_num : %2.10e\n", fp_num);

      // fp_num = truncate_to_bf16(fp_num);

      uint32_t number =
          (uint32_t)(fp_num * 65536.0 - 32768); // in range : [-32768, 32767]

      dst[r * width + c] = number;
    }
  }
}

inline void fp32_to_uint8(uint8_t *dst, float *src, int height, int width) {
  for (int r = 0; r < height; r++) {
    for (int c = 0; c < width; c++) {

      float fp_num = src[r * width + c]; // softmax output, range : [0.0, 1.0]

      // fp_num = truncate_to_bf16(fp_num);

      // int32_t number = (int32_t)(fp_num * 255.0);// - 128.0);   //in range :
      // [-128, 127]
      uint32_t number =
          (uint32_t)(fp_num * 255.0); // - 128.0);   //in range : [-128, 127]

      dst[r * width + c] = (uint8_t)number;
    }
    // printf("\n");
  }
}

static void uint32_to_int16(int16_t *dst, uint32_t *src, int height, int width,
                            int shift) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      uint32_t val32 = src[idx];
      dst[idx] = (int16_t)(val32 >> shift);
    }
  }
}

static void uint32_to_int8(uint8_t *dst, int32_t *src, int height, int width,
                           int shift) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      uint32_t val32 = src[idx];
      dst[idx] = (uint8_t)(val32 >> shift);
    }
  }
}

template <typename T1, typename T2>
void dequant_int8_to_float(T1 in_mat, T2 out_mat, uint8_t zero_point,
                           float scale) {
  for (int i = 0; i < in_mat.num_rows; ++i) {
    for (int j = 0; j < in_mat.num_cols; ++j) {
      out_mat.at(i, j) = (float(in_mat.at(i, j) - zero_point) * scale);
    }
  }
}

template <typename T1, typename T2> // T1 float, T2 uint8
void quant_float_to_uint8(T1 in_mat, T2 out_mat, uint8_t zero_point,
                          float scale) {
  for (int i = 0; i < in_mat.num_rows; ++i) {
    for (int j = 0; j < in_mat.num_cols; ++j) {
      out_mat.at(i, j) = (uint8_t)((int16_t)(in_mat.at(i, j) * (1.0 / scale)) +
                                   (int16_t)zero_point);
    }
  }
}

static void uint32_to_uint8(uint8_t *dst, uint32_t *src, int height, int width,
                            int shift) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      uint32_t val32 = src[idx];
      // dst[idx] = (uint8_t)(val32 >> shift);
      dst[idx] = (uint8_t)(val32 & 0x000000FF);
    }
  }
}

static float approx_exp2(float x) {
  // inp = self.f2bf( inp, rounding=False )
  x = truncate_to_bf16(x);
  // out = ( 1+( inp-np.floor( inp ))) * 2**np.floor( inp )
  float y = (1 + (x - floor(x))) * pow(2.0f, floor(x));

  // out[inp<0] -= 2**( np.floor( np.log2( out[inp<0] ))-23 )
  if (x < 0)
    y -= pow(2.0f, floor(log2(y)) - 23);
  return truncate_to_bf16(y);
}

static void softmax(float *dst, float *src, int height, int width) {
  for (int y = 0; y < height; y++) {
    float rowsum = 0.0f;
    float rowmax = std::numeric_limits<float>::min();
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      rowmax = std::max(src[idx], rowmax);
    }

    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      // rowsum += pow(2, (src[idx]));
      rowsum += approx_exp2(src[idx] - rowmax); // pow(2, (src[idx] - rowmax));
    }
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      float exp = src[idx] - rowmax;

      float pow2 = approx_exp2(src[idx] - rowmax); // pow(2, src[idx]-rowmax);

      /*
      if(exp == 0.0f || pow2 != 0.0f)
      {
              printf("exp[%d] == %2.10e  ", idx, exp );
              printf("pow[%d] == %2.10e  ", idx, pow2);
              printf("rowsum  == %2.10e\n", rowsum   );
      }
      */

      dst[idx] = pow2 / rowsum;
      // dst[idx] = truncate_to_bf16(pow(2, src[idx]) / rowsum);

      // if(dst[idx] != 0.0f)
      //	printf("dst[%d] : %2.10e\n", idx, dst[idx]);
    }
  }
}

template <typename T>
static void elemw_ups(uint32_t *pdst, T *psrc, int height, int width) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      pdst[idx] = (uint32_t)psrc[idx];
    }
  }
}

template <typename T>
static void elemw_add(T *pdst, T *psrc1, T *psrc2, int height, int width) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      pdst[idx] = psrc1[idx] + psrc2[idx];
    }
  }
}

template <typename T>
static void broadcast_vertically(T *pdst, T *bcast_array, int height,
                                 int width) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      pdst[idx] = bcast_array[x];
    }
  }
}

template <typename T>
void ref_qxkxv(T *q_mat,                                       // (mxk)
               T *k_mat,                                       // (nxk)
               T *v_mat,                                       // (nxl)
               T *o8,                                          // (mxl)
               uint32_t *z32,                                  // (mxl)
               int m_subv, int k_subv, int n_subv, int l_subv, //(m, k, n, l)
               gemm_qdq_param<T> qdq_params[4], bool k_mat_transposed = true) {
  uint32_t *qkt_32 = (uint32_t *)malloc(m_subv * n_subv * sizeof(uint32_t));
  T *qkt_qtzed = (T *)malloc(m_subv * n_subv * sizeof(T));
  float *qkt_fp32 = (float *)malloc(m_subv * n_subv * sizeof(float));
  float *sfmx_fp32 = (float *)malloc(m_subv * n_subv * sizeof(float));
  T *sfmx_qtzed = (T *)malloc(m_subv * n_subv * sizeof(T));

  RowMajorMatrix<T> Q(m_subv, k_subv, q_mat);
  RowMajorMatrix<T> K(n_subv, k_subv, k_mat);
  RowMajorMatrix<T> V(n_subv, l_subv, v_mat);
  RowMajorMatrix<uint32_t> QKT(m_subv, n_subv, qkt_32);
  RowMajorMatrix<T> QKT_QTZED(m_subv, n_subv, qkt_qtzed);
  RowMajorMatrix<float> QKT_FP32(m_subv, n_subv, qkt_fp32);
  RowMajorMatrix<T> SM_U8(m_subv, n_subv, sfmx_qtzed);
  RowMajorMatrix<float> SM_FP32(m_subv, n_subv, sfmx_fp32);
  RowMajorMatrix<T> O_S8(m_subv, l_subv, o8);
  RowMajorMatrix<uint32_t> Z_S32(m_subv, l_subv, z32);

  bool perf_tranpose_k = !(k_mat_transposed);
  ref_mac<T, T>(q_mat, k_mat, qkt_32, m_subv, k_subv, n_subv, k_mat_transposed);

  qdq_asym_golden<RowMajorMatrix<T>, RowMajorMatrix<T>,
                  RowMajorMatrix<uint32_t>, RowMajorMatrix<T>>(
      Q, K, QKT, qdq_params[0], QKT_QTZED, perf_tranpose_k);

  dequant_int8_to_float(QKT_QTZED, QKT_FP32, T(qdq_params[2].zero_point),
                        qdq_params[2].scale);

  softmax(sfmx_fp32, QKT_FP32.data, m_subv, n_subv);

  quant_float_to_uint8(SM_FP32, SM_U8, T(qdq_params[3].zero_point),
                       (qdq_params[3].scale));

  qdq_asym_golden<RowMajorMatrix<T>, RowMajorMatrix<T>,
                  RowMajorMatrix<uint32_t>, RowMajorMatrix<T>, T>(
      SM_U8, V, Z_S32, qdq_params[1], O_S8, false);
}

template <typename T>
void ref_qxkxv(T *q_mat,           // (mxk)
               T *k_mat,           // (nxk)
               T *v_mat,           // (nxl)
               float *bias_32,     // (mxn)
               uint16_t *m_mat_16, // (1xn)    uint16_t to represent bfloat16
               T *o8,              // (mxl)
               uint32_t *z32,      // (mxl)
               int m_subv, int k_subv, int n_subv, int l_subv, //(m, k, n, l)
               gemm_qdq_param<T> qdq_params[4], bool k_mat_transposed = true) {
  uint32_t *qkt_32 = (uint32_t *)malloc(m_subv * n_subv * sizeof(uint32_t));
  T *qkt_int8 = (T *)malloc(m_subv * n_subv * sizeof(T));
  float *qkt_fp32 = (float *)malloc(m_subv * n_subv * sizeof(float));
  float *bcasted_mask = (float *)malloc(m_subv * n_subv * sizeof(float));
  float *qkt_plus_m_32 = (float *)malloc(m_subv * n_subv * sizeof(float));
  float *sfmx_fp32 = (float *)malloc(m_subv * n_subv * sizeof(float));
  T *sfmx_uint8 = (T *)malloc(m_subv * n_subv * sizeof(T));
  float *m_mat = (float *)malloc(1 * n_subv * sizeof(float));
  T *tmp_int8 = (T *)malloc(m_subv * n_subv * sizeof(T));

  for (int i = 0; i < n_subv; i++)
    m_mat[i] = bfloat16_to_float(bfloat16_t{m_mat_16[i]});

  RowMajorMatrix<T> Q(m_subv, k_subv, q_mat);
  RowMajorMatrix<T> K(n_subv, k_subv, k_mat);
  RowMajorMatrix<T> V(n_subv, l_subv, v_mat);
  RowMajorMatrix<uint32_t> QKT(m_subv, n_subv, qkt_32);
  RowMajorMatrix<T> QKT_QTZED(m_subv, n_subv, qkt_int8);
  RowMajorMatrix<float> QKT_FP32(m_subv, n_subv, qkt_fp32);
  RowMajorMatrix<T> SM_U8(m_subv, n_subv, sfmx_uint8);
  RowMajorMatrix<float> SM_FP32(m_subv, n_subv, sfmx_fp32);
  RowMajorMatrix<T> O_S8(m_subv, l_subv, o8);
  RowMajorMatrix<uint32_t> Z_S32(m_subv, l_subv, z32);

  bool perf_tranpose_k = true;
  ref_mac<T, T>(q_mat, k_mat, qkt_32, m_subv, k_subv, n_subv, k_mat_transposed);

  // int32_to_int8(tmp_int8, qkt_32, m_subv, n_subv, 11);
  qdq_asym_golden<RowMajorMatrix<T>, RowMajorMatrix<T>,
                  RowMajorMatrix<uint32_t>, RowMajorMatrix<T>>(
      Q, K, QKT, qdq_params[0], QKT_QTZED, perf_tranpose_k);

  dequant_int8_to_float<RowMajorMatrix<T>, RowMajorMatrix<float>>(
      QKT_QTZED, QKT_FP32, qdq_params[2].zero_point, qdq_params[2].scale);
  // broadcast_vertically<float>(    bcasted_mask,    m_mat,  m_subv, n_subv);

  // elemw_add<float>(bcasted_mask,  bcasted_mask,  bias_32,  m_subv, n_subv);

  // elemw_add<float>(qkt_plus_m_32, bcasted_mask, qkt_fp32,  m_subv, n_subv);
  elemw_add<float>(qkt_plus_m_32, bias_32, qkt_fp32, m_subv, n_subv);
  softmax(sfmx_fp32, qkt_plus_m_32, m_subv, n_subv);

  // dequant_int8_to_float<RowMajorMatrix<int8_t>,
  // RowMajorMatrix<float>>(QKT_QTZED, QKT_FP32, qdq_params[2].zero_point,
  // qdq_params[2].scale); softmax(sfmx_fp32, qkt_fp32, m_subv, n_subv);
  quant_float_to_uint8(SM_FP32, SM_U8, qdq_params[3].zero_point,
                       (qdq_params[3].scale));

  ref_mac<T, T>(sfmx_uint8, v_mat, z32, m_subv, n_subv, l_subv);
  // ref_mac<int8_t, T>(qkt_int8, v_mat, z32, m_subv, n_subv, l_subv);
  // ref_mac<int8_t, T>(tmp_int8, v_mat, z32, m_subv, n_subv, l_subv);
  // int32_to_int8(o8, z32, m_subv, l_subv, 11);
  //  replace this with qdq_asym_golden function call
  // int32_to_uint8((uint8_t*)o8, z32, m_subv, l_subv, 11);
  qdq_asym_golden<RowMajorMatrix<T>, RowMajorMatrix<T>,
                  RowMajorMatrix<uint32_t>, RowMajorMatrix<T>, T>(
      SM_U8, V, Z_S32, qdq_params[1], O_S8, false);
  //qdq_asym_golden<RowMajorMatrix<int8_t>, RowMajorMatrix<T>, RowMajorMatrix<int32_t>, RowMajorMatrix<int8_t>> \
	//(QKT_QTZED, V, Z_S32, qdq_params[1], O_S8, false);
}

template <class T>
static bool check_consistent(T *ref, T *buf, int height, int width) {
  bool has_discrep = false;
  // for(int i = 0; i < len; i++)
  float max_rel_error = 0.0f;
  float sum_rel_error = 0.0f;
  float avg_rel_error;
  for (int y = 0; y < height; y++) {

    for (int x = 0; x < width; x++) {
      // if(x % 64 == 0)
      //	printf("\n-------------------------------\n");
      int i = y * width + x;
      if (ref[i] != buf[i]) {
        has_discrep = true;
        // printf("%c ", 'X');
      } else {
        // printf("%c ", 'O');
      }
      float rel_error = std::abs((float)(ref[i] - buf[i]) / (float)abs(ref[i]));
      max_rel_error = fmax(rel_error, max_rel_error);
      sum_rel_error += rel_error;
      printf("@(%3d, %3d)-->(%10d, %10d). Error %4.4f\n", y, x, ref[i], buf[i],
             rel_error * 100);
    }
    printf("\n");
  }
  avg_rel_error = (sum_rel_error / (float)(height * width));
  // printf("Maximum relative error: %4.4f % \n", max_rel_error*100);
  printf("Average relative error: %4.4f % \n", avg_rel_error * 100);
  return avg_rel_error * 100 < 0.2;
  // return !has_discrep;
}

static void vis_buf(int32_t *buf, int height, int width) {
  for (int y = 0; y < height; y++) {
    if (y % 32 == 0)
      printf("\n-------------------------------\n");
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      // if(buf[idx]==0)
      //	printf("O");
      // else
      //	printf("X");
      printf("%7d ", buf[idx]);
    }
    printf("\n");
  }
}
} // namespace mhagprb_matrix
