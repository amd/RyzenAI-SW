#include "kernels.h"

extern int8_t buf_wgt1[CORE_WGT_SIZE / 2];
extern int8_t buf_wgt2[CORE_WGT_SIZE / 2];

void printbuf(v32bfloat16 buf){
    v32accfloat buf_fp = ups(buf); 
    float* elem = (float*)(&buf_fp);
    for(int i=0; i<32; i++){
        printf("%f ", *(elem+i));
    }
    printf("\n");
}
//
// This GeMM computes
//
//      matC = matA * matB + matC
//
// where matB is interleaved between two banks as matB1, matB2
// according to the format specified in dequantize_weights_32.
// The matrices matA and matC are expected to be in row-major
// order. The matrix matB is also effectively row major,
// except for the bank interleaving scheme.
//
// NOTE: This kernel uses a linear algebra trick that the product
//       matA * matB can be expressed as a linear combination of
//       the row vectors of matB with scalars from matA. This
//       factorization is required because AIE 2p intrinsics only
//       support elementwise bfloat16 MAC operations.
//
//       This kernel is unrolled in a very specific way
//       to achieve full vector pipelining in the inner loop.
//
//       The inner loop computes a 2x2 grid of vector results
//
//          concat(a0, a1) * concat(b0, b1) | concat(a0, a1) * concat(b2, b3)
//          --------------------------------+--------------------------------
//          concat(a2, a3) * concat(b0, b1) | concat(a2, a3) * concat(b2, b3)
//
//       where a0, a1, a2, a3 are broadcasted scalars and b0, b1, b2, b3 are
//       vectors loaded from L1. Each vector is re-used twice so that the
//       load and broadcast instructions fit in the same slots as the four MACs
//       without any pipeline bubbles. The loads for b0/b1 and b2/b3 must be
//       from different banks so that they can be scheduled in the same VLIW
//       slot. Otherwise, the read from the next loop iteration will cause a
//       pipeline hazard, forcing the compiler to insert a NOP.
//


template<int Kt = 32>
void gemm_MxK_Kx128_bf16(
    int8_t* matA,
    int8_t* matB1,
    int8_t* matB2,
    int8_t* matC,
    int num_rows)
{
    v32bfloat16* v_matA  = (v32bfloat16*) matA;
    v32bfloat16* v_matB1 = (v32bfloat16*) matB1;
    v32bfloat16* v_matB2 = (v32bfloat16*) matB2;
    v64accfloat* v_matC  = (v64accfloat*) matC;

    v64accfloat acc0;
    v64accfloat acc1;
    v64accfloat acc2;
    v64accfloat acc3;

    v32bfloat16 v0;
    v32bfloat16 v1;

    v32bfloat16 a0;
    v32bfloat16 a1;
    v32bfloat16 a2;
    v32bfloat16 a3;

    v32bfloat16 b0;
    v32bfloat16 b1;
    v32bfloat16 b2;
    v32bfloat16 b3;
    // tile for a1 in matA
    int row_pitch = Kt/32;


    for (int i = 0; i < num_rows; i += 2) {

        int e = 0;

        acc0 = v_matC[2*i + 0];
        acc1 = v_matC[2*i + 1];
        acc2 = v_matC[2*i + 2];
        acc3 = v_matC[2*i + 3];
        int row_id_n_0 = i*row_pitch;
        int row_id_n_1 = (i+1)*row_pitch; 
        
        for(int r=0; r<row_pitch; r++){

            v0 = v_matA[ row_id_n_0 + r];
            v1 = v_matA[ row_id_n_1 + r];
            int b_row = r*64;

            for (int k = 0; k < 64; k += 2) {

                a0 = broadcast_to_v32bfloat16(ext_elem(v0, e));
                a1 = broadcast_to_v32bfloat16(ext_elem(v0, e));
                a2 = broadcast_to_v32bfloat16(ext_elem(v1, e));
                a3 = broadcast_to_v32bfloat16(ext_elem(v1, e));
                e += 1;

                b0 = v_matB1[b_row + k];
                b1 = v_matB2[b_row + k];
                b2 = v_matB1[b_row + k + 1];
                b3 = v_matB2[b_row + k + 1];

                acc0 = mac_elem_64(concat(a0, a1), concat(b0, b1), acc0);
                acc1 = mac_elem_64(concat(a0, a1), concat(b2, b3), acc1);
                acc2 = mac_elem_64(concat(a2, a3), concat(b0, b1), acc2);
                acc3 = mac_elem_64(concat(a2, a3), concat(b2, b3), acc3);
            }
        }

        v_matC[2*i + 0] = acc0;
        v_matC[2*i + 1] = acc1;
        v_matC[2*i + 2] = acc2;
        v_matC[2*i + 3] = acc3;
    }
}

//
// This dequantization computes
//
//      w = (q - z) * s
//
// where q is a vector of int4 quantized weights, z is
// an int4 zero point, and s is a bfloat16 scaling factor.
// The same zero point and scaling factor are shared across
// groups of 32 weights.
//
// Group vectors are stored into dst1 and dst2. They are interleaved
// so that every even group is stored in dst1 and every odd group
// is stored in dst2.
//
// The GeMM kernel requires this interleaved format to avoid
// bank conflicts when loading from L1 memory.
//
// NOTE: The inner loop of the dequantization is very carefully
//       unrolled and vectorized to use the full pipeline vector width.
//       This results in an over 4x speedup compared to just using
//       the templated APIs with the group size as vector width.
//       This speedup is required to get full bandwidth utilization during
//       GeMMs with a small M dimension.
//
template <int Kt = 32>
void dequant_weights_Kx128(int8_t* src, int8_t* dst1, int8_t* dst2, int sign)
{
    // Unpack subvolume components
    auto subv   = (CoreSubv*)    src;
    auto quants = (v64int4*)     subv->wgts.quants;
    auto zeros  = (v64int4*)     subv->wgts.zeros;
    auto scales = (v64bfloat16*) subv->wgts.scales;
    auto wgts1  = (v32bfloat16*) dst1;
    auto wgts2  = (v32bfloat16*) dst2;

    v64int8 z0 = unpack(zeros[0], sign);
    v64int8 z1 = unpack(zeros[1], sign);

    v64bfloat16 s0 = scales[0];
    v64bfloat16 s1 = scales[1];

    for (int i = 0; i < Kt; ++i) {
        // Compute first 64 elements of row
        {
            v64int8     quant = unpack(*quants++, sign);
            v64int8     diff  = sub(quant, z0);
            v64bfloat16 bf    = aie::to_float<bfloat16>(aie::vector<int8, 64>(diff));
            v64accfloat wgt   = mul_elem_64(bf, s0);
            *wgts1++ = srs_to_v32bfloat16(extract_v32accfloat(wgt, 0));
            *wgts2++ = srs_to_v32bfloat16(extract_v32accfloat(wgt, 1));
        }

        // Compute next 64 elements of row
        {
            v64int8     quant = unpack(*quants++, sign);
            v64int8     diff  = sub(quant, z1);
            v64bfloat16 bf    = aie::to_float<bfloat16>(aie::vector<int8, 64>(diff));
            v64accfloat wgt   = mul_elem_64(bf, s1);
            *wgts1++ = srs_to_v32bfloat16(extract_v32accfloat(wgt, 0));
            *wgts2++ = srs_to_v32bfloat16(extract_v32accfloat(wgt, 1));
        }
    }
}

//
// Broadcast and upshift the bias coefficients in src
// to the accumulator buffer located at dst.
//
void init_acc_bias(int8_t* dst, int8_t* src)
{
    static_assert(N_SUBV % 32 == 0, "Invalid vectorization!");
    CoreSubv* subv = (CoreSubv*) src;
    v32bfloat16* v_src = (v32bfloat16*) subv->bias.coefs;
    v32accfloat* v_dst = (v32accfloat*) dst;
    for (int i = 0; i < M_SUBV; ++i) {
        for (int j = 0; j < N_SUBV / 32; ++j) {
            *v_dst++ = ups_to_v32accfloat(v_src[j]);
        }
    }
}

//
// Copy data from src to dst assuming 64 byte alignment
// and no overlap between buffers.
//
void vec_copy(int8_t* dst, int8_t* src, int len)
{
    v64int8* v_dst = (v64int8*) dst;
    v64int8* v_src = (v64int8*) src;
    for (int i = 0; i < len / 64; ++i) {
        v_dst[i] = v_src[i];
    }
}

// function pointer to select different GeMM functions
void (*gemm)(int8_t* ,int8_t*, int8_t*, int8_t*, int);

void (*dequant_weights)(int8_t*, int8_t*, int8_t*, int);

void gemm_wrapper(
    adf::input_async_buffer<int8_t>& buf_in1,
    adf::input_async_buffer<int8_t>& buf_in2,
    adf::output_async_buffer<int8_t>& buf_out)
{
    // NOTE: This rounding mode will control how weights are rounded
    //       from fp32 --> bfloat16 after dequantization is calculated.
    //       We choose round to nearest even, since this matches
    //       the behavior of numpy.
    set_rnd(rnd_conv_even);

    int outer_loop, inner_loop, kernel_rows, group_size, sign;
    {
        buf_in1.acquire();
        ParamSubv* params = (ParamSubv*) buf_in1.data();
        outer_loop  = params->outer_loop;
        inner_loop  = params->inner_loop;
        kernel_rows = params->kernel_rows;
        group_size  = params->group_size;
        sign        = params->sign;
        buf_in1.release();
    }
    if (group_size == 32){
        gemm = gemm_MxK_Kx128_bf16<32>;
        dequant_weights = dequant_weights_Kx128<32>;
    }else if(group_size == 128){
        gemm = gemm_MxK_Kx128_bf16<128>;
        dequant_weights = dequant_weights_Kx128<128>;
    }else{
        RUNTIME_ASSERT(false, "grp not supported");
    }
    for (int outer = 0; outer < outer_loop; ++outer) {
        buf_out.acquire();
        int8_t* buf_out_p = buf_out.data();
        {
            buf_in2.acquire();
            init_acc_bias(buf_out.data(), buf_in2.data());
            buf_in2.release();
        }
        uint64_t start, end;
        for (int inner = 0; inner < inner_loop; ++inner) {
            buf_in1.acquire();
            buf_in2.acquire();
            start = get_cycles();
            dequant_weights(buf_in2.data(), buf_wgt1, buf_wgt2, sign);
            gemm(buf_in1.data(),
                 buf_wgt1,
                 buf_wgt2,
                 buf_out_p,
                 kernel_rows);
            end = get_cycles();
            buf_in2.release();
            buf_in1.release();
        }
        {
            //vec_copy(buf_out_p, buf_acc, kernel_rows * N_SUBV * sizeof(float));
            // *((int64_t*) buf_out.data()) = end - start;
            buf_out.release();
        }
    }
}
