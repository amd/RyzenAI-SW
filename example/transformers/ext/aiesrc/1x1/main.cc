#include "graph.h"
#include "data_helpers.h"

SingleCoreGraph g_compute_graph;

int main(void)
{
    //
    // Allocate memory for single subvolume
    //

    ParamSubv* gemm_params = (ParamSubv*) adf::GMIO::malloc(CORE_IN1_SIZE);
    uint16_t* matA = (uint16_t*) adf::GMIO::malloc(CORE_IN1_SIZE);
    CoreSubv* matB = (CoreSubv*) adf::GMIO::malloc(2 * sizeof(CoreSubv));
    float*    matC = (float*)    adf::GMIO::malloc(CORE_OUT_SIZE);

    //
    // Init inputs
    //

    gemm_params->inner_loop = 1;
    gemm_params->outer_loop = 1;
    gemm_params->kernel_rows = M_SUBV;

    srand(42);
    for (int i = 0; i < M_SUBV * K_SUBV; ++i) {
        matA[i] = rand_bfloat16();
    }

    for (int i = 0; i < N_SUBV; ++i) {
        matB[0].bias.coefs[i] = rand_bfloat16();
    }
    for (int i = 0; i < K_SUBV * N_SUBV / 2; ++i) {
        int x = rand_int4();
        int y = rand_int4();
        matB[1].wgts.quants[i] = pack_v2int4(x, y);
    }
    for (int i = 0; i < K_SUBV * N_SUBV / GRP_SIZE / 2; ++i) {
        int x = rand_int4();
        int y = rand_int4();
        matB[1].wgts.zeros[i] = pack_v2int4(x, y);
    }
    for (int i = 0; i < K_SUBV * N_SUBV / GRP_SIZE; ++i) {
        matB[1].wgts.scales[i] = rand_bfloat16();
    }

    //
    // Compute golden data
    //

    std::vector<float> matA_golden(M_SUBV * K_SUBV);
    std::vector<float> matB_golden(K_SUBV * N_SUBV);
    std::vector<float> matC_golden(M_SUBV * N_SUBV);
    
    // Upshift matA to fp32
    for (int i = 0; i < matA_golden.size(); ++i) {
        matA_golden.at(i) = bfloat16_to_float(matA[i]);
    }
    // Dequantize matB to fp32
    //      NOTE: We simulate the bfloat16 round to nearest even
    //            so that the CPU DI model is bit-accurate with
    //            the HW weight dequantization.
    for (int i = 0; i < K_SUBV; ++i) {
        for (int j = 0; j < N_SUBV; ++j) {
            v2int quants = unpack_v2int4(matB[1].wgts.quants[((i * N_SUBV) + j) / 2]);
            v2int zeros = unpack_v2int4(matB[1].wgts.zeros[(((i / GRP_SIZE) * N_SUBV) + j) / 2]);
            uint16_t scale = matB[1].wgts.scales[((i / GRP_SIZE) * N_SUBV) + j];
            int q = (j % 2 == 0) ? quants.x : quants.y;
            int z = (j % 2 == 0) ? zeros.x : zeros.y;
            float s = bfloat16_to_float(scale);
            float w = (q - z) * s;
            matB_golden.at((i * N_SUBV) + j) = bfloat16_rnd_even(w);
        }
        
    }

    // Broadcast bias to matC
    for (int i = 0; i < M_SUBV; ++i) {
        for (int j = 0; j < N_SUBV; ++j) {
            matC_golden.at(i * N_SUBV + j) = bfloat16_to_float(matB[0].bias.coefs[j]);
        }
    }

    // Compute matC += matA * matB
    for (int r = 0; r < M_SUBV; ++r) {
        for (int c = 0; c < N_SUBV; ++c) {
            for (int k = 0; k < K_SUBV; ++k) {
                matC_golden.at(r * N_SUBV + c) +=
                    matA_golden.at(r * K_SUBV + k) * matB_golden.at(k * N_SUBV + c);
            }
        }
    }

    //
    // Run graph
    //

    g_compute_graph.init();
    g_compute_graph.run(1);
    g_compute_graph.gmio_in1.gm2aie_nb(gemm_params, CORE_IN1_SIZE);
    g_compute_graph.gmio_in1.gm2aie_nb(matA, CORE_IN1_SIZE);
    g_compute_graph.gmio_in2.gm2aie_nb(matB, 2 * sizeof(CoreSubv));
    g_compute_graph.gmio_out.aie2gm_nb(matC, CORE_OUT_SIZE);
    g_compute_graph.gmio_in1.wait();
    g_compute_graph.gmio_in2.wait();
    g_compute_graph.gmio_out.wait();
    g_compute_graph.end();

    //
    // Log results
    //

    printf("matA = \n");
    for (int r = 0; r < M_SUBV; ++r) {
        for (int c = 0; c < K_SUBV; ++c) {
            printf("%f ", bfloat16_to_float(matA[r * K_SUBV + c]));
        }
        printf("\n");
    }
    printf("matB.quant = \n");
    for (int r = 0; r < K_SUBV; ++r) {
        for (int c = 0; c < N_SUBV / 2; ++c) {
            v2int v = unpack_v2int4(matB[1].wgts.quants[r * (N_SUBV / 2) + c]);
            printf("%d %d ", v.x, v.y);
        }
        printf("\n");
    }
    printf("matB.zero = \n");
    for (int i = 0; i < K_SUBV * N_SUBV / GRP_SIZE / 2; ++i) {
        v2int v = unpack_v2int4(matB[1].wgts.zeros[i]);
        printf("%d %d ", v.x, v.y);
    }
    printf("\n");
    printf("matB.scale = \n");
    for (int i = 0; i < K_SUBV * N_SUBV / GRP_SIZE; ++i) {
        printf("%f ", bfloat16_to_float(matB[1].wgts.scales[i]));
    }
    printf("\n");
    printf("matB_golden = \n");
    for (int r = 0; r < K_SUBV; ++r) {
        for (int c = 0; c < N_SUBV; ++c) {
            printf("%f ", matB_golden.at(r * N_SUBV + c));
        }
        printf("\n");
    }
    printf("matC = \n");
    for (int r = 0; r < M_SUBV; ++r) {
        for (int c = 0; c < N_SUBV; ++c) {
            printf("%f ", matC[r * N_SUBV + c]);
        }
        printf("\n");
    }
    printf("matC_golden = \n");
    for (int r = 0; r < M_SUBV; ++r) {
        for (int c = 0; c < N_SUBV; ++c) {
            printf("%f ", matC_golden.at(r * N_SUBV + c));
        }
        printf("\n");
    }

    //
    // Test DI
    //

    // NOTE: There can be slight differences in rounding, since AIE is doing
    //       fused multiply-accumulate instructions. This is why we allow
    //       for a small epsilon difference between AIE and the golden model.
    float const EPSILON = 1e-4;
    int fail = 0;
    for (int i = 0; i < M_SUBV * N_SUBV; ++i) {
        float diff = matC[i] - matC_golden.at(i);
        if (std::abs(diff) >= EPSILON) {
            fail = 1;
            printf("matC[%d]: Expected: %f, Received: %f\n", i, matC_golden.at(i), matC[i]);
        }
    }
    if (!fail) {
        printf("DI: PASS\n");
    } else {
        printf("DI: FAIL\n");
    }

    //
    // Report cycle count
    //

    // printf("GeMM Cycles = %lu\n", *((uint64_t*) matC));

    //
    // Free memory
    //

    adf::GMIO::free(gemm_params);
    adf::GMIO::free(matA);
    adf::GMIO::free(matB);
    adf::GMIO::free(matC);

    return fail;
}
