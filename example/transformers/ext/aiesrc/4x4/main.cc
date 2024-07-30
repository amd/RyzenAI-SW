#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#define TXN_FLOW 1

#include "graph.h"
#include "bd_config.h"
#include "config.h"
#include "subv_formatting.h"
#include "data_helpers.h"
#include "matrix_formatting.h"

std::string OUTPUT_FOLDER = "hw_package/workspace";
std::string matA_FILE = "/ifm.txt";
std::string matB_FILE = "/param.txt";
std::string matC_FILE = "/ofm.txt";
std::string txn_FILE = "/txn.bin";

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// NOTE: THe following changes are for TXN generation
#if TXN_FLOW == 1
    #ifdef __AIESIM__ 
        #include "xaiengine.h"
        extern XAie_DevInst DevInst;
        int32_t patch_op_code = XAIE_IO_CUSTOM_OP_DDR_PATCH;
    #endif
#endif

typedef struct{
    int temp;
    unsigned int size_in_bytes;
} op_base;
struct PatchShimOp {
    op_base op;
    uint32_t action;
    uint64_t regAddr;
    uint64_t extBufId;
    uint64_t argplus;
};
uint64_t ComputePatchRegAddr(uint8_t shimColumn, uint8_t bdId); 
void AddDDRCustomOp(uint32_t action, uint64_t bufId, uint64_t offsetInBytes, uint8_t shimCol, uint8_t bdId, int32_t patch_op_code);


#if TXN_FLOW == 1
    #ifdef __AIESIM__
        uint64_t ComputePatchRegAddr(uint8_t shimColumn, uint8_t bdId) {
            // get tile info
            XAie_LocType tileLoc = XAie_TileLoc(shimColumn, 0); //XAie_TileLoc(Col, Row)
            XAie_DmaDesc dmaInst;
            int driverStatus = XAie_DmaDescInit(&DevInst, &dmaInst, tileLoc);
            if (driverStatus != AieRC::XAIE_OK)
              std::cout << "########## ComputePatchRegAddr: AIE Driver Error" << std::endl;
            
            // compute reg addr
            //https://gitenterprise.xilinx.com/ai-engine/aie-rt/blob/main/driver/src/dma/xaie_dma_aieml.c#L1308
            // _XAie_GetTileAddr: https://gitenterprise.xilinx.com/ai-engine/aie-rt/blob/main/driver/src/common/xaie_helper.h
            u64 regAddr = dmaInst.DmaMod->BaseAddr 
              + bdId * dmaInst.DmaMod->IdxOffset 
              + _XAie_GetTileAddr(&DevInst, tileLoc.Row, tileLoc.Col) 
              + dmaInst.DmaMod->BdProp->Buffer->ShimDmaBuff.AddrLow.Idx * 4U;
            
            
            return regAddr;
        }
        
        void AddDDRCustomOp(uint32_t action, uint64_t bufId, uint64_t offsetInBytes, uint8_t shimCol, uint8_t bdId, int32_t patch_op_code)
        {
            PatchShimOp op;
            op.action = action;
            op.regAddr = ComputePatchRegAddr(shimCol, bdId);
            op.extBufId = bufId;
            op.argplus = offsetInBytes;
            XAie_AddCustomTxnOp(&DevInst, patch_op_code, (void*)&op, sizeof(op));
        }
    #endif
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif

ComputeGraph g_compute_graph;

template<typename inT>
void write_mat_to_file(std::string fileName, inT* buf, const size_t bufSize)
{
    std::ofstream ofs(fileName.c_str());
    if (!ofs) {
        std::cerr << "Error writing file " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }
    for (size_t i=0; i<bufSize; ++i) {
        ofs << buf[i] << std::endl;
    }
    ofs.close();
}

int main(void)
{
    // GeMM shape
    int const Mgemm = 1;
    int const Kgemm = 4096;
    int const Ngemm = 4096;
    bool ShareIfmChannel = false;
    if (Mgemm == 1) {
        ShareIfmChannel = true;
    } else {
        ShareIfmChannel = false;
    }
    std::string runtime_txn_FILE = "/a16fw4acc32f_"+std::to_string(Mgemm)+"_"+std::to_string(Kgemm)+"_"+std::to_string(Ngemm)+"_"+std::to_string(GRP_SIZE)+".bin";

    // NOTE: The Data in1 size is the unpadded subvol dimension from L3->L2
    //  Has to be the minimum of the Mgemm (for token phase) and M_SUBV (for prefill phase)
    int const unpadded_M_SUBV = (Mgemm > M_SUBV) ? M_SUBV : Mgemm;
    int const Outer_M_loop  = (Mgemm > M_SUBV) ? (Mgemm / M_SUBV) : 1;
    int const Outer_N_loop  = Ngemm / (NUM_ROWS * NUM_COLS * N_SUBV);
    int const InnerLoop  = Kgemm / K_SUBV;
    int const KernelRows = (Mgemm > M_SUBV) ? M_SUBV : Mgemm;

    //assert(Mgemm <= M_SUBV);
    assert(Kgemm % K_SUBV == 0);
    assert(Ngemm % (NUM_ROWS * NUM_COLS * N_SUBV) == 0);

    // Allocate memory

    ActMatrix<uint16_t> dev_A(Mgemm, Kgemm, 1, 1);
    QuantMatrix         dev_B(Kgemm, Ngemm);
    ActMatrix<float>    dev_C(Mgemm, Ngemm, 1, 1);
    ActMatrix<float>    cpu_C(Mgemm, Ngemm, 1, 1);
    ParamSubv*          dev_params = nullptr;

    int in1_size = dev_A.data_size + CORE_IN1_SIZE;
    int in2_size = dev_B.data_size;
    int out_size = dev_C.data_size;

    int8_t* dev_in1 = (int8_t*) adf::GMIO::malloc(in1_size);
    int8_t* dev_in2 = (int8_t*) adf::GMIO::malloc(in2_size);
    int8_t* dev_out = (int8_t*) adf::GMIO::malloc(out_size);
    int8_t* cpu_out = (int8_t*) malloc(out_size);

    dev_params = (ParamSubv*) &dev_in1[dev_A.data_size];
    dev_A.data = (uint16_t*)  &dev_in1[0];
    dev_B.data = (CoreSubv*)  &dev_in2[0];
    dev_C.data = (float*)     &dev_out[0];
    cpu_C.data = (float*)     &cpu_out[0];

    // Initialize inputs
    printf("==========================================\n");
    printf("Mgemm = %d \n", Mgemm);
    printf("Kgemm = %d \n", Kgemm);
    printf("Ngemm = %d \n", Ngemm);
    printf("M_SUBV = %d \n", M_SUBV);
    printf("K_SUBV = %d \n", K_SUBV);
    printf("N_SUBV = %d \n", N_SUBV);
    printf("GRP_SIZE = %d \n", GRP_SIZE);
    printf("CORE_IN1_SIZE = %d \n", CORE_IN1_SIZE);
    printf("CORE_IN2_SIZE = %d \n", CORE_IN2_SIZE);
    printf("CORE_OUT_SIZE = %d \n", CORE_OUT_SIZE);
    printf("CORE_WGT_SIZE = %d \n", CORE_WGT_SIZE);
    printf("CORE_ACC_SIZE = %d \n", CORE_ACC_SIZE);
    printf("CORE_IN1_PING_ADDR = %d \n", CORE_IN1_PING_ADDR);
    printf("CORE_IN1_PONG_ADDR = %d \n", CORE_IN1_PONG_ADDR);
    printf("CORE_IN2_PING_ADDR = %d \n", CORE_IN2_PING_ADDR);
    printf("CORE_IN2_PONG_ADDR = %d \n", CORE_IN2_PONG_ADDR);
    printf("CORE_OUT_PING_ADDR = %d \n", CORE_OUT_PING_ADDR);
    printf("CORE_WGT1_ADDR = %d \n", CORE_WGT1_ADDR);
    printf("CORE_WGT2_ADDR = %d \n", CORE_WGT2_ADDR);
    printf("outer_loop = %d \n", Outer_M_loop * Outer_N_loop);
    printf("inner_loop = %d \n", InnerLoop);
    printf("Channel sharing = %d \n", ShareIfmChannel);
    printf("==========================================\n");

    dev_params->outer_loop = Outer_M_loop * Outer_N_loop;
    dev_params->inner_loop = InnerLoop;
    dev_params->kernel_rows = KernelRows;
    dev_params->group_size = GRP_SIZE;
    dev_params->sign = 1;

    for (int i = 0; i < dev_A.num_rows; ++i) {
        for (int j = 0; j < dev_A.num_cols; ++j) {
            dev_A.act(i, j) = rand_bfloat16();
        }
    }

    for (int i = 0; i < dev_B.num_cols; ++i) {
        dev_B.bias(i) = rand_bfloat16();
    }
    for (int i = 0; i < dev_B.num_rows; ++i) {
        for (int j = 0; j < dev_B.num_cols; j += 2) {
            int x = rand_int4();
            int y = rand_int4();
            dev_B.quant(i, j) = pack_v2int4(x, y);
        }
    }
    for (int i = 0; i < dev_B.num_rows; i += GRP_SIZE) {
        for (int j = 0; j < dev_B.num_cols; j += 2) {
            int x = rand_int4();
            int y = rand_int4();
            dev_B.zero(i, j) = pack_v2int4(x, y);
        }
    }
    for (int i = 0; i < dev_B.num_rows; i += GRP_SIZE) {
        for (int j = 0; j < dev_B.num_cols; ++j) {
            dev_B.scale(i, j) = rand_bfloat16();
        }
    }

    // Compute golden

    for (int i = 0; i < cpu_C.num_rows; ++i) {
        for (int j = 0; j < cpu_C.num_cols; ++j) {
            cpu_C.act(i, j) = bfloat16_to_float(dev_B.bias(j));
            for (int k = 0; k < dev_A.num_cols; ++k) {
                float a = bfloat16_to_float(dev_A.act(i, k));
                float b = dev_B.weight(k, j);
                cpu_C.act(i, j) += a * b;
            }
        }
    }

    // Run graph

    g_compute_graph.init();
    g_compute_graph.run(1);
    #if TXN_FLOW == 1
        #ifdef __AIESIM__
            printf("Start Transaction\n");
            XAie_StartTransaction(&DevInst, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
            printf("DevInst->\n");
            XAie_LocType shimdma[NUM_COLS];
            XAie_LocType memtiledma[NUM_COLS];
            // Create Shim tile location instances
            for (int col = 0; col < NUM_COLS; ++col) 
            {
                shimdma[col] = XAie_TileLoc(col, 0);  // XAie_TileLoc(u8 col, u8 row);
                memtiledma[col] = XAie_TileLoc(col, 1);  // XAie_TileLoc(u8 col, u8 row);
            }
        #endif
    #endif
    // Dump the start offset of each shim BD to a txt to be consumed by the DPU sequence generator 
    std::ofstream bo_offset_file("HW_BO_offsets.txt");
    int const DATA_IN1_SIZE = (unpadded_M_SUBV * K_SUBV * 2);

    adf::dma_buffer_descriptor bd_prm;
    adf::dma_buffer_descriptor bd_ifm;
    adf::dma_buffer_descriptor bd_wgt;
    adf::dma_buffer_descriptor bd_wgt1;
    adf::dma_buffer_descriptor bd_wgt2;
    adf::dma_buffer_descriptor bd_ofm;
    if (!ShareIfmChannel) {
        
        for (int col = 0; col < NUM_COLS; ++col) {
            // NOTE: Address offset overide in DPU sequence has to be in Bytes
            
            bd_prm.address  = (dev_A.data_size / sizeof(int32_t));
            bd_prm.length   = (DATA_IN1_SIZE / sizeof(int32_t));
            bd_prm.stepsize = {1};
            bd_prm.wrap     = {};
            bo_offset_file << "instr_" << col << "=0x" << std::hex << (bd_prm.address * sizeof(int32_t)) << std::endl;

            
            bd_ifm.address  = 0;
            bd_ifm.length   = (dev_A.data_size / sizeof(int32_t));
            bd_ifm.stepsize = {1};
            bd_ifm.wrap     = {};
            bo_offset_file << "matA_" << col << "=0x" << std::hex << (bd_ifm.address * sizeof(int32_t)) << std::endl;

            #if TXN_FLOW == 1
                #ifdef __AIESIM__
                    // NOTE: Assert and deassert the reset bit for all DMAs of each mem tile
                    XAie_DmaChannelResetAll(&DevInst, memtiledma[col], 1);                                                          // XAie_DmaChannelResetAll(XAie_DevInst *DevInst, XAie_LocType Loc, XAie_DmaChReset Reset)
                    XAie_DmaChannelResetAll(&DevInst, memtiledma[col], 0);
                    // NOTE: All the address and length values are specified in terms of Bytes, the driver API converts them the 32-bit words
                    // INSTR BD
                    XAie_DmaDesc prm_desc;
                    XAie_DmaDescInit(&DevInst, &prm_desc, shimdma[col]);                                                            // XAie_DmaChannelDescInit(XAie_DevInst *DevInst, XAie_DmaChannelDesc *DmaChannelDesc, XAie_LocType Loc);
                    XAie_DmaSetAddrLen(&prm_desc, (u64)(bd_prm.address * sizeof(int32_t)), (bd_prm.length * sizeof(int32_t)));      // XAie_DmaSetAddrLen(XAie_DmaDesc *DmaDesc, u64 Addr, u32 Len);
                    XAie_DmaSetAxi(&prm_desc, 0U, 32U, 0U, 2U, 0U);                                                                 // XAie_DmaSetAxi(XAie_DmaDesc *DmaDesc, u8 Smid, u8 BurstLen, u8 Qos, u8 Cache, u8 Secure);
                    // Work around to manually  override the burst length CR-1188249
                    prm_desc.AxiDesc.BurstLen = 3;
                    prm_desc.AxiDesc.AxCache = 2;
                    XAie_DmaEnableBd(&prm_desc);
                    XAie_DmaWriteBd(&DevInst, &prm_desc, shimdma[col], (u8)3);                                                      // XAie_DmaWriteBd(XAie_DevInst *DevInst, XAie_DmaDesc *DmaDesc, XAie_LocType Loc, u8 BdNum);
                    AddDDRCustomOp(0, 1, (bd_prm.address * sizeof(int32_t)), col, 3, patch_op_code);
                    XAie_DmaChannelSetStartQueue(&DevInst, shimdma[col], (u8)0, DMA_MM2S, (u8)3, 1, 0);                             // XAie_DmaChannelSetStartQueue(XAie_DevInst *DevInst, XAie_LocType Loc, u8 ChNum, XAie_DmaDirection Dir, u8 BdNum, u32 RepeatCount, u8 EnTokenIssue)
                    // IFM BD
                    XAie_DmaDesc ifm_desc;
                    XAie_DmaDescInit(&DevInst, &ifm_desc, shimdma[col]);
                    XAie_DmaSetAddrLen(&ifm_desc, (u64)(bd_ifm.address * sizeof(int32_t)), (bd_ifm.length * sizeof(int32_t)));
                    XAie_DmaSetAxi(&ifm_desc, 0U, 32U, 0U, 2U, 0U);
                    // Work around to manually  override the burst length CR-1188249
                    ifm_desc.AxiDesc.BurstLen = 3;
                    ifm_desc.AxiDesc.AxCache = 2;
                    XAie_DmaEnableBd(&ifm_desc);
                    XAie_DmaWriteBd(&DevInst, &ifm_desc, shimdma[col], (u8)0);
                    AddDDRCustomOp(0, 1, (bd_ifm.address * sizeof(int32_t)), col, 0, patch_op_code);
                    XAie_DmaChannelSetStartQueue(&DevInst, shimdma[col], (u8)0, DMA_MM2S, (u8)0, 1, 0);
                    // Enable the MM2s channel 0 of the shim DMA after all BDs are queued
                    // Enabled the shim MM2S channel 0
                    XAie_DmaChannelEnable(&DevInst, shimdma[col], (u8)0, DMA_MM2S);                                                 // XAie_DmaChannelEnable(XAie_DevInst *DevInst, XAie_LocType Loc, u8 ChNum, XAie_DmaDirection Dir)
                #endif
           #endif
        }
        // The repeat count is capped to 256 when there is no iterstep involved 
        int MAX_REPEAT_COUNT = 256;
        int WGT_REPEAT = Outer_M_loop > MAX_REPEAT_COUNT ? MAX_REPEAT_COUNT : Outer_M_loop; 
        int WGT_REENQUEUE = Outer_M_loop > MAX_REPEAT_COUNT ? Outer_M_loop/MAX_REPEAT_COUNT : 1; 
        if (WGT_REENQUEUE > 4){
            printf("MAX shim BD task queue depth is 4 !!");
            exit(0);
        }
        for (int col = 0; col < NUM_COLS; ++col) {
            
            bd_wgt.address  = ((dev_B.data_size / NUM_COLS) / sizeof(int32_t)) * col;
            bd_wgt.length   = ((dev_B.data_size / NUM_COLS) / sizeof(int32_t));
            bd_wgt.stepsize = {1};
            bd_wgt.wrap     = {};
            bo_offset_file << "matB_" << col << "=0x" << std::hex << (bd_wgt.address * sizeof(int32_t)) << std::endl;

            #if TXN_FLOW == 1
                #ifdef __AIESIM__
                    // Dedicated WGT BD
                    XAie_DmaDesc wgt_desc;
                    XAie_DmaDescInit(&DevInst, &wgt_desc, shimdma[col]);
                    XAie_DmaSetAddrLen(&wgt_desc, (u64)(bd_wgt.address * sizeof(int32_t)), (bd_wgt.length * sizeof(int32_t)));
                    XAie_DmaSetAxi(&wgt_desc, 0U, 32U, 0U, 2U, 0U);
                    // Work around to manually  override the burst length CR-1188249
                    wgt_desc.AxiDesc.BurstLen = 3;
                    wgt_desc.AxiDesc.AxCache = 2;
                    XAie_DmaEnableBd(&wgt_desc);
                    XAie_DmaWriteBd(&DevInst, &wgt_desc, shimdma[col], (u8)1);
                    AddDDRCustomOp(0, 2, (bd_wgt.address * sizeof(int32_t)), col, 1, patch_op_code);
                    for(int wgt_reenqueue = 0; wgt_reenqueue < WGT_REENQUEUE; wgt_reenqueue++) {
                        printf("Enqueuing WGT shim BD with %d repeat count \n", WGT_REPEAT);
                        XAie_DmaChannelSetStartQueue(&DevInst, shimdma[col], (u8)1, DMA_MM2S, (u8)1, WGT_REPEAT, 0);
                    }
                    XAie_DmaChannelEnable(&DevInst, shimdma[col], (u8)1, DMA_MM2S);
                #endif
            #endif
        }
        int MAX_ITER_COUNT = 64;
        int ITER_REPEAT = Outer_M_loop > MAX_ITER_COUNT ? MAX_ITER_COUNT : Outer_M_loop;
        int ofm_chain_count = Outer_M_loop > MAX_ITER_COUNT ? Outer_M_loop/MAX_ITER_COUNT : 1;
        for (int col = 0; col < NUM_COLS; ++col) {
            // Compute OFM access pattern
            int const OFM_D0_STEP = 1;
            int const OFM_D1_STEP = Ngemm;
            int const OFM_D2_STEP = (NUM_ROWS * N_SUBV);
            int const OFM_D0_WRAP = (NUM_ROWS * N_SUBV);
            int const OFM_D1_WRAP = M_SUBV;
            
            bd_ofm.address  = ((Ngemm / NUM_COLS) * col);
            bd_ofm.length   = ((M_SUBV * Ngemm) / NUM_COLS);
            bd_ofm.stepsize = {OFM_D0_STEP, OFM_D1_STEP, OFM_D2_STEP};
            bd_ofm.wrap     = {OFM_D0_WRAP, OFM_D1_WRAP};
            bd_ofm.iteration_stepsize = ((M_SUBV * Ngemm * 4) / sizeof(int32_t));
            bo_offset_file << "matC_" << col << "=0x" << std::hex << (bd_ofm.address * sizeof(int32_t)) << std::endl;

            #if TXN_FLOW == 1
                #ifdef __AIESIM__
                    // Step/wrap of OFM matrix per column 
                    XAie_DmaTensor ofm_tensor;
                    XAie_DmaDimDesc ofm_dim[3];
                    ofm_dim[0].AieMlDimDesc.StepSize = OFM_D0_STEP;
                    ofm_dim[1].AieMlDimDesc.StepSize = OFM_D1_STEP;
                    ofm_dim[2].AieMlDimDesc.StepSize = OFM_D2_STEP;
                    ofm_dim[0].AieMlDimDesc.Wrap = OFM_D0_WRAP;
                    ofm_dim[1].AieMlDimDesc.Wrap = OFM_D1_WRAP;
                    ofm_tensor.NumDim = 3U;
                    ofm_tensor.Dim = ofm_dim;
                    // If the Outer_M_loop is larger than the max repeat count of the BD (64) then create a chain of BDs
                    // else set the chain to just 1 BD
                    // The repeat count is capped to 64 since the iterstep_current fields of the BD is only 6 bit wide
                    for(int ofm_bd_offset = 0; ofm_bd_offset < ofm_chain_count; ofm_bd_offset++) {
                        // OFM BD
                        XAie_DmaDesc ofm_desc;
                        XAie_DmaDescInit(&DevInst, &ofm_desc, shimdma[col]);
                        XAie_DmaSetMultiDimAddr(&ofm_desc, &ofm_tensor, (u64)( (bd_ofm.address * sizeof(int32_t)) + (ofm_bd_offset*M_SUBV*Ngemm*sizeof(int32_t)) ) , (bd_ofm.length * sizeof(int32_t)));
                        if (ofm_bd_offset != (ofm_chain_count) - 1) {
                            printf("Col %d chaining OFM BD %d to BD %d \n", col, 2+(ofm_bd_offset), 3+(ofm_bd_offset));
                            XAie_DmaSetNextBd(&ofm_desc, (u8)3+(ofm_bd_offset), (u8)1);
                        }
                        XAie_DmaSetBdIteration(&ofm_desc, (ofm_chain_count*M_SUBV*Ngemm), 0U, 0U);
                        XAie_DmaSetAxi(&ofm_desc, 0U, 32U, 0U, 2U, 0U);
                        // Work around to manually  override the burst length CR-1188249
                        ofm_desc.AxiDesc.BurstLen = 3;
                        ofm_desc.AxiDesc.AxCache = 2;
                        XAie_DmaEnableBd(&ofm_desc);
                        XAie_DmaWriteBd(&DevInst, &ofm_desc, shimdma[col], (u8)2+(ofm_bd_offset));
                        AddDDRCustomOp(0, 0, ( (bd_ofm.address * sizeof(int32_t)) + (ofm_bd_offset*M_SUBV*Ngemm*sizeof(int32_t))), col, 2+(ofm_bd_offset), patch_op_code);
                    }
                    XAie_DmaChannelSetStartQueue(&DevInst, shimdma[col], (u8)0, DMA_S2MM, (u8)2, ITER_REPEAT, 0);
                    XAie_DmaChannelEnable(&DevInst, shimdma[col], (u8)0, DMA_S2MM);
                #endif
            #endif
        }
        #if TXN_FLOW != 1
            for (int col = 0; col < NUM_COLS; ++col) {
                g_compute_graph.gmio_in1[col].gm2aie_nb(dev_in1, {bd_prm}, {3}, 1);
                g_compute_graph.gmio_in1[col].gm2aie_nb(dev_in1, {bd_ifm}, {0}, 1);
                g_compute_graph.gmio_in2[col].gm2aie_nb(dev_in2, {bd_wgt}, {1}, Outer_M_loop);
                g_compute_graph.gmio_out[col].aie2gm_nb(dev_out, {bd_ofm}, {2}, Outer_M_loop);
            }
        #endif
    } else {
       for (int col = 0; col < NUM_COLS; ++col) {
            // NOTE: Address offset overide in DPU sequence has to be in Bytes
            bd_prm.address  = (dev_A.data_size / sizeof(int32_t));
            bd_prm.length   = (DATA_IN1_SIZE / sizeof(int32_t));
            bd_prm.stepsize = {1};
            bd_prm.wrap     = {};
            bo_offset_file << "instr_" << col << "=0x" << std::hex << (bd_prm.address * sizeof(int32_t)) << std::endl;

            bd_ifm.address  = 0;
            bd_ifm.length   = (dev_A.data_size / sizeof(int32_t));
            bd_ifm.stepsize = {1};
            bd_ifm.wrap     = {};
            bo_offset_file << "matA_" << col << "=0x" << std::hex << (bd_ifm.address * sizeof(int32_t)) << std::endl;

            // Compute shared weights access pattern
            int const WRAP_DIV = 4;
            int const DATA_IN2_WORDS = DATA_IN2_SIZE / sizeof(int32_t);
            int const D0_STEP = 1;
            int const D1_STEP = DATA_IN2_WORDS / WRAP_DIV;
            int const D2_STEP = DATA_IN2_WORDS * NUM_ROWS;
            int const D0_WRAP = DATA_IN2_WORDS / WRAP_DIV;
            int const D1_WRAP = NUM_ROWS / 2 * WRAP_DIV;
            int const WGT2_OFFSET = DATA_IN2_WORDS * (NUM_ROWS / 2);

            assert(D0_WRAP < 1024);
            assert(D0_WRAP == D1_STEP);
            assert(D0_WRAP * D1_WRAP == 2 * DATA_IN2_WORDS);

            bd_wgt1.address  = (((dev_B.data_size / NUM_COLS) / sizeof(int32_t)) * col);
            bd_wgt1.length   = (((dev_B.data_size / NUM_COLS) / sizeof(int32_t)) / 2);
            bd_wgt1.stepsize = {D0_STEP, D1_STEP, D2_STEP};
            bd_wgt1.wrap     = {D0_WRAP, D1_WRAP};
            bo_offset_file << "matB2_" << col << "=0x" << std::hex << (bd_wgt1.address * sizeof(int32_t)) << std::endl;

            bd_wgt2.address  = (((dev_B.data_size / NUM_COLS) / sizeof(int32_t)) * col) + WGT2_OFFSET;
            bd_wgt2.length   = (((dev_B.data_size / NUM_COLS) / sizeof(int32_t)) / 2);
            bd_wgt2.stepsize = {D0_STEP, D1_STEP, D2_STEP};
            bd_wgt2.wrap     = {D0_WRAP, D1_WRAP};
            bo_offset_file << "matB_" << col << "=0x" << std::hex << (bd_wgt2.address * sizeof(int32_t)) << std::endl;

            bd_ofm.address  = ((dev_C.data_size / NUM_COLS) / sizeof(int32_t)) * col;
            bd_ofm.length   = ((dev_C.data_size / NUM_COLS) / sizeof(int32_t));
            bd_ofm.stepsize = {1, };
            bd_ofm.wrap     = {};
            bo_offset_file << "matC_" << col << "=0x" << std::hex << (bd_ofm.address * sizeof(int32_t)) << std::endl;

            #if TXN_FLOW == 1
                #ifdef __AIESIM__
                    // NOTE: Assert and deassert the reset bit for all DMAs of each mem tile
                    XAie_DmaChannelResetAll(&DevInst, memtiledma[col], 1);                                                          // XAie_DmaChannelResetAll(XAie_DevInst *DevInst, XAie_LocType Loc, XAie_DmaChReset Reset)
                    XAie_DmaChannelResetAll(&DevInst, memtiledma[col], 0);
                    // NOTE: All the address and length values are specified in terms of Bytes, the driver API converts them the 32-bit words
                    // INSTR BD
                    XAie_DmaDesc prm_desc;
                    XAie_DmaDescInit(&DevInst, &prm_desc, shimdma[col]);                                                            // XAie_DmaChannelDescInit(XAie_DevInst *DevInst, XAie_DmaChannelDesc *DmaChannelDesc, XAie_LocType Loc);
                    XAie_DmaSetAddrLen(&prm_desc, (u64)(bd_prm.address * sizeof(int32_t)), (bd_prm.length * sizeof(int32_t)));      // XAie_DmaSetAddrLen(XAie_DmaDesc *DmaDesc, u64 Addr, u32 Len);
                    XAie_DmaSetAxi(&prm_desc, 0U, 32U, 0U, 2U, 0U);                                                                 // XAie_DmaSetAxi(XAie_DmaDesc *DmaDesc, u8 Smid, u8 BurstLen, u8 Qos, u8 Cache, u8 Secure);
                    // Work around to manually  override the burst length CR-1188249
                    prm_desc.AxiDesc.BurstLen = 3;
                    prm_desc.AxiDesc.AxCache = 2;
                    XAie_DmaEnableBd(&prm_desc);
                    XAie_DmaWriteBd(&DevInst, &prm_desc, shimdma[col], (u8)3);                                                      // XAie_DmaWriteBd(XAie_DevInst *DevInst, XAie_DmaDesc *DmaDesc, XAie_LocType Loc, u8 BdNum);
                    AddDDRCustomOp(0, 1, (bd_prm.address * sizeof(int32_t)), col, 3, patch_op_code);
                    //XAie_DmaChannelPushBdToQueue(&DevInst, shimdma[col], (u8)0, DMA_MM2S, (u8)3);                                 // XAie_DmaChannelPushBdToQueue(XAie_DevInst *DevInst, XAie_LocType Loc, u8 ChNum, XAie_DmaDirection Dir, u8 BdNum)
                    // NOTE: Minimum repeat_count=1
                    XAie_DmaChannelSetStartQueue(&DevInst, shimdma[col], (u8)0, DMA_MM2S, (u8)3, 1, 0);                             // XAie_DmaChannelSetStartQueue(XAie_DevInst *DevInst, XAie_LocType Loc, u8 ChNum, XAie_DmaDirection Dir, u8 BdNum, u32 RepeatCount, u8 EnTokenIssue)
                    // IFM BD
                    XAie_DmaDesc ifm_desc;
                    XAie_DmaDescInit(&DevInst, &ifm_desc, shimdma[col]);
                    XAie_DmaSetAddrLen(&ifm_desc, (u64)(bd_ifm.address * sizeof(int32_t)), (bd_ifm.length * sizeof(int32_t)));
                    XAie_DmaSetAxi(&ifm_desc, 0U, 32U, 0U, 2U, 0U);
                    // Work around to manually  override the burst length CR-1188249
                    ifm_desc.AxiDesc.BurstLen = 3;
                    ifm_desc.AxiDesc.AxCache = 2;
                    XAie_DmaEnableBd(&ifm_desc);
                    XAie_DmaWriteBd(&DevInst, &ifm_desc, shimdma[col], (u8)0);
                    AddDDRCustomOp(0, 1, (bd_ifm.address * sizeof(int32_t)), col, 0, patch_op_code);
                    //XAie_DmaChannelPushBdToQueue(&DevInst, shimdma[col], (u8)0, DMA_MM2S, (u8)0);
                    XAie_DmaChannelSetStartQueue(&DevInst, shimdma[col], (u8)0, DMA_MM2S, (u8)0, 1, 0);
                    // Step/wrap of shared WGT per column
                    XAie_DmaTensor wgt_tensor;
                    XAie_DmaDimDesc wgt_dim[3];
                    wgt_dim[0].AieMlDimDesc.StepSize = D0_STEP;
                    wgt_dim[1].AieMlDimDesc.StepSize = D1_STEP;
                    wgt_dim[2].AieMlDimDesc.StepSize = D2_STEP;
                    wgt_dim[0].AieMlDimDesc.Wrap = D0_WRAP;
                    wgt_dim[1].AieMlDimDesc.Wrap = D1_WRAP;
                    wgt_tensor.NumDim = 3U;
                    wgt_tensor.Dim = wgt_dim;
                    // Shared WGT BD
                    XAie_DmaDesc wgt_shrd_desc;
                    XAie_DmaDescInit(&DevInst, &wgt_shrd_desc, shimdma[col]);
                    XAie_DmaSetMultiDimAddr(&wgt_shrd_desc, &wgt_tensor, (u64)(bd_wgt1.address * sizeof(int32_t)), (bd_wgt1.length * sizeof(int32_t)));
                    XAie_DmaSetAxi(&wgt_shrd_desc, 0U, 32U, 0U, 2U, 0U);
                    // Work around to manually  override the burst length CR-1188249
                    wgt_shrd_desc.AxiDesc.BurstLen = 3;
                    wgt_shrd_desc.AxiDesc.AxCache = 2;
                    XAie_DmaEnableBd(&wgt_shrd_desc);
                    XAie_DmaWriteBd(&DevInst, &wgt_shrd_desc, shimdma[col], (u8)4);
                    AddDDRCustomOp(0, 2, (bd_wgt1.address * sizeof(int32_t)), col, 4, patch_op_code);
                    //XAie_DmaChannelPushBdToQueue(&DevInst, shimdma[col], (u8)0, DMA_MM2S, (u8)4);
                    XAie_DmaChannelSetStartQueue(&DevInst, shimdma[col], (u8)0, DMA_MM2S, (u8)4, 1, 0);
                    // Enable the MM2s channel 0 of the shim DMA after all BDs are queued
                    // Enabled the shim MM2S channel 0
                    XAie_DmaChannelEnable(&DevInst, shimdma[col], (u8)0, DMA_MM2S);                                                 // XAie_DmaChannelEnable(XAie_DevInst *DevInst, XAie_LocType Loc, u8 ChNum, XAie_DmaDirection Dir)
                    // Dedicated WGT BD
                    XAie_DmaDesc wgt_desc;
                    XAie_DmaDescInit(&DevInst, &wgt_desc, shimdma[col]);
                    XAie_DmaSetMultiDimAddr(&wgt_desc, &wgt_tensor, (u64)(bd_wgt2.address * sizeof(int32_t)), (bd_wgt2.length * sizeof(int32_t)));
                    XAie_DmaSetAxi(&wgt_desc, 0U, 32U, 0U, 2U, 0U);
                    // Work around to manually  override the burst length CR-1188249
                    wgt_desc.AxiDesc.BurstLen = 3;
                    wgt_desc.AxiDesc.AxCache = 2;
                    XAie_DmaEnableBd(&wgt_desc);
                    XAie_DmaWriteBd(&DevInst, &wgt_desc, shimdma[col], (u8)1);
                    AddDDRCustomOp(0, 2, (bd_wgt2.address * sizeof(int32_t)), col, 1, patch_op_code);
                    //XAie_DmaChannelPushBdToQueue(&DevInst, shimdma[col], (u8)1, DMA_MM2S, (u8)1);
                    XAie_DmaChannelSetStartQueue(&DevInst, shimdma[col], (u8)1, DMA_MM2S, (u8)1, 1, 0);
                    XAie_DmaChannelEnable(&DevInst, shimdma[col], (u8)1, DMA_MM2S);
                    // OFM BD
                    XAie_DmaDesc ofm_desc;
                    XAie_DmaDescInit(&DevInst, &ofm_desc, shimdma[col]);
                    XAie_DmaSetAddrLen(&ofm_desc, (u64)(bd_ofm.address * sizeof(int32_t)), (bd_ofm.length * sizeof(int32_t)));
                    XAie_DmaSetAxi(&ofm_desc, 0U, 32U, 0U, 2U, 0U);
                    // Work around to manually  override the burst length CR-1188249
                    ofm_desc.AxiDesc.BurstLen = 3;
                    ofm_desc.AxiDesc.AxCache = 2;
                    XAie_DmaEnableBd(&ofm_desc);
                    XAie_DmaWriteBd(&DevInst, &ofm_desc, shimdma[col], (u8)2);
                    AddDDRCustomOp(0, 0, (bd_ofm.address * sizeof(int32_t)), col, 2, patch_op_code);
                    //XAie_DmaChannelPushBdToQueue(&DevInst, shimdma[col], (u8)0, DMA_S2MM, (u8)2);
                    XAie_DmaChannelSetStartQueue(&DevInst, shimdma[col], (u8)0, DMA_S2MM, (u8)2, 1, 0);
                    XAie_DmaChannelEnable(&DevInst, shimdma[col], (u8)0, DMA_S2MM);
                #endif
            #endif
            #if TXN_FLOW != 1
                g_compute_graph.gmio_in1[col].gm2aie_nb(dev_in1, {bd_prm},  {3}, 1);
                g_compute_graph.gmio_in1[col].gm2aie_nb(dev_in1, {bd_ifm},  {0}, 1);
                g_compute_graph.gmio_in1[col].gm2aie_nb(dev_in2, {bd_wgt1}, {4}, 1);
                g_compute_graph.gmio_in2[col].gm2aie_nb(dev_in2, {bd_wgt2}, {1}, 1);
                g_compute_graph.gmio_out[col].aie2gm_nb(dev_out, {bd_ofm},  {2}, 1);
            #endif
        }
    }
    bo_offset_file.close();
    {   // This section dumps IO data for HW validation
        std::cout << "Writing matrices to ./" << OUTPUT_FOLDER << std::endl;
        system(("mkdir -p "+OUTPUT_FOLDER).c_str());
        write_mat_to_file<uint16_t>(OUTPUT_FOLDER+matA_FILE, (uint16_t*)dev_in1, in1_size/sizeof(uint16_t));
        write_mat_to_file<uint16_t>(OUTPUT_FOLDER+matB_FILE, (uint16_t*)dev_in2, in2_size/sizeof(uint16_t));
        write_mat_to_file<float>(OUTPUT_FOLDER+matC_FILE, (float*)cpu_out, out_size/sizeof(float));
        // NOTE: Dump the No. of elements of each matrix
        //       THis is used by the XRT host app for memory allocation
        std::ofstream ddr_range(OUTPUT_FOLDER+"/ddr_range.txt");
        ddr_range  << "ifm_addr " << 0 << std::endl;
        ddr_range  << "ifm_size " << in1_size/sizeof(uint16_t) << std::endl;
        ddr_range  << "param_addr " << 0 << std::endl;
        ddr_range  << "param_size " << in2_size/sizeof(uint16_t) << std::endl;
        ddr_range  << "ofm_addr " << 0 << std::endl;
        ddr_range  << "ofm_size " << out_size/sizeof(float) << std::endl;
        ddr_range  << "inter_addr " << 0 << std::endl;
        ddr_range  << "inter_size " << 0 << std::endl;
        ddr_range.close();
    }
    run_host_bd_config();
    #if TXN_FLOW == 1
        #ifdef __AIESIM__
            for (int col = 0; col < NUM_COLS; ++col) {
                // Wait for DMA done on the output buffer indefninitely
                XAie_DmaWaitForDone(&DevInst, shimdma[col], 0, DMA_S2MM, 0);                    // XAie_DmaWaitForDone(XAie_DevInst *DevInst, XAie_LocType Loc, u8 ChNum, XAie_DmaDirection Dir, u32 TimeOutUs)
            }
            char *txn_ptr =  XAie_ExportSerializedTransaction(&DevInst, 0, 0);
            auto TxnInst =  XAie_ExportTransactionInstance(&DevInst);
            fstream txn_file (OUTPUT_FOLDER+txn_FILE, ios::out  | ios::binary | ios::app);
            fstream runtime_txn_file (OUTPUT_FOLDER+runtime_txn_FILE, ios::out  | ios::binary | ios::app);
            printf("Exported transaction\n");
            XAie_TxnHeader *TxnHeader = (XAie_TxnHeader *)txn_ptr;
            printf("TxnHeader->Major=%d \n", TxnHeader->Major);
            printf("TxnHeader->Minor=%d \n", TxnHeader->Minor);
            printf("TxnHeader->DevGen=%d \n", TxnHeader->DevGen);
            printf("TxnHeader->NumRows=%d \n", TxnHeader->NumRows);
            printf("TxnHeader->NumCols=%d \n", TxnHeader->NumCols);
            printf("TxnHeader->NumMemTileRows=%d \n", TxnHeader->NumMemTileRows);
            printf("TxnHeader->NumOps=%d \n", TxnHeader->NumOps);
            printf("TxnHeader->TxnSize=%d \n", TxnHeader->TxnSize);
            txn_file.write(txn_ptr, TxnHeader->TxnSize);
            txn_file.close();
            runtime_txn_file.write(txn_ptr, TxnHeader->TxnSize);
            runtime_txn_file.close();
            XAie_SubmitTransaction(&DevInst, TxnInst);
            printf("Submiting transactions \n");
	        XAie_ClearTransaction(&DevInst);
        #endif
    #endif
    // NOTE: For transaction binary generation, all gmio.wait() calls are replaced with driver calls
    #if TXN_FLOW != 1
        for (int col = 0; col < NUM_COLS; ++col) {
            g_compute_graph.gmio_out[col].wait();
        }
        g_compute_graph.end();
    #endif

    // Test DI
    int fail = 0;
    #if TXN_FLOW != 1
        printf("here \n");
        float const EPSILON = 1e-2;
        int mismatch_count = 0;
        for (int i = 0; i < cpu_C.num_rows; ++i) {
            for (int j = 0; j < cpu_C.num_cols; ++j) {
                float diff = cpu_C.act(i, j) - dev_C.act(i, j);
                if ( (std::abs(diff) >= EPSILON) | std::isnan(dev_C.act(i, j)) ) {
                    fail = 1;
                    mismatch_count++;
                    printf("C[%d, %d]: Expected: %f, Received: %f\n", i, j,
                           cpu_C.act(i, j), dev_C.act(i, j));
                } else {
                    printf("C[%d, %d]: Expected: %f, Received: %f\n", i, j,
                           cpu_C.act(i, j), dev_C.act(i, j));
                }
            }
        }
        if (!fail) {
            printf("%dx%dx%d GEMM DI: PASS\n", Mgemm, Kgemm, Ngemm);
        } else {
            printf("%dx%dx%d GEMM DI: FAIL\n", Mgemm, Kgemm, Ngemm);
            printf("mismatch count=%d \n", mismatch_count);
        }
    #endif

    // Free memory

    free(dev_in1);
    free(dev_in2);
    free(dev_out);
    free(cpu_out);

    return fail;
}
