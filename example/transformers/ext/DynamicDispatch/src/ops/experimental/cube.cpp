#include <any>
#include <iostream>
#include <vector>

#include <ops/experimental/cube.hpp>
#include <ops/op_interface.hpp>
#include <ops/pm_load/pm_load.hpp>

// XRT headers
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// dpu kernel metadata
#include <utils/dpu_mdata.hpp>

// Subroutines to read the transaction binary
#include "op_buf.hpp"
#include "op_types.h"

#include <xaiengine.h>

#define XAIE_NUM_ROWS 6
#define XAIE_NUM_COLS 4
#define XAIE_BASE_ADDR 0
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20
#define XAIE_SHIM_ROW 0
#define XAIE_MEM_TILE_ROW_START 1
#define XAIE_MEM_TILE_NUM_ROWS 1
#define XAIE_AIE_TILE_ROW_START 2
#define XAIE_AIE_TILE_NUM_ROWS 4

namespace ryzenai {

template <typename InT, typename OutT> void cube<InT, OutT>::setup_instr_bo() {
  std::vector<Tensor> dummy_tensor;

  auto txn_bin = get_transaction_bin(dummy_tensor, dummy_tensor);
  aiectrl::op_buf instr_buf;
  instr_buf.addOP(aiectrl::transaction_op(txn_bin.data()));
  size_t instr_bo_size = instr_buf.ibuf_.size();
  instr_bo_ =
      xrt::bo(xrt_ctx_->get_context(), instr_bo_size, xrt::bo::flags::cacheable,
              xrt_ctx_->get_kernel().group_id(1));
  instr_bo_.write(instr_buf.ibuf_.data());
  instr_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

template <typename InT, typename OutT> cube<InT, OutT>::cube(bool load_xrt) {

  std::string XCLBIN_FNAME = OpInterface::get_dod_base_dir() + "\\xclbin\\" +
                             "stx" + "\\square_cube.xclbin";

  if (load_xrt) {
    // DONOT call instruction registry for this op. All txns are generated on
    // the fly.
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
    setup_instr_bo();
  }
}

template <typename InT, typename OutT>
const std::vector<uint8_t> cube<InT, OutT>::get_super_kernel_params(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  return {};
}

template <typename InT, typename OutT>
void cube<InT, OutT>::execute(const std::vector<Tensor> &input,
                              std::vector<Tensor> &output) {

  pm_load pm_loader_(true);
  pm_loader_.execute("cube", "int32");

  input_bo_.write(input.at(0).data);
  input_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  output_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto kernel = xrt_ctx_->get_kernel();
  auto run = kernel(2, instr_bo_, instr_bo_.size() / sizeof(int),
                    input_bo_.address() + DDR_AIE_ADDR_OFFSET,
                    output_bo_.address() + DDR_AIE_ADDR_OFFSET, 0, 0, 0);

  run.wait2();

  output_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  output_bo_.read(output.at(0).data);
}

template <typename InT, typename OutT>
void cube<InT, OutT>::initialize_const_params(
    const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {
  input_bo_ =
      xrt::bo(xrt_ctx_->get_device(), 32 * sizeof(InT), XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(0));
  output_bo_ =
      xrt::bo(xrt_ctx_->get_device(), 32 * sizeof(InT), XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(0));
}

template <typename InT, typename OutT>
void cube<InT, OutT>::initialize_const_params(
    void *dest, const std::vector<Tensor> &const_params,
    const std::map<std::string, std::any> &attr) {}

template <typename InT, typename OutT>
const std::vector<uint8_t> cube<InT, OutT>::get_transaction_bin(
    std::vector<Tensor> &input, std::vector<Tensor> &output,
    const std::map<std::string, std::any> &attr) {
  // Initialize AIE Driver. Hardcode for STRIX for now
  XAie_Config ConfigPtr{
      XAIE_DEV_GEN_AIE2P,      XAIE_BASE_ADDR,          XAIE_COL_SHIFT,
      XAIE_ROW_SHIFT,          XAIE_NUM_ROWS,           XAIE_NUM_COLS,
      XAIE_SHIM_ROW,           XAIE_MEM_TILE_ROW_START, XAIE_MEM_TILE_NUM_ROWS,
      XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS,  {0}};

  XAie_InstDeclare(DevInst, &ConfigPtr);
  XAie_CfgInitialize(&(DevInst), &ConfigPtr);

  XAie_LocType ShimDma;
  XAie_DmaDesc DmaDesc1, DmaDesc2;
  patch_op_t patch_op;

  ShimDma = XAie_TileLoc(2, 0);

  XAie_StartTransaction(&DevInst, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
  // Setting lock initial value for gr.sq.in[0]
  XAie_LockSetValue(&DevInst, XAie_TileLoc(2, 2), XAie_LockInit(0, 2));

  // Setting buffer buf0 of gr.sq.in[0]
  {
    XAie_DmaDesc DmaInst;
    XAie_DmaDescInit(&DevInst, &DmaInst, XAie_TileLoc(2, 2));
    XAie_DmaSetAddrLen(&DmaInst, 0x540, 128);
    XAie_DmaSetLock(&DmaInst, XAie_LockInit(0, -1), XAie_LockInit(1, 1));
    XAie_DmaSetNextBd(&DmaInst, 1, XAIE_ENABLE);
    XAie_DmaEnableBd(&DmaInst);
    XAie_DmaWriteBd(&DevInst, &DmaInst, XAie_TileLoc(2, 2), 0);
  }

  // Setting buffer buf0d of gr.sq.in[0]
  {
    XAie_DmaDesc DmaInst;
    XAie_DmaDescInit(&DevInst, &DmaInst, XAie_TileLoc(2, 2));
    XAie_DmaSetAddrLen(&DmaInst, 0x4000, 128);
    XAie_DmaSetLock(&DmaInst, XAie_LockInit(0, -1), XAie_LockInit(1, 1));
    XAie_DmaSetNextBd(&DmaInst, 0, XAIE_ENABLE);
    XAie_DmaEnableBd(&DmaInst);
    XAie_DmaWriteBd(&DevInst, &DmaInst, XAie_TileLoc(2, 2), 1);
  }
  {
    XAie_DmaChannelSetStartQueue(&DevInst, XAie_TileLoc(2, 2), 0, DMA_S2MM, 0,
                                 1, XAIE_DISABLE);
  }

  // Setting lock initial value for gr.sq.out[0]
  XAie_LockSetValue(&DevInst, XAie_TileLoc(2, 2), XAie_LockInit(2, 2));

  // Setting buffer buf1 of gr.sq.out[0]
  {
    XAie_DmaDesc DmaInst;
    XAie_DmaDescInit(&DevInst, &DmaInst, XAie_TileLoc(2, 2));
    XAie_DmaSetAddrLen(&DmaInst, 0x8540, 128);
    XAie_DmaSetLock(&DmaInst, XAie_LockInit(3, -1), XAie_LockInit(2, 1));
    XAie_DmaSetNextBd(&DmaInst, 3, XAIE_ENABLE);
    XAie_DmaEnableBd(&DmaInst);
    XAie_DmaWriteBd(&DevInst, &DmaInst, XAie_TileLoc(2, 2), 2);
  }

  // Setting buffer buf1d of gr.sq.out[0]
  {
    XAie_DmaDesc DmaInst;
    XAie_DmaDescInit(&DevInst, &DmaInst, XAie_TileLoc(2, 2));
    XAie_DmaSetAddrLen(&DmaInst, 0xc5c0, 128);
    XAie_DmaSetLock(&DmaInst, XAie_LockInit(3, -1), XAie_LockInit(2, 1));
    XAie_DmaSetNextBd(&DmaInst, 2, XAIE_ENABLE);
    XAie_DmaEnableBd(&DmaInst);
    XAie_DmaWriteBd(&DevInst, &DmaInst, XAie_TileLoc(2, 2), 3);
  }
  {
    XAie_DmaChannelSetStartQueue(&DevInst, XAie_TileLoc(2, 2), 0, DMA_MM2S, 2,
                                 1, XAIE_DISABLE);
  }

  // Connect mux stream id 3 to shim DMA.
  XAie_EnableShimDmaToAieStrmPort(&DevInst, XAie_TileLoc(0, 0), 3);
  // Connect mux stream id 3 to shim DMA.
  XAie_EnableShimDmaToAieStrmPort(&DevInst, XAie_TileLoc(2, 0), 3);
  // Connect demux stream id 2 to shim DMA.
  XAie_EnableAieToShimDmaStrmPort(&DevInst, XAie_TileLoc(2, 0), 2);

  // Shim DMA Configurations
  XAie_DmaDescInit(&DevInst, &DmaDesc1, ShimDma);
  XAie_DmaSetAddrLen(&DmaDesc1, 0, num_elems * sizeof(InT));
  XAie_DmaEnableBd(&DmaDesc1);
  XAie_DmaSetAxi(&DmaDesc1, 0U, 16U, 0U, 0U, 0U);
  XAie_DmaWriteBd(&DevInst, &DmaDesc1, ShimDma, 2);
  // insert patch op for input
  patch_op.action = 0;
  patch_op.argidx = 0;
  patch_op.argplus = 0;
  uint8_t BdId = 2;
  patch_op.regaddr = DmaDesc1.DmaMod->BaseAddr +
                     BdId * DmaDesc1.DmaMod->IdxOffset +
                     +_XAie_GetTileAddr(&DevInst, 0, 2) +
                     DmaDesc1.DmaMod->BdProp->Buffer->ShimDmaBuff.AddrLow.Idx *
                         4; // hardcode for now
  XAie_AddCustomTxnOp(&DevInst, XAIE_IO_CUSTOM_OP_DDR_PATCH, (void *)&patch_op,
                      sizeof(patch_op));

  XAie_DmaDescInit(&DevInst, &DmaDesc2, ShimDma);
  XAie_DmaSetAddrLen(&DmaDesc2, 0, num_elems * sizeof(OutT));
  XAie_DmaEnableBd(&DmaDesc2);
  XAie_DmaSetAxi(&DmaDesc2, 0U, 16U, 0U, 0U, 0U);
  XAie_DmaWriteBd(&DevInst, &DmaDesc2, ShimDma, 5);
  // insert patch op for input
  patch_op.action = 0;
  patch_op.argidx = 1;
  patch_op.argplus = 0;
  BdId = 5;
  patch_op.regaddr = DmaDesc1.DmaMod->BaseAddr +
                     BdId * DmaDesc1.DmaMod->IdxOffset +
                     +_XAie_GetTileAddr(&DevInst, 0, 2) +
                     DmaDesc1.DmaMod->BdProp->Buffer->ShimDmaBuff.AddrLow.Idx *
                         4; // hardcode for now
  XAie_AddCustomTxnOp(&DevInst, XAIE_IO_CUSTOM_OP_DDR_PATCH, (void *)&patch_op,
                      sizeof(patch_op));

  XAie_DmaChannelPushBdToQueue(&DevInst, ShimDma, 0, DMA_S2MM, 5);
  XAie_DmaChannelPushBdToQueue(&DevInst, ShimDma, 0, DMA_MM2S, 2);

  XAie_DmaChannelEnable(&DevInst, ShimDma, 0, DMA_S2MM);
  XAie_DmaChannelEnable(&DevInst, ShimDma, 0, DMA_MM2S);

  XAie_DmaWaitForDone(&DevInst, ShimDma, 0, DMA_S2MM, 0);

  uint8_t *txn_ptr = XAie_ExportSerializedTransaction(&DevInst, 0, 0);
  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn_ptr;
  auto size = Hdr->TxnSize;

  std::vector<uint8_t> txn(size, 0);
  memcpy((void *)txn.data(), (void *)txn_ptr, size);

  // check if there is an API to free txn pointer
  free(txn_ptr);
  XAie_Finish(&DevInst);

  return txn;
}

template <typename InT, typename OutT>
std::vector<OpArgMap>
cube<InT, OutT>::get_buffer_reqs(std::vector<Tensor> &input,
                                 std::vector<Tensor> &output,
                                 const std::map<std::string, std::any> &attr) {
  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::INPUT, 0, 0, 0, num_elems * sizeof(InT)},
      {OpArgMap::OpArgType::OUTPUT, 1, 1, 0, num_elems * sizeof(InT)}};

  return arg_map;
}

template class cube<int32_t, int32_t>;

} // namespace ryzenai
