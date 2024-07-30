#include "../include/scalar_mult.hpp"
#include <vector>

// AIE Driver headers
#include "xaiengine.h"

// Headers to create Txn binary
#include "op_buf.hpp"
#include "op_types.h"

// dpu kernel metadata
#include "dpu_kernel_metadata.hpp"

#define XAIE_NUM_ROWS 6
#define XAIE_NUM_COLS 5
#define XAIE_BASE_ADDR 0
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20
#define XAIE_SHIM_ROW 0
#define XAIE_MEM_TILE_ROW_START 1
#define XAIE_MEM_TILE_NUM_ROWS 1
#define XAIE_AIE_TILE_ROW_START 2
#define XAIE_AIE_TILE_NUM_ROWS 4

aie::scalar_mult::scalar_mult(size_t size) {
  std::string xclbin_fname = std::string(std::getenv("PYTORCH_AIE_PATH")) +
                             "\\xclbin\\phx\\int32_scalar.xclbin";
  initialize_device(xclbin_fname);

  // Only 32 bit int supported today
  input_ = xrt::bo(context_, size * sizeof(int), xrt::bo::flags::host_only,
                   kernel_.group_id(0));
  output_ = xrt::bo(context_, size * sizeof(int), xrt::bo::flags::host_only,
                    kernel_.group_id(0));

  generate_txn(input_.address() + DDR_AIE_ADDR_OFFSET,
               output_.address() + DDR_AIE_ADDR_OFFSET, size * sizeof(int));
}

void aie::scalar_mult::initialize_device(std::string xcl) {
  std::string xclbinFileName = xcl.data();
  std::cout << xclbinFileName << std::endl;
  unsigned int device_index = 0;
  device_ = xrt::device(device_index);
  xclbin_ = xrt::xclbin(xclbinFileName);

  device_.register_xclbin(xclbin_);
  context_ = xrt::hw_context(device_, xclbin_.get_uuid());
  kernel_ = xrt::kernel(context_, KERNEL_NAME);
}

aie::scalar_mult::~scalar_mult() {}

void aie::scalar_mult::generate_txn(uint64_t src, uint64_t dest,
                                    uint32_t size) {
  XAie_Config ConfigPtr{
      XAIE_DEV_GEN_AIE2IPU,    XAIE_BASE_ADDR,          XAIE_COL_SHIFT,
      XAIE_ROW_SHIFT,          XAIE_NUM_ROWS,           XAIE_NUM_COLS,
      XAIE_SHIM_ROW,           XAIE_MEM_TILE_ROW_START, XAIE_MEM_TILE_NUM_ROWS,
      XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS,  {0}};

  XAie_InstDeclare(DevInst, &ConfigPtr);
  XAie_CfgInitialize(&(DevInst), &ConfigPtr);

  AieRC RC;
  XAie_LocType ShimDma;
  XAie_DmaDesc DmaDesc1, DmaDesc2;
  uint8_t col = 2;

  ShimDma = XAie_TileLoc(col, 0);

  XAie_StartTransaction(&DevInst, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
  XAie_DataMemWrWord(&DevInst, XAie_TileLoc(3, 3), 0x0, 0xDEADBEEF);
  RC = XAie_DmaDescInit(&DevInst, &DmaDesc1, ShimDma);
  RC = XAie_DmaSetAddrLen(&DmaDesc1, src, size);
  RC = XAie_DmaEnableBd(&DmaDesc1);
  RC = XAie_DmaSetAxi(&DmaDesc1, 0U, 16U, 0U, 0U, 0U);
  RC = XAie_DmaWriteBd(&DevInst, &DmaDesc1, ShimDma, 2);

  RC = XAie_DmaDescInit(&DevInst, &DmaDesc2, ShimDma);
  RC = XAie_DmaSetAddrLen(&DmaDesc2, dest, size);
  RC = XAie_DmaEnableBd(&DmaDesc2);
  RC = XAie_DmaSetAxi(&DmaDesc2, 0U, 16U, 0U, 0U, 0U);
  RC = XAie_DmaWriteBd(&DevInst, &DmaDesc2, ShimDma, 5);

  RC = XAie_DmaChannelPushBdToQueue(&DevInst, ShimDma, 0, DMA_S2MM, 5);
  RC = XAie_DmaChannelPushBdToQueue(&DevInst, ShimDma, 0, DMA_MM2S, 2);

  RC = XAie_DmaChannelEnable(&DevInst, ShimDma, 0, DMA_S2MM);
  RC = XAie_DmaChannelEnable(&DevInst, ShimDma, 0, DMA_MM2S);

  RC = XAie_DmaWaitForDone(&DevInst, ShimDma, 0, DMA_S2MM, 0);

  uint8_t *txn_ptr = XAie_ExportSerializedTransaction(&DevInst, 0, 0);
  aiectrl::op_buf instr_buf_;
  instr_buf_.addOP(aiectrl::transaction_op(txn_ptr));

  instr_bo_ = xrt::bo(context_, instr_buf_.ibuf_.size(),
                      xrt::bo::flags::cacheable, kernel_.group_id(1));
  instr_bo_.write(instr_buf_.ibuf_.data());
  instr_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

torch::Tensor aie::scalar_mult::execute(torch::Tensor x) {
  // Copy input / output vectors into XRT BOs
  input_.write(x.data_ptr());
  input_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  output_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::vector<u64> kargv(5, 0);
  auto run = kernel_(OPCODE, instr_bo_, instr_bo_.size() / sizeof(int),
                     kargv[0], kargv[1], kargv[2], kargv[3], kargv[4]);

  run.wait();

  torch::Tensor out = torch::zeros(x.sizes(), torch::kInt32);
  output_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  output_.read(out.data_ptr());

  return out;
}
