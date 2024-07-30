#include <any>
#include <iostream>
#include <vector>

#include <ops/op_interface.hpp>
#include <ops/pm_load/pm_load.hpp>
#include <utils/tfuncs.hpp>

#include "utils/dpu_mdata.hpp"
#include "utils/ipu_hw_config.hpp"

#include <xaiengine.h>

namespace ryzenai {

pm_load::pm_load(bool load_xrt) {
  std::string XCLBIN_FNAME = OpInterface::get_dod_base_dir() + "\\xclbin\\" +
                             "stx" + "\\square_cube.xclbin";

  if (load_xrt) {
    xrt_ctx_ = dynamic_dispatch::xrt_context::get_instance(XCLBIN_FNAME);
  }
}

const std::map<std::string, overlay_pm_meta> pm_load::overlay_meta_ = {
    // one entry in the map per xclbin
    // TODO: Add nother field to capture PDI info if needed.
    {
        "square_cube.xclbin",
        {
            4,
            {
                {
                    0, 0, 0, // pkt_id, col, dma_ch_num
                },
            },
        },
    },
};

const std::map<std::string, op_xclbin_meta> pm_load::op_xclbin_meta_ = {
    {"cube_int32",
     {
         "square_cube.xclbin",
         "PM_0_cube.bin",
     }},
    {"square_int32",
     {
         "square_cube.xclbin",
         "PM_1_sq.bin",
     }},
};

const overlay_pm_meta &
pm_load::get_overlay_meta(const std::string &xclbin_name) const {
  if (overlay_meta_.find(xclbin_name) == overlay_meta_.end()) {
    throw std::runtime_error("Cannot find overlay meta for xclbin : " +
                             xclbin_name);
  }

  return overlay_meta_.at(xclbin_name);
}

const op_xclbin_meta &
pm_load::get_op_xclbin_meta(const std::string &op_name,
                            const std::string &dtype) const {

  std::string op = op_name + "_" + dtype;
  if (op_xclbin_meta_.find(op) == op_xclbin_meta_.end()) {
    throw std::runtime_error("Cannot find PM elf file for op: " + op_name +
                             " dtype: " + dtype);
  }

  return op_xclbin_meta_.at(op);
}

void pm_load::execute(std::string op_name, std::string dtype) {

  auto xclbin_meta = get_op_xclbin_meta(op_name, dtype);
  std::string pm_file_name = xclbin_meta.pm_elf_fname;
  std::string pm_file = OpInterface::get_dod_base_dir() +
                        "\\xclbin\\stx\\aie_elf_ctrl_pkt\\" + pm_file_name;

  std::vector<uint8_t> pm_bin = OpsFusion::read_bin_file<uint8_t>(pm_file);
  std::cout << "pm_bin_file_size  " << pm_bin.size() << std::endl;

  auto pm_bo =
      xrt::bo(xrt_ctx_->get_device(), pm_bin.size(), XRT_BO_FLAGS_HOST_ONLY,
              xrt_ctx_->get_kernel().group_id(0));
  pm_bo.write(pm_bin.data());
  pm_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Prepare txn to load PM bin
  std::vector<Tensor> input, output;
  const std::map<std::string, std::any> &attr{{"op_type", op_name},
                                              {"op_dtype", dtype}};

  auto txn_bin = get_transaction_bin(input, output, attr);
  aiectrl::op_buf instr_buf;
  instr_buf.addOP(aiectrl::transaction_op(txn_bin.data()));
  size_t instr_bo_size = instr_buf.ibuf_.size();
  auto instr_bo_ =
      xrt::bo(xrt_ctx_->get_context(), instr_bo_size, xrt::bo::flags::cacheable,
              xrt_ctx_->get_kernel().group_id(1));
  instr_bo_.write(instr_buf.ibuf_.data());
  instr_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Execute kernel to load PM Bin
  auto kernel = xrt_ctx_->get_kernel();
  auto run = kernel(2, instr_bo_, instr_bo_.size() / sizeof(int),
                    pm_bo.address() + DDR_AIE_ADDR_OFFSET, 0, 0, 0, 0);

  run.wait2();
}

const std::vector<uint8_t>
pm_load::get_transaction_bin(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             const std::map<std::string, std::any> &attr) {

  // find op_name
  std::string op_type;
  if (attr.find("op_type") != attr.end()) {
    op_type = std::any_cast<std::string>(attr.find("op_type")->second);
  } else {
    throw std::runtime_error("Can't find op_type in attrs asdf asdf");
  }

  std::string op_dtype;
  if (attr.find("op_dtype") != attr.end()) {
    op_dtype = std::any_cast<std::string>(attr.find("op_dtype")->second);
  } else {
    throw std::runtime_error("Can't find op_dtype in attrs");
  }

  auto xclbin_meta = get_op_xclbin_meta(op_type, op_dtype);
  auto overlay_meta = get_overlay_meta(xclbin_meta.xclbin_name);

  std::vector<uint8_t> pm_bin = get_pm_bin(attr);

  // Initialize AIE Driver. Hardcode for STRIX for now
  XAie_Config ConfigPtr{
      XAIE_DEV_GEN_AIE2P,      XAIE_BASE_ADDR,          XAIE_COL_SHIFT,
      XAIE_ROW_SHIFT,          XAIE_NUM_ROWS,           XAIE_NUM_COLS,
      XAIE_SHIM_ROW,           XAIE_MEM_TILE_ROW_START, XAIE_MEM_TILE_NUM_ROWS,
      XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS,  {0}};

  XAie_InstDeclare(DevInst, &ConfigPtr);
  XAie_CfgInitialize(&(DevInst), &ConfigPtr);

  XAie_LocType ShimDma;
  XAie_DmaDesc DmaDesc;
  patch_op_t patch_op;

  XAie_StartTransaction(&DevInst, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
  // Reset Core tiles and Tile DMAs
  for (int c = 0; c < overlay_meta.num_cols; c++) {
    for (int r = XAIE_AIE_TILE_ROW_START;
         r < XAIE_AIE_TILE_NUM_ROWS + XAIE_AIE_TILE_ROW_START; r++) {
      XAie_CoreDisable(&DevInst, XAie_TileLoc(c, r));
      XAie_CoreReset(&DevInst, XAie_TileLoc(c, r));
      XAie_CoreUnreset(&DevInst, XAie_TileLoc(c, r));
      XAie_DmaChannelResetAll(&DevInst, XAie_TileLoc(c, r), DMA_CHANNEL_RESET);
      XAie_DmaChannelResetAll(&DevInst, XAie_TileLoc(c, r),
                              DMA_CHANNEL_UNRESET);
    }
  }

  for (int c = 0; c < overlay_meta.num_cols; c++) {
    for (int r = XAIE_MEM_TILE_ROW_START;
         r < XAIE_MEM_TILE_NUM_ROWS + XAIE_MEM_TILE_ROW_START; r++) {
      XAie_DmaChannelResetAll(&DevInst, XAie_TileLoc(c, r), DMA_CHANNEL_RESET);
      XAie_DmaChannelResetAll(&DevInst, XAie_TileLoc(c, r),
                              DMA_CHANNEL_UNRESET);
    }
  }
  auto &pkt_sw_meta = overlay_meta.pkt_sw_meta_;

  for (auto &meta_ : pkt_sw_meta) {
    // TODO: Handle packet id and come up with BD sequence for it.
    ShimDma = XAie_TileLoc(meta_.col, 0);
    XAie_DmaDescInit(&DevInst, &DmaDesc, ShimDma);
    XAie_DmaSetAddrLen(&DmaDesc, 0, pm_bin.size());
    XAie_DmaEnableBd(&DmaDesc);
    XAie_DmaSetAxi(&DmaDesc, 0U, 16U, 0U, 0U, 0U);
    XAie_DmaWriteBd(&DevInst, &DmaDesc, ShimDma, 0);
    // insert patch op for input
    patch_op.action = 0;
    patch_op.argidx = 0;
    patch_op.argplus = 0;
    uint8_t BdId = 0; // Always use BD 0 for PM configuration
    u64 regaddr = DmaDesc.DmaMod->BaseAddr + BdId * DmaDesc.DmaMod->IdxOffset +
                  +_XAie_GetTileAddr(&DevInst, 0, meta_.col) +
                  DmaDesc.DmaMod->BdProp->Buffer->ShimDmaBuff.AddrLow.Idx * 4;
    patch_op.regaddr = regaddr;
    XAie_AddCustomTxnOp(&DevInst, XAIE_IO_CUSTOM_OP_DDR_PATCH,
                        (void *)&patch_op, sizeof(patch_op));

    XAie_DmaChannelPushBdToQueue(&DevInst, ShimDma, 0, DMA_MM2S, 0);
    XAie_DmaChannelEnable(&DevInst, ShimDma, 0, DMA_MM2S);
  }

  // Poll for completition after all BD Writes are done
  for (auto &meta_ : pkt_sw_meta) {
    ShimDma = XAie_TileLoc(meta_.col, 0);
    XAie_DmaWaitForDone(&DevInst, ShimDma, 0, DMA_MM2S, 0);
  }

  // Reset all DMA Channels
  for (auto &meta_ : pkt_sw_meta) {
    ShimDma = XAie_TileLoc(meta_.col, 0);
    // TODO: Is this needed?
    // XAie_DmaChannelResetAll(&DevInst, ShimDma, DMA_CHANNEL_RESET);
    // XAie_DmaChannelResetAll(&DevInst, ShimDma, DMA_CHANNEL_UNRESET);
  }

  // Enable all cores
  for (int c = 0; c < overlay_meta.num_cols; c++) {
    for (int r = XAIE_AIE_TILE_ROW_START;
         r < XAIE_AIE_TILE_NUM_ROWS + XAIE_AIE_TILE_ROW_START; r++) {
      XAie_CoreEnable(&DevInst, XAie_TileLoc(c, r));
    }
  }

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

std::vector<OpArgMap>
pm_load::get_buffer_reqs(std::vector<Tensor> &input,
                         std::vector<Tensor> &output,
                         const std::map<std::string, std::any> &attr) {

  auto pm_bin = get_pm_bin(attr);
  // Load PM in super_kernel_param_input
  std::vector<OpArgMap> arg_map{
      {OpArgMap::OpArgType::CONST_KERNEL_PARAM_INPUT, 0, 0, 0, pm_bin.size()}};
  return arg_map;
}

const std::vector<uint8_t>
pm_load::get_super_kernel_params(std::vector<Tensor> &input,
                                 std::vector<Tensor> &output,
                                 const std::map<std::string, std::any> &attr) {

  return get_pm_bin(attr);
}

const std::vector<uint8_t>
pm_load::get_pm_bin(const std::map<std::string, std::any> &attr) {
  // find op_name
  std::string op_type;
  if (attr.find("op_type") != attr.end()) {
    op_type = std::any_cast<std::string>(attr.find("op_type")->second);
  } else {
    throw std::runtime_error("Can't find op_type in attrs");
  }

  std::string op_dtype;
  if (attr.find("op_dtype") != attr.end()) {
    op_dtype = std::any_cast<std::string>(attr.find("op_dtype")->second);
  } else {
    throw std::runtime_error("Can't find op_dtype in attrs");
  }

  auto xclbin_meta = get_op_xclbin_meta(op_type, op_dtype);
  auto overlay_meta = get_overlay_meta(xclbin_meta.xclbin_name);

  std::string pm_file_name = xclbin_meta.pm_elf_fname;
  std::string pm_file = OpInterface::get_dod_base_dir() +
                        "\\xclbin\\stx\\aie_elf_ctrl_pkt\\" + pm_file_name;

  std::vector<uint8_t> pm_bin = OpsFusion::read_bin_file<uint8_t>(pm_file);
  return pm_bin;
}

} // namespace ryzenai
