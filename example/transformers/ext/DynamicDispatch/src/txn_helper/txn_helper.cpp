#include <vector>

#include <utils/ipu_hw_config.hpp>
#include <xaiengine.h>

#include "txn/txn_utils.hpp"
#include "txn_helper.hpp"

namespace ryzenai {

static inline std::tuple<uint32_t, uint32_t, uint32_t>
decode_regoff(const uint64_t &regoff) {
  // get register offset, col and row
  uint32_t col = (regoff >> XAIE_COL_SHIFT) & 0xFF;
  uint32_t row = (regoff >> XAIE_ROW_SHIFT) & 0xFF;
  uint32_t reg = (regoff & 0xFFFFF);

  return std::make_tuple(col, row, reg);
}

static inline void check_if_const_pad_config_exists(const uint64_t &regoff) {
  constexpr uint32_t const_pad_reg_start = 0x000A06E0;
  constexpr uint32_t const_pad_reg_end = 0x000A06F4;
  auto [col, row, reg] = decode_regoff(regoff);
  if ((row >= XAIE_MEM_TILE_ROW_START) &&
      (row < (XAIE_MEM_TILE_ROW_START + XAIE_MEM_TILE_NUM_ROWS))) {
    if ((reg >= const_pad_reg_start) && (reg <= const_pad_reg_end)) {
      DOD_THROW(OpsFusion::dod_format(
          "Prepend const padding error. Base transaction binary configures "
          "constant padding register. [Col: {}, Row: {}, Offset: {}]",
          col, row, reg));
    }
  }
}

static void is_const_pad_configured(const std::vector<uint8_t> &base_txn) {
  auto txn_ptr = (uint8_t *)base_txn.data();
  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn_ptr;
  auto num_ops = Hdr->NumOps;
  uint8_t *ptr = txn_ptr + sizeof(*Hdr);

  for (int n = 0; n < num_ops; n++) {
    auto op_hdr = (XAie_OpHdr *)ptr;
    switch (op_hdr->Op) {
    case XAIE_IO_WRITE: {
      XAie_Write32Hdr *w_hdr = (XAie_Write32Hdr *)(ptr);
      check_if_const_pad_config_exists(w_hdr->RegOff);
      ptr = ptr + w_hdr->Size;
      break;
    }
    default: {
      utils::txn_util::pass_through(&ptr);
      break;
    }
    }
  }
}

/**
 * @brief Generate txn to configure mem tile const padding register and prepend
 * it to the base transaction provided by the op.
 *
 * @param base_txn
 * @param pad_value
 * @param num_channels
 * @param num_cols
 * @return std::vector<uint8_t> new txn bin with const padding configuration
 */
std::vector<uint8_t>
prepend_mtile_const_pad_txn(const std::vector<uint8_t> &base_txn,
                            const uint32_t pad_value, uint8_t num_channels,
                            uint8_t num_cols) {

  // Check if base transaction has constant padding configuration embedded.
  is_const_pad_configured(base_txn);

  RYZENAI_LOG_TRACE(OpsFusion::dod_format(
      "[INFO] Prepending constant padding register in transaction binary with "
      "Value: {}, Num Channels: {}, Num Columns: {}",
      pad_value, static_cast<std::uint32_t>(num_channels),
      static_cast<std::uint32_t>(num_cols)));
  // Initialize AIE Driver. Hardcode for STRIX for now
  XAie_Config ConfigPtr{
      XAIE_DEV_GEN_AIE2P,      XAIE_BASE_ADDR,          XAIE_COL_SHIFT,
      XAIE_ROW_SHIFT,          XAIE_NUM_ROWS,           XAIE_NUM_COLS,
      XAIE_SHIM_ROW,           XAIE_MEM_TILE_ROW_START, XAIE_MEM_TILE_NUM_ROWS,
      XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS,  {0}};

  XAie_InstDeclare(DevInst, &ConfigPtr);
  ConfigPtr.NumCols = num_cols;
  XAie_CfgInitialize(&(DevInst), &ConfigPtr);

  XAie_StartTransaction(&DevInst, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
  for (uint8_t col = 0; col < num_cols; col++) {
    for (uint8_t row = XAIE_MEM_TILE_ROW_START;
         row < XAIE_MEM_TILE_ROW_START + XAIE_MEM_TILE_NUM_ROWS; row++) {
      for (uint8_t ch = 0; ch < num_channels; ch++) {
        XAie_DmaSetPadValue(&DevInst, XAie_TileLoc(col, row), ch, pad_value);
      }
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

  // concatenate with base txn
  // the order of txn, base_txn for below call is important.
  // txn must come before base_txn for const pad value configuration to be
  // valid.
  utils::txn_util t_util;
  return t_util.fuse_txns({txn, base_txn});
}

} // namespace ryzenai
