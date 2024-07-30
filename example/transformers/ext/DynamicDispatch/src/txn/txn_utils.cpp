#include <utils/tfuncs.hpp>

#include "txn_utils.hpp"

namespace utils {

static constexpr size_t SUPER_KERNEL_ARGIDX = 4;

txn_util::~txn_util() {
  delete[] txn_ptr_;
  delete[] fused_txn_ptr_;
}

void txn_util::pass_through(uint8_t **ptr) {
  auto op_hdr = (XAie_OpHdr *)(*ptr);
  switch (op_hdr->Op) {
  case XAIE_IO_WRITE: {
    XAie_Write32Hdr *w_hdr = (XAie_Write32Hdr *)(*ptr);
    *ptr = *ptr + w_hdr->Size;
    break;
  }
  case XAIE_IO_BLOCKWRITE: {
    XAie_BlockWrite32Hdr *bw_header = (XAie_BlockWrite32Hdr *)(*ptr);
    *ptr = *ptr + bw_header->Size;
    break;
  }
  case XAIE_IO_MASKWRITE: {
    XAie_MaskWrite32Hdr *mw_header = (XAie_MaskWrite32Hdr *)(*ptr);
    *ptr = *ptr + mw_header->Size;
    break;
  }
  case XAIE_IO_MASKPOLL: {
    XAie_MaskPoll32Hdr *mp_header = (XAie_MaskPoll32Hdr *)(*ptr);
    *ptr = *ptr + mp_header->Size;
    break;
  }
  case (XAIE_IO_CUSTOM_OP_BEGIN):
  case (XAIE_IO_CUSTOM_OP_BEGIN + 1):
  case (XAIE_IO_CUSTOM_OP_BEGIN + 2):
  case (XAIE_IO_CUSTOM_OP_BEGIN + 3):
  case (XAIE_IO_CUSTOM_OP_BEGIN + 4): {
    XAie_CustomOpHdr *Hdr = (XAie_CustomOpHdr *)(*ptr);
    *ptr = *ptr + Hdr->Size;
    break;
  }
  default:
    throw std::runtime_error("Unknown op to pass through");
  }
}

static auto OpArgMapLT = [](const OpArgMap &lhs, const OpArgMap &rhs) {
  return lhs.xrt_arg_idx < rhs.xrt_arg_idx;
};

// Input argmap contains multiple args with different xrt_arg_ids.
// Partition it to multiple slots based on each xrt_arg_id
// And sort each partition for binary search.
static std::vector<std::vector<OpArgMap>>
partition_argmap(const std::vector<OpArgMap> &arg_map) {
  std::vector<std::vector<OpArgMap>> res;
  if (arg_map.size() == 0) {
    RYZENAI_LOG_TRACE(OpsFusion::dod_format(
        "Operator with arg_map size 0, skipping partition_argmap"));
    return res;
  }
  auto max_xrt_arg_id =
      *std::max_element(arg_map.begin(), arg_map.end(), OpArgMapLT);
  for (size_t i = 0; i <= max_xrt_arg_id.xrt_arg_idx; ++i) {
    std::vector<OpArgMap> args;
    std::copy_if(arg_map.begin(), arg_map.end(), std::back_inserter(args),
                 [i](const OpArgMap &arg) { return arg.xrt_arg_idx == i; });
    std::sort(args.begin(), args.end(), OpArgMapLT);
    res.push_back(std::move(args));
  }
  return res;
}

// Given an offset and xrt_arg_id, find the block(OpArg) in partition to which
// the offset belongs to. Returns reference to the corresponding OpArg
static const OpArgMap &
find_op_arg(const std::vector<std::vector<OpArgMap>> &argmaps,
            size_t xrt_arg_id, size_t offset) {
  const auto &partition = argmaps.at(xrt_arg_id);
  auto iter = std::lower_bound(
      partition.begin(), partition.end(), offset,
      [](const OpArgMap &lhs, size_t val) { return lhs.offset <= val; });

  size_t idx = std::distance(partition.begin(), iter);
  RYZENAI_LOG_TRACE(OpsFusion::dod_format(
      "find_op_arg: xrt_arg_id {} offset {} idx {}", xrt_arg_id, offset, idx));
  return argmaps.at(xrt_arg_id).at(idx - 1);
}

void txn_util::patch(const OpsFusion::Metadata::OpInfo &op_info,
                     const OpsFusion::Metadata &meta,
                     const std::vector<OpArgMap> &args_map) {
  RYZENAI_LOG_TRACE(OpsFusion::dod_format("Patching Instructions for op:{} ...",
                                          op_info.name));
  const auto &tensor_map = meta.tensor_map;
  const auto &super_instr_map = meta.super_instr_map;
  const auto &const_map = meta.const_map;
  const auto intermediate_scratch_size =
      MAP_AT(meta.fused_tensors, "scratch").size;
  const auto max_op_scratch_pad_size = meta.max_op_scratch_pad_size;

  const auto argmap_partition = partition_argmap(args_map);

  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn_ptr_;
  int num_ops = Hdr->NumOps;
  uint8_t *ptr = txn_ptr_ + sizeof(*Hdr);
  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("Total #ops : {}, max_op_scratch_pad_size: {}",
                            num_ops, max_op_scratch_pad_size));

  for (int n = 0; n < num_ops; n++) {
    auto op_hdr = (XAie_OpHdr *)ptr;
    switch (op_hdr->Op) {
    case XAIE_IO_CUSTOM_OP_BEGIN + 1: {
      XAie_CustomOpHdr *hdr = (XAie_CustomOpHdr *)(ptr);
      u32 size = hdr->Size;
      patch_op_t *op = (patch_op_t *)((ptr) + sizeof(*hdr));

      const auto curr_argidx = op->argidx;
      const auto curr_offset = op->argplus;

      // support two additional args for super kernels and initlize const params
      // super kernel params can be sent to NPU - ONNX node may not have this as
      // an input to the op some operators may need to send LUTs to NPU from DDR
      // for functionality. This will not be represented as an input in onnx
      // node. Example kernels - bf16 Silu/Gelu ops.
      DOD_THROW_IF((curr_argidx >= op_info.args.size() + 2),
                   OpsFusion::dod_format("curr_argidx({}) >= # op_args({}) + 2",
                                         curr_argidx, op_info.args.size()));
      DOD_THROW_IF(curr_argidx >= args_map.size(),
                   OpsFusion::dod_format("curr_argidx({}) >= args_map size({})",
                                         curr_argidx, args_map.size()));

      const auto &op_arg =
          find_op_arg(argmap_partition, curr_argidx, curr_offset);

      if (op_arg.arg_type == OpArgMap::CONST_KERNEL_PARAM_INPUT) {
        op->argidx = OpArgMap::CONST_KERNEL_PARAM_INPUT;
        op->argplus = curr_offset + super_instr_map.at(op_info.name).offset;
        RYZENAI_LOG_TRACE(OpsFusion::dod_format(
            "Patched : [{}:{}] -> [ super kernel instr ] -> [{}:{}] ",
            curr_argidx, curr_offset, op->argidx, op->argplus));
      } else if (op_arg.arg_type == OpArgMap::CONST_INPUT) {
        op->argidx = OpArgMap::CONST_INPUT;
        op->argplus = curr_offset + MAP_AT(const_map, op_info.name).offset;
        RYZENAI_LOG_TRACE(OpsFusion::dod_format(
            "Patched : [{}:{}] -> [ const ] -> [{}:{}] ", curr_argidx,
            curr_offset, op->argidx, op->argplus));
      } else if (op_arg.arg_type == OpArgMap::SCRATCH_PAD) {
        op->argidx = OpArgMap::SCRATCH_PAD;
        DOD_THROW_IF(
            curr_offset >= max_op_scratch_pad_size,
            OpsFusion::dod_format(
                "curr_offset({}) >= args_map max_op_scratch_pad_size({})",
                curr_offset, max_op_scratch_pad_size));
        // Note: Internal scratch pad for each op is shared, since it
        //       is assumed ops will execute sequentially
        //       Offset by scratch buffer size since beginning will store
        //       intermediate outputs, i.e. memory layout will be
        //       [intermediate_outputs | internal_scratch_pad]
        op->argplus = curr_offset + intermediate_scratch_size;
        RYZENAI_LOG_TRACE(OpsFusion::dod_format(
            "Patched : [{}:{}] -> [ scratch pad mem ] -> [{}:{}] ", curr_argidx,
            curr_offset, op->argidx, op->argplus));
      } else {
        const size_t onnx_argidx = op_arg.onnx_arg_idx;
        const auto &arg_label = ARRAY_AT(op_info.args, onnx_argidx);
        const auto &tensor = MAP_AT(tensor_map, arg_label);

        size_t new_argidx = tensor.arg_idx;
        size_t block_offset = tensor.offset;
        size_t curr_offset_delta = curr_offset - op_arg.offset;
        // tensor.offset tells where data actually is
        // op_arg.padding_offset is op requirement on whether it needs address
        // of actual data or beginning of padding
        size_t final_offset =
            block_offset + curr_offset_delta - op_arg.padding_offset;

        op->argidx = new_argidx;
        op->argplus = final_offset;
        RYZENAI_LOG_TRACE(OpsFusion::dod_format(
            "Patched : [{}:{}] -> [ onnx_argid:{} ] -> [{}:{}] ", curr_argidx,
            curr_offset, onnx_argidx, op->argidx, op->argplus));
      }

      ptr = ptr + size;

    } break;
    default:
      // no modification for other ops
      pass_through(&ptr);
      break;
    }
  }
  RYZENAI_LOG_TRACE(OpsFusion::dod_format(
      "Patching Instructions for op:{} ... DONE", op_info.name));
}

std::vector<uint8_t> txn_util::to_vector() {
  std::vector<uint8_t> txn_vec(txn_ptr_, txn_ptr_ + txn_size_);
  return txn_vec;
}

void txn_util::write_fused_txn_to_file(std::string file_name) {
  std::ofstream txn_bin(file_name, std::ios::binary);
  txn_bin.write((char *)fused_txn_ptr_, fused_size_);
}

std::string txn_util::summarize() {
  std::stringstream ss;
  ss << ss_hdr_.str() << ss_summary_.str();

  return ss.str();
}

std::string txn_util::text_dump() {
  std::stringstream ss;
  ss << ss_hdr_.str() << ss_ops_.str();
  return ss.str();
}

txn_util::txn_util(std::string file_name) {

  // std::cout << "Reading Txn from file: " << file_name << std::endl;
  std::ifstream txn_bin(file_name, std::ios::binary);
  if (!txn_bin.is_open()) {
    throw std::runtime_error(std::string("couldn't open the file : ") +
                             file_name);
  }

  XAie_TxnHeader *Hdr = new XAie_TxnHeader();
  txn_bin.read((char *)Hdr, sizeof(XAie_TxnHeader));

  uint8_t *Ptr = new uint8_t[Hdr->TxnSize];
  std::memcpy(Ptr, Hdr, sizeof(XAie_TxnHeader));

  uint8_t *txnPtr = Ptr + sizeof(*Hdr);
  txn_bin.read((char *)txnPtr, Hdr->TxnSize - sizeof(XAie_TxnHeader));

  delete Hdr;

  txn_ptr_ = Ptr;
  stringify_txn_bin();
  prepare_summary();
}

txn_util::txn_util(const std::vector<uint8_t> &txn_vec) {
  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn_vec.data();
  if (txn_vec.size() != Hdr->TxnSize) {
    throw std::runtime_error(
        "Invalid Transaction Vec : Size of input transaction vector and the "
        "size specified in its header doesn't match.");
  }

  txn_ptr_ = new uint8_t[Hdr->TxnSize];
  std::memcpy(txn_ptr_, txn_vec.data(), Hdr->TxnSize);
  stringify_txn_bin();
  prepare_summary();
}

void txn_util::prepare_summary() {
  ss_summary_ << "Summary of transaction binary" << std::endl;
  ss_summary_ << "Number of write ops: " << std::to_string(num_w_ops)
              << std::endl;
  ss_summary_ << "Number of block_write ops: " << std::to_string(num_bw_ops)
              << std::endl;
  ss_summary_ << "Number of mask_write ops: " << std::to_string(num_mw_ops)
              << std::endl;
  ss_summary_ << "Number of mask_poll ops: " << std::to_string(num_mp_ops)
              << std::endl;
  ss_summary_ << "Number of tct ops: " << std::to_string(num_tct_ops)
              << std::endl;
  ss_summary_ << "Number of patch ops: " << std::to_string(num_patch_ops)
              << std::endl;
  ss_summary_ << "Number of read ops: " << std::to_string(num_read_ops)
              << std::endl;
  ss_summary_ << "Number of timer ops: " << std::to_string(num_readtimer_ops)
              << std::endl;
  ss_summary_ << "Number of merge sync ops: "
              << std::to_string(num_mergesync_ops) << std::endl;
}

void txn_util::stringify_w32(uint8_t **ptr) {
  XAie_Write32Hdr *w_hdr = (XAie_Write32Hdr *)(*ptr);
  ss_ops_ << "W: 0x" << std::hex << w_hdr->RegOff << " 0x" << w_hdr->Value
          << std::endl;
  *ptr = *ptr + w_hdr->Size;
  num_w_ops++;
}

void txn_util::stringify_bw32(uint8_t **ptr) {
  XAie_BlockWrite32Hdr *bw_header = (XAie_BlockWrite32Hdr *)(*ptr);
  u32 bw_size = bw_header->Size;
  u32 Size = (bw_size - sizeof(*bw_header)) / 4;
  u32 *Payload = (u32 *)((*ptr) + sizeof(*bw_header));
  ss_ops_ << "BW: 0x" << std::hex << bw_header->RegOff << " ";
  // ss_ops_ << "Payload: ";
  for (u32 i = 0; i < Size; i++) {
    ss_ops_ << "0x" << std::hex << *Payload << " ";
    Payload++;
  }
  ss_ops_ << std::endl;
  *ptr = *ptr + bw_size;
  num_bw_ops++;
}

void txn_util::stringify_mw32(uint8_t **ptr) {
  XAie_MaskWrite32Hdr *mw_header = (XAie_MaskWrite32Hdr *)(*ptr);
  ss_ops_ << "MW: 0x" << std::hex << mw_header->RegOff << " " << mw_header->Mask
          << " " << mw_header->Value << std::endl;
  *ptr = *ptr + mw_header->Size;
  num_mw_ops++;
}

void txn_util::stringify_mp32(uint8_t **ptr) {
  XAie_MaskPoll32Hdr *mp_header = (XAie_MaskPoll32Hdr *)(*ptr);
  ss_ops_ << "MP: 0x" << std::hex << mp_header->RegOff << " " << mp_header->Mask
          << " " << mp_header->Value << std::endl;
  *ptr = *ptr + mp_header->Size;
  num_mp_ops++;
}

void txn_util::stringify_tct(uint8_t **ptr) {
  XAie_CustomOpHdr *co_header = (XAie_CustomOpHdr *)(*ptr);
  ss_ops_ << "TCT: " << std::endl;
  *ptr = *ptr + co_header->Size;
  num_tct_ops++;
}

void txn_util::stringify_patchop(uint8_t **ptr) {
  XAie_CustomOpHdr *hdr = (XAie_CustomOpHdr *)(*ptr);
  u32 size = hdr->Size;
  ss_ops_ << "PatchOp: ";
  patch_op_t *op = (patch_op_t *)((*ptr) + sizeof(*hdr));
  auto reg_off = op->regaddr;
  auto arg_idx = op->argidx;
  auto addr_offset = op->argplus;
  ss_ops_ << "(RegAddr: " << std::hex << reg_off << " Arg Idx: " << arg_idx
          << " Addr Offset: " << addr_offset << ")" << std::endl;
  *ptr = *ptr + size;
  num_patch_ops++;
}

void txn_util::stringify_rdreg(uint8_t **ptr) {
  XAie_CustomOpHdr *Hdr = (XAie_CustomOpHdr *)(*ptr);
  u32 size = Hdr->Size;
  ss_ops_ << "ReadOp: " << std::endl;
  *ptr = *ptr + size;
  num_read_ops++;
}

void txn_util::stringify_rectimer(uint8_t **ptr) {
  XAie_CustomOpHdr *Hdr = (XAie_CustomOpHdr *)(*ptr);
  u32 size = Hdr->Size;
  ss_ops_ << "TimerOp: " << std::endl;
  *ptr = *ptr + size;
  num_readtimer_ops++;
}

void txn_util::stringify_mergesync(uint8_t **ptr) {
  XAie_CustomOpHdr *Hdr = (XAie_CustomOpHdr *)(*ptr);
  u32 size = Hdr->Size;
  ss_ops_ << "MergeSyncOp: " << std::endl;
  *ptr = *ptr + size;
  num_mergesync_ops++;
}

void txn_util::stringify_txn_ops() {
  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn_ptr_;
  auto num_ops = Hdr->NumOps;
  auto ptr = txn_ptr_ + sizeof(*Hdr);

  XAie_OpHdr *op_hdr;
  for (uint32_t i = 0; i < num_ops; i++) {
    op_hdr = (XAie_OpHdr *)ptr;
    // printf("i: %d, OpCode: %d\n", i, op_hdr->Op);
    switch (op_hdr->Op) {
    case XAIE_IO_WRITE:
      stringify_w32(&ptr);
      break;
    case XAIE_IO_BLOCKWRITE:
      stringify_bw32(&ptr);
      break;
    case XAIE_IO_MASKWRITE:
      stringify_mw32(&ptr);
      break;
    case XAIE_IO_MASKPOLL:
      stringify_mp32(&ptr);
      break;
    case XAIE_IO_CUSTOM_OP_BEGIN:
      stringify_tct(&ptr);
      break;
    case XAIE_IO_CUSTOM_OP_BEGIN + 1:
      stringify_patchop(&ptr);
      break;
    case XAIE_IO_CUSTOM_OP_BEGIN + 2:
      stringify_rdreg(&ptr);
      break;
    case XAIE_IO_CUSTOM_OP_BEGIN + 3:
      stringify_rectimer(&ptr);
      break;
    case XAIE_IO_CUSTOM_OP_BEGIN + 4:
      stringify_mergesync(&ptr);
      break;
    default:
      throw std::runtime_error("Error: Unknown op code at offset at " +
                               std::to_string(ptr - txn_ptr_) +
                               ". OpCode: " + std::to_string(op_hdr->Op));
    }
  }
}

void txn_util::stringify_txn_bin() {

  XAie_TxnHeader *Hdr = (XAie_TxnHeader *)txn_ptr_;

  ss_hdr_ << "Header version: " << std::to_string(Hdr->Major) << "."
          << std::to_string(Hdr->Minor) << std::endl;
  ss_hdr_ << "Device Generation: " << std::to_string(Hdr->DevGen) << std::endl;
  ss_hdr_ << "Partition Info: " << std::endl;
  ss_hdr_ << "     Num Cols:" << std::to_string(Hdr->NumCols) << std::endl;
  ss_hdr_ << "     Num Rows:" << std::to_string(Hdr->NumRows) << std::endl;
  ss_hdr_ << "     Num MemTile Rows:" << std::to_string(Hdr->NumMemTileRows)
          << std::endl;
  ss_hdr_ << "Transaction Metadata:" << std::endl;
  ss_hdr_ << "     Size: " << std::to_string(Hdr->TxnSize) << std::endl;
  ss_hdr_ << "     NumOps: " << std::to_string(Hdr->NumOps) << std::endl;
  txn_size_ = Hdr->TxnSize;
  num_txn_ops_ = Hdr->NumOps;

  stringify_txn_ops();
}

std::vector<uint8_t>
txn_util::fuse_txns(const std::vector<std::vector<uint8_t>> &txns) {
  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("Fuse {} transactions ...", txns.size()));
  DOD_ASSERT(!txns.empty(), "No transactions to fuse");

  std::vector<uint8_t> fused_txn;

  size_t NumOps = 0;
  size_t TxnSize = sizeof(XAie_TxnHeader);

  // First go through all txn and figure out size to pre-allocate
  // this is to avoid unnecessary vector re-allocation
  for (size_t i = 0; i < txns.size(); ++i) {
    const auto &txn = ARRAY_AT(txns, i);
    const XAie_TxnHeader *txn_hdr = (const XAie_TxnHeader *)txn.data();
    NumOps += txn_hdr->NumOps;

    DOD_ASSERT(txn_hdr->TxnSize > sizeof(XAie_TxnHeader),
               OpsFusion::dod_format(
                   "Size of fused_transaction {} smaller than its header {}",
                   txn_hdr->TxnSize, sizeof(XAie_TxnHeader)));
    size_t instr_size = txn_hdr->TxnSize - sizeof(XAie_TxnHeader);
    TxnSize += instr_size;
  }

  fused_txn.reserve(TxnSize);

  // First txn - copy over header too
  const auto &txn1 = ARRAY_AT(txns, 0);
  const XAie_TxnHeader *txn1_hdr = (const XAie_TxnHeader *)txn1.data();
  fused_txn.insert(fused_txn.end(), txn1.data(),
                   txn1.data() + txn1_hdr->TxnSize);

  // Rest of txns
  for (size_t i = 1; i < txns.size(); ++i) {
    const auto &txn = ARRAY_AT(txns, i);
    const XAie_TxnHeader *txn_hdr = (const XAie_TxnHeader *)txn.data();
    const uint8_t *instr_ptr = txn.data() + sizeof(XAie_TxnHeader);
    // skip copying over the header for the rest of txns
    size_t instr_size = txn_hdr->TxnSize - sizeof(XAie_TxnHeader);
    fused_txn.insert(fused_txn.end(), instr_ptr, instr_ptr + instr_size);
  }

  // Update the header
  XAie_TxnHeader *txn_vec_hdr = (XAie_TxnHeader *)(fused_txn.data());
  txn_vec_hdr->NumOps = NumOps;
  txn_vec_hdr->TxnSize = TxnSize;
  DOD_ASSERT(fused_txn.size() == TxnSize,
             OpsFusion::dod_format(
                 "Size of fused_transaction {} doesn't match the size "
                 "in its header {}",
                 fused_txn.size(), TxnSize));

  // Just print summary.
  txn_util res(fused_txn);
  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("Fused Ops Summary\n{}", res.summarize()));
  RYZENAI_LOG_TRACE(
      OpsFusion::dod_format("Fuse {} transactions ... DONE", txns.size()));
  return fused_txn;
}

} // namespace utils
