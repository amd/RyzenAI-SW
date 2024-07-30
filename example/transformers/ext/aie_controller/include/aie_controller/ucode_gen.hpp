/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef __UCODE_GEN_HPP__
#define __UCODE_GEN_HPP__

#include <queue>
#include <vector>

#include <adf/adf_api_message.h>
#include <adf/adf_types.hpp>
#include <memory>
#include <unordered_map>

extern "C"
{
#include "xaiengine.h"

}

namespace aiectrl
{

struct hash_bd {
  template <class T1, class T2, class T3>
  size_t operator()(const std::tuple<T1, T2, T3>& x) const {
    return std::get<0>(x) ^ std::get<1>(x) ^ std::get<2>(x); 
  }
};

class shimTileHandle
{
public:

    shimTileHandle() = delete;
    ~shimTileHandle() {}

    shimTileHandle( XAie_DevInst * devInst, const GMIOConfig pConfig);

    err_code program_only(const void* address, const std::vector<dma_buffer_descriptor>& buffer_descriptors, const std::vector<uint32_t>& bd_ids, uint32_t repeat_count = 1, bool enable_task_complete_token = false);
    err_code enqueuetask_only(uint8_t startBdId, uint32_t repeat_count=1, bool enable_task_complete_token=false);
    std::pair<u64, err_code> compute_patch_regaddr(uint8_t bdid);

    err_code enqueue(const void* address, uint32_t size, uint32_t repeat_count = 1, bool enable_task_complete_token = false);
    err_code enqueue(const void* address, const std::vector<dma_buffer_descriptor>& buffer_descriptors, const std::vector<uint32_t>& bd_ids, uint32_t repeat_count = 1, bool enable_task_complete_token = false);

    err_code enqueue(XAie_DevInst* devInst, XAie_LocType tileLoc, XAie_DmaDirection dir, uint8_t channel, uint32_t repeatCount, bool enableTaskCompleteToken, std::vector<uint8_t> bdIds, std::vector< std::shared_ptr< dma_buffer_descriptor> > bdParams);

    void init ( XAie_DevInst * devInst);
    
    XAie_LocType getTileLoc() const { return gmioTileLoc_;}

    const GMIOConfig* getGMIOConfig() const { return &gmio_cfg_; }

    void wait();
    std::vector<uint32_t> wait_tct();
    void clearJournal() { journal_.clear(); }

private:
    std::pair<uint8_t, err_code> configBD( gmio_buffer_descriptor * bd, bool use_next_bd, uint32_t repeatCount = 1, bool enableTaskCompleteToken = false);

    err_code configBD(XAie_DevInst* devInst, XAie_LocType tileLoc, XAie_DmaDirection dir, uint8_t bdId, const dma_buffer_descriptor * bdParam);

//    err_code enqueueTask(XAie_DevInst* devInst, XAie_LocType tileLoc, XAie_DmaDirection dir, uint8_t channel, uint32_t repeatCount, bool enableTaskCompleteToken, std::vector<uint8_t> bdIds, std::vector< std::shared_ptr< dma_buffer_descriptor> > bdParams);

    AieRC configure();
    XAie_DevInst *devInst_;
    /// GMIO shim DMA physical configuration compiled by the AIE compiler
    GMIOConfig gmio_cfg_;
    XAie_LocType gmioTileLoc_;

    uint8_t dmaStartQMaxSize_ = 4;
    std::queue<size_t> enqueuedBDs_;
    std::queue<size_t> availableBDs_;   
    std::unordered_map<std::tuple<uint8_t,uint8_t,uint8_t>, dma_buffer_descriptor, hash_bd> journal_;
};

class memTilePortHandle
{
public:
    memTilePortHandle() = delete;
    memTilePortHandle(XAie_DevInst* devInst, const DMAChannelConfig & dma_cfg): devInst_(devInst), dma_cfg_(dma_cfg)
    {
    }
    ~memTilePortHandle(){}

    err_code enqueue_task(const std::vector<dma_buffer_descriptor>& buffer_descriptors, const std::vector<uint32_t>& bd_ids, uint32_t repeat_count, bool enable_task_complete_token);
    err_code wait();
    std::vector<uint32_t> wait_tct();
    XAie_LocType getTileLoc() const;
    const DMAChannelConfig & getDMAChConfig () const{
        return dma_cfg_;
    }

    void clearJournal() { journal_.clear(); }

private:
    XAie_DevInst* devInst_;
    const DMAChannelConfig dma_cfg_;
    std::unordered_map<std::tuple<uint8_t,uint8_t,uint8_t>, dma_buffer_descriptor, hash_bd> journal_;
};

class dma_api
{
public:
    /// AIE2 DMA Buffer descriptor.
    /// Data types in this class are considered to match AIE driver.
    struct buffer_descriptor
    {
        /// Address in bytes
        uint64_t address = 0;
        /// Length in bytes
        uint32_t length = 0;
        /// D0, D1, D2, D3(memory tile only) stepsize in 32-bit word
        std::vector<uint32_t> stepsize;
        /// D0, D1, D2(memory tile only) wrap in 32-bit word
        std::vector<uint32_t> wrap;
        /// D0, D1, D2 zero-before and zero-after in 32-bit word
        std::vector<std::pair<uint32_t, uint32_t>> padding;
        /// Enable adding packet header at start of transfer. MM2S only.
        bool enable_packet = false;
        /// Packet ID. MM2S only.
        uint8_t packet_id = 0;
        /// Out of order BD ID
        uint8_t out_of_order_bd_id = 0;
        /// TLAST suppress. Memory tile only. MM2S only.
        bool tlast_suppress = false;
        /// Iteration stepsize in 32-bit word
        uint32_t iteration_stepsize = 0;
        /// Iteration wrap
        uint16_t iteration_wrap = 0;
        /// Iteration current
        uint8_t iteration_current = 0;
        /// Enable compression for MM2S or enable decompression for S2MM. AIE tile and memory tile only.
        bool enable_compression = false;
        /// Enable lock acquire
        bool lock_acq_enable = false;
        /// Lock acquire value (signed). acq_ge if less than 0. acq_eq if larger than or equal to 0.
        int8_t lock_acq_value = 0;
        /// Lock id to acquire
        uint8_t lock_acq_id = 0;
        /// Lock release value (signed). 0: do not release a lock.
        int8_t lock_rel_value = 0;
        /// Lock id to release
        uint8_t lock_rel_id = 0;
        /// Continue with next BD
        bool use_next_bd = false;
        /// Next BD ID
        uint8_t next_bd = 0;
        /// AXI burst length. Shim tile only. In binary format 00: BLEN = 4 (64B), 01: BLEN = 8 (128B), 10: BLEN = 16 (256B), 11: Undefined
        uint8_t burst_length = 4;
    };

    /// Configure BD, wait task queue space, then enqueue task.
    /// @param tileType 0 (tile_type::aie_tile), 1 (tile_type::shim_tile), 2 (tile_type::memory_tile)
    /// @param column AIE array column
    /// @param row AIE array row relative to tileType
    /// @param dir 0 (XAie_DmaDirection::DMA_S2MM), 1 (XAie_DmaDirection::DMA_MM2S)
    static err_code configureBdWaitQueueEnqueueTask(XAie_DevInst* devInst,int tileType, uint8_t column, uint8_t row, int dir, uint8_t channel, uint32_t repeatCount, bool enableTaskCompleteToken, std::vector<uint8_t> bdIds, std::vector<const dma_buffer_descriptor*> bdParams);

    static err_code configureBD(XAie_DevInst* devInst,int tileType, uint8_t column, uint8_t row, uint8_t bdId, const dma_buffer_descriptor* bdParam);
    static err_code configureBDAddressOnly(XAie_DevInst* devInst,int tileType, uint8_t column, uint8_t row, uint8_t bdId, const dma_buffer_descriptor* bdParam);
    static err_code enqueueTask(XAie_DevInst* devInst,int tileType, uint8_t column, uint8_t row, int dir, uint8_t channel, uint32_t repeatCount, bool enableTaskCompleteToken, uint8_t startBdId);
    static err_code waitDMAChannelTaskQueue(XAie_DevInst* devInst, int tileType, uint8_t column, uint8_t row, int dir, uint8_t channel);
    static err_code waitDMAChannelDone(XAie_DevInst* devInst, int tileType, uint8_t column, uint8_t row, int dir, uint8_t channel);
};

}

#endif
