/*
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */


#ifndef __ADF_TYPES_HPP__
#define __ADF_TYPES_HPP__

#include <string>
#include <vector>

namespace aiectrl{


struct gmio_config
{
    enum gmio_type { gm2aie, aie2gm, gm2pl, pl2gm };

    /// GMIO object id
    int id;
    /// GMIO variable name
    std::string name;
    /// GMIO loginal name
    std::string logicalName;
    /// GMIO type
    gmio_type type;
    /// Shim tile column to where the GMIO is mapped
    short shimColumn;
    /// Channel number (0-S2MM0,1-S2MM1,2-MM2S0,3-MM2S1).
    short channelNum;
    /// Shim stream switch port id (slave: gm-->me, master: me-->gm)
    short streamId;
    /// For type == gm2aie or type == aie2gm, burstLength is the burst length for the AXI-MM transfer
    /// (4 or 8 or 16 in C_RTS API). The burst length in bytes is burstLength * 16 bytes (128-bit aligned).
    /// For type == gm2pl or type == pl2gm, burstLength is the burst length in bytes.
    short burstLength;

#ifndef __XRT__
    // Does this gmio use packet switching?
    bool isPacketSwitch;
    // Map packet split output port id to packet id
    std::vector<unsigned short> packetIDs;
    /// PL kernel instance name
    std::string plKernelInstanceName;
    /// PL parameter index
    int plParameterIndex;
    /// PL IP Id for AIESIM (for type == gm2pl or type == pl2gm)
    int plId;
#ifdef __AIECOMPILER_BACKEND__
    std::string plDriverSetAxiMMAddr;
#else
    /// Driver function pointer to set PL m_axi port address (for aiesimulator and hardware) and size (for aiesimulator only)
    void(*plDriverSetAxiMMAddr)(unsigned long long, unsigned int);
#endif
#endif
};

struct dma_channel_config
{
    /// Port instance id
    int portId;
    /// Port name
    std::string portName;
    /// Parent id
    int parentId;
    /// Tile type
    int tileType;
    /// DMA channel column
    short column;
    /// DMA channel row relative to tileType
    short row;
    /// S2MM or MM2S. 0:S2MM, 1:MM2S. Should resemble the XAie_DmaDirection driver enum.
    int S2MMOrMM2S;
    /// DMA channel number
    short channel;
};

struct dma_buffer_descriptor
{
    /// Address in 32bit word
    /// Memory tile: West 0x0_0000 - 0x7_FFFF (channel 0-3 only); Local 0x8_0000 - 0xF_FFFF; East 0x10_0000 - 0x17FFFF (channel 0-3 only)
    uint64_t address = 0;
    /// Transaction length in 32bit word
    uint32_t length = 0;
    /// D0, D1, D2, D3(memory tile only) stepsize in 32-bit word
    std::vector<uint32_t> stepsize;
    /// D0, D1, D2(memory tile only) wrap in 32-bit word
    std::vector<uint32_t> wrap;
    /// D0, D1, D2 zero-before and zero-after in 32-bit word. MM2S only.
    std::vector<std::pair<uint32_t, uint32_t>> padding;
    /// Enable adding packet header at start of transfer. MM2S only. enable_pkt_mode::automatic enables packet when there is a connected pktsplit.
    bool enable_packet = false;
    /// Packet id
    uint32_t packet_id = 0;
    /// Out of order BD ID. MM2S only.
    uint32_t out_of_order_bd_id = 0;
    /// TLAST suppress. Memory tile only. MM2S only.
    bool tlast_suppress = false;
    /// Iteration stepsize in 32-bit word
    uint32_t iteration_stepsize = 0;
    /// Iteration wrap
    uint32_t iteration_wrap = 0;
    /// Iteration current
    uint32_t iteration_current = 0;
    /// Enable compression for MM2S or enable decompression for S2MM. AIE tile and memory tile only.
    bool enable_compression = false;
    /// Enable lock acquire
    bool lock_acq_enable = false;
    /// Lock acquire value V (signed). V<0: acq_ge; V>=0: acq_eq.
    int32_t lock_acq_value = 0;
    /// ID of lock to acquire
    /// Memory tile: West 0-63 (channel 0-3 only); Local 64-127; East 128-191 (channel 0-3 only)
    uint32_t lock_acq_id = 0;
    /// Lock release value (signed). 0 = do not release a lock.
    int32_t lock_rel_value = 0;
    /// ID of lock to release.
    /// Memory tile: West 0-63 (channel 0-3 only); Local 64-127; East 128-191 (channel 0-3 only)
    uint32_t lock_rel_id = 0;
    /// Use next BD
    bool use_next_bd = false;
    /// Next BD ID
    uint32_t next_bd = 0;
};

struct gmio_buffer_descriptor : public dma_buffer_descriptor
{
    /// AXI burst length. Shim tile only. In binary format 00: BLEN = 4 (64B), 01: BLEN = 8 (128B), 10: BLEN = 16 (256B), 11: Undefined
    uint8_t burst_length = 4;
};

enum tile_type {
    undefined_tile = -1,
    aie_tile,
    shim_tile,
    memory_tile
};

typedef gmio_config GMIOConfig;
typedef dma_channel_config DMAChannelConfig;

}

#endif