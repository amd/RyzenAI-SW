// Channnel sharing:  True
// Mgemm:  1
// Kgemm:  4096
// Ngemm:  4096
// Max-Sub vol dim that the core will support:
// M_SUBV:  8
// K_SUBV:  128
// N_SUBV:  128
// GRP_SIZE:  128
// Actual sub vol dim computed:
// M_SUBV:  8
// K_SUBV:  32
// N_SUBV:  128
// GRP_SIZE:  32
// INNER_LOOP:  128
// OUTER_M_LOOP:  1
// OUTER_N_LOOP:  2
// Generating BDs for token phase 

#include <adf/adf_api/AIERuntimeControl.h>

//
// NOTE: This code is auto-generated, so do not modify.
//

void run_host_bd_config()
{

    //
    // 1 to 1 Data Transfer
    //
    // Location: memtile_0_0
    //
    // Writers
    // ----------------
    // s2mm_0 BDs: 0
    //
    // Readers
    // ----------------
    // mm2s_0 BDs: 1
    //
    // Locks
    // ----------------
    // Id: 1, Init: +1
    // Id: 2, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_0_0_id0;
    bd_memtile_0_0_id0.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_0_0_id0.length = 16;
    bd_memtile_0_0_id0.stepsize = {1};
    bd_memtile_0_0_id0.wrap = {};
    bd_memtile_0_0_id0.padding = {};
    bd_memtile_0_0_id0.lock_acq_enable = true;
    bd_memtile_0_0_id0.lock_acq_value = -1;
    bd_memtile_0_0_id0.lock_acq_id = 64 + 1;
    bd_memtile_0_0_id0.lock_rel_value = +1;
    bd_memtile_0_0_id0.lock_rel_id = 64 + 2;
    bd_memtile_0_0_id0.use_next_bd = false;
    bd_memtile_0_0_id0.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 0, bd_memtile_0_0_id0);

    adf::dma_buffer_descriptor bd_memtile_0_0_id1;
    bd_memtile_0_0_id1.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_0_0_id1.length = 512;
    bd_memtile_0_0_id1.stepsize = {1};
    bd_memtile_0_0_id1.wrap = {};
    bd_memtile_0_0_id1.padding = {};
    bd_memtile_0_0_id1.lock_acq_enable = true;
    bd_memtile_0_0_id1.lock_acq_value = -1;
    bd_memtile_0_0_id1.lock_acq_id = 64 + 2;
    bd_memtile_0_0_id1.lock_rel_value = +1;
    bd_memtile_0_0_id1.lock_rel_id = 64 + 1;
    bd_memtile_0_0_id1.use_next_bd = false;
    bd_memtile_0_0_id1.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 1, bd_memtile_0_0_id1);

    adf::initializeLock(adf::memory_tile, 0, 0, 1, +1);

    adf::initializeLock(adf::memory_tile, 0, 0, 2, +0);

    //
    // 1 to 1 Data Transfer
    //
    // Location: memtile_1_0
    //
    // Writers
    // ----------------
    // s2mm_0 BDs: 0
    //
    // Readers
    // ----------------
    // mm2s_0 BDs: 1
    //
    // Locks
    // ----------------
    // Id: 1, Init: +1
    // Id: 2, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_1_0_id0;
    bd_memtile_1_0_id0.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_1_0_id0.length = 16;
    bd_memtile_1_0_id0.stepsize = {1};
    bd_memtile_1_0_id0.wrap = {};
    bd_memtile_1_0_id0.padding = {};
    bd_memtile_1_0_id0.lock_acq_enable = true;
    bd_memtile_1_0_id0.lock_acq_value = -1;
    bd_memtile_1_0_id0.lock_acq_id = 64 + 1;
    bd_memtile_1_0_id0.lock_rel_value = +1;
    bd_memtile_1_0_id0.lock_rel_id = 64 + 2;
    bd_memtile_1_0_id0.use_next_bd = false;
    bd_memtile_1_0_id0.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 0, bd_memtile_1_0_id0);

    adf::dma_buffer_descriptor bd_memtile_1_0_id1;
    bd_memtile_1_0_id1.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_1_0_id1.length = 512;
    bd_memtile_1_0_id1.stepsize = {1};
    bd_memtile_1_0_id1.wrap = {};
    bd_memtile_1_0_id1.padding = {};
    bd_memtile_1_0_id1.lock_acq_enable = true;
    bd_memtile_1_0_id1.lock_acq_value = -1;
    bd_memtile_1_0_id1.lock_acq_id = 64 + 2;
    bd_memtile_1_0_id1.lock_rel_value = +1;
    bd_memtile_1_0_id1.lock_rel_id = 64 + 1;
    bd_memtile_1_0_id1.use_next_bd = false;
    bd_memtile_1_0_id1.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 1, bd_memtile_1_0_id1);

    adf::initializeLock(adf::memory_tile, 1, 0, 1, +1);

    adf::initializeLock(adf::memory_tile, 1, 0, 2, +0);

    //
    // 1 to 1 Data Transfer
    //
    // Location: memtile_2_0
    //
    // Writers
    // ----------------
    // s2mm_0 BDs: 0
    //
    // Readers
    // ----------------
    // mm2s_0 BDs: 1
    //
    // Locks
    // ----------------
    // Id: 1, Init: +1
    // Id: 2, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_2_0_id0;
    bd_memtile_2_0_id0.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_2_0_id0.length = 16;
    bd_memtile_2_0_id0.stepsize = {1};
    bd_memtile_2_0_id0.wrap = {};
    bd_memtile_2_0_id0.padding = {};
    bd_memtile_2_0_id0.lock_acq_enable = true;
    bd_memtile_2_0_id0.lock_acq_value = -1;
    bd_memtile_2_0_id0.lock_acq_id = 64 + 1;
    bd_memtile_2_0_id0.lock_rel_value = +1;
    bd_memtile_2_0_id0.lock_rel_id = 64 + 2;
    bd_memtile_2_0_id0.use_next_bd = false;
    bd_memtile_2_0_id0.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 0, bd_memtile_2_0_id0);

    adf::dma_buffer_descriptor bd_memtile_2_0_id1;
    bd_memtile_2_0_id1.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_2_0_id1.length = 512;
    bd_memtile_2_0_id1.stepsize = {1};
    bd_memtile_2_0_id1.wrap = {};
    bd_memtile_2_0_id1.padding = {};
    bd_memtile_2_0_id1.lock_acq_enable = true;
    bd_memtile_2_0_id1.lock_acq_value = -1;
    bd_memtile_2_0_id1.lock_acq_id = 64 + 2;
    bd_memtile_2_0_id1.lock_rel_value = +1;
    bd_memtile_2_0_id1.lock_rel_id = 64 + 1;
    bd_memtile_2_0_id1.use_next_bd = false;
    bd_memtile_2_0_id1.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 1, bd_memtile_2_0_id1);

    adf::initializeLock(adf::memory_tile, 2, 0, 1, +1);

    adf::initializeLock(adf::memory_tile, 2, 0, 2, +0);

    //
    // 1 to 1 Data Transfer
    //
    // Location: memtile_3_0
    //
    // Writers
    // ----------------
    // s2mm_0 BDs: 0
    //
    // Readers
    // ----------------
    // mm2s_0 BDs: 1
    //
    // Locks
    // ----------------
    // Id: 1, Init: +1
    // Id: 2, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_3_0_id0;
    bd_memtile_3_0_id0.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_3_0_id0.length = 16;
    bd_memtile_3_0_id0.stepsize = {1};
    bd_memtile_3_0_id0.wrap = {};
    bd_memtile_3_0_id0.padding = {};
    bd_memtile_3_0_id0.lock_acq_enable = true;
    bd_memtile_3_0_id0.lock_acq_value = -1;
    bd_memtile_3_0_id0.lock_acq_id = 64 + 1;
    bd_memtile_3_0_id0.lock_rel_value = +1;
    bd_memtile_3_0_id0.lock_rel_id = 64 + 2;
    bd_memtile_3_0_id0.use_next_bd = false;
    bd_memtile_3_0_id0.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 0, bd_memtile_3_0_id0);

    adf::dma_buffer_descriptor bd_memtile_3_0_id1;
    bd_memtile_3_0_id1.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_3_0_id1.length = 512;
    bd_memtile_3_0_id1.stepsize = {1};
    bd_memtile_3_0_id1.wrap = {};
    bd_memtile_3_0_id1.padding = {};
    bd_memtile_3_0_id1.lock_acq_enable = true;
    bd_memtile_3_0_id1.lock_acq_value = -1;
    bd_memtile_3_0_id1.lock_acq_id = 64 + 2;
    bd_memtile_3_0_id1.lock_rel_value = +1;
    bd_memtile_3_0_id1.lock_rel_id = 64 + 1;
    bd_memtile_3_0_id1.use_next_bd = false;
    bd_memtile_3_0_id1.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 1, bd_memtile_3_0_id1);

    adf::initializeLock(adf::memory_tile, 3, 0, 1, +1);

    adf::initializeLock(adf::memory_tile, 3, 0, 2, +0);

    //
    // 1 to 1 Data Transfer
    //
    // Location: memtile_0_0
    //
    // Writers
    // ----------------
    // s2mm_0 BDs: 2
    //
    // Readers
    // ----------------
    // mm2s_0 BDs: 3 -> 4 -> 5 -> 6
    //
    // Locks
    // ----------------
    // Id: 3, Init: +2
    // Id: 4, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_0_0_id2;
    bd_memtile_0_0_id2.address = 0 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_0_0_id2.length = 2048;
    bd_memtile_0_0_id2.stepsize = {1};
    bd_memtile_0_0_id2.wrap = {};
    bd_memtile_0_0_id2.padding = {};
    bd_memtile_0_0_id2.lock_acq_enable = true;
    bd_memtile_0_0_id2.lock_acq_value = -2;
    bd_memtile_0_0_id2.lock_acq_id = 64 + 3;
    bd_memtile_0_0_id2.lock_rel_value = +2;
    bd_memtile_0_0_id2.lock_rel_id = 64 + 4;
    bd_memtile_0_0_id2.use_next_bd = false;
    bd_memtile_0_0_id2.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 2, bd_memtile_0_0_id2);

    adf::dma_buffer_descriptor bd_memtile_0_0_id3;
    bd_memtile_0_0_id3.address = 0 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_0_0_id3.length = 16384;
    bd_memtile_0_0_id3.stepsize = {1, 16};
    bd_memtile_0_0_id3.wrap = {512};
    bd_memtile_0_0_id3.padding = {};
    bd_memtile_0_0_id3.lock_acq_enable = true;
    bd_memtile_0_0_id3.lock_acq_value = -1;
    bd_memtile_0_0_id3.lock_acq_id = 64 + 4;
    bd_memtile_0_0_id3.lock_rel_value = +0;
    bd_memtile_0_0_id3.lock_rel_id = 64 + 0;
    bd_memtile_0_0_id3.use_next_bd = true;
    bd_memtile_0_0_id3.next_bd = 4;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 3, bd_memtile_0_0_id3);

    adf::dma_buffer_descriptor bd_memtile_0_0_id4;
    bd_memtile_0_0_id4.address = 512 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_0_0_id4.length = 16384;
    bd_memtile_0_0_id4.stepsize = {1, 16};
    bd_memtile_0_0_id4.wrap = {512};
    bd_memtile_0_0_id4.padding = {};
    bd_memtile_0_0_id4.lock_acq_enable = false;
    bd_memtile_0_0_id4.lock_acq_value = +0;
    bd_memtile_0_0_id4.lock_acq_id = 64 + 0;
    bd_memtile_0_0_id4.lock_rel_value = +0;
    bd_memtile_0_0_id4.lock_rel_id = 64 + 0;
    bd_memtile_0_0_id4.use_next_bd = true;
    bd_memtile_0_0_id4.next_bd = 5;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 4, bd_memtile_0_0_id4);

    adf::dma_buffer_descriptor bd_memtile_0_0_id5;
    bd_memtile_0_0_id5.address = 1024 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_0_0_id5.length = 16384;
    bd_memtile_0_0_id5.stepsize = {1, 16};
    bd_memtile_0_0_id5.wrap = {512};
    bd_memtile_0_0_id5.padding = {};
    bd_memtile_0_0_id5.lock_acq_enable = false;
    bd_memtile_0_0_id5.lock_acq_value = +0;
    bd_memtile_0_0_id5.lock_acq_id = 64 + 0;
    bd_memtile_0_0_id5.lock_rel_value = +0;
    bd_memtile_0_0_id5.lock_rel_id = 64 + 0;
    bd_memtile_0_0_id5.use_next_bd = true;
    bd_memtile_0_0_id5.next_bd = 6;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 5, bd_memtile_0_0_id5);

    adf::dma_buffer_descriptor bd_memtile_0_0_id6;
    bd_memtile_0_0_id6.address = 1536 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_0_0_id6.length = 16384;
    bd_memtile_0_0_id6.stepsize = {1, 16};
    bd_memtile_0_0_id6.wrap = {512};
    bd_memtile_0_0_id6.padding = {};
    bd_memtile_0_0_id6.lock_acq_enable = true;
    bd_memtile_0_0_id6.lock_acq_value = +0;
    bd_memtile_0_0_id6.lock_acq_id = 64 + 0;
    bd_memtile_0_0_id6.lock_rel_value = +1;
    bd_memtile_0_0_id6.lock_rel_id = 64 + 3;
    bd_memtile_0_0_id6.use_next_bd = false;
    bd_memtile_0_0_id6.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 6, bd_memtile_0_0_id6);

    adf::initializeLock(adf::memory_tile, 0, 0, 3, +2);

    adf::initializeLock(adf::memory_tile, 0, 0, 4, +0);

    //
    // 1 to 1 Data Transfer
    //
    // Location: memtile_1_0
    //
    // Writers
    // ----------------
    // s2mm_0 BDs: 2
    //
    // Readers
    // ----------------
    // mm2s_0 BDs: 3 -> 4 -> 5 -> 6
    //
    // Locks
    // ----------------
    // Id: 3, Init: +2
    // Id: 4, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_1_0_id2;
    bd_memtile_1_0_id2.address = 0 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_1_0_id2.length = 2048;
    bd_memtile_1_0_id2.stepsize = {1};
    bd_memtile_1_0_id2.wrap = {};
    bd_memtile_1_0_id2.padding = {};
    bd_memtile_1_0_id2.lock_acq_enable = true;
    bd_memtile_1_0_id2.lock_acq_value = -2;
    bd_memtile_1_0_id2.lock_acq_id = 64 + 3;
    bd_memtile_1_0_id2.lock_rel_value = +2;
    bd_memtile_1_0_id2.lock_rel_id = 64 + 4;
    bd_memtile_1_0_id2.use_next_bd = false;
    bd_memtile_1_0_id2.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 2, bd_memtile_1_0_id2);

    adf::dma_buffer_descriptor bd_memtile_1_0_id3;
    bd_memtile_1_0_id3.address = 0 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_1_0_id3.length = 16384;
    bd_memtile_1_0_id3.stepsize = {1, 16};
    bd_memtile_1_0_id3.wrap = {512};
    bd_memtile_1_0_id3.padding = {};
    bd_memtile_1_0_id3.lock_acq_enable = true;
    bd_memtile_1_0_id3.lock_acq_value = -1;
    bd_memtile_1_0_id3.lock_acq_id = 64 + 4;
    bd_memtile_1_0_id3.lock_rel_value = +0;
    bd_memtile_1_0_id3.lock_rel_id = 64 + 0;
    bd_memtile_1_0_id3.use_next_bd = true;
    bd_memtile_1_0_id3.next_bd = 4;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 3, bd_memtile_1_0_id3);

    adf::dma_buffer_descriptor bd_memtile_1_0_id4;
    bd_memtile_1_0_id4.address = 512 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_1_0_id4.length = 16384;
    bd_memtile_1_0_id4.stepsize = {1, 16};
    bd_memtile_1_0_id4.wrap = {512};
    bd_memtile_1_0_id4.padding = {};
    bd_memtile_1_0_id4.lock_acq_enable = false;
    bd_memtile_1_0_id4.lock_acq_value = +0;
    bd_memtile_1_0_id4.lock_acq_id = 64 + 0;
    bd_memtile_1_0_id4.lock_rel_value = +0;
    bd_memtile_1_0_id4.lock_rel_id = 64 + 0;
    bd_memtile_1_0_id4.use_next_bd = true;
    bd_memtile_1_0_id4.next_bd = 5;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 4, bd_memtile_1_0_id4);

    adf::dma_buffer_descriptor bd_memtile_1_0_id5;
    bd_memtile_1_0_id5.address = 1024 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_1_0_id5.length = 16384;
    bd_memtile_1_0_id5.stepsize = {1, 16};
    bd_memtile_1_0_id5.wrap = {512};
    bd_memtile_1_0_id5.padding = {};
    bd_memtile_1_0_id5.lock_acq_enable = false;
    bd_memtile_1_0_id5.lock_acq_value = +0;
    bd_memtile_1_0_id5.lock_acq_id = 64 + 0;
    bd_memtile_1_0_id5.lock_rel_value = +0;
    bd_memtile_1_0_id5.lock_rel_id = 64 + 0;
    bd_memtile_1_0_id5.use_next_bd = true;
    bd_memtile_1_0_id5.next_bd = 6;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 5, bd_memtile_1_0_id5);

    adf::dma_buffer_descriptor bd_memtile_1_0_id6;
    bd_memtile_1_0_id6.address = 1536 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_1_0_id6.length = 16384;
    bd_memtile_1_0_id6.stepsize = {1, 16};
    bd_memtile_1_0_id6.wrap = {512};
    bd_memtile_1_0_id6.padding = {};
    bd_memtile_1_0_id6.lock_acq_enable = true;
    bd_memtile_1_0_id6.lock_acq_value = +0;
    bd_memtile_1_0_id6.lock_acq_id = 64 + 0;
    bd_memtile_1_0_id6.lock_rel_value = +1;
    bd_memtile_1_0_id6.lock_rel_id = 64 + 3;
    bd_memtile_1_0_id6.use_next_bd = false;
    bd_memtile_1_0_id6.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 6, bd_memtile_1_0_id6);

    adf::initializeLock(adf::memory_tile, 1, 0, 3, +2);

    adf::initializeLock(adf::memory_tile, 1, 0, 4, +0);

    //
    // 1 to 1 Data Transfer
    //
    // Location: memtile_2_0
    //
    // Writers
    // ----------------
    // s2mm_0 BDs: 2
    //
    // Readers
    // ----------------
    // mm2s_0 BDs: 3 -> 4 -> 5 -> 6
    //
    // Locks
    // ----------------
    // Id: 3, Init: +2
    // Id: 4, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_2_0_id2;
    bd_memtile_2_0_id2.address = 0 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_2_0_id2.length = 2048;
    bd_memtile_2_0_id2.stepsize = {1};
    bd_memtile_2_0_id2.wrap = {};
    bd_memtile_2_0_id2.padding = {};
    bd_memtile_2_0_id2.lock_acq_enable = true;
    bd_memtile_2_0_id2.lock_acq_value = -2;
    bd_memtile_2_0_id2.lock_acq_id = 64 + 3;
    bd_memtile_2_0_id2.lock_rel_value = +2;
    bd_memtile_2_0_id2.lock_rel_id = 64 + 4;
    bd_memtile_2_0_id2.use_next_bd = false;
    bd_memtile_2_0_id2.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 2, bd_memtile_2_0_id2);

    adf::dma_buffer_descriptor bd_memtile_2_0_id3;
    bd_memtile_2_0_id3.address = 0 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_2_0_id3.length = 16384;
    bd_memtile_2_0_id3.stepsize = {1, 16};
    bd_memtile_2_0_id3.wrap = {512};
    bd_memtile_2_0_id3.padding = {};
    bd_memtile_2_0_id3.lock_acq_enable = true;
    bd_memtile_2_0_id3.lock_acq_value = -1;
    bd_memtile_2_0_id3.lock_acq_id = 64 + 4;
    bd_memtile_2_0_id3.lock_rel_value = +0;
    bd_memtile_2_0_id3.lock_rel_id = 64 + 0;
    bd_memtile_2_0_id3.use_next_bd = true;
    bd_memtile_2_0_id3.next_bd = 4;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 3, bd_memtile_2_0_id3);

    adf::dma_buffer_descriptor bd_memtile_2_0_id4;
    bd_memtile_2_0_id4.address = 512 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_2_0_id4.length = 16384;
    bd_memtile_2_0_id4.stepsize = {1, 16};
    bd_memtile_2_0_id4.wrap = {512};
    bd_memtile_2_0_id4.padding = {};
    bd_memtile_2_0_id4.lock_acq_enable = false;
    bd_memtile_2_0_id4.lock_acq_value = +0;
    bd_memtile_2_0_id4.lock_acq_id = 64 + 0;
    bd_memtile_2_0_id4.lock_rel_value = +0;
    bd_memtile_2_0_id4.lock_rel_id = 64 + 0;
    bd_memtile_2_0_id4.use_next_bd = true;
    bd_memtile_2_0_id4.next_bd = 5;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 4, bd_memtile_2_0_id4);

    adf::dma_buffer_descriptor bd_memtile_2_0_id5;
    bd_memtile_2_0_id5.address = 1024 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_2_0_id5.length = 16384;
    bd_memtile_2_0_id5.stepsize = {1, 16};
    bd_memtile_2_0_id5.wrap = {512};
    bd_memtile_2_0_id5.padding = {};
    bd_memtile_2_0_id5.lock_acq_enable = false;
    bd_memtile_2_0_id5.lock_acq_value = +0;
    bd_memtile_2_0_id5.lock_acq_id = 64 + 0;
    bd_memtile_2_0_id5.lock_rel_value = +0;
    bd_memtile_2_0_id5.lock_rel_id = 64 + 0;
    bd_memtile_2_0_id5.use_next_bd = true;
    bd_memtile_2_0_id5.next_bd = 6;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 5, bd_memtile_2_0_id5);

    adf::dma_buffer_descriptor bd_memtile_2_0_id6;
    bd_memtile_2_0_id6.address = 1536 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_2_0_id6.length = 16384;
    bd_memtile_2_0_id6.stepsize = {1, 16};
    bd_memtile_2_0_id6.wrap = {512};
    bd_memtile_2_0_id6.padding = {};
    bd_memtile_2_0_id6.lock_acq_enable = true;
    bd_memtile_2_0_id6.lock_acq_value = +0;
    bd_memtile_2_0_id6.lock_acq_id = 64 + 0;
    bd_memtile_2_0_id6.lock_rel_value = +1;
    bd_memtile_2_0_id6.lock_rel_id = 64 + 3;
    bd_memtile_2_0_id6.use_next_bd = false;
    bd_memtile_2_0_id6.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 6, bd_memtile_2_0_id6);

    adf::initializeLock(adf::memory_tile, 2, 0, 3, +2);

    adf::initializeLock(adf::memory_tile, 2, 0, 4, +0);

    //
    // 1 to 1 Data Transfer
    //
    // Location: memtile_3_0
    //
    // Writers
    // ----------------
    // s2mm_0 BDs: 2
    //
    // Readers
    // ----------------
    // mm2s_0 BDs: 3 -> 4 -> 5 -> 6
    //
    // Locks
    // ----------------
    // Id: 3, Init: +2
    // Id: 4, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_3_0_id2;
    bd_memtile_3_0_id2.address = 0 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_3_0_id2.length = 2048;
    bd_memtile_3_0_id2.stepsize = {1};
    bd_memtile_3_0_id2.wrap = {};
    bd_memtile_3_0_id2.padding = {};
    bd_memtile_3_0_id2.lock_acq_enable = true;
    bd_memtile_3_0_id2.lock_acq_value = -2;
    bd_memtile_3_0_id2.lock_acq_id = 64 + 3;
    bd_memtile_3_0_id2.lock_rel_value = +2;
    bd_memtile_3_0_id2.lock_rel_id = 64 + 4;
    bd_memtile_3_0_id2.use_next_bd = false;
    bd_memtile_3_0_id2.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 2, bd_memtile_3_0_id2);

    adf::dma_buffer_descriptor bd_memtile_3_0_id3;
    bd_memtile_3_0_id3.address = 0 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_3_0_id3.length = 16384;
    bd_memtile_3_0_id3.stepsize = {1, 16};
    bd_memtile_3_0_id3.wrap = {512};
    bd_memtile_3_0_id3.padding = {};
    bd_memtile_3_0_id3.lock_acq_enable = true;
    bd_memtile_3_0_id3.lock_acq_value = -1;
    bd_memtile_3_0_id3.lock_acq_id = 64 + 4;
    bd_memtile_3_0_id3.lock_rel_value = +0;
    bd_memtile_3_0_id3.lock_rel_id = 64 + 0;
    bd_memtile_3_0_id3.use_next_bd = true;
    bd_memtile_3_0_id3.next_bd = 4;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 3, bd_memtile_3_0_id3);

    adf::dma_buffer_descriptor bd_memtile_3_0_id4;
    bd_memtile_3_0_id4.address = 512 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_3_0_id4.length = 16384;
    bd_memtile_3_0_id4.stepsize = {1, 16};
    bd_memtile_3_0_id4.wrap = {512};
    bd_memtile_3_0_id4.padding = {};
    bd_memtile_3_0_id4.lock_acq_enable = false;
    bd_memtile_3_0_id4.lock_acq_value = +0;
    bd_memtile_3_0_id4.lock_acq_id = 64 + 0;
    bd_memtile_3_0_id4.lock_rel_value = +0;
    bd_memtile_3_0_id4.lock_rel_id = 64 + 0;
    bd_memtile_3_0_id4.use_next_bd = true;
    bd_memtile_3_0_id4.next_bd = 5;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 4, bd_memtile_3_0_id4);

    adf::dma_buffer_descriptor bd_memtile_3_0_id5;
    bd_memtile_3_0_id5.address = 1024 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_3_0_id5.length = 16384;
    bd_memtile_3_0_id5.stepsize = {1, 16};
    bd_memtile_3_0_id5.wrap = {512};
    bd_memtile_3_0_id5.padding = {};
    bd_memtile_3_0_id5.lock_acq_enable = false;
    bd_memtile_3_0_id5.lock_acq_value = +0;
    bd_memtile_3_0_id5.lock_acq_id = 64 + 0;
    bd_memtile_3_0_id5.lock_rel_value = +0;
    bd_memtile_3_0_id5.lock_rel_id = 64 + 0;
    bd_memtile_3_0_id5.use_next_bd = true;
    bd_memtile_3_0_id5.next_bd = 6;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 5, bd_memtile_3_0_id5);

    adf::dma_buffer_descriptor bd_memtile_3_0_id6;
    bd_memtile_3_0_id6.address = 1536 + ((0x80000 + 0x800) / sizeof(uint32_t));
    bd_memtile_3_0_id6.length = 16384;
    bd_memtile_3_0_id6.stepsize = {1, 16};
    bd_memtile_3_0_id6.wrap = {512};
    bd_memtile_3_0_id6.padding = {};
    bd_memtile_3_0_id6.lock_acq_enable = true;
    bd_memtile_3_0_id6.lock_acq_value = +0;
    bd_memtile_3_0_id6.lock_acq_id = 64 + 0;
    bd_memtile_3_0_id6.lock_rel_value = +1;
    bd_memtile_3_0_id6.lock_rel_id = 64 + 3;
    bd_memtile_3_0_id6.use_next_bd = false;
    bd_memtile_3_0_id6.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 6, bd_memtile_3_0_id6);

    adf::initializeLock(adf::memory_tile, 3, 0, 3, +2);

    adf::initializeLock(adf::memory_tile, 3, 0, 4, +0);

    //
    // 1 to 2 Data Transfer
    //
    // Location: memtile_0_0
    //
    // Writers
    // ----------------
    // s2mm_0 BDs: 7 -> 8 -> 10 -> 11
    //
    // Readers
    // ----------------
    // mm2s_1 BDs: 24 -> 25
    // mm2s_2 BDs: 9 -> 12
    //
    // Locks
    // ----------------
    // Id: 5, Init: +2
    // Id: 6, Init: +0
    // Id: 7, Init: +0
    // Id: 8, Init: +2
    // Id: 9, Init: +0
    // Id: 10, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_0_0_id7;
    bd_memtile_0_0_id7.address = 0 + ((0x80000 + 0x2800) / sizeof(uint32_t));
    bd_memtile_0_0_id7.length = 1184;
    bd_memtile_0_0_id7.stepsize = {1};
    bd_memtile_0_0_id7.wrap = {};
    bd_memtile_0_0_id7.padding = {};
    bd_memtile_0_0_id7.lock_acq_enable = true;
    bd_memtile_0_0_id7.lock_acq_value = -2;
    bd_memtile_0_0_id7.lock_acq_id = 64 + 5;
    bd_memtile_0_0_id7.lock_rel_value = +1;
    bd_memtile_0_0_id7.lock_rel_id = 64 + 6;
    bd_memtile_0_0_id7.use_next_bd = true;
    bd_memtile_0_0_id7.next_bd = 8;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 7, bd_memtile_0_0_id7);

    adf::dma_buffer_descriptor bd_memtile_0_0_id8;
    bd_memtile_0_0_id8.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_0_0_id8.length = 0;
    bd_memtile_0_0_id8.stepsize = {1};
    bd_memtile_0_0_id8.wrap = {};
    bd_memtile_0_0_id8.padding = {};
    bd_memtile_0_0_id8.lock_acq_enable = true;
    bd_memtile_0_0_id8.lock_acq_value = +0;
    bd_memtile_0_0_id8.lock_acq_id = 64 + 0;
    bd_memtile_0_0_id8.lock_rel_value = +1;
    bd_memtile_0_0_id8.lock_rel_id = 64 + 7;
    bd_memtile_0_0_id8.use_next_bd = true;
    bd_memtile_0_0_id8.next_bd = 10;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 8, bd_memtile_0_0_id8);

    adf::dma_buffer_descriptor bd_memtile_0_0_id24;
    bd_memtile_0_0_id24.address = 0 + ((0x80000 + 0x2800) / sizeof(uint32_t));
    bd_memtile_0_0_id24.length = 2128;
    bd_memtile_0_0_id24.stepsize = {1};
    bd_memtile_0_0_id24.wrap = {};
    bd_memtile_0_0_id24.padding = {};
    bd_memtile_0_0_id24.lock_acq_enable = true;
    bd_memtile_0_0_id24.lock_acq_value = -1;
    bd_memtile_0_0_id24.lock_acq_id = 64 + 6;
    bd_memtile_0_0_id24.lock_rel_value = +1;
    bd_memtile_0_0_id24.lock_rel_id = 64 + 5;
    bd_memtile_0_0_id24.use_next_bd = true;
    bd_memtile_0_0_id24.next_bd = 25;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 24, bd_memtile_0_0_id24);

    adf::dma_buffer_descriptor bd_memtile_0_0_id9;
    bd_memtile_0_0_id9.address = 592 + ((0x80000 + 0x2800) / sizeof(uint32_t));
    bd_memtile_0_0_id9.length = 2128;
    bd_memtile_0_0_id9.stepsize = {1};
    bd_memtile_0_0_id9.wrap = {};
    bd_memtile_0_0_id9.padding = {};
    bd_memtile_0_0_id9.lock_acq_enable = true;
    bd_memtile_0_0_id9.lock_acq_value = -1;
    bd_memtile_0_0_id9.lock_acq_id = 64 + 7;
    bd_memtile_0_0_id9.lock_rel_value = +1;
    bd_memtile_0_0_id9.lock_rel_id = 64 + 5;
    bd_memtile_0_0_id9.use_next_bd = true;
    bd_memtile_0_0_id9.next_bd = 12;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 9, bd_memtile_0_0_id9);

    adf::dma_buffer_descriptor bd_memtile_0_0_id10;
    bd_memtile_0_0_id10.address = 0 + ((0x80000 + 0x4d00) / sizeof(uint32_t));
    bd_memtile_0_0_id10.length = 1184;
    bd_memtile_0_0_id10.stepsize = {1};
    bd_memtile_0_0_id10.wrap = {};
    bd_memtile_0_0_id10.padding = {};
    bd_memtile_0_0_id10.lock_acq_enable = true;
    bd_memtile_0_0_id10.lock_acq_value = -2;
    bd_memtile_0_0_id10.lock_acq_id = 64 + 8;
    bd_memtile_0_0_id10.lock_rel_value = +1;
    bd_memtile_0_0_id10.lock_rel_id = 64 + 9;
    bd_memtile_0_0_id10.use_next_bd = true;
    bd_memtile_0_0_id10.next_bd = 11;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 10, bd_memtile_0_0_id10);

    adf::dma_buffer_descriptor bd_memtile_0_0_id11;
    bd_memtile_0_0_id11.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_0_0_id11.length = 0;
    bd_memtile_0_0_id11.stepsize = {1};
    bd_memtile_0_0_id11.wrap = {};
    bd_memtile_0_0_id11.padding = {};
    bd_memtile_0_0_id11.lock_acq_enable = true;
    bd_memtile_0_0_id11.lock_acq_value = +0;
    bd_memtile_0_0_id11.lock_acq_id = 64 + 0;
    bd_memtile_0_0_id11.lock_rel_value = +1;
    bd_memtile_0_0_id11.lock_rel_id = 64 + 10;
    bd_memtile_0_0_id11.use_next_bd = false;
    bd_memtile_0_0_id11.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 11, bd_memtile_0_0_id11);

    adf::dma_buffer_descriptor bd_memtile_0_0_id25;
    bd_memtile_0_0_id25.address = 0 + ((0x80000 + 0x4d00) / sizeof(uint32_t));
    bd_memtile_0_0_id25.length = 2128;
    bd_memtile_0_0_id25.stepsize = {1};
    bd_memtile_0_0_id25.wrap = {};
    bd_memtile_0_0_id25.padding = {};
    bd_memtile_0_0_id25.lock_acq_enable = true;
    bd_memtile_0_0_id25.lock_acq_value = -1;
    bd_memtile_0_0_id25.lock_acq_id = 64 + 9;
    bd_memtile_0_0_id25.lock_rel_value = +1;
    bd_memtile_0_0_id25.lock_rel_id = 64 + 8;
    bd_memtile_0_0_id25.use_next_bd = false;
    bd_memtile_0_0_id25.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 25, bd_memtile_0_0_id25);

    adf::dma_buffer_descriptor bd_memtile_0_0_id12;
    bd_memtile_0_0_id12.address = 592 + ((0x80000 + 0x4d00) / sizeof(uint32_t));
    bd_memtile_0_0_id12.length = 2128;
    bd_memtile_0_0_id12.stepsize = {1};
    bd_memtile_0_0_id12.wrap = {};
    bd_memtile_0_0_id12.padding = {};
    bd_memtile_0_0_id12.lock_acq_enable = true;
    bd_memtile_0_0_id12.lock_acq_value = -1;
    bd_memtile_0_0_id12.lock_acq_id = 64 + 10;
    bd_memtile_0_0_id12.lock_rel_value = +1;
    bd_memtile_0_0_id12.lock_rel_id = 64 + 8;
    bd_memtile_0_0_id12.use_next_bd = false;
    bd_memtile_0_0_id12.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 12, bd_memtile_0_0_id12);

    adf::initializeLock(adf::memory_tile, 0, 0, 5, +2);

    adf::initializeLock(adf::memory_tile, 0, 0, 6, +0);

    adf::initializeLock(adf::memory_tile, 0, 0, 7, +0);

    adf::initializeLock(adf::memory_tile, 0, 0, 8, +2);

    adf::initializeLock(adf::memory_tile, 0, 0, 9, +0);

    adf::initializeLock(adf::memory_tile, 0, 0, 10, +0);

    //
    // 1 to 2 Data Transfer
    //
    // Location: memtile_1_0
    //
    // Writers
    // ----------------
    // s2mm_0 BDs: 7 -> 8 -> 10 -> 11
    //
    // Readers
    // ----------------
    // mm2s_1 BDs: 24 -> 25
    // mm2s_2 BDs: 9 -> 12
    //
    // Locks
    // ----------------
    // Id: 5, Init: +2
    // Id: 6, Init: +0
    // Id: 7, Init: +0
    // Id: 8, Init: +2
    // Id: 9, Init: +0
    // Id: 10, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_1_0_id7;
    bd_memtile_1_0_id7.address = 0 + ((0x80000 + 0x2800) / sizeof(uint32_t));
    bd_memtile_1_0_id7.length = 1184;
    bd_memtile_1_0_id7.stepsize = {1};
    bd_memtile_1_0_id7.wrap = {};
    bd_memtile_1_0_id7.padding = {};
    bd_memtile_1_0_id7.lock_acq_enable = true;
    bd_memtile_1_0_id7.lock_acq_value = -2;
    bd_memtile_1_0_id7.lock_acq_id = 64 + 5;
    bd_memtile_1_0_id7.lock_rel_value = +1;
    bd_memtile_1_0_id7.lock_rel_id = 64 + 6;
    bd_memtile_1_0_id7.use_next_bd = true;
    bd_memtile_1_0_id7.next_bd = 8;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 7, bd_memtile_1_0_id7);

    adf::dma_buffer_descriptor bd_memtile_1_0_id8;
    bd_memtile_1_0_id8.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_1_0_id8.length = 0;
    bd_memtile_1_0_id8.stepsize = {1};
    bd_memtile_1_0_id8.wrap = {};
    bd_memtile_1_0_id8.padding = {};
    bd_memtile_1_0_id8.lock_acq_enable = true;
    bd_memtile_1_0_id8.lock_acq_value = +0;
    bd_memtile_1_0_id8.lock_acq_id = 64 + 0;
    bd_memtile_1_0_id8.lock_rel_value = +1;
    bd_memtile_1_0_id8.lock_rel_id = 64 + 7;
    bd_memtile_1_0_id8.use_next_bd = true;
    bd_memtile_1_0_id8.next_bd = 10;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 8, bd_memtile_1_0_id8);

    adf::dma_buffer_descriptor bd_memtile_1_0_id24;
    bd_memtile_1_0_id24.address = 0 + ((0x80000 + 0x2800) / sizeof(uint32_t));
    bd_memtile_1_0_id24.length = 2128;
    bd_memtile_1_0_id24.stepsize = {1};
    bd_memtile_1_0_id24.wrap = {};
    bd_memtile_1_0_id24.padding = {};
    bd_memtile_1_0_id24.lock_acq_enable = true;
    bd_memtile_1_0_id24.lock_acq_value = -1;
    bd_memtile_1_0_id24.lock_acq_id = 64 + 6;
    bd_memtile_1_0_id24.lock_rel_value = +1;
    bd_memtile_1_0_id24.lock_rel_id = 64 + 5;
    bd_memtile_1_0_id24.use_next_bd = true;
    bd_memtile_1_0_id24.next_bd = 25;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 24, bd_memtile_1_0_id24);

    adf::dma_buffer_descriptor bd_memtile_1_0_id9;
    bd_memtile_1_0_id9.address = 592 + ((0x80000 + 0x2800) / sizeof(uint32_t));
    bd_memtile_1_0_id9.length = 2128;
    bd_memtile_1_0_id9.stepsize = {1};
    bd_memtile_1_0_id9.wrap = {};
    bd_memtile_1_0_id9.padding = {};
    bd_memtile_1_0_id9.lock_acq_enable = true;
    bd_memtile_1_0_id9.lock_acq_value = -1;
    bd_memtile_1_0_id9.lock_acq_id = 64 + 7;
    bd_memtile_1_0_id9.lock_rel_value = +1;
    bd_memtile_1_0_id9.lock_rel_id = 64 + 5;
    bd_memtile_1_0_id9.use_next_bd = true;
    bd_memtile_1_0_id9.next_bd = 12;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 9, bd_memtile_1_0_id9);

    adf::dma_buffer_descriptor bd_memtile_1_0_id10;
    bd_memtile_1_0_id10.address = 0 + ((0x80000 + 0x4d00) / sizeof(uint32_t));
    bd_memtile_1_0_id10.length = 1184;
    bd_memtile_1_0_id10.stepsize = {1};
    bd_memtile_1_0_id10.wrap = {};
    bd_memtile_1_0_id10.padding = {};
    bd_memtile_1_0_id10.lock_acq_enable = true;
    bd_memtile_1_0_id10.lock_acq_value = -2;
    bd_memtile_1_0_id10.lock_acq_id = 64 + 8;
    bd_memtile_1_0_id10.lock_rel_value = +1;
    bd_memtile_1_0_id10.lock_rel_id = 64 + 9;
    bd_memtile_1_0_id10.use_next_bd = true;
    bd_memtile_1_0_id10.next_bd = 11;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 10, bd_memtile_1_0_id10);

    adf::dma_buffer_descriptor bd_memtile_1_0_id11;
    bd_memtile_1_0_id11.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_1_0_id11.length = 0;
    bd_memtile_1_0_id11.stepsize = {1};
    bd_memtile_1_0_id11.wrap = {};
    bd_memtile_1_0_id11.padding = {};
    bd_memtile_1_0_id11.lock_acq_enable = true;
    bd_memtile_1_0_id11.lock_acq_value = +0;
    bd_memtile_1_0_id11.lock_acq_id = 64 + 0;
    bd_memtile_1_0_id11.lock_rel_value = +1;
    bd_memtile_1_0_id11.lock_rel_id = 64 + 10;
    bd_memtile_1_0_id11.use_next_bd = false;
    bd_memtile_1_0_id11.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 11, bd_memtile_1_0_id11);

    adf::dma_buffer_descriptor bd_memtile_1_0_id25;
    bd_memtile_1_0_id25.address = 0 + ((0x80000 + 0x4d00) / sizeof(uint32_t));
    bd_memtile_1_0_id25.length = 2128;
    bd_memtile_1_0_id25.stepsize = {1};
    bd_memtile_1_0_id25.wrap = {};
    bd_memtile_1_0_id25.padding = {};
    bd_memtile_1_0_id25.lock_acq_enable = true;
    bd_memtile_1_0_id25.lock_acq_value = -1;
    bd_memtile_1_0_id25.lock_acq_id = 64 + 9;
    bd_memtile_1_0_id25.lock_rel_value = +1;
    bd_memtile_1_0_id25.lock_rel_id = 64 + 8;
    bd_memtile_1_0_id25.use_next_bd = false;
    bd_memtile_1_0_id25.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 25, bd_memtile_1_0_id25);

    adf::dma_buffer_descriptor bd_memtile_1_0_id12;
    bd_memtile_1_0_id12.address = 592 + ((0x80000 + 0x4d00) / sizeof(uint32_t));
    bd_memtile_1_0_id12.length = 2128;
    bd_memtile_1_0_id12.stepsize = {1};
    bd_memtile_1_0_id12.wrap = {};
    bd_memtile_1_0_id12.padding = {};
    bd_memtile_1_0_id12.lock_acq_enable = true;
    bd_memtile_1_0_id12.lock_acq_value = -1;
    bd_memtile_1_0_id12.lock_acq_id = 64 + 10;
    bd_memtile_1_0_id12.lock_rel_value = +1;
    bd_memtile_1_0_id12.lock_rel_id = 64 + 8;
    bd_memtile_1_0_id12.use_next_bd = false;
    bd_memtile_1_0_id12.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 12, bd_memtile_1_0_id12);

    adf::initializeLock(adf::memory_tile, 1, 0, 5, +2);

    adf::initializeLock(adf::memory_tile, 1, 0, 6, +0);

    adf::initializeLock(adf::memory_tile, 1, 0, 7, +0);

    adf::initializeLock(adf::memory_tile, 1, 0, 8, +2);

    adf::initializeLock(adf::memory_tile, 1, 0, 9, +0);

    adf::initializeLock(adf::memory_tile, 1, 0, 10, +0);

    //
    // 1 to 2 Data Transfer
    //
    // Location: memtile_2_0
    //
    // Writers
    // ----------------
    // s2mm_0 BDs: 7 -> 8 -> 10 -> 11
    //
    // Readers
    // ----------------
    // mm2s_1 BDs: 24 -> 25
    // mm2s_2 BDs: 9 -> 12
    //
    // Locks
    // ----------------
    // Id: 5, Init: +2
    // Id: 6, Init: +0
    // Id: 7, Init: +0
    // Id: 8, Init: +2
    // Id: 9, Init: +0
    // Id: 10, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_2_0_id7;
    bd_memtile_2_0_id7.address = 0 + ((0x80000 + 0x2800) / sizeof(uint32_t));
    bd_memtile_2_0_id7.length = 1184;
    bd_memtile_2_0_id7.stepsize = {1};
    bd_memtile_2_0_id7.wrap = {};
    bd_memtile_2_0_id7.padding = {};
    bd_memtile_2_0_id7.lock_acq_enable = true;
    bd_memtile_2_0_id7.lock_acq_value = -2;
    bd_memtile_2_0_id7.lock_acq_id = 64 + 5;
    bd_memtile_2_0_id7.lock_rel_value = +1;
    bd_memtile_2_0_id7.lock_rel_id = 64 + 6;
    bd_memtile_2_0_id7.use_next_bd = true;
    bd_memtile_2_0_id7.next_bd = 8;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 7, bd_memtile_2_0_id7);

    adf::dma_buffer_descriptor bd_memtile_2_0_id8;
    bd_memtile_2_0_id8.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_2_0_id8.length = 0;
    bd_memtile_2_0_id8.stepsize = {1};
    bd_memtile_2_0_id8.wrap = {};
    bd_memtile_2_0_id8.padding = {};
    bd_memtile_2_0_id8.lock_acq_enable = true;
    bd_memtile_2_0_id8.lock_acq_value = +0;
    bd_memtile_2_0_id8.lock_acq_id = 64 + 0;
    bd_memtile_2_0_id8.lock_rel_value = +1;
    bd_memtile_2_0_id8.lock_rel_id = 64 + 7;
    bd_memtile_2_0_id8.use_next_bd = true;
    bd_memtile_2_0_id8.next_bd = 10;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 8, bd_memtile_2_0_id8);

    adf::dma_buffer_descriptor bd_memtile_2_0_id24;
    bd_memtile_2_0_id24.address = 0 + ((0x80000 + 0x2800) / sizeof(uint32_t));
    bd_memtile_2_0_id24.length = 2128;
    bd_memtile_2_0_id24.stepsize = {1};
    bd_memtile_2_0_id24.wrap = {};
    bd_memtile_2_0_id24.padding = {};
    bd_memtile_2_0_id24.lock_acq_enable = true;
    bd_memtile_2_0_id24.lock_acq_value = -1;
    bd_memtile_2_0_id24.lock_acq_id = 64 + 6;
    bd_memtile_2_0_id24.lock_rel_value = +1;
    bd_memtile_2_0_id24.lock_rel_id = 64 + 5;
    bd_memtile_2_0_id24.use_next_bd = true;
    bd_memtile_2_0_id24.next_bd = 25;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 24, bd_memtile_2_0_id24);

    adf::dma_buffer_descriptor bd_memtile_2_0_id9;
    bd_memtile_2_0_id9.address = 592 + ((0x80000 + 0x2800) / sizeof(uint32_t));
    bd_memtile_2_0_id9.length = 2128;
    bd_memtile_2_0_id9.stepsize = {1};
    bd_memtile_2_0_id9.wrap = {};
    bd_memtile_2_0_id9.padding = {};
    bd_memtile_2_0_id9.lock_acq_enable = true;
    bd_memtile_2_0_id9.lock_acq_value = -1;
    bd_memtile_2_0_id9.lock_acq_id = 64 + 7;
    bd_memtile_2_0_id9.lock_rel_value = +1;
    bd_memtile_2_0_id9.lock_rel_id = 64 + 5;
    bd_memtile_2_0_id9.use_next_bd = true;
    bd_memtile_2_0_id9.next_bd = 12;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 9, bd_memtile_2_0_id9);

    adf::dma_buffer_descriptor bd_memtile_2_0_id10;
    bd_memtile_2_0_id10.address = 0 + ((0x80000 + 0x4d00) / sizeof(uint32_t));
    bd_memtile_2_0_id10.length = 1184;
    bd_memtile_2_0_id10.stepsize = {1};
    bd_memtile_2_0_id10.wrap = {};
    bd_memtile_2_0_id10.padding = {};
    bd_memtile_2_0_id10.lock_acq_enable = true;
    bd_memtile_2_0_id10.lock_acq_value = -2;
    bd_memtile_2_0_id10.lock_acq_id = 64 + 8;
    bd_memtile_2_0_id10.lock_rel_value = +1;
    bd_memtile_2_0_id10.lock_rel_id = 64 + 9;
    bd_memtile_2_0_id10.use_next_bd = true;
    bd_memtile_2_0_id10.next_bd = 11;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 10, bd_memtile_2_0_id10);

    adf::dma_buffer_descriptor bd_memtile_2_0_id11;
    bd_memtile_2_0_id11.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_2_0_id11.length = 0;
    bd_memtile_2_0_id11.stepsize = {1};
    bd_memtile_2_0_id11.wrap = {};
    bd_memtile_2_0_id11.padding = {};
    bd_memtile_2_0_id11.lock_acq_enable = true;
    bd_memtile_2_0_id11.lock_acq_value = +0;
    bd_memtile_2_0_id11.lock_acq_id = 64 + 0;
    bd_memtile_2_0_id11.lock_rel_value = +1;
    bd_memtile_2_0_id11.lock_rel_id = 64 + 10;
    bd_memtile_2_0_id11.use_next_bd = false;
    bd_memtile_2_0_id11.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 11, bd_memtile_2_0_id11);

    adf::dma_buffer_descriptor bd_memtile_2_0_id25;
    bd_memtile_2_0_id25.address = 0 + ((0x80000 + 0x4d00) / sizeof(uint32_t));
    bd_memtile_2_0_id25.length = 2128;
    bd_memtile_2_0_id25.stepsize = {1};
    bd_memtile_2_0_id25.wrap = {};
    bd_memtile_2_0_id25.padding = {};
    bd_memtile_2_0_id25.lock_acq_enable = true;
    bd_memtile_2_0_id25.lock_acq_value = -1;
    bd_memtile_2_0_id25.lock_acq_id = 64 + 9;
    bd_memtile_2_0_id25.lock_rel_value = +1;
    bd_memtile_2_0_id25.lock_rel_id = 64 + 8;
    bd_memtile_2_0_id25.use_next_bd = false;
    bd_memtile_2_0_id25.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 25, bd_memtile_2_0_id25);

    adf::dma_buffer_descriptor bd_memtile_2_0_id12;
    bd_memtile_2_0_id12.address = 592 + ((0x80000 + 0x4d00) / sizeof(uint32_t));
    bd_memtile_2_0_id12.length = 2128;
    bd_memtile_2_0_id12.stepsize = {1};
    bd_memtile_2_0_id12.wrap = {};
    bd_memtile_2_0_id12.padding = {};
    bd_memtile_2_0_id12.lock_acq_enable = true;
    bd_memtile_2_0_id12.lock_acq_value = -1;
    bd_memtile_2_0_id12.lock_acq_id = 64 + 10;
    bd_memtile_2_0_id12.lock_rel_value = +1;
    bd_memtile_2_0_id12.lock_rel_id = 64 + 8;
    bd_memtile_2_0_id12.use_next_bd = false;
    bd_memtile_2_0_id12.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 12, bd_memtile_2_0_id12);

    adf::initializeLock(adf::memory_tile, 2, 0, 5, +2);

    adf::initializeLock(adf::memory_tile, 2, 0, 6, +0);

    adf::initializeLock(adf::memory_tile, 2, 0, 7, +0);

    adf::initializeLock(adf::memory_tile, 2, 0, 8, +2);

    adf::initializeLock(adf::memory_tile, 2, 0, 9, +0);

    adf::initializeLock(adf::memory_tile, 2, 0, 10, +0);

    //
    // 1 to 2 Data Transfer
    //
    // Location: memtile_3_0
    //
    // Writers
    // ----------------
    // s2mm_0 BDs: 7 -> 8 -> 10 -> 11
    //
    // Readers
    // ----------------
    // mm2s_1 BDs: 24 -> 25
    // mm2s_2 BDs: 9 -> 12
    //
    // Locks
    // ----------------
    // Id: 5, Init: +2
    // Id: 6, Init: +0
    // Id: 7, Init: +0
    // Id: 8, Init: +2
    // Id: 9, Init: +0
    // Id: 10, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_3_0_id7;
    bd_memtile_3_0_id7.address = 0 + ((0x80000 + 0x2800) / sizeof(uint32_t));
    bd_memtile_3_0_id7.length = 1184;
    bd_memtile_3_0_id7.stepsize = {1};
    bd_memtile_3_0_id7.wrap = {};
    bd_memtile_3_0_id7.padding = {};
    bd_memtile_3_0_id7.lock_acq_enable = true;
    bd_memtile_3_0_id7.lock_acq_value = -2;
    bd_memtile_3_0_id7.lock_acq_id = 64 + 5;
    bd_memtile_3_0_id7.lock_rel_value = +1;
    bd_memtile_3_0_id7.lock_rel_id = 64 + 6;
    bd_memtile_3_0_id7.use_next_bd = true;
    bd_memtile_3_0_id7.next_bd = 8;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 7, bd_memtile_3_0_id7);

    adf::dma_buffer_descriptor bd_memtile_3_0_id8;
    bd_memtile_3_0_id8.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_3_0_id8.length = 0;
    bd_memtile_3_0_id8.stepsize = {1};
    bd_memtile_3_0_id8.wrap = {};
    bd_memtile_3_0_id8.padding = {};
    bd_memtile_3_0_id8.lock_acq_enable = true;
    bd_memtile_3_0_id8.lock_acq_value = +0;
    bd_memtile_3_0_id8.lock_acq_id = 64 + 0;
    bd_memtile_3_0_id8.lock_rel_value = +1;
    bd_memtile_3_0_id8.lock_rel_id = 64 + 7;
    bd_memtile_3_0_id8.use_next_bd = true;
    bd_memtile_3_0_id8.next_bd = 10;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 8, bd_memtile_3_0_id8);

    adf::dma_buffer_descriptor bd_memtile_3_0_id24;
    bd_memtile_3_0_id24.address = 0 + ((0x80000 + 0x2800) / sizeof(uint32_t));
    bd_memtile_3_0_id24.length = 2128;
    bd_memtile_3_0_id24.stepsize = {1};
    bd_memtile_3_0_id24.wrap = {};
    bd_memtile_3_0_id24.padding = {};
    bd_memtile_3_0_id24.lock_acq_enable = true;
    bd_memtile_3_0_id24.lock_acq_value = -1;
    bd_memtile_3_0_id24.lock_acq_id = 64 + 6;
    bd_memtile_3_0_id24.lock_rel_value = +1;
    bd_memtile_3_0_id24.lock_rel_id = 64 + 5;
    bd_memtile_3_0_id24.use_next_bd = true;
    bd_memtile_3_0_id24.next_bd = 25;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 24, bd_memtile_3_0_id24);

    adf::dma_buffer_descriptor bd_memtile_3_0_id9;
    bd_memtile_3_0_id9.address = 592 + ((0x80000 + 0x2800) / sizeof(uint32_t));
    bd_memtile_3_0_id9.length = 2128;
    bd_memtile_3_0_id9.stepsize = {1};
    bd_memtile_3_0_id9.wrap = {};
    bd_memtile_3_0_id9.padding = {};
    bd_memtile_3_0_id9.lock_acq_enable = true;
    bd_memtile_3_0_id9.lock_acq_value = -1;
    bd_memtile_3_0_id9.lock_acq_id = 64 + 7;
    bd_memtile_3_0_id9.lock_rel_value = +1;
    bd_memtile_3_0_id9.lock_rel_id = 64 + 5;
    bd_memtile_3_0_id9.use_next_bd = true;
    bd_memtile_3_0_id9.next_bd = 12;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 9, bd_memtile_3_0_id9);

    adf::dma_buffer_descriptor bd_memtile_3_0_id10;
    bd_memtile_3_0_id10.address = 0 + ((0x80000 + 0x4d00) / sizeof(uint32_t));
    bd_memtile_3_0_id10.length = 1184;
    bd_memtile_3_0_id10.stepsize = {1};
    bd_memtile_3_0_id10.wrap = {};
    bd_memtile_3_0_id10.padding = {};
    bd_memtile_3_0_id10.lock_acq_enable = true;
    bd_memtile_3_0_id10.lock_acq_value = -2;
    bd_memtile_3_0_id10.lock_acq_id = 64 + 8;
    bd_memtile_3_0_id10.lock_rel_value = +1;
    bd_memtile_3_0_id10.lock_rel_id = 64 + 9;
    bd_memtile_3_0_id10.use_next_bd = true;
    bd_memtile_3_0_id10.next_bd = 11;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 10, bd_memtile_3_0_id10);

    adf::dma_buffer_descriptor bd_memtile_3_0_id11;
    bd_memtile_3_0_id11.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_3_0_id11.length = 0;
    bd_memtile_3_0_id11.stepsize = {1};
    bd_memtile_3_0_id11.wrap = {};
    bd_memtile_3_0_id11.padding = {};
    bd_memtile_3_0_id11.lock_acq_enable = true;
    bd_memtile_3_0_id11.lock_acq_value = +0;
    bd_memtile_3_0_id11.lock_acq_id = 64 + 0;
    bd_memtile_3_0_id11.lock_rel_value = +1;
    bd_memtile_3_0_id11.lock_rel_id = 64 + 10;
    bd_memtile_3_0_id11.use_next_bd = false;
    bd_memtile_3_0_id11.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 11, bd_memtile_3_0_id11);

    adf::dma_buffer_descriptor bd_memtile_3_0_id25;
    bd_memtile_3_0_id25.address = 0 + ((0x80000 + 0x4d00) / sizeof(uint32_t));
    bd_memtile_3_0_id25.length = 2128;
    bd_memtile_3_0_id25.stepsize = {1};
    bd_memtile_3_0_id25.wrap = {};
    bd_memtile_3_0_id25.padding = {};
    bd_memtile_3_0_id25.lock_acq_enable = true;
    bd_memtile_3_0_id25.lock_acq_value = -1;
    bd_memtile_3_0_id25.lock_acq_id = 64 + 9;
    bd_memtile_3_0_id25.lock_rel_value = +1;
    bd_memtile_3_0_id25.lock_rel_id = 64 + 8;
    bd_memtile_3_0_id25.use_next_bd = false;
    bd_memtile_3_0_id25.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 25, bd_memtile_3_0_id25);

    adf::dma_buffer_descriptor bd_memtile_3_0_id12;
    bd_memtile_3_0_id12.address = 592 + ((0x80000 + 0x4d00) / sizeof(uint32_t));
    bd_memtile_3_0_id12.length = 2128;
    bd_memtile_3_0_id12.stepsize = {1};
    bd_memtile_3_0_id12.wrap = {};
    bd_memtile_3_0_id12.padding = {};
    bd_memtile_3_0_id12.lock_acq_enable = true;
    bd_memtile_3_0_id12.lock_acq_value = -1;
    bd_memtile_3_0_id12.lock_acq_id = 64 + 10;
    bd_memtile_3_0_id12.lock_rel_value = +1;
    bd_memtile_3_0_id12.lock_rel_id = 64 + 8;
    bd_memtile_3_0_id12.use_next_bd = false;
    bd_memtile_3_0_id12.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 12, bd_memtile_3_0_id12);

    adf::initializeLock(adf::memory_tile, 3, 0, 5, +2);

    adf::initializeLock(adf::memory_tile, 3, 0, 6, +0);

    adf::initializeLock(adf::memory_tile, 3, 0, 7, +0);

    adf::initializeLock(adf::memory_tile, 3, 0, 8, +2);

    adf::initializeLock(adf::memory_tile, 3, 0, 9, +0);

    adf::initializeLock(adf::memory_tile, 3, 0, 10, +0);

    //
    // 1 to 2 Data Transfer
    //
    // Location: memtile_0_0
    //
    // Writers
    // ----------------
    // s2mm_1 BDs: 26 -> 27 -> 29 -> 30
    //
    // Readers
    // ----------------
    // mm2s_3 BDs: 28 -> 31
    // mm2s_4 BDs: 13 -> 14
    //
    // Locks
    // ----------------
    // Id: 11, Init: +2
    // Id: 12, Init: +0
    // Id: 13, Init: +0
    // Id: 14, Init: +2
    // Id: 15, Init: +0
    // Id: 16, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_0_0_id26;
    bd_memtile_0_0_id26.address = 0 + ((0x80000 + 0x3a80) / sizeof(uint32_t));
    bd_memtile_0_0_id26.length = 1184;
    bd_memtile_0_0_id26.stepsize = {1};
    bd_memtile_0_0_id26.wrap = {};
    bd_memtile_0_0_id26.padding = {};
    bd_memtile_0_0_id26.lock_acq_enable = true;
    bd_memtile_0_0_id26.lock_acq_value = -2;
    bd_memtile_0_0_id26.lock_acq_id = 64 + 11;
    bd_memtile_0_0_id26.lock_rel_value = +1;
    bd_memtile_0_0_id26.lock_rel_id = 64 + 12;
    bd_memtile_0_0_id26.use_next_bd = true;
    bd_memtile_0_0_id26.next_bd = 27;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 26, bd_memtile_0_0_id26);

    adf::dma_buffer_descriptor bd_memtile_0_0_id27;
    bd_memtile_0_0_id27.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_0_0_id27.length = 0;
    bd_memtile_0_0_id27.stepsize = {1};
    bd_memtile_0_0_id27.wrap = {};
    bd_memtile_0_0_id27.padding = {};
    bd_memtile_0_0_id27.lock_acq_enable = true;
    bd_memtile_0_0_id27.lock_acq_value = +0;
    bd_memtile_0_0_id27.lock_acq_id = 64 + 0;
    bd_memtile_0_0_id27.lock_rel_value = +1;
    bd_memtile_0_0_id27.lock_rel_id = 64 + 13;
    bd_memtile_0_0_id27.use_next_bd = true;
    bd_memtile_0_0_id27.next_bd = 29;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 27, bd_memtile_0_0_id27);

    adf::dma_buffer_descriptor bd_memtile_0_0_id28;
    bd_memtile_0_0_id28.address = 0 + ((0x80000 + 0x3a80) / sizeof(uint32_t));
    bd_memtile_0_0_id28.length = 2128;
    bd_memtile_0_0_id28.stepsize = {1};
    bd_memtile_0_0_id28.wrap = {};
    bd_memtile_0_0_id28.padding = {};
    bd_memtile_0_0_id28.lock_acq_enable = true;
    bd_memtile_0_0_id28.lock_acq_value = -1;
    bd_memtile_0_0_id28.lock_acq_id = 64 + 12;
    bd_memtile_0_0_id28.lock_rel_value = +1;
    bd_memtile_0_0_id28.lock_rel_id = 64 + 11;
    bd_memtile_0_0_id28.use_next_bd = true;
    bd_memtile_0_0_id28.next_bd = 31;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 28, bd_memtile_0_0_id28);

    adf::dma_buffer_descriptor bd_memtile_0_0_id13;
    bd_memtile_0_0_id13.address = 592 + ((0x80000 + 0x3a80) / sizeof(uint32_t));
    bd_memtile_0_0_id13.length = 2128;
    bd_memtile_0_0_id13.stepsize = {1};
    bd_memtile_0_0_id13.wrap = {};
    bd_memtile_0_0_id13.padding = {};
    bd_memtile_0_0_id13.lock_acq_enable = true;
    bd_memtile_0_0_id13.lock_acq_value = -1;
    bd_memtile_0_0_id13.lock_acq_id = 64 + 13;
    bd_memtile_0_0_id13.lock_rel_value = +1;
    bd_memtile_0_0_id13.lock_rel_id = 64 + 11;
    bd_memtile_0_0_id13.use_next_bd = true;
    bd_memtile_0_0_id13.next_bd = 14;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 13, bd_memtile_0_0_id13);

    adf::dma_buffer_descriptor bd_memtile_0_0_id29;
    bd_memtile_0_0_id29.address = 0 + ((0x80000 + 0x5f80) / sizeof(uint32_t));
    bd_memtile_0_0_id29.length = 1184;
    bd_memtile_0_0_id29.stepsize = {1};
    bd_memtile_0_0_id29.wrap = {};
    bd_memtile_0_0_id29.padding = {};
    bd_memtile_0_0_id29.lock_acq_enable = true;
    bd_memtile_0_0_id29.lock_acq_value = -2;
    bd_memtile_0_0_id29.lock_acq_id = 64 + 14;
    bd_memtile_0_0_id29.lock_rel_value = +1;
    bd_memtile_0_0_id29.lock_rel_id = 64 + 15;
    bd_memtile_0_0_id29.use_next_bd = true;
    bd_memtile_0_0_id29.next_bd = 30;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 29, bd_memtile_0_0_id29);

    adf::dma_buffer_descriptor bd_memtile_0_0_id30;
    bd_memtile_0_0_id30.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_0_0_id30.length = 0;
    bd_memtile_0_0_id30.stepsize = {1};
    bd_memtile_0_0_id30.wrap = {};
    bd_memtile_0_0_id30.padding = {};
    bd_memtile_0_0_id30.lock_acq_enable = true;
    bd_memtile_0_0_id30.lock_acq_value = +0;
    bd_memtile_0_0_id30.lock_acq_id = 64 + 0;
    bd_memtile_0_0_id30.lock_rel_value = +1;
    bd_memtile_0_0_id30.lock_rel_id = 64 + 16;
    bd_memtile_0_0_id30.use_next_bd = false;
    bd_memtile_0_0_id30.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 30, bd_memtile_0_0_id30);

    adf::dma_buffer_descriptor bd_memtile_0_0_id31;
    bd_memtile_0_0_id31.address = 0 + ((0x80000 + 0x5f80) / sizeof(uint32_t));
    bd_memtile_0_0_id31.length = 2128;
    bd_memtile_0_0_id31.stepsize = {1};
    bd_memtile_0_0_id31.wrap = {};
    bd_memtile_0_0_id31.padding = {};
    bd_memtile_0_0_id31.lock_acq_enable = true;
    bd_memtile_0_0_id31.lock_acq_value = -1;
    bd_memtile_0_0_id31.lock_acq_id = 64 + 15;
    bd_memtile_0_0_id31.lock_rel_value = +1;
    bd_memtile_0_0_id31.lock_rel_id = 64 + 14;
    bd_memtile_0_0_id31.use_next_bd = false;
    bd_memtile_0_0_id31.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 31, bd_memtile_0_0_id31);

    adf::dma_buffer_descriptor bd_memtile_0_0_id14;
    bd_memtile_0_0_id14.address = 592 + ((0x80000 + 0x5f80) / sizeof(uint32_t));
    bd_memtile_0_0_id14.length = 2128;
    bd_memtile_0_0_id14.stepsize = {1};
    bd_memtile_0_0_id14.wrap = {};
    bd_memtile_0_0_id14.padding = {};
    bd_memtile_0_0_id14.lock_acq_enable = true;
    bd_memtile_0_0_id14.lock_acq_value = -1;
    bd_memtile_0_0_id14.lock_acq_id = 64 + 16;
    bd_memtile_0_0_id14.lock_rel_value = +1;
    bd_memtile_0_0_id14.lock_rel_id = 64 + 14;
    bd_memtile_0_0_id14.use_next_bd = false;
    bd_memtile_0_0_id14.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 14, bd_memtile_0_0_id14);

    adf::initializeLock(adf::memory_tile, 0, 0, 11, +2);

    adf::initializeLock(adf::memory_tile, 0, 0, 12, +0);

    adf::initializeLock(adf::memory_tile, 0, 0, 13, +0);

    adf::initializeLock(adf::memory_tile, 0, 0, 14, +2);

    adf::initializeLock(adf::memory_tile, 0, 0, 15, +0);

    adf::initializeLock(adf::memory_tile, 0, 0, 16, +0);

    //
    // 1 to 2 Data Transfer
    //
    // Location: memtile_1_0
    //
    // Writers
    // ----------------
    // s2mm_1 BDs: 26 -> 27 -> 29 -> 30
    //
    // Readers
    // ----------------
    // mm2s_3 BDs: 28 -> 31
    // mm2s_4 BDs: 13 -> 14
    //
    // Locks
    // ----------------
    // Id: 11, Init: +2
    // Id: 12, Init: +0
    // Id: 13, Init: +0
    // Id: 14, Init: +2
    // Id: 15, Init: +0
    // Id: 16, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_1_0_id26;
    bd_memtile_1_0_id26.address = 0 + ((0x80000 + 0x3a80) / sizeof(uint32_t));
    bd_memtile_1_0_id26.length = 1184;
    bd_memtile_1_0_id26.stepsize = {1};
    bd_memtile_1_0_id26.wrap = {};
    bd_memtile_1_0_id26.padding = {};
    bd_memtile_1_0_id26.lock_acq_enable = true;
    bd_memtile_1_0_id26.lock_acq_value = -2;
    bd_memtile_1_0_id26.lock_acq_id = 64 + 11;
    bd_memtile_1_0_id26.lock_rel_value = +1;
    bd_memtile_1_0_id26.lock_rel_id = 64 + 12;
    bd_memtile_1_0_id26.use_next_bd = true;
    bd_memtile_1_0_id26.next_bd = 27;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 26, bd_memtile_1_0_id26);

    adf::dma_buffer_descriptor bd_memtile_1_0_id27;
    bd_memtile_1_0_id27.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_1_0_id27.length = 0;
    bd_memtile_1_0_id27.stepsize = {1};
    bd_memtile_1_0_id27.wrap = {};
    bd_memtile_1_0_id27.padding = {};
    bd_memtile_1_0_id27.lock_acq_enable = true;
    bd_memtile_1_0_id27.lock_acq_value = +0;
    bd_memtile_1_0_id27.lock_acq_id = 64 + 0;
    bd_memtile_1_0_id27.lock_rel_value = +1;
    bd_memtile_1_0_id27.lock_rel_id = 64 + 13;
    bd_memtile_1_0_id27.use_next_bd = true;
    bd_memtile_1_0_id27.next_bd = 29;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 27, bd_memtile_1_0_id27);

    adf::dma_buffer_descriptor bd_memtile_1_0_id28;
    bd_memtile_1_0_id28.address = 0 + ((0x80000 + 0x3a80) / sizeof(uint32_t));
    bd_memtile_1_0_id28.length = 2128;
    bd_memtile_1_0_id28.stepsize = {1};
    bd_memtile_1_0_id28.wrap = {};
    bd_memtile_1_0_id28.padding = {};
    bd_memtile_1_0_id28.lock_acq_enable = true;
    bd_memtile_1_0_id28.lock_acq_value = -1;
    bd_memtile_1_0_id28.lock_acq_id = 64 + 12;
    bd_memtile_1_0_id28.lock_rel_value = +1;
    bd_memtile_1_0_id28.lock_rel_id = 64 + 11;
    bd_memtile_1_0_id28.use_next_bd = true;
    bd_memtile_1_0_id28.next_bd = 31;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 28, bd_memtile_1_0_id28);

    adf::dma_buffer_descriptor bd_memtile_1_0_id13;
    bd_memtile_1_0_id13.address = 592 + ((0x80000 + 0x3a80) / sizeof(uint32_t));
    bd_memtile_1_0_id13.length = 2128;
    bd_memtile_1_0_id13.stepsize = {1};
    bd_memtile_1_0_id13.wrap = {};
    bd_memtile_1_0_id13.padding = {};
    bd_memtile_1_0_id13.lock_acq_enable = true;
    bd_memtile_1_0_id13.lock_acq_value = -1;
    bd_memtile_1_0_id13.lock_acq_id = 64 + 13;
    bd_memtile_1_0_id13.lock_rel_value = +1;
    bd_memtile_1_0_id13.lock_rel_id = 64 + 11;
    bd_memtile_1_0_id13.use_next_bd = true;
    bd_memtile_1_0_id13.next_bd = 14;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 13, bd_memtile_1_0_id13);

    adf::dma_buffer_descriptor bd_memtile_1_0_id29;
    bd_memtile_1_0_id29.address = 0 + ((0x80000 + 0x5f80) / sizeof(uint32_t));
    bd_memtile_1_0_id29.length = 1184;
    bd_memtile_1_0_id29.stepsize = {1};
    bd_memtile_1_0_id29.wrap = {};
    bd_memtile_1_0_id29.padding = {};
    bd_memtile_1_0_id29.lock_acq_enable = true;
    bd_memtile_1_0_id29.lock_acq_value = -2;
    bd_memtile_1_0_id29.lock_acq_id = 64 + 14;
    bd_memtile_1_0_id29.lock_rel_value = +1;
    bd_memtile_1_0_id29.lock_rel_id = 64 + 15;
    bd_memtile_1_0_id29.use_next_bd = true;
    bd_memtile_1_0_id29.next_bd = 30;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 29, bd_memtile_1_0_id29);

    adf::dma_buffer_descriptor bd_memtile_1_0_id30;
    bd_memtile_1_0_id30.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_1_0_id30.length = 0;
    bd_memtile_1_0_id30.stepsize = {1};
    bd_memtile_1_0_id30.wrap = {};
    bd_memtile_1_0_id30.padding = {};
    bd_memtile_1_0_id30.lock_acq_enable = true;
    bd_memtile_1_0_id30.lock_acq_value = +0;
    bd_memtile_1_0_id30.lock_acq_id = 64 + 0;
    bd_memtile_1_0_id30.lock_rel_value = +1;
    bd_memtile_1_0_id30.lock_rel_id = 64 + 16;
    bd_memtile_1_0_id30.use_next_bd = false;
    bd_memtile_1_0_id30.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 30, bd_memtile_1_0_id30);

    adf::dma_buffer_descriptor bd_memtile_1_0_id31;
    bd_memtile_1_0_id31.address = 0 + ((0x80000 + 0x5f80) / sizeof(uint32_t));
    bd_memtile_1_0_id31.length = 2128;
    bd_memtile_1_0_id31.stepsize = {1};
    bd_memtile_1_0_id31.wrap = {};
    bd_memtile_1_0_id31.padding = {};
    bd_memtile_1_0_id31.lock_acq_enable = true;
    bd_memtile_1_0_id31.lock_acq_value = -1;
    bd_memtile_1_0_id31.lock_acq_id = 64 + 15;
    bd_memtile_1_0_id31.lock_rel_value = +1;
    bd_memtile_1_0_id31.lock_rel_id = 64 + 14;
    bd_memtile_1_0_id31.use_next_bd = false;
    bd_memtile_1_0_id31.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 31, bd_memtile_1_0_id31);

    adf::dma_buffer_descriptor bd_memtile_1_0_id14;
    bd_memtile_1_0_id14.address = 592 + ((0x80000 + 0x5f80) / sizeof(uint32_t));
    bd_memtile_1_0_id14.length = 2128;
    bd_memtile_1_0_id14.stepsize = {1};
    bd_memtile_1_0_id14.wrap = {};
    bd_memtile_1_0_id14.padding = {};
    bd_memtile_1_0_id14.lock_acq_enable = true;
    bd_memtile_1_0_id14.lock_acq_value = -1;
    bd_memtile_1_0_id14.lock_acq_id = 64 + 16;
    bd_memtile_1_0_id14.lock_rel_value = +1;
    bd_memtile_1_0_id14.lock_rel_id = 64 + 14;
    bd_memtile_1_0_id14.use_next_bd = false;
    bd_memtile_1_0_id14.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 14, bd_memtile_1_0_id14);

    adf::initializeLock(adf::memory_tile, 1, 0, 11, +2);

    adf::initializeLock(adf::memory_tile, 1, 0, 12, +0);

    adf::initializeLock(adf::memory_tile, 1, 0, 13, +0);

    adf::initializeLock(adf::memory_tile, 1, 0, 14, +2);

    adf::initializeLock(adf::memory_tile, 1, 0, 15, +0);

    adf::initializeLock(adf::memory_tile, 1, 0, 16, +0);

    //
    // 1 to 2 Data Transfer
    //
    // Location: memtile_2_0
    //
    // Writers
    // ----------------
    // s2mm_1 BDs: 26 -> 27 -> 29 -> 30
    //
    // Readers
    // ----------------
    // mm2s_3 BDs: 28 -> 31
    // mm2s_4 BDs: 13 -> 14
    //
    // Locks
    // ----------------
    // Id: 11, Init: +2
    // Id: 12, Init: +0
    // Id: 13, Init: +0
    // Id: 14, Init: +2
    // Id: 15, Init: +0
    // Id: 16, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_2_0_id26;
    bd_memtile_2_0_id26.address = 0 + ((0x80000 + 0x3a80) / sizeof(uint32_t));
    bd_memtile_2_0_id26.length = 1184;
    bd_memtile_2_0_id26.stepsize = {1};
    bd_memtile_2_0_id26.wrap = {};
    bd_memtile_2_0_id26.padding = {};
    bd_memtile_2_0_id26.lock_acq_enable = true;
    bd_memtile_2_0_id26.lock_acq_value = -2;
    bd_memtile_2_0_id26.lock_acq_id = 64 + 11;
    bd_memtile_2_0_id26.lock_rel_value = +1;
    bd_memtile_2_0_id26.lock_rel_id = 64 + 12;
    bd_memtile_2_0_id26.use_next_bd = true;
    bd_memtile_2_0_id26.next_bd = 27;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 26, bd_memtile_2_0_id26);

    adf::dma_buffer_descriptor bd_memtile_2_0_id27;
    bd_memtile_2_0_id27.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_2_0_id27.length = 0;
    bd_memtile_2_0_id27.stepsize = {1};
    bd_memtile_2_0_id27.wrap = {};
    bd_memtile_2_0_id27.padding = {};
    bd_memtile_2_0_id27.lock_acq_enable = true;
    bd_memtile_2_0_id27.lock_acq_value = +0;
    bd_memtile_2_0_id27.lock_acq_id = 64 + 0;
    bd_memtile_2_0_id27.lock_rel_value = +1;
    bd_memtile_2_0_id27.lock_rel_id = 64 + 13;
    bd_memtile_2_0_id27.use_next_bd = true;
    bd_memtile_2_0_id27.next_bd = 29;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 27, bd_memtile_2_0_id27);

    adf::dma_buffer_descriptor bd_memtile_2_0_id28;
    bd_memtile_2_0_id28.address = 0 + ((0x80000 + 0x3a80) / sizeof(uint32_t));
    bd_memtile_2_0_id28.length = 2128;
    bd_memtile_2_0_id28.stepsize = {1};
    bd_memtile_2_0_id28.wrap = {};
    bd_memtile_2_0_id28.padding = {};
    bd_memtile_2_0_id28.lock_acq_enable = true;
    bd_memtile_2_0_id28.lock_acq_value = -1;
    bd_memtile_2_0_id28.lock_acq_id = 64 + 12;
    bd_memtile_2_0_id28.lock_rel_value = +1;
    bd_memtile_2_0_id28.lock_rel_id = 64 + 11;
    bd_memtile_2_0_id28.use_next_bd = true;
    bd_memtile_2_0_id28.next_bd = 31;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 28, bd_memtile_2_0_id28);

    adf::dma_buffer_descriptor bd_memtile_2_0_id13;
    bd_memtile_2_0_id13.address = 592 + ((0x80000 + 0x3a80) / sizeof(uint32_t));
    bd_memtile_2_0_id13.length = 2128;
    bd_memtile_2_0_id13.stepsize = {1};
    bd_memtile_2_0_id13.wrap = {};
    bd_memtile_2_0_id13.padding = {};
    bd_memtile_2_0_id13.lock_acq_enable = true;
    bd_memtile_2_0_id13.lock_acq_value = -1;
    bd_memtile_2_0_id13.lock_acq_id = 64 + 13;
    bd_memtile_2_0_id13.lock_rel_value = +1;
    bd_memtile_2_0_id13.lock_rel_id = 64 + 11;
    bd_memtile_2_0_id13.use_next_bd = true;
    bd_memtile_2_0_id13.next_bd = 14;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 13, bd_memtile_2_0_id13);

    adf::dma_buffer_descriptor bd_memtile_2_0_id29;
    bd_memtile_2_0_id29.address = 0 + ((0x80000 + 0x5f80) / sizeof(uint32_t));
    bd_memtile_2_0_id29.length = 1184;
    bd_memtile_2_0_id29.stepsize = {1};
    bd_memtile_2_0_id29.wrap = {};
    bd_memtile_2_0_id29.padding = {};
    bd_memtile_2_0_id29.lock_acq_enable = true;
    bd_memtile_2_0_id29.lock_acq_value = -2;
    bd_memtile_2_0_id29.lock_acq_id = 64 + 14;
    bd_memtile_2_0_id29.lock_rel_value = +1;
    bd_memtile_2_0_id29.lock_rel_id = 64 + 15;
    bd_memtile_2_0_id29.use_next_bd = true;
    bd_memtile_2_0_id29.next_bd = 30;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 29, bd_memtile_2_0_id29);

    adf::dma_buffer_descriptor bd_memtile_2_0_id30;
    bd_memtile_2_0_id30.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_2_0_id30.length = 0;
    bd_memtile_2_0_id30.stepsize = {1};
    bd_memtile_2_0_id30.wrap = {};
    bd_memtile_2_0_id30.padding = {};
    bd_memtile_2_0_id30.lock_acq_enable = true;
    bd_memtile_2_0_id30.lock_acq_value = +0;
    bd_memtile_2_0_id30.lock_acq_id = 64 + 0;
    bd_memtile_2_0_id30.lock_rel_value = +1;
    bd_memtile_2_0_id30.lock_rel_id = 64 + 16;
    bd_memtile_2_0_id30.use_next_bd = false;
    bd_memtile_2_0_id30.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 30, bd_memtile_2_0_id30);

    adf::dma_buffer_descriptor bd_memtile_2_0_id31;
    bd_memtile_2_0_id31.address = 0 + ((0x80000 + 0x5f80) / sizeof(uint32_t));
    bd_memtile_2_0_id31.length = 2128;
    bd_memtile_2_0_id31.stepsize = {1};
    bd_memtile_2_0_id31.wrap = {};
    bd_memtile_2_0_id31.padding = {};
    bd_memtile_2_0_id31.lock_acq_enable = true;
    bd_memtile_2_0_id31.lock_acq_value = -1;
    bd_memtile_2_0_id31.lock_acq_id = 64 + 15;
    bd_memtile_2_0_id31.lock_rel_value = +1;
    bd_memtile_2_0_id31.lock_rel_id = 64 + 14;
    bd_memtile_2_0_id31.use_next_bd = false;
    bd_memtile_2_0_id31.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 31, bd_memtile_2_0_id31);

    adf::dma_buffer_descriptor bd_memtile_2_0_id14;
    bd_memtile_2_0_id14.address = 592 + ((0x80000 + 0x5f80) / sizeof(uint32_t));
    bd_memtile_2_0_id14.length = 2128;
    bd_memtile_2_0_id14.stepsize = {1};
    bd_memtile_2_0_id14.wrap = {};
    bd_memtile_2_0_id14.padding = {};
    bd_memtile_2_0_id14.lock_acq_enable = true;
    bd_memtile_2_0_id14.lock_acq_value = -1;
    bd_memtile_2_0_id14.lock_acq_id = 64 + 16;
    bd_memtile_2_0_id14.lock_rel_value = +1;
    bd_memtile_2_0_id14.lock_rel_id = 64 + 14;
    bd_memtile_2_0_id14.use_next_bd = false;
    bd_memtile_2_0_id14.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 14, bd_memtile_2_0_id14);

    adf::initializeLock(adf::memory_tile, 2, 0, 11, +2);

    adf::initializeLock(adf::memory_tile, 2, 0, 12, +0);

    adf::initializeLock(adf::memory_tile, 2, 0, 13, +0);

    adf::initializeLock(adf::memory_tile, 2, 0, 14, +2);

    adf::initializeLock(adf::memory_tile, 2, 0, 15, +0);

    adf::initializeLock(adf::memory_tile, 2, 0, 16, +0);

    //
    // 1 to 2 Data Transfer
    //
    // Location: memtile_3_0
    //
    // Writers
    // ----------------
    // s2mm_1 BDs: 26 -> 27 -> 29 -> 30
    //
    // Readers
    // ----------------
    // mm2s_3 BDs: 28 -> 31
    // mm2s_4 BDs: 13 -> 14
    //
    // Locks
    // ----------------
    // Id: 11, Init: +2
    // Id: 12, Init: +0
    // Id: 13, Init: +0
    // Id: 14, Init: +2
    // Id: 15, Init: +0
    // Id: 16, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_3_0_id26;
    bd_memtile_3_0_id26.address = 0 + ((0x80000 + 0x3a80) / sizeof(uint32_t));
    bd_memtile_3_0_id26.length = 1184;
    bd_memtile_3_0_id26.stepsize = {1};
    bd_memtile_3_0_id26.wrap = {};
    bd_memtile_3_0_id26.padding = {};
    bd_memtile_3_0_id26.lock_acq_enable = true;
    bd_memtile_3_0_id26.lock_acq_value = -2;
    bd_memtile_3_0_id26.lock_acq_id = 64 + 11;
    bd_memtile_3_0_id26.lock_rel_value = +1;
    bd_memtile_3_0_id26.lock_rel_id = 64 + 12;
    bd_memtile_3_0_id26.use_next_bd = true;
    bd_memtile_3_0_id26.next_bd = 27;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 26, bd_memtile_3_0_id26);

    adf::dma_buffer_descriptor bd_memtile_3_0_id27;
    bd_memtile_3_0_id27.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_3_0_id27.length = 0;
    bd_memtile_3_0_id27.stepsize = {1};
    bd_memtile_3_0_id27.wrap = {};
    bd_memtile_3_0_id27.padding = {};
    bd_memtile_3_0_id27.lock_acq_enable = true;
    bd_memtile_3_0_id27.lock_acq_value = +0;
    bd_memtile_3_0_id27.lock_acq_id = 64 + 0;
    bd_memtile_3_0_id27.lock_rel_value = +1;
    bd_memtile_3_0_id27.lock_rel_id = 64 + 13;
    bd_memtile_3_0_id27.use_next_bd = true;
    bd_memtile_3_0_id27.next_bd = 29;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 27, bd_memtile_3_0_id27);

    adf::dma_buffer_descriptor bd_memtile_3_0_id28;
    bd_memtile_3_0_id28.address = 0 + ((0x80000 + 0x3a80) / sizeof(uint32_t));
    bd_memtile_3_0_id28.length = 2128;
    bd_memtile_3_0_id28.stepsize = {1};
    bd_memtile_3_0_id28.wrap = {};
    bd_memtile_3_0_id28.padding = {};
    bd_memtile_3_0_id28.lock_acq_enable = true;
    bd_memtile_3_0_id28.lock_acq_value = -1;
    bd_memtile_3_0_id28.lock_acq_id = 64 + 12;
    bd_memtile_3_0_id28.lock_rel_value = +1;
    bd_memtile_3_0_id28.lock_rel_id = 64 + 11;
    bd_memtile_3_0_id28.use_next_bd = true;
    bd_memtile_3_0_id28.next_bd = 31;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 28, bd_memtile_3_0_id28);

    adf::dma_buffer_descriptor bd_memtile_3_0_id13;
    bd_memtile_3_0_id13.address = 592 + ((0x80000 + 0x3a80) / sizeof(uint32_t));
    bd_memtile_3_0_id13.length = 2128;
    bd_memtile_3_0_id13.stepsize = {1};
    bd_memtile_3_0_id13.wrap = {};
    bd_memtile_3_0_id13.padding = {};
    bd_memtile_3_0_id13.lock_acq_enable = true;
    bd_memtile_3_0_id13.lock_acq_value = -1;
    bd_memtile_3_0_id13.lock_acq_id = 64 + 13;
    bd_memtile_3_0_id13.lock_rel_value = +1;
    bd_memtile_3_0_id13.lock_rel_id = 64 + 11;
    bd_memtile_3_0_id13.use_next_bd = true;
    bd_memtile_3_0_id13.next_bd = 14;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 13, bd_memtile_3_0_id13);

    adf::dma_buffer_descriptor bd_memtile_3_0_id29;
    bd_memtile_3_0_id29.address = 0 + ((0x80000 + 0x5f80) / sizeof(uint32_t));
    bd_memtile_3_0_id29.length = 1184;
    bd_memtile_3_0_id29.stepsize = {1};
    bd_memtile_3_0_id29.wrap = {};
    bd_memtile_3_0_id29.padding = {};
    bd_memtile_3_0_id29.lock_acq_enable = true;
    bd_memtile_3_0_id29.lock_acq_value = -2;
    bd_memtile_3_0_id29.lock_acq_id = 64 + 14;
    bd_memtile_3_0_id29.lock_rel_value = +1;
    bd_memtile_3_0_id29.lock_rel_id = 64 + 15;
    bd_memtile_3_0_id29.use_next_bd = true;
    bd_memtile_3_0_id29.next_bd = 30;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 29, bd_memtile_3_0_id29);

    adf::dma_buffer_descriptor bd_memtile_3_0_id30;
    bd_memtile_3_0_id30.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_3_0_id30.length = 0;
    bd_memtile_3_0_id30.stepsize = {1};
    bd_memtile_3_0_id30.wrap = {};
    bd_memtile_3_0_id30.padding = {};
    bd_memtile_3_0_id30.lock_acq_enable = true;
    bd_memtile_3_0_id30.lock_acq_value = +0;
    bd_memtile_3_0_id30.lock_acq_id = 64 + 0;
    bd_memtile_3_0_id30.lock_rel_value = +1;
    bd_memtile_3_0_id30.lock_rel_id = 64 + 16;
    bd_memtile_3_0_id30.use_next_bd = false;
    bd_memtile_3_0_id30.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 30, bd_memtile_3_0_id30);

    adf::dma_buffer_descriptor bd_memtile_3_0_id31;
    bd_memtile_3_0_id31.address = 0 + ((0x80000 + 0x5f80) / sizeof(uint32_t));
    bd_memtile_3_0_id31.length = 2128;
    bd_memtile_3_0_id31.stepsize = {1};
    bd_memtile_3_0_id31.wrap = {};
    bd_memtile_3_0_id31.padding = {};
    bd_memtile_3_0_id31.lock_acq_enable = true;
    bd_memtile_3_0_id31.lock_acq_value = -1;
    bd_memtile_3_0_id31.lock_acq_id = 64 + 15;
    bd_memtile_3_0_id31.lock_rel_value = +1;
    bd_memtile_3_0_id31.lock_rel_id = 64 + 14;
    bd_memtile_3_0_id31.use_next_bd = false;
    bd_memtile_3_0_id31.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 31, bd_memtile_3_0_id31);

    adf::dma_buffer_descriptor bd_memtile_3_0_id14;
    bd_memtile_3_0_id14.address = 592 + ((0x80000 + 0x5f80) / sizeof(uint32_t));
    bd_memtile_3_0_id14.length = 2128;
    bd_memtile_3_0_id14.stepsize = {1};
    bd_memtile_3_0_id14.wrap = {};
    bd_memtile_3_0_id14.padding = {};
    bd_memtile_3_0_id14.lock_acq_enable = true;
    bd_memtile_3_0_id14.lock_acq_value = -1;
    bd_memtile_3_0_id14.lock_acq_id = 64 + 16;
    bd_memtile_3_0_id14.lock_rel_value = +1;
    bd_memtile_3_0_id14.lock_rel_id = 64 + 14;
    bd_memtile_3_0_id14.use_next_bd = false;
    bd_memtile_3_0_id14.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 14, bd_memtile_3_0_id14);

    adf::initializeLock(adf::memory_tile, 3, 0, 11, +2);

    adf::initializeLock(adf::memory_tile, 3, 0, 12, +0);

    adf::initializeLock(adf::memory_tile, 3, 0, 13, +0);

    adf::initializeLock(adf::memory_tile, 3, 0, 14, +2);

    adf::initializeLock(adf::memory_tile, 3, 0, 15, +0);

    adf::initializeLock(adf::memory_tile, 3, 0, 16, +0);

    //
    // 4 to 1 Data Transfer
    //
    // Location: memtile_0_0
    //
    // Writers
    // ----------------
    // s2mm_2 BDs: 15 -> 17
    // s2mm_3 BDs: 32 -> 38
    // s2mm_4 BDs: 16 -> 18
    // s2mm_5 BDs: 33 -> 39
    //
    // Readers
    // ----------------
    // mm2s_5 BDs: 34 -> 35 -> 36 -> 37 -> 40 -> 41 -> 42 -> 43
    //
    // Locks
    // ----------------
    // Id: 17, Init: +1
    // Id: 18, Init: +1
    // Id: 19, Init: +1
    // Id: 20, Init: +1
    // Id: 21, Init: +0
    // Id: 22, Init: +1
    // Id: 23, Init: +1
    // Id: 24, Init: +1
    // Id: 25, Init: +1
    // Id: 26, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_0_0_id15;
    bd_memtile_0_0_id15.address = 0 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_0_0_id15.length = 1024;
    bd_memtile_0_0_id15.stepsize = {1};
    bd_memtile_0_0_id15.wrap = {};
    bd_memtile_0_0_id15.padding = {};
    bd_memtile_0_0_id15.lock_acq_enable = true;
    bd_memtile_0_0_id15.lock_acq_value = -1;
    bd_memtile_0_0_id15.lock_acq_id = 64 + 17;
    bd_memtile_0_0_id15.lock_rel_value = +1;
    bd_memtile_0_0_id15.lock_rel_id = 64 + 21;
    bd_memtile_0_0_id15.use_next_bd = true;
    bd_memtile_0_0_id15.next_bd = 17;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 15, bd_memtile_0_0_id15);

    adf::dma_buffer_descriptor bd_memtile_0_0_id32;
    bd_memtile_0_0_id32.address = 1024 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_0_0_id32.length = 1024;
    bd_memtile_0_0_id32.stepsize = {1};
    bd_memtile_0_0_id32.wrap = {};
    bd_memtile_0_0_id32.padding = {};
    bd_memtile_0_0_id32.lock_acq_enable = true;
    bd_memtile_0_0_id32.lock_acq_value = -1;
    bd_memtile_0_0_id32.lock_acq_id = 64 + 18;
    bd_memtile_0_0_id32.lock_rel_value = +1;
    bd_memtile_0_0_id32.lock_rel_id = 64 + 21;
    bd_memtile_0_0_id32.use_next_bd = true;
    bd_memtile_0_0_id32.next_bd = 38;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 32, bd_memtile_0_0_id32);

    adf::dma_buffer_descriptor bd_memtile_0_0_id16;
    bd_memtile_0_0_id16.address = 2048 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_0_0_id16.length = 1024;
    bd_memtile_0_0_id16.stepsize = {1};
    bd_memtile_0_0_id16.wrap = {};
    bd_memtile_0_0_id16.padding = {};
    bd_memtile_0_0_id16.lock_acq_enable = true;
    bd_memtile_0_0_id16.lock_acq_value = -1;
    bd_memtile_0_0_id16.lock_acq_id = 64 + 19;
    bd_memtile_0_0_id16.lock_rel_value = +1;
    bd_memtile_0_0_id16.lock_rel_id = 64 + 21;
    bd_memtile_0_0_id16.use_next_bd = true;
    bd_memtile_0_0_id16.next_bd = 18;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 16, bd_memtile_0_0_id16);

    adf::dma_buffer_descriptor bd_memtile_0_0_id33;
    bd_memtile_0_0_id33.address = 3072 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_0_0_id33.length = 1024;
    bd_memtile_0_0_id33.stepsize = {1};
    bd_memtile_0_0_id33.wrap = {};
    bd_memtile_0_0_id33.padding = {};
    bd_memtile_0_0_id33.lock_acq_enable = true;
    bd_memtile_0_0_id33.lock_acq_value = -1;
    bd_memtile_0_0_id33.lock_acq_id = 64 + 20;
    bd_memtile_0_0_id33.lock_rel_value = +1;
    bd_memtile_0_0_id33.lock_rel_id = 64 + 21;
    bd_memtile_0_0_id33.use_next_bd = true;
    bd_memtile_0_0_id33.next_bd = 39;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 33, bd_memtile_0_0_id33);

    adf::dma_buffer_descriptor bd_memtile_0_0_id34;
    bd_memtile_0_0_id34.address = 0 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_0_0_id34.length = 512;
    bd_memtile_0_0_id34.stepsize = {1, 128, 1024};
    bd_memtile_0_0_id34.wrap = {128, 1};
    bd_memtile_0_0_id34.padding = {};
    bd_memtile_0_0_id34.lock_acq_enable = true;
    bd_memtile_0_0_id34.lock_acq_value = -4;
    bd_memtile_0_0_id34.lock_acq_id = 64 + 21;
    bd_memtile_0_0_id34.lock_rel_value = +1;
    bd_memtile_0_0_id34.lock_rel_id = 64 + 17;
    bd_memtile_0_0_id34.use_next_bd = true;
    bd_memtile_0_0_id34.next_bd = 35;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 34, bd_memtile_0_0_id34);

    adf::dma_buffer_descriptor bd_memtile_0_0_id35;
    bd_memtile_0_0_id35.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_0_0_id35.length = 0;
    bd_memtile_0_0_id35.stepsize = {1};
    bd_memtile_0_0_id35.wrap = {};
    bd_memtile_0_0_id35.padding = {};
    bd_memtile_0_0_id35.lock_acq_enable = true;
    bd_memtile_0_0_id35.lock_acq_value = +0;
    bd_memtile_0_0_id35.lock_acq_id = 64 + 0;
    bd_memtile_0_0_id35.lock_rel_value = +1;
    bd_memtile_0_0_id35.lock_rel_id = 64 + 18;
    bd_memtile_0_0_id35.use_next_bd = true;
    bd_memtile_0_0_id35.next_bd = 36;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 35, bd_memtile_0_0_id35);

    adf::dma_buffer_descriptor bd_memtile_0_0_id36;
    bd_memtile_0_0_id36.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_0_0_id36.length = 0;
    bd_memtile_0_0_id36.stepsize = {1};
    bd_memtile_0_0_id36.wrap = {};
    bd_memtile_0_0_id36.padding = {};
    bd_memtile_0_0_id36.lock_acq_enable = true;
    bd_memtile_0_0_id36.lock_acq_value = +0;
    bd_memtile_0_0_id36.lock_acq_id = 64 + 0;
    bd_memtile_0_0_id36.lock_rel_value = +1;
    bd_memtile_0_0_id36.lock_rel_id = 64 + 19;
    bd_memtile_0_0_id36.use_next_bd = true;
    bd_memtile_0_0_id36.next_bd = 37;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 36, bd_memtile_0_0_id36);

    adf::dma_buffer_descriptor bd_memtile_0_0_id37;
    bd_memtile_0_0_id37.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_0_0_id37.length = 0;
    bd_memtile_0_0_id37.stepsize = {1};
    bd_memtile_0_0_id37.wrap = {};
    bd_memtile_0_0_id37.padding = {};
    bd_memtile_0_0_id37.lock_acq_enable = true;
    bd_memtile_0_0_id37.lock_acq_value = +0;
    bd_memtile_0_0_id37.lock_acq_id = 64 + 0;
    bd_memtile_0_0_id37.lock_rel_value = +1;
    bd_memtile_0_0_id37.lock_rel_id = 64 + 20;
    bd_memtile_0_0_id37.use_next_bd = true;
    bd_memtile_0_0_id37.next_bd = 40;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 37, bd_memtile_0_0_id37);

    adf::dma_buffer_descriptor bd_memtile_0_0_id17;
    bd_memtile_0_0_id17.address = 0 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_0_0_id17.length = 1024;
    bd_memtile_0_0_id17.stepsize = {1};
    bd_memtile_0_0_id17.wrap = {};
    bd_memtile_0_0_id17.padding = {};
    bd_memtile_0_0_id17.lock_acq_enable = true;
    bd_memtile_0_0_id17.lock_acq_value = -1;
    bd_memtile_0_0_id17.lock_acq_id = 64 + 22;
    bd_memtile_0_0_id17.lock_rel_value = +1;
    bd_memtile_0_0_id17.lock_rel_id = 64 + 26;
    bd_memtile_0_0_id17.use_next_bd = false;
    bd_memtile_0_0_id17.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 17, bd_memtile_0_0_id17);

    adf::dma_buffer_descriptor bd_memtile_0_0_id38;
    bd_memtile_0_0_id38.address = 1024 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_0_0_id38.length = 1024;
    bd_memtile_0_0_id38.stepsize = {1};
    bd_memtile_0_0_id38.wrap = {};
    bd_memtile_0_0_id38.padding = {};
    bd_memtile_0_0_id38.lock_acq_enable = true;
    bd_memtile_0_0_id38.lock_acq_value = -1;
    bd_memtile_0_0_id38.lock_acq_id = 64 + 23;
    bd_memtile_0_0_id38.lock_rel_value = +1;
    bd_memtile_0_0_id38.lock_rel_id = 64 + 26;
    bd_memtile_0_0_id38.use_next_bd = false;
    bd_memtile_0_0_id38.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 38, bd_memtile_0_0_id38);

    adf::dma_buffer_descriptor bd_memtile_0_0_id18;
    bd_memtile_0_0_id18.address = 2048 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_0_0_id18.length = 1024;
    bd_memtile_0_0_id18.stepsize = {1};
    bd_memtile_0_0_id18.wrap = {};
    bd_memtile_0_0_id18.padding = {};
    bd_memtile_0_0_id18.lock_acq_enable = true;
    bd_memtile_0_0_id18.lock_acq_value = -1;
    bd_memtile_0_0_id18.lock_acq_id = 64 + 24;
    bd_memtile_0_0_id18.lock_rel_value = +1;
    bd_memtile_0_0_id18.lock_rel_id = 64 + 26;
    bd_memtile_0_0_id18.use_next_bd = false;
    bd_memtile_0_0_id18.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 18, bd_memtile_0_0_id18);

    adf::dma_buffer_descriptor bd_memtile_0_0_id39;
    bd_memtile_0_0_id39.address = 3072 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_0_0_id39.length = 1024;
    bd_memtile_0_0_id39.stepsize = {1};
    bd_memtile_0_0_id39.wrap = {};
    bd_memtile_0_0_id39.padding = {};
    bd_memtile_0_0_id39.lock_acq_enable = true;
    bd_memtile_0_0_id39.lock_acq_value = -1;
    bd_memtile_0_0_id39.lock_acq_id = 64 + 25;
    bd_memtile_0_0_id39.lock_rel_value = +1;
    bd_memtile_0_0_id39.lock_rel_id = 64 + 26;
    bd_memtile_0_0_id39.use_next_bd = false;
    bd_memtile_0_0_id39.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 39, bd_memtile_0_0_id39);

    adf::dma_buffer_descriptor bd_memtile_0_0_id40;
    bd_memtile_0_0_id40.address = 0 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_0_0_id40.length = 512;
    bd_memtile_0_0_id40.stepsize = {1, 128, 1024};
    bd_memtile_0_0_id40.wrap = {128, 1};
    bd_memtile_0_0_id40.padding = {};
    bd_memtile_0_0_id40.lock_acq_enable = true;
    bd_memtile_0_0_id40.lock_acq_value = -4;
    bd_memtile_0_0_id40.lock_acq_id = 64 + 26;
    bd_memtile_0_0_id40.lock_rel_value = +1;
    bd_memtile_0_0_id40.lock_rel_id = 64 + 22;
    bd_memtile_0_0_id40.use_next_bd = true;
    bd_memtile_0_0_id40.next_bd = 41;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 40, bd_memtile_0_0_id40);

    adf::dma_buffer_descriptor bd_memtile_0_0_id41;
    bd_memtile_0_0_id41.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_0_0_id41.length = 0;
    bd_memtile_0_0_id41.stepsize = {1};
    bd_memtile_0_0_id41.wrap = {};
    bd_memtile_0_0_id41.padding = {};
    bd_memtile_0_0_id41.lock_acq_enable = true;
    bd_memtile_0_0_id41.lock_acq_value = +0;
    bd_memtile_0_0_id41.lock_acq_id = 64 + 0;
    bd_memtile_0_0_id41.lock_rel_value = +1;
    bd_memtile_0_0_id41.lock_rel_id = 64 + 23;
    bd_memtile_0_0_id41.use_next_bd = true;
    bd_memtile_0_0_id41.next_bd = 42;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 41, bd_memtile_0_0_id41);

    adf::dma_buffer_descriptor bd_memtile_0_0_id42;
    bd_memtile_0_0_id42.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_0_0_id42.length = 0;
    bd_memtile_0_0_id42.stepsize = {1};
    bd_memtile_0_0_id42.wrap = {};
    bd_memtile_0_0_id42.padding = {};
    bd_memtile_0_0_id42.lock_acq_enable = true;
    bd_memtile_0_0_id42.lock_acq_value = +0;
    bd_memtile_0_0_id42.lock_acq_id = 64 + 0;
    bd_memtile_0_0_id42.lock_rel_value = +1;
    bd_memtile_0_0_id42.lock_rel_id = 64 + 24;
    bd_memtile_0_0_id42.use_next_bd = true;
    bd_memtile_0_0_id42.next_bd = 43;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 42, bd_memtile_0_0_id42);

    adf::dma_buffer_descriptor bd_memtile_0_0_id43;
    bd_memtile_0_0_id43.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_0_0_id43.length = 0;
    bd_memtile_0_0_id43.stepsize = {1};
    bd_memtile_0_0_id43.wrap = {};
    bd_memtile_0_0_id43.padding = {};
    bd_memtile_0_0_id43.lock_acq_enable = true;
    bd_memtile_0_0_id43.lock_acq_value = +0;
    bd_memtile_0_0_id43.lock_acq_id = 64 + 0;
    bd_memtile_0_0_id43.lock_rel_value = +1;
    bd_memtile_0_0_id43.lock_rel_id = 64 + 25;
    bd_memtile_0_0_id43.use_next_bd = false;
    bd_memtile_0_0_id43.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 0, 0, 43, bd_memtile_0_0_id43);

    adf::initializeLock(adf::memory_tile, 0, 0, 17, +1);

    adf::initializeLock(adf::memory_tile, 0, 0, 18, +1);

    adf::initializeLock(adf::memory_tile, 0, 0, 19, +1);

    adf::initializeLock(adf::memory_tile, 0, 0, 20, +1);

    adf::initializeLock(adf::memory_tile, 0, 0, 21, +0);

    adf::initializeLock(adf::memory_tile, 0, 0, 22, +1);

    adf::initializeLock(adf::memory_tile, 0, 0, 23, +1);

    adf::initializeLock(adf::memory_tile, 0, 0, 24, +1);

    adf::initializeLock(adf::memory_tile, 0, 0, 25, +1);

    adf::initializeLock(adf::memory_tile, 0, 0, 26, +0);

    //
    // 4 to 1 Data Transfer
    //
    // Location: memtile_1_0
    //
    // Writers
    // ----------------
    // s2mm_2 BDs: 15 -> 17
    // s2mm_3 BDs: 32 -> 38
    // s2mm_4 BDs: 16 -> 18
    // s2mm_5 BDs: 33 -> 39
    //
    // Readers
    // ----------------
    // mm2s_5 BDs: 34 -> 35 -> 36 -> 37 -> 40 -> 41 -> 42 -> 43
    //
    // Locks
    // ----------------
    // Id: 17, Init: +1
    // Id: 18, Init: +1
    // Id: 19, Init: +1
    // Id: 20, Init: +1
    // Id: 21, Init: +0
    // Id: 22, Init: +1
    // Id: 23, Init: +1
    // Id: 24, Init: +1
    // Id: 25, Init: +1
    // Id: 26, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_1_0_id15;
    bd_memtile_1_0_id15.address = 0 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_1_0_id15.length = 1024;
    bd_memtile_1_0_id15.stepsize = {1};
    bd_memtile_1_0_id15.wrap = {};
    bd_memtile_1_0_id15.padding = {};
    bd_memtile_1_0_id15.lock_acq_enable = true;
    bd_memtile_1_0_id15.lock_acq_value = -1;
    bd_memtile_1_0_id15.lock_acq_id = 64 + 17;
    bd_memtile_1_0_id15.lock_rel_value = +1;
    bd_memtile_1_0_id15.lock_rel_id = 64 + 21;
    bd_memtile_1_0_id15.use_next_bd = true;
    bd_memtile_1_0_id15.next_bd = 17;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 15, bd_memtile_1_0_id15);

    adf::dma_buffer_descriptor bd_memtile_1_0_id32;
    bd_memtile_1_0_id32.address = 1024 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_1_0_id32.length = 1024;
    bd_memtile_1_0_id32.stepsize = {1};
    bd_memtile_1_0_id32.wrap = {};
    bd_memtile_1_0_id32.padding = {};
    bd_memtile_1_0_id32.lock_acq_enable = true;
    bd_memtile_1_0_id32.lock_acq_value = -1;
    bd_memtile_1_0_id32.lock_acq_id = 64 + 18;
    bd_memtile_1_0_id32.lock_rel_value = +1;
    bd_memtile_1_0_id32.lock_rel_id = 64 + 21;
    bd_memtile_1_0_id32.use_next_bd = true;
    bd_memtile_1_0_id32.next_bd = 38;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 32, bd_memtile_1_0_id32);

    adf::dma_buffer_descriptor bd_memtile_1_0_id16;
    bd_memtile_1_0_id16.address = 2048 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_1_0_id16.length = 1024;
    bd_memtile_1_0_id16.stepsize = {1};
    bd_memtile_1_0_id16.wrap = {};
    bd_memtile_1_0_id16.padding = {};
    bd_memtile_1_0_id16.lock_acq_enable = true;
    bd_memtile_1_0_id16.lock_acq_value = -1;
    bd_memtile_1_0_id16.lock_acq_id = 64 + 19;
    bd_memtile_1_0_id16.lock_rel_value = +1;
    bd_memtile_1_0_id16.lock_rel_id = 64 + 21;
    bd_memtile_1_0_id16.use_next_bd = true;
    bd_memtile_1_0_id16.next_bd = 18;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 16, bd_memtile_1_0_id16);

    adf::dma_buffer_descriptor bd_memtile_1_0_id33;
    bd_memtile_1_0_id33.address = 3072 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_1_0_id33.length = 1024;
    bd_memtile_1_0_id33.stepsize = {1};
    bd_memtile_1_0_id33.wrap = {};
    bd_memtile_1_0_id33.padding = {};
    bd_memtile_1_0_id33.lock_acq_enable = true;
    bd_memtile_1_0_id33.lock_acq_value = -1;
    bd_memtile_1_0_id33.lock_acq_id = 64 + 20;
    bd_memtile_1_0_id33.lock_rel_value = +1;
    bd_memtile_1_0_id33.lock_rel_id = 64 + 21;
    bd_memtile_1_0_id33.use_next_bd = true;
    bd_memtile_1_0_id33.next_bd = 39;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 33, bd_memtile_1_0_id33);

    adf::dma_buffer_descriptor bd_memtile_1_0_id34;
    bd_memtile_1_0_id34.address = 0 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_1_0_id34.length = 512;
    bd_memtile_1_0_id34.stepsize = {1, 128, 1024};
    bd_memtile_1_0_id34.wrap = {128, 1};
    bd_memtile_1_0_id34.padding = {};
    bd_memtile_1_0_id34.lock_acq_enable = true;
    bd_memtile_1_0_id34.lock_acq_value = -4;
    bd_memtile_1_0_id34.lock_acq_id = 64 + 21;
    bd_memtile_1_0_id34.lock_rel_value = +1;
    bd_memtile_1_0_id34.lock_rel_id = 64 + 17;
    bd_memtile_1_0_id34.use_next_bd = true;
    bd_memtile_1_0_id34.next_bd = 35;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 34, bd_memtile_1_0_id34);

    adf::dma_buffer_descriptor bd_memtile_1_0_id35;
    bd_memtile_1_0_id35.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_1_0_id35.length = 0;
    bd_memtile_1_0_id35.stepsize = {1};
    bd_memtile_1_0_id35.wrap = {};
    bd_memtile_1_0_id35.padding = {};
    bd_memtile_1_0_id35.lock_acq_enable = true;
    bd_memtile_1_0_id35.lock_acq_value = +0;
    bd_memtile_1_0_id35.lock_acq_id = 64 + 0;
    bd_memtile_1_0_id35.lock_rel_value = +1;
    bd_memtile_1_0_id35.lock_rel_id = 64 + 18;
    bd_memtile_1_0_id35.use_next_bd = true;
    bd_memtile_1_0_id35.next_bd = 36;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 35, bd_memtile_1_0_id35);

    adf::dma_buffer_descriptor bd_memtile_1_0_id36;
    bd_memtile_1_0_id36.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_1_0_id36.length = 0;
    bd_memtile_1_0_id36.stepsize = {1};
    bd_memtile_1_0_id36.wrap = {};
    bd_memtile_1_0_id36.padding = {};
    bd_memtile_1_0_id36.lock_acq_enable = true;
    bd_memtile_1_0_id36.lock_acq_value = +0;
    bd_memtile_1_0_id36.lock_acq_id = 64 + 0;
    bd_memtile_1_0_id36.lock_rel_value = +1;
    bd_memtile_1_0_id36.lock_rel_id = 64 + 19;
    bd_memtile_1_0_id36.use_next_bd = true;
    bd_memtile_1_0_id36.next_bd = 37;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 36, bd_memtile_1_0_id36);

    adf::dma_buffer_descriptor bd_memtile_1_0_id37;
    bd_memtile_1_0_id37.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_1_0_id37.length = 0;
    bd_memtile_1_0_id37.stepsize = {1};
    bd_memtile_1_0_id37.wrap = {};
    bd_memtile_1_0_id37.padding = {};
    bd_memtile_1_0_id37.lock_acq_enable = true;
    bd_memtile_1_0_id37.lock_acq_value = +0;
    bd_memtile_1_0_id37.lock_acq_id = 64 + 0;
    bd_memtile_1_0_id37.lock_rel_value = +1;
    bd_memtile_1_0_id37.lock_rel_id = 64 + 20;
    bd_memtile_1_0_id37.use_next_bd = true;
    bd_memtile_1_0_id37.next_bd = 40;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 37, bd_memtile_1_0_id37);

    adf::dma_buffer_descriptor bd_memtile_1_0_id17;
    bd_memtile_1_0_id17.address = 0 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_1_0_id17.length = 1024;
    bd_memtile_1_0_id17.stepsize = {1};
    bd_memtile_1_0_id17.wrap = {};
    bd_memtile_1_0_id17.padding = {};
    bd_memtile_1_0_id17.lock_acq_enable = true;
    bd_memtile_1_0_id17.lock_acq_value = -1;
    bd_memtile_1_0_id17.lock_acq_id = 64 + 22;
    bd_memtile_1_0_id17.lock_rel_value = +1;
    bd_memtile_1_0_id17.lock_rel_id = 64 + 26;
    bd_memtile_1_0_id17.use_next_bd = false;
    bd_memtile_1_0_id17.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 17, bd_memtile_1_0_id17);

    adf::dma_buffer_descriptor bd_memtile_1_0_id38;
    bd_memtile_1_0_id38.address = 1024 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_1_0_id38.length = 1024;
    bd_memtile_1_0_id38.stepsize = {1};
    bd_memtile_1_0_id38.wrap = {};
    bd_memtile_1_0_id38.padding = {};
    bd_memtile_1_0_id38.lock_acq_enable = true;
    bd_memtile_1_0_id38.lock_acq_value = -1;
    bd_memtile_1_0_id38.lock_acq_id = 64 + 23;
    bd_memtile_1_0_id38.lock_rel_value = +1;
    bd_memtile_1_0_id38.lock_rel_id = 64 + 26;
    bd_memtile_1_0_id38.use_next_bd = false;
    bd_memtile_1_0_id38.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 38, bd_memtile_1_0_id38);

    adf::dma_buffer_descriptor bd_memtile_1_0_id18;
    bd_memtile_1_0_id18.address = 2048 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_1_0_id18.length = 1024;
    bd_memtile_1_0_id18.stepsize = {1};
    bd_memtile_1_0_id18.wrap = {};
    bd_memtile_1_0_id18.padding = {};
    bd_memtile_1_0_id18.lock_acq_enable = true;
    bd_memtile_1_0_id18.lock_acq_value = -1;
    bd_memtile_1_0_id18.lock_acq_id = 64 + 24;
    bd_memtile_1_0_id18.lock_rel_value = +1;
    bd_memtile_1_0_id18.lock_rel_id = 64 + 26;
    bd_memtile_1_0_id18.use_next_bd = false;
    bd_memtile_1_0_id18.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 18, bd_memtile_1_0_id18);

    adf::dma_buffer_descriptor bd_memtile_1_0_id39;
    bd_memtile_1_0_id39.address = 3072 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_1_0_id39.length = 1024;
    bd_memtile_1_0_id39.stepsize = {1};
    bd_memtile_1_0_id39.wrap = {};
    bd_memtile_1_0_id39.padding = {};
    bd_memtile_1_0_id39.lock_acq_enable = true;
    bd_memtile_1_0_id39.lock_acq_value = -1;
    bd_memtile_1_0_id39.lock_acq_id = 64 + 25;
    bd_memtile_1_0_id39.lock_rel_value = +1;
    bd_memtile_1_0_id39.lock_rel_id = 64 + 26;
    bd_memtile_1_0_id39.use_next_bd = false;
    bd_memtile_1_0_id39.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 39, bd_memtile_1_0_id39);

    adf::dma_buffer_descriptor bd_memtile_1_0_id40;
    bd_memtile_1_0_id40.address = 0 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_1_0_id40.length = 512;
    bd_memtile_1_0_id40.stepsize = {1, 128, 1024};
    bd_memtile_1_0_id40.wrap = {128, 1};
    bd_memtile_1_0_id40.padding = {};
    bd_memtile_1_0_id40.lock_acq_enable = true;
    bd_memtile_1_0_id40.lock_acq_value = -4;
    bd_memtile_1_0_id40.lock_acq_id = 64 + 26;
    bd_memtile_1_0_id40.lock_rel_value = +1;
    bd_memtile_1_0_id40.lock_rel_id = 64 + 22;
    bd_memtile_1_0_id40.use_next_bd = true;
    bd_memtile_1_0_id40.next_bd = 41;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 40, bd_memtile_1_0_id40);

    adf::dma_buffer_descriptor bd_memtile_1_0_id41;
    bd_memtile_1_0_id41.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_1_0_id41.length = 0;
    bd_memtile_1_0_id41.stepsize = {1};
    bd_memtile_1_0_id41.wrap = {};
    bd_memtile_1_0_id41.padding = {};
    bd_memtile_1_0_id41.lock_acq_enable = true;
    bd_memtile_1_0_id41.lock_acq_value = +0;
    bd_memtile_1_0_id41.lock_acq_id = 64 + 0;
    bd_memtile_1_0_id41.lock_rel_value = +1;
    bd_memtile_1_0_id41.lock_rel_id = 64 + 23;
    bd_memtile_1_0_id41.use_next_bd = true;
    bd_memtile_1_0_id41.next_bd = 42;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 41, bd_memtile_1_0_id41);

    adf::dma_buffer_descriptor bd_memtile_1_0_id42;
    bd_memtile_1_0_id42.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_1_0_id42.length = 0;
    bd_memtile_1_0_id42.stepsize = {1};
    bd_memtile_1_0_id42.wrap = {};
    bd_memtile_1_0_id42.padding = {};
    bd_memtile_1_0_id42.lock_acq_enable = true;
    bd_memtile_1_0_id42.lock_acq_value = +0;
    bd_memtile_1_0_id42.lock_acq_id = 64 + 0;
    bd_memtile_1_0_id42.lock_rel_value = +1;
    bd_memtile_1_0_id42.lock_rel_id = 64 + 24;
    bd_memtile_1_0_id42.use_next_bd = true;
    bd_memtile_1_0_id42.next_bd = 43;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 42, bd_memtile_1_0_id42);

    adf::dma_buffer_descriptor bd_memtile_1_0_id43;
    bd_memtile_1_0_id43.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_1_0_id43.length = 0;
    bd_memtile_1_0_id43.stepsize = {1};
    bd_memtile_1_0_id43.wrap = {};
    bd_memtile_1_0_id43.padding = {};
    bd_memtile_1_0_id43.lock_acq_enable = true;
    bd_memtile_1_0_id43.lock_acq_value = +0;
    bd_memtile_1_0_id43.lock_acq_id = 64 + 0;
    bd_memtile_1_0_id43.lock_rel_value = +1;
    bd_memtile_1_0_id43.lock_rel_id = 64 + 25;
    bd_memtile_1_0_id43.use_next_bd = false;
    bd_memtile_1_0_id43.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 1, 0, 43, bd_memtile_1_0_id43);

    adf::initializeLock(adf::memory_tile, 1, 0, 17, +1);

    adf::initializeLock(adf::memory_tile, 1, 0, 18, +1);

    adf::initializeLock(adf::memory_tile, 1, 0, 19, +1);

    adf::initializeLock(adf::memory_tile, 1, 0, 20, +1);

    adf::initializeLock(adf::memory_tile, 1, 0, 21, +0);

    adf::initializeLock(adf::memory_tile, 1, 0, 22, +1);

    adf::initializeLock(adf::memory_tile, 1, 0, 23, +1);

    adf::initializeLock(adf::memory_tile, 1, 0, 24, +1);

    adf::initializeLock(adf::memory_tile, 1, 0, 25, +1);

    adf::initializeLock(adf::memory_tile, 1, 0, 26, +0);

    //
    // 4 to 1 Data Transfer
    //
    // Location: memtile_2_0
    //
    // Writers
    // ----------------
    // s2mm_2 BDs: 15 -> 17
    // s2mm_3 BDs: 32 -> 38
    // s2mm_4 BDs: 16 -> 18
    // s2mm_5 BDs: 33 -> 39
    //
    // Readers
    // ----------------
    // mm2s_5 BDs: 34 -> 35 -> 36 -> 37 -> 40 -> 41 -> 42 -> 43
    //
    // Locks
    // ----------------
    // Id: 17, Init: +1
    // Id: 18, Init: +1
    // Id: 19, Init: +1
    // Id: 20, Init: +1
    // Id: 21, Init: +0
    // Id: 22, Init: +1
    // Id: 23, Init: +1
    // Id: 24, Init: +1
    // Id: 25, Init: +1
    // Id: 26, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_2_0_id15;
    bd_memtile_2_0_id15.address = 0 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_2_0_id15.length = 1024;
    bd_memtile_2_0_id15.stepsize = {1};
    bd_memtile_2_0_id15.wrap = {};
    bd_memtile_2_0_id15.padding = {};
    bd_memtile_2_0_id15.lock_acq_enable = true;
    bd_memtile_2_0_id15.lock_acq_value = -1;
    bd_memtile_2_0_id15.lock_acq_id = 64 + 17;
    bd_memtile_2_0_id15.lock_rel_value = +1;
    bd_memtile_2_0_id15.lock_rel_id = 64 + 21;
    bd_memtile_2_0_id15.use_next_bd = true;
    bd_memtile_2_0_id15.next_bd = 17;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 15, bd_memtile_2_0_id15);

    adf::dma_buffer_descriptor bd_memtile_2_0_id32;
    bd_memtile_2_0_id32.address = 1024 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_2_0_id32.length = 1024;
    bd_memtile_2_0_id32.stepsize = {1};
    bd_memtile_2_0_id32.wrap = {};
    bd_memtile_2_0_id32.padding = {};
    bd_memtile_2_0_id32.lock_acq_enable = true;
    bd_memtile_2_0_id32.lock_acq_value = -1;
    bd_memtile_2_0_id32.lock_acq_id = 64 + 18;
    bd_memtile_2_0_id32.lock_rel_value = +1;
    bd_memtile_2_0_id32.lock_rel_id = 64 + 21;
    bd_memtile_2_0_id32.use_next_bd = true;
    bd_memtile_2_0_id32.next_bd = 38;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 32, bd_memtile_2_0_id32);

    adf::dma_buffer_descriptor bd_memtile_2_0_id16;
    bd_memtile_2_0_id16.address = 2048 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_2_0_id16.length = 1024;
    bd_memtile_2_0_id16.stepsize = {1};
    bd_memtile_2_0_id16.wrap = {};
    bd_memtile_2_0_id16.padding = {};
    bd_memtile_2_0_id16.lock_acq_enable = true;
    bd_memtile_2_0_id16.lock_acq_value = -1;
    bd_memtile_2_0_id16.lock_acq_id = 64 + 19;
    bd_memtile_2_0_id16.lock_rel_value = +1;
    bd_memtile_2_0_id16.lock_rel_id = 64 + 21;
    bd_memtile_2_0_id16.use_next_bd = true;
    bd_memtile_2_0_id16.next_bd = 18;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 16, bd_memtile_2_0_id16);

    adf::dma_buffer_descriptor bd_memtile_2_0_id33;
    bd_memtile_2_0_id33.address = 3072 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_2_0_id33.length = 1024;
    bd_memtile_2_0_id33.stepsize = {1};
    bd_memtile_2_0_id33.wrap = {};
    bd_memtile_2_0_id33.padding = {};
    bd_memtile_2_0_id33.lock_acq_enable = true;
    bd_memtile_2_0_id33.lock_acq_value = -1;
    bd_memtile_2_0_id33.lock_acq_id = 64 + 20;
    bd_memtile_2_0_id33.lock_rel_value = +1;
    bd_memtile_2_0_id33.lock_rel_id = 64 + 21;
    bd_memtile_2_0_id33.use_next_bd = true;
    bd_memtile_2_0_id33.next_bd = 39;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 33, bd_memtile_2_0_id33);

    adf::dma_buffer_descriptor bd_memtile_2_0_id34;
    bd_memtile_2_0_id34.address = 0 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_2_0_id34.length = 512;
    bd_memtile_2_0_id34.stepsize = {1, 128, 1024};
    bd_memtile_2_0_id34.wrap = {128, 1};
    bd_memtile_2_0_id34.padding = {};
    bd_memtile_2_0_id34.lock_acq_enable = true;
    bd_memtile_2_0_id34.lock_acq_value = -4;
    bd_memtile_2_0_id34.lock_acq_id = 64 + 21;
    bd_memtile_2_0_id34.lock_rel_value = +1;
    bd_memtile_2_0_id34.lock_rel_id = 64 + 17;
    bd_memtile_2_0_id34.use_next_bd = true;
    bd_memtile_2_0_id34.next_bd = 35;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 34, bd_memtile_2_0_id34);

    adf::dma_buffer_descriptor bd_memtile_2_0_id35;
    bd_memtile_2_0_id35.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_2_0_id35.length = 0;
    bd_memtile_2_0_id35.stepsize = {1};
    bd_memtile_2_0_id35.wrap = {};
    bd_memtile_2_0_id35.padding = {};
    bd_memtile_2_0_id35.lock_acq_enable = true;
    bd_memtile_2_0_id35.lock_acq_value = +0;
    bd_memtile_2_0_id35.lock_acq_id = 64 + 0;
    bd_memtile_2_0_id35.lock_rel_value = +1;
    bd_memtile_2_0_id35.lock_rel_id = 64 + 18;
    bd_memtile_2_0_id35.use_next_bd = true;
    bd_memtile_2_0_id35.next_bd = 36;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 35, bd_memtile_2_0_id35);

    adf::dma_buffer_descriptor bd_memtile_2_0_id36;
    bd_memtile_2_0_id36.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_2_0_id36.length = 0;
    bd_memtile_2_0_id36.stepsize = {1};
    bd_memtile_2_0_id36.wrap = {};
    bd_memtile_2_0_id36.padding = {};
    bd_memtile_2_0_id36.lock_acq_enable = true;
    bd_memtile_2_0_id36.lock_acq_value = +0;
    bd_memtile_2_0_id36.lock_acq_id = 64 + 0;
    bd_memtile_2_0_id36.lock_rel_value = +1;
    bd_memtile_2_0_id36.lock_rel_id = 64 + 19;
    bd_memtile_2_0_id36.use_next_bd = true;
    bd_memtile_2_0_id36.next_bd = 37;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 36, bd_memtile_2_0_id36);

    adf::dma_buffer_descriptor bd_memtile_2_0_id37;
    bd_memtile_2_0_id37.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_2_0_id37.length = 0;
    bd_memtile_2_0_id37.stepsize = {1};
    bd_memtile_2_0_id37.wrap = {};
    bd_memtile_2_0_id37.padding = {};
    bd_memtile_2_0_id37.lock_acq_enable = true;
    bd_memtile_2_0_id37.lock_acq_value = +0;
    bd_memtile_2_0_id37.lock_acq_id = 64 + 0;
    bd_memtile_2_0_id37.lock_rel_value = +1;
    bd_memtile_2_0_id37.lock_rel_id = 64 + 20;
    bd_memtile_2_0_id37.use_next_bd = true;
    bd_memtile_2_0_id37.next_bd = 40;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 37, bd_memtile_2_0_id37);

    adf::dma_buffer_descriptor bd_memtile_2_0_id17;
    bd_memtile_2_0_id17.address = 0 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_2_0_id17.length = 1024;
    bd_memtile_2_0_id17.stepsize = {1};
    bd_memtile_2_0_id17.wrap = {};
    bd_memtile_2_0_id17.padding = {};
    bd_memtile_2_0_id17.lock_acq_enable = true;
    bd_memtile_2_0_id17.lock_acq_value = -1;
    bd_memtile_2_0_id17.lock_acq_id = 64 + 22;
    bd_memtile_2_0_id17.lock_rel_value = +1;
    bd_memtile_2_0_id17.lock_rel_id = 64 + 26;
    bd_memtile_2_0_id17.use_next_bd = false;
    bd_memtile_2_0_id17.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 17, bd_memtile_2_0_id17);

    adf::dma_buffer_descriptor bd_memtile_2_0_id38;
    bd_memtile_2_0_id38.address = 1024 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_2_0_id38.length = 1024;
    bd_memtile_2_0_id38.stepsize = {1};
    bd_memtile_2_0_id38.wrap = {};
    bd_memtile_2_0_id38.padding = {};
    bd_memtile_2_0_id38.lock_acq_enable = true;
    bd_memtile_2_0_id38.lock_acq_value = -1;
    bd_memtile_2_0_id38.lock_acq_id = 64 + 23;
    bd_memtile_2_0_id38.lock_rel_value = +1;
    bd_memtile_2_0_id38.lock_rel_id = 64 + 26;
    bd_memtile_2_0_id38.use_next_bd = false;
    bd_memtile_2_0_id38.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 38, bd_memtile_2_0_id38);

    adf::dma_buffer_descriptor bd_memtile_2_0_id18;
    bd_memtile_2_0_id18.address = 2048 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_2_0_id18.length = 1024;
    bd_memtile_2_0_id18.stepsize = {1};
    bd_memtile_2_0_id18.wrap = {};
    bd_memtile_2_0_id18.padding = {};
    bd_memtile_2_0_id18.lock_acq_enable = true;
    bd_memtile_2_0_id18.lock_acq_value = -1;
    bd_memtile_2_0_id18.lock_acq_id = 64 + 24;
    bd_memtile_2_0_id18.lock_rel_value = +1;
    bd_memtile_2_0_id18.lock_rel_id = 64 + 26;
    bd_memtile_2_0_id18.use_next_bd = false;
    bd_memtile_2_0_id18.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 18, bd_memtile_2_0_id18);

    adf::dma_buffer_descriptor bd_memtile_2_0_id39;
    bd_memtile_2_0_id39.address = 3072 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_2_0_id39.length = 1024;
    bd_memtile_2_0_id39.stepsize = {1};
    bd_memtile_2_0_id39.wrap = {};
    bd_memtile_2_0_id39.padding = {};
    bd_memtile_2_0_id39.lock_acq_enable = true;
    bd_memtile_2_0_id39.lock_acq_value = -1;
    bd_memtile_2_0_id39.lock_acq_id = 64 + 25;
    bd_memtile_2_0_id39.lock_rel_value = +1;
    bd_memtile_2_0_id39.lock_rel_id = 64 + 26;
    bd_memtile_2_0_id39.use_next_bd = false;
    bd_memtile_2_0_id39.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 39, bd_memtile_2_0_id39);

    adf::dma_buffer_descriptor bd_memtile_2_0_id40;
    bd_memtile_2_0_id40.address = 0 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_2_0_id40.length = 512;
    bd_memtile_2_0_id40.stepsize = {1, 128, 1024};
    bd_memtile_2_0_id40.wrap = {128, 1};
    bd_memtile_2_0_id40.padding = {};
    bd_memtile_2_0_id40.lock_acq_enable = true;
    bd_memtile_2_0_id40.lock_acq_value = -4;
    bd_memtile_2_0_id40.lock_acq_id = 64 + 26;
    bd_memtile_2_0_id40.lock_rel_value = +1;
    bd_memtile_2_0_id40.lock_rel_id = 64 + 22;
    bd_memtile_2_0_id40.use_next_bd = true;
    bd_memtile_2_0_id40.next_bd = 41;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 40, bd_memtile_2_0_id40);

    adf::dma_buffer_descriptor bd_memtile_2_0_id41;
    bd_memtile_2_0_id41.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_2_0_id41.length = 0;
    bd_memtile_2_0_id41.stepsize = {1};
    bd_memtile_2_0_id41.wrap = {};
    bd_memtile_2_0_id41.padding = {};
    bd_memtile_2_0_id41.lock_acq_enable = true;
    bd_memtile_2_0_id41.lock_acq_value = +0;
    bd_memtile_2_0_id41.lock_acq_id = 64 + 0;
    bd_memtile_2_0_id41.lock_rel_value = +1;
    bd_memtile_2_0_id41.lock_rel_id = 64 + 23;
    bd_memtile_2_0_id41.use_next_bd = true;
    bd_memtile_2_0_id41.next_bd = 42;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 41, bd_memtile_2_0_id41);

    adf::dma_buffer_descriptor bd_memtile_2_0_id42;
    bd_memtile_2_0_id42.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_2_0_id42.length = 0;
    bd_memtile_2_0_id42.stepsize = {1};
    bd_memtile_2_0_id42.wrap = {};
    bd_memtile_2_0_id42.padding = {};
    bd_memtile_2_0_id42.lock_acq_enable = true;
    bd_memtile_2_0_id42.lock_acq_value = +0;
    bd_memtile_2_0_id42.lock_acq_id = 64 + 0;
    bd_memtile_2_0_id42.lock_rel_value = +1;
    bd_memtile_2_0_id42.lock_rel_id = 64 + 24;
    bd_memtile_2_0_id42.use_next_bd = true;
    bd_memtile_2_0_id42.next_bd = 43;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 42, bd_memtile_2_0_id42);

    adf::dma_buffer_descriptor bd_memtile_2_0_id43;
    bd_memtile_2_0_id43.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_2_0_id43.length = 0;
    bd_memtile_2_0_id43.stepsize = {1};
    bd_memtile_2_0_id43.wrap = {};
    bd_memtile_2_0_id43.padding = {};
    bd_memtile_2_0_id43.lock_acq_enable = true;
    bd_memtile_2_0_id43.lock_acq_value = +0;
    bd_memtile_2_0_id43.lock_acq_id = 64 + 0;
    bd_memtile_2_0_id43.lock_rel_value = +1;
    bd_memtile_2_0_id43.lock_rel_id = 64 + 25;
    bd_memtile_2_0_id43.use_next_bd = false;
    bd_memtile_2_0_id43.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 2, 0, 43, bd_memtile_2_0_id43);

    adf::initializeLock(adf::memory_tile, 2, 0, 17, +1);

    adf::initializeLock(adf::memory_tile, 2, 0, 18, +1);

    adf::initializeLock(adf::memory_tile, 2, 0, 19, +1);

    adf::initializeLock(adf::memory_tile, 2, 0, 20, +1);

    adf::initializeLock(adf::memory_tile, 2, 0, 21, +0);

    adf::initializeLock(adf::memory_tile, 2, 0, 22, +1);

    adf::initializeLock(adf::memory_tile, 2, 0, 23, +1);

    adf::initializeLock(adf::memory_tile, 2, 0, 24, +1);

    adf::initializeLock(adf::memory_tile, 2, 0, 25, +1);

    adf::initializeLock(adf::memory_tile, 2, 0, 26, +0);

    //
    // 4 to 1 Data Transfer
    //
    // Location: memtile_3_0
    //
    // Writers
    // ----------------
    // s2mm_2 BDs: 15 -> 17
    // s2mm_3 BDs: 32 -> 38
    // s2mm_4 BDs: 16 -> 18
    // s2mm_5 BDs: 33 -> 39
    //
    // Readers
    // ----------------
    // mm2s_5 BDs: 34 -> 35 -> 36 -> 37 -> 40 -> 41 -> 42 -> 43
    //
    // Locks
    // ----------------
    // Id: 17, Init: +1
    // Id: 18, Init: +1
    // Id: 19, Init: +1
    // Id: 20, Init: +1
    // Id: 21, Init: +0
    // Id: 22, Init: +1
    // Id: 23, Init: +1
    // Id: 24, Init: +1
    // Id: 25, Init: +1
    // Id: 26, Init: +0
    //

    adf::dma_buffer_descriptor bd_memtile_3_0_id15;
    bd_memtile_3_0_id15.address = 0 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_3_0_id15.length = 1024;
    bd_memtile_3_0_id15.stepsize = {1};
    bd_memtile_3_0_id15.wrap = {};
    bd_memtile_3_0_id15.padding = {};
    bd_memtile_3_0_id15.lock_acq_enable = true;
    bd_memtile_3_0_id15.lock_acq_value = -1;
    bd_memtile_3_0_id15.lock_acq_id = 64 + 17;
    bd_memtile_3_0_id15.lock_rel_value = +1;
    bd_memtile_3_0_id15.lock_rel_id = 64 + 21;
    bd_memtile_3_0_id15.use_next_bd = true;
    bd_memtile_3_0_id15.next_bd = 17;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 15, bd_memtile_3_0_id15);

    adf::dma_buffer_descriptor bd_memtile_3_0_id32;
    bd_memtile_3_0_id32.address = 1024 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_3_0_id32.length = 1024;
    bd_memtile_3_0_id32.stepsize = {1};
    bd_memtile_3_0_id32.wrap = {};
    bd_memtile_3_0_id32.padding = {};
    bd_memtile_3_0_id32.lock_acq_enable = true;
    bd_memtile_3_0_id32.lock_acq_value = -1;
    bd_memtile_3_0_id32.lock_acq_id = 64 + 18;
    bd_memtile_3_0_id32.lock_rel_value = +1;
    bd_memtile_3_0_id32.lock_rel_id = 64 + 21;
    bd_memtile_3_0_id32.use_next_bd = true;
    bd_memtile_3_0_id32.next_bd = 38;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 32, bd_memtile_3_0_id32);

    adf::dma_buffer_descriptor bd_memtile_3_0_id16;
    bd_memtile_3_0_id16.address = 2048 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_3_0_id16.length = 1024;
    bd_memtile_3_0_id16.stepsize = {1};
    bd_memtile_3_0_id16.wrap = {};
    bd_memtile_3_0_id16.padding = {};
    bd_memtile_3_0_id16.lock_acq_enable = true;
    bd_memtile_3_0_id16.lock_acq_value = -1;
    bd_memtile_3_0_id16.lock_acq_id = 64 + 19;
    bd_memtile_3_0_id16.lock_rel_value = +1;
    bd_memtile_3_0_id16.lock_rel_id = 64 + 21;
    bd_memtile_3_0_id16.use_next_bd = true;
    bd_memtile_3_0_id16.next_bd = 18;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 16, bd_memtile_3_0_id16);

    adf::dma_buffer_descriptor bd_memtile_3_0_id33;
    bd_memtile_3_0_id33.address = 3072 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_3_0_id33.length = 1024;
    bd_memtile_3_0_id33.stepsize = {1};
    bd_memtile_3_0_id33.wrap = {};
    bd_memtile_3_0_id33.padding = {};
    bd_memtile_3_0_id33.lock_acq_enable = true;
    bd_memtile_3_0_id33.lock_acq_value = -1;
    bd_memtile_3_0_id33.lock_acq_id = 64 + 20;
    bd_memtile_3_0_id33.lock_rel_value = +1;
    bd_memtile_3_0_id33.lock_rel_id = 64 + 21;
    bd_memtile_3_0_id33.use_next_bd = true;
    bd_memtile_3_0_id33.next_bd = 39;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 33, bd_memtile_3_0_id33);

    adf::dma_buffer_descriptor bd_memtile_3_0_id34;
    bd_memtile_3_0_id34.address = 0 + ((0x80000 + 0x7200) / sizeof(uint32_t));
    bd_memtile_3_0_id34.length = 512;
    bd_memtile_3_0_id34.stepsize = {1, 128, 1024};
    bd_memtile_3_0_id34.wrap = {128, 1};
    bd_memtile_3_0_id34.padding = {};
    bd_memtile_3_0_id34.lock_acq_enable = true;
    bd_memtile_3_0_id34.lock_acq_value = -4;
    bd_memtile_3_0_id34.lock_acq_id = 64 + 21;
    bd_memtile_3_0_id34.lock_rel_value = +1;
    bd_memtile_3_0_id34.lock_rel_id = 64 + 17;
    bd_memtile_3_0_id34.use_next_bd = true;
    bd_memtile_3_0_id34.next_bd = 35;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 34, bd_memtile_3_0_id34);

    adf::dma_buffer_descriptor bd_memtile_3_0_id35;
    bd_memtile_3_0_id35.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_3_0_id35.length = 0;
    bd_memtile_3_0_id35.stepsize = {1};
    bd_memtile_3_0_id35.wrap = {};
    bd_memtile_3_0_id35.padding = {};
    bd_memtile_3_0_id35.lock_acq_enable = true;
    bd_memtile_3_0_id35.lock_acq_value = +0;
    bd_memtile_3_0_id35.lock_acq_id = 64 + 0;
    bd_memtile_3_0_id35.lock_rel_value = +1;
    bd_memtile_3_0_id35.lock_rel_id = 64 + 18;
    bd_memtile_3_0_id35.use_next_bd = true;
    bd_memtile_3_0_id35.next_bd = 36;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 35, bd_memtile_3_0_id35);

    adf::dma_buffer_descriptor bd_memtile_3_0_id36;
    bd_memtile_3_0_id36.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_3_0_id36.length = 0;
    bd_memtile_3_0_id36.stepsize = {1};
    bd_memtile_3_0_id36.wrap = {};
    bd_memtile_3_0_id36.padding = {};
    bd_memtile_3_0_id36.lock_acq_enable = true;
    bd_memtile_3_0_id36.lock_acq_value = +0;
    bd_memtile_3_0_id36.lock_acq_id = 64 + 0;
    bd_memtile_3_0_id36.lock_rel_value = +1;
    bd_memtile_3_0_id36.lock_rel_id = 64 + 19;
    bd_memtile_3_0_id36.use_next_bd = true;
    bd_memtile_3_0_id36.next_bd = 37;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 36, bd_memtile_3_0_id36);

    adf::dma_buffer_descriptor bd_memtile_3_0_id37;
    bd_memtile_3_0_id37.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_3_0_id37.length = 0;
    bd_memtile_3_0_id37.stepsize = {1};
    bd_memtile_3_0_id37.wrap = {};
    bd_memtile_3_0_id37.padding = {};
    bd_memtile_3_0_id37.lock_acq_enable = true;
    bd_memtile_3_0_id37.lock_acq_value = +0;
    bd_memtile_3_0_id37.lock_acq_id = 64 + 0;
    bd_memtile_3_0_id37.lock_rel_value = +1;
    bd_memtile_3_0_id37.lock_rel_id = 64 + 20;
    bd_memtile_3_0_id37.use_next_bd = true;
    bd_memtile_3_0_id37.next_bd = 40;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 37, bd_memtile_3_0_id37);

    adf::dma_buffer_descriptor bd_memtile_3_0_id17;
    bd_memtile_3_0_id17.address = 0 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_3_0_id17.length = 1024;
    bd_memtile_3_0_id17.stepsize = {1};
    bd_memtile_3_0_id17.wrap = {};
    bd_memtile_3_0_id17.padding = {};
    bd_memtile_3_0_id17.lock_acq_enable = true;
    bd_memtile_3_0_id17.lock_acq_value = -1;
    bd_memtile_3_0_id17.lock_acq_id = 64 + 22;
    bd_memtile_3_0_id17.lock_rel_value = +1;
    bd_memtile_3_0_id17.lock_rel_id = 64 + 26;
    bd_memtile_3_0_id17.use_next_bd = false;
    bd_memtile_3_0_id17.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 17, bd_memtile_3_0_id17);

    adf::dma_buffer_descriptor bd_memtile_3_0_id38;
    bd_memtile_3_0_id38.address = 1024 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_3_0_id38.length = 1024;
    bd_memtile_3_0_id38.stepsize = {1};
    bd_memtile_3_0_id38.wrap = {};
    bd_memtile_3_0_id38.padding = {};
    bd_memtile_3_0_id38.lock_acq_enable = true;
    bd_memtile_3_0_id38.lock_acq_value = -1;
    bd_memtile_3_0_id38.lock_acq_id = 64 + 23;
    bd_memtile_3_0_id38.lock_rel_value = +1;
    bd_memtile_3_0_id38.lock_rel_id = 64 + 26;
    bd_memtile_3_0_id38.use_next_bd = false;
    bd_memtile_3_0_id38.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 38, bd_memtile_3_0_id38);

    adf::dma_buffer_descriptor bd_memtile_3_0_id18;
    bd_memtile_3_0_id18.address = 2048 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_3_0_id18.length = 1024;
    bd_memtile_3_0_id18.stepsize = {1};
    bd_memtile_3_0_id18.wrap = {};
    bd_memtile_3_0_id18.padding = {};
    bd_memtile_3_0_id18.lock_acq_enable = true;
    bd_memtile_3_0_id18.lock_acq_value = -1;
    bd_memtile_3_0_id18.lock_acq_id = 64 + 24;
    bd_memtile_3_0_id18.lock_rel_value = +1;
    bd_memtile_3_0_id18.lock_rel_id = 64 + 26;
    bd_memtile_3_0_id18.use_next_bd = false;
    bd_memtile_3_0_id18.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 18, bd_memtile_3_0_id18);

    adf::dma_buffer_descriptor bd_memtile_3_0_id39;
    bd_memtile_3_0_id39.address = 3072 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_3_0_id39.length = 1024;
    bd_memtile_3_0_id39.stepsize = {1};
    bd_memtile_3_0_id39.wrap = {};
    bd_memtile_3_0_id39.padding = {};
    bd_memtile_3_0_id39.lock_acq_enable = true;
    bd_memtile_3_0_id39.lock_acq_value = -1;
    bd_memtile_3_0_id39.lock_acq_id = 64 + 25;
    bd_memtile_3_0_id39.lock_rel_value = +1;
    bd_memtile_3_0_id39.lock_rel_id = 64 + 26;
    bd_memtile_3_0_id39.use_next_bd = false;
    bd_memtile_3_0_id39.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 39, bd_memtile_3_0_id39);

    adf::dma_buffer_descriptor bd_memtile_3_0_id40;
    bd_memtile_3_0_id40.address = 0 + ((0x80000 + 0xb200) / sizeof(uint32_t));
    bd_memtile_3_0_id40.length = 512;
    bd_memtile_3_0_id40.stepsize = {1, 128, 1024};
    bd_memtile_3_0_id40.wrap = {128, 1};
    bd_memtile_3_0_id40.padding = {};
    bd_memtile_3_0_id40.lock_acq_enable = true;
    bd_memtile_3_0_id40.lock_acq_value = -4;
    bd_memtile_3_0_id40.lock_acq_id = 64 + 26;
    bd_memtile_3_0_id40.lock_rel_value = +1;
    bd_memtile_3_0_id40.lock_rel_id = 64 + 22;
    bd_memtile_3_0_id40.use_next_bd = true;
    bd_memtile_3_0_id40.next_bd = 41;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 40, bd_memtile_3_0_id40);

    adf::dma_buffer_descriptor bd_memtile_3_0_id41;
    bd_memtile_3_0_id41.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_3_0_id41.length = 0;
    bd_memtile_3_0_id41.stepsize = {1};
    bd_memtile_3_0_id41.wrap = {};
    bd_memtile_3_0_id41.padding = {};
    bd_memtile_3_0_id41.lock_acq_enable = true;
    bd_memtile_3_0_id41.lock_acq_value = +0;
    bd_memtile_3_0_id41.lock_acq_id = 64 + 0;
    bd_memtile_3_0_id41.lock_rel_value = +1;
    bd_memtile_3_0_id41.lock_rel_id = 64 + 23;
    bd_memtile_3_0_id41.use_next_bd = true;
    bd_memtile_3_0_id41.next_bd = 42;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 41, bd_memtile_3_0_id41);

    adf::dma_buffer_descriptor bd_memtile_3_0_id42;
    bd_memtile_3_0_id42.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_3_0_id42.length = 0;
    bd_memtile_3_0_id42.stepsize = {1};
    bd_memtile_3_0_id42.wrap = {};
    bd_memtile_3_0_id42.padding = {};
    bd_memtile_3_0_id42.lock_acq_enable = true;
    bd_memtile_3_0_id42.lock_acq_value = +0;
    bd_memtile_3_0_id42.lock_acq_id = 64 + 0;
    bd_memtile_3_0_id42.lock_rel_value = +1;
    bd_memtile_3_0_id42.lock_rel_id = 64 + 24;
    bd_memtile_3_0_id42.use_next_bd = true;
    bd_memtile_3_0_id42.next_bd = 43;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 42, bd_memtile_3_0_id42);

    adf::dma_buffer_descriptor bd_memtile_3_0_id43;
    bd_memtile_3_0_id43.address = 0 + ((0x80000 + 0x0) / sizeof(uint32_t));
    bd_memtile_3_0_id43.length = 0;
    bd_memtile_3_0_id43.stepsize = {1};
    bd_memtile_3_0_id43.wrap = {};
    bd_memtile_3_0_id43.padding = {};
    bd_memtile_3_0_id43.lock_acq_enable = true;
    bd_memtile_3_0_id43.lock_acq_value = +0;
    bd_memtile_3_0_id43.lock_acq_id = 64 + 0;
    bd_memtile_3_0_id43.lock_rel_value = +1;
    bd_memtile_3_0_id43.lock_rel_id = 64 + 25;
    bd_memtile_3_0_id43.use_next_bd = false;
    bd_memtile_3_0_id43.next_bd = 0;
    adf::configureBufferDescriptor(adf::memory_tile, 3, 0, 43, bd_memtile_3_0_id43);

    adf::initializeLock(adf::memory_tile, 3, 0, 17, +1);

    adf::initializeLock(adf::memory_tile, 3, 0, 18, +1);

    adf::initializeLock(adf::memory_tile, 3, 0, 19, +1);

    adf::initializeLock(adf::memory_tile, 3, 0, 20, +1);

    adf::initializeLock(adf::memory_tile, 3, 0, 21, +0);

    adf::initializeLock(adf::memory_tile, 3, 0, 22, +1);

    adf::initializeLock(adf::memory_tile, 3, 0, 23, +1);

    adf::initializeLock(adf::memory_tile, 3, 0, 24, +1);

    adf::initializeLock(adf::memory_tile, 3, 0, 25, +1);

    adf::initializeLock(adf::memory_tile, 3, 0, 26, +0);

    //
    // 0 to 0 Data Transfer
    //
    // Location: None
    //
    // Writers
    // ----------------
    // None
    //
    // Readers
    // ----------------
    // None
    //
    // Locks
    // ----------------
    // Id: 0, Init: +0
    // Id: 0, Init: +0
    // Id: 0, Init: +0
    // Id: 0, Init: +0
    // Id: 0, Init: +0
    // Id: 0, Init: +0
    // Id: 0, Init: +0
    // Id: 0, Init: +0
    //

    adf::initializeLock(adf::memory_tile, 0, 0, 0, +0);

    adf::initializeLock(adf::memory_tile, 1, 0, 0, +0);

    adf::initializeLock(adf::memory_tile, 2, 0, 0, +0);

    adf::initializeLock(adf::memory_tile, 3, 0, 0, +0);

    adf::initializeLock(adf::shim_tile, 0, 0, 0, +0);

    adf::initializeLock(adf::shim_tile, 1, 0, 0, +0);

    adf::initializeLock(adf::shim_tile, 2, 0, 0, +0);

    adf::initializeLock(adf::shim_tile, 3, 0, 0, +0);

    //
    // 1 to 1 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_0_0_id0, Channel: s2mm_0, Repeat: 1
    //
    // Readers
    // ----------------
    // BD: bd_memtile_0_0_id1, Channel: mm2s_0, Repeat: 1
    //

    adf::enqueueTask(adf::memory_tile, 0, 0, adf::dma_s2mm, 0, 0, 1, false);

    adf::enqueueTask(adf::memory_tile, 0, 0, adf::dma_mm2s, 0, 1, 1, false);

    //
    // 1 to 1 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_1_0_id0, Channel: s2mm_0, Repeat: 1
    //
    // Readers
    // ----------------
    // BD: bd_memtile_1_0_id1, Channel: mm2s_0, Repeat: 1
    //

    adf::enqueueTask(adf::memory_tile, 1, 0, adf::dma_s2mm, 0, 0, 1, false);

    adf::enqueueTask(adf::memory_tile, 1, 0, adf::dma_mm2s, 0, 1, 1, false);

    //
    // 1 to 1 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_2_0_id0, Channel: s2mm_0, Repeat: 1
    //
    // Readers
    // ----------------
    // BD: bd_memtile_2_0_id1, Channel: mm2s_0, Repeat: 1
    //

    adf::enqueueTask(adf::memory_tile, 2, 0, adf::dma_s2mm, 0, 0, 1, false);

    adf::enqueueTask(adf::memory_tile, 2, 0, adf::dma_mm2s, 0, 1, 1, false);

    //
    // 1 to 1 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_3_0_id0, Channel: s2mm_0, Repeat: 1
    //
    // Readers
    // ----------------
    // BD: bd_memtile_3_0_id1, Channel: mm2s_0, Repeat: 1
    //

    adf::enqueueTask(adf::memory_tile, 3, 0, adf::dma_s2mm, 0, 0, 1, false);

    adf::enqueueTask(adf::memory_tile, 3, 0, adf::dma_mm2s, 0, 1, 1, false);

    //
    // 1 to 1 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_0_0_id2, Channel: s2mm_0, Repeat: 1
    //
    // Readers
    // ----------------
    // BD: bd_memtile_0_0_id3, Channel: mm2s_0, Repeat: 2
    //

    adf::enqueueTask(adf::memory_tile, 0, 0, adf::dma_s2mm, 0, 2, 1, false);

    adf::enqueueTask(adf::memory_tile, 0, 0, adf::dma_mm2s, 0, 3, 2, false);

    //
    // 1 to 1 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_1_0_id2, Channel: s2mm_0, Repeat: 1
    //
    // Readers
    // ----------------
    // BD: bd_memtile_1_0_id3, Channel: mm2s_0, Repeat: 2
    //

    adf::enqueueTask(adf::memory_tile, 1, 0, adf::dma_s2mm, 0, 2, 1, false);

    adf::enqueueTask(adf::memory_tile, 1, 0, adf::dma_mm2s, 0, 3, 2, false);

    //
    // 1 to 1 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_2_0_id2, Channel: s2mm_0, Repeat: 1
    //
    // Readers
    // ----------------
    // BD: bd_memtile_2_0_id3, Channel: mm2s_0, Repeat: 2
    //

    adf::enqueueTask(adf::memory_tile, 2, 0, adf::dma_s2mm, 0, 2, 1, false);

    adf::enqueueTask(adf::memory_tile, 2, 0, adf::dma_mm2s, 0, 3, 2, false);

    //
    // 1 to 1 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_3_0_id2, Channel: s2mm_0, Repeat: 1
    //
    // Readers
    // ----------------
    // BD: bd_memtile_3_0_id3, Channel: mm2s_0, Repeat: 2
    //

    adf::enqueueTask(adf::memory_tile, 3, 0, adf::dma_s2mm, 0, 2, 1, false);

    adf::enqueueTask(adf::memory_tile, 3, 0, adf::dma_mm2s, 0, 3, 2, false);

    //
    // 1 to 2 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_0_0_id7, Channel: s2mm_0, Repeat: 129
    //
    // Readers
    // ----------------
    // BD: bd_memtile_0_0_id24, Channel: mm2s_1, Repeat: 129
    // BD: bd_memtile_0_0_id9, Channel: mm2s_2, Repeat: 129
    //

    adf::enqueueTask(adf::memory_tile, 0, 0, adf::dma_s2mm, 0, 7, 129, false);

    adf::enqueueTask(adf::memory_tile, 0, 0, adf::dma_mm2s, 1, 24, 129, false);

    adf::enqueueTask(adf::memory_tile, 0, 0, adf::dma_mm2s, 2, 9, 129, false);

    //
    // 1 to 2 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_1_0_id7, Channel: s2mm_0, Repeat: 129
    //
    // Readers
    // ----------------
    // BD: bd_memtile_1_0_id24, Channel: mm2s_1, Repeat: 129
    // BD: bd_memtile_1_0_id9, Channel: mm2s_2, Repeat: 129
    //

    adf::enqueueTask(adf::memory_tile, 1, 0, adf::dma_s2mm, 0, 7, 129, false);

    adf::enqueueTask(adf::memory_tile, 1, 0, adf::dma_mm2s, 1, 24, 129, false);

    adf::enqueueTask(adf::memory_tile, 1, 0, adf::dma_mm2s, 2, 9, 129, false);

    //
    // 1 to 2 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_2_0_id7, Channel: s2mm_0, Repeat: 129
    //
    // Readers
    // ----------------
    // BD: bd_memtile_2_0_id24, Channel: mm2s_1, Repeat: 129
    // BD: bd_memtile_2_0_id9, Channel: mm2s_2, Repeat: 129
    //

    adf::enqueueTask(adf::memory_tile, 2, 0, adf::dma_s2mm, 0, 7, 129, false);

    adf::enqueueTask(adf::memory_tile, 2, 0, adf::dma_mm2s, 1, 24, 129, false);

    adf::enqueueTask(adf::memory_tile, 2, 0, adf::dma_mm2s, 2, 9, 129, false);

    //
    // 1 to 2 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_3_0_id7, Channel: s2mm_0, Repeat: 129
    //
    // Readers
    // ----------------
    // BD: bd_memtile_3_0_id24, Channel: mm2s_1, Repeat: 129
    // BD: bd_memtile_3_0_id9, Channel: mm2s_2, Repeat: 129
    //

    adf::enqueueTask(adf::memory_tile, 3, 0, adf::dma_s2mm, 0, 7, 129, false);

    adf::enqueueTask(adf::memory_tile, 3, 0, adf::dma_mm2s, 1, 24, 129, false);

    adf::enqueueTask(adf::memory_tile, 3, 0, adf::dma_mm2s, 2, 9, 129, false);

    //
    // 1 to 2 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_0_0_id26, Channel: s2mm_1, Repeat: 129
    //
    // Readers
    // ----------------
    // BD: bd_memtile_0_0_id28, Channel: mm2s_3, Repeat: 129
    // BD: bd_memtile_0_0_id13, Channel: mm2s_4, Repeat: 129
    //

    adf::enqueueTask(adf::memory_tile, 0, 0, adf::dma_s2mm, 1, 26, 129, false);

    adf::enqueueTask(adf::memory_tile, 0, 0, adf::dma_mm2s, 3, 28, 129, false);

    adf::enqueueTask(adf::memory_tile, 0, 0, adf::dma_mm2s, 4, 13, 129, false);

    //
    // 1 to 2 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_1_0_id26, Channel: s2mm_1, Repeat: 129
    //
    // Readers
    // ----------------
    // BD: bd_memtile_1_0_id28, Channel: mm2s_3, Repeat: 129
    // BD: bd_memtile_1_0_id13, Channel: mm2s_4, Repeat: 129
    //

    adf::enqueueTask(adf::memory_tile, 1, 0, adf::dma_s2mm, 1, 26, 129, false);

    adf::enqueueTask(adf::memory_tile, 1, 0, adf::dma_mm2s, 3, 28, 129, false);

    adf::enqueueTask(adf::memory_tile, 1, 0, adf::dma_mm2s, 4, 13, 129, false);

    //
    // 1 to 2 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_2_0_id26, Channel: s2mm_1, Repeat: 129
    //
    // Readers
    // ----------------
    // BD: bd_memtile_2_0_id28, Channel: mm2s_3, Repeat: 129
    // BD: bd_memtile_2_0_id13, Channel: mm2s_4, Repeat: 129
    //

    adf::enqueueTask(adf::memory_tile, 2, 0, adf::dma_s2mm, 1, 26, 129, false);

    adf::enqueueTask(adf::memory_tile, 2, 0, adf::dma_mm2s, 3, 28, 129, false);

    adf::enqueueTask(adf::memory_tile, 2, 0, adf::dma_mm2s, 4, 13, 129, false);

    //
    // 1 to 2 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_3_0_id26, Channel: s2mm_1, Repeat: 129
    //
    // Readers
    // ----------------
    // BD: bd_memtile_3_0_id28, Channel: mm2s_3, Repeat: 129
    // BD: bd_memtile_3_0_id13, Channel: mm2s_4, Repeat: 129
    //

    adf::enqueueTask(adf::memory_tile, 3, 0, adf::dma_s2mm, 1, 26, 129, false);

    adf::enqueueTask(adf::memory_tile, 3, 0, adf::dma_mm2s, 3, 28, 129, false);

    adf::enqueueTask(adf::memory_tile, 3, 0, adf::dma_mm2s, 4, 13, 129, false);

    //
    // 4 to 1 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_0_0_id15, Channel: s2mm_2, Repeat: 1
    // BD: bd_memtile_0_0_id32, Channel: s2mm_3, Repeat: 1
    // BD: bd_memtile_0_0_id16, Channel: s2mm_4, Repeat: 1
    // BD: bd_memtile_0_0_id33, Channel: s2mm_5, Repeat: 1
    //
    // Readers
    // ----------------
    // BD: bd_memtile_0_0_id34, Channel: mm2s_5, Repeat: 1
    //

    adf::enqueueTask(adf::memory_tile, 0, 0, adf::dma_s2mm, 2, 15, 1, false);

    adf::enqueueTask(adf::memory_tile, 0, 0, adf::dma_s2mm, 3, 32, 1, false);

    adf::enqueueTask(adf::memory_tile, 0, 0, adf::dma_s2mm, 4, 16, 1, false);

    adf::enqueueTask(adf::memory_tile, 0, 0, adf::dma_s2mm, 5, 33, 1, false);

    adf::enqueueTask(adf::memory_tile, 0, 0, adf::dma_mm2s, 5, 34, 1, false);

    //
    // 4 to 1 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_1_0_id15, Channel: s2mm_2, Repeat: 1
    // BD: bd_memtile_1_0_id32, Channel: s2mm_3, Repeat: 1
    // BD: bd_memtile_1_0_id16, Channel: s2mm_4, Repeat: 1
    // BD: bd_memtile_1_0_id33, Channel: s2mm_5, Repeat: 1
    //
    // Readers
    // ----------------
    // BD: bd_memtile_1_0_id34, Channel: mm2s_5, Repeat: 1
    //

    adf::enqueueTask(adf::memory_tile, 1, 0, adf::dma_s2mm, 2, 15, 1, false);

    adf::enqueueTask(adf::memory_tile, 1, 0, adf::dma_s2mm, 3, 32, 1, false);

    adf::enqueueTask(adf::memory_tile, 1, 0, adf::dma_s2mm, 4, 16, 1, false);

    adf::enqueueTask(adf::memory_tile, 1, 0, adf::dma_s2mm, 5, 33, 1, false);

    adf::enqueueTask(adf::memory_tile, 1, 0, adf::dma_mm2s, 5, 34, 1, false);

    //
    // 4 to 1 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_2_0_id15, Channel: s2mm_2, Repeat: 1
    // BD: bd_memtile_2_0_id32, Channel: s2mm_3, Repeat: 1
    // BD: bd_memtile_2_0_id16, Channel: s2mm_4, Repeat: 1
    // BD: bd_memtile_2_0_id33, Channel: s2mm_5, Repeat: 1
    //
    // Readers
    // ----------------
    // BD: bd_memtile_2_0_id34, Channel: mm2s_5, Repeat: 1
    //

    adf::enqueueTask(adf::memory_tile, 2, 0, adf::dma_s2mm, 2, 15, 1, false);

    adf::enqueueTask(adf::memory_tile, 2, 0, adf::dma_s2mm, 3, 32, 1, false);

    adf::enqueueTask(adf::memory_tile, 2, 0, adf::dma_s2mm, 4, 16, 1, false);

    adf::enqueueTask(adf::memory_tile, 2, 0, adf::dma_s2mm, 5, 33, 1, false);

    adf::enqueueTask(adf::memory_tile, 2, 0, adf::dma_mm2s, 5, 34, 1, false);

    //
    // 4 to 1 Task Enqueue
    //
    // Writers
    // ----------------
    // BD: bd_memtile_3_0_id15, Channel: s2mm_2, Repeat: 1
    // BD: bd_memtile_3_0_id32, Channel: s2mm_3, Repeat: 1
    // BD: bd_memtile_3_0_id16, Channel: s2mm_4, Repeat: 1
    // BD: bd_memtile_3_0_id33, Channel: s2mm_5, Repeat: 1
    //
    // Readers
    // ----------------
    // BD: bd_memtile_3_0_id34, Channel: mm2s_5, Repeat: 1
    //

    adf::enqueueTask(adf::memory_tile, 3, 0, adf::dma_s2mm, 2, 15, 1, false);

    adf::enqueueTask(adf::memory_tile, 3, 0, adf::dma_s2mm, 3, 32, 1, false);

    adf::enqueueTask(adf::memory_tile, 3, 0, adf::dma_s2mm, 4, 16, 1, false);

    adf::enqueueTask(adf::memory_tile, 3, 0, adf::dma_s2mm, 5, 33, 1, false);

    adf::enqueueTask(adf::memory_tile, 3, 0, adf::dma_mm2s, 5, 34, 1, false);

    //
    // 0 to 0 Task Enqueue
    //
    // Writers
    // ----------------
    // 
    //
    // Readers
    // ----------------
    // 
    //

    adf::waitDMAChannelDone(adf::memory_tile, 0, 0, adf::dma_s2mm, 0);

    adf::waitDMAChannelDone(adf::memory_tile, 0, 0, adf::dma_mm2s, 0);

    adf::waitDMAChannelDone(adf::memory_tile, 1, 0, adf::dma_s2mm, 0);

    adf::waitDMAChannelDone(adf::memory_tile, 1, 0, adf::dma_mm2s, 0);

    adf::waitDMAChannelDone(adf::memory_tile, 2, 0, adf::dma_s2mm, 0);

    adf::waitDMAChannelDone(adf::memory_tile, 2, 0, adf::dma_mm2s, 0);

    adf::waitDMAChannelDone(adf::memory_tile, 3, 0, adf::dma_s2mm, 0);

    adf::waitDMAChannelDone(adf::memory_tile, 3, 0, adf::dma_mm2s, 0);

    adf::waitDMAChannelDone(adf::memory_tile, 0, 0, adf::dma_mm2s, 1);

    adf::waitDMAChannelDone(adf::memory_tile, 0, 0, adf::dma_mm2s, 2);

    adf::waitDMAChannelDone(adf::memory_tile, 1, 0, adf::dma_mm2s, 1);

    adf::waitDMAChannelDone(adf::memory_tile, 1, 0, adf::dma_mm2s, 2);

    adf::waitDMAChannelDone(adf::memory_tile, 2, 0, adf::dma_mm2s, 1);

    adf::waitDMAChannelDone(adf::memory_tile, 2, 0, adf::dma_mm2s, 2);

    adf::waitDMAChannelDone(adf::memory_tile, 3, 0, adf::dma_mm2s, 1);

    adf::waitDMAChannelDone(adf::memory_tile, 3, 0, adf::dma_mm2s, 2);

    adf::waitDMAChannelDone(adf::memory_tile, 0, 0, adf::dma_s2mm, 1);

    adf::waitDMAChannelDone(adf::memory_tile, 0, 0, adf::dma_mm2s, 3);

    adf::waitDMAChannelDone(adf::memory_tile, 0, 0, adf::dma_mm2s, 4);

    adf::waitDMAChannelDone(adf::memory_tile, 1, 0, adf::dma_s2mm, 1);

    adf::waitDMAChannelDone(adf::memory_tile, 1, 0, adf::dma_mm2s, 3);

    adf::waitDMAChannelDone(adf::memory_tile, 1, 0, adf::dma_mm2s, 4);

    adf::waitDMAChannelDone(adf::memory_tile, 2, 0, adf::dma_s2mm, 1);

    adf::waitDMAChannelDone(adf::memory_tile, 2, 0, adf::dma_mm2s, 3);

    adf::waitDMAChannelDone(adf::memory_tile, 2, 0, adf::dma_mm2s, 4);

    adf::waitDMAChannelDone(adf::memory_tile, 3, 0, adf::dma_s2mm, 1);

    adf::waitDMAChannelDone(adf::memory_tile, 3, 0, adf::dma_mm2s, 3);

    adf::waitDMAChannelDone(adf::memory_tile, 3, 0, adf::dma_mm2s, 4);

    adf::waitDMAChannelDone(adf::memory_tile, 0, 0, adf::dma_s2mm, 2);

    adf::waitDMAChannelDone(adf::memory_tile, 0, 0, adf::dma_s2mm, 3);

    adf::waitDMAChannelDone(adf::memory_tile, 0, 0, adf::dma_s2mm, 4);

    adf::waitDMAChannelDone(adf::memory_tile, 0, 0, adf::dma_s2mm, 5);

    adf::waitDMAChannelDone(adf::memory_tile, 0, 0, adf::dma_mm2s, 5);

    adf::waitDMAChannelDone(adf::memory_tile, 1, 0, adf::dma_s2mm, 2);

    adf::waitDMAChannelDone(adf::memory_tile, 1, 0, adf::dma_s2mm, 3);

    adf::waitDMAChannelDone(adf::memory_tile, 1, 0, adf::dma_s2mm, 4);

    adf::waitDMAChannelDone(adf::memory_tile, 1, 0, adf::dma_s2mm, 5);

    adf::waitDMAChannelDone(adf::memory_tile, 1, 0, adf::dma_mm2s, 5);

    adf::waitDMAChannelDone(adf::memory_tile, 2, 0, adf::dma_s2mm, 2);

    adf::waitDMAChannelDone(adf::memory_tile, 2, 0, adf::dma_s2mm, 3);

    adf::waitDMAChannelDone(adf::memory_tile, 2, 0, adf::dma_s2mm, 4);

    adf::waitDMAChannelDone(adf::memory_tile, 2, 0, adf::dma_s2mm, 5);

    adf::waitDMAChannelDone(adf::memory_tile, 2, 0, adf::dma_mm2s, 5);

    adf::waitDMAChannelDone(adf::memory_tile, 3, 0, adf::dma_s2mm, 2);

    adf::waitDMAChannelDone(adf::memory_tile, 3, 0, adf::dma_s2mm, 3);

    adf::waitDMAChannelDone(adf::memory_tile, 3, 0, adf::dma_s2mm, 4);

    adf::waitDMAChannelDone(adf::memory_tile, 3, 0, adf::dma_s2mm, 5);

    adf::waitDMAChannelDone(adf::memory_tile, 3, 0, adf::dma_mm2s, 5);

}

