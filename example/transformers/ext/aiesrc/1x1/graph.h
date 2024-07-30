#ifndef GRAPH_H
#define GRAPH_H

#include "kernels.h"

extern int8_t buf_wgt1[CORE_WGT_SIZE / 2];
extern int8_t buf_wgt2[CORE_WGT_SIZE / 2];

static int const MEMTILE_IN1_PING_ADDR = 0x00000;
static int const MEMTILE_IN2_PING_ADDR = 0x08000;
static int const MEMTILE_OUT_PING_ADDR = 0x10000;

static int const MEMTILE_IN1_PONG_ADDR = 0x40000;
static int const MEMTILE_IN2_PONG_ADDR = 0x48000;
static int const MEMTILE_OUT_PONG_ADDR = 0x50000;

class SingleCoreGraph : public adf::graph
{
public:
    adf::input_gmio gmio_in1;
    adf::input_gmio gmio_in2;
    adf::output_gmio gmio_out;
private:
    adf::kernel core;
    adf::shared_buffer<int8_t> buf_in1;
    adf::shared_buffer<int8_t> buf_in2;
    adf::shared_buffer<int8_t> buf_out;

public:
    SingleCoreGraph()
    {
        //
        // Core Initialization
        //

        core = adf::kernel::create(gemm_wrapper);
        adf::runtime<adf::ratio>(core) = 1.0;
        adf::source(core) = "kernels.cc";
        adf::headers(core) = {"core_buffers.h"};
        adf::location<adf::kernel>(core) = adf::tile(0, 0);

        adf::dimensions(core.in[0]) = {CORE_IN1_SIZE};
        adf::location<adf::buffer>(core.in[0]) = {adf::address(0, 0, CORE_IN1_PING_ADDR),
                                                  adf::address(0, 0, CORE_IN1_PONG_ADDR)};
        adf::location<adf::dma>(core.in[0]) = adf::dma_channel(adf::aie_tile, 0, 0, 0);

        adf::dimensions(core.in[1]) = {CORE_IN2_SIZE};
        adf::location<adf::buffer>(core.in[1]) = {adf::address(0, 0, CORE_IN2_PING_ADDR),
                                                  adf::address(0, 0, CORE_IN2_PONG_ADDR)};
        adf::location<adf::dma>(core.in[1]) = adf::dma_channel(adf::aie_tile, 0, 0, 1);

        adf::dimensions(core.out[0]) = {CORE_OUT_SIZE};
        adf::single_buffer(core.out[0]);
        adf::location<adf::buffer>(core.out[0]) = adf::address(0, 0, CORE_OUT_PING_ADDR);
        adf::location<adf::dma>(core.out[0]) = adf::dma_channel(adf::aie_tile, 0, 0, 0);

        adf::parameter param_wgt1 = adf::parameter::array(buf_wgt1);
        adf::connect(param_wgt1, core);
        adf::location<adf::parameter>(param_wgt1) = adf::address(0, 0, CORE_WGT1_ADDR);

        adf::parameter param_wgt2 = adf::parameter::array(buf_wgt2);
        adf::connect(param_wgt2, core);
        adf::location<adf::parameter>(param_wgt2) = adf::address(0, 0, CORE_WGT2_ADDR);

        adf::location<adf::stack>(core) = adf::address(0, 0, CORE_STACK_ADDR);

        //
        // MemTile Initialization
        //

        buf_in1 = adf::shared_buffer<int8_t>::create({CORE_IN1_SIZE}, 1, 1);
        adf::num_buffers(buf_in1) = 2;
        adf::location<adf::buffer>(buf_in1) = {adf::address(0, 0, MEMTILE_IN1_PING_ADDR),
                                               adf::address(0, 0, MEMTILE_IN1_PONG_ADDR)};
        adf::location<adf::dma>(buf_in1.in[0]) = adf::dma_channel(adf::memory_tile, 0, 0, 0);
        adf::location<adf::dma>(buf_in1.out[0]) = adf::dma_channel(adf::memory_tile, 0, 0, 0);

        buf_in2 = adf::shared_buffer<int8_t>::create({CORE_IN2_SIZE}, 1, 1);
        adf::num_buffers(buf_in2) = 2;
        adf::location<adf::buffer>(buf_in2) = {adf::address(0, 0, MEMTILE_IN2_PING_ADDR),
                                               adf::address(0, 0, MEMTILE_IN2_PONG_ADDR)};
        adf::location<adf::dma>(buf_in2.in[0]) = adf::dma_channel(adf::memory_tile, 0, 0, 1);
        adf::location<adf::dma>(buf_in2.out[0]) = adf::dma_channel(adf::memory_tile, 0, 0, 1);

        buf_out = adf::shared_buffer<int8_t>::create({CORE_OUT_SIZE}, 1, 1);
        adf::num_buffers(buf_out) = 2;
        adf::location<adf::buffer>(buf_out) = {adf::address(0, 0, MEMTILE_OUT_PING_ADDR),
                                               adf::address(0, 0, MEMTILE_OUT_PONG_ADDR)};
        adf::location<adf::dma>(buf_out.in[0]) = adf::dma_channel(adf::memory_tile, 0, 0, 2);
        adf::location<adf::dma>(buf_out.out[0]) = adf::dma_channel(adf::memory_tile, 0, 0, 2);

        //
        // GMIO Initialization
        //

        gmio_in1 = adf::input_gmio::create(GMIO_BURST_LENGTH, GMIO_BANDWIDTH);
        adf::location<adf::GMIO>(gmio_in1) = adf::shim(0);
        adf::location<adf::dma>(gmio_in1.out[0]) = adf::dma_channel(adf::shim_tile, 0, 0, 0);

        gmio_in2 = adf::input_gmio::create(GMIO_BURST_LENGTH, GMIO_BANDWIDTH);
        adf::location<adf::GMIO>(gmio_in2) = adf::shim(0);
        adf::location<adf::dma>(gmio_in2.out[0]) = adf::dma_channel(adf::shim_tile, 0, 0, 1);

        gmio_out = adf::output_gmio::create(GMIO_BURST_LENGTH, GMIO_BANDWIDTH);
        adf::location<adf::GMIO>(gmio_out) = adf::shim(0);
        adf::location<adf::dma>(gmio_out.in[0]) = adf::dma_channel(adf::shim_tile, 0, 0, 0);

        //
        // Graph connections
        //

        // connect GMIO --> MemTile
        adf::connect(gmio_in1.out[0], buf_in1.in[0]);
        adf::connect(gmio_in2.out[0], buf_in2.in[0]);

        // connect MemTile --> Core
        adf::connect(buf_in1.out[0], core.in[0]);
        adf::connect(buf_in2.out[0], core.in[1]);

        // connect Core --> MemTile
        adf::connect(core.out[0], buf_out.in[0]);

        // connect Memtile --> GMIO
        adf::connect(buf_out.out[0], gmio_out.in[0]);

        //
        // Tiling parameters
        //

        adf::tiling_parameters full_in1;
        full_in1.buffer_dimension = {CORE_IN1_SIZE};
        full_in1.tiling_dimension = {CORE_IN1_SIZE};
        full_in1.offset = {0};

        adf::read_access(gmio_in1.out[0]) = full_in1;
        adf::write_access(buf_in1.in[0]) = full_in1;
        adf::read_access(buf_in1.out[0]) = full_in1;

        adf::tiling_parameters full_in2;
        full_in2.buffer_dimension = {CORE_IN2_SIZE};
        full_in2.tiling_dimension = {CORE_IN2_SIZE};
        full_in2.offset = {0};

        adf::read_access(gmio_in2.out[0]) = full_in2;
        adf::write_access(buf_in2.in[0]) = full_in2;
        adf::read_access(buf_in2.out[0]) = full_in2;

        adf::tiling_parameters full_out;
        full_out.buffer_dimension = {CORE_OUT_SIZE};
        full_out.tiling_dimension = {CORE_OUT_SIZE};
        full_out.offset = {0};

        adf::write_access(buf_out.in[0]) = full_out;
        adf::read_access(buf_out.out[0]) = full_out;
        adf::write_access(gmio_out.in[0]) = full_out;
    }
};

#endif // GRAPH_H
