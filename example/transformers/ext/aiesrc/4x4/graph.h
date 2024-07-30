#ifndef GRAPH_H
#define GRAPH_H

#include "kernels.h"

extern int8_t buf_wgt1[CORE_WGT_SIZE / 2];
extern int8_t buf_wgt2[CORE_WGT_SIZE / 2];
//extern int8_t buf_acc[CORE_ACC_SIZE];

class ComputeGraph : public adf::graph
{
public:
    adf::input_gmio  gmio_in1[NUM_COLS];
    adf::input_gmio  gmio_in2[NUM_COLS];
    adf::output_gmio gmio_out[NUM_COLS];
private:
    adf::kernel core[NUM_COLS][NUM_ROWS];
    adf::shared_buffer<int8_t> buf_mem[NUM_COLS];

public:
    ComputeGraph()
    {
        for (int col = 0; col < NUM_COLS; ++col) {

            //
            // Core Initialization
            //

            for (int row = 0; row < NUM_ROWS; ++row) {
                core[col][row] = adf::kernel::create(gemm_wrapper);
                adf::runtime<adf::ratio>(core[col][row]) = 1.0;
                adf::source(core[col][row]) = "kernels.cc";
                adf::headers(core[col][row]) = {"core_buffers.h"};
                adf::location<adf::kernel>(core[col][row]) = adf::tile(col, row);

                adf::dimensions(core[col][row].in[0]) = {CORE_IN1_SIZE};
                adf::location<adf::buffer>(core[col][row].in[0]) = {adf::address(col, row, CORE_IN1_PING_ADDR),
                                                                    adf::address(col, row, CORE_IN1_PONG_ADDR)};
                adf::location<adf::dma>(core[col][row].in[0]) = adf::dma_channel(adf::aie_tile, col, row, 0);

                adf::dimensions(core[col][row].in[1]) = {CORE_IN2_SIZE};
                adf::location<adf::buffer>(core[col][row].in[1]) = {adf::address(col, row, CORE_IN2_PING_ADDR),
                                                                    adf::address(col, row, CORE_IN2_PONG_ADDR)};
                adf::location<adf::dma>(core[col][row].in[1]) = adf::dma_channel(adf::aie_tile, col, row, 1);

                adf::dimensions(core[col][row].out[0]) = {CORE_OUT_SIZE};
                adf::single_buffer(core[col][row].out[0]);
                adf::location<adf::buffer>(core[col][row].out[0]) = adf::address(col, row, CORE_OUT_PING_ADDR);
                adf::location<adf::dma>(core[col][row].out[0]) = adf::dma_channel(adf::aie_tile, col, row, 0);

                adf::parameter param_wgt1 = adf::parameter::array(buf_wgt1);
                adf::connect(param_wgt1, core[col][row]);
                adf::location<adf::parameter>(param_wgt1) = adf::address(col, row, CORE_WGT1_ADDR);

                adf::parameter param_wgt2 = adf::parameter::array(buf_wgt2);
                adf::connect(param_wgt2, core[col][row]);
                adf::location<adf::parameter>(param_wgt2) = adf::address(col, row, CORE_WGT2_ADDR);

                // adf::parameter param_acc = adf::parameter::array(buf_acc);
                // adf::connect(param_acc, core[col][row]);
                // adf::location<adf::parameter>(param_acc) = adf::address(col, row, CORE_ACC_ADDR);

                adf::location<adf::stack>(core[col][row]) = adf::address(col, row, CORE_STACK_ADDR);
            }

            //
            // MemTile Initialization
            //

            buf_mem[col] = adf::shared_buffer<int8_t>::create({1 << 19}, 6, 6);
            adf::num_buffers(buf_mem[col]) = 1;
            adf::location<adf::buffer>(buf_mem[col]) = adf::address(col, 0, 0x0);
            for (int i = 0; i < 6; ++i) {
                adf::location<adf::dma>(buf_mem[col].in[i]) = adf::dma_channel(adf::memory_tile, col, 0, i);
                adf::location<adf::dma>(buf_mem[col].out[i]) = adf::dma_channel(adf::memory_tile, col, 0, i);
            }
            for (int i = 0; i < 6; ++i) {
                disable_dma_autostart(buf_mem[col].in[i]);
                disable_dma_autostart(buf_mem[col].out[i]);
            }

            //
            // GMIO Initialization
            //

            gmio_in1[col] = adf::input_gmio::create(GMIO_BURST_LENGTH, GMIO_BANDWIDTH);
            adf::location<adf::GMIO>(gmio_in1[col]) = adf::shim(col);
            adf::location<adf::dma>(gmio_in1[col].out[0]) = adf::dma_channel(adf::shim_tile, col, 0, 0);

            gmio_in2[col] = adf::input_gmio::create(GMIO_BURST_LENGTH, GMIO_BANDWIDTH);
            adf::location<adf::GMIO>(gmio_in2[col]) = adf::shim(col);
            adf::location<adf::dma>(gmio_in2[col].out[0]) = adf::dma_channel(adf::shim_tile, col, 0, 1);

            gmio_out[col] = adf::output_gmio::create(GMIO_BURST_LENGTH, GMIO_BANDWIDTH);
            adf::location<adf::GMIO>(gmio_out[col]) = adf::shim(col);
            adf::location<adf::dma>(gmio_out[col].in[0]) = adf::dma_channel(adf::shim_tile, col, 0, 0);
        }

        for (int col = 0; col < NUM_COLS; ++col) {

            //
            // Graph connections
            //

            // Connect GMIO --> Memtile
            adf::connect(gmio_in1[col].out[0], buf_mem[col].in[0]);
            adf::connect(gmio_in2[col].out[0], buf_mem[col].in[1]);
            // Connect Memtile --> Core input 1 (multi-cast)
            adf::connect(buf_mem[col].out[0], core[0][col].in[0]);
            adf::connect(buf_mem[col].out[0], core[1][col].in[0]);
            adf::connect(buf_mem[col].out[0], core[2][col].in[0]);
            adf::connect(buf_mem[col].out[0], core[3][col].in[0]);
            // Connect Memtile --> Core input 2 (uni-cast)
            adf::connect(buf_mem[col].out[1], core[col][0].in[1]);
            adf::connect(buf_mem[col].out[2], core[col][1].in[1]);
            adf::connect(buf_mem[col].out[3], core[col][2].in[1]);
            adf::connect(buf_mem[col].out[4], core[col][3].in[1]);
            // Connect Core --> Memtile (uni-cast)
            adf::connect(core[col][0].out[0], buf_mem[col].in[2]);
            adf::connect(core[col][1].out[0], buf_mem[col].in[3]);
            adf::connect(core[col][2].out[0], buf_mem[col].in[4]);
            adf::connect(core[col][3].out[0], buf_mem[col].in[5]);
            // Connect Memtile --> GMIO
            adf::connect(buf_mem[col].out[5], gmio_out[col].in[0]);
        }
    }
};

#endif // GRAPH_H
