/*
 * Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#ifndef WGT_MATRIX_H
#define WGT_MATRIX_H

#include <assert.h>

/*
 * Weight Matrix Indexing
 *
 * This class implements the interleaved weight matrix indexing scheme
 * to preformat weights for the performance-optimized manual BDs
 * in the vertical cascade GEMM kernels. This formatting allows weights
 * to be sent on all GMIO ports in parallel with stride 1 memory accesses.
 *
 * At a basic level, the core receives 128 x 64 blocks that are
 * W-8 aligned. W-8 alignment can be considered a special case
 * of "Zig-Zag" indexing implemented below with zz = 8.
 * Zig-Zag indexing is a generalization of row-major indexing,
 * where the pitch between rows doesn't necessarily match the
 * number of columns in the matrix.
 *
 * After computing the block zig-zag index, there are three
 * levels of offsets. The matrix is split into a grid with
 * vertical shards for each AIE column, major horizontal shards
 * for each GMIO input in a single column, and minor horizontal shards
 * for each core that shares a GMIO input. For Strix, there are
 * 4 vertical shards and 2 major horizontal shards that are each
 * split into 2 minor horizontal shards.
 *
 * 1) Groups of four blocks are stacked verically
 *    to form a block group.
 *
 * 2) Block groups are interleaved between adjacent
 *    minor horizontal shards. This allows block groups
 *    for all cores sharing a single GMIO port to be loaded
 *    into the memtile with stride 1 (avoids bank conflicts).
 *
 * 3) Major shard chunks (each representing that data sent
 *    to one GMIO port) are column-major ordered.
 *
 * The closed form expressions for this three leveled offset scheme
 * are implemented in the wgt_idx function.
 */
namespace ryzenai {
template <typename T, int BpG = 4> class WgtMatrix {
  /* static constants */
private:
  static int const SUBV_ROWS = 128;
  static int const SUBV_COLS = 64;
  static int const BLOCKS_PER_GROUP = BpG;
  static int const AIE_ROWS = 4;
  static int const AIE_COLS = 4;
  static int const GMIO_INPUTS_PER_COL = 2;
  // subvolume blocks
  static int const block_rows = SUBV_ROWS;
  static int const block_cols = SUBV_COLS;
  static int const block_size = block_rows * block_cols;
  // NOTE: block zig-zag width is detemined by the width
  //       of the MAC unit (8 x 8 x 8 for strix)
  static int const block_zz = 8;
  // interleaved block groups
  static int const inter_blocks = BLOCKS_PER_GROUP;
  static int const inter_rows = block_rows * inter_blocks;
  static int const inter_cols = block_cols;
  static int const inter_size = inter_rows * inter_cols;
  // number of major/minor horizontal shards
  static int const minor_shards = AIE_ROWS / GMIO_INPUTS_PER_COL;
  static int const major_shards = GMIO_INPUTS_PER_COL;
  // number of vertical shards
  static int const vert_shards = AIE_COLS;

  /* instance constants*/
public:
  // raw data buffer
  // NOTE: this must be allocated to num_rows * num_cols
  T *const data;
  // weight matrix shape
  int const num_rows;
  int const num_cols;

private:
  // data interleaving constants
  int const minor_rows;
  int const major_rows;
  int const major_cols;
  int const shard_size;
  int const inter_pitch;

public:
  WgtMatrix(T *data, int num_rows, int num_cols)
      : data(data), num_rows(num_rows), num_cols(num_cols),
        minor_rows(num_rows / (major_shards * minor_shards)),
        major_rows(num_rows / major_shards), major_cols(num_cols / vert_shards),
        shard_size(major_rows * major_cols),
        inter_pitch(minor_rows / inter_rows) {}

  T &operator()(int row, int col) {
    assert(row < num_rows);
    assert(col < num_cols);
    int const idx = wgt_idx(row, col);
    assert(idx < num_rows * num_cols);
    return data[idx];
  }

private:
  // Zig-Zag index
  int zz_idx(int row, int col, int num_rows, int zz) {
    return (row * zz) + (col % zz) + ((col / zz) * (num_rows * zz));
  }

  // Column-Major index
  int cm_idx(int row, int col, int num_rows) { return (col * num_rows) + row; }

  // Weight Matrix index
  int wgt_idx(int row, int col) {
    int zz = zz_idx(row % block_rows, col % block_cols, block_rows, block_zz);
    int inter = cm_idx((row % minor_rows) / inter_rows,
                       (col % major_cols) / inter_cols, inter_pitch);
    int off1 = ((row / block_rows) % inter_blocks);
    int off2 = ((row / minor_rows) % minor_shards) + (inter * minor_shards);
    int off3 = cm_idx(row / major_rows, col / major_cols, major_shards);
    return zz + (off1 * block_size) + (off2 * inter_size) + (off3 * shard_size);
  }
};
} // namespace ryzenai
#endif // WGT_MATRIX_H
