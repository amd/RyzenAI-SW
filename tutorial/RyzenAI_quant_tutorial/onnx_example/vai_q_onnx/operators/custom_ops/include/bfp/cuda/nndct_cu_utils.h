//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#ifndef _NNDCT_CU_UTILS_H_
#define _NNDCT_CU_UTILS_H_
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <algorithm>
#include <assert.h>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define BLOCKSIZE_COL 64
#define BLOCKSIZE_ROW 4
#define BLOCKSIZE 256
#define NNDCT_CUDA_NUM_THREADS 512
#define CU2DBLOCK 16
#define CU1DBLOCK 256
#define NNDCT_CUDA_NUM_THREADS_COL 512
#define NNDCT_CUDA_NUM_THREADS_ROW 8

#define NNDCT_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
		  i < (n); \
		  i += blockDim.x * gridDim.x)

#define NNDCT_KERNEL_LOOP_2D(i, j, m, n) \
  for (int i = blockIdx.y * blockDim.y + threadIdx.y; \
		  i < (m); \
		  i += blockDim.y * gridDim.y) \
		for (int j = blockIdx.x * blockDim.x + threadIdx.x; \
		  	j < (n); \
		  	j += blockDim.x * gridDim.x)

inline int NNDCT_GET_BLOCKS(const int N){
	return (N + NNDCT_CUDA_NUM_THREADS - 1) / NNDCT_CUDA_NUM_THREADS;
}

inline int n_blocks(int size, int block_size) {
	return size / block_size + ((size % block_size == 0)? 0 : 1);
}

inline int NNDCT_GET_BLOCKS1D(const int N){
	int dimGrid = n_blocks(N, CU1DBLOCK);
	if (dimGrid > 256) {
		dimGrid = 256;
	}
	return dimGrid;
}

dim3 GetGridSizeF(unsigned n);

void GetBlockSizesForSimpleMatrixOperation(int num_rows,
		int num_cols,dim3 *dimGrid,dim3 *dimBlock);

inline dim3 NNDCT_GET_BLOCKS_2D(const int N_row, const int N_col){
	// dim3 blockSize2d((N_col + NNDCT_CUDA_NUM_THREADS_COL - 1) / NNDCT_CUDA_NUM_THREADS_COL, 
	// 			(N_row + NNDCT_CUDA_NUM_THREADS_ROW - 1) / NNDCT_CUDA_NUM_THREADS_ROW);
	dim3 blockSize2d((N_row + NNDCT_CUDA_NUM_THREADS_ROW - 1) / NNDCT_CUDA_NUM_THREADS_ROW, 
				(N_col + NNDCT_CUDA_NUM_THREADS_COL - 1) / NNDCT_CUDA_NUM_THREADS_COL);
	return blockSize2d;
}

#endif //_NNDCT_CU_UTILS_H_