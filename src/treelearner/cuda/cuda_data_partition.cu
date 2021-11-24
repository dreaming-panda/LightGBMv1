/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_data_partition.hpp"

#include <LightGBM/cuda/cuda_algorithms.hpp>
#include <LightGBM/tree.h>

#include <algorithm>
#include <vector>

namespace LightGBM {

__global__ void FillDataIndicesBeforeTrainKernel(const data_size_t num_data,
  data_size_t* data_indices, int* cuda_data_index_to_leaf_index) {
  const unsigned int data_index = threadIdx.x + blockIdx.x * blockDim.x;
  if (data_index < num_data) {
    data_indices[data_index] = data_index;
    cuda_data_index_to_leaf_index[data_index] = 0;
  }
}

__global__ void FillDataIndexToLeafIndexKernel(
  const data_size_t num_data,
  const data_size_t* data_indices,
  int* data_index_to_leaf_index) {
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (data_index < num_data) {
    data_index_to_leaf_index[data_indices[data_index]] = 0;
  }
}

void CUDADataPartition::LaunchFillDataIndicesBeforeTrain() {
  const data_size_t num_data_in_root = root_num_data();
  const int num_blocks = (num_data_in_root + FILL_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / FILL_INDICES_BLOCK_SIZE_DATA_PARTITION;
  FillDataIndicesBeforeTrainKernel<<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(num_data_in_root, cuda_data_indices_, cuda_data_index_to_leaf_index_);
}

void CUDADataPartition::LaunchFillDataIndexToLeafIndex() {
  const data_size_t num_data_in_root = root_num_data();
  const int num_blocks = (num_data_in_root + FILL_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / FILL_INDICES_BLOCK_SIZE_DATA_PARTITION;
  FillDataIndexToLeafIndexKernel<<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(num_data_in_root, cuda_data_indices_, cuda_data_index_to_leaf_index_);
}

__device__ __forceinline__ void PrepareOffset(const data_size_t num_data_in_leaf, uint16_t* block_to_left_offset,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer,
  const uint16_t thread_to_left_offset_cnt, uint16_t* shared_mem_buffer) {
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  const uint16_t thread_to_left_offset = ShufflePrefixSum<uint16_t>(thread_to_left_offset_cnt, shared_mem_buffer);
  const data_size_t num_data_in_block = (blockIdx.x + 1) * blockDim_x <= num_data_in_leaf ? static_cast<data_size_t>(blockDim_x) :
    num_data_in_leaf - static_cast<data_size_t>(blockIdx.x * blockDim_x);
  if (static_cast<data_size_t>(threadIdx_x) < num_data_in_block) {
    block_to_left_offset[threadIdx_x] = thread_to_left_offset;
  }
  if (threadIdx_x == blockDim_x - 1) {
    if (num_data_in_block > 0) {
      const data_size_t data_to_left = static_cast<data_size_t>(thread_to_left_offset);
      block_to_left_offset_buffer[blockIdx.x + 1] = data_to_left;
      block_to_right_offset_buffer[blockIdx.x + 1] = num_data_in_block - data_to_left;
    } else {
      block_to_left_offset_buffer[blockIdx.x + 1] = 0;
      block_to_right_offset_buffer[blockIdx.x + 1] = 0;
    }
  }
}

template <typename T>
__device__ bool CUDAFindInBitset(const uint32_t* bits, int n, T pos) {
  int i1 = pos / 32;
  if (i1 >= n) {
    return false;
  }
  int i2 = pos % 32;
  return (bits[i1] >> i2) & 1;
}



#define UpdateDataIndexToLeafIndexKernel_PARAMS \
  const BIN_TYPE* column_data, \
  const data_size_t num_data_in_leaf, \
  const data_size_t* data_indices_in_leaf, \
  const uint32_t th, \
  const uint32_t t_zero_bin, \
  const uint32_t max_bin, \
  const uint32_t min_bin, \
  const int left_leaf_index, \
  const int right_leaf_index, \
  const int default_leaf_index, \
  const int missing_default_leaf_index

#define UpdateDataIndexToLeafIndex_ARGS \
  column_data, \
  num_data_in_leaf, \
  data_indices_in_leaf, th, \
  t_zero_bin, \
  max_bin, \
  min_bin, \
  left_leaf_index, \
  right_leaf_index, \
  default_leaf_index, \
  missing_default_leaf_index

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, bool MAX_TO_LEFT, typename BIN_TYPE>
__global__ void UpdateDataIndexToLeafIndexKernel(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  int* cuda_data_index_to_leaf_index) {
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(column_data[global_data_index]);
    if (!MIN_IS_MAX) {
      if ((MISSING_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) ||
        (MISSING_IS_NA && !MFB_IS_NA && bin == max_bin)) {
        cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
      } else if (bin < min_bin || bin > max_bin) {
        if ((MISSING_IS_NA && MFB_IS_NA) || (MISSING_IS_ZERO && MFB_IS_ZERO)) {
          cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
        } else {
          cuda_data_index_to_leaf_index[global_data_index] = default_leaf_index;
        }
      } else if (bin > th) {
        cuda_data_index_to_leaf_index[global_data_index] = right_leaf_index;
      } else {
        cuda_data_index_to_leaf_index[global_data_index] = left_leaf_index;
      }
    } else {
      if (MISSING_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) {
        cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
      } else if (bin != max_bin) {
        if ((MISSING_IS_NA && MFB_IS_NA) || (MISSING_IS_ZERO && MFB_IS_ZERO)) {
          cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
        } else {
          cuda_data_index_to_leaf_index[global_data_index] = default_leaf_index;
        }
      } else {
        if (MISSING_IS_NA && !MFB_IS_NA) {
          cuda_data_index_to_leaf_index[global_data_index] = missing_default_leaf_index;
        } else {
          if (!MAX_TO_LEFT) {
            cuda_data_index_to_leaf_index[global_data_index] = right_leaf_index;
          } else {
            cuda_data_index_to_leaf_index[global_data_index] = left_leaf_index;
          }
        }
      }
    }
  }
}

template <typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool missing_is_zero,
  const bool missing_is_na,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_to_left) {
  if (min_bin < max_bin) {
    if (!missing_is_zero) {
      LaunchUpdateDataIndexToLeafIndexKernel_Inner0<false, false, BIN_TYPE>
        (UpdateDataIndexToLeafIndex_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_to_left);
    } else {
      LaunchUpdateDataIndexToLeafIndexKernel_Inner0<false, true, BIN_TYPE>
        (UpdateDataIndexToLeafIndex_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_to_left);
    }
  } else {
    if (!missing_is_zero) {
      LaunchUpdateDataIndexToLeafIndexKernel_Inner0<true, false, BIN_TYPE>
        (UpdateDataIndexToLeafIndex_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_to_left);
    } else {
      LaunchUpdateDataIndexToLeafIndexKernel_Inner0<true, true, BIN_TYPE>
        (UpdateDataIndexToLeafIndex_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_to_left);
    }
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel_Inner0(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool missing_is_na,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_to_left) {
  if (!missing_is_na) {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner1<MIN_IS_MAX, MISSING_IS_ZERO, false, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, mfb_is_zero, mfb_is_na, max_to_left);
  } else {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner1<MIN_IS_MAX, MISSING_IS_ZERO, true, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, mfb_is_zero, mfb_is_na, max_to_left);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel_Inner1(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_to_left) {
  if (!mfb_is_zero) {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner2<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, false, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, mfb_is_na, max_to_left);
  } else {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner2<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, true, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, mfb_is_na, max_to_left);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel_Inner2(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool mfb_is_na,
  const bool max_to_left) {
  if (!mfb_is_na) {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner3<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, false, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, max_to_left);
  } else {
    LaunchUpdateDataIndexToLeafIndexKernel_Inner3<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, true, BIN_TYPE>
      (UpdateDataIndexToLeafIndex_ARGS, max_to_left);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, typename BIN_TYPE>
void CUDADataPartition::LaunchUpdateDataIndexToLeafIndexKernel_Inner3(
  UpdateDataIndexToLeafIndexKernel_PARAMS,
  const bool max_to_left) {
  if (!max_to_left) {
    UpdateDataIndexToLeafIndexKernel<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, false, BIN_TYPE>
      <<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index_);
  } else {
    UpdateDataIndexToLeafIndexKernel<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, true, BIN_TYPE>
      <<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index_);
  }
}

#define GenDataToLeftBitVectorKernel_PARMS \
  const BIN_TYPE* column_data, \
  const data_size_t num_data_in_leaf, \
  const data_size_t* data_indices_in_leaf, \
  const uint32_t th, \
  const uint32_t t_zero_bin, \
  const uint32_t max_bin, \
  const uint32_t min_bin, \
  const uint8_t split_default_to_left, \
  const uint8_t split_missing_default_to_left

#define GenBitVector_ARGS \
  column_data, \
  num_data_in_leaf, \
  data_indices_in_leaf, \
  th, \
  t_zero_bin, \
  max_bin, \
  min_bin, \
  split_default_to_left,  \
  split_missing_default_to_left

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, bool MAX_TO_LEFT, typename BIN_TYPE>
__global__ void GenDataToLeftBitVectorKernel(
  GenDataToLeftBitVectorKernel_PARMS,
  uint16_t* block_to_left_offset,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer) {
  __shared__ uint16_t shared_mem_buffer[32];
  uint16_t thread_to_left_offset_cnt = 0;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(column_data[global_data_index]);
    if (!MIN_IS_MAX) {
      if ((MISSING_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) ||
        (MISSING_IS_NA && !MFB_IS_NA && bin == max_bin)) {
        thread_to_left_offset_cnt = split_missing_default_to_left;
      } else if ((bin < min_bin || bin > max_bin)) {
        if ((MISSING_IS_NA && MFB_IS_NA) || (MISSING_IS_ZERO || MFB_IS_ZERO)) {
          thread_to_left_offset_cnt = split_missing_default_to_left;
        } else {
          thread_to_left_offset_cnt = split_default_to_left;
        }
      } else if (bin <= th) {
        thread_to_left_offset_cnt = 1;
      }
    } else {
      if (MISSING_IS_ZERO && !MFB_IS_ZERO && bin == t_zero_bin) {
        thread_to_left_offset_cnt = split_missing_default_to_left;
      } else if (bin != max_bin) {
        if ((MISSING_IS_NA && MFB_IS_NA) || (MISSING_IS_ZERO && MFB_IS_ZERO)) {
          thread_to_left_offset_cnt = split_missing_default_to_left;
        } else {
          thread_to_left_offset_cnt = split_default_to_left;
        }
      } else {
        if (MISSING_IS_NA && !MFB_IS_NA) {
          thread_to_left_offset_cnt = split_missing_default_to_left;
        } else if (MAX_TO_LEFT) {
          thread_to_left_offset_cnt = 1;
        }
      }
    }
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, block_to_left_offset + blockIdx.x * blockDim.x, block_to_left_offset_buffer, block_to_right_offset_buffer,
    thread_to_left_offset_cnt, shared_mem_buffer);
}

template <typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner(
  GenDataToLeftBitVectorKernel_PARMS,
  const bool missing_is_zero,
  const bool missing_is_na,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_bin_to_left) {
  if (min_bin < max_bin) {
    if (!missing_is_zero) {
      LaunchGenDataToLeftBitVectorKernelInner0<false, false, BIN_TYPE>
        (GenBitVector_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_bin_to_left);
    } else {
      LaunchGenDataToLeftBitVectorKernelInner0<false, true, BIN_TYPE>
        (GenBitVector_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_bin_to_left);
    }
  } else {
    if (!missing_is_zero) {
      LaunchGenDataToLeftBitVectorKernelInner0<true, false, BIN_TYPE>
        (GenBitVector_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_bin_to_left);
    } else {
      LaunchGenDataToLeftBitVectorKernelInner0<true, true, BIN_TYPE>
        (GenBitVector_ARGS, missing_is_na, mfb_is_zero, mfb_is_na, max_bin_to_left);
    }
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner0(
  GenDataToLeftBitVectorKernel_PARMS,
  const bool missing_is_na,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_bin_to_left) {
  if (!missing_is_na) {
    LaunchGenDataToLeftBitVectorKernelInner1<MIN_IS_MAX, MISSING_IS_ZERO, false, BIN_TYPE>
      (GenBitVector_ARGS, mfb_is_zero, mfb_is_na, max_bin_to_left);
  } else {
    LaunchGenDataToLeftBitVectorKernelInner1<MIN_IS_MAX, MISSING_IS_ZERO, true, BIN_TYPE>
      (GenBitVector_ARGS, mfb_is_zero, mfb_is_na, max_bin_to_left);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner1(
  GenDataToLeftBitVectorKernel_PARMS,
  const bool mfb_is_zero,
  const bool mfb_is_na,
  const bool max_bin_to_left) {
  if (!mfb_is_zero) {
    LaunchGenDataToLeftBitVectorKernelInner2<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, false, BIN_TYPE>
      (GenBitVector_ARGS, mfb_is_na, max_bin_to_left);
  } else {
    LaunchGenDataToLeftBitVectorKernelInner2<MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, true, BIN_TYPE>
      (GenBitVector_ARGS, mfb_is_na, max_bin_to_left);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner2(
  GenDataToLeftBitVectorKernel_PARMS,
  const bool mfb_is_na,
  const bool max_bin_to_left) {
  if (!mfb_is_na) {
    LaunchGenDataToLeftBitVectorKernelInner3
      <MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, false, BIN_TYPE>
      (GenBitVector_ARGS, max_bin_to_left);
  } else {
    LaunchGenDataToLeftBitVectorKernelInner3
      <MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, true, BIN_TYPE>
      (GenBitVector_ARGS, max_bin_to_left);
  }
}

template <bool MIN_IS_MAX, bool MISSING_IS_ZERO, bool MISSING_IS_NA, bool MFB_IS_ZERO, bool MFB_IS_NA, typename BIN_TYPE>
void CUDADataPartition::LaunchGenDataToLeftBitVectorKernelInner3(
  GenDataToLeftBitVectorKernel_PARMS,
  const bool max_bin_to_left) {
  if (!max_bin_to_left) {
    GenDataToLeftBitVectorKernel
      <MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, false, BIN_TYPE>
      <<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_ARGS,
        cuda_block_to_left_offset_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_);
  } else {
    GenDataToLeftBitVectorKernel
      <MIN_IS_MAX, MISSING_IS_ZERO, MISSING_IS_NA, MFB_IS_ZERO, MFB_IS_NA, true, BIN_TYPE>
      <<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_ARGS,
        cuda_block_to_left_offset_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_);
  }
}

__device__ void CUDADataPartition::CalcBlockDim(const data_size_t num_data_in_leaf, int* grid_dim, int* block_dim) {
  const int min_num_blocks = num_data_in_leaf <= 100 ? 1 : 80;
  const int num_blocks = max(min_num_blocks, (num_data_in_leaf + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION);
  int split_indices_block_size_data_partition = (num_data_in_leaf + num_blocks - 1) / num_blocks - 1;
  int split_indices_block_size_data_partition_aligned = 1;
  while (split_indices_block_size_data_partition > 0) {
    split_indices_block_size_data_partition_aligned <<= 1;
    split_indices_block_size_data_partition >>= 1;
  }
  const int num_blocks_final = (num_data_in_leaf + split_indices_block_size_data_partition_aligned - 1) / split_indices_block_size_data_partition_aligned;
  *grid_dim = num_blocks_final;
  *block_dim = split_indices_block_size_data_partition_aligned;
}


__global__ void AggregateBlockOffsetKernel0(
  const int left_leaf_index,
  const int right_leaf_index,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer, data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end, data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices,
  const data_size_t num_blocks) {
  __shared__ uint32_t shared_mem_buffer[32];
  __shared__ uint32_t to_left_total_count;
  const data_size_t num_data_in_leaf = cuda_leaf_num_data[left_leaf_index];
  const unsigned int blockDim_x = blockDim.x;
  const unsigned int threadIdx_x = threadIdx.x;
  const data_size_t num_blocks_plus_1 = num_blocks + 1;
  const uint32_t num_blocks_per_thread = (num_blocks_plus_1 + blockDim_x - 1) / blockDim_x;
  const uint32_t remain = num_blocks_plus_1 - ((num_blocks_per_thread - 1) * blockDim_x);
  const uint32_t remain_offset = remain * num_blocks_per_thread;
  uint32_t thread_start_block_index = 0;
  uint32_t thread_end_block_index = 0;
  if (threadIdx_x < remain) {
    thread_start_block_index = threadIdx_x * num_blocks_per_thread;
    thread_end_block_index = min(thread_start_block_index + num_blocks_per_thread, num_blocks_plus_1);
  } else {
    thread_start_block_index = remain_offset + (num_blocks_per_thread - 1) * (threadIdx_x - remain);
    thread_end_block_index = min(thread_start_block_index + num_blocks_per_thread - 1, num_blocks_plus_1);
  }
  if (threadIdx.x == 0) {
    block_to_right_offset_buffer[0] = 0;
  }
  __syncthreads();
  for (uint32_t block_index = thread_start_block_index + 1; block_index < thread_end_block_index; ++block_index) {
    block_to_left_offset_buffer[block_index] += block_to_left_offset_buffer[block_index - 1];
    block_to_right_offset_buffer[block_index] += block_to_right_offset_buffer[block_index - 1];
  }
  __syncthreads();
  uint32_t block_to_left_offset = 0;
  uint32_t block_to_right_offset = 0;
  if (thread_start_block_index < thread_end_block_index && thread_start_block_index > 1) {
    block_to_left_offset = block_to_left_offset_buffer[thread_start_block_index - 1];
    block_to_right_offset = block_to_right_offset_buffer[thread_start_block_index - 1];
  }
  block_to_left_offset = ShufflePrefixSum<uint32_t>(block_to_left_offset, shared_mem_buffer);
  __syncthreads();
  block_to_right_offset = ShufflePrefixSum<uint32_t>(block_to_right_offset, shared_mem_buffer);
  if (threadIdx_x == blockDim_x - 1) {
    to_left_total_count = block_to_left_offset + block_to_left_offset_buffer[num_blocks];
  }
  __syncthreads();
  const uint32_t to_left_thread_block_offset = block_to_left_offset;
  const uint32_t to_right_thread_block_offset = block_to_right_offset + to_left_total_count;
  for (uint32_t block_index = thread_start_block_index; block_index < thread_end_block_index; ++block_index) {
    block_to_left_offset_buffer[block_index] += to_left_thread_block_offset;
    block_to_right_offset_buffer[block_index] += to_right_thread_block_offset;
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const data_size_t old_leaf_data_end = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[left_leaf_index] = cuda_leaf_data_start[left_leaf_index] + static_cast<data_size_t>(to_left_total_count);
    cuda_leaf_num_data[left_leaf_index] = static_cast<data_size_t>(to_left_total_count);
    cuda_leaf_data_start[right_leaf_index] = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[right_leaf_index] = old_leaf_data_end;
    cuda_leaf_num_data[right_leaf_index] = num_data_in_leaf - static_cast<data_size_t>(to_left_total_count);
  }
}

__global__ void AggregateBlockOffsetKernel1(
  const int left_leaf_index,
  const int right_leaf_index,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer, data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end, data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices,
  const data_size_t num_blocks) {
  __shared__ uint32_t shared_mem_buffer[32];
  __shared__ uint32_t to_left_total_count;
  const data_size_t num_data_in_leaf = cuda_leaf_num_data[left_leaf_index];
  const unsigned int threadIdx_x = threadIdx.x;
  uint32_t block_to_left_offset = 0;
  uint32_t block_to_right_offset = 0;
  if (threadIdx_x < static_cast<unsigned int>(num_blocks)) {
    block_to_left_offset = block_to_left_offset_buffer[threadIdx_x + 1];
    block_to_right_offset = block_to_right_offset_buffer[threadIdx_x + 1];
  }
  block_to_left_offset = ShufflePrefixSum<uint32_t>(block_to_left_offset, shared_mem_buffer);
  __syncthreads();
  block_to_right_offset = ShufflePrefixSum<uint32_t>(block_to_right_offset, shared_mem_buffer);
  if (threadIdx.x == blockDim.x - 1) {
    to_left_total_count = block_to_left_offset;
  }
  __syncthreads();
  if (threadIdx_x < static_cast<unsigned int>(num_blocks)) {
    block_to_left_offset_buffer[threadIdx_x + 1] = block_to_left_offset;
    block_to_right_offset_buffer[threadIdx_x + 1] = block_to_right_offset + to_left_total_count;
  }
  if (threadIdx_x == 0) {
    block_to_right_offset_buffer[0] = to_left_total_count;
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const data_size_t old_leaf_data_end = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[left_leaf_index] = cuda_leaf_data_start[left_leaf_index] + static_cast<data_size_t>(to_left_total_count);
    cuda_leaf_num_data[left_leaf_index] = static_cast<data_size_t>(to_left_total_count);
    cuda_leaf_data_start[right_leaf_index] = cuda_leaf_data_end[left_leaf_index];
    cuda_leaf_data_end[right_leaf_index] = old_leaf_data_end;
    cuda_leaf_num_data[right_leaf_index] = num_data_in_leaf - static_cast<data_size_t>(to_left_total_count);
  }
}

template <typename BIN_TYPE>
__global__ void GenDataToLeftBitVectorKernelLaunch(
  // data
  const BIN_TYPE* column_data,
  const data_size_t* cuda_data_indices,
  data_size_t* cuda_leaf_data_start,
  const uint8_t* cuda_feature_missing_is_zero,
  const uint8_t* cuda_feature_missing_is_na,
  const uint8_t* cuda_feature_mfb_is_zero,
  const uint8_t* cuda_feature_mfb_is_na,
  const uint32_t* cuda_feature_default_bin,
  const uint32_t* cuda_feature_most_freq_bin,
  const uint32_t* cuda_feature_max_bin,
  const uint32_t* cuda_feature_min_bin,
  // input best split info
  const CUDASplitInfo* best_split_info,
  const int right_leaf_index,
  const uint32_t* categorical_bitset,
  const int categorical_bitset_len,
  // for leaf information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  // members of CUDADataPartition
  uint16_t* block_to_left_offset,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer,
  int* cuda_data_index_to_leaf_index,
  data_size_t* cuda_leaf_data_end,
  data_size_t* cuda_leaf_num_data) {
  const data_size_t num_data_in_leaf = best_split_info->left_count + best_split_info->right_count;
  const int left_leaf_index = best_split_info->leaf_index;
  const data_size_t data_start = cuda_leaf_data_start[left_leaf_index];
  const data_size_t* data_indices_in_leaf = cuda_data_indices + data_start;
  const int inner_feature_index = best_split_info->inner_feature_index;
  const bool missing_is_zero = static_cast<bool>(cuda_feature_missing_is_zero[inner_feature_index]);
  const bool missing_is_na = static_cast<bool>(cuda_feature_missing_is_na[inner_feature_index]);
  const bool mfb_is_zero = static_cast<bool>(cuda_feature_mfb_is_zero[inner_feature_index]);
  const bool mfb_is_na = static_cast<bool>(cuda_feature_mfb_is_na[inner_feature_index]);
  const uint32_t default_bin = cuda_feature_default_bin[inner_feature_index];
  const uint32_t most_freq_bin = cuda_feature_most_freq_bin[inner_feature_index];
  const uint32_t max_bin = cuda_feature_max_bin[inner_feature_index];
  const uint32_t min_bin = cuda_feature_min_bin[inner_feature_index];
  uint32_t th = best_split_info->threshold + min_bin;
  uint32_t t_zero_bin = min_bin + default_bin;
  if (most_freq_bin == 0) {
    --th;
    --t_zero_bin;
  }
  uint8_t split_default_to_left = 0;
  uint8_t split_missing_default_to_left = 0;
  int default_leaf_index = right_leaf_index;
  int missing_default_leaf_index = right_leaf_index;
  if (most_freq_bin <= best_split_info->threshold) {
    split_default_to_left = 1;
    default_leaf_index = left_leaf_index;
  }
  if (missing_is_zero || missing_is_na) {
    if (best_split_info->default_left) {
      split_missing_default_to_left = 1;
      missing_default_leaf_index = left_leaf_index;
    }
  }
  const bool max_bin_to_left = (max_bin <= th);
  const bool min_is_max = (min_bin == max_bin);

  cudaStream_t gen_bit_stream;
  cudaStream_t update_leaf_index_stream;
  cudaStreamCreateWithFlags(&gen_bit_stream, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&update_leaf_index_stream, cudaStreamNonBlocking);

  int grid_dim = 0;
  int block_dim = 0;
  CUDADataPartition::CalcBlockDim(num_data_in_leaf, &grid_dim, &block_dim);
  if (!min_is_max && !missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, false, false, false, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, false, false, false, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && !missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, false, false, false, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, false, false, false, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && !missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, false, false, false, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, false, false, false, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && !missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, false, false, false, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, false, false, false, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && !missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, false, false, true, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, false, false, true, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && !missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, false, false, true, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, false, false, true, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && !missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, false, false, true, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, false, false, true, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && !missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, false, false, true, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, false, false, true, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && !missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, false, true, false, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, false, true, false, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && !missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, false, true, false, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, false, true, false, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && !missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, false, true, false, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, false, true, false, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && !missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, false, true, false, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, false, true, false, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && !missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, false, true, true, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, false, true, true, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && !missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, false, true, true, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, false, true, true, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && !missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, false, true, true, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, false, true, true, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && !missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, false, true, true, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, false, true, true, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, true, false, false, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, true, false, false, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, true, false, false, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, true, false, false, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, true, false, false, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, true, false, false, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, true, false, false, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, true, false, false, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, true, false, true, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, true, false, true, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, true, false, true, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, true, false, true, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, true, false, true, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, true, false, true, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, true, false, true, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, true, false, true, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, true, true, false, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, true, true, false, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, true, true, false, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, true, true, false, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, true, true, false, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, true, true, false, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, true, true, false, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, true, true, false, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, true, true, true, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, true, true, true, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, true, true, true, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, true, true, true, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, true, true, true, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, true, true, true, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (!min_is_max && missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<false, true, true, true, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<false, true, true, true, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && !missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, false, false, false, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, false, false, false, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && !missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, false, false, false, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, false, false, false, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && !missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, false, false, false, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, false, false, false, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && !missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, false, false, false, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, false, false, false, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && !missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, false, false, true, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, false, false, true, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && !missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, false, false, true, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, false, false, true, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && !missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, false, false, true, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, false, false, true, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && !missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, false, false, true, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, false, false, true, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && !missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, false, true, false, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, false, true, false, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && !missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, false, true, false, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, false, true, false, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && !missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, false, true, false, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, false, true, false, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && !missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, false, true, false, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, false, true, false, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && !missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, false, true, true, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, false, true, true, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && !missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, false, true, true, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, false, true, true, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && !missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, false, true, true, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, false, true, true, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && !missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, false, true, true, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, false, true, true, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, true, false, false, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, true, false, false, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && missing_is_zero && !missing_is_na && !mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, true, false, false, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, true, false, false, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, true, false, false, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, true, false, false, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && missing_is_zero && !missing_is_na && !mfb_is_zero && mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, true, false, false, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, true, false, false, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, true, false, true, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, true, false, true, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && missing_is_zero && !missing_is_na && mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, true, false, true, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, true, false, true, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, true, false, true, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, true, false, true, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && missing_is_zero && !missing_is_na && mfb_is_zero && mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, true, false, true, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, true, false, true, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, true, true, false, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, true, true, false, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && missing_is_zero && missing_is_na && !mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, true, true, false, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, true, true, false, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, true, true, false, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, true, true, false, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && missing_is_zero && missing_is_na && !mfb_is_zero && mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, true, true, false, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, true, true, false, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, true, true, true, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, true, true, true, false, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && missing_is_zero && missing_is_na && mfb_is_zero && !mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, true, true, true, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, true, true, true, false, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && !max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, true, true, true, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, true, true, true, true, false, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  } else if (min_is_max && missing_is_zero && missing_is_na && mfb_is_zero && mfb_is_na && max_bin_to_left) {
    GenDataToLeftBitVectorKernel<true, true, true, true, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, gen_bit_stream>>>(
        GenBitVector_ARGS,
        block_to_left_offset,
        block_to_left_offset_buffer,
        block_to_right_offset_buffer);
    UpdateDataIndexToLeafIndexKernel<true, true, true, true, true, true, BIN_TYPE>
      <<<grid_dim, block_dim, 0, update_leaf_index_stream>>>(
        UpdateDataIndexToLeafIndex_ARGS,
        cuda_data_index_to_leaf_index);
  }

  int num_blocks_final_ref = grid_dim - 1;
  int num_blocks_final_aligned = 1;
  while (num_blocks_final_ref > 0) {
    num_blocks_final_aligned <<= 1;
    num_blocks_final_ref >>= 1;
  }
  if (grid_dim > AGGREGATE_BLOCK_SIZE_DATA_PARTITION) {
    AggregateBlockOffsetKernel0<<<1, AGGREGATE_BLOCK_SIZE_DATA_PARTITION, 0, gen_bit_stream>>>(
      left_leaf_index,
      right_leaf_index,
      block_to_left_offset_buffer,
      block_to_right_offset_buffer, cuda_leaf_data_start, cuda_leaf_data_end,
      cuda_leaf_num_data, cuda_data_indices,
      grid_dim);
  } else {
    AggregateBlockOffsetKernel1<<<1, num_blocks_final_aligned, 0, gen_bit_stream>>>(
      left_leaf_index,
      right_leaf_index,
      block_to_left_offset_buffer,
      block_to_right_offset_buffer, cuda_leaf_data_start, cuda_leaf_data_end,
      cuda_leaf_num_data, cuda_data_indices,
      grid_dim);
  }

  cudaStreamDestroy(gen_bit_stream);
  cudaStreamDestroy(update_leaf_index_stream);
}

__global__ void GenDataToLeftBitVectorKernelLaunchOuter(
  // data
  const int* cuda_feature_to_column,
  const uint8_t* cuda_column_bit_type,
  void* const* cuda_data_by_column,
  const data_size_t* cuda_data_indices,
  data_size_t* cuda_leaf_data_start,
  const uint8_t* cuda_feature_missing_is_zero,
  const uint8_t* cuda_feature_missing_is_na,
  const uint8_t* cuda_feature_mfb_is_zero,
  const uint8_t* cuda_feature_mfb_is_na,
  const uint32_t* cuda_feature_default_bin,
  const uint32_t* cuda_feature_most_freq_bin,
  const uint32_t* cuda_feature_max_bin,
  const uint32_t* cuda_feature_min_bin,
  // input best split info
  CUDASplitInfo* const* best_split_info_pointer,
  const int right_leaf_index,
  const uint32_t* categorical_bitset,
  const int categorical_bitset_len,
  // for leaf information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  // members of CUDADataPartition
  uint16_t* block_to_left_offset,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer,
  int* cuda_data_index_to_leaf_index,
  data_size_t* cuda_leaf_data_end,
  data_size_t* cuda_leaf_num_data) {
  const CUDASplitInfo* best_split_info = *best_split_info_pointer;
  const int inner_feature_index = best_split_info->inner_feature_index;
  const int column_index = cuda_feature_to_column[inner_feature_index];
  const uint8_t bit_type = cuda_column_bit_type[column_index];
  const void* column_data_pointer = cuda_data_by_column[column_index];
  cudaStream_t cuda_stream;
  cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);
  if (bit_type == 8) {
    const uint8_t* column_data = reinterpret_cast<const uint8_t*>(column_data_pointer);
    GenDataToLeftBitVectorKernelLaunch<uint8_t><<<1, 1, 0, cuda_stream>>>(
      column_data,
      cuda_data_indices,
      cuda_leaf_data_start,
      cuda_feature_missing_is_zero,
      cuda_feature_missing_is_na,
      cuda_feature_mfb_is_zero,
      cuda_feature_mfb_is_na,
      cuda_feature_default_bin,
      cuda_feature_most_freq_bin,
      cuda_feature_max_bin,
      cuda_feature_min_bin,
      best_split_info,
      right_leaf_index,
      categorical_bitset,
      categorical_bitset_len,
      smaller_leaf_splits,
      larger_leaf_splits,
      block_to_left_offset,
      block_to_left_offset_buffer,
      block_to_right_offset_buffer,
      cuda_data_index_to_leaf_index,
      cuda_leaf_data_end,
      cuda_leaf_num_data);
  } else if (bit_type == 16) {
    const uint16_t* column_data = reinterpret_cast<const uint16_t*>(column_data_pointer);
    GenDataToLeftBitVectorKernelLaunch<uint16_t><<<1, 1, 0, cuda_stream>>>(
      column_data,
      cuda_data_indices,
      cuda_leaf_data_start,
      cuda_feature_missing_is_zero,
      cuda_feature_missing_is_na,
      cuda_feature_mfb_is_zero,
      cuda_feature_mfb_is_na,
      cuda_feature_default_bin,
      cuda_feature_most_freq_bin,
      cuda_feature_max_bin,
      cuda_feature_min_bin,
      best_split_info,
      right_leaf_index,
      categorical_bitset,
      categorical_bitset_len,
      smaller_leaf_splits,
      larger_leaf_splits,
      block_to_left_offset,
      block_to_left_offset_buffer,
      block_to_right_offset_buffer,
      cuda_data_index_to_leaf_index,
      cuda_leaf_data_end,
      cuda_leaf_num_data);
  } else if (bit_type == 32) {
    const uint32_t* column_data = reinterpret_cast<const uint32_t*>(column_data_pointer);
    GenDataToLeftBitVectorKernelLaunch<uint32_t><<<1, 1, 0, cuda_stream>>>(
      column_data,
      cuda_data_indices,
      cuda_leaf_data_start,
      cuda_feature_missing_is_zero,
      cuda_feature_missing_is_na,
      cuda_feature_mfb_is_zero,
      cuda_feature_mfb_is_na,
      cuda_feature_default_bin,
      cuda_feature_most_freq_bin,
      cuda_feature_max_bin,
      cuda_feature_min_bin,
      best_split_info,
      right_leaf_index,
      categorical_bitset,
      categorical_bitset_len,
      smaller_leaf_splits,
      larger_leaf_splits,
      block_to_left_offset,
      block_to_left_offset_buffer,
      block_to_right_offset_buffer,
      cuda_data_index_to_leaf_index,
      cuda_leaf_data_end,
      cuda_leaf_num_data);
  }
  cudaStreamDestroy(cuda_stream);
}

void CUDADataPartition::LaunchGenDataToLeftBitVectorKernel(
  // input best split info
  CUDASplitInfo* const* best_split_info,
  const int right_leaf_index,
  const uint32_t* categorical_bitset,
  const int categorical_bitset_len,
  // for leaf information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits) {
  GenDataToLeftBitVectorKernelLaunchOuter<<<1, 1, 0, cuda_streams_[0]>>>(
    cuda_column_data_->cuda_feature_to_column(),
    cuda_column_data_->cuda_column_bit_type(),
    cuda_column_data_->cuda_data_by_column(),
    cuda_data_indices_,
    cuda_leaf_data_start_,
    cuda_column_data_->cuda_feature_missing_is_zero(),
    cuda_column_data_->cuda_feature_missing_is_na(),
    cuda_column_data_->cuda_feature_mfb_is_zero(),
    cuda_column_data_->cuda_feature_mfb_is_na(),
    cuda_column_data_->cuda_feature_default_bin(),
    cuda_column_data_->cuda_feature_most_freq_bin(),
    cuda_column_data_->cuda_feature_max_bin(),
    cuda_column_data_->cuda_feature_min_bin(),
    best_split_info,
    right_leaf_index,
    categorical_bitset,
    categorical_bitset_len,
    smaller_leaf_splits,
    larger_leaf_splits,
    cuda_block_to_left_offset_,
    cuda_block_data_to_left_offset_,
    cuda_block_data_to_right_offset_,
    cuda_data_index_to_leaf_index_,
    cuda_leaf_data_end_,
    cuda_leaf_num_data_);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

#undef UpdateDataIndexToLeafIndexKernel_PARAMS
#undef UpdateDataIndexToLeafIndex_ARGS
#undef GenDataToLeftBitVectorKernel_PARMS
#undef GenBitVector_ARGS

template <typename BIN_TYPE, bool USE_MIN_BIN>
__global__ void UpdateDataIndexToLeafIndexKernel_Categorical(
  const data_size_t num_data_in_leaf, const data_size_t* data_indices_in_leaf,
  const uint32_t* bitset, const int bitset_len, const BIN_TYPE* column_data,
  // values from feature
  const uint32_t max_bin, const uint32_t min_bin, const int8_t mfb_offset,
  int* cuda_data_index_to_leaf_index, const int left_leaf_index, const int right_leaf_index,
  const int default_leaf_index) {
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(column_data[global_data_index]);
    if (USE_MIN_BIN && (bin < min_bin || bin > max_bin)) {
      cuda_data_index_to_leaf_index[global_data_index] = default_leaf_index;
    } else if (!USE_MIN_BIN && bin == 0) {
      cuda_data_index_to_leaf_index[global_data_index] = default_leaf_index;
    } else if (CUDAFindInBitset(bitset, bitset_len, bin - min_bin + mfb_offset)) {
      cuda_data_index_to_leaf_index[global_data_index] = left_leaf_index;
    } else {
      cuda_data_index_to_leaf_index[global_data_index] = right_leaf_index;
    }
  }
}

// for categorical features
template <typename BIN_TYPE, bool USE_MIN_BIN>
__global__ void GenDataToLeftBitVectorKernel_Categorical(
  const data_size_t num_data_in_leaf, const data_size_t* data_indices_in_leaf,
  const uint32_t* bitset, int bitset_len, const BIN_TYPE* column_data,
  // values from feature
  const uint32_t max_bin, const uint32_t min_bin, const int8_t mfb_offset,
  const uint8_t split_default_to_left,
  uint16_t* block_to_left_offset,
  data_size_t* block_to_left_offset_buffer, data_size_t* block_to_right_offset_buffer) {
  __shared__ uint16_t shared_mem_buffer[32];
  uint16_t thread_to_left_offset_cnt = 0;
  const unsigned int local_data_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_data_index < num_data_in_leaf) {
    const unsigned int global_data_index = data_indices_in_leaf[local_data_index];
    const uint32_t bin = static_cast<uint32_t>(column_data[global_data_index]);
    if (USE_MIN_BIN && (bin < min_bin || bin > max_bin)) {
      thread_to_left_offset_cnt = split_default_to_left;
    } else if (!USE_MIN_BIN && bin == 0) {
      thread_to_left_offset_cnt = split_default_to_left;
    } else if (CUDAFindInBitset(bitset, bitset_len, bin - min_bin + mfb_offset)) {
      thread_to_left_offset_cnt = 1;
    }
  }
  __syncthreads();
  PrepareOffset(num_data_in_leaf, block_to_left_offset + blockIdx.x * blockDim.x, block_to_left_offset_buffer, block_to_right_offset_buffer,
    thread_to_left_offset_cnt, shared_mem_buffer);
}

#define GenBitVector_Categorical_ARGS \
  num_data_in_leaf, data_indices_in_leaf, \
  bitset, bitset_len, \
  column_data, max_bin, min_bin, mfb_offset, split_default_to_left, \
  cuda_block_to_left_offset_, cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_

#define UpdateDataIndexToLeafIndex_Categorical_ARGS \
  num_data_in_leaf, data_indices_in_leaf, \
  bitset, bitset_len, \
  column_data, max_bin, min_bin, mfb_offset,  \
  cuda_data_index_to_leaf_index_, left_leaf_index, right_leaf_index, default_leaf_index

void CUDADataPartition::LaunchGenDataToLeftBitVectorCategoricalKernel(
  const data_size_t num_data_in_leaf,
  const int split_feature_index,
  const uint32_t* bitset,
  const int bitset_len,
  const uint8_t split_default_left,
  const data_size_t leaf_data_start,
  const int left_leaf_index,
  const int right_leaf_index) {
  const data_size_t* data_indices_in_leaf = cuda_data_indices_ + leaf_data_start;
  const int column_index = cuda_column_data_->feature_to_column(split_feature_index);
  const uint8_t bit_type = cuda_column_data_->column_bit_type(column_index);
  const bool is_single_feature_in_column = is_single_feature_in_column_[split_feature_index];
  const uint32_t min_bin = is_single_feature_in_column ? 1 : cuda_column_data_->feature_min_bin(split_feature_index);
  const uint32_t max_bin = cuda_column_data_->feature_max_bin(split_feature_index);
  const uint32_t most_freq_bin = cuda_column_data_->feature_most_freq_bin(split_feature_index);
  const uint32_t default_bin = cuda_column_data_->feature_default_bin(split_feature_index);
  const void* column_data_pointer = cuda_column_data_->GetColumnData(column_index);
  const int8_t mfb_offset = static_cast<int8_t>(most_freq_bin == 0);
  std::vector<uint32_t> host_bitset(bitset_len, 0);
  CopyFromCUDADeviceToHost<uint32_t>(host_bitset.data(), bitset, bitset_len, __FILE__, __LINE__);
  uint8_t split_default_to_left = 0;
  int default_leaf_index = right_leaf_index;
  if (most_freq_bin > 0 && Common::FindInBitset(host_bitset.data(), bitset_len, most_freq_bin)) {
    split_default_to_left = 1;
    default_leaf_index = left_leaf_index;
  }
  if (bit_type == 8) {
    const uint8_t* column_data = reinterpret_cast<const uint8_t*>(column_data_pointer);
    if (is_single_feature_in_column) {
      GenDataToLeftBitVectorKernel_Categorical<uint8_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint8_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    } else {
      GenDataToLeftBitVectorKernel_Categorical<uint8_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint8_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    }
  } else if (bit_type == 16) {
    const uint16_t* column_data = reinterpret_cast<const uint16_t*>(column_data_pointer);
    if (is_single_feature_in_column) {
      GenDataToLeftBitVectorKernel_Categorical<uint16_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint16_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    } else {
      GenDataToLeftBitVectorKernel_Categorical<uint16_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint16_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    }
  } else if (bit_type == 32) {
    const uint32_t* column_data = reinterpret_cast<const uint32_t*>(column_data_pointer);
    if (is_single_feature_in_column) {
      GenDataToLeftBitVectorKernel_Categorical<uint32_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint32_t, false><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    } else {
      GenDataToLeftBitVectorKernel_Categorical<uint32_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[0]>>>(GenBitVector_Categorical_ARGS);
      UpdateDataIndexToLeafIndexKernel_Categorical<uint32_t, true><<<grid_dim_, block_dim_, 0, cuda_streams_[3]>>>(UpdateDataIndexToLeafIndex_Categorical_ARGS);
    }
  }
}

#undef GenBitVector_Categorical_ARGS
#undef UpdateDataIndexToLeafIndex_Categorical_ARGS


__global__ void SplitTreeStructureKernel(const int left_leaf_index,
  const int right_leaf_index,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer, data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end, data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices,
  const CUDASplitInfo* best_split_info,
  // for leaf splits information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  const int num_total_bin,
  hist_t* cuda_hist, hist_t** cuda_hist_pool,
  double* cuda_leaf_output) {
  const unsigned int to_left_total_cnt = cuda_leaf_num_data[left_leaf_index];
  const unsigned int global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_thread_index == 0) {
    cuda_leaf_output[left_leaf_index] = best_split_info->left_value;
  } else if (global_thread_index == 1) {
    cuda_leaf_output[right_leaf_index] = best_split_info->right_value;
  }

  if (cuda_leaf_num_data[left_leaf_index] < cuda_leaf_num_data[right_leaf_index]) {
    if (global_thread_index == 0) {
      hist_t* parent_hist_ptr = cuda_hist_pool[left_leaf_index];
      cuda_hist_pool[right_leaf_index] = parent_hist_ptr;
      cuda_hist_pool[left_leaf_index] = cuda_hist + 2 * right_leaf_index * num_total_bin;
      smaller_leaf_splits->hist_in_leaf = cuda_hist_pool[left_leaf_index];
      larger_leaf_splits->hist_in_leaf = cuda_hist_pool[right_leaf_index];
    } else if (global_thread_index == 1) {
      smaller_leaf_splits->sum_of_gradients = best_split_info->left_sum_gradients;
    } else if (global_thread_index == 2) {
      smaller_leaf_splits->sum_of_hessians = best_split_info->left_sum_hessians;
    } else if (global_thread_index == 3) {
      smaller_leaf_splits->num_data_in_leaf = to_left_total_cnt;
    } else if (global_thread_index == 4) {
      smaller_leaf_splits->gain = best_split_info->left_gain;
    } else if (global_thread_index == 5) {
      smaller_leaf_splits->leaf_value = best_split_info->left_value;
    } else if (global_thread_index == 6) {
      smaller_leaf_splits->data_indices_in_leaf = cuda_data_indices;
    } else if (global_thread_index == 7) {
      larger_leaf_splits->leaf_index = right_leaf_index;
    } else if (global_thread_index == 8) {
      larger_leaf_splits->sum_of_gradients = best_split_info->right_sum_gradients;
    } else if (global_thread_index == 9) {
      larger_leaf_splits->sum_of_hessians = best_split_info->right_sum_hessians;
    } else if (global_thread_index == 10) {
      larger_leaf_splits->num_data_in_leaf = cuda_leaf_num_data[right_leaf_index];
    } else if (global_thread_index == 11) {
      larger_leaf_splits->gain = best_split_info->right_gain;
    } else if (global_thread_index == 12) {
      larger_leaf_splits->leaf_value = best_split_info->right_value;
    } else if (global_thread_index == 13) {
      larger_leaf_splits->data_indices_in_leaf = cuda_data_indices + cuda_leaf_num_data[left_leaf_index];
    } else if (global_thread_index == 14) {
      smaller_leaf_splits->leaf_index = left_leaf_index;
    }
  } else {
    if (global_thread_index == 0) {
      larger_leaf_splits->leaf_index = left_leaf_index;
    } else if (global_thread_index == 1) {
      larger_leaf_splits->sum_of_gradients = best_split_info->left_sum_gradients;
    } else if (global_thread_index == 2) {
      larger_leaf_splits->sum_of_hessians = best_split_info->left_sum_hessians;
    } else if (global_thread_index == 3) {
      larger_leaf_splits->num_data_in_leaf = to_left_total_cnt;
    } else if (global_thread_index == 4) {
      larger_leaf_splits->gain = best_split_info->left_gain;
    } else if (global_thread_index == 5) {
      larger_leaf_splits->leaf_value = best_split_info->left_value;
    } else if (global_thread_index == 6) {
      larger_leaf_splits->data_indices_in_leaf = cuda_data_indices;
    } else if (global_thread_index == 7) {
      smaller_leaf_splits->leaf_index = right_leaf_index;
    } else if (global_thread_index == 8) {
      smaller_leaf_splits->sum_of_gradients = best_split_info->right_sum_gradients;
    } else if (global_thread_index == 9) {
      smaller_leaf_splits->sum_of_hessians = best_split_info->right_sum_hessians;
    } else if (global_thread_index == 10) {
      smaller_leaf_splits->num_data_in_leaf = cuda_leaf_num_data[right_leaf_index];
    } else if (global_thread_index == 11) {
      smaller_leaf_splits->gain = best_split_info->right_gain;
    } else if (global_thread_index == 12) {
      smaller_leaf_splits->leaf_value = best_split_info->right_value;
    } else if (global_thread_index == 13) {
      smaller_leaf_splits->data_indices_in_leaf = cuda_data_indices + cuda_leaf_num_data[left_leaf_index];
    } else if (global_thread_index == 14) {
      cuda_hist_pool[right_leaf_index] = cuda_hist + 2 * right_leaf_index * num_total_bin;
      smaller_leaf_splits->hist_in_leaf = cuda_hist_pool[right_leaf_index];
    } else if (global_thread_index == 15) {
      larger_leaf_splits->hist_in_leaf = cuda_hist_pool[left_leaf_index];
    }
  }
}

__global__ void SplitTreeStructureKernelLaunch(
  CUDASplitInfo* const* best_split_info_pointer,
  const int right_leaf_index,
  data_size_t* block_to_left_offset_buffer,
  data_size_t* block_to_right_offset_buffer, data_size_t* cuda_leaf_data_start,
  data_size_t* cuda_leaf_data_end, data_size_t* cuda_leaf_num_data, const data_size_t* cuda_data_indices,
  // for leaf splits information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  const int num_total_bin,
  hist_t* cuda_hist, hist_t** cuda_hist_pool,
  double* cuda_leaf_output) {
  const CUDASplitInfo* best_split_info = *best_split_info_pointer;
  const int left_leaf_index = best_split_info->leaf_index;
  cudaStream_t cuda_stream;
  cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);
  SplitTreeStructureKernel<<<4, 4, 0, cuda_stream>>>(left_leaf_index, right_leaf_index,
    block_to_left_offset_buffer,
    block_to_right_offset_buffer, cuda_leaf_data_start, cuda_leaf_data_end,
    cuda_leaf_num_data, cuda_data_indices,
    best_split_info,
    smaller_leaf_splits,
    larger_leaf_splits,
    num_total_bin,
    cuda_hist,
    cuda_hist_pool,
    cuda_leaf_output);
  cudaStreamDestroy(cuda_stream);
}

__global__ void SplitInnerKernel(const int left_leaf_index, const int right_leaf_index,
  const data_size_t* cuda_leaf_data_start, const data_size_t* cuda_leaf_num_data,
  const data_size_t* cuda_data_indices,
  const data_size_t* block_to_left_offset_buffer, const data_size_t* block_to_right_offset_buffer,
  const uint16_t* block_to_left_offset, data_size_t* out_data_indices_in_leaf) {
  const data_size_t leaf_num_data_offset = cuda_leaf_data_start[left_leaf_index];
  const data_size_t num_data_in_leaf = cuda_leaf_num_data[left_leaf_index] + cuda_leaf_num_data[right_leaf_index];
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  const unsigned int global_thread_index = blockIdx.x * blockDim_x + threadIdx_x;
  const data_size_t* cuda_data_indices_in_leaf = cuda_data_indices + leaf_num_data_offset;
  const uint16_t* block_to_left_offset_ptr = block_to_left_offset + blockIdx.x * blockDim_x;
  const uint32_t to_right_block_offset = block_to_right_offset_buffer[blockIdx.x];
  const uint32_t to_left_block_offset = block_to_left_offset_buffer[blockIdx.x];
  data_size_t* left_out_data_indices_in_leaf = out_data_indices_in_leaf + to_left_block_offset;
  data_size_t* right_out_data_indices_in_leaf = out_data_indices_in_leaf + to_right_block_offset;
  if (static_cast<data_size_t>(global_thread_index) < num_data_in_leaf) {
    const uint32_t thread_to_left_offset = (threadIdx_x == 0 ? 0 : block_to_left_offset_ptr[threadIdx_x - 1]);
    const bool to_left = block_to_left_offset_ptr[threadIdx_x] > thread_to_left_offset;
    if (to_left) {
      left_out_data_indices_in_leaf[thread_to_left_offset] = cuda_data_indices_in_leaf[global_thread_index];
    } else {
      const uint32_t thread_to_right_offset = threadIdx.x - thread_to_left_offset;
      right_out_data_indices_in_leaf[thread_to_right_offset] = cuda_data_indices_in_leaf[global_thread_index];
    }
  }
}

__global__ void SplitInnerKernelLaunch(
  CUDASplitInfo* const* best_split_info_pointer,
  const int right_leaf_index,
  const data_size_t* cuda_leaf_data_start, const data_size_t* cuda_leaf_num_data,
  const data_size_t* cuda_data_indices,
  const data_size_t* block_to_left_offset_buffer, const data_size_t* block_to_right_offset_buffer,
  const uint16_t* block_to_left_offset, data_size_t* out_data_indices_in_leaf) {
  const CUDASplitInfo* best_split_info = *best_split_info_pointer;
  const int left_leaf_index = best_split_info->leaf_index;
  const data_size_t num_data_in_leaf = cuda_leaf_num_data[left_leaf_index] + cuda_leaf_num_data[right_leaf_index];
  int grid_dim = 0;
  int block_dim = 0;
  cudaStream_t cuda_stream;
  cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);
  CUDADataPartition::CalcBlockDim(num_data_in_leaf, &grid_dim, &block_dim);
  SplitInnerKernel<<<grid_dim, block_dim, 0, cuda_stream>>>(
    left_leaf_index,
    right_leaf_index,
    cuda_leaf_data_start,
    cuda_leaf_num_data,
    cuda_data_indices,
    block_to_left_offset_buffer,
    block_to_right_offset_buffer,
    block_to_left_offset,
    out_data_indices_in_leaf);
  cudaStreamDestroy(cuda_stream);
}

__global__ void CopyDataIndicesKernel(
  const data_size_t num_data_in_leaf,
  const data_size_t* out_data_indices_in_leaf,
  data_size_t* cuda_data_indices) {
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int global_thread_index = blockIdx.x * blockDim.x + threadIdx_x;
  if (global_thread_index < num_data_in_leaf) {
    cuda_data_indices[global_thread_index] = out_data_indices_in_leaf[global_thread_index];
  }
}

__global__ void CopyDataIndicesKernelLaunch(
  CUDASplitInfo* const* best_split_info_pointer,
  const data_size_t* cuda_leaf_data_start,
  const data_size_t* out_data_indices_in_leaf,
  data_size_t* cuda_data_indices) {
  const CUDASplitInfo* best_split_info = *best_split_info_pointer;
  const int left_leaf_index = best_split_info->leaf_index;
  const data_size_t num_data_in_leaf = best_split_info->left_count + best_split_info->right_count;
  int grid_dim = 0;
  int block_dim = 0;
  cudaStream_t cuda_stream;
  cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);
  CUDADataPartition::CalcBlockDim(num_data_in_leaf, &grid_dim, &block_dim);
  CopyDataIndicesKernel<<<grid_dim, block_dim, 0, cuda_stream>>>(
    num_data_in_leaf,
    out_data_indices_in_leaf,
    cuda_data_indices + cuda_leaf_data_start[left_leaf_index]);
  cudaStreamDestroy(cuda_stream);
}

void CUDADataPartition::LaunchSplitInnerKernel(
  // input best split info
  CUDASplitInfo* const* best_split_info,
  const int right_leaf_index,
  const uint32_t* categorical_bitset,
  const int categorical_bitset_len,
  // for leaf information update
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits) {
  global_timer.Start("CUDADataPartition::SplitInnerKernel");
  SplitInnerKernelLaunch<<<1, 1, 0, cuda_streams_[1]>>>(
    best_split_info, right_leaf_index, cuda_leaf_data_start_, cuda_leaf_num_data_, cuda_data_indices_,
    cuda_block_data_to_left_offset_, cuda_block_data_to_right_offset_, cuda_block_to_left_offset_,
    cuda_out_data_indices_in_leaf_);
  global_timer.Stop("CUDADataPartition::SplitInnerKernel");

  // TODO: can remove this synchronization ?
  SynchronizeCUDADevice(__FILE__, __LINE__);

  global_timer.Start("CUDADataPartition::SplitTreeStructureKernel");
  SplitTreeStructureKernelLaunch<<<1, 1, 0, cuda_streams_[0]>>>(best_split_info, right_leaf_index,
    cuda_block_data_to_left_offset_,
    cuda_block_data_to_right_offset_, cuda_leaf_data_start_, cuda_leaf_data_end_,
    cuda_leaf_num_data_, cuda_out_data_indices_in_leaf_,
    smaller_leaf_splits,
    larger_leaf_splits,
    num_total_bin_,
    cuda_hist_,
    cuda_hist_pool_,
    cuda_leaf_output_);
  global_timer.Stop("CUDADataPartition::SplitTreeStructureKernel");
  SynchronizeCUDADevice(__FILE__, __LINE__);
  global_timer.Start("CUDADataPartition::CopyDataIndicesKernel");
  CopyDataIndicesKernelLaunch<<<1, 1, 0, cuda_streams_[2]>>>(
    best_split_info,
    cuda_leaf_data_start_,
    cuda_out_data_indices_in_leaf_,
    cuda_data_indices_);
  global_timer.Stop("CUDADataPartition::CopyDataIndicesKernel");
}

template <bool USE_BAGGING>
__global__ void AddPredictionToScoreKernel(
  const data_size_t* used_data_indices,
  const double* leaf_value, double* cuda_scores,
  const int* cuda_data_index_to_leaf_index, const data_size_t num_data) {
  const unsigned int threadIdx_x = threadIdx.x;
  const unsigned int blockIdx_x = blockIdx.x;
  const unsigned int blockDim_x = blockDim.x;
  const data_size_t local_data_index = static_cast<data_size_t>(blockIdx_x * blockDim_x + threadIdx_x);
  if (local_data_index < num_data) {
    if (USE_BAGGING) {
      const data_size_t global_data_index = used_data_indices[local_data_index];
      const int leaf_index = cuda_data_index_to_leaf_index[global_data_index];
      const double leaf_prediction_value = leaf_value[leaf_index];
      cuda_scores[global_data_index] += leaf_prediction_value;
    } else {
      const int leaf_index = cuda_data_index_to_leaf_index[local_data_index];
      const double leaf_prediction_value = leaf_value[leaf_index];
      cuda_scores[local_data_index] += leaf_prediction_value;
    }
  }
}

void CUDADataPartition::LaunchAddPredictionToScoreKernel(const double* leaf_value, double* cuda_scores) {
  global_timer.Start("CUDADataPartition::AddPredictionToScoreKernel");
  const data_size_t num_data_in_root = root_num_data();
  const int num_blocks = (num_data_in_root + FILL_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) / FILL_INDICES_BLOCK_SIZE_DATA_PARTITION;
  if (use_bagging_) {
    AddPredictionToScoreKernel<true><<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(
      cuda_data_indices_, leaf_value, cuda_scores, cuda_data_index_to_leaf_index_, num_data_in_root);
  } else {
    AddPredictionToScoreKernel<false><<<num_blocks, FILL_INDICES_BLOCK_SIZE_DATA_PARTITION>>>(
      cuda_data_indices_, leaf_value, cuda_scores, cuda_data_index_to_leaf_index_, num_data_in_root);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  global_timer.Stop("CUDADataPartition::AddPredictionToScoreKernel");
}

}  // namespace LightGBM

#endif  // USE_CUDA
