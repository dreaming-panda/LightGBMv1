/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_IO_CUDA_CUDA_ROW_DATA_HPP_
#define LIGHTGBM_IO_CUDA_CUDA_ROW_DATA_HPP_

#include <LightGBM/bin.h>
#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/dataset.h>
#include <LightGBM/train_share_states.h>

namespace LightGBM {

class CUDARowData {
 public:
  CUDARowData(const Config* config);

  ~CUDARowData();

  void Init(const Dataset* train_data, const TrainingShareStates* train_share_states);

 private:
  void DivideCUDAFeatureGroups(const Dataset* train_data, const TrainingShareStates* share_state);

  template <typename BIN_TYPE>
  void GetDenseDataPartitioned(const BIN_TYPE* row_wise_data, std::vector<BIN_TYPE>* partitioned_data);

  template <typename BIN_TYPE, typename ROW_PTR_TYPE>
  void GetSparseDataPartitioned(const BIN_TYPE* row_wise_data,
    const ROW_PTR_TYPE* row_ptr,
    std::vector<std::vector<BIN_TYPE>>* partitioned_data,
    std::vector<std::vector<ROW_PTR_TYPE>>* partitioned_row_ptr,
    std::vector<ROW_PTR_TYPE>* partition_ptr);

  template <typename BIN_TYPE, typename ROW_PTR_TYPE>
  void InitCUDADataInner(const size_t total_size, const BIN_TYPE* host_data, const ROW_PTR_TYPE* host_row_ptr);

  void InitCUDAData(const TrainingShareStates* train_share_states);

  int num_threads_;
  int num_features_;
  data_size_t num_data_;

  int num_feature_partitions_;
  int max_num_bin_per_partition_;
  int max_num_column_per_partition_;

  int* cuda_feature_partition_column_index_offsets_;
  uint32_t* cuda_column_hist_offsets_;
  uint32_t* cuda_column_hist_offsets_full_;

  bool is_row_wise_sparse_;
  int8_t cuda_data_by_row_bit_type_;
  int8_t cuda_row_ptr_bit_type_;
  void* cuda_data_by_row_;
  void* cuda_row_ptr_;
  void* cuda_partition_ptr_;

  std::vector<int> feature_partition_column_index_offsets_;
  std::vector<uint32_t> column_hist_offsets_;
  std::vector<uint32_t> column_hist_offsets_full_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_IO_CUDA_CUDA_ROW_DATA_HPP_
