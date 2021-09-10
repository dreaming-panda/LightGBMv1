/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_CUDA_ROW_DATA_HPP_
#define LIGHTGBM_CUDA_ROW_DATA_HPP_

#include <LightGBM/config.h>
#include <LightGBM/cuda/cuda_utils.h>
#include <LightGBM/dataset.h>
#include <LightGBM/bin.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include "../train_share_states.h"

#define SHRAE_HIST_SIZE (6144 * 2)

namespace LightGBM {

class CUDARowData {
 public:
  CUDARowData(const Dataset* train_data,
              const TrainingShareStates* train_share_state, const int gpu_device_id);

  void Init(const Dataset* train_data,
            TrainingShareStates* train_share_state);

  int num_feature_partitions() const { return num_feature_partitions_; }

  int max_num_column_per_partition() const { return max_num_column_per_partition_; }

  bool is_sparse() const { return is_sparse_; }

  uint8_t bit_type() const { return bit_type_; }

  uint8_t row_ptr_bit_type() const { return row_ptr_bit_type_; }

  const uint8_t* cuda_data_uint8() const { return cuda_data_uint8_t_; }

  const uint16_t* cuda_data_uint16() const { return cuda_data_uint16_t_; }

  const uint32_t* cuda_data_uint32() const { return cuda_data_uint32_t_; }

  const uint16_t* cuda_row_ptr_uint16() const { return cuda_row_ptr_uint16_t_; }

  const uint32_t* cuda_row_ptr_uint32() const { return cuda_row_ptr_uint32_t_; }

  const uint64_t* cuda_row_ptr_uint64() const { return cuda_row_ptr_uint64_t_; }

  const uint16_t* cuda_partition_ptr_uint16() const { return cuda_partition_ptr_uint16_t_; }

  const uint32_t* cuda_partition_ptr_uint32() const { return cuda_partition_ptr_uint32_t_; }

  const uint64_t* cuda_partition_ptr_uint64() const { return cuda_partition_ptr_uint64_t_; }

  const int* cuda_feature_partition_column_index_offsets() const { return cuda_feature_partition_column_index_offsets_; }

  const uint32_t* cuda_column_hist_offsets() const { return cuda_column_hist_offsets_; }

  const uint32_t* cuda_partition_hist_offsets() const { return cuda_partition_hist_offsets_; }

 private:
  void DivideCUDAFeatureGroups(const Dataset* train_data, TrainingShareStates* share_state);

  template <typename BIN_TYPE>
  void GetDenseDataPartitioned(const BIN_TYPE* row_wise_data, std::vector<BIN_TYPE>* partitioned_data);

  template <typename BIN_TYPE, typename ROW_PTR_TYPE>
  void GetSparseDataPartitioned(const BIN_TYPE* row_wise_data,
    const ROW_PTR_TYPE* row_ptr,
    std::vector<std::vector<BIN_TYPE>>* partitioned_data,
    std::vector<std::vector<ROW_PTR_TYPE>>* partitioned_row_ptr,
    std::vector<ROW_PTR_TYPE>* partition_ptr);

  template <typename BIN_TYPE, typename ROW_PTR_TYPE>
  void InitSparseData(const BIN_TYPE* host_data,
                      const ROW_PTR_TYPE* host_row_ptr,
                      BIN_TYPE** cuda_data,
                      ROW_PTR_TYPE** cuda_row_ptr,
                      ROW_PTR_TYPE** cuda_partition_ptr);

  /*! \brief number of threads to use */
  int num_threads_;
  /*! \brief number of training data */
  data_size_t num_data_;
  /*! \brief number of bins of all features */
  int num_total_bin_;
  /*! \brief number of feature groups in dataset */
  int num_feature_group_;
  /*! \brief number of features in dataset */
  int num_feature_;
  /*! \brief number of bits used to store each bin value */
  uint8_t bit_type_;
  /*! \brief number of bits used to store each row pointer value */
  uint8_t row_ptr_bit_type_;
  /*! \brief is sparse row wise data */
  bool is_sparse_;
  /*! \brief start column index of each feature partition */
  std::vector<int> feature_partition_column_index_offsets_;
  /*! \brief histogram offset of each column */
  std::vector<uint32_t> column_hist_offsets_;
  /*! \brief hisotgram offset of each partition */
  std::vector<uint32_t> partition_hist_offsets_;
  /*! \brief maximum number of columns among all feature partitions */
  int max_num_column_per_partition_;
  /*! \brief number of partitions */
  int num_feature_partitions_;

  // CUDA memory

  /*! \brief row-wise data stored in CUDA, 8 bits */
  uint8_t* cuda_data_uint8_t_;
  /*! \brief row-wise data stored in CUDA, 16 bits */
  uint16_t* cuda_data_uint16_t_;
  /*! \brief row-wise data stored in CUDA, 32 bits */
  uint32_t* cuda_data_uint32_t_;
  /*! \brief row pointer stored in CUDA, 16 bits */
  uint16_t* cuda_row_ptr_uint16_t_;
  /*! \brief row pointer stored in CUDA, 32 bits */
  uint32_t* cuda_row_ptr_uint32_t_;
  /*! \brief row pointer stored in CUDA, 64 bits */
  uint64_t* cuda_row_ptr_uint64_t_;
  /*! \brief partition bin offsets, 16 bits */
  uint16_t* cuda_partition_ptr_uint16_t_;
  /*! \brief partition bin offsets, 32 bits */
  uint32_t* cuda_partition_ptr_uint32_t_;
  /*! \brief partition bin offsets, 64 bits */
  uint64_t* cuda_partition_ptr_uint64_t_;
  /*! \brief start column index of each feature partition */
  int* cuda_feature_partition_column_index_offsets_;
  /*! \brief histogram offset of each column */
  uint32_t* cuda_column_hist_offsets_;
  /*! \brief hisotgram offset of each partition */
  uint32_t* cuda_partition_hist_offsets_;
};

}  // namespace LightGBM
#endif  // LIGHTGBM_CUDA_COLUMN_DATA_HPP_