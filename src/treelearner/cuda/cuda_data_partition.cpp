/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_data_partition.hpp"

namespace LightGBM {

CUDADataPartition::CUDADataPartition(const data_size_t num_data, const int num_features, const int num_leaves,
  const int num_threads, const data_size_t* cuda_num_data, const int* cuda_num_leaves, const uint8_t* cuda_data,
  const int* cuda_num_features, const std::vector<uint32_t>& feature_hist_offsets, const Dataset* train_data,
  hist_t* cuda_hist):
  num_data_(num_data), num_features_(num_features), num_leaves_(num_leaves), num_threads_(num_threads),
  num_total_bin_(feature_hist_offsets.back()), cuda_data_(cuda_data), cuda_num_features_(cuda_num_features),
  cuda_hist_(cuda_hist) {
  cuda_num_data_ = cuda_num_data;
  cuda_num_leaves_ = cuda_num_leaves;
  max_num_split_indices_blocks_ = (num_data_ + SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION - 1) /
    SPLIT_INDICES_BLOCK_SIZE_DATA_PARTITION;
  cur_num_leaves_ = 1;
  feature_default_bins_.resize(train_data->num_features());
  feature_most_freq_bins_.resize(train_data->num_features());
  feature_max_bins_.resize(train_data->num_features());
  feature_min_bins_.resize(train_data->num_features());
  feature_missing_is_zero_.resize(train_data->num_features());
  feature_missing_is_na_.resize(train_data->num_features());
  feature_mfb_is_zero_.resize(train_data->num_features());
  feature_mfb_is_na_.resize(train_data->num_features());
  int cur_group = 0;
  uint32_t prev_group_bins = 0;
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    const int group = train_data->Feature2Group(feature_index);
    if (cur_group != group) {
      prev_group_bins += static_cast<uint32_t>(train_data->FeatureGroupNumBin(cur_group));
      cur_group = group;
    }
    const BinMapper* bin_mapper = train_data->FeatureBinMapper(feature_index);
    feature_default_bins_[feature_index] = bin_mapper->GetDefaultBin();
    feature_most_freq_bins_[feature_index] = bin_mapper->GetMostFreqBin();
    /*Log::Warning("feature_index = %d, feature_hist_offsets[feature_index] = %d, prev_group_bins = %d",
      feature_index, feature_hist_offsets[feature_index], prev_group_bins);*/
    feature_min_bins_[feature_index] = feature_hist_offsets[feature_index] - prev_group_bins;
    feature_max_bins_[feature_index] = feature_hist_offsets[feature_index + 1] - prev_group_bins - 1;
    const MissingType missing_type = bin_mapper->missing_type();
    if (missing_type == MissingType::None) {
      feature_missing_is_zero_[feature_index] = 0;
      feature_missing_is_na_[feature_index] = 0;
      feature_mfb_is_zero_[feature_index] = 0;
      feature_mfb_is_na_[feature_index] = 0;
    } else if (missing_type == MissingType::Zero) {
      feature_missing_is_zero_[feature_index] = 1;
      feature_missing_is_na_[feature_index] = 0;
      if (bin_mapper->GetMostFreqBin() == bin_mapper->GetDefaultBin()) {
        feature_mfb_is_zero_[feature_index] = 1;
      } else {
        feature_mfb_is_zero_[feature_index] = 0;
      }
      feature_mfb_is_na_[feature_index] = 0;
    } else if (missing_type == MissingType::NaN) {
      feature_missing_is_zero_[feature_index] = 0;
      feature_missing_is_na_[feature_index] = 1;
      feature_mfb_is_zero_[feature_index] = 0;
      if (bin_mapper->GetMostFreqBin() == bin_mapper->GetDefaultBin()) {
        feature_mfb_is_na_[feature_index] = 1;
      } else {
        feature_mfb_is_na_[feature_index] = 0;
      }
    }
  }
  num_data_in_leaf_.resize(num_leaves_, 0);
  num_data_in_leaf_[0] = num_data_;
}

void CUDADataPartition::Init() {
  // allocate CUDA memory
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_data_), &cuda_data_indices_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_leaves_), &cuda_leaf_data_start_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_leaves_), &cuda_leaf_data_end_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_leaves_), &cuda_leaf_num_data_);
  InitCUDAValueFromConstant<int>(&cuda_num_total_bin_, num_total_bin_);
  InitCUDAValueFromConstant<int>(&cuda_cur_num_leaves_, 1);
  AllocateCUDAMemory<uint8_t>(static_cast<size_t>(num_data_), &cuda_data_to_left_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(max_num_split_indices_blocks_), &cuda_block_data_to_left_offset_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(max_num_split_indices_blocks_), &cuda_block_data_to_right_offset_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_data_), &cuda_out_data_indices_in_leaf_);
  AllocateCUDAMemory<hist_t*>(static_cast<size_t>(num_leaves_), &cuda_hist_pool_);
  CopyFromHostToCUDADevice<hist_t*>(cuda_hist_pool_, &cuda_hist_, 1);

  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_most_freq_bins_, feature_most_freq_bins_.data(), static_cast<size_t>(num_features_));
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_default_bins_, feature_default_bins_.data(), static_cast<size_t>(num_features_));
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_max_bins_, feature_max_bins_.data(), static_cast<size_t>(num_features_));
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_feature_min_bins_, feature_min_bins_.data(), static_cast<size_t>(num_features_));
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_feature_missing_is_zero_, feature_missing_is_zero_.data(), static_cast<size_t>(num_features_));
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_feature_missing_is_na_, feature_missing_is_na_.data(), static_cast<size_t>(num_features_));
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_feature_mfb_is_zero_, feature_mfb_is_zero_.data(), static_cast<size_t>(num_features_));
  InitCUDAMemoryFromHostMemory<uint8_t>(&cuda_feature_mfb_is_na_, feature_mfb_is_na_.data(), static_cast<size_t>(num_features_));
  AllocateCUDAMemory<int>(12, &cuda_split_info_buffer_);

  AllocateCUDAMemory<int>(static_cast<size_t>(num_leaves_), &tree_split_leaf_index_);
  AllocateCUDAMemory<int>(static_cast<size_t>(num_leaves_), &tree_inner_feature_index_);
  AllocateCUDAMemory<uint32_t>(static_cast<size_t>(num_leaves_), &tree_threshold_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &tree_left_output_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &tree_right_output_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_leaves_), &tree_left_count_);
  AllocateCUDAMemory<data_size_t>(static_cast<size_t>(num_leaves_), &tree_right_count_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &tree_left_sum_hessian_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &tree_right_sum_hessian_);
  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &tree_gain_);
  AllocateCUDAMemory<uint8_t>(static_cast<size_t>(num_leaves_), &tree_default_left_);

  AllocateCUDAMemory<double>(static_cast<size_t>(num_leaves_), &data_partition_leaf_output_);

  AllocateCUDAMemory<double>(static_cast<size_t>(num_data_), &train_data_score_tmp_);

  AllocateCUDAMemory<uint8_t>(static_cast<size_t>(num_data_) * static_cast<size_t>(num_features_), &cuda_data_col_wise_);

  CopyColWiseData();

  cpu_train_data_score_tmp_.resize(num_data_, 0.0f);
  cpu_split_info_buffer_.resize(6, 0);

  cuda_streams_.resize(5);
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[0]));
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[1]));
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[2]));
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[3]));
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[4]));

  const size_t max_num_blocks_in_debug = static_cast<size_t>((num_data_ + 1023) / 1024);
  AllocateCUDAMemory<double>(max_num_blocks_in_debug, &cuda_gradients_sum_buffer_);
  AllocateCUDAMemory<double>(max_num_blocks_in_debug, &cuda_hessians_sum_buffer_);
}

void CUDADataPartition::CopyColWiseData() {
  LaunchCopyColWiseDataKernel();
}

void CUDADataPartition::BeforeTrain(const data_size_t* data_indices) {
  if (data_indices == nullptr) {
    // no bagging
    LaunchFillDataIndicesBeforeTrain();
    SetCUDAMemory<data_size_t>(cuda_leaf_num_data_, 0, static_cast<size_t>(num_leaves_));
    SetCUDAMemory<data_size_t>(cuda_leaf_data_start_, 0, static_cast<size_t>(num_leaves_));
    SetCUDAMemory<data_size_t>(cuda_leaf_data_end_, 0, static_cast<size_t>(num_leaves_));
    SynchronizeCUDADevice();
    CopyFromCUDADeviceToCUDADevice<data_size_t>(cuda_leaf_num_data_, cuda_num_data_, 1);
    CopyFromCUDADeviceToCUDADevice<data_size_t>(cuda_leaf_data_end_, cuda_num_data_, 1);
    SynchronizeCUDADevice();
    cur_num_leaves_ = 1;
    CopyFromHostToCUDADevice<int>(cuda_cur_num_leaves_, &cur_num_leaves_, 1);
    num_data_in_leaf_.clear();
    num_data_in_leaf_.resize(num_leaves_, 0);
    num_data_in_leaf_[0] = num_data_;
    CopyFromHostToCUDADevice<hist_t*>(cuda_hist_pool_, &cuda_hist_, 1);
  } else {
    Log::Fatal("bagging is not supported by GPU");
  }
}

void CUDADataPartition::Split(const int* leaf_id,
  const double* best_split_gain,
  const int* best_split_feature,
  const uint32_t* best_split_threshold,
  const uint8_t* best_split_default_left,
  const double* best_left_sum_gradients, const double* best_left_sum_hessians, const data_size_t* best_left_count,
  const double* best_left_gain, const double* best_left_leaf_value,
  const double* best_right_sum_gradients, const double* best_right_sum_hessians, const data_size_t* best_right_count,
  const double* best_right_gain, const double* best_right_leaf_value,
  uint8_t* best_split_found,
  // for leaf splits information update
  int* smaller_leaf_cuda_leaf_index_pointer, double* smaller_leaf_cuda_sum_of_gradients_pointer,
  double* smaller_leaf_cuda_sum_of_hessians_pointer, data_size_t* smaller_leaf_cuda_num_data_in_leaf_pointer,
  double* smaller_leaf_cuda_gain_pointer, double* smaller_leaf_cuda_leaf_value_pointer,
  const data_size_t** smaller_leaf_cuda_data_indices_in_leaf_pointer_pointer,
  hist_t** smaller_leaf_cuda_hist_pointer_pointer,
  int* larger_leaf_cuda_leaf_index_pointer, double* larger_leaf_cuda_sum_of_gradients_pointer,
  double* larger_leaf_cuda_sum_of_hessians_pointer, data_size_t* larger_leaf_cuda_num_data_in_leaf_pointer,
  double* larger_leaf_cuda_gain_pointer, double* larger_leaf_cuda_leaf_value_pointer,
  const data_size_t** larger_leaf_cuda_data_indices_in_leaf_pointer_pointer,
  hist_t** larger_leaf_cuda_hist_pointer_pointer,
  std::vector<data_size_t>* cpu_leaf_num_data,
  std::vector<data_size_t>* cpu_leaf_data_start,
  std::vector<double>* cpu_leaf_sum_hessians,
  const std::vector<int>& cpu_leaf_best_split_feature,
  const std::vector<uint32_t>& cpu_leaf_best_split_threshold,
  const std::vector<uint8_t>& cpu_leaf_best_split_default_left,
  int* smaller_leaf_index, int* larger_leaf_index,
  const int cpu_leaf_index) {
  global_timer.Start("GenDataToLeftBitVector");
  global_timer.Start("SplitInner Copy CUDA To Host");
  const data_size_t num_data_in_leaf = cpu_leaf_num_data->at(cpu_leaf_index);
  const int split_feature_index = cpu_leaf_best_split_feature[cpu_leaf_index];
  const uint32_t split_threshold = cpu_leaf_best_split_threshold[cpu_leaf_index];
  const uint8_t split_default_left = cpu_leaf_best_split_default_left[cpu_leaf_index];
  const data_size_t leaf_data_start = cpu_leaf_data_start->at(cpu_leaf_index);
  global_timer.Stop("SplitInner Copy CUDA To Host");
  auto start = std::chrono::steady_clock::now();
  //GenDataToLeftBitVector(leaf_id, cpu_num_data_in_leaf, best_split_feature, best_split_threshold, best_split_default_left);
  GenDataToLeftBitVector2(num_data_in_leaf, split_feature_index, split_threshold, split_default_left, leaf_data_start);
  auto end = std::chrono::steady_clock::now();
  double duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  global_timer.Stop("GenDataToLeftBitVector");
  //Log::Warning("CUDADataPartition::GenDataToLeftBitVector time %f", duration);
  global_timer.Start("SplitInner");

  start = std::chrono::steady_clock::now();
  SplitInner(leaf_id, num_data_in_leaf,
    best_split_feature, best_split_threshold, best_split_default_left, best_split_gain,
    best_left_sum_gradients, best_left_sum_hessians, best_left_count,
    best_left_gain, best_left_leaf_value,
    best_right_sum_gradients, best_right_sum_hessians, best_right_count,
    best_right_gain, best_right_leaf_value, best_split_found,
    smaller_leaf_cuda_leaf_index_pointer, smaller_leaf_cuda_sum_of_gradients_pointer,
    smaller_leaf_cuda_sum_of_hessians_pointer, smaller_leaf_cuda_num_data_in_leaf_pointer,
    smaller_leaf_cuda_gain_pointer, smaller_leaf_cuda_leaf_value_pointer,
    smaller_leaf_cuda_data_indices_in_leaf_pointer_pointer,
    smaller_leaf_cuda_hist_pointer_pointer,
    larger_leaf_cuda_leaf_index_pointer, larger_leaf_cuda_sum_of_gradients_pointer,
    larger_leaf_cuda_sum_of_hessians_pointer, larger_leaf_cuda_num_data_in_leaf_pointer,
    larger_leaf_cuda_gain_pointer, larger_leaf_cuda_leaf_value_pointer,
    larger_leaf_cuda_data_indices_in_leaf_pointer_pointer,
    larger_leaf_cuda_hist_pointer_pointer, cpu_leaf_num_data, cpu_leaf_data_start, cpu_leaf_sum_hessians,
    smaller_leaf_index, larger_leaf_index);
  end = std::chrono::steady_clock::now();
  duration = (static_cast<std::chrono::duration<double>>(end - start)).count();
  global_timer.Stop("SplitInner");
  //Log::Warning("CUDADataPartition::SplitInner time %f", duration);
}

void CUDADataPartition::GenDataToLeftBitVector(const int* leaf_id,
  const data_size_t num_data_in_leaf,
  const int* best_split_feature,
  const uint32_t* best_split_threshold,
  const uint8_t* best_split_default_left) {
  LaunchGenDataToLeftBitVectorKernel(leaf_id, num_data_in_leaf, best_split_feature, best_split_threshold, best_split_default_left);
}

void CUDADataPartition::GenDataToLeftBitVector2(const data_size_t num_data_in_leaf,
    const int split_feature_index, const uint32_t split_threshold,
    const uint8_t split_default_left, const data_size_t leaf_data_start) {
  LaunchGenDataToLeftBitVectorKernel2(num_data_in_leaf, split_feature_index, split_threshold, split_default_left, leaf_data_start);
}

void CUDADataPartition::SplitInner(const int* leaf_index, const data_size_t num_data_in_leaf,
  const int* best_split_feature, const uint32_t* best_split_threshold,
  const uint8_t* best_split_default_left, const double* best_split_gain,
  const double* best_left_sum_gradients, const double* best_left_sum_hessians, const data_size_t* best_left_count,
  const double* best_left_gain, const double* best_left_leaf_value,
  const double* best_right_sum_gradients, const double* best_right_sum_hessians, const data_size_t* best_right_count,
  const double* best_right_gain, const double* best_right_leaf_value, uint8_t* best_split_found,
  // for leaf splits information update
  int* smaller_leaf_cuda_leaf_index_pointer, double* smaller_leaf_cuda_sum_of_gradients_pointer,
  double* smaller_leaf_cuda_sum_of_hessians_pointer, data_size_t* smaller_leaf_cuda_num_data_in_leaf_pointer,
  double* smaller_leaf_cuda_gain_pointer, double* smaller_leaf_cuda_leaf_value_pointer,
  const data_size_t** smaller_leaf_cuda_data_indices_in_leaf_pointer_pointer,
  hist_t** smaller_leaf_cuda_hist_pointer_pointer,
  int* larger_leaf_cuda_leaf_index_pointer, double* larger_leaf_cuda_sum_of_gradients_pointer,
  double* larger_leaf_cuda_sum_of_hessians_pointer, data_size_t* larger_leaf_cuda_num_data_in_leaf_pointer,
  double* larger_leaf_cuda_gain_pointer, double* larger_leaf_cuda_leaf_value_pointer,
  const data_size_t** larger_leaf_cuda_data_indices_in_leaf_pointer_pointer,
  hist_t** larger_leaf_cuda_hist_pointer_pointer,
  std::vector<data_size_t>* cpu_leaf_num_data, std::vector<data_size_t>* cpu_leaf_data_start,
  std::vector<double>* cpu_leaf_sum_hessians,
  int* smaller_leaf_index, int* larger_leaf_index) {
  LaunchSplitInnerKernel(leaf_index, num_data_in_leaf,
    best_split_feature, best_split_threshold, best_split_default_left, best_split_gain,
    best_left_sum_gradients, best_left_sum_hessians, best_left_count,
    best_left_gain, best_left_leaf_value,
    best_right_sum_gradients, best_right_sum_hessians, best_right_count,
    best_right_gain, best_right_leaf_value, best_split_found,
    smaller_leaf_cuda_leaf_index_pointer, smaller_leaf_cuda_sum_of_gradients_pointer,
    smaller_leaf_cuda_sum_of_hessians_pointer, smaller_leaf_cuda_num_data_in_leaf_pointer,
    smaller_leaf_cuda_gain_pointer, smaller_leaf_cuda_leaf_value_pointer,
    smaller_leaf_cuda_data_indices_in_leaf_pointer_pointer,
    smaller_leaf_cuda_hist_pointer_pointer,
    larger_leaf_cuda_leaf_index_pointer, larger_leaf_cuda_sum_of_gradients_pointer,
    larger_leaf_cuda_sum_of_hessians_pointer, larger_leaf_cuda_num_data_in_leaf_pointer,
    larger_leaf_cuda_gain_pointer, larger_leaf_cuda_leaf_value_pointer,
    larger_leaf_cuda_data_indices_in_leaf_pointer_pointer,
    larger_leaf_cuda_hist_pointer_pointer, cpu_leaf_num_data, cpu_leaf_data_start, cpu_leaf_sum_hessians,
    smaller_leaf_index, larger_leaf_index);
  ++cur_num_leaves_;
}

Tree* CUDADataPartition::GetCPUTree() {}

void CUDADataPartition::UpdateTrainScore(const double learning_rate, double* train_score, double* cuda_scores) {
  LaunchAddPredictionToScoreKernel(learning_rate, cuda_scores);
  /*#pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (data_size_t i = 0; i < num_data_; ++i) {
    train_score[i] += cpu_train_data_score_tmp_[i];
  }*/
}

void CUDADataPartition::CUDACheck(
    const int smaller_leaf_index,
    const int larger_leaf_index,
    const std::vector<data_size_t>& num_data_in_leaf,
    const CUDALeafSplits* smaller_leaf_splits,
    const CUDALeafSplits* larger_leaf_splits,
    const score_t* gradients,
    const score_t* hessians) {
  LaunchCUDACheckKernel(smaller_leaf_index, larger_leaf_index, num_data_in_leaf, smaller_leaf_splits, larger_leaf_splits, gradients, hessians);
}

}  // namespace LightGBM

#endif  // USE_CUDA