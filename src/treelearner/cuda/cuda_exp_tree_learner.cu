/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include "cuda_exp_tree_learner.hpp"

namespace LightGBM {

__global__ void ReduceRootNodeInformationKernel(
  const CUDALeafSplitsStruct* leaf_splits_buffer,
  const int num_gpu,
  const double lambda_l1,
  const double lambda_l2,
  const data_size_t num_data,
  CUDALeafSplitsStruct* out,
  double* out_sum_hessians) {
  double sum_of_gradients = 0.0;
  double sum_of_hessians = 0.0f;
  int64_t sum_of_gradients_hessians = 0;
  data_size_t num_data_in_leaf = 0;
  for (int gpu_index = 0; gpu_index < num_gpu; ++gpu_index) {
    const CUDALeafSplitsStruct* leaf_splits = leaf_splits_buffer + gpu_index;
    const double gpu_sum_of_gradients = leaf_splits->sum_of_gradients;
    const double gpu_sum_of_hessians = leaf_splits->sum_of_hessians;
    const int64_t gpu_sum_of_gradients_hessians = leaf_splits->sum_of_gradients_hessians;
    const data_size_t gpu_num_data_in_leaf = leaf_splits->num_data_in_leaf;
    sum_of_gradients += gpu_sum_of_gradients;
    sum_of_hessians += gpu_sum_of_hessians;
    sum_of_gradients_hessians += gpu_sum_of_gradients_hessians;
    num_data_in_leaf += gpu_num_data_in_leaf;
  }
  out->sum_of_gradients = sum_of_gradients;
  out->sum_of_hessians = sum_of_hessians;
  *out_sum_hessians = sum_of_hessians;
  out->sum_of_gradients_hessians = sum_of_gradients_hessians;
  out->num_data_in_leaf = num_data_in_leaf;
  assert(num_data_in_leaf == num_data);
  out->leaf_index = 0;
  const bool use_l1 = lambda_l1 > 0.0f;
  if (!use_l1) {
    // no smoothing on root node
    out->gain = CUDALeafSplits::GetLeafGain<false, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
  } else {
    // no smoothing on root node
    out->gain = CUDALeafSplits::GetLeafGain<true, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
  }
  if (!use_l1) {
    // no smoothing on root node
    out->leaf_value =
      CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
  } else {
    // no smoothing on root node
    out->leaf_value =
      CUDALeafSplits::CalculateSplittedLeafOutput<true, false>(sum_of_gradients, sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
  }
}

void CUDAExpTreeLearner::LaunchReduceRootNodeInformationKernel(CUDALeafSplitsStruct* out) {
  ReduceRootNodeInformationKernel<<<1, 1>>>(
    leaf_splits_buffer_[0].RawData(),
    config_->num_gpu,
    config_->lambda_l1,
    config_->lambda_l2,
    // TODO(shiyu1994): bagging is not supported by now
    train_data_->num_data(),
    out,
    cuda_root_sum_hessians_.RawData());
}

__global__ void ReduceBestSplitsForLeafKernel(
  int* leaf_best_split_info_buffer,
  const int num_gpu) {
  int best_gpu_for_smaller_leaf = -1;
  double best_gain = kMinScore;
  for (int gpu_index = 0; gpu_index < num_gpu; ++gpu_index) {
    const double gain = (reinterpret_cast<const double*>(leaf_best_split_info_buffer + 6))[0];
    if (gain > best_gain) {
      best_gain = gain;
      best_gpu_for_smaller_leaf = gpu_index;
    }
  }
  if (best_gpu_for_smaller_leaf >= 0) {
    const int* buffer = leaf_best_split_info_buffer + best_gpu_for_smaller_leaf * 10;
    leaf_best_split_info_buffer[0] = buffer[0];
    leaf_best_split_info_buffer[1] = buffer[1];
    leaf_best_split_info_buffer[2] = buffer[2];
    double* gain_buffer = reinterpret_cast<double*>(leaf_best_split_info_buffer + 6);
    gain_buffer[0] = (reinterpret_cast<const double*>(buffer + 6))[0];
  }
  
  int best_gpu_for_larger_leaf = -1;
  best_gain = kMinScore;
  for (int gpu_index = 0; gpu_index < num_gpu; ++gpu_index) {
    const double gain = (reinterpret_cast<const double*>(leaf_best_split_info_buffer + 6))[1];
    if (gain > best_gain) {
      best_gain = gain;
      best_gpu_for_larger_leaf = gpu_index;
    }
  }
  if (best_gpu_for_larger_leaf >= 0) {
    const int* buffer = leaf_best_split_info_buffer + best_gpu_for_larger_leaf * 10;
    leaf_best_split_info_buffer[3] = buffer[3];
    leaf_best_split_info_buffer[4] = buffer[4];
    leaf_best_split_info_buffer[5] = buffer[5];
    double* gain_buffer = reinterpret_cast<double*>(leaf_best_split_info_buffer + 6);
    gain_buffer[1] = (reinterpret_cast<const double*>(buffer + 6))[1];
  }
  leaf_best_split_info_buffer[10] = best_gpu_for_smaller_leaf;
  leaf_best_split_info_buffer[11] = best_gpu_for_larger_leaf;
}

void CUDAExpTreeLearner::LaunchReduceBestSplitsForLeafKernel() {
  ReduceBestSplitsForLeafKernel<<<1, 1>>>(
    best_split_info_buffer_.RawData(),
    config_->num_gpu);
}

}  // namespace LightGBM
