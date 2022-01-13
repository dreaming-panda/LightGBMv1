/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include "cuda_exp_tree_learner.hpp"

namespace LightGBM {

template <bool IS_ROOT>
__global__ void ReduceRootNodeInformationKernel(
  const CUDALeafSplitsStruct* smaller_leaf_splits_buffer,
  const CUDALeafSplitsStruct* larger_leaf_splits_buffer,
  const int num_gpu,
  const double lambda_l1,
  const double lambda_l2,
  const data_size_t num_data,
  CUDALeafSplitsStruct* smaller_leaf_splits,
  CUDALeafSplitsStruct* larger_leaf_splits,
  double* out_sum_hessians) {
  double smaller_sum_of_gradients = 0.0;
  double smaller_sum_of_hessians = 0.0f;
  int64_t smaller_sum_of_gradients_hessians = 0;
  data_size_t smaller_num_data_in_leaf = 0;
  for (int gpu_index = 0; gpu_index < num_gpu; ++gpu_index) {
    const CUDALeafSplitsStruct* leaf_splits = smaller_leaf_splits_buffer + gpu_index;
    const double smaller_gpu_sum_of_gradients = leaf_splits->sum_of_gradients;
    const double smaller_gpu_sum_of_hessians = leaf_splits->sum_of_hessians;
    const int64_t smaller_gpu_sum_of_gradients_hessians = leaf_splits->sum_of_gradients_hessians;
    const data_size_t smaller_gpu_num_data_in_leaf = leaf_splits->num_data_in_leaf;
    if (IS_ROOT) {
      smaller_sum_of_gradients += smaller_gpu_sum_of_gradients;
      smaller_sum_of_hessians += smaller_gpu_sum_of_hessians;
      smaller_sum_of_gradients_hessians += smaller_gpu_sum_of_gradients_hessians;
    } else {
      if (gpu_index == 0) {
        smaller_sum_of_gradients = smaller_gpu_sum_of_gradients;
        smaller_sum_of_hessians = smaller_gpu_sum_of_hessians;
        smaller_sum_of_gradients_hessians = smaller_gpu_sum_of_gradients_hessians;
      } else {
        assert(smaller_sum_of_gradients == smaller_gpu_sum_of_gradients);
        assert(smaller_sum_of_hessians == smaller_gpu_sum_of_hessians);
        assert(smaller_sum_of_gradients_hessians == smaller_gpu_sum_of_gradients_hessians);
      }
    }
    smaller_num_data_in_leaf += smaller_gpu_num_data_in_leaf;
  }
  smaller_leaf_splits->sum_of_gradients = smaller_sum_of_gradients;
  smaller_leaf_splits->sum_of_hessians = smaller_sum_of_hessians;
  if (IS_ROOT) {
    *out_sum_hessians = smaller_sum_of_hessians;
  }
  smaller_leaf_splits->sum_of_gradients_hessians = smaller_sum_of_gradients_hessians;
  smaller_leaf_splits->num_data_in_leaf = smaller_num_data_in_leaf;
  if (IS_ROOT) {
    assert(num_data_in_leaf == num_data);
    smaller_leaf_splits->leaf_index = 0;
  }
  const bool use_l1 = lambda_l1 > 0.0f;
  if (!use_l1) {
    // no smoothing on root node
    smaller_leaf_splits->gain = CUDALeafSplits::GetLeafGain<false, false>(smaller_sum_of_gradients, smaller_sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
  } else {
    // no smoothing on root node
    smaller_leaf_splits->gain = CUDALeafSplits::GetLeafGain<true, false>(smaller_sum_of_gradients, smaller_sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
  }
  if (!use_l1) {
    // no smoothing on root node
    smaller_leaf_splits->leaf_value =
      CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(smaller_sum_of_gradients, smaller_sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
  } else {
    // no smoothing on root node
    smaller_leaf_splits->leaf_value =
      CUDALeafSplits::CalculateSplittedLeafOutput<true, false>(smaller_sum_of_gradients, smaller_sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
  }

  if (!IS_ROOT) {
    double larger_sum_of_gradients = 0.0;
    double larger_sum_of_hessians = 0.0f;
    int64_t larger_sum_of_gradients_hessians = 0;
    data_size_t larger_num_data_in_leaf = 0;
    for (int gpu_index = 0; gpu_index < num_gpu; ++gpu_index) {
      const CUDALeafSplitsStruct* leaf_splits = larger_leaf_splits_buffer + gpu_index;
      const double larger_gpu_sum_of_gradients = leaf_splits->sum_of_gradients;
      const double larger_gpu_sum_of_hessians = leaf_splits->sum_of_hessians;
      const int64_t larger_gpu_sum_of_gradients_hessians = leaf_splits->sum_of_gradients_hessians;
      const data_size_t larger_gpu_num_data_in_leaf = leaf_splits->num_data_in_leaf;
      if (gpu_index == 0) {
        larger_sum_of_gradients = larger_gpu_sum_of_gradients;
        larger_sum_of_hessians = larger_gpu_sum_of_hessians;
        larger_sum_of_gradients_hessians = larger_gpu_sum_of_gradients_hessians;
      } else {
        assert(larger_sum_of_gradients == larger_gpu_sum_of_gradients);
        assert(larger_sum_of_hessians == larger_gpu_sum_of_hessians);
        assert(larger_sum_of_gradients_hessians == larger_gpu_sum_of_gradients_hessians);
      }
      larger_num_data_in_leaf += larger_gpu_num_data_in_leaf;
    }
    larger_leaf_splits->sum_of_gradients = larger_sum_of_gradients;
    larger_leaf_splits->sum_of_hessians = larger_sum_of_hessians;
    larger_leaf_splits->sum_of_gradients_hessians = larger_sum_of_gradients_hessians;
    larger_leaf_splits->num_data_in_leaf = larger_num_data_in_leaf;
    const bool use_l1 = lambda_l1 > 0.0f;
    if (!use_l1) {
      // no smoothing on root node
      larger_leaf_splits->gain = CUDALeafSplits::GetLeafGain<false, false>(larger_sum_of_gradients, larger_sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    } else {
      // no smoothing on root node
      larger_leaf_splits->gain = CUDALeafSplits::GetLeafGain<true, false>(larger_sum_of_gradients, larger_sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    }
    if (!use_l1) {
      // no smoothing on root node
      larger_leaf_splits->leaf_value =
        CUDALeafSplits::CalculateSplittedLeafOutput<false, false>(larger_sum_of_gradients, larger_sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    } else {
      // no smoothing on root node
      larger_leaf_splits->leaf_value =
        CUDALeafSplits::CalculateSplittedLeafOutput<true, false>(larger_sum_of_gradients, larger_sum_of_hessians, lambda_l1, lambda_l2, 0.0f, 0, 0.0f);
    }
  }
}

void CUDAExpTreeLearner::LaunchReduceLeafInformationKernel() {
  if (larger_leaf_index_ < 0) {
    // is root node
    ReduceRootNodeInformationKernel<true><<<1, 1>>>(
      smaller_leaf_splits_buffer_[0].RawData(),
      nullptr,
      config_->num_gpu,
      config_->lambda_l1,
      config_->lambda_l2,
      // TODO(shiyu1994): bagging is not supported by now
      train_data_->num_data(),
      cuda_smaller_leaf_splits_->GetCUDAStructRef(),
      nullptr,
      cuda_root_sum_hessians_.RawData());
  } else {
    ReduceRootNodeInformationKernel<false><<<1, 1>>>(
      smaller_leaf_splits_buffer_[0].RawData(),
      larger_leaf_splits_buffer_[0].RawData(),
      config_->num_gpu,
      config_->lambda_l1,
      config_->lambda_l2,
      // TODO(shiyu1994): bagging is not supported by now
      train_data_->num_data(),
      cuda_smaller_leaf_splits_->GetCUDAStructRef(),
      cuda_larger_leaf_splits_->GetCUDAStructRef(),
      nullptr);
  }
}

template <bool IS_ROOT>
__global__ void ReduceBestSplitsForLeafKernel(
  int* leaf_best_split_info_buffer,
  const int num_gpu) {
  int best_gpu_for_smaller_leaf = -1;
  double best_gain = kMinScore;
  for (int gpu_index = 0; gpu_index < num_gpu; ++gpu_index) {
    const double gain = (reinterpret_cast<const double*>(leaf_best_split_info_buffer + gpu_index * 10 + 6))[0];
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
  if (!IS_ROOT) {
    best_gain = kMinScore;
    for (int gpu_index = 0; gpu_index < num_gpu; ++gpu_index) {
      const double gain = (reinterpret_cast<const double*>(leaf_best_split_info_buffer + gpu_index * 10 + 6))[1];
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
  } else {
    leaf_best_split_info_buffer[3] = -1;
    leaf_best_split_info_buffer[4] = 0;
    leaf_best_split_info_buffer[5] = 0;
    double* gain_buffer = reinterpret_cast<double*>(leaf_best_split_info_buffer + 6);
    gain_buffer[1] = kMinScore;
  }
  leaf_best_split_info_buffer[10] = best_gpu_for_smaller_leaf;
  leaf_best_split_info_buffer[11] = best_gpu_for_larger_leaf;
}

void CUDAExpTreeLearner::LaunchReduceBestSplitsForLeafKernel() {
  if (larger_leaf_index_ < 0) {
    ReduceBestSplitsForLeafKernel<true><<<1, 1>>>(
      best_split_info_buffer_[0]->RawData(),
      config_->num_gpu);
  } else {
    ReduceBestSplitsForLeafKernel<false><<<1, 1>>>(
      best_split_info_buffer_[0]->RawData(),
      config_->num_gpu);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

}  // namespace LightGBM
