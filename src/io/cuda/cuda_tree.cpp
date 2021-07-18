/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/cuda/cuda_tree.hpp>

namespace LightGBM {

CUDATree::CUDATree(int max_leaves, bool track_branch_features, bool is_linear):
Tree(max_leaves, track_branch_features, is_linear),
num_threads_per_block_add_prediction_to_score_(1024) {
  is_cuda_tree_ = true;
  Log::Fatal("CUDATree can be only created from host Tree.");
}

CUDATree::CUDATree(const Tree* host_tree):
  Tree(*host_tree),
  num_threads_per_block_add_prediction_to_score_(1024) {
  is_cuda_tree_ = true;
  InitCUDA();
}

CUDATree::~CUDATree() {}

void CUDATree::InitCUDA() {
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_left_child_,
                                    left_child_.data(),
                                    left_child_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_right_child_,
                                    right_child_.data(),
                                    right_child_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_split_feature_inner_,
                                    split_feature_inner_.data(),
                                    split_feature_inner_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_split_feature_,
                                    split_feature_.data(),
                                    split_feature_.size(),
                                    __FILE__,
                                    __LINE__);
  for (int i = 0; i < num_leaves_ - 1; ++i) {
    Log::Warning("CUDATree threshold_in_bin_[%d] = %d", i, threshold_in_bin_[i]);
  }
  InitCUDAMemoryFromHostMemoryOuter<uint32_t>(&cuda_threshold_in_bin_,
                                    threshold_in_bin_.data(),
                                    threshold_in_bin_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_threshold_,
                                    threshold_.data(),
                                    threshold_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<int8_t>(&cuda_decision_type_,
                                    decision_type_.data(),
                                    decision_type_.size(),
                                    __FILE__,
                                    __LINE__);
  for (int i = 0; i < num_leaves_; ++i) {
    Log::Warning("CUDATree leaf_value_[%d] = %f", i, leaf_value_[i]);
  }
  InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_leaf_value_,
                                    leaf_value_.data(),
                                    leaf_value_.size(),
                                    __FILE__,
                                    __LINE__);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
}

void CUDATree::AddPredictionToScore(const Dataset* data,
                                    data_size_t num_data,
                                    double* score) const {
  LaunchAddPredictionToScoreKernel(data, num_data, score);
  SynchronizeCUDADeviceOuter(__FILE__, __LINE__);
  Log::Warning("after add prediction to score");
}

void CUDATree::AddPredictionToScore(const Dataset* /*data*/,
                            const data_size_t* /*used_data_indices*/,
                            data_size_t /*num_data*/, double* /*score*/) const {
  Log::Fatal("bagging is not supported by CUDA version yet.");
}

inline void CUDATree::Shrinkage(double rate) {
  Tree::Shrinkage(rate);
  LaunchShrinkageKernel(rate);
}

}  // namespace LightGBM
