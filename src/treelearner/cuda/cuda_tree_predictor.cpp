/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_tree_predictor.hpp"

namespace LightGBM {

CUDATreePredictor::CUDATreePredictor() {}

void CUDATreePredictor::Init(const Dataset* train_data, const int max_num_leaves, const int max_num_trees, const double learning_rate,
  const uint8_t* cuda_data_uint8_t, const uint16_t* cuda_data_uint16_t, const uint32_t* cuda_data_uint32_t) {
  const int cur_num_leaves = 1;
  InitCUDAMemoryFromHostMemory<int>(&cuda_cur_num_leaves_, &cur_num_leaves, 1);

  std::vector<double> flatten_bin_upper_bounds;
  std::vector<int> feature_num_bin_offsets;
  int offset = 0;
  const int num_used_features = train_data->num_features();
  feature_num_bin_offsets.resize(num_used_features + 1, 0);
  feature_num_bin_offsets[0] = offset;
  real_feature_index_.resize(num_used_features);
  feature_missing_types_.resize(num_used_features);
  max_bin_.resize(num_used_features, 0);
  default_bin_.resize(num_used_features, 0);
  for (int inner_feature_index = 0; inner_feature_index < num_used_features; ++inner_feature_index) {
    const BinMapper* bin_mapper = train_data->FeatureBinMapper(inner_feature_index);
    const int num_bin = bin_mapper->num_bin();
    max_bin_[inner_feature_index] = static_cast<uint32_t>(bin_mapper->num_bin()) - 1;
    default_bin_[inner_feature_index] = static_cast<uint32_t>(bin_mapper->GetDefaultBin());
    const std::vector<double>& bin_upper_bound = bin_mapper->bin_upper_bound();
    CHECK_EQ(static_cast<size_t>(num_bin), bin_upper_bound.size());
    for (const auto value : bin_upper_bound) {
      flatten_bin_upper_bounds.emplace_back(value);
    }
    offset += num_bin;
    feature_num_bin_offsets[inner_feature_index + 1] = offset;
    real_feature_index_[inner_feature_index] = train_data->RealFeatureIndex(inner_feature_index);
    feature_missing_types_[inner_feature_index] = static_cast<int8_t>(bin_mapper->missing_type());
  }
  InitCUDAMemoryFromHostMemory<int>(&cuda_feature_num_bin_offsets_, feature_num_bin_offsets.data(), feature_num_bin_offsets.size());
  InitCUDAMemoryFromHostMemory<double>(&cuda_bin_upper_bounds_, flatten_bin_upper_bounds.data(), flatten_bin_upper_bounds.size());
  InitCUDAMemoryFromHostMemory<int>(&cuda_real_feature_index_, real_feature_index_.data(), real_feature_index_.size());
  InitCUDAMemoryFromHostMemory<int8_t>(&cuda_feature_missing_types_, feature_missing_types_.data(), feature_missing_types_.size());
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_max_bin_, max_bin_.data(), max_bin_.size());
  InitCUDAMemoryFromHostMemory<uint32_t>(&cuda_default_bin_, default_bin_.data(), default_bin_.size());

  size_t num_tree_structure_items = static_cast<size_t>(max_num_leaves * max_num_trees);
  AllocateCUDAMemory<int>(static_cast<size_t>(max_num_trees), &num_leaves_per_tree_);
  AllocateCUDAMemory<int>(num_tree_structure_items, &leaf_parent_);
  AllocateCUDAMemory<int>(num_tree_structure_items, &left_child_);
  AllocateCUDAMemory<int>(num_tree_structure_items, &right_child_);
  AllocateCUDAMemory<int>(num_tree_structure_items, &split_feature_inner_);
  AllocateCUDAMemory<int>(num_tree_structure_items, &split_feature_);
  AllocateCUDAMemory<double>(num_tree_structure_items, &split_gain_);
  AllocateCUDAMemory<double>(num_tree_structure_items, &internal_weight_);
  AllocateCUDAMemory<double>(num_tree_structure_items, &internal_value_);
  AllocateCUDAMemory<data_size_t>(num_tree_structure_items, &internal_count_);
  AllocateCUDAMemory<double>(num_tree_structure_items, &leaf_value_);
  AllocateCUDAMemory<double>(num_tree_structure_items, &leaf_weight_);
  AllocateCUDAMemory<data_size_t>(num_tree_structure_items, &leaf_count_);
  AllocateCUDAMemory<int>(num_tree_structure_items, &leaf_depth_);
  AllocateCUDAMemory<int8_t>(num_tree_structure_items, &decision_type_);
  AllocateCUDAMemory<uint32_t>(num_tree_structure_items, &threshold_in_bin_);
  AllocateCUDAMemory<double>(num_tree_structure_items, &threshold_);

  num_trees_ = 0;
  max_num_leaves_ = max_num_leaves;
  learning_rate_ = learning_rate;

  cuda_data_uint8_t_ = cuda_data_uint8_t;
  cuda_data_uint16_t_ = cuda_data_uint16_t;
  cuda_data_uint32_t_ = cuda_data_uint32_t;
}

void CUDATreePredictor::BeforeTrain() {
  num_trees_ = 0;
}

void CUDATreePredictor::BuildTree(const int* num_leaves,
  const int* tree_split_leaf_index,
  const int* tree_inner_feature_index,
  const uint32_t* tree_threshold,
  const double* tree_left_output,
  const double* tree_right_output,
  const data_size_t* tree_left_count,
  const data_size_t* tree_right_count,
  const double* tree_left_sum_hessian,
  const double* tree_right_sum_hessian,
  const double* tree_gain,
  const uint8_t* tree_default_left) {
  const int cur_num_leaves = 1;
  CopyFromHostToCUDADevice<int>(cuda_cur_num_leaves_, &cur_num_leaves, 1);
  LaunchBuildTreeKernel(num_leaves,
    tree_split_leaf_index,
    tree_inner_feature_index,
    tree_threshold,
    tree_left_output,
    tree_right_output,
    tree_left_count,
    tree_right_count,
    tree_left_sum_hessian,
    tree_right_sum_hessian,
    tree_gain,
    tree_default_left);
  ++num_trees_;
}

void CUDATreePredictor::Predict(const int num_data, const int num_feature, const double* data, double* out_score) const {
  LaunchPredictKernel(num_data, num_feature, data, out_score, learning_rate_);
}

}  // namespace LightGBM

#endif  // USE_CUDA
