/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_tree_predictor.hpp"

namespace LightGBM {

__device__ void SetDecisionType(int8_t* decision_type, bool input, int8_t mask) {
  if (input) {
    (*decision_type) |= mask;
  } else {
    (*decision_type) &= (127 - mask);
  }
}

__device__ void SetMissingType(int8_t* decision_type, int8_t input) {
  (*decision_type) &= 3;
  (*decision_type) |= (input << 2);
}

__device__ bool GetDecisionType(int8_t decision_type, int8_t mask) {
  return (decision_type & mask) > 0;
}

__device__ int8_t GetMissingType(int8_t decision_type) {
  return (decision_type >> 2) & 3;
}

__device__ bool IsZero(double fval) {
  return (fval >= -kZeroThreshold && fval <= kZeroThreshold);
}

__device__ int NumericalDecision(double fval,
  const int8_t decision_type,
  const uint16_t left_child,
  const uint16_t right_child,
  const double threshold) {
  int8_t missing_type = GetMissingType(decision_type);
  if (isnan(fval) && missing_type != 2) {
    fval = 0.0f;
  }
  if ((missing_type == 1 && IsZero(fval))
      || (missing_type == 2 && isnan(fval))) {
    if (GetDecisionType(decision_type, kDefaultLeftMask)) {
      return left_child;
    } else {
      return right_child;
    }
  }
  if (fval <= threshold) {
    return left_child;
  } else {
    return right_child;
  }
}

#define InnerNumericalDecision(fval, max_bin, default_bin, decision_type, left_child, right_child, threshold) \
  int8_t missing_type = GetMissingType(decision_type); \
  if ((missing_type == 1 && fval == default_bin) || \
    (missing_type == 2 && fval == max_bin)) { \
    if (GetDecisionType(decision_type, kDefaultLeftMask)) { \
      leaf = left_child; \
    } else { \
      leaf = right_child; \
    } \
  } \
  if (fval <= threshold) { \
    leaf = left_child; \
  } else { \
    leaf = right_child; \
  }

__device__ int Split(
  const int leaf,
  const int feature,
  const int real_feature,
  const double left_value,
  const double right_value,
  const int left_cnt,
  const int right_cnt,
  const double left_weight,
  const double right_weight,
  const float gain,
  const int8_t missing_type,
  const bool default_left,
  const uint32_t threshold_bin,
  const double threshold_double,

  int* cur_num_leaves,
  int* leaf_parent,
  int* left_child,
  int* right_child,
  int* split_feature_inner,
  int* split_feature,
  double* split_gain,
  double* internal_weight,
  double* internal_value,
  data_size_t* internal_count,
  double* leaf_value,
  double* leaf_weight,
  data_size_t* leaf_count,
  int* leaf_depth,
  int8_t* decision_type,
  uint32_t* threshold_in_bin,
  double* threshold) {
  int& cur_num_leaves_ref = *cur_num_leaves;
  int new_node_idx = cur_num_leaves_ref - 1;
  // update parent info
  int parent = leaf_parent[leaf];
  if (parent >= 0) {
    // if cur node is left child
    if (left_child[parent] == ~leaf) {
      left_child[parent] = new_node_idx;
    } else {
      right_child[parent] = new_node_idx;
    }
  }
  // add new node
  split_feature_inner[new_node_idx] = feature;
  split_feature[new_node_idx] = real_feature;
  split_gain[new_node_idx] = gain;
  // add two new leaves
  left_child[new_node_idx] = ~leaf;
  right_child[new_node_idx] = ~cur_num_leaves_ref;
  // update new leaves
  leaf_parent[leaf] = new_node_idx;
  leaf_parent[cur_num_leaves_ref] = new_node_idx;
  // save current leaf value to internal node before change
  internal_weight[new_node_idx] = leaf_weight[leaf];
  internal_value[new_node_idx] = leaf_value[leaf];
  internal_count[new_node_idx] = left_cnt + right_cnt;
  /*if (leaf == 0) {
    printf("leaf_value[%d] from left_value = %f\n", 0, left_value);
  }*/
  leaf_value[leaf] = std::isnan(left_value) ? 0.0f : left_value;
  leaf_weight[leaf] = left_weight;
  leaf_count[leaf] = left_cnt;
  leaf_value[cur_num_leaves_ref] = std::isnan(right_value) ? 0.0f : right_value;
  leaf_weight[cur_num_leaves_ref] = right_weight;
  leaf_count[cur_num_leaves_ref] = right_cnt;
  // update leaf depth
  leaf_depth[cur_num_leaves_ref] = leaf_depth[leaf] + 1;
  leaf_depth[leaf]++;

  SetDecisionType(&decision_type[new_node_idx], false, kCategoricalMask);
  SetDecisionType(&decision_type[new_node_idx], default_left, kDefaultLeftMask);
  SetMissingType(&decision_type[new_node_idx], missing_type);

  threshold_in_bin[new_node_idx] = threshold_bin;
  threshold[new_node_idx] = threshold_double;
  ++cur_num_leaves_ref;
  return cur_num_leaves_ref - 1;
}

__global__ void BuildTreeKernel(
  const int cur_num_trees,
  int* cur_num_leaves,
  const int* num_leaves,
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
  const uint8_t* tree_default_left,

  const double* cuda_bin_upper_bounds,
  const int* cuda_feature_num_bin_offsets,
  const int* cuda_real_feature_index,
  const int8_t* cuda_feature_missing_types,

  int* num_leaves_per_tree,
  int* leaf_parent,
  int* left_child,
  int* right_child,
  int* split_feature_inner,
  int* split_feature,
  double* split_gain,
  double* internal_weight,
  double* internal_value,
  data_size_t* internal_count,
  double* leaf_value,
  double* leaf_weight,
  data_size_t* leaf_count,
  int* leaf_depth,
  int8_t* decision_type,
  uint32_t* threshold_in_bin,
  double* threshold) {
  const int num_leaves_ref = *num_leaves;
  num_leaves_per_tree[cur_num_trees] = num_leaves_ref;
  for (int i = 0; i < num_leaves_ref - 1; ++i) {
    const int leaf = tree_split_leaf_index[i];
    const int inner_feature_index = tree_inner_feature_index[i];
    const uint32_t threshold_bin = tree_threshold[i];
    const double threshold_real = cuda_bin_upper_bounds[cuda_feature_num_bin_offsets[inner_feature_index] + threshold_bin];
    const int real_feature_index = cuda_real_feature_index[inner_feature_index];
    const int8_t missing_type = cuda_feature_missing_types[inner_feature_index];
    Split(leaf,
      inner_feature_index,
      real_feature_index,
      tree_left_output[i],
      tree_right_output[i],
      tree_left_count[i],
      tree_right_count[i],
      tree_left_sum_hessian[i],
      tree_right_sum_hessian[i],
      tree_gain[i],
      missing_type,
      tree_default_left[i],
      threshold_bin,
      threshold_real,

      cur_num_leaves,
      leaf_parent,
      left_child,
      right_child,
      split_feature_inner,
      split_feature,
      split_gain,
      internal_weight,
      internal_value,
      internal_count,
      leaf_value,
      leaf_weight,
      leaf_count,
      leaf_depth,
      decision_type,
      threshold_in_bin,
      threshold);
  }
}

void CUDATreePredictor::LaunchBuildTreeKernel(const int* num_leaves,
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
  const int offset = num_trees_ * max_num_leaves_;
  BuildTreeKernel<<<1, 1>>>(
    num_trees_,
    cuda_cur_num_leaves_,
    num_leaves,
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
    tree_default_left,
  
    cuda_bin_upper_bounds_,
    cuda_feature_num_bin_offsets_,
    cuda_real_feature_index_,
    cuda_feature_missing_types_,

    num_leaves_per_tree_ + num_trees_,
    leaf_parent_ + offset,
    left_child_ + offset,
    right_child_ + offset,
    split_feature_inner_ + offset,
    split_feature_ + offset,
    split_gain_ + offset,
    internal_weight_ + offset,
    internal_value_ + offset,
    internal_count_ + offset,
    leaf_value_ + offset,
    leaf_weight_ + offset,
    leaf_count_ + offset,
    leaf_depth_ + offset,
    decision_type_ + offset,
    threshold_in_bin_ + offset,
    threshold_ + offset);
  //SynchronizeCUDADevice();
  /*std::vector<double> cpu_leaf_values(256);
  CopyFromCUDADeviceToHost(cpu_leaf_values.data(), leaf_value_, cpu_leaf_values.size());
  for (size_t i = 0; i < cpu_leaf_values.size(); ++i) {
    Log::Warning("cpu_leaf_values[%d] = %f", i, cpu_leaf_values[i]);
  }*/
  /*const int check_size = 256;
  std::vector<double> leaf_values(check_size);
  std::vector<int> node_split_feature(check_size);
  std::vector<uint32_t> node_split_threshold(check_size);
  CopyFromCUDADeviceToHost<double>(leaf_values.data(), leaf_value_, check_size);
  CopyFromCUDADeviceToHost<int>(node_split_feature.data(), split_feature_inner_, check_size);
  CopyFromCUDADeviceToHost<uint32_t>(node_split_threshold.data(), threshold_in_bin_, check_size);
  for (int i = 0; i < check_size; ++i) {
    Log::Warning("leaf_values[%d] = %f, node_split_feature[%d] = %d, node_split_threshold[%d] = %d", i, leaf_values[i], i, node_split_feature[i], i, node_split_threshold[i]);
  }*/
}

__global__ void PredictKernel(
  const int max_num_leaves,
  const int num_trees,
  const int num_data,
  const int num_original_feature,
  const double* data,
  double* out_score,

  const int* num_leaves_pre_tree,
  const int* leaf_parent,
  const int* left_child,
  const int* right_child,
  const int* split_feature_inner,
  const int* split_feature,
  const double* split_gain,
  const double* internal_weight,
  const double* internal_value,
  const data_size_t* internal_count,
  const double* leaf_value,
  const double* leaf_weight,
  const data_size_t* leaf_count,
  const int* leaf_depth,
  const int8_t* decision_type,
  const uint32_t* threshold_in_bin,
  const double* threshold,
  const double* leaf_output) {
  __shared__ int8_t shared_decision_type[PREDICT_SHARE_MEM_SIZE];
  __shared__ double shared_threshold[PREDICT_SHARE_MEM_SIZE];
  __shared__ uint16_t shared_left_child[PREDICT_SHARE_MEM_SIZE];
  __shared__ uint16_t shared_right_child[PREDICT_SHARE_MEM_SIZE];
  __shared__ double shared_out_score[PREDICT_BLOCK_SIZE];
  __shared__ double shared_leaf_output[PREDICT_SHARE_MEM_SIZE];
  shared_out_score[threadIdx.x] = 0.0f;
  __syncthreads();
  const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  for (int tree_index = 0; tree_index < num_trees; ++tree_index) {
    const int offset = tree_index * max_num_leaves;
    const int num_leaves_in_this_tree = num_leaves_pre_tree[tree_index];
    const int8_t* decision_type_pointer = decision_type + offset;
    const double* threshold_pointer = threshold + offset;
    const int* left_child_pointer = left_child + offset;
    const int* right_child_pointer = right_child + offset;
    const double* leaf_output_pointer = leaf_output + offset;
    for (unsigned int i = threadIdx.x; i < num_leaves_in_this_tree; i += PREDICT_BLOCK_SIZE) {
      shared_decision_type[i] = decision_type_pointer[i];
      shared_threshold[i] = threshold_pointer[i];
      shared_left_child[i] = static_cast<uint16_t>(left_child_pointer[i]);
      shared_right_child[i] = static_cast<uint16_t>(right_child_pointer[i]);
      shared_leaf_output[i] = leaf_output_pointer[i];
    }
    __syncthreads();
    const double* data_pointer = data + thread_index * num_original_feature;
    int leaf = 0;
    while (leaf >= 0) {
      NumericalDecision(data_pointer[split_feature[leaf]],
        shared_decision_type[leaf],
        shared_left_child[leaf],
        shared_right_child[leaf],
        shared_threshold[leaf]);
    }
    leaf = ~leaf;
    shared_out_score[threadIdx.x] += shared_leaf_output[leaf];
  }
  __syncthreads();
  out_score[thread_index] = shared_out_score[threadIdx.x];
}

template <typename BIN_TYPE>
__global__ void InnerPredictKernel(
  const int max_num_leaves,
  const int num_trees,
  const int num_data,
  const int num_original_feature,
  const BIN_TYPE* data,
  double* out_score,
  const double learning_rate,

  const uint32_t* max_bin,
  const uint32_t* default_bin,

  const int* num_leaves_pre_tree,
  const int* leaf_parent,
  const int* left_child,
  const int* right_child,
  const int* split_feature_inner,
  const int* split_feature,
  const double* split_gain,
  const double* internal_weight,
  const double* internal_value,
  const data_size_t* internal_count,
  const double* leaf_value,
  const double* leaf_weight,
  const data_size_t* leaf_count,
  const int* leaf_depth,
  const int8_t* decision_type,
  const uint32_t* threshold_in_bin,
  const double* threshold,
  const double* leaf_output) {
  __shared__ int8_t shared_decision_type[PREDICT_SHARE_MEM_SIZE];
  __shared__ uint32_t shared_threshold[PREDICT_SHARE_MEM_SIZE];
  __shared__ int shared_left_child[PREDICT_SHARE_MEM_SIZE];
  __shared__ int shared_right_child[PREDICT_SHARE_MEM_SIZE];
  __shared__ double shared_out_score[PREDICT_BLOCK_SIZE];
  __shared__ double shared_leaf_output[PREDICT_SHARE_MEM_SIZE];
  shared_out_score[threadIdx.x] = 0.0f;
  __syncthreads();
  const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  for (int tree_index = 0; tree_index < num_trees; ++tree_index) {
    const int offset = tree_index * max_num_leaves;
    const int num_leaves_in_this_tree = num_leaves_pre_tree[tree_index];
    const int8_t* decision_type_pointer = decision_type + offset;
    const uint32_t* threshold_pointer = threshold_in_bin + offset;
    const int* left_child_pointer = left_child + offset;
    const int* right_child_pointer = right_child + offset;
    const double* leaf_output_pointer = leaf_output + offset;
    for (unsigned int i = threadIdx.x; i < num_leaves_in_this_tree; i += PREDICT_BLOCK_SIZE) {
      shared_decision_type[i] = decision_type_pointer[i];
      shared_threshold[i] = threshold_pointer[i];
      shared_left_child[i] = left_child_pointer[i];
      shared_right_child[i] = right_child_pointer[i];
      shared_leaf_output[i] = leaf_output_pointer[i];
    }
    __syncthreads();
    if (thread_index < num_data) {
      const BIN_TYPE* data_pointer = data + thread_index * num_original_feature;
      int leaf = 0;
      while (leaf >= 0) {
        const int inner_feature_index = split_feature_inner[leaf];
        InnerNumericalDecision(data_pointer[inner_feature_index],
          max_bin[inner_feature_index],
          default_bin[inner_feature_index],
          shared_decision_type[leaf],
          shared_left_child[leaf],
          shared_right_child[leaf],
          shared_threshold[leaf]);
      }
      leaf = ~leaf;
      shared_out_score[threadIdx.x] += shared_leaf_output[leaf];
    }
  }
  __syncthreads();
  if (thread_index < num_data) {
    out_score[thread_index] += shared_out_score[threadIdx.x] * learning_rate;
  }
}

void CUDATreePredictor::LaunchPredictKernel(
  const int num_data,
  const int num_original_feature,
  const double* data,
  double* out_score,
  const double learning_rate) const {
  const int num_blocks = (num_data + PREDICT_BLOCK_SIZE - 1) / PREDICT_BLOCK_SIZE;
  InnerPredictKernel<<<num_blocks, PREDICT_BLOCK_SIZE>>>(
    max_num_leaves_,
    num_trees_,
    num_data,
    num_original_feature,
    cuda_data_uint8_t_,
    out_score,
    learning_rate,

    cuda_max_bin_,
    cuda_default_bin_,

    num_leaves_per_tree_,
    leaf_parent_,
    left_child_,
    right_child_,
    split_feature_inner_,
    split_feature_,
    split_gain_,
    internal_weight_,
    internal_value_,
    internal_count_,
    leaf_value_,
    leaf_weight_,
    leaf_count_,
    leaf_depth_,
    decision_type_,
    threshold_in_bin_,
    threshold_,
    leaf_value_);
}

}  // namespace LightGBM

#endif  // USE_CUDA
