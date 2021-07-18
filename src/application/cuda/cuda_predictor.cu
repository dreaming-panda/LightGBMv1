/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_predictor.hpp"

namespace LightGBM {

__device__ int FindIndex(const int* feature_index, const int num_elements, const int target) {
  int low = 0;
  int high = num_elements;
  int mid = (low + high) / 2;
  while (low <= high) {
    if (feature_index[mid] < target) {
      low = mid + 1;
    } else if (feature_index[mid] > target) {
      high = mid - 1;
    } else {
      return mid;
    }
    mid = (low + high) / 2;
  }
  return num_elements;
}

__global__ void PredictorForMapKernel(
  // dataset information
  const data_size_t num_data,
  const int* cuda_row_ptr,
  const int* cuda_feature_index_per_row,
  const double* cuda_feature_values,
  // boosting information
  const int num_trees,
  // tree information
  const int* cuda_num_leaves,
  const double** cuda_threshold,
  const int8_t** cuda_decision_type,
  const int** cuda_split_feature,
  const int** cuda_left_child,
  const int** cuda_right_child,
  const double** cuda_leaf_value,
  // output
  double* score) {
  __shared__ double cuda_tree_threshold[CDUA_PREDICTOR_MAX_TREE_SIZE];
  __shared__ int8_t cuda_tree_decision_type[CDUA_PREDICTOR_MAX_TREE_SIZE];
  __shared__ int cuda_tree_split_feature[CDUA_PREDICTOR_MAX_TREE_SIZE];
  __shared__ int cuda_tree_left_child[CDUA_PREDICTOR_MAX_TREE_SIZE];
  __shared__ int cuda_tree_right_child[CDUA_PREDICTOR_MAX_TREE_SIZE];
  __shared__ double cuda_tree_leaf_value[CDUA_PREDICTOR_MAX_TREE_SIZE];
  const data_size_t data_index = static_cast<data_size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const unsigned int thread_index = threadIdx.x;
  double all_tree_score = 0.0f;
  for (int tree_index = 0; tree_index < num_trees; ++tree_index) {
    const int cuda_tree_num_leaves = cuda_num_leaves[tree_index];
    const double* cuda_tree_threshold_pointer = cuda_threshold[tree_index];
    const int8_t* cuda_tree_decision_type_pointer = cuda_decision_type[tree_index];
    const int* cuda_tree_split_feature_pointer = cuda_split_feature[tree_index];
    const int* cuda_tree_left_child_pointer = cuda_left_child[tree_index];
    const int* cuda_tree_right_child_pointer = cuda_right_child[tree_index];
    const double* cuda_tree_leaf_value_pointer = cuda_leaf_value[tree_index];
    if (thread_index < cuda_tree_num_leaves - 1) {
      cuda_tree_threshold[thread_index] = cuda_tree_threshold_pointer[thread_index];
      cuda_tree_decision_type[thread_index] = cuda_tree_decision_type_pointer[thread_index];
      cuda_tree_split_feature[thread_index] = cuda_tree_split_feature_pointer[thread_index];
      cuda_tree_left_child[thread_index] = cuda_tree_left_child_pointer[thread_index];
      cuda_tree_right_child[thread_index] = cuda_tree_right_child_pointer[thread_index];
      cuda_tree_leaf_value[thread_index] = cuda_tree_leaf_value_pointer[thread_index];
    }
    __syncthreads();
    if (data_index < num_data) {
      const int start = cuda_row_ptr[data_index];
      const int end = cuda_row_ptr[data_index + 1];
      const int num_elements = end - start;
      int node = 0;
      while (node >= 0) {
        const int split_feature = cuda_tree_split_feature[node];
        const int8_t decision_type = cuda_tree_decision_type[node];
        const int8_t missing_type = ((decision_type >> 2) & 3);
        const int position = FindIndex(cuda_feature_index_per_row + start, split_feature, num_elements);
        double value = nanf("");
        if (position == num_elements && missing_type != 2) {
          value = 0.0f;
        }
        const bool default_left = ((decision_type & kDefaultLeftMask) > 0);
        if ((missing_type == 1 && fabs(value) <= kZeroThreshold) ||
            (missing_type == 2 && isnan(value))) {
          if (default_left) {
            node = cuda_tree_left_child[node];
          } else {
            node = cuda_tree_right_child[node];
          }
        } else {
          if (value <= cuda_tree_threshold[node]) {
            node = cuda_tree_left_child[node];
          } else {
            node = cuda_tree_right_child[node];
          }
        }
      }
      all_tree_score += cuda_tree_leaf_value[~node];
    }
  }
  if (data_index < num_data) {
    score[data_index] = all_tree_score;
  }
}

void CUDAPredictor::LaunchPredictForMapKernel(double* score) {
  const int num_blocks = (num_data_ + CDUA_PREDICTOR_MAX_TREE_SIZE - 1) / CDUA_PREDICTOR_MAX_TREE_SIZE;
  PredictorForMapKernel<<<num_blocks, CDUA_PREDICTOR_MAX_TREE_SIZE>>>(
    // dataset information
    num_data_,
    cuda_row_ptr_,
    cuda_feature_index_per_row_,
    cuda_feature_values_,
    // boosting information
    static_cast<int>(cuda_models_.size()),
    // tree information
    cuda_all_tree_num_leaves_,
    cuda_all_tree_threshold_,
    cuda_all_tree_decision_type_,
    cuda_all_tree_split_feature_,
    cuda_all_tree_left_child_,
    cuda_all_tree_right_child_,
    cuda_all_tree_leaf_value_,
    // output
    score);
}

}  // namespace LightGBM
