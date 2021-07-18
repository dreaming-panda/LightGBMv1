/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_CUDA_TREE_PREDICTOR_HPP_
#define LIGHTGBM_CUDA_TREE_PREDICTOR_HPP_

#ifdef USE_CUDA

#include <LightGBM/meta.h>
#include <LightGBM/config.h>
#include <LightGBM/tree.h>
#include "new_cuda_utils.hpp"

#include <fstream>
#include <vector>

#define PREDICT_BLOCK_SIZE (1024)
#define PREDICT_SHARE_MEM_SIZE (1024)

namespace LightGBM {

class CUDATreePredictor {
 public:
  CUDATreePredictor();

  void Init(const Dataset* train_data, const int max_num_leaves, const int max_num_trees, const double learning_rate,
    const uint8_t* cuda_data_uint8_t, const uint16_t* cuda_data_uint16_t, const uint32_t* cuda_data_uint32_t);

  void Predict(const int num_data, const int num_original_feature, const double* data, double* out_score) const;

  void BeforeTrain();

  void BuildTree(const int* num_leaves,
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
    const uint8_t* tree_default_left);

 private:
  void LaunchBuildTreeKernel(const int* num_leaves,
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
    const uint8_t* tree_default_left);

  void LaunchPredictKernel(const int num_data, const int num_original_feature, const double* data, double* out_score, const double learning_rate) const;

  template <typename DATA_TYPE>
  void OutputToFile(const std::string filename, const std::vector<DATA_TYPE>& data) const {
    std::ofstream fout(filename.c_str());
    for (size_t i = 0; i < data.size(); ++i) {
      fout << data[i] << "\n";
    }
    fout.close();
  }

  // CUDA memory, held by this object
  // for tree structure
  int* num_leaves_per_tree_;
  int* leaf_parent_;
  int* left_child_;
  int* right_child_;
  int* split_feature_inner_;
  int* split_feature_;
  double* split_gain_;
  double* internal_weight_;
  double* internal_value_;
  data_size_t* internal_count_;
  double* leaf_value_;
  double* leaf_weight_;
  data_size_t* leaf_count_;
  int* leaf_depth_;
  int8_t* decision_type_;
  uint32_t* threshold_in_bin_;
  double* threshold_;
  // for tree structure building
  double* cuda_bin_upper_bounds_;
  int* cuda_feature_num_bin_offsets_;
  int* cuda_cur_num_leaves_;
  int* cuda_real_feature_index_;
  int8_t* cuda_feature_missing_types_;
  // for prediction
  uint32_t* cuda_max_bin_;
  uint32_t* cuda_default_bin_;

  // CUDA memory, held by other objects
  const uint8_t* cuda_data_uint8_t_;
  const uint16_t* cuda_data_uint16_t_;
  const uint32_t* cuda_data_uint32_t_;

  // Host memory
  std::vector<int> real_feature_index_;
  std::vector<int8_t> feature_missing_types_;
  std::vector<uint32_t> max_bin_;
  std::vector<uint32_t> default_bin_;
  int num_trees_;
  int max_num_leaves_;
  double learning_rate_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_TREE_PREDICTOR_HPP_
