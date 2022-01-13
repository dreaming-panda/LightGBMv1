/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_NEW_CUDA_TREE_LEARNER_HPP_
#define LIGHTGBM_NEW_CUDA_TREE_LEARNER_HPP_

#include <memory>
#include <vector>

#ifdef USE_CUDA

#include "cuda_leaf_splits.hpp"
#include "cuda_histogram_constructor.hpp"
#include "cuda_data_partition.hpp"
#include "cuda_best_split_finder.hpp"

#include "cuda_gradient_discretizer.hpp"
#include "../serial_tree_learner.h"

#include <nccl.h>

namespace LightGBM {

#define CUDA_SINGLE_GPU_TREE_LEARNER_BLOCK_SIZE (1024)

class CUDASingleGPUTreeLearner: public SerialTreeLearner {
 public:
  explicit CUDASingleGPUTreeLearner(const Config* config);

  ~CUDASingleGPUTreeLearner();

  void Init(const Dataset* train_data, bool is_constant_hessian) override;

  void Init(const Dataset* train_data, bool is_constant_hessian, int gpu_index);

  void ResetTrainingData(const Dataset* train_data,
                         bool is_constant_hessian) override;

  Tree* Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) override;

  void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data) override;

  void AddPredictionToScore(const Tree* tree, double* out_score) const override;

  void AddPredictionToScore(double* out_score, const int num_leaves, const double shrinkage_rate) const;

  void RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                       const double* score, data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const override;

  void ResetConfig(const Config* config) override;

  Tree* FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t* hessians) const override;

  Tree* FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                          const score_t* gradients, const score_t* hessians) const override;

  virtual void BeforeTrainWithGrad(const score_t* gradients, const score_t* hessians, const std::vector<int8_t>& is_feature_used_by_tree, ncclComm_t* comm = nullptr, cudaStream_t* stream = nullptr);

  void CUDAConstructHistograms(
    data_size_t global_num_data_in_smaller_leaf,
    data_size_t global_num_data_in_larger_leaf,
    double global_sum_hessians_in_smaller_leaf,
    double global_sum_hessians_in_larger_leaf);

  void CUDASubtractHistograms();

  void CUDAFindBestSplitsForLeaf(
    data_size_t global_num_data_in_smaller_leaf,
    data_size_t global_num_data_in_larger_leaf,
    double global_sum_hessians_in_smaller_leaf,
    double global_sum_hessians_in_larger_leaf);

  void CUDASplit(int right_leaf_index, ncclComm_t* comm = nullptr);

  hist_t* cuda_hist_pointer() {
    return cuda_histogram_constructor_->cuda_hist_pointer();
  }

  const CUDALeafSplitsStruct* GetSmallerLeafSplitsStruct() const {
    return cuda_smaller_leaf_splits_->GetCUDAStruct();
  }

  const CUDALeafSplitsStruct* GetLargerLeafSplitsStruct() const {
    return cuda_larger_leaf_splits_->GetCUDAStruct();
  }

  const int* cuda_best_split_info_buffer() const {
    return cuda_best_split_finder_->cuda_best_split_info_buffer();
  }

  CUDASplitInfo* cuda_leaf_best_split_info() {
    return cuda_best_split_finder_->cuda_leaf_best_split_info();
  }

  void SetBestSplit(
    const int best_leaf_index,
    const int smaller_leaf_best_split_feature,
    const uint32_t smaller_leaf_best_split_threshold,
    const uint8_t smaller_leaf_best_split_default_left,
    const double smaller_leaf_best_split_gain,
    const int larger_leaf_best_split_feature,
    const uint32_t larger_leaf_best_split_threshold,
    const uint8_t larger_leaf_best_split_default_left,
    const double larger_leaf_best_split_gain);

  data_size_t leaf_num_data(const int leaf_index) const {
    return leaf_num_data_[leaf_index];
  }

  double leaf_sum_hessians(const int leaf_index) const {
    return leaf_sum_hessians_[leaf_index];
  }

  void SetSmallerAndLargerLeaves(const int smaller_leaf_index, const int larger_leaf_index) {
    smaller_leaf_index_ = smaller_leaf_index;
    larger_leaf_index_ = larger_leaf_index;
  }

 protected:
  void ReduceLeafStat(CUDATree* old_tree, const score_t* gradients, const score_t* hessians, const data_size_t* num_data_in_leaf) const;

  void LaunchReduceLeafStatKernel(const score_t* gradients, const score_t* hessians, const data_size_t* num_data_in_leaf,
    const int* leaf_parent, const int* left_child, const int* right_child,
    const int num_leaves, const data_size_t num_data, double* cuda_leaf_value, const double shrinkage_rate) const;

  void ConstructBitsetForCategoricalSplit(const CUDASplitInfo* best_split_info);

  void LaunchConstructBitsetForCategoricalSplitKernel(const CUDASplitInfo* best_split_info);

  void AllocateBitset();

  void CheckSplitValid(
    const int inner_split_feature,
    const int left_leaf, const int right_leaf, const double sum_left_gradients, const double sum_right_gradients);

  void RenewDiscretizedTreeLeaves(CUDATree* cuda_tree);

  void LaunchCalcLeafValuesGivenGradStat(CUDATree* cuda_tree, const data_size_t* num_data_in_leaf);

  // GPU device ID
  int gpu_device_id_;
  // number of threads on CPU
  int num_threads_;

  // CUDA components for tree training

  // leaf splits information for smaller and larger leaves
  std::unique_ptr<CUDALeafSplits> cuda_smaller_leaf_splits_;
  std::unique_ptr<CUDALeafSplits> cuda_larger_leaf_splits_;
  // data partition that partitions data indices into different leaves
  std::unique_ptr<CUDADataPartition> cuda_data_partition_;
  // for histogram construction
  std::unique_ptr<CUDAHistogramConstructor> cuda_histogram_constructor_;
  // for best split information finding, given the histograms
  std::unique_ptr<CUDABestSplitFinder> cuda_best_split_finder_;
  std::unique_ptr<CUDAGradientDiscretizer> cuda_gradient_discretizer_;

  std::vector<int> leaf_best_split_feature_;
  std::vector<uint32_t> leaf_best_split_threshold_;
  std::vector<uint8_t> leaf_best_split_default_left_;
  std::vector<data_size_t> leaf_num_data_;
  std::vector<data_size_t> leaf_data_start_;
  std::vector<double> leaf_sum_hessians_;
  std::vector<double> leaf_best_split_gain_;
  int smaller_leaf_index_;
  int larger_leaf_index_;
  int best_leaf_index_;
  int num_cat_threshold_;
  bool has_categorical_feature_;

  std::vector<int> categorical_bin_to_value_;
  std::vector<int> categorical_bin_offsets_;

  mutable CUDAVector<double> cuda_leaf_gradient_stat_buffer_;
  mutable CUDAVector<double> cuda_leaf_hessian_stat_buffer_;
  mutable data_size_t refit_num_data_;
  uint32_t* cuda_bitset_;
  size_t cuda_bitset_len_;
  uint32_t* cuda_bitset_inner_;
  size_t cuda_bitset_inner_len_;
  size_t* cuda_block_bitset_len_buffer_;
  int* cuda_categorical_bin_to_value_;
  int* cuda_categorical_bin_offsets_;
};

}  // namespace LightGBM

#else  // USE_CUDA

// When GPU support is not compiled in, quit with an error message

namespace LightGBM {

class CUDASingleGPUTreeLearner: public SerialTreeLearner {
 public:
    #pragma warning(disable : 4702)
    explicit CUDASingleGPUTreeLearner(const Config* tree_config) : SerialTreeLearner(tree_config) {
      Log::Fatal("CUDA Tree Learner was not enabled in this build.\n"
                 "Please recompile with CMake option -DUSE_CUDA=1");
    }
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_NEW_CUDA_TREE_LEARNER_HPP_
