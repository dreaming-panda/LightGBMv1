/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifndef LIGHTGBM_CUDA_CUDA_EXP_TREE_LEARNER_HPP_
#define LIGHTGBM_CUDA_CUDA_EXP_TREE_LEARNER_HPP_

#ifdef USE_CUDA

#include "cuda_single_gpu_tree_learner.hpp"
#include <nccl.h>

namespace LightGBM {

class CUDAExpTreeLearner: public CUDASingleGPUTreeLearner {
 public:
  explicit CUDAExpTreeLearner(const Config* config);

  ~CUDAExpTreeLearner();

  void Init(const Dataset* train_data, bool is_constant_hessian) override;

  void ResetTrainingData(const Dataset* train_data,
                         bool is_constant_hessian) override;

  Tree* Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) override;

  void AddPredictionToScore(const Tree* tree, double* out_score) const override;

  /*void SetBaggingData(const Dataset* subset, const data_size_t* used_indices, data_size_t num_data) override;

  void RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                       const double* score, data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const override;

  void ResetConfig(const Config* config) override;

  Tree* FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t* hessians) const override;

  Tree* FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                          const score_t* gradients, const score_t* hessians) const override;*/

  void BeforeTrainWithGrad(const score_t* gradients, const score_t* hessians, const std::vector<int8_t>& is_feature_used_by_tree, ncclComm_t* comm = nullptr, cudaStream_t* stream = nullptr) override;

 private:
  void NCCLReduceHistograms();

  void LaunchReduceLeafInformationKernel();

  void NCCLReduceBestSplitsForLeaf(CUDATree* tree);

  void LaunchReduceBestSplitsForLeafKernel();

  void BroadCastBestSplit();

  void ReduceLeafInformationAfterSplit(const int best_leaf_index, const int right_leaf_index);

  std::vector<std::unique_ptr<CUDASingleGPUTreeLearner>> tree_learners_;
  std::vector<std::unique_ptr<Dataset>> datasets_;
  std::vector<std::unique_ptr<Config>> configs_;
  data_size_t num_data_per_gpu_;
  std::vector<ncclComm_t> nccl_communicators_;
  std::vector<cudaStream_t> cuda_send_streams_;
  std::vector<cudaStream_t> cuda_recv_streams_;
  std::vector<std::vector<int8_t>> is_feature_used_by_tree_per_gpu_;
  std::vector<std::unique_ptr<CUDAVector<score_t>>> per_gpu_gradients_;
  std::vector<std::unique_ptr<CUDAVector<score_t>>> per_gpu_hessians_;

  int num_total_bin_;

  std::vector<CUDAVector<CUDALeafSplitsStruct>> smaller_leaf_splits_buffer_;
  std::vector<CUDAVector<CUDALeafSplitsStruct>> larger_leaf_splits_buffer_;
  //std::vector<std::unique_ptr<CUDAVector<CUDALeafSplitsStruct>>> per_gpu_smaller_leaf_splits_;
  //std::vector<std::unique_ptr<CUDAVector<CUDALeafSplitsStruct>>> per_gpu_larger_leaf_splits_;
  std::vector<std::unique_ptr<CUDAVector<int>>> best_split_info_buffer_;
  int* host_split_info_buffer_;
  CUDAVector<double> cuda_root_sum_hessians_;
  std::vector<int> leaf_to_hist_index_map_;
  std::vector<std::unique_ptr<CUDAVector<double>>> per_gpu_scores_;
  mutable int cur_iter_;
};

}  // namespace LightGBM

#endif  // USE_CUDA
#endif  // LIGHTGBM_CUDA_CUDA_EXP_TREE_LEARNER_HPP_
