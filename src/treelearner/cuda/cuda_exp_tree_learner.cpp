/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_exp_tree_learner.hpp"

namespace LightGBM {

CUDAExpTreeLearner::CUDAExpTreeLearner(const Config* config): CUDASingleGPUTreeLearner(config) {
  CHECK_GE(config_->num_gpu, 1);
}

CUDAExpTreeLearner::~CUDAExpTreeLearner() {
  tree_learners_.clear();
  tree_learners_.shrink_to_fit();
  datasets_.clear();
  datasets_.shrink_to_fit();
  configs_.clear();
  configs_.shrink_to_fit();
  CUDASUCCESS_OR_FATAL(cudaFreeHost(host_split_info_buffer_));
}

void CUDAExpTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  SerialTreeLearner::Init(train_data, is_constant_hessian);
  num_threads_ = OMP_NUM_THREADS();
  std::vector<data_size_t> all_data_indices(num_data_, 0);
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (data_size_t data_index = 0; data_index < train_data_->num_data(); ++data_index) {
    all_data_indices[data_index] = data_index;
  }
  const data_size_t num_data_per_gpu_ = (num_data_ + config_->num_gpu - 1) / config_->num_gpu;
  tree_learners_.resize(config_->num_gpu);
  datasets_.resize(config_->num_gpu);
  configs_.resize(config_->num_gpu);
  #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
    const data_size_t data_start = gpu_index * num_data_per_gpu_;
    const data_size_t data_end = std::min(data_start + num_data_per_gpu_, num_data_);
    const data_size_t num_data_in_gpu = data_end - data_start;
    configs_[gpu_index].reset(new Config(*config_));
    configs_[gpu_index]->gpu_device_id = gpu_index;
    tree_learners_[gpu_index].reset(new CUDASingleGPUTreeLearner(configs_[gpu_index].get()));
    datasets_[gpu_index].reset(new Dataset(num_data_in_gpu));
    datasets_[gpu_index]->CopySubrow(train_data_, all_data_indices.data() + data_start, num_data_in_gpu, true);
    //datasets_[gpu_index]->MoveToCUDADevice(config_->gpu_device_id);
    tree_learners_.back()->Init(datasets_[gpu_index].get(), is_constant_hessian);
  }

  cuda_smaller_leaf_splits_.reset(new CUDALeafSplits(num_data_));
  cuda_smaller_leaf_splits_->Init(config_->gpu_use_discretized_grad);
  cuda_larger_leaf_splits_.reset(new CUDALeafSplits(num_data_));
  cuda_larger_leaf_splits_->Init(config_->gpu_use_discretized_grad);

  leaf_best_split_feature_.resize(config_->num_leaves, -1);
  leaf_best_split_threshold_.resize(config_->num_leaves, 0);
  leaf_best_split_default_left_.resize(config_->num_leaves, 0);
  leaf_best_split_gain_.resize(config_->num_leaves, kMinScore);
  leaf_num_data_.resize(config_->num_leaves, 0);
  leaf_data_start_.resize(config_->num_leaves, 0);
  leaf_sum_hessians_.resize(config_->num_leaves, 0.0f);

  nccl_communicators_.resize(config_->num_gpu);
  ncclUniqueId nccl_unique_id;
  ncclGetUniqueId(&nccl_unique_id);
  ncclGroupStart();
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    cudaSetDevice(gpu_index);
    ncclCommInitRank(&nccl_communicators_[gpu_index], config_->num_gpu, nccl_unique_id, gpu_index);
  }
  ncclGroupEnd();

  cuda_streams_.resize(config_->num_gpu);
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_streams_[gpu_index]));
  }

  const auto& feature_hist_offsets = share_state_->feature_hist_offsets();
  num_total_bin_ = feature_hist_offsets.empty() ? 0 : static_cast<int>(feature_hist_offsets.back());

  is_feature_used_by_tree_per_gpu_.resize(config_->num_gpu);
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    is_feature_used_by_tree_per_gpu_[gpu_index].resize(num_features_, 0);
  }

  CUDASUCCESS_OR_FATAL(cudaSetDevice(0));
  leaf_splits_buffer_.Resize(config_->num_gpu);
  best_split_info_buffer_.Resize(12 * config_->num_gpu);
  CUDASUCCESS_OR_FATAL(cudaMallocHost(&host_split_info_buffer_, 12 * sizeof(int)));
}

void CUDAExpTreeLearner::ResetTrainingData(const Dataset* train_data, bool is_constant_hessian) {
  SerialTreeLearner::ResetTrainingData(train_data, is_constant_hessian);
  std::vector<data_size_t> all_data_indices(num_data_, 0);
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (data_size_t data_index = 0; data_index < train_data_->num_data(); ++data_index) {
    all_data_indices[data_index] = data_index;
  }
  const data_size_t num_data_per_gpu_ = (num_data_ + config_->num_gpu - 1) / config_->num_gpu;
  #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
    const data_size_t data_start = gpu_index * num_data_per_gpu_;
    const data_size_t data_end = std::min(data_start + num_data_per_gpu_, num_data_);
    const data_size_t num_data_in_gpu = data_end - data_start;
    datasets_[gpu_index].reset(new Dataset(num_data_in_gpu));
    datasets_[gpu_index]->CopySubrow(train_data_, all_data_indices.data() + data_start, num_data_in_gpu, true);
    //datasets_[gpu_index]->MoveToCUDADevice(config_->gpu_device_id);
    tree_learners_.back()->ResetTrainingData(datasets_[gpu_index].get(), is_constant_hessian);
  }
}

void CUDAExpTreeLearner::NCCLReduceRootNodeInformation() {
  const size_t send_count = (sizeof(int) + sizeof(double) + sizeof(double) + sizeof(int64_t) + sizeof(data_size_t)) / sizeof(int);
  CUDASUCCESS_OR_FATAL(cudaSetDevice(0));
  ncclRecv(leaf_splits_buffer_.RawData(), send_count, ncclInt32, 0, nccl_communicators_[0], cuda_streams_[0]);
  #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
    ncclSend(tree_learners_[gpu_index]->GetSmallerLeafSplitsStruct(),
      send_count, ncclInt32, gpu_index, nccl_communicators_[gpu_index], cuda_streams_[gpu_index]);
  }
  LaunchReduceRootNodeInformationKernel(cuda_smaller_leaf_splits_->GetCUDAStructRef());
}

void CUDAExpTreeLearner::BeforeTrainWithGrad(const score_t* gradients, const score_t* hessians, const std::vector<int8_t>& is_feature_used_by_tree) {
  gradients_ = gradients;
  hessians_ = hessians;

  int num_used_features = 0;
  for (size_t i = 0; i < is_feature_used_by_tree.size(); ++i) {
    if (is_feature_used_by_tree[i]) {
      ++num_used_features;
    }
  }
  const int num_used_features_per_gpu = (num_used_features + config_->num_gpu - 1) / config_->num_gpu;
  int cur_num_used_features = 0;
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
      is_feature_used_by_tree_per_gpu_[gpu_index][feature_index] = 0;
    }
  }
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (is_feature_used_by_tree[feature_index]) {
      const int gpu_index = cur_num_used_features / num_used_features_per_gpu;
      is_feature_used_by_tree_per_gpu_[gpu_index][feature_index] = 1;
      ++cur_num_used_features;
    }
  }

  #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
    const data_size_t data_start = num_data_per_gpu_ * gpu_index;
    tree_learners_[gpu_index]->BeforeTrainWithGrad(
      gradients_ + data_start,
      hessians_ + data_start,
      is_feature_used_by_tree_per_gpu_[gpu_index]);
  }

  // reduce root node information
  NCCLReduceRootNodeInformation();

  leaf_num_data_[0] = train_data_->num_data();
  col_sampler_.ResetByTree();
  leaf_data_start_[0] = 0;
  smaller_leaf_index_ = 0;
  larger_leaf_index_ = -1;
}

void CUDAExpTreeLearner::NCCLReduceHistograms() {
  if (config_->gpu_use_discretized_grad) {
    #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
    for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      ncclAllReduce(
        reinterpret_cast<const int64_t*>(tree_learners_[gpu_index]->smaller_leaf_hist()),
        reinterpret_cast<int64_t*>(tree_learners_[gpu_index]->smaller_leaf_hist()),
        static_cast<size_t>(num_total_bin_),
        ncclInt64,
        ncclSum,
        nccl_communicators_[gpu_index],
        cuda_streams_[gpu_index]);
    }
  } else {
    #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
    for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      ncclAllReduce(
        tree_learners_[gpu_index]->smaller_leaf_hist(),
        tree_learners_[gpu_index]->smaller_leaf_hist(),
        static_cast<size_t>(num_total_bin_) * 2,
        ncclFloat64,
        ncclSum,
        nccl_communicators_[gpu_index],
        cuda_streams_[gpu_index]);
    }
  }
}

void CUDAExpTreeLearner::NCCLReduceBestSplitsForLeaf(CUDATree* tree) {
  CUDASUCCESS_OR_FATAL(cudaSetDevice(0));
  ncclRecv(leaf_splits_buffer_.RawData(), 10, ncclInt32, 0, nccl_communicators_[0], cuda_streams_[0]);
  #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
    ncclSend(tree_learners_[gpu_index]->cuda_best_split_info_buffer(),
      10, ncclInt32, gpu_index, nccl_communicators_[gpu_index], cuda_streams_[gpu_index]);
  }
  LaunchReduceBestSplitsForLeafKernel();
  CopyFromCUDADeviceToHost<int>(
    host_split_info_buffer_,
    best_split_info_buffer_.RawData(),
    12, __FILE__, __LINE__);
  const int smaller_leaf_best_gpu_index = host_split_info_buffer_[10];
  const int larger_leaf_best_gpu_index = host_split_info_buffer_[11];
  const size_t count = (sizeof(int8_t) + sizeof(int) + sizeof(double) + sizeof(int) + sizeof(uint32_t) + sizeof(int8_t) +
    sizeof(double) + sizeof(double) + sizeof(int64_t) + sizeof(data_size_t) + sizeof(double) + sizeof(double) +
    sizeof(double) + sizeof(double) + sizeof(int64_t) + sizeof(data_size_t) + sizeof(double) + sizeof(double) +
    sizeof(int) + sizeof(uint32_t*) + sizeof(int*)) / sizeof(int8_t);

  // synchronize best split info across devices
  if (smaller_leaf_best_gpu_index >= 0) {
    #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
    for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      if (gpu_index == smaller_leaf_best_gpu_index) {
        CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
        // TODO(shiyu1994): categorical features are not supported yet
        ncclSend(
          tree_learners_[gpu_index]->cuda_leaf_best_split_info() + smaller_leaf_index_,
          count, ncclInt8, gpu_index, nccl_communicators_[gpu_index], cuda_streams_[gpu_index]);
      } else {
        CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
        ncclRecv(
          tree_learners_[gpu_index]->cuda_leaf_best_split_info() + smaller_leaf_index_,
          count, ncclInt8, gpu_index, nccl_communicators_[gpu_index], cuda_streams_[gpu_index]);
      }
    }
  }
  if (larger_leaf_index_ >= 0 && larger_leaf_best_gpu_index >= 0) {
    #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
    for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      if (gpu_index == larger_leaf_best_gpu_index) {
        CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
        // TODO(shiyu1994): categorical features are not supported yet
        ncclSend(
          tree_learners_[gpu_index]->cuda_leaf_best_split_info() + larger_leaf_index_,
          count, ncclInt8, gpu_index, nccl_communicators_[gpu_index], cuda_streams_[gpu_index]);
      } else {
        CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
        ncclRecv(
          tree_learners_[gpu_index]->cuda_leaf_best_split_info() + larger_leaf_index_,
          count, ncclInt8, gpu_index, nccl_communicators_[gpu_index], cuda_streams_[gpu_index]);
      }
    }
  }

  const double* gain_buffer = reinterpret_cast<const double*>(host_split_info_buffer_ + 6);
  leaf_best_split_feature_[smaller_leaf_index_] = host_split_info_buffer_[0];
  leaf_best_split_threshold_[smaller_leaf_index_] = static_cast<uint32_t>(host_split_info_buffer_[1]);
  leaf_best_split_default_left_[smaller_leaf_index_] = static_cast<uint8_t>(host_split_info_buffer_[2]);
  leaf_best_split_gain_[smaller_leaf_index_] = gain_buffer[0];
  if (larger_leaf_index_ >= 0) {
    leaf_best_split_feature_[larger_leaf_index_] = host_split_info_buffer_[3];
    leaf_best_split_threshold_[larger_leaf_index_] = static_cast<uint32_t>(host_split_info_buffer_[4]);
    leaf_best_split_default_left_[larger_leaf_index_] = static_cast<uint8_t>(host_split_info_buffer_[5]);
    leaf_best_split_gain_[larger_leaf_index_] = gain_buffer[1];
  }

  best_leaf_index_ = -1;
  double best_gain = kMinScore;
  for (int leaf_index = 0; leaf_index < tree->num_leaves(); ++leaf_index) {
    if (leaf_best_split_gain_[leaf_index] > best_gain) {
      best_gain = leaf_best_split_gain_[leaf_index];
      best_leaf_index_ = leaf_index;
    }
  }
}

void CUDAExpTreeLearner::BroadCastBestSplit() {
  if (larger_leaf_index_ >= 0) {
    for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      tree_learners_[gpu_index]->SetBestSplit(
        best_leaf_index_,
        leaf_best_split_feature_[smaller_leaf_index_],
        leaf_best_split_threshold_[smaller_leaf_index_],
        leaf_best_split_default_left_[smaller_leaf_index_],
        leaf_best_split_gain_[smaller_leaf_index_],
        leaf_best_split_feature_[larger_leaf_index_],
        leaf_best_split_threshold_[larger_leaf_index_],
        leaf_best_split_default_left_[larger_leaf_index_],
        leaf_best_split_gain_[larger_leaf_index_]);
    }
  } else {
    for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      tree_learners_[gpu_index]->SetBestSplit(
        best_leaf_index_,
        leaf_best_split_feature_[smaller_leaf_index_],
        leaf_best_split_threshold_[smaller_leaf_index_],
        leaf_best_split_default_left_[smaller_leaf_index_],
        leaf_best_split_gain_[smaller_leaf_index_],
        -1,
        0,
        0,
        kMinScore);
    }
  }
}

Tree* CUDAExpTreeLearner::Train(const score_t* gradients, const score_t* hessians, bool /*is_first_tree*/) {
  BeforeTrainWithGrad(gradients, hessians, col_sampler_.is_feature_used_bytree());
  const bool track_branch_features = !(config_->interaction_constraints_vector.empty());
  std::unique_ptr<CUDATree> tree(new CUDATree(config_->num_leaves, track_branch_features,
    config_->linear_tree, config_->gpu_device_id, has_categorical_feature_));
  for (int i = 0; i < config_->num_leaves - 1; ++i) {
    const data_size_t num_data_in_smaller_leaf = leaf_num_data_[smaller_leaf_index_];
    const data_size_t num_data_in_larger_leaf = larger_leaf_index_ < 0 ? 0 : leaf_num_data_[larger_leaf_index_];
    const double sum_hessians_in_smaller_leaf = leaf_sum_hessians_[smaller_leaf_index_];
    const double sum_hessians_in_larger_leaf = larger_leaf_index_ < 0 ? 0 : leaf_sum_hessians_[larger_leaf_index_];
    #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
    for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
      tree_learners_[gpu_index]->CUDAConstructHistograms(
        num_data_in_smaller_leaf,
        num_data_in_larger_leaf,
        sum_hessians_in_smaller_leaf,
        sum_hessians_in_larger_leaf);
      tree_learners_[gpu_index]->CUDASubtractHistograms(
        cuda_smaller_leaf_splits_->GetCUDAStruct(),
        cuda_larger_leaf_splits_->GetCUDAStruct());
      SynchronizeCUDADevice(__FILE__, __LINE__);
    }
    NCCLReduceHistograms();
    #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
    for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
      tree_learners_[gpu_index]->CUDAFindBestSplitsForLeaf(
        num_data_in_smaller_leaf,
        num_data_in_larger_leaf,
        sum_hessians_in_smaller_leaf,
        sum_hessians_in_larger_leaf,
        cuda_smaller_leaf_splits_->GetCUDAStruct(),
        cuda_larger_leaf_splits_->GetCUDAStruct());
      SynchronizeCUDADevice(__FILE__, __LINE__);
    }
    NCCLReduceBestSplitsForLeaf(tree.get());

    if (best_leaf_index_ == -1) {
      Log::Warning("No further splits with positive gain, training stopped with %d leaves.", (i + 1));
      break;
    }

    BroadCastBestSplit();
  }
}

}  // namespace LightGBM

#endif  // USE_CUDA
