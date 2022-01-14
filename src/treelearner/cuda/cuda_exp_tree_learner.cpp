/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_exp_tree_learner.hpp"

#include <fstream>

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
  num_data_per_gpu_ = (num_data_ + config_->num_gpu - 1) / config_->num_gpu;
  tree_learners_.resize(config_->num_gpu);
  datasets_.resize(config_->num_gpu);
  configs_.resize(config_->num_gpu);
  per_gpu_gradients_.resize(config_->num_gpu);
  per_gpu_hessians_.resize(config_->num_gpu);
  smaller_leaf_splits_buffer_.resize(config_->num_gpu);
  larger_leaf_splits_buffer_.resize(config_->num_gpu);
  per_gpu_scores_.resize(config_->num_gpu);
  best_split_info_buffer_.resize(config_->num_gpu);
  #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
    const data_size_t data_start = gpu_index * num_data_per_gpu_;
    const data_size_t data_end = std::min(data_start + num_data_per_gpu_, num_data_);
    const data_size_t num_data_in_gpu = data_end - data_start;
    per_gpu_gradients_[gpu_index].reset(new CUDAVector<score_t>(static_cast<size_t>(num_data_in_gpu)));
    per_gpu_hessians_[gpu_index].reset(new CUDAVector<score_t>(static_cast<size_t>(num_data_in_gpu)));
    configs_[gpu_index].reset(new Config(*config_));
    configs_[gpu_index]->gpu_device_id = gpu_index;
    tree_learners_[gpu_index].reset(new CUDASingleGPUTreeLearner(configs_[gpu_index].get()));
    datasets_[gpu_index].reset(new Dataset(num_data_in_gpu));
    datasets_[gpu_index]->CopyFeatureMapperFrom(train_data_);
    datasets_[gpu_index]->ReSize(num_data_in_gpu);
    datasets_[gpu_index]->CopySubrow(train_data_, all_data_indices.data() + data_start, num_data_in_gpu, true, data_start, data_end, gpu_index);
    tree_learners_[gpu_index]->Init(datasets_[gpu_index].get(), is_constant_hessian, gpu_index);
    smaller_leaf_splits_buffer_[gpu_index].Resize(config_->num_gpu);
    larger_leaf_splits_buffer_[gpu_index].Resize(config_->num_gpu);
    best_split_info_buffer_[gpu_index].reset(new CUDAVector<int>());
    best_split_info_buffer_[gpu_index]->Resize(12 * config_->num_gpu);
    per_gpu_scores_[gpu_index].reset(new CUDAVector<double>());
    per_gpu_scores_[gpu_index]->Resize(num_data_in_gpu);
    SynchronizeCUDADevice(__FILE__, __LINE__);
  }

  CUDASUCCESS_OR_FATAL(cudaSetDevice(0));

  leaf_best_split_feature_.resize(config_->num_leaves, -1);
  leaf_best_split_threshold_.resize(config_->num_leaves, 0);
  leaf_best_split_default_left_.resize(config_->num_leaves, 0);
  leaf_best_split_gain_.resize(config_->num_leaves, kMinScore);
  leaf_num_data_.resize(config_->num_leaves, 0);
  leaf_data_start_.resize(config_->num_leaves, 0);
  leaf_sum_hessians_.resize(config_->num_leaves, 0.0f);

  nccl_communicators_.resize(config_->num_gpu);
  ncclUniqueId nccl_unique_id;
  NCCLCHECK(ncclGetUniqueId(&nccl_unique_id));
  NCCLCHECK(ncclGroupStart());
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
    NCCLCHECK(ncclCommInitRank(&nccl_communicators_[gpu_index], config_->num_gpu, nccl_unique_id, gpu_index));
  }
  NCCLCHECK(ncclGroupEnd());

  cuda_send_streams_.resize(config_->num_gpu);
  cuda_recv_streams_.resize(config_->num_gpu);
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
    CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_send_streams_[gpu_index]));
    CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_recv_streams_[gpu_index]));
  }

  CUDASUCCESS_OR_FATAL(cudaSetDevice(0));
  const auto& feature_hist_offsets = share_state_->feature_hist_offsets();
  num_total_bin_ = feature_hist_offsets.empty() ? 0 : static_cast<int>(feature_hist_offsets.back());

  is_feature_used_by_tree_per_gpu_.resize(config_->num_gpu);
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    is_feature_used_by_tree_per_gpu_[gpu_index].resize(num_features_, 0);
  }

  CUDASUCCESS_OR_FATAL(cudaMallocHost(&host_split_info_buffer_, 12 * sizeof(int)));
  cuda_root_sum_hessians_.Resize(1);

  leaf_to_hist_index_map_.resize(config_->num_leaves, -1);

  cur_iter_ = 0;
}

void CUDAExpTreeLearner::ResetTrainingData(const Dataset* train_data, bool is_constant_hessian) {
  SerialTreeLearner::ResetTrainingData(train_data, is_constant_hessian);
  std::vector<data_size_t> all_data_indices(num_data_, 0);
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (data_size_t data_index = 0; data_index < train_data_->num_data(); ++data_index) {
    all_data_indices[data_index] = data_index;
  }
  num_data_per_gpu_ = (num_data_ + config_->num_gpu - 1) / config_->num_gpu;
  #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
    const data_size_t data_start = gpu_index * num_data_per_gpu_;
    const data_size_t data_end = std::min(data_start + num_data_per_gpu_, num_data_);
    const data_size_t num_data_in_gpu = data_end - data_start;
    datasets_[gpu_index].reset(new Dataset(num_data_in_gpu));
    datasets_[gpu_index]->CopySubrow(train_data_, all_data_indices.data() + data_start, num_data_in_gpu, true);
    tree_learners_.back()->ResetTrainingData(datasets_[gpu_index].get(), is_constant_hessian);
  }
}

void CUDAExpTreeLearner::BeforeTrainWithGrad(const score_t* gradients, const score_t* hessians, const std::vector<int8_t>& is_feature_used_by_tree, ncclComm_t* /*comm*/, cudaStream_t* /*stream*/) {
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
    const int gpu_device_id = config_->gpu_device_id >= 0 ? config_->gpu_device_id : 0;
    const data_size_t data_start = num_data_per_gpu_ * gpu_index;
    const data_size_t data_end = std::min(data_start + num_data_per_gpu_, num_data_);
    const data_size_t num_data_in_gpu = data_end - data_start;
    CopyPeerFromCUDADeviceToCUDADevice<score_t>(
      per_gpu_gradients_[gpu_index]->RawData(),
      gpu_index,
      gradients_ + data_start,
      gpu_device_id,
      static_cast<size_t>(num_data_in_gpu),
      __FILE__,
      __LINE__);
    CopyPeerFromCUDADeviceToCUDADevice<score_t>(
      per_gpu_hessians_[gpu_index]->RawData(),
      gpu_index,
      hessians_ + data_start,
      gpu_device_id,
      static_cast<size_t>(num_data_in_gpu),
      __FILE__,
      __LINE__);
    tree_learners_[gpu_index]->BeforeTrainWithGrad(
      per_gpu_gradients_[gpu_index]->RawData(),
      per_gpu_hessians_[gpu_index]->RawData(),
      is_feature_used_by_tree_per_gpu_[gpu_index],
      &nccl_communicators_[gpu_index],
      &cuda_send_streams_[gpu_index]);
  }

  leaf_to_hist_index_map_[0] = 0;
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int leaf_index = 1; leaf_index < config_->num_leaves; ++leaf_index) {
    leaf_to_hist_index_map_[leaf_index] = -1;
  }

  leaf_num_data_[0] = train_data_->num_data();
  leaf_sum_hessians_[0] = tree_learners_[0]->leaf_sum_hessians(0);
  col_sampler_.ResetByTree();
  leaf_data_start_[0] = 0;
  smaller_leaf_index_ = 0;
  larger_leaf_index_ = -1;
}

void CUDAExpTreeLearner::NCCLReduceHistograms() {
  if (config_->gpu_use_discretized_grad) {
    NCCLCHECK(ncclGroupStart());
    for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {  
      hist_t* smaller_leaf_hist_pointer = tree_learners_[gpu_index]->cuda_hist_pointer() +
        leaf_to_hist_index_map_[smaller_leaf_index_] * num_total_bin_;
      NCCLCHECK(ncclAllReduce(
        reinterpret_cast<const int64_t*>(smaller_leaf_hist_pointer),
        reinterpret_cast<int64_t*>(smaller_leaf_hist_pointer),
        static_cast<size_t>(num_total_bin_),
        ncclInt64,
        ncclSum,
        nccl_communicators_[gpu_index],
        cuda_send_streams_[gpu_index]));
    }
    NCCLCHECK(ncclGroupEnd());
  } else {
    NCCLCHECK(ncclGroupStart());
    for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      hist_t* smaller_leaf_hist_pointer = tree_learners_[gpu_index]->cuda_hist_pointer() +
        leaf_to_hist_index_map_[smaller_leaf_index_] * num_total_bin_ * 2;
      NCCLCHECK(ncclAllReduce(
        smaller_leaf_hist_pointer,
        smaller_leaf_hist_pointer,
        static_cast<size_t>(num_total_bin_) * 2,
        ncclFloat64,
        ncclSum,
        nccl_communicators_[gpu_index],
        cuda_send_streams_[gpu_index]));
    }
    NCCLCHECK(ncclGroupEnd());
  }
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
    CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(cuda_send_streams_[gpu_index]));
  }
}

void CUDAExpTreeLearner::NCCLReduceBestSplitsForLeaf(CUDATree* tree) {
  //NCCLCHECK(ncclGroupStart());
  #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
    NCCLCHECK(ncclAllGather(tree_learners_[gpu_index]->cuda_best_split_info_buffer(),
      best_split_info_buffer_[gpu_index]->RawData(),
      10, ncclInt32, nccl_communicators_[gpu_index], cuda_send_streams_[gpu_index]));
    CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(cuda_send_streams_[gpu_index]));
  }
  //NCCLCHECK(ncclGroupEnd());
  /*for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
    CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(cuda_send_streams_[gpu_index]));
  }*/
  CUDASUCCESS_OR_FATAL(cudaSetDevice(0));
  global_timer.Start("CUDAExpTreeLearner::LaunchReduceBestSplitsForLeafKernel");
  LaunchReduceBestSplitsForLeafKernel();
  CopyFromCUDADeviceToHost<int>(
    host_split_info_buffer_,
    best_split_info_buffer_[0]->RawData(),
    12, __FILE__, __LINE__);
  global_timer.Stop("CUDAExpTreeLearner::LaunchReduceBestSplitsForLeafKernel");
  const int smaller_leaf_best_gpu_index = host_split_info_buffer_[10];
  const int larger_leaf_best_gpu_index = host_split_info_buffer_[11];
  const size_t count = sizeof(CUDASplitInfo) / sizeof(int8_t);

  // synchronize best split info across devices
  if (smaller_leaf_best_gpu_index >= 0) {
    //NCCLCHECK(ncclGroupStart());
    #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
    for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
      NCCLCHECK(ncclBroadcast(tree_learners_[gpu_index]->cuda_leaf_best_split_info() + smaller_leaf_index_,
        tree_learners_[gpu_index]->cuda_leaf_best_split_info() + smaller_leaf_index_,
        count, ncclInt8, smaller_leaf_best_gpu_index, nccl_communicators_[gpu_index], cuda_send_streams_[gpu_index]));
      CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(cuda_send_streams_[gpu_index]));
    }
    //NCCLCHECK(ncclGroupEnd());

    /*for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
      CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(cuda_send_streams_[gpu_index]));
    }*/
  }

  if (larger_leaf_index_ >= 0 && larger_leaf_best_gpu_index >= 0) {
    //NCCLCHECK(ncclGroupStart());
    #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
    for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
      NCCLCHECK(ncclBroadcast(tree_learners_[gpu_index]->cuda_leaf_best_split_info() + larger_leaf_index_,
        tree_learners_[gpu_index]->cuda_leaf_best_split_info() + larger_leaf_index_,
        count, ncclInt8, larger_leaf_best_gpu_index, nccl_communicators_[gpu_index], cuda_send_streams_[gpu_index]));
      CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(cuda_send_streams_[gpu_index]));
    }
    //NCCLCHECK(ncclGroupEnd());

    /*for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
      CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(cuda_send_streams_[gpu_index]));
    }*/
  }

  const double* gain_buffer = reinterpret_cast<const double*>(host_split_info_buffer_ + 6);
  leaf_best_split_feature_[smaller_leaf_index_] = host_split_info_buffer_[0];
  leaf_best_split_threshold_[smaller_leaf_index_] = static_cast<uint32_t>(host_split_info_buffer_[1]);
  leaf_best_split_default_left_[smaller_leaf_index_] = static_cast<uint8_t>(host_split_info_buffer_[2]);
  leaf_best_split_gain_[smaller_leaf_index_] = gain_buffer[0];
  /*Log::Warning("leaf best split real feature = %d", train_data_->RealFeatureIndex(leaf_best_split_feature_[smaller_leaf_index_]));
  Log::Warning("leaf_best_split_feature[%d] = %d", smaller_leaf_index_, leaf_best_split_feature_[smaller_leaf_index_]);
  Log::Warning("leaf_best_split_threshold[%d] = %d", smaller_leaf_index_, leaf_best_split_threshold_[smaller_leaf_index_]);
  Log::Warning("leaf_best_split_gain[%d] = %f", smaller_leaf_index_, leaf_best_split_gain_[smaller_leaf_index_]);*/
  if (larger_leaf_index_ >= 0) {
    leaf_best_split_feature_[larger_leaf_index_] = host_split_info_buffer_[3];
    leaf_best_split_threshold_[larger_leaf_index_] = static_cast<uint32_t>(host_split_info_buffer_[4]);
    leaf_best_split_default_left_[larger_leaf_index_] = static_cast<uint8_t>(host_split_info_buffer_[5]);
    leaf_best_split_gain_[larger_leaf_index_] = gain_buffer[1];
    /*Log::Warning("leaf best split real feature = %d", train_data_->RealFeatureIndex(leaf_best_split_feature_[larger_leaf_index_]));
    Log::Warning("leaf_best_split_feature[%d] = %d", larger_leaf_index_, leaf_best_split_feature_[larger_leaf_index_]);
    Log::Warning("leaf_best_split_threshold[%d] = %d", larger_leaf_index_, leaf_best_split_threshold_[larger_leaf_index_]);
    Log::Warning("leaf_best_split_gain[%d] = %f", larger_leaf_index_, leaf_best_split_gain_[larger_leaf_index_]);*/
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

void CUDAExpTreeLearner::ReduceLeafInformationAfterSplit(const int best_leaf_index, const int right_leaf_index) {
  leaf_num_data_[best_leaf_index] = 0;
  leaf_num_data_[right_leaf_index] = 0;
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    leaf_num_data_[best_leaf_index] += tree_learners_[gpu_index]->leaf_num_data(best_leaf_index);
    leaf_num_data_[right_leaf_index] += tree_learners_[gpu_index]->leaf_num_data(right_leaf_index);
  }
  leaf_sum_hessians_[best_leaf_index] = tree_learners_[0]->leaf_sum_hessians(best_leaf_index);
  leaf_sum_hessians_[right_leaf_index] = tree_learners_[0]->leaf_sum_hessians(right_leaf_index);
  //Log::Warning("best_leaf_index = %d, leaf_num_data = %d, leaf_sum_hessians = %f", best_leaf_index, leaf_num_data_[best_leaf_index], leaf_sum_hessians_[best_leaf_index]);
  //Log::Warning("right_leaf_index = %d, leaf_num_data = %d, leaf_sum_hessians = %f", right_leaf_index, leaf_num_data_[right_leaf_index], leaf_sum_hessians_[right_leaf_index]);
}

Tree* CUDAExpTreeLearner::Train(const score_t* gradients, const score_t* hessians, bool /*is_first_tree*/) {
  global_timer.Start("CUDAExpTreeLearner::BeforeTrainWithGrad");
  BeforeTrainWithGrad(gradients, hessians, col_sampler_.is_feature_used_bytree());
  global_timer.Stop("CUDAExpTreeLearner::BeforeTrainWithGrad");
  const bool track_branch_features = !(config_->interaction_constraints_vector.empty());
  std::unique_ptr<CUDATree> tree(new CUDATree(config_->num_leaves, track_branch_features,
    config_->linear_tree, config_->gpu_device_id, has_categorical_feature_));
  for (int i = 0; i < config_->num_leaves - 1; ++i) {
    const data_size_t num_data_in_smaller_leaf = leaf_num_data_[smaller_leaf_index_];
    const data_size_t num_data_in_larger_leaf = larger_leaf_index_ < 0 ? 0 : leaf_num_data_[larger_leaf_index_];
    const double sum_hessians_in_smaller_leaf = leaf_sum_hessians_[smaller_leaf_index_];
    const double sum_hessians_in_larger_leaf = larger_leaf_index_ < 0 ? 0 : leaf_sum_hessians_[larger_leaf_index_];

  global_timer.Start("CUDAExpTreeLearner::CUDAConstructHistograms");
  #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    global_timer.Start("CUDAExpTreeLearner::CUDAConstructHistograms::cudaSetDevice");
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
    global_timer.Stop("CUDAExpTreeLearner::CUDAConstructHistograms::cudaSetDevice");
    tree_learners_[gpu_index]->CUDAConstructHistograms(
      num_data_in_smaller_leaf,
      num_data_in_larger_leaf,
      sum_hessians_in_smaller_leaf,
      sum_hessians_in_larger_leaf);
    //SynchronizeCUDADevice(__FILE__, __LINE__);
  }
  global_timer.Stop("CUDAExpTreeLearner::CUDAConstructHistograms");
  global_timer.Start("CUDAExpTreeLearner::NCCLReduceHistograms");
  NCCLReduceHistograms();
  global_timer.Stop("CUDAExpTreeLearner::NCCLReduceHistograms");
  global_timer.Start("CUDAExpTreeLearner::CUDASubtractHistograms");
  #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
    tree_learners_[gpu_index]->CUDASubtractHistograms();
    //SynchronizeCUDADevice(__FILE__, __LINE__);
  }
  global_timer.Stop("CUDAExpTreeLearner::CUDASubtractHistograms");
    /*{
      if (smaller_leaf_index_ == 0 && larger_leaf_index_ == -1) {
        std::ofstream fout("cuda_gradients.txt");
        CUDASUCCESS_OR_FATAL(cudaSetDevice(0));
        std::vector<hist_t> host_hist(2 * num_total_bin_, 0.0f);
        CopyFromCUDADeviceToHost<hist_t>(host_hist.data(), tree_learners_[0]->cuda_hist_pointer(), 2 * num_total_bin_, __FILE__, __LINE__);
        for (int real_feature_index = 0; real_feature_index < train_data_->num_total_features(); ++real_feature_index) {
          const int inner_feature_index = train_data_->InnerFeatureIndex(real_feature_index);
          if (inner_feature_index >= 0) {
            const auto bin_start = share_state_->feature_hist_offsets()[inner_feature_index];
            const auto bin_end = share_state_->feature_hist_offsets()[inner_feature_index + 1];
            fout << "feature " << real_feature_index << " inner feature " << train_data_->InnerFeatureIndex(real_feature_index) << " offset " << (train_data_->FeatureBinMapper(inner_feature_index)->GetMostFreqBin() == 0) << std::endl;
            for (auto bin = bin_start; bin < bin_end; ++bin) {
              fout << host_hist[2 * bin] << "," << host_hist[2 * bin + 1] << std::endl;
            }
          }
        }
      }
    }*/
  global_timer.Start("CUDAExpTreeLearner::CUDAFindBestSplitsForLeaf");
    #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
    for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
      tree_learners_[gpu_index]->CUDAFindBestSplitsForLeaf(
        num_data_in_smaller_leaf,
        num_data_in_larger_leaf,
        sum_hessians_in_smaller_leaf,
        sum_hessians_in_larger_leaf);
      //SynchronizeCUDADevice(__FILE__, __LINE__);
    }
  global_timer.Stop("CUDAExpTreeLearner::CUDAFindBestSplitsForLeaf");
  global_timer.Start("CUDAExpTreeLearner::NCCLReduceBestSplitsForLeaf");
    NCCLReduceBestSplitsForLeaf(tree.get());
  global_timer.Stop("CUDAExpTreeLearner::NCCLReduceBestSplitsForLeaf");

    if (best_leaf_index_ == -1) {
      Log::Warning("No further splits with positive gain, training stopped with %d leaves.", (i + 1));
      break;
    }

    BroadCastBestSplit();

    // TODO(shiyu1994): categorical features are not supported yet
    CUDASUCCESS_OR_FATAL(cudaSetDevice(0));
    const int right_leaf_index = tree->Split(
      best_leaf_index_,
      train_data_->RealFeatureIndex(leaf_best_split_feature_[best_leaf_index_]),
      train_data_->RealThreshold(leaf_best_split_feature_[best_leaf_index_],
      leaf_best_split_threshold_[best_leaf_index_]),
      train_data_->FeatureBinMapper(leaf_best_split_feature_[best_leaf_index_])->missing_type(),
      tree_learners_[0]->cuda_leaf_best_split_info() + best_leaf_index_);

  global_timer.Start("CUDAExpTreeLearner::CUDASplit");
    #pragma omp parallel for schedule(static, 1) num_threads(config_->num_gpu)
    for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
      tree_learners_[gpu_index]->CUDASplit(right_leaf_index, &nccl_communicators_[gpu_index]);
      //SynchronizeCUDADevice(__FILE__, __LINE__);
    }
  global_timer.Stop("CUDAExpTreeLearner::CUDASplit");

    ReduceLeafInformationAfterSplit(best_leaf_index_, right_leaf_index);

    smaller_leaf_index_ = (leaf_num_data_[best_leaf_index_] < leaf_num_data_[right_leaf_index] ? best_leaf_index_ : right_leaf_index);
    larger_leaf_index_ = (smaller_leaf_index_ == best_leaf_index_ ? right_leaf_index : best_leaf_index_);

    const int best_leaf_hist_index = leaf_to_hist_index_map_[best_leaf_index_];
    leaf_to_hist_index_map_[smaller_leaf_index_] = right_leaf_index;
    leaf_to_hist_index_map_[larger_leaf_index_] = best_leaf_hist_index;

    for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      tree_learners_[gpu_index]->SetSmallerAndLargerLeaves(smaller_leaf_index_, larger_leaf_index_);
    }
  }
  CUDASUCCESS_OR_FATAL(cudaSetDevice(0));
  SynchronizeCUDADevice(__FILE__, __LINE__);
  return tree.release();
}

void CUDAExpTreeLearner::AddPredictionToScore(const Tree* tree, double* out_score) const {
  global_timer.Start("CUDAExpTreeLearner::AddPredictionToScore");
  SynchronizeCUDADevice(__FILE__, __LINE__);
  if (cur_iter_ == 0) {
    for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
      const data_size_t data_start = gpu_index * num_data_per_gpu_;
      const data_size_t data_end = std::min(data_start + num_data_per_gpu_, num_data_);
      const data_size_t num_data_in_gpu = data_end - data_start;
      const int gpu_device_id = config_->gpu_device_id >= 0 ? config_->gpu_device_id : 0;
      CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
      CopyPeerFromCUDADeviceToCUDADevice<double>(
        per_gpu_scores_[gpu_index]->RawData(),
        gpu_index,
        out_score + data_start,
        gpu_device_id,
        num_data_in_gpu,
        __FILE__,
        __LINE__);
    }
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  #pragma omp parallel for schedule(static) num_threads(config_->num_gpu)
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(gpu_index));
    tree_learners_[gpu_index]->AddPredictionToScore(per_gpu_scores_[gpu_index]->RawData(), tree->num_leaves(), config_->learning_rate);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  for (int gpu_index = 0; gpu_index < config_->num_gpu; ++gpu_index) {
    const data_size_t data_start = gpu_index * num_data_per_gpu_;
    const data_size_t data_end = std::min(data_start + num_data_per_gpu_, num_data_);
    const data_size_t num_data_in_gpu = data_end - data_start;
    CopyPeerFromCUDADeviceToCUDADevice<double>(
      out_score + data_start,
      0,
      per_gpu_scores_[gpu_index]->RawData(),
      gpu_index,
      num_data_in_gpu,
      __FILE__,
      __LINE__);
  }
  SynchronizeCUDADevice(__FILE__, __LINE__);
  global_timer.Stop("CUDAExpTreeLearner::AddPredictionToScore");
  ++cur_iter_;
}

}  // namespace LightGBM

#endif  // USE_CUDA
