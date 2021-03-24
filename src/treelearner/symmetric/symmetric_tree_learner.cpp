/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "symmetric_tree_learner.hpp"

namespace LightGBM {

SymmetricTreeLearner::SymmetricTreeLearner(const Config* config): 
SerialTreeLearner(config), symmetric_data_partition_(config),
max_depth_(config->max_depth), max_num_leaves_(1 << max_depth_),
num_threads_(OMP_NUM_THREADS()) {}

void SymmetricTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();

  GetShareStates(train_data_, is_constant_hessian, true);
  col_sampler_.SetTrainingData(train_data_);

  ordered_gradients_.resize(num_data_, 0.0f);
  ordered_hessians_.resize(num_data_, 0.0f);

  paired_leaf_indices_in_cur_level_.resize(max_num_leaves_);


  thread_best_inner_feature_index_cur_level_.resize(num_threads_, -1);
  thread_best_threshold_cur_level_.resize(num_threads_, -1);
  thread_best_gain_cur_level_.resize(num_threads_, kMinScore);
  thread_best_split_direction_.resize(num_threads_, 0);
  thread_leaf_in_level_should_be_split_.resize(num_threads_, 0);

  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    thread_leaf_in_level_should_be_split_[thread_id].resize(max_num_leaves_, 0);
  }

  num_leaves_in_cur_level_ = 1;

  leaf_ids_in_current_level_.resize(max_num_leaves_, -1);
}

Tree* SymmetricTreeLearner::Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) override {
  gradients_ = gradients;
  hessians_ = hessians;
  BeforeTrain();
  // TODO(shiyu1994) support interaction constraints and linear tree
  std::unique_ptr<Tree> tree(new Tree(max_num_leaves_, false, false));
  for (int depth = 0; depth < config_->max_depth; ++depth) {
    SetUpLevelInfo(depth);
    PrepareLevelHistograms();
    // construct and subtract
    symmetric_data_partition_.ConstructLevelHistograms(&level_feature_histograms_, train_data_);
    // find best splits
    FindBestLevelSplits();
    SplitLevel(tree.get());
  }
  return tree.release();
}

void SymmetricTreeLearner::PrepareLevelHistograms() {
  for (int i = 0; i < num_leaves_in_cur_level_; ++i) {
    const int leaf_id = leaf_ids_in_current_level_[i];
    const bool get = symmetric_histogram_pool_.Get(leaf_id, &level_feature_histograms_[i]);
    if (!get) {
      // TODO(shiyu1994): handle the case when the feature histogram cache is not enough
    }
  }
}

void SymmetricTreeLearner::BeforeTrain() {
  leaf_ids_in_current_level_[0] = 0;
  for (int i = 1; i < max_num_leaves_; ++i) {
    leaf_ids_in_current_level[i] = -1;
  }
  level_feature_histograms_.resize(max_num_leaves_, nullptr);
  
  // initalize leaf splits
  level_leaf_splits_.resize(max_num_leaves_, nullptr);
  level_leaf_splits_[0]->reset(new LeafSplits(num_data_, config_));
  level_leaf_splits_[0]->Init(gradients_, hessians_);

  paired_leaf_indices_in_cur_level_.emplace_back(1, 0);
}

void SymmetricTreeLearner::FindBestLevelSplits() {
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int inner_feature_index = 0; inner_feature_index < num_features_; ++inner_feature_index) {
    FindBestLevelSplitsForFeature(inner_feature_index);
  }
}

void SymmetricTreeLearner::SetUpLevelInfo(const int depth) {
  cur_depth_ = depth;

  best_inner_feature_index_cur_level_ = -1;
  best_threshold_cur_level_ = -1;
  best_gain_cur_level_ = kMinScore;
  //TODO(shiyu1994) add level-wise feature subsampling
}

void SymmetricTreeLearner::FindBestLevelSplitsForFeature(const int inner_feature_index, const int thread_id) {
  SymmetricHistogramPool::FindBestThresholdFromLevelHistograms(inner_feature_index,
    paired_leaf_indices_in_cur_level_,
    &thread_best_inner_feature_index_cur_level_[thread_id],
    &thread_best_threshold_cur_level_[thread_id],
    &thread_best_gain_cur_level_[thread_id],
    &thread_best_split_direction_cur_level_[thread_id],
    &thread_leaf_in_level_should_be_split_[thread_id]);
  best_gain_cur_level_ = kMinScore;
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    if (thread_best_gain_cur_level_[thread_id] > best_gain_cur_level_) {
      best_inner_feature_index_cur_level_ = thread_best_inner_feature_index_cur_level_[thread_id];
      best_threshold_cur_level_ = thread_best_threshold_cur_level_[thread_id];
      best_gain_cur_level_ = thread_best_gain_cur_level_[thread_id];
      best_split_direction_cur_level_ = thread_best_split_direction_cur_level_[thread_id];
      #pragma omp parallel for schedule(static) num_threads(num_threads_) if (num_leaves_in_cur_level_ >= 1024)
      for (int leaf_index_in_level = 0; leaf_index_in_level < num_leaves_in_cur_level_; ++leaf_index_in_level) {
        best_leaf_in_level_should_be_split_[leaf_index_in_level] = thread_leaf_in_level_should_be_split_[leaf_index_in_level];
      }
    }
  }
  if (best_inner_feature_index_cur_level_ != -1) {
    
  }
}

void SymmetricTreeLearner::SplitLevel(Tree* tree) {
  symmetric_data_partition_.Split(best_inner_feature_index_cur_level_,
    best_threshold_cur_level_, best_split_direction_cur_level_,
    best_leaf_in_level_should_be_split_);
  if (best_inner_feature_index_cur_level_ != -1) {
    for (int leaf_index_in_level = 0; leaf_index_in_level < num_leaves_in_cur_level_; ++leaf_index_in_level) {
      if (best_leaf_in_level_should_be_split_[leaf_index_in_level]) {
        const int real_leaf_index = leaf_ids_in_current_level_[leaf_index_in_level];
        //TODO(shiyu1994) may be change the type of best_threshold_cur_level_;
        const uint32_t uint32_best_threshold_cur_level = static_cast<uint32_t>(best_threshold_cur_level_);
        tree->Split(real_leaf_index, best_inner_feature_index_cur_level_,
          train_data_->RealFeatureIndex(best_inner_feature_index_cur_level_),
          uint32_best_threshold_cur_level,
          train_data_->RealThreshold(best_inner_feature_cur_level_, uint32_best_threshold_cur_level),
          )
      }
    }
  }
}

}  // namespace LightGBM
