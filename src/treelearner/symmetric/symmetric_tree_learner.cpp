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
  thread_leaf_in_level_should_be_split_.resize(num_threads_, 0);
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    thread_leaf_in_level_should_be_split_[thread_id].resize(max_num_leaves_, 0);
  }
  thread_best_split_default_left_.resize(num_threads_, 0);

  leaf_indices_in_cur_level_.resize(max_num_leaves_, -1);

  best_level_left_output_.resize(max_num_leaves_, 0.0f);
  best_level_right_output_.resize(max_num_leaves_, 0.0f);
  best_level_left_sum_gradient_.resize(max_num_leaves_, 0.0f);
  best_level_right_sum_gradient_.resize(max_num_leaves_, 0.0f);
  best_level_left_sum_hessian_.resize(max_num_leaves_, 0.0f);
  best_level_right_sum_hessian_.resize(max_num_leaves_, 0.0f);
  best_level_left_count_.resize(max_num_leaves_, 0);
  best_level_right_count_.resize(max_num_leaves_, 0);
  best_level_split_gain_.resize(max_num_leaves_, 0.0f);

  best_level_split_info_.resize(max_num_leaves_);
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
    const int leaf_id = leaf_indices_in_cur_level_[i];
    const bool get = symmetric_histogram_pool_.Get(leaf_id, &level_feature_histograms_[i]);
    if (!get) {
      // TODO(shiyu1994): handle the case when the feature histogram cache is not enough
    }
  }
}

void SymmetricTreeLearner::BeforeTrain() {
  leaf_indices_in_cur_level_[0] = 0;
  for (int i = 1; i < max_num_leaves_; ++i) {
    leaf_indices_in_cur_level[i] = -1;
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
  cur_level_ = depth;
  num_leaves_in_cur_level_ = 1 << cur_level_;
  best_inner_feature_index_cur_level_ = -1;
  best_threshold_cur_level_ = -1;
  best_gain_cur_level_ = kMinScore;
  best_split_direction_cur_level_ = 0;
  best_split_default_left_ = -1;

  for (int i = 0; i < num_leaves_in_cur_level_; ++i) {
    best_leaf_in_level_should_be_split_[i] = 0;
  }
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    thread_best_gain_cur_level_[thread_id] = kMinScore;
  }
  //TODO(shiyu1994) add level-wise feature subsampling
}

void SymmetricTreeLearner::FindBestLevelSplitsForFeature(const int inner_feature_index, const int thread_id) {
  SymmetricHistogramPool::FindBestThreaholdFromLevelHistograms(inner_feature_index,
    paired_leaf_indices_in_cur_level_,
    &thread_best_inner_feature_index_cur_level_[thread_id],
    &thread_best_threshold_cur_level_[thread_id],
    &thread_best_gain_cur_level_[thread_id],
    &thread_best_split_direction_cur_level_[thread_id],
    &thread_leaf_in_level_should_be_split_[thread_id],
    &thread_best_split_default_left_[thread_id]);
  best_gain_cur_level_ = kMinScore;
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    if (thread_best_gain_cur_level_[thread_id] > best_gain_cur_level_) {
      best_inner_feature_index_cur_level_ = thread_best_inner_feature_index_cur_level_[thread_id];
      best_threshold_cur_level_ = thread_best_threshold_cur_level_[thread_id];
      best_gain_cur_level_ = thread_best_gain_cur_level_[thread_id];
      best_split_direction_cur_level_ = thread_best_split_direction_cur_level_[thread_id];
      #pragma omp parallel for schedule(static) num_threads(num_threads_) if (num_leaves_in_cur_level_ >= 1024)
      for (int leaf_index_in_level = 0; leaf_index_in_level < num_leaves_in_cur_level_; ++leaf_index_in_level) {
        best_leaf_in_level_should_be_split_[leaf_index_in_level] = thread_leaf_in_level_should_be_split_[thread_id][leaf_index_in_level];
      }
      best_split_default_left_ = thread_best_split_default_left_[thread_id];
    }
  }
  if (best_inner_feature_index_cur_level_ != -1) {
    #pragma omp parallel for schedule(static) num_threads(num_threads_) if (num_leaves_in_cur_level_ >= 1024)
    for (int i = 0; i < num_leaves_in_cur_level_; ++i) {
      if (best_leaf_in_level_should_be_split_[i]) {
        const int leaf_index = leaf_indices_in_cur_level_[i];
        SymmetricHistogramPool::GetSplitLeafOutput(leaf_index, best_inner_feature_index_cur_level_,
          best_threshold_cur_level_, best_split_direction_cur_level_, &best_level_left_output_[i], &best_level_right_output_[i],
          &best_level_left_gradient_[i], &best_level_left_hessian_[i], &best_level_right_gradient_[i], &best_level_right_hessian_[i],
          &best_level_left_count_[i], &best_level_right_count_[i], &best_level_split_gain_[i]);
      }
    }
  }
}

void SymmetricTreeLearner::SplitLevel(Tree* tree) {
  symmetric_data_partition_.Split(best_inner_feature_index_cur_level_,
    best_threshold_cur_level_, best_split_direction_cur_level_,
    best_leaf_in_level_should_be_split_);
  if (best_inner_feature_index_cur_level_ != -1) {
    std::vector<int> old_leaf_indices_in_cur_level = leaf_indices_in_cur_level_;
    std::vector<std::unique_ptr<LeafSplits>> old_level_leaf_splits_(num_leaves_in_cur_level_, nullptr);
    for (int i = 0; i < num_leaves_in_cur_level_; ++i) {
      old_level_leaf_splits_[i].reset(level_leaf_splits_[i].release());
    }
    int num_leaves_in_next_level = 0;
    for (int leaf_index_in_level = 0; leaf_index_in_level < num_leaves_in_cur_level_; ++leaf_index_in_level) {
      const int real_leaf_index = old_leaf_indices_in_cur_level[leaf_index_in_level];
      if (best_leaf_in_level_should_be_split_[leaf_index_in_level]) {
        //TODO(shiyu1994) may be change the type of best_threshold_cur_level_;
        const uint32_t uint32_best_threshold_cur_level = static_cast<uint32_t>(best_threshold_cur_level_);
        const int right_leaf_index = tree->Split(real_leaf_index, best_inner_feature_index_cur_level_,
          train_data_->RealFeatureIndex(best_inner_feature_index_cur_level_),
          uint32_best_threshold_cur_level,
          train_data_->RealThreshold(best_inner_feature_index_cur_level_, uint32_best_threshold_cur_level),
          best_level_left_output_[leaf_index_in_level], best_level_right_output_[leaf_index_in_level],
          best_level_left_gradient_[leaf_index_in_level], best_level_right_gradient_[leaf_index_in_level],
          best_level_left_count_[leaf_index_in_level], best_level_right_count_[leaf_index_in_level],
          best_level_left_hessian_[leaf_index_in_level], best_level_right_hessian_[leaf_index_in_level],
          best_level_split_gain_[leaf_index_in_level]
        );
        symmetric_data_partition_.Split(real_leaf_index, train_data_, best_inner_feature_index_cur_level_,
          uint32_best_threshold_cur_level, best_split_default_left_, right_leaf_index);
        const data_size_t left_count = symmetric_data_partition_.leaf_count(real_leaf_index);
        const data_size_t right_count = symmetric_data_partition_.leaf_count(right_leaf_index);
        paired_leaf_indices_in_cur_level_[leaf_index_in_level].resize(2);
        if (left_count <= right_count) {
          paired_leaf_indices_in_cur_level_[leaf_index_in_level][0] = num_leaves_in_next_level;
          paired_leaf_indices_in_cur_level_[leaf_index_in_level][1] = num_leaves_in_next_level + 1;
        } else {
          paired_leaf_indices_in_cur_level_[leaf_index_in_level][0] = num_leaves_in_next_level + 1;
          paired_leaf_indices_in_cur_level_[leaf_index_in_level][1] = num_leaves_in_next_level;
        }
        level_leaf_splits_[num_leaves_in_next_level]->reset(new LeafSplits(left_count, config_));
        level_leaf_splits_[num_leaves_in_next_level]->Init(real_leaf_index, &symmetric_data_partition_, gradients_, hessians_);
        leaf_indices_in_cur_level_[num_leaves_in_next_level++] = real_leaf_index;
        level_leaf_splits_[num_leaves_in_next_level]->reset(new LeafSplits(symmetric_data_partition_.leaf_count(right_leaf_index), config_));
        level_leaf_splits_[num_leaves_in_next_level]->Init(right_leaf_index, &symmetric_data_partition_, gradients_, hessians_);
        leaf_indices_in_cur_level_[num_leaves_in_next_level++] = right_leaf_index;
      } else {
        paired_leaf_indices_in_cur_level_[leaf_index_in_level].resize(1);
        paired_leaf_indices_in_cur_level_[leaf_index_in_level][0] = num_leaves_in_next_level;
        level_leaf_splits_[num_leaves_in_next_level].reset(old_level_leaf_splits_[leaf_index_in_level].release());
        leaf_indices_in_cur_level_[num_leaves_in_next_level++] = real_leaf_index;
      }
    }
  }
}

}  // namespace LightGBM
