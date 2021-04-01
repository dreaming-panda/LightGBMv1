/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/objective_function.h>

#include "symmetric_tree_learner.hpp"

namespace LightGBM {

SymmetricTreeLearner::SymmetricTreeLearner(const Config* config): 
SerialTreeLearner(config),
max_depth_(config->max_depth), max_num_leaves_(1 << max_depth_),
num_threads_(OMP_NUM_THREADS()) {}

void SymmetricTreeLearner::Init(const Dataset* train_data, bool is_constant_hessian) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();

  symmetric_data_partition_.reset(new SymmetricDataPartition(num_data_, max_num_leaves_, num_threads_));
  symmetric_histogram_pool_.reset(new SymmetricHistogramPool(num_threads_, max_num_leaves_));

  GetShareStates(train_data_, is_constant_hessian, true);
  col_sampler_.SetTrainingData(train_data_);

  ordered_gradients_.resize(num_data_, 0.0f);
  ordered_hessians_.resize(num_data_, 0.0f);

  paired_leaf_indices_in_cur_level_.resize(max_num_leaves_);

  thread_best_inner_feature_index_cur_level_.resize(num_threads_, -1);
  thread_best_threshold_cur_level_.resize(num_threads_, -1);
  thread_best_gain_cur_level_.resize(num_threads_, kMinScore);
  thread_best_split_default_left_cur_level_.resize(num_threads_, 0);

  leaf_indices_in_cur_level_.resize(max_num_leaves_, -1);

  best_level_split_info_.resize(max_num_leaves_);

  level_leaf_splits_.resize(max_num_leaves_);
}

Tree* SymmetricTreeLearner::Train(const score_t* gradients, const score_t *hessians, bool /*is_first_tree*/) {
  gradients_ = gradients;
  hessians_ = hessians;
  BeforeTrain();
  // TODO(shiyu1994) support interaction constraints and linear tree
  std::unique_ptr<Tree> tree(new Tree(max_num_leaves_, false, false));
  for (int depth = 0; depth < config_->max_depth; ++depth) {
    SetUpLevelInfo(depth);
    PrepareLevelHistograms();
    // construct and subtract
    symmetric_data_partition_->ConstructLevelHistograms(&level_feature_histograms_, train_data_,
      col_sampler_.is_feature_used_bytree(), gradients, hessians, share_state_.get());
    // find best splits
    FindBestLevelSplits();
    SplitLevel(tree.get());
  }
  return tree.release();
}

void SymmetricTreeLearner::PrepareLevelHistograms() {
  for (int i = 0; i < num_leaves_in_cur_level_; ++i) {
    const int leaf_id = leaf_indices_in_cur_level_[i];
    const bool get = symmetric_histogram_pool_->Get(leaf_id, &level_feature_histograms_[i]);
    if (!get) {
      // TODO(shiyu1994): handle the case when the feature histogram cache is not enough
    }
  }
}

void SymmetricTreeLearner::BeforeTrain() {
  leaf_indices_in_cur_level_[0] = 0;
  for (int i = 1; i < max_num_leaves_; ++i) {
    leaf_indices_in_cur_level_[i] = -1;
  }
  level_feature_histograms_.resize(max_num_leaves_, nullptr);
  
  // initalize leaf splits
  level_leaf_splits_[0].reset(new LeafSplits(num_data_, config_));
  level_leaf_splits_[0]->Init(gradients_, hessians_);

  paired_leaf_indices_in_cur_level_[0].resize(1);
  paired_leaf_indices_in_cur_level_[0][0] = 0;

  col_sampler_.ResetByTree();
}

void SymmetricTreeLearner::FindBestLevelSplits() {
  Threading::For<int>(0, num_features_, 1, [this] (int thread_id, int start, int end) {
    for (int inner_feature_index = start; inner_feature_index < end; ++inner_feature_index) {
      FindBestLevelSplitsForFeature(inner_feature_index, thread_id);
    }
  });
}

void SymmetricTreeLearner::SetUpLevelInfo(const int depth) {
  cur_level_ = depth;
  num_leaves_in_cur_level_ = 1 << cur_level_;
  best_inner_feature_index_cur_level_ = -1;
  best_threshold_cur_level_ = -1;
  best_gain_cur_level_ = kMinScore;
  best_split_default_left_cur_level_ = -1;

  for (int i = 0; i < num_leaves_in_cur_level_; ++i) {
    best_leaf_in_level_should_be_split_[i] = 0;
  }
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    thread_best_gain_cur_level_[thread_id] = kMinScore;
  }
  //TODO(shiyu1994) add level-wise feature subsampling
}

void SymmetricTreeLearner::FindBestLevelSplitsForFeature(const int inner_feature_index, const int thread_id) {
  symmetric_histogram_pool_->FindBestThresholdFromLevelHistograms(inner_feature_index,
    paired_leaf_indices_in_cur_level_,
    leaf_indices_in_cur_level_,
    level_leaf_splits_,
    &thread_best_inner_feature_index_cur_level_[thread_id],
    &thread_best_threshold_cur_level_[thread_id],
    &thread_best_gain_cur_level_[thread_id],
    &thread_best_split_default_left_cur_level_[thread_id],
    thread_id,
    num_leaves_in_cur_level_);
  best_gain_cur_level_ = kMinScore;
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    if (thread_best_gain_cur_level_[thread_id] > best_gain_cur_level_) {
      best_inner_feature_index_cur_level_ = thread_best_inner_feature_index_cur_level_[thread_id];
      best_threshold_cur_level_ = thread_best_threshold_cur_level_[thread_id];
      best_gain_cur_level_ = thread_best_gain_cur_level_[thread_id];
      best_split_default_left_cur_level_ = thread_best_split_default_left_cur_level_[thread_id];
    }
  }
  if (best_inner_feature_index_cur_level_ != -1) {
    #pragma omp parallel for schedule(static) num_threads(num_threads_) if (num_leaves_in_cur_level_ >= 1024)
    for (int i = 0; i < num_leaves_in_cur_level_; ++i) {
      if (best_leaf_in_level_should_be_split_[i]) {
        const int leaf_index = leaf_indices_in_cur_level_[i];
        SplitInfo& split_info = best_level_split_info_[i];
        symmetric_histogram_pool_->GetSplitLeafOutput(leaf_index, best_inner_feature_index_cur_level_,
          best_threshold_cur_level_, best_split_default_left_cur_level_, level_leaf_splits_[i],
          &split_info.left_output, &split_info.right_output,
          &split_info.left_sum_gradient, &split_info.left_sum_hessian, &split_info.right_sum_gradient, &split_info.right_sum_hessian,
          &split_info.left_count, &split_info.right_count, &split_info.gain);
      }
    }
  }
}

void SymmetricTreeLearner::SplitLevel(Tree* tree) {
  symmetric_data_partition_->Split(
    train_data_,
    best_inner_feature_index_cur_level_,
    best_threshold_cur_level_,
    best_split_default_left_cur_level_,
    best_leaf_in_level_should_be_split_,
    best_level_split_info_);
  if (best_inner_feature_index_cur_level_ != -1) {
    std::vector<int> old_leaf_indices_in_cur_level = leaf_indices_in_cur_level_;
    std::vector<std::unique_ptr<LeafSplits>> old_level_leaf_splits(num_leaves_in_cur_level_);
    for (int i = 0; i < num_leaves_in_cur_level_; ++i) {
      old_level_leaf_splits[i].reset(level_leaf_splits_[i].release());
    }
    int num_leaves_in_next_level = 0;
    for (int leaf_index_in_level = 0; leaf_index_in_level < num_leaves_in_cur_level_; ++leaf_index_in_level) {
      const int real_leaf_index = old_leaf_indices_in_cur_level[leaf_index_in_level];
      if (best_leaf_in_level_should_be_split_[leaf_index_in_level]) {
        //TODO(shiyu1994) may be change the type of best_threshold_cur_level_;
        const uint32_t uint32_best_threshold_cur_level = static_cast<uint32_t>(best_threshold_cur_level_);
        SplitInfo& split_info = best_level_split_info_[leaf_index_in_level];
        const int right_leaf_index = tree->Split(real_leaf_index, best_inner_feature_index_cur_level_,
          train_data_->RealFeatureIndex(best_inner_feature_index_cur_level_),
          uint32_best_threshold_cur_level,
          train_data_->RealThreshold(best_inner_feature_index_cur_level_, uint32_best_threshold_cur_level),
          split_info.left_output, split_info.right_output,
          split_info.left_count, split_info.right_count,
          split_info.left_sum_hessian, split_info.right_sum_hessian,
          split_info.gain,
          train_data_->FeatureBinMapper(best_inner_feature_index_cur_level_)->missing_type(),
          split_info.default_left);
        symmetric_data_partition_->SplitInnerLeafIndex(real_leaf_index, real_leaf_index, right_leaf_index);
        const data_size_t left_count = symmetric_data_partition_->leaf_count(real_leaf_index);
        const data_size_t right_count = symmetric_data_partition_->leaf_count(right_leaf_index);
        // correct leaf count in split info, which was originally estimated from sum of hessians
        split_info.left_count = left_count;
        split_info.right_count = right_count;
        paired_leaf_indices_in_cur_level_[leaf_index_in_level].resize(2);
        if (left_count <= right_count) {
          paired_leaf_indices_in_cur_level_[leaf_index_in_level][0] = real_leaf_index;
          paired_leaf_indices_in_cur_level_[leaf_index_in_level][1] = right_leaf_index;
        } else {
          paired_leaf_indices_in_cur_level_[leaf_index_in_level][0] = right_leaf_index;
          paired_leaf_indices_in_cur_level_[leaf_index_in_level][1] = real_leaf_index;
        }
        level_leaf_splits_[num_leaves_in_next_level].reset(new LeafSplits(left_count, config_));
        level_leaf_splits_[num_leaves_in_next_level]->Init(
          real_leaf_index,
          split_info.left_count,
          split_info.left_sum_gradient,
          split_info.left_sum_hessian,
          split_info.left_output);
        leaf_indices_in_cur_level_[num_leaves_in_next_level++] = real_leaf_index;
        level_leaf_splits_[num_leaves_in_next_level].reset(new LeafSplits(symmetric_data_partition_->leaf_count(right_leaf_index), config_));
        level_leaf_splits_[num_leaves_in_next_level]->Init(
          right_leaf_index,
          split_info.right_count,
          split_info.right_sum_gradient,
          split_info.right_sum_hessian,
          split_info.right_output);
        leaf_indices_in_cur_level_[num_leaves_in_next_level++] = right_leaf_index;
      } else {
        // update inner leaf index map of data partition
        symmetric_data_partition_->SplitInnerLeafIndex(real_leaf_index, -1, -1);
        paired_leaf_indices_in_cur_level_[leaf_index_in_level].resize(1);
        paired_leaf_indices_in_cur_level_[leaf_index_in_level][0] = real_leaf_index;
        level_leaf_splits_[num_leaves_in_next_level].reset(old_level_leaf_splits[leaf_index_in_level].release());
        leaf_indices_in_cur_level_[num_leaves_in_next_level++] = real_leaf_index;
      }
    }
  }
}

void SymmetricTreeLearner::ResetTrainingDataInner(const Dataset* train_data,
  bool is_constant_hessian,
  bool reset_multi_val_bin) {
  train_data_ = train_data;
  num_data_ = train_data_->num_data();
  CHECK_EQ(num_features_, train_data_->num_features());

  if (reset_multi_val_bin) {
    col_sampler_.SetTrainingData(train_data_);
    GetShareStates(train_data_, is_constant_hessian, false);
  }

  ordered_gradients_.resize(num_data_, 0.0f);
  ordered_hessians_.resize(num_data_, 0.0f);

  //TODO(shiyu1994): handle cost efficient (cegb_)
}

Tree* SymmetricTreeLearner::FitByExistingTree(const Tree* /*old_tree*/,
  const score_t* /*gradients*/, const score_t* /*hessians*/) const {
  //TODO(shiyu1994)
  return nullptr;
}

Tree* SymmetricTreeLearner::FitByExistingTree(const Tree* /*old_tree*/, const std::vector<int>& /*leaf_pred*/,
  const score_t* /*gradients*/, const score_t* /*hessians*/) const {
  // TODO(shiyu1994)
  return nullptr;
}

void SymmetricTreeLearner::AddPredictionToScore(const Tree* tree, double* out_score) const {
  CHECK_LE(tree->num_leaves(), data_partition_->num_leaves());
  if (tree->num_leaves() <= 1) {
    return;
  }
  Threading::For<data_size_t>(0, num_data_, 512,
    [this, tree, out_score] (int /*thread_id*/, data_size_t start, data_size_t end) {
    for (data_size_t i = start; i < end; ++i) {
      const int real_leaf_index = symmetric_data_partition_->GetDataLeafIndex(i);
      const double output = static_cast<double>(tree->LeafOutput(real_leaf_index));
      out_score[i] += output;
    }
  });
}

void SymmetricTreeLearner::RenewTreeOutput(Tree* /*tree*/, const ObjectiveFunction* obj,
  std::function<double(const label_t*, int)> /*residual_getter*/,
  data_size_t /*total_num_data*/, const data_size_t* /*bag_indices*/, data_size_t /*bag_cnt*/) const {
  if (obj != nullptr && obj->IsRenewTreeOutput()) {
    Log::Fatal("renew output is not supported with symmetric tree yet");
  }
}

void SymmetricTreeLearner::SetBaggingData(const Dataset* /*subset*/,
  const data_size_t* /*used_indices*/, data_size_t /*num_data*/) {
  Log::Fatal("bagging is not supported with symmetric tree yet");
}

}  // namespace LightGBM
