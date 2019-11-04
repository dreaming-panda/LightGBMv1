/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "serial_tree_learner.h"
#include <queue>

namespace LightGBM {

Tree* SymmetricTreeShareThresholdMultiFeatureLearner::Train(const score_t* gradients, const score_t *hessians, bool is_constant_hessian,
              Json& /*forced_split_json*/) {
  sum_gradients_ = 0.0;
  sum_hessians_ = 0.0;
  gradients_ = gradients;
  hessians_ = hessians;
  is_constant_hessian_ = is_constant_hessian;
  hist_time_ = std::chrono::duration<double>(0.0);
  #ifdef TIMETAG
  auto start_time = std::chrono::steady_clock::now();
  #endif
  // some initial works before training
  BeforeTrain();

  #ifdef TIMETAG
  init_train_time += std::chrono::steady_clock::now() - start_time;
  #endif

  auto tree = std::unique_ptr<Tree>(new Tree(config_->num_leaves));
  // root leaf
  int left_leaf = 0;
  // only root leaf can be splitted on first time
  int right_leaf = -1;
  
  std::vector<SplitInfo> level_splits;
  std::queue<int> level_leaf_queue;
  bool is_left_right_update = false;
  used_features_.clear();
  is_feature_used_.clear();
  is_feature_used_.resize(num_features_, 1);
  for(int level = 0; level < config_->max_depth; ++level) {
      std::queue<int> next_level_leaf_queue;
      if(level == 0) {
          BeforeFindBestSplit(tree.get(), left_leaf, right_leaf);
          FindBestSplits();
          is_feature_used_.clear();
          is_feature_used_.resize(num_features_, 0);
          for(int feature_index = 0; feature_index < num_features_; ++feature_index) {
              level_splits.push_back(splits_per_leaf_[feature_index]);
          }
          std::sort(level_splits.begin(), level_splits.end(), [] (SplitInfo a, SplitInfo b) { return a.gain > b.gain; });
          int best_leaf = 0;
          Split(tree.get(), best_leaf, &left_leaf, &right_leaf);
          for(int i = 0; i < num_features_; ++i) {
            int inner_feature_idx = train_data_->InnerFeatureIndex(level_splits[i].feature);
            used_features_.push_back(inner_feature_idx);
            is_feature_used_[inner_feature_idx] = 1;
          }
          InitializeThresholdStats(1 << (config_->max_depth - 1));
          is_left_right_update = true;
          next_level_leaf_queue.push(left_leaf);
          next_level_leaf_queue.push(right_leaf);
          cur_leaf_id_in_level_ = 0;
      }
      else {
        int level_size = static_cast<int>(level_leaf_queue.size());
        while(!level_leaf_queue.empty()) {
            int best_leaf = level_leaf_queue.front();
            const int node_in_level = level_size - static_cast<int>(level_leaf_queue.size());
            if(is_left_right_update) {
              BeforeFindBestSplit(tree.get(), left_leaf, right_leaf);
              FindBestSplitForFeature(left_leaf, right_leaf, -1, -1);
              cur_leaf_id_in_level_ += 2;
            }
            if(node_in_level == 0) {
              feature_threshold_gain_ = next_feature_threshold_gain_;
              feature_threshold_split_info_ = next_feature_threshold_split_info_;

              ClearGainVector();
              SetShareThreshold(level_leaf_queue, -1);
              CHECK(cur_leaf_id_in_level_ == level_size);
              cur_leaf_id_in_level_ = 0;
            }

            level_leaf_queue.pop();
            
            Split(tree.get(), best_leaf, &left_leaf, &right_leaf);
            is_left_right_update = true;

            next_level_leaf_queue.push(left_leaf);
            next_level_leaf_queue.push(right_leaf);
        }
      }

      level_leaf_queue = next_level_leaf_queue;
      if(level_leaf_queue.empty()) {
        break;
      }
  }
  Log::Warning("histogram time %f s", hist_time_.count());
  return tree.release();
}

void SymmetricTreeShareThresholdMultiFeatureLearner::FindBestSplitForFeature(int left_leaf, int right_leaf, int /*left_inner_feature_index*/, int /*right_inner_feature_index*/) {

  #ifdef TIMETAG
  auto start_time = std::chrono::steady_clock::now();
  #endif

  int smaller_in_level = cur_leaf_id_in_level_;
  int larger_in_level = cur_leaf_id_in_level_ + 1;
  int left_leaf_data_count = GetGlobalDataCountInLeaf(left_leaf);
  int right_leaf_data_count = GetGlobalDataCountInLeaf(right_leaf);
  bool use_subtract = parent_leaf_histogram_array_ != nullptr;
  if(right_leaf != -1 && left_leaf_data_count >= right_leaf_data_count) {
    smaller_in_level = cur_leaf_id_in_level_ + 1;
    larger_in_level = cur_leaf_id_in_level_;
  }

  auto hist_start_time = std::chrono::steady_clock::now();
  // construct smaller leaf
  HistogramBinEntry* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - 1;
  train_data_->ConstructHistograms(is_feature_used_,
                                  smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
                                  smaller_leaf_splits_->LeafIndex(),
                                  ordered_bins_, gradients_, hessians_,
                                  ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
                                  ptr_smaller_leaf_hist_data);
  if(right_leaf != -1) {
    //construct larger leaf
    HistogramBinEntry* ptr_larger_leaf_hist_data = larger_leaf_histogram_array_[0].RawData() - 1;
    if(!use_subtract) {
      train_data_->ConstructHistograms(is_feature_used_,
                                        larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
                                        larger_leaf_splits_->LeafIndex(),
                                        ordered_bins_, gradients_, hessians_,
                                        ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
                                        ptr_larger_leaf_hist_data);
    }
  }

  hist_time_ += std::chrono::steady_clock::now() - hist_start_time;

  //find best threshold for smaller leaf
  #pragma omp parallel for schedule(static) num_threads(config_->num_threads)
  for(size_t i = 0; i < used_features_.size(); ++i) {
    SplitInfo smaller_split;
    int smaller_inner_feature_index = used_features_[i];
    train_data_->FixHistogram(smaller_inner_feature_index,
                              smaller_leaf_splits_->sum_gradients(), smaller_leaf_splits_->sum_hessians(),
                              smaller_leaf_splits_->num_data_in_leaf(),
                              smaller_leaf_histogram_array_[smaller_inner_feature_index].RawData());
    int smaller_real_fidx = train_data_->RealFeatureIndex(smaller_inner_feature_index);
    smaller_leaf_histogram_array_[smaller_inner_feature_index].FindBestThreshold(
      smaller_leaf_splits_->sum_gradients(),
      smaller_leaf_splits_->sum_hessians(),
      smaller_leaf_splits_->num_data_in_leaf(),
      smaller_leaf_splits_->min_constraint(),
      smaller_leaf_splits_->max_constraint(),
      &smaller_split,
      &next_feature_threshold_gain_[i],
      &next_feature_threshold_split_info_[i][smaller_in_level]);
    smaller_split.feature = smaller_real_fidx;
  }

  if(right_leaf != -1) {
    #pragma omp parallel for schedule(static) num_threads(config_->num_threads)
    for(size_t i = 0; i < used_features_.size(); ++i) {
      SplitInfo larger_split;
      int larger_inner_feature_index = used_features_[i];
      if(use_subtract) {
        larger_leaf_histogram_array_[larger_inner_feature_index].Subtract(smaller_leaf_histogram_array_[larger_inner_feature_index]);
      }
      train_data_->FixHistogram(larger_inner_feature_index,
                                larger_leaf_splits_->sum_gradients(), larger_leaf_splits_->sum_hessians(),
                                larger_leaf_splits_->num_data_in_leaf(),
                                larger_leaf_histogram_array_[larger_inner_feature_index].RawData());
      int larger_real_fidx = train_data_->RealFeatureIndex(larger_inner_feature_index);
      larger_leaf_histogram_array_[larger_inner_feature_index].FindBestThreshold(
        larger_leaf_splits_->sum_gradients(),
        larger_leaf_splits_->sum_hessians(),
        larger_leaf_splits_->num_data_in_leaf(),
        larger_leaf_splits_->min_constraint(),
        larger_leaf_splits_->max_constraint(),
        &larger_split,
        &next_feature_threshold_gain_[i],
        &next_feature_threshold_split_info_[i][larger_in_level]);
      larger_split.feature = larger_real_fidx;
    }
  }
}

void SymmetricTreeShareThresholdMultiFeatureLearner::SetShareThreshold(const std::queue<int>& level_leaf_queue, int /*feature*/) {
  std::queue<int> copy_level_leaf_queue = level_leaf_queue;
  uint32_t best_threshold = 0;
  int best_dir = -1;
  double best_gain = kMinScore;
  size_t best_i = 0;
  for(size_t i = 0; i < feature_threshold_gain_.size(); ++i) {
    for(size_t j = 0; j < feature_threshold_gain_[i].size(); ++j) {
      if(feature_threshold_gain_[i][j][0] > best_gain) {
        best_gain = feature_threshold_gain_[i][j][0];
        best_threshold = j;
        best_dir = 0;
        best_i = i;
      }
      if(feature_threshold_gain_[i][j][1] > best_gain) {
        best_gain = feature_threshold_gain_[i][j][1];
        best_threshold = j;
        best_dir = 1;
        best_i = i;
      }
    }
  }
  int feature = train_data_->RealFeatureIndex(used_features_[best_i]);
  int cur_leaf_id_in_level = 0;
  double tmp_sum_gradients = 0.0, tmp_sum_hessians = 0.0;
  bool is_first = false;
  if(sum_gradients_ == 0.0 && sum_hessians_ == 0.0) {
    is_first = true;
  }
  while(!copy_level_leaf_queue.empty()) {
    int leaf = copy_level_leaf_queue.front();
    copy_level_leaf_queue.pop();
    best_split_per_leaf_[leaf] = feature_threshold_split_info_[best_i][cur_leaf_id_in_level][best_threshold][best_dir];
    best_split_per_leaf_[leaf].feature = feature;
    SplitInfo& split_info = best_split_per_leaf_[leaf];
    split_info.left_output = FeatureHistogram::CalculateSplittedLeafOutput(split_info.left_sum_gradient, split_info.left_sum_hessian, config_->lambda_l1, config_->lambda_l2, config_->max_delta_step);
    split_info.right_output = FeatureHistogram::CalculateSplittedLeafOutput(split_info.right_sum_gradient, split_info.right_sum_hessian, config_->lambda_l1, config_->lambda_l2, config_->max_delta_step);

    CHECK(feature == split_info.feature);
    if(is_first) {
      sum_gradients_ += split_info.left_sum_gradient;
      sum_hessians_ += split_info.left_sum_hessian;
      sum_gradients_ += split_info.right_sum_gradient;
      sum_hessians_ += split_info.right_sum_hessian;
    }
    else {
      tmp_sum_gradients += split_info.left_sum_gradient;
      tmp_sum_hessians += split_info.left_sum_hessian;
      tmp_sum_gradients += split_info.right_sum_gradient;
      tmp_sum_hessians += split_info.right_sum_hessian;
    }
    ++cur_leaf_id_in_level;
  }
  if(!is_first) {
    if(std::fabs(sum_gradients_ - tmp_sum_gradients) > 1e-6) {
      Log::Warning("sum_gradients %f, tmp_sum_gradients %f, feature %d, threshold %d", sum_gradients_, tmp_sum_gradients, feature, best_threshold);
    }
    CHECK(std::fabs(sum_gradients_ - tmp_sum_gradients) <= 1e-6);
    if(std::fabs(sum_hessians_ - tmp_sum_hessians) > 1e-6) {
      Log::Warning("sum_hessians %f, tmp_sum_hessians %f, feature %d, threshold %d", sum_hessians_, tmp_sum_hessians, feature, best_threshold);
    }
    CHECK(std::fabs(sum_hessians_ - tmp_sum_hessians) <= 1e-6);
  }
}

void SymmetricTreeShareThresholdMultiFeatureLearner::InitializeThresholdStats(const size_t level_size) {
  next_feature_threshold_gain_.clear();
  next_feature_threshold_gain_.resize(used_features_.size());
  for(size_t i = 0; i < next_feature_threshold_gain_.size(); ++i) {
    int num_bin = train_data_->FeatureBinMapper(used_features_[i])->num_bin();
    next_feature_threshold_gain_[i].resize(num_bin);
    for(int j = 0; j < num_bin; ++j) {
      next_feature_threshold_gain_[i][j].resize(2, 0.0);
    }
  }

  next_feature_threshold_split_info_.clear();
  next_feature_threshold_split_info_.resize(used_features_.size());
  for(size_t i = 0; i < next_feature_threshold_split_info_.size(); ++i) {
    next_feature_threshold_split_info_[i].clear();
    next_feature_threshold_split_info_[i].resize(level_size);
    for(size_t j = 0; j < level_size; ++j) {
      next_feature_threshold_split_info_[i][j].resize(train_data_->FeatureBinMapper(used_features_[i])->num_bin());
      for(size_t k = 0; k < next_feature_threshold_split_info_[i][j].size(); ++k) {
        next_feature_threshold_split_info_[i][j][k].resize(2);
      }
    }
  }
}

void SymmetricTreeShareThresholdMultiFeatureLearner::ClearGainVector() {
  #pragma omp parallel for schedule(static) num_threads(config_->num_threads)
  for(size_t i = 0; i < used_features_.size(); ++i) {
    int num_bin = train_data_->FeatureBinMapper(used_features_[i])->num_bin();
    next_feature_threshold_gain_[i].resize(num_bin);
    for(int j = 0; j < num_bin; ++j) {
      next_feature_threshold_gain_[i][j].resize(2, 0.0);
      next_feature_threshold_gain_[i][j][0] = 0.0;
      next_feature_threshold_gain_[i][j][1] = 0.0;
    }
  }
}

} // namespace LightGBM