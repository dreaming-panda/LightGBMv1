/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "serial_tree_learner.h"
#include <queue>

namespace LightGBM {

Tree* SymmetricTreeShareThresholdLearner::Train(const score_t* gradients, const score_t *hessians, bool is_constant_hessian,
              Json& /*forced_split_json*/) {
  sum_gradients_ = 0.0;
  sum_hessians_ = 0.0;
  gradients_ = gradients;
  hessians_ = hessians;
  is_constant_hessian_ = is_constant_hessian;
  hist_time_ = std::chrono::duration<double>(0.0);
  auto split_time = std::chrono::duration<double>(0.0);
  auto before_time = std::chrono::duration<double>(0.0);
  auto train_start_time = std::chrono::steady_clock::now();
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
  int feature = -1;
  int level_inner_feature_index = -1;
  int max_num_bin = 0;
  for(int level = 0; level < config_->max_depth; ++level) {
      std::queue<int> next_level_leaf_queue;
      if(level == 0) {
          auto root_time_start = std::chrono::steady_clock::now();
          BeforeFindBestSplit(tree.get(), left_leaf, right_leaf);
          before_time += std::chrono::steady_clock::now() - root_time_start;
          FindBestSplits();
          Log::Warning("root time for histogram %f s", (static_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - root_time_start)).count());
          for(int feature_index = 0; feature_index < num_features_; ++feature_index) {
              level_splits.push_back(splits_per_leaf_[feature_index]);
          }
          std::sort(level_splits.begin(), level_splits.end(), [] (SplitInfo a, SplitInfo b) { return a.gain > b.gain; });
          SetOrderedBin(level_splits);
          int best_leaf = 0;
          auto split_start = std::chrono::steady_clock::now();
          Split(tree.get(), best_leaf, &left_leaf, &right_leaf);
          split_time += std::chrono::steady_clock::now() - split_start;
          for(int i = 0; i < config_->max_depth; ++i) {
            const BinMapper* bin_mapper = train_data_->FeatureBinMapper(train_data_->InnerFeatureIndex(level_splits[i].feature));
            if(bin_mapper->num_bin() > max_num_bin) {
              max_num_bin = bin_mapper->num_bin();
            }
          }
          InitializeThresholdStats(1 << (config_->max_depth - 1), max_num_bin);
          is_left_right_update = true;
          next_level_leaf_queue.push(left_leaf);
          next_level_leaf_queue.push(right_leaf);
          cur_leaf_id_in_level_ = 0;
      }
      else {
        feature = level - 1 >= num_features_ ? level_splits.back().feature : level_splits[level - 1].feature;
        level_inner_feature_index = train_data_->InnerFeatureIndex(feature);
        //bin_mapper = train_data_->FeatureBinMapper(level_inner_feature_index);
        //CHECK(bin_mapper->num_bin() == static_cast<int>(next_feature_threshold_gain_.size()));
        int level_size = static_cast<int>(level_leaf_queue.size());
        while(!level_leaf_queue.empty()) {
            int best_leaf = level_leaf_queue.front();
            int right_inner_feature_index = right_leaf == -1 ? -1 : level_inner_feature_index;
            const int node_in_level = level_size - static_cast<int>(level_leaf_queue.size());
            if(is_left_right_update) {
              auto before_start = std::chrono::steady_clock::now();
              BeforeFindBestSplit(tree.get(), left_leaf, right_leaf);
              before_time += std::chrono::steady_clock::now() - before_start;
              FindBestSplitForFeature(left_leaf, right_leaf, level_inner_feature_index, right_inner_feature_index);
              cur_leaf_id_in_level_ += 2;
            }

            if(node_in_level == 0) {
              feature_threshold_gain_ = next_feature_threshold_gain_;
              feature_threshold_split_info_ = next_feature_threshold_split_info_;

              feature = level >= num_features_ ? level_splits.back().feature : level_splits[level].feature;
              level_inner_feature_index = train_data_->InnerFeatureIndex(feature);

              ClearGainVector();
              SetShareThreshold(level_leaf_queue, level_splits[level - 1].feature);
              CHECK(cur_leaf_id_in_level_ == level_size);
              cur_leaf_id_in_level_ = 0;
            }

            level_leaf_queue.pop();
            
            auto split_start = std::chrono::steady_clock::now();
            Split(tree.get(), best_leaf, &left_leaf, &right_leaf);
            split_time += std::chrono::steady_clock::now() - split_start;
            is_left_right_update = level < config_->max_depth - 1;

            next_level_leaf_queue.push(left_leaf);
            next_level_leaf_queue.push(right_leaf);
        }
      }

      level_leaf_queue = next_level_leaf_queue;
      if(level_leaf_queue.empty()) {
        break;
      }
  }
  Log::Warning("split for histogram time %f s", split_time.count());
  Log::Warning("before time for histogram %f s", before_time.count());
  Log::Warning("histogram time %f s", hist_time_.count());
  Log::Warning("train tree with histogram time %f s", (static_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - train_start_time)).count());
  return tree.release();
}

void SymmetricTreeShareThresholdLearner::FindBestSplitForFeature(int left_leaf, int right_leaf, int left_inner_feature_index, int right_inner_feature_index) {
  std::vector<int8_t> is_feature_used(num_features_, 0);

  #ifdef TIMETAG
  auto start_time = std::chrono::steady_clock::now();
  #endif

  int smaller_inner_feature_index = left_inner_feature_index;
  int larger_inner_feature_index = right_inner_feature_index;
  int smaller_in_level = cur_leaf_id_in_level_;
  int larger_in_level = cur_leaf_id_in_level_ + 1;
  int left_leaf_data_count = GetGlobalDataCountInLeaf(left_leaf);
  int right_leaf_data_count = GetGlobalDataCountInLeaf(right_leaf);
  int larger_data_count = left_leaf_data_count;
  int smaller_data_count = right_leaf_data_count;
  if(right_leaf != -1 && left_leaf_data_count >= right_leaf_data_count) {
    smaller_inner_feature_index = right_inner_feature_index;
    larger_inner_feature_index = left_inner_feature_index;
    smaller_in_level = cur_leaf_id_in_level_ + 1;
    larger_in_level = cur_leaf_id_in_level_;
    larger_data_count = right_leaf_data_count;
    smaller_data_count = left_leaf_data_count;
  }

  auto hist_start_time = std::chrono::steady_clock::now();
  if(smaller_inner_feature_index != -1) {
    // construct smaller leaf
    is_feature_used[smaller_inner_feature_index] = 1;
    HistogramBinEntry* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - 1;
    if(smaller_data_count >= 10000) {
      PrepareThreadHistogramVectors();
      //Log::Warning("should success");
      train_data_->ConstructHistograms(is_feature_used,
                                      smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
                                      smaller_leaf_splits_->LeafIndex(),
                                      ordered_bins_, gradients_, hessians_,
                                      ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
                                      ptr_smaller_leaf_hist_data, config_->num_threads, &histogram_grad_, &histogram_hess_, &histogram_cnt_);
    }
    else {
      train_data_->ConstructHistograms(is_feature_used,
                                      smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
                                      smaller_leaf_splits_->LeafIndex(),
                                      ordered_bins_, gradients_, hessians_,
                                      ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
                                      ptr_smaller_leaf_hist_data);
    }

    is_feature_used[smaller_inner_feature_index] = 0;
  }

  if(larger_inner_feature_index != -1) {
    //construct larger leaf
    is_feature_used[larger_inner_feature_index] = 1;
    HistogramBinEntry* ptr_larger_leaf_hist_data = larger_leaf_histogram_array_[0].RawData() - 1;
    if(larger_data_count >= 10000) {
      //Log::Warning("should success");
      PrepareThreadHistogramVectors();
      train_data_->ConstructHistograms(is_feature_used,
                                        larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
                                        larger_leaf_splits_->LeafIndex(),
                                        ordered_bins_, gradients_, hessians_,
                                        ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
                                        ptr_larger_leaf_hist_data, config_->num_threads, &histogram_grad_, &histogram_hess_, &histogram_cnt_);
    }
    else {
      train_data_->ConstructHistograms(is_feature_used,
                                        larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
                                        larger_leaf_splits_->LeafIndex(),
                                        ordered_bins_, gradients_, hessians_,
                                        ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
                                        ptr_larger_leaf_hist_data);
    }
    is_feature_used[larger_inner_feature_index] = 0;
  }

  hist_time_ += std::chrono::steady_clock::now() - hist_start_time;

  if(smaller_inner_feature_index != -1) {
    //find best threshold for smaller leaf
    SplitInfo smaller_split;
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
      &next_feature_threshold_gain_,
      &next_feature_threshold_split_info_[smaller_in_level]);
    smaller_split.feature = smaller_real_fidx;
    smaller_split.gain -= config_->cegb_tradeoff * config_->cegb_penalty_split * smaller_leaf_splits_->num_data_in_leaf();
    if (!config_->cegb_penalty_feature_coupled.empty()) {
      smaller_split.gain -= config_->cegb_tradeoff * config_->cegb_penalty_feature_coupled[smaller_real_fidx];
    }
    if (!config_->cegb_penalty_feature_lazy.empty()) {
      smaller_split.gain -= config_->cegb_tradeoff * CalculateOndemandCosts(smaller_real_fidx, smaller_leaf_splits_->LeafIndex());
    }
    splits_per_leaf_[smaller_leaf_splits_->LeafIndex()*train_data_->num_features() + smaller_inner_feature_index] = smaller_split;
    best_split_per_leaf_[smaller_leaf_splits_->LeafIndex()] = smaller_split;
  }

  if(larger_inner_feature_index != -1) {
    SplitInfo larger_split;
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
      &next_feature_threshold_gain_,
      &next_feature_threshold_split_info_[larger_in_level]);
    larger_split.feature = larger_real_fidx;
    larger_split.gain -= config_->cegb_tradeoff * config_->cegb_penalty_split * larger_leaf_splits_->num_data_in_leaf();
    if (!config_->cegb_penalty_feature_coupled.empty()) {
      larger_split.gain -= config_->cegb_tradeoff * config_->cegb_penalty_feature_coupled[larger_real_fidx];
    }
    if (!config_->cegb_penalty_feature_lazy.empty()) {
      larger_split.gain -= config_->cegb_tradeoff * CalculateOndemandCosts(larger_real_fidx, larger_leaf_splits_->LeafIndex());
    }
    splits_per_leaf_[larger_leaf_splits_->LeafIndex()*train_data_->num_features() + larger_inner_feature_index] = larger_split;
    best_split_per_leaf_[larger_leaf_splits_->LeafIndex()] = larger_split;
  }
}

void SymmetricTreeShareThresholdLearner::SetShareThreshold(const std::queue<int>& level_leaf_queue, int feature) {
  std::queue<int> copy_level_leaf_queue = level_leaf_queue;
  uint32_t best_threshold = 0;
  double best_gain = feature_threshold_gain_[0][0];
  int best_dir = 0;
  for(uint32_t i = 1; i < feature_threshold_gain_.size(); ++i) {
    if(feature_threshold_gain_[i][0] > best_gain) {
      best_gain = feature_threshold_gain_[i][0];
      best_threshold = i;
    }
  }
  for(uint32_t i = 0; i < feature_threshold_gain_.size(); ++i) {
    if(feature_threshold_gain_[i][1] > best_gain) {
      best_gain = feature_threshold_gain_[i][1];
      best_threshold = i;
      best_dir = 1;
    }
  }
  int cur_leaf_id_in_level = 0;
  double tmp_sum_gradients = 0.0, tmp_sum_hessians = 0.0;
  bool is_first = false;
  if(sum_gradients_ == 0.0 && sum_hessians_ == 0.0) {
    is_first = true;
  }
  while(!copy_level_leaf_queue.empty()) {
    int leaf = copy_level_leaf_queue.front();
    copy_level_leaf_queue.pop();
    best_split_per_leaf_[leaf] = feature_threshold_split_info_[cur_leaf_id_in_level][best_threshold][best_dir];
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

void SymmetricTreeShareThresholdLearner::InitializeThresholdStats(const size_t level_size, const int num_bin) {
  next_feature_threshold_gain_.clear();
  next_feature_threshold_gain_.resize(num_bin);
  for(size_t i = 0; i < next_feature_threshold_gain_.size(); ++i) {
    next_feature_threshold_gain_[i].resize(2, 0.0);
  }
  next_feature_threshold_split_info_.clear();
  next_feature_threshold_split_info_.resize(level_size);
  #pragma omp parallel for schedule(static) num_threads(config_->num_threads)
  for(size_t i = 0; i < level_size; ++i) {
    next_feature_threshold_split_info_[i].resize(num_bin);
    for(size_t j = 0; j < next_feature_threshold_split_info_[i].size(); ++j) {
      next_feature_threshold_split_info_[i][j].resize(2);
    }
  }
}

void SymmetricTreeShareThresholdLearner::ClearGainVector() {
  for(size_t i = 0; i < next_feature_threshold_gain_.size(); ++i) {
    next_feature_threshold_gain_[i][0] = 0.0;
    next_feature_threshold_gain_[i][1] = 0.0;
  }
}

void SymmetricTreeShareThresholdLearner::PrepareThreadHistogramVectors() {
  histogram_grad_.resize(num_threads_);
  histogram_hess_.resize(num_threads_);
  histogram_cnt_.resize(num_threads_);

  #pragma omp parallel for schedule(static) num_threads(config_->num_threads)
  for(int thread_id = 0; thread_id < config_->num_threads; ++thread_id) {
    if(histogram_grad_[thread_id].size() == 0) {
      histogram_grad_[thread_id].resize(2 * config_->max_bin, 0.0);
      histogram_hess_[thread_id].resize(2 * config_->max_bin, 0.0);
      histogram_cnt_[thread_id].resize(2 * config_->max_bin, 0);
    }
    else {
      for(int i = 0; i < 2 * config_->max_bin; ++i) {
        histogram_grad_[thread_id][i] = 0.0;
        histogram_hess_[thread_id][i] = 0.0;
        histogram_cnt_[thread_id][i] = 0;
      }
    }
  }
}

void SymmetricTreeShareThresholdLearner::SetOrderedBin(const std::vector<SplitInfo>& level_split) {
  ordered_bin_indices_.clear();
  std::vector<bool> group_has_used_ordered_bin(train_data_->num_feature_groups(), false);
  for(int i = 0; i < config_->max_depth; ++i) {
    int inner_feature = train_data_->InnerFeatureIndex(level_split[i].feature);
    int group = train_data_->Feature2Group(inner_feature);
    if(ordered_bins_[group] != nullptr) {
      group_has_used_ordered_bin[group] = true;
    }
  }
  for(int i = 0; i < static_cast<int>(group_has_used_ordered_bin.size()); ++i) {
    if(group_has_used_ordered_bin[i]) {
      ordered_bin_indices_.push_back(i);
    }
  }
}

} // namespace LightGBM