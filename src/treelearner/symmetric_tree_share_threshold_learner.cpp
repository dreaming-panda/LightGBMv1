/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "serial_tree_learner.h"
#include <queue>

namespace LightGBM {

Tree* SymmetricTreeShareThresholdLearner::Train(const score_t* gradients, const score_t *hessians, bool is_constant_hessian,
              Json& /*forced_split_json*/) {
  gradients_ = gradients;
  hessians_ = hessians;
  is_constant_hessian_ = is_constant_hessian;
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
  const BinMapper* bin_mapper = nullptr;
  for(int level = 0; level < config_->max_depth; ++level) {
      std::queue<int> next_level_leaf_queue;
      if(level == 0) {
          BeforeFindBestSplit(tree.get(), left_leaf, right_leaf);
          FindBestSplits();
          for(int feature_index = 0; feature_index < num_features_; ++feature_index) {
              level_splits.push_back(splits_per_leaf_[feature_index]);
          }
          std::sort(level_splits.begin(), level_splits.end(), [] (SplitInfo a, SplitInfo b) { return a.gain > b.gain; });
          int best_leaf = 0;
          const SplitInfo& best_leaf_split_info = best_split_per_leaf_[best_leaf];
          if (best_leaf_split_info.gain <= 0.0) {
            is_left_right_update = false;
            Log::Warning("No further splits with positive gain, best gain: %f", best_leaf_split_info.gain);
            continue;
          }
          Split(tree.get(), best_leaf, &left_leaf, &right_leaf);
          bin_mapper = train_data_->FeatureBinMapper(train_data_->InnerFeatureIndex(level_splits[0].feature));
          next_feature_threshold_gain_.clear();
          next_feature_threshold_gain_.resize(bin_mapper->num_bin(), 0.0);
          next_feature_threshold_split_info_.clear();
          next_feature_threshold_split_info_.resize(2);
          for(size_t i = 0; i < 2; ++i) {
            next_feature_threshold_split_info_[i].resize(bin_mapper->num_bin());
          }
          is_left_right_update = true;
          next_level_leaf_queue.push(left_leaf);
          next_level_leaf_queue.push(right_leaf);
          cur_leaf_id_in_level_ = 0;
      }
      else {
        feature = level - 1 >= num_features_ ? level_splits.back().feature : level_splits[level - 1].feature;
        //Log::Warning("feature 1 %d", feature);
        level_inner_feature_index = train_data_->InnerFeatureIndex(feature);
        bin_mapper = train_data_->FeatureBinMapper(level_inner_feature_index);
        CHECK(bin_mapper->num_bin() == static_cast<int>(next_feature_threshold_gain_.size()));
        int level_size = static_cast<int>(level_leaf_queue.size());
        while(!level_leaf_queue.empty()) {
            int best_leaf = level_leaf_queue.front();
            int right_inner_feature_index = right_leaf == -1 ? -1 : level_inner_feature_index;
            const int node_in_level = level_size - static_cast<int>(level_leaf_queue.size());
            if(is_left_right_update) {
              BeforeFindBestSplit(tree.get(), left_leaf, right_leaf);
              FindBestSplitForFeature(left_leaf, right_leaf, level_inner_feature_index, right_inner_feature_index);
              cur_leaf_id_in_level_ += 2;
            }

            if(node_in_level == 0) {
              feature_threshold_gain_ = next_feature_threshold_gain_;
              feature_threshold_split_info_ = next_feature_threshold_split_info_;

              feature = level >= num_features_ ? level_splits.back().feature : level_splits[level].feature;
              //Log::Warning("feature 2 %d", feature);
              level_inner_feature_index = train_data_->InnerFeatureIndex(feature);
              bin_mapper = train_data_->FeatureBinMapper(level_inner_feature_index);

              next_feature_threshold_gain_.clear();
              next_feature_threshold_gain_.resize(bin_mapper->num_bin(), 0.0);
              next_feature_threshold_split_info_.clear();
              next_feature_threshold_split_info_.resize(2 * level_leaf_queue.size());
              for(size_t i = 0; i < 2 * level_leaf_queue.size(); ++i) {
                next_feature_threshold_split_info_[i].resize(bin_mapper->num_bin());
              }
              SetShareThreshold(level_leaf_queue, level_splits[level - 1].feature);
              CHECK(cur_leaf_id_in_level_ == level_size);
              cur_leaf_id_in_level_ = 0;
            }

            //const SplitInfo& best_leaf_split_info = best_split_per_leaf_[best_leaf];
            level_leaf_queue.pop();

            //if (best_leaf_split_info.gain <= 0.0) {
              //Log::Warning("No further splits with positive gain, best gain: %f", best_leaf_split_info.gain);
            //  is_left_right_update = false;
            //  continue;
            //}
            
            Split(tree.get(), best_leaf, &left_leaf, &right_leaf);
            is_left_right_update = true;

            next_level_leaf_queue.push(left_leaf);
            next_level_leaf_queue.push(right_leaf);
        }
        //Log::Warning("level %d", level);
      }

      level_leaf_queue = next_level_leaf_queue;
      if(level_leaf_queue.empty()) {
        //Log::Warning("stop at level %d", level);
        break;
      }
  }
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
  if(right_leaf != -1 && left_leaf_data_count >= right_leaf_data_count) {
    smaller_inner_feature_index = right_inner_feature_index;
    larger_inner_feature_index = left_inner_feature_index;
    smaller_in_level = cur_leaf_id_in_level_ + 1;
    larger_in_level = cur_leaf_id_in_level_;
  }

  if(smaller_inner_feature_index != -1) {
    // construct smaller leaf
    is_feature_used[smaller_inner_feature_index] = 1;
    HistogramBinEntry* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - 1;
    train_data_->ConstructHistograms(is_feature_used,
                                    smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
                                    smaller_leaf_splits_->LeafIndex(),
                                    ordered_bins_, gradients_, hessians_,
                                    ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
                                    ptr_smaller_leaf_hist_data);

    is_feature_used[smaller_inner_feature_index] = 0;
  }

  if(larger_inner_feature_index != -1) {
    //construct larger leaf
    is_feature_used[larger_inner_feature_index] = 1;
    HistogramBinEntry* ptr_larger_leaf_hist_data = larger_leaf_histogram_array_[0].RawData() - 1;
    train_data_->ConstructHistograms(is_feature_used,
                                      larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
                                      larger_leaf_splits_->LeafIndex(),
                                      ordered_bins_, gradients_, hessians_,
                                      ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
                                      ptr_larger_leaf_hist_data);
    is_feature_used[larger_inner_feature_index] = 0;
  }

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
  double best_gain = feature_threshold_gain_[0];
  for(uint32_t i = 1; i < feature_threshold_gain_.size(); ++i) {
    if(feature_threshold_gain_[i] > best_gain) {
      best_gain = feature_threshold_gain_[i];
      best_threshold = i;
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
    best_split_per_leaf_[leaf] = feature_threshold_split_info_[cur_leaf_id_in_level][best_threshold];
    best_split_per_leaf_[leaf].feature = feature;
    SplitInfo& split_info = best_split_per_leaf_[leaf];
    //Log::Warning("feature %d, left_sum_gradient %f, left_sum_hessian %f, right_sum_gradient %f, right_sum_hessian %f", split_info.feature, split_info.left_sum_gradient, split_info.left_sum_hessian, split_info.right_sum_gradient, split_info.right_sum_hessian);
    split_info.left_output = FeatureHistogram::CalculateSplittedLeafOutput(split_info.left_sum_gradient, split_info.left_sum_hessian, config_->lambda_l1, config_->lambda_l2, config_->max_delta_step);
    split_info.right_output = FeatureHistogram::CalculateSplittedLeafOutput(split_info.right_sum_gradient, split_info.right_sum_hessian, config_->lambda_l1, config_->lambda_l2, config_->max_delta_step);
    //Log::Warning("left_out %f, right_output %f, gain %f", split_info.left_output, split_info.right_output, split_info.gain);
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
      Log::Warning("sum_gradients %f, tmp_sum_gradients %f", sum_gradients_, tmp_sum_gradients);
    }
    CHECK(std::fabs(sum_gradients_ - tmp_sum_gradients) <= 1e-6);
    if(std::fabs(sum_gradients_ - tmp_sum_gradients) > 1e-6) {
      Log::Warning("sum_gradients %f, tmp_sum_gradients %f", sum_gradients_, tmp_sum_gradients);
    }
    CHECK(std::fabs(sum_hessians_ - tmp_sum_hessians) <= 1e-6);
  }
}

} // namespace LightGBM