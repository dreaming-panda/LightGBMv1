/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "symmetric_data_partition.hpp"

namespace LightGBM {

SymmetricDataPartition::SymmetricDataPartition(data_size_t num_data, int max_num_leaves, int num_threads):
num_data_(num_data), num_threads_(num_threads) {
  ordered_small_leaf_index_.resize(num_data_, 0);
  data_indices_in_small_leaf_.resize(num_data_, 0);
  num_data_in_small_leaf_ = num_data_;
  is_col_wise_ = true;

  data_index_to_leaf_index_.resize(num_data_, 0);
  leaf_count_.resize(max_num_leaves, 0);
  thread_leaf_count_.resize(num_threads_);
  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    thread_leaf_count_[thread_id].resize(max_num_leaves, 0);
  }
  thread_data_in_small_leaf_count_.resize(num_threads_, 0);
  left_child_index_.resize(max_num_leaves, -1);
  right_child_index_.resize(max_num_leaves, -1);
  left_child_smaller_.resize(max_num_leaves, false);
  is_data_in_small_leaf_.resize(num_data_, 0);
  small_leaf_positions_.resize(max_num_leaves, -1);
}

void SymmetricDataPartition::Init() {
  num_data_in_small_leaf_ = num_data_;
  num_leaf_in_level_ = 1;
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (data_size_t i = 0; i < num_data_; ++i) {
    data_index_to_leaf_index_[i] = 0;
    data_indices_in_small_leaf_[i] = i;
    ordered_small_leaf_index_[i] = 0;
  }
  num_small_leaf_ = 0;
}

void SymmetricDataPartition::ConstructLevelHistograms(
  std::vector<FeatureHistogram*>* level_feature_histogram,
  const Dataset* train_data,
  const std::vector<int8_t>& is_feature_used,
  const score_t* gradients, const score_t* hessians) const {
  int num_threads = OMP_NUM_THREADS();
  if (is_col_wise_) {
    std::vector<int> used_feature_groups;
    int used_multi_val_feature_group = -1;
    GetUsedFeatureGroups(is_feature_used, train_data, &used_feature_groups, &used_multi_val_feature_group);
    const int num_used_dense_feature_groups = static_cast<int>(used_feature_groups.size());
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < num_used_dense_feature_groups; ++i) {
      const int group_index = used_feature_groups[i];
      std::vector<hist_t*> group_hist_ptr;
      for (int j = 0; j < num_small_leaf_; ++j) {
        FeatureHistogram* feature_histograms = (*level_feature_histogram)[small_leaf_positions_[j]];
        hist_t* hist_ptr = feature_histograms[train_data->group_feature_start(group_index)].RawData() - 1;
        group_hist_ptr.emplace_back(hist_ptr);
      }
      train_data->ConstructSymmetricLevelHistogram<uint32_t>(group_index, group_hist_ptr, gradients, hessians,
        num_data_in_small_leaf_, data_indices_in_small_leaf_.data(), ordered_small_leaf_index_.data());
    }
  } else {
    Log::Fatal("symmetric tree with row-wise histogram construction is currently unsupported.");
  }
}

void SymmetricDataPartition::GetUsedFeatureGroups(
  const std::vector<int8_t>& is_feature_used, const Dataset* train_data,
  std::vector<int>* used_dense_feature_groups,
  int* used_multi_val_feature_group) const {
  std::vector<int>& used_feature_groups_ref = *used_dense_feature_groups;
  // assuming that the group index monotonically increases with the feature index
  for (int feature_index = 0; feature_index < train_data->num_features(); ++feature_index) {
    const int group_index = train_data->Feature2Group(feature_index);
    if (!train_data->IsMultiGroup(group_index)) {
      if (used_feature_groups_ref.empty() || used_dense_feature_groups->back() != group_index) {
        if (is_feature_used[feature_index]) {
          used_feature_groups_ref.emplace_back(group_index);
        }
      }
    } else {
      // assume that there's only multi value group
      *used_multi_val_feature_group = group_index;
    }
  }
}

#define SplitLevelLeafIndices_ARGS train_data,\
  inner_feature_index,\
  uint_threshold,\
  leaf_should_be_split,\
  level_split_info,\
  most_freq_bin,\
  max_bin,\
  zero_bin\

void SymmetricDataPartition::Split(const Dataset* train_data, const int inner_feature_index,
    const int threshold, const int8_t default_left,
    const std::vector<int8_t>& leaf_should_be_split,
    const std::vector<SplitInfo>& level_split_info) {
  const BinMapper* bin_mapper = train_data->FeatureBinMapper(inner_feature_index);
  const uint32_t most_freq_bin = bin_mapper->GetMostFreqBin();
  const uint32_t zero_bin = bin_mapper->ValueToBin(0.0f);
  const uint32_t max_bin = static_cast<uint32_t>(bin_mapper->num_bin() - 1);
  const uint32_t uint_threshold = static_cast<uint32_t>(threshold);
  const MissingType missing_type = bin_mapper->missing_type();
  if (default_left > 0) {
    if (most_freq_bin <= uint_threshold) {
      SplitLevelLeafIndices<true, true>(SplitLevelLeafIndices_ARGS);
    } else {
      SplitLevelLeafIndices<true, false>(SplitLevelLeafIndices_ARGS);
    }
  } else {
    if (most_freq_bin <= uint_threshold) {
      SplitLevelLeafIndices<false, true>(SplitLevelLeafIndices_ARGS);
    } else {
      SplitLevelLeafIndices<false, false>(SplitLevelLeafIndices_ARGS);
    }
  }
}

template <bool DEFAULT_LEFT, bool MOST_FREQ_LEFT>
  void SymmetricDataPartition::SplitLevelLeafIndices(
    const Dataset* train_data,
    const int inner_feature_index,
    const int32_t threshold,
    const std::vector<int8_t>& leaf_should_be_split,
    const std::vector<SplitInfo>& level_split_info,
    const int32_t most_freq_bin,
    const int32_t max_bin,
    const int32_t zero_bin) {
  if (missing_type == MissingType::None) {
    SplitLevelLeafIndicesInner<false, false, false, false, DEFAULT_LEFT, MOST_FREQ_LEFT>(SplitLevelLeafIndices_ARGS);
  } else if (missing_type == MissingType::Zero) {
    if (zero_bin == most_freq_bin) {
      SplitLevelLeafIndicesInner<true, false, true, false, DEFAULT_LEFT, MOST_FREQ_LEFT>(SplitLevelLeafIndices_ARGS);
    } else {
      SplitLevelLeafIndicesInner<true, false, false, false, DEFAULT_LEFT, MOST_FREQ_LEFT>(SplitLevelLeafIndices_ARGS);
    }
  } else {
    if (max_bin == most_freq_bin && most_freq_bin > 0) {
      SplitLevelLeafIndicesInner<false, true, false, true, DEFAULT_LEFT, MOST_FREQ_LEFT>(SplitLevelLeafIndices_ARGS);
    } else {
      SplitLevelLeafIndicesInner<false, true, false, false, DEFAULT_LEFT, MOST_FREQ_LEFT>(SplitLevelLeafIndices_ARGS);
    }
  }
}

#undef SplitLevelLeafIndices_ARGS

template <bool MISS_IS_ZERO, bool MISS_IS_NA, bool MFB_IS_ZERO,
            bool MFB_IS_NA, bool DEFAULT_LEFT, bool MOST_FREQ_LEFT>
void SymmetricDataPartition::SplitLevelLeafIndicesInner(
  const Dataset* train_data,
  const int inner_feature_index,
  const int32_t threshold,
  const std::vector<int8_t>& leaf_should_be_split,
  const std::vector<SplitInfo>& level_split_info,
  const int32_t most_freq_bin,
  const int32_t max_bin,
  const int32_t zero_bin) {
  std::vector<std::unique_ptr<BinIterator>> iters(num_threads_, nullptr);

  for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
    iters[thread_id].reset(train_data->FeatureIterator(inner_feature_index));
  }

  int small_leaf_index_in_next_level = 0;
  int large_leaf_index_in_next_level = 0;
  int non_split_leaf_index_in_next_level = 0;
  
  for (int leaf_index = 0; leaf_index < num_leaf_in_level_; ++leaf_index) {
    if (leaf_should_be_split[leaf_index] > 0) {
      ++large_leaf_index_in_next_level;
    }
    non_split_leaf_index_in_next_level += 2;
  }
  for (int leaf_index = 0; leaf_index < num_leaf_in_level_; ++leaf_index) {
    const SplitInfo& split_info = level_split_info[leaf_index];
    if (leaf_should_be_split[leaf_index] > 0) {
      if (split_info.left_count <= split_info.right_count) {
        left_child_index_[leaf_index] = small_leaf_index_in_next_level;
        right_child_index_[leaf_index] = large_leaf_index_in_next_level;
        left_child_smaller_[leaf_index] = true;
      } else {
        left_child_index_[leaf_index] = large_leaf_index_in_next_level;
        right_child_index_[leaf_index] = small_leaf_index_in_next_level;
        left_child_smaller_[leaf_index] = false;
      }
      small_leaf_positions_[small_leaf_index_in_next_level] = leaf_index;
      ++small_leaf_index_in_next_level;
      ++large_leaf_index_in_next_level;
    } else {
      left_child_index_[leaf_index] = non_split_leaf_index_in_next_level;
      ++non_split_leaf_index_in_next_level;
    }
  }
  const int num_leaf_in_next_level = non_split_leaf_index_in_next_level;
  num_small_leaf_ = small_leaf_index_in_next_level;
  Threading::For<data_size_t>(0, num_data_, 512,
    [this, &iters, most_freq_bin, zero_bin, max_bin]
    (int thread_id, data_size_t start, data_size_t end) {
      BinIterator* iter = iters[thread_id].get();
      iter->Reset(start);
      thread_data_in_small_leaf_count_[thread_id] = 0;
      std::vector<int>& leaf_count_ref = thread_leaf_count_[thread_id];
      int& data_in_small_leaf_count = thread_data_in_small_leaf_count_[thread_id];
      for (int i = 0; i < num_leaf_in_next_level; ++i) {
        leaf_count_ref[i] = 0;
      }
      for (data_size_t i = start; i < end; ++i) {
        const uint32_t bin = iter->Get(i);
        const int leaf_index = data_index_to_leaf_index_[i];
        if (leaf_should_be_split[leaf_index] > 0) {
          if ((MISS_IS_ZERO && !MFB_IS_ZERO && bin == zero_bin) ||
              (MISS_IS_NA && !MFB_IS_NA && bin == maxb)) {
            const new_leaf_index = DEFAULT_LEFT ?
              left_child_index_[leaf_index] :
              right_child_index_[leaf_index];
            ++leaf_count_ref[new_leaf_index];
            data_index_to_leaf_index[i] = new_leaf_index;
            if (DEFAULT_LEFT) {
              if (left_child_smaller_[leaf_index]) {
                ++data_in_small_leaf_count;
                is_data_in_small_leaf_[i] = 1;
              } else {
                is_data_in_small_leaf_[i] = 0;
              }
            } else {
              if (!left_child_smaller_[leaf_index]) {
                ++data_in_small_leaf_count;
                is_data_in_small_leaf_[i] = 1;
              } else {
                is_data_in_small_leaf_[i] = 0;
              }
            }
          } else if (bin == most_freq_bin) {
            if ((MISS_IS_ZERO && MFB_IS_ZERO) || (MISS_IS_NA && MFB_IS_NA)) {
              const new_leaf_index = DEFAULT_LEFT ?
                left_child_index_[leaf_index] :
                right_child_index_[leaf_index];
              ++leaf_count_ref[new_leaf_index];
              data_index_to_leaf_index[i] = new_leaf_index;
              if (DEFAULT_LEFT) {
                if (left_child_smaller_[leaf_index]) {
                  ++data_in_small_leaf_count;
                  is_data_in_small_leaf_[i] = 1;
                } else {
                  is_data_in_small_leaf_[i] = 0;
                }
              } else {
                if (!left_child_smaller_[leaf_index]) {
                  ++data_in_small_leaf_count;
                  is_data_in_small_leaf_[i] = 1;
                } else {
                  is_data_in_small_leaf_[i] = 0;
                }
              }
            } else {
              const new_leaf_index = MOST_FREQ_LEFT ?
                left_child_index_[leaf_index] :
                right_child_index_[leaf_index];
              ++leaf_count_ref[new_leaf_index];
              data_index_to_leaf_index[i] = new_leaf_index;
              if (MOST_FREQ_LEFT) {
                if (left_child_smaller_[leaf_index]) {
                  ++data_in_small_leaf_count;
                  is_data_in_small_leaf_[i] = 1;
                } else {
                  is_data_in_small_leaf_[i] = 0;
                }
              } else {
                if (!left_child_smaller_[leaf_index]) {
                  ++data_in_small_leaf_count;
                  is_data_in_small_leaf_[i] = 1;
                } else {
                  is_data_in_small_leaf_[i] = 0;
                }
              }
            }
          } else if (bin > threshold) {
            const new_leaf_index = right_child_index_[leaf_index];
            ++leaf_count_ref[new_leaf_index];
            data_index_to_leaf_index[i] = new_leaf_index;
            if (!left_child_smaller_[leaf_index]) {
              ++data_in_small_leaf_count;
              is_data_in_small_leaf_[i] = 1;
            } else {
              is_data_in_small_leaf_[i] = 0;
            }
          } else {
            const new_leaf_index = left_child_index_[leaf_index];
            ++leaf_count_ref[new_leaf_index];
            data_index_to_leaf_index[i] = new_leaf_index;
            if (left_child_smaller_[leaf_index]) {
              ++data_in_small_leaf_count;
              is_data_in_small_leaf_[i] = 1;
            } else {
              is_data_in_small_leaf_[i] = 0;
            }
          }
        } else {
          const new_leaf_index = left_child_smaller_[leaf_index];
          ++leaf_count_ref[new_leaf_index];
          data_index_to_leaf_index[i] = new_leaf_index;
          is_data_in_small_leaf_[i] = 0;
        }
      }
    });
  
  #pragma omp parallel for schedule(static) num_threads(num_threads_) if (num_leaf_in_next_level >= 512)
  for (int leaf_index = 0; leaf_index < num_leaf_in_next_level; ++leaf_index) {
    leaf_count_[leaf_index] = thread_leaf_count_[0][leaf_index];
    for (int thread_id = 1; thread_id < num_threads_; ++thread_id) {
      leaf_count_[leaf_index] += thread_leaf_count_[thread_id][leaf_index];
    }
  }

  for (int i = 1; i < num_threads_; ++i) {
    thread_data_in_small_leaf_count_[i] += thread_data_in_small_leaf_count_[i - 1];
  }
  Threading::For<data_size_t>(0, num_data_,
    [this] (int thread_id, data_size_t start, data_size_t end) {
      int thread_pos = thread_data_in_small_leaf_count_[thread_id];
      for (data_size_t i = start; i < end; ++i) {
        if (is_data_in_small_leaf_[i] > 0) {
          data_indices_in_small_leaf_[thread_pos] = i;
          ordered_small_leaf_index_[thread_pos] = static_cast<uint32_t>(data_index_to_leaf_index_[i]);
          ++thread_pos;
        }
      }
    });

  num_leaf_in_level_ = num_leaf_in_next_level;
}

} //  namespace LightGBM
