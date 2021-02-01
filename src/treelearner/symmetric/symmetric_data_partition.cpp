/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "symmetric_data_partition.hpp"

namespace LightGBM {

SymmetricDataPartition::SymmetricDataPartition(data_size_t num_data):
num_data_(num_data) {
  data_index_to_leaf_index_.resize(num_data, 0);
  is_col_wise_ = true;
}

void SymmetricDataPartition::ConstructLevelHistograms(
  std::vector<FeatureHistogram*>* level_feature_histogram,
  const Dataset* train_data,
  std::vector<uint8_t> is_feature_group_used,
  const score_t* gradients, const score_t* hessians) const {
  int num_threads = OMP_NUM_THREADS();
  if (is_col_wise_) {
    std::vector<int> used_feature_groups;
    GetUsedFeatureGroups(is_feature_group_used, train_data, &used_feature_groups);
    const int num_used_feature_groups = static_cast<int>(used_feature_groups.size());
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < num_used_feature_groups; ++i) {
      const int group_index = used_feature_groups[i];
      std::vector<hist_t*> group_hist_ptr;
      for (size_t i = 0; i < level_feature_histogram->size(); ++i) {
        FeatureHistogram* feature_histograms = level_feature_histogram->operator[](i);
        hist_t* hist_ptr = feature_histograms[0].RawData() - 1 +
           * train_data->FeatureGroupNumBin(group_index) * kHistOffset;
          group_hist_ptr.emplace_back(hist_ptr);
      }
      train_data->ConstructSymmetricLevelHistogram(group_index, group_hist_ptr, gradients, hessians,
        );
    }
  }
}

void SymmetricDataPartition::GetUsedFeatureGroups(
  const std::vector<uint8_t>& is_feature_used, const Dataset* train_data,
  std::vector<int>* used_feature_groups) const {
  std::vector<int>& used_feature_groups_ref = *used_feature_groups;
  // assuming that the group index monotonically increases with the feature index
  for (int feature_index = 0; feature_index < train_data->num_features(); ++feature_index) {
    const int group_index = train_data->Feature2Group(feature_index);
    if (used_feature_groups_ref.empty() || used_feature_groups->back() != group_index) {
      if (is_feature_used[feature_index]) {
        used_feature_groups_ref.emplace_back(group_index);
      }
    }
  }
}

} //  namespace LightGBM