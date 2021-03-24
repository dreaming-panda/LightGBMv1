/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_SYMMETRIC_DATA_PARTITION_HPP_
#define LIGHTGBM_TREELEARNER_SYMMETRIC_DATA_PARTITION_HPP_

#include <LightGBM/config.h>
#include <LightGBM/dataset.h>

#include "symmetric_feature_histogram.hpp"

namespace LightGBM {

class SymmetricDataPartition {
 public:
  SymmetricDataPartition(data_size_t num_data);

  void ConstructLevelHistograms(std::vector<FeatureHistogram*>* level_feature_histogram,
                                const Dataset* train_data,
                                std::vector<uint8_t> is_feature_group_used,
                                const score_t* gradients, const score_t* hessians) const;
 private:
  void GetUsedFeatureGroups(const std::vector<uint8_t>& is_feature_used,
                            const Dataset* train_data,
                            std::vector<int>* used_feature_groups) const;

  std::vector<uint32_t> data_index_to_small_leaf_index_;
  std::vector<data_size_t> data_indices_in_small_leaf_;
  data_size_t num_data_in_small_leaf_;
  const data_size_t num_data_;
  bool is_col_wise_;

  std::vector<int> leaf_count_;
  std::vector<int> level_leaf_index_to_real_leaf_index_;
  const int num_small_leaf_in_level_;
};

} //  namespace LightGBM

#endif //  LIGHTGBM_TREELEARNER_SYMMETRIC_DATA_PARTITION_HPP_
