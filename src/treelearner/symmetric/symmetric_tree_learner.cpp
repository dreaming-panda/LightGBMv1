/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "symmetric_tree_learner.hpp"

namespace LightGBM {

SymmetricTreeLearner::SymmetricTreeLearner(const Config* config): 
SerialTreeLearner(config), symmetric_data_partition_(config) {}

Tree* SymmetricTreeLearner::Train(const score_t* gradients, const score_t *hessians, bool is_first_tree) {
  for (int depth = 0; depth < config_->max_depth; ++depth) {
    PrepareLevelHistograms(depth);
    // construct and subtract
    symmetric_data_partition_.ConstructLevelHistograms(&level_feature_histograms_, train_data_);
    // find best splits
    FindBestLevelSplits();
    SplitLevel();
  }
}

void SymmetricTreeLearner::PrepareLevelHistograms(const int depth) {
  const int num_leaves_in_level = 1 << depth;
  level_feature_histograms_.resize(leaf_ids_in_current_level_.size(), nullptr);
  for (size_t i = 0; i < leaf_ids_in_current_level_.size(); ++i) {
    const int leaf_id = leaf_ids_in_current_level_[i];
    const bool get = symmetric_histogram_pool_.Get(leaf_id, &level_feature_histograms_[i]);
    if (!get) {
      // TODO(shiyu1994): handle the case when the feature histogram cache is not enough
    }
  }
}

}  // namespace LightGBM
