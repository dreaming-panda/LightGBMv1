/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_SYMMETRIC_FEATURE_HISTOGRAM_H_
#define LIGHTGBM_TREELEARNER_SYMMETRIC_FEATURE_HISTOGRAM_H_


#include "../feature_histogram.hpp"

namespace LightGBM {

class SymmetricHistogramPool : public HistogramPool {
  public:
    template <>
    void FindBestThreaholdFromLevelHistograms(const int inner_feature_index,
        const std::vector<std::vector<int>>& paired_leaf_indices_in_cur_level,
        int* best_inner_feature_index, int* best_threshold,
        double* best_gain, int* best_direction, std::vector<int>* thread_leaf_in_level_should_be_split);
};

}  // namespace LightGBM

#endif  // LIGHTGBM_TREELEARNER_SYMMETRIC_FEATURE_HISTOGRAM_H_