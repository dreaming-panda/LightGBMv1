/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_SYMMETRIC_FEATURE_HISTOGRAM_H_
#define LIGHTGBM_TREELEARNER_SYMMETRIC_FEATURE_HISTOGRAM_H_

#include <vector>

#include "../feature_histogram.hpp"

namespace LightGBM {

class SymmetricHistogramPool : public HistogramPool {
  public:
    SymmetricHistogramPool(const int num_threads, const int max_num_leaves):
      HistogramPool(), num_threads_(num_threads) {
      thread_left_sum_gradient_.resize(num_threads_);
      thread_left_sum_hessian_.resize(num_threads_);
      thread_left_count_.resize(num_threads_);
      thread_right_sum_gradient_.resize(num_threads_);
      thread_right_sum_hessian_.resize(num_threads_);
      thread_right_count_.resize(num_threads_);
      #pragma omp parallel for schedule(static) num_threads(num_threads_)
      for (int thread_id = 0; thread_id < num_threads_; ++thread_id) {
        thread_left_sum_gradient_[thread_id].resize(max_num_leaves, 0.0f);
        thread_left_sum_hessian_[thread_id].resize(max_num_leaves, 0.0f);
        thread_left_count_[thread_id].resize(max_num_leaves, 0);
        thread_right_sum_gradient_[thread_id].resize(max_num_leaves, 0.0f);
        thread_right_sum_hessian_[thread_id].resize(max_num_leaves, 0.0f);
        thread_right_count_[thread_id].resize(max_num_leaves, 0);
      }
    }

    static void FindBestThreaholdFromLevelHistograms(const int inner_feature_index,
        const std::vector<std::vector<int>>& paired_leaf_indices_in_cur_level,
        const std::vector<int>& leaf_indices_in_cur_level,
        int* best_inner_feature_index, int* best_threshold,
        double* best_gain, int* best_direction, std::vector<int>* thread_leaf_in_level_should_be_split,
        int8_t* default_left);

    void GetSplitLeafOutput(const int leaf_index, const int feature_index, const int threshold, const int direction,
      double* left_output, double* right_output, double* left_sum_gradient, double* left_sum_hessian,
      double* right_sum_gradient, double* right_sum_hessian, data_size_t* left_cnt, data_size_t* right_cnt,
      double* gain) const;

  private:
    template <bool USE_L1, bool REVERSE, bool SKIP_DEFAULT_BIN, bool NA_AS_MISSING>
    void FindBestThreaholdFromLevelHistogramsInner(
      const std::vector<double>& parent_sum_gradient,
      const std::vector<double>& parent_sum_hessian,
      const std::vector<data_size_t>& parent_num_data,
      const std::vector<double>& parent_output,
      const std::vector<double>& min_gain_shift,
      const std::vector<hist_t*>& level_histogram_ptr,
      const int num_leaves_in_cur_level,
      const int inner_feature_index,
      const int thread_id,
      int* best_threshold,
      double* best_gain,
      int8_t* best_default_left);

    const int num_threads_;

    std::vector<std::vector<double>> thread_left_sum_gradient_;
    std::vector<std::vector<double>> thread_left_sum_hessian_;
    std::vector<std::vector<data_size_t>> thread_left_count_;
    std::vector<std::vector<double>> thread_right_sum_gradient_;
    std::vector<std::vector<double>> thread_right_sum_hessian_;
    std::vector<std::vector<data_size_t>> thread_right_count_;
};

void SymmetricHistogramPool::FindBestThreaholdFromLevelHistograms(const int inner_feature_index,
  const std::vector<std::vector<int>>& paired_leaf_indices_in_cur_level,
  const std::vector<int>& leaf_indices_in_cur_level,
  int* best_inner_feature_index, int* best_threshold,
  double* best_gain, int* best_direction,
  std::vector<int>* thread_leaf_in_level_should_be_split,
  int8_t* default_left, const int thread_id,
  const int num_leaves_in_cur_level) {
  const int num_pairs = static_cast<int>(paired_leaf_indices_in_cur_level.size());
  for (int i = 0; i < num_pairs; ++i) {
    const std::vector<int>& pair = paired_leaf_indices_in_cur_level[i];
    if (pair.size() == 2) {
      const int smaller_leaf_index_in_level = pair[0];
      const int larger_leaf_index_in_level = pair[1];
      const int smaller_leaf_index = leaf_indices_in_cur_level[smaller_leaf_index_in_level];
      const int larger_leaf_index = leaf_indices_in_cur_level[larger_leaf_index_in_level];
      FeatureHistogram *smaller_leaf_histogram = nullptr, *larger_leaf_histogrma = nullptr;
      const bool get_smaller_leaf_histogram = Get(smaller_leaf_index, &smaller_leaf_histogram);
      const bool get_larger_leaf_histogram = Get(larger_leaf_index, &larger_leaf_histogrma);
      CHECK(get_smaller_leaf_histogram);
      CHECK(get_larger_leaf_histogram);
      larger_leaf_histogrma->Subtract(*smaller_leaf_histogram);
    }
  }
  std::vector<hist_t*> level_histogram_ptr(num_leaves_in_cur_level, nullptr);
  for (int leaf_index_in_level = 0; leaf_index_in_level < num_leaves_in_cur_level; ++leaf_index_in_level) {
    const int leaf_index = leaf_indices_in_cur_level[leaf_index_in_level];
    FeatureHistogram* feature_histogram = nullptr;
    const bool get = Get(leaf_index, &feature_histogram);
    level_histogram_ptr[leaf_index_in_level] = feature_histogram->RawData();
    CHECK(get);
  }
  
  FindBestThreaholdFromLevelHistogramsInner()
}

#define GET_GRAD(hist, i) hist[(i) << 1]
#define GET_HESS(hist, i) hist[((i) << 1) + 1]

template <bool USE_L1, bool REVERSE, bool SKIP_DEFAULT_BIN, bool NA_AS_MISSING>
void SymmetricHistogramPool::FindBestThreaholdFromLevelHistogramsInner(
  const std::vector<double>& parent_sum_gradient,
  const std::vector<double>& parent_sum_hessian,
  const std::vector<data_size_t>& parent_num_data,
  const std::vector<double>& parent_output,
  const std::vector<double>& min_gain_shift,
  const std::vector<hist_t*>& level_histogram_ptr,
  const int num_leaves_in_cur_level,
  const int inner_feature_index,
  const int thread_id,
  int* best_threshold,
  double* best_gain,
  int8_t* best_default_left) {

  const auto& meta = feature_metas_[inner_feature_index];
  const int8_t offset = meta.offset;
  const double cnt_factor = num_data / sum_hessian;

  std::vector<double>& left_sum_gradient_ref = thread_left_sum_gradient_[thread_id];
  std::vector<double>& left_sum_hessian_ref = thread_left_sum_hessian_[thread_id];
  std::vector<double>& left_count_ref = thread_left_count_[thread_id];
  std::vector<double>& right_sum_gradient_ref = thread_right_sum_gradient_[thread_id];
  std::vector<double>& right_sum_hessian_ref = thread_right_sum_hessian_[thread_id];
  std::vector<double>& right_count_ref = thread_right_count_[thread_id];

  if (REVERSE) {
    int t = meta.num_bin - 1 - offset - static_cast<int>(NA_AS_MISSING);
    const int t_end = 1 - offset;

    for (int leaf_index_in_level = 0; leaf_index_in_level < num_leaves_in_cur_level; ++leaf_index_in_level) {
      left_sum_gradient_ref[leaf_index_in_level] = 0.0f;
      left_sum_hessian_ref[leaf_index_in_level] = kEpsilon;
      left_count_ref[leaf_index_in_level] = 0;
      right_sum_gradient_ref[leaf_index_in_level] = 0.0f;
      right_sum_hessian_ref[leaf_index_in_level] = kEpsilon;
      right_count_ref[leaf_index_in_level] = 0;
    }

    for (; t >= t_end; --t) {
      if (SKIP_DEFAULT_BIN) {
        if ((t + offset) == static_cast<int>(meta.default_bin)) {
          continue;
        }
      }
      double threshold_gain = 0.0f;
      bool has_leaf_to_split = false;
      bool is_valid = false;
      for (int leaf_index_in_level = 0; leaf_index_in_level < num_leaves_in_cur_level; ++leaf_index_in_level) {
        const auto grad = GET_GRAD(level_histogram_ptr[leaf_index_in_level], t);
        const auto hess = GET_HESS(level_histogram_ptr[leaf_index_in_level], t);
        const data_size_t cnt =
            static_cast<data_size_t>(Common::RoundInt(hess * cnt_factor));
        right_sum_gradient_ref[leaf_index_in_level] += grad;
        right_sum_hessian_ref[leaf_index_in_level] += hess;
        right_count_ref[leaf_index_in_level] += cnt;
        if (right_count_ref[leaf_index_in_level] < meta.config->min_data_in_leaf ||
          right_sum_hessian_ref[leaf_index_in_level] < mdata.config->min_sum_hessian_in_leaf) {
          has_leaf_to_split = true;
          continue;
        }
        left_count_ref[leaf_index_in_level] = parent_num_data[leaf_index_in_level] - right_count_ref[leaf_index_in_level];
        if (left_count_ref[leaf_index_in_level] < meta.config->min_data_in_leaf) {
          continue;
        }
        left_sum_hessian_ref[leaf_index_in_level] = parent_sum_hessian[leaf_index_in_level] - right_sum_hessian_ref[leaf_index_in_level];
        if (left_sum_hessian_ref[leaf_index_in_level] < meta.config->min_sum_hessian_in_leaf) {
          continue;
        }
        has_leaf_to_split = true;
        left_sum_gradient_ref[leaf_index_in_level] = parent_sum_gradient[leaf_index_in_level] - right_sum_gradient_ref[leaf_index_in_level];
        double current_gain = GetSplitGains<false, USE_L1, false, false>(
            left_sum_gradient_ref[leaf_index_in_level],
            left_sum_hessian_ref[leaf_index_in_level],
            right_sum_gradient_ref[leaf_index_in_level],
            right_sum_hessian_ref[leaf_index_in_level],
            meta.config->lambda_l1,
            meta.config->lambda_l2,
            meta.config->max_delta_step,
            nullptr, meta.monotone_type, meta.config->path_smooth,
            left_count_ref[leaf_index_in_level],
            right_count_ref[leaf_index_in_level],
            parent_output[leaf_index_in_level]);
        if (current_gain <= min_gain_shift[leaf_index_in_level]) {
          continue;
        }
        threshold_gain += current_gain;
        is_valid = true;
      }
      if (!has_leaf_to_split) {
        break;
      }
      if (is_valid) {
        if (threshold_gain > *best_gain) {
          *best_gain = threshold_gain;
          *best_threshold = t - 1 + offset;
          *best_default_left = REVSERSE;
        }
      }
    }
  } else {
    for (int leaf_index_in_level = 0; leaf_index_in_level < num_leaves_in_cur_level; ++leaf_index_in_level) {
      left_sum_gradient_ref[leaf_index_in_level] = 0.0f;
      left_sum_hessian_ref[leaf_index_in_level] = kEpsilon;
      left_count_ref[leaf_index_in_level] = 0;
      
      int t = 0;
      const int t_end = meta.num_bin - 2 - offset;

      if (NA_AS_MISSING) {
        if (offset == 1) {
          for (int leaf_index_in_level = 0; leaf_index_in_level < num_leaves_in_cur_level; ++leaf_index_in_level) {
            left_sum_gradient_ref[leaf_index_in_level] = parent_sum_gradient[leaf_index_in_level];
            left_sum_hessian_ref[leaf_index_in_level] = parent_sum_hessian[leaf_index_in_level];
            left_count_ref[leaf_index_in_level] = parent_num_data[leaf_index_in_level];
            for (int i = 0; i < meta.num_bin - 1; ++i) {
              const auto grad = GET_GRAD(level_histogram_ptr[leaf_index_in_level], i);
              const auto hess = GET_HESS(level_histogram_ptr[leaf_index_in_level], i);
              const data_size_t cnt =
                static_cast<data_size_t>(Common::RoundInt(hess * cnt_factor));
              left_sum_gradient_ref[leaf_index_in_level] -= grad;
              left_sum_hessian_ref[leaf_index_in_level] -= hess;
              left_count_ref[leaf_index_in_level] -= cnt;
            }
          }
          t -= 1;
        }
      }

      for (; t <= t_end; ++t) {
        if (SKIP_DEFAULT_BIN) {
          if ((t + offset) == static_cast<int>(meta.default_bin)) {
            continue;
          }
        }
        double threshold_gain = 0.0f;
        bool has_leaf_to_split = false;
        bool is_valid = false;
        if (t >= 0) {
          for (int leaf_index_in_level = 0; leaf_index_in_level < num_leaves_in_cur_level; ++leaf_index_in_level) {
            const auto grad = GET_GRAD(level_histogram_ptr[leaf_index_in_level], t);
            const auto hess = GET_HESS(level_histogram_ptr[leaf_index_in_level], t);
            const data_size_t cnt =
                static_cast<data_size_t>(Common::RoundInt(hess * cnt_factor));
            left_sum_gradient_ref[leaf_index_in_level] += grad;
            left_sum_hessian_ref[leaf_index_in_level] +- hess;
            left_count_ref[leaf_index_in_level] += cnt;

            if (left_count_ref[leaf_index_in_level] < meta.config->min_data_in_leaf ||
              left_sum_hessian_ref[leaf_index_in_level] < meta.config->min_sum_hessian_in_leaf) {
              has_leaf_to_split = true;
              continue;
            }

            right_count_ref[leaf_index_in_level] = parent_num_data[leaf_index_in_level] - left_count_ref[leaf_index_in_level];
            if (right_count_ref[leaf_index_in_level] < meta.config->min_data_in_leaf) {
              continue;
            }

            right_sum_hessian_ref[leaf_index_in_level] = parent_sum_hessian[leaf_index_in_level] - left_sum_hessian_ref[leaf_index_in_level];
            if (right_sum_hessian_ref[leaf_index_in_level] < meta.config->min_sum_hessian_in_level) {
              continue;
            }
            has_leaf_to_split = true;
            right_sum_gradient_ref[leaf_index_in_level] = parent_sum_gradient[leaf_index_in_level] - left_sum_gradient_ref[leaf_index_in_level];

            double current_gain = GetSplitGains<false, USE_L1, false, false>(
              left_sum_gradient_ref[leaf_index_in_level],
              left_sum_hessian_ref[leaf_index_in_level],
              right_sum_gradient_ref[leaf_index_in_level],
              right_sum_hessian_ref[leaf_index_in_level],
              meta.config->lambda_l1,
              meta.config->lambda_l2,
              meta.config->max_delta_step,
              nullptr, meta.monotone_type, meta.config->path_smooth,
              left_count_ref[leaf_index_in_level],
              right_count_ref[leaf_index_in_level],
              parent_output[leaf_index_in_level]);
            if (current_gain <= min_gain_shift[leaf_index_in_level]) {
              continue;
            }
            threshold_gain += current_gain;
            is_valid = true;
          }
          if (!has_leaf_to_split) {
            break;
          }
          if (is_valid) {
            if (threshold_gain > *best_gain) {
              *best_gain = threshold_gain;
              *best_threshold = t + offset;
              *best_default_left = REVSERSE;
            }
          }
        }
      }
    }
  }
}

#undef GET_GRAD
#undef GET_HESS

}  // namespace LightGBM

#endif  // LIGHTGBM_TREELEARNER_SYMMETRIC_FEATURE_HISTOGRAM_H_
