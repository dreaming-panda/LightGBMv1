/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_SYMMETRIC_FEATURE_HISTOGRAM_HPP_
#define LIGHTGBM_TREELEARNER_SYMMETRIC_FEATURE_HISTOGRAM_HPP_

#include <vector>

#include "../feature_histogram.hpp"
#include "../leaf_splits.hpp"

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

    void FindBestThresholdFromLevelHistograms(const int inner_feature_index,
      const std::vector<std::vector<int>>& paired_leaf_indices_in_cur_level,
      const std::vector<int>& leaf_indices_in_cur_level,
      const std::vector<std::unique_ptr<LeafSplits>>& parent_leaf_splits,
      int* best_inner_feature_index, int* best_threshold,
      double* best_gain, int8_t* default_left, const int thread_id,
      const int num_leaves_in_cur_level);

    void GetSplitLeafOutput(const int leaf_index, const int feature_index, const int threshold, const int8_t default_left,
      double* left_output, double* right_output, double* left_sum_gradient, double* left_sum_hessian,
      double* right_sum_gradient, double* right_sum_hessian, data_size_t* left_cnt, data_size_t* right_cnt,
      double* gain) const;

  private:
    template <bool USE_L1, bool REVERSE, bool SKIP_DEFAULT_BIN, bool NA_AS_MISSING>
    void FindBestThresholdFromLevelHistogramsInner(
      const std::vector<std::unique_ptr<LeafSplits>>& parent_leaf_splits,
      const std::vector<double>& min_gain_shift,
      const std::vector<hist_t*>& level_histogram_ptr,
      const int num_leaves_in_cur_level,
      const int inner_feature_index,
      const int thread_id,
      int* best_inner_feature_index,
      int* best_threshold,
      double* best_gain,
      int8_t* best_default_left);

    template <bool USE_MC, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
    static double GetSplitGains(double sum_left_gradients,
                                double sum_left_hessians,
                                double sum_right_gradients,
                                double sum_right_hessians, double l1, double l2,
                                double max_delta_step,
                                const FeatureConstraint* constraints,
                                int8_t monotone_constraint,
                                double smoothing,
                                data_size_t left_count,
                                data_size_t right_count,
                                double parent_output) {
      if (!USE_MC) {
        return GetLeafGain<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(sum_left_gradients,
                                                                  sum_left_hessians, l1, l2,
                                                                  max_delta_step, smoothing,
                                                                  left_count, parent_output) +
              GetLeafGain<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(sum_right_gradients,
                                                                  sum_right_hessians, l1, l2,
                                                                  max_delta_step, smoothing,
                                                                  right_count, parent_output);
      } else {
        double left_output =
            CalculateSplittedLeafOutput<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
                sum_left_gradients, sum_left_hessians, l1, l2, max_delta_step,
                constraints->LeftToBasicConstraint(), smoothing, left_count, parent_output);
        double right_output =
            CalculateSplittedLeafOutput<USE_MC, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
                sum_right_gradients, sum_right_hessians, l1, l2, max_delta_step,
                constraints->RightToBasicConstraint(), smoothing, right_count, parent_output);
        if (((monotone_constraint > 0) && (left_output > right_output)) ||
            ((monotone_constraint < 0) && (left_output < right_output))) {
          return 0;
        }
        return GetLeafGainGivenOutput<USE_L1>(
                  sum_left_gradients, sum_left_hessians, l1, l2, left_output) +
              GetLeafGainGivenOutput<USE_L1>(
                  sum_right_gradients, sum_right_hessians, l1, l2, right_output);
      }
    }

    template <bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
    static double GetLeafGain(double sum_gradients, double sum_hessians,
                              double l1, double l2, double max_delta_step,
                              double smoothing, data_size_t num_data, double parent_output) {
      if (!USE_MAX_OUTPUT && !USE_SMOOTHING) {
        if (USE_L1) {
          const double sg_l1 = ThresholdL1(sum_gradients, l1);
          return (sg_l1 * sg_l1) / (sum_hessians + l2);
        } else {
          return (sum_gradients * sum_gradients) / (sum_hessians + l2);
        }
      } else {
        double output = CalculateSplittedLeafOutput<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
            sum_gradients, sum_hessians, l1, l2, max_delta_step, smoothing, num_data, parent_output);
        return GetLeafGainGivenOutput<USE_L1>(sum_gradients, sum_hessians, l1, l2, output);
      }
    }

    template <bool USE_MC, bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
    static double CalculateSplittedLeafOutput(
        double sum_gradients, double sum_hessians, double l1, double l2,
        double max_delta_step, const BasicConstraint& constraints,
        double smoothing, data_size_t num_data, double parent_output) {
      double ret = CalculateSplittedLeafOutput<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>(
          sum_gradients, sum_hessians, l1, l2, max_delta_step, smoothing, num_data, parent_output);
      if (USE_MC) {
        if (ret < constraints.min) {
          ret = constraints.min;
        } else if (ret > constraints.max) {
          ret = constraints.max;
        }
      }
      return ret;
    }

    template <bool USE_L1, bool USE_MAX_OUTPUT, bool USE_SMOOTHING>
    static double CalculateSplittedLeafOutput(double sum_gradients,
                                              double sum_hessians, double l1,
                                              double l2, double max_delta_step,
                                              double smoothing, data_size_t num_data,
                                              double parent_output) {
      double ret;
      if (USE_L1) {
        ret = -ThresholdL1(sum_gradients, l1) / (sum_hessians + l2);
      } else {
        ret = -sum_gradients / (sum_hessians + l2);
      }
      if (USE_MAX_OUTPUT) {
        if (max_delta_step > 0 && std::fabs(ret) > max_delta_step) {
          ret = Common::Sign(ret) * max_delta_step;
        }
      }
      if (USE_SMOOTHING) {
        ret = ret * (num_data / smoothing) / (num_data / smoothing + 1) \
            + parent_output / (num_data / smoothing + 1);
      }
      return ret;
    }

    template <bool USE_L1>
    static double GetLeafGainGivenOutput(double sum_gradients,
                                        double sum_hessians, double l1,
                                        double l2, double output) {
      if (USE_L1) {
        const double sg_l1 = ThresholdL1(sum_gradients, l1);
        return -(2.0 * sg_l1 * output + (sum_hessians + l2) * output * output);
      } else {
        return -(2.0 * sum_gradients * output +
                (sum_hessians + l2) * output * output);
      }
    }

    static double ThresholdL1(double s, double l1) {
      const double reg_s = std::max(0.0, std::fabs(s) - l1);
      return Common::Sign(s) * reg_s;
    }

    const int num_threads_;

    std::vector<std::vector<double>> thread_left_sum_gradient_;
    std::vector<std::vector<double>> thread_left_sum_hessian_;
    std::vector<std::vector<data_size_t>> thread_left_count_;
    std::vector<std::vector<double>> thread_right_sum_gradient_;
    std::vector<std::vector<double>> thread_right_sum_hessian_;
    std::vector<std::vector<data_size_t>> thread_right_count_;
};

#define FindBestThresholdFromLevelHistogramsInner_ARGS \
  parent_leaf_splits,\
  min_gain_shift,\
  level_histogram_ptr,\
  num_leaves_in_cur_level,\
  inner_feature_index,\
  thread_id,\
  best_inner_feature_index,\
  best_threshold,\
  best_gain,\
  default_left

void SymmetricHistogramPool::FindBestThresholdFromLevelHistograms(const int inner_feature_index,
  const std::vector<std::vector<int>>& paired_leaf_indices_in_cur_level,
  const std::vector<int>& leaf_indices_in_cur_level,
  const std::vector<std::unique_ptr<LeafSplits>>& parent_leaf_splits,
  int* best_inner_feature_index, int* best_threshold,
  double* best_gain, int8_t* default_left, const int thread_id,
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
  std::vector<double> min_gain_shift(num_leaves_in_cur_level, 0.0f);
  const auto& meta = feature_metas_[inner_feature_index];
  for (int leaf_index_in_level = 0; leaf_index_in_level < num_leaves_in_cur_level; ++leaf_index_in_level) {
    const int leaf_index = leaf_indices_in_cur_level[leaf_index_in_level];
    FeatureHistogram* feature_histogram = nullptr;
    const bool get = Get(leaf_index, &feature_histogram);
    level_histogram_ptr[leaf_index_in_level] = feature_histogram->RawData();
    CHECK(get);
  }
  const bool use_l1 = meta.config->lambda_l1 > 0.0f;
  for (int leaf_index_in_level = 0; leaf_index_in_level < num_leaves_in_cur_level; ++leaf_index_in_level) {
    if (use_l1) {
      min_gain_shift[leaf_index_in_level] = GetLeafGain<true, false, false>(
        parent_leaf_splits[leaf_index_in_level]->sum_gradients(),
        parent_leaf_splits[leaf_index_in_level]->sum_hessians(),
        meta.config->lambda_l1, meta.config->lambda_l2,
        meta.config->max_delta_step, meta.config->path_smooth,
        parent_leaf_splits[leaf_index_in_level]->num_data_in_leaf(),
        parent_leaf_splits[leaf_index_in_level]->weight()) + meta.config->min_gain_to_split;
    } else {
      min_gain_shift[leaf_index_in_level] = GetLeafGain<false, false, false>(
        parent_leaf_splits[leaf_index_in_level]->sum_gradients(),
        parent_leaf_splits[leaf_index_in_level]->sum_hessians(),
        meta.config->lambda_l1, meta.config->lambda_l2,
        meta.config->max_delta_step, meta.config->path_smooth,
        parent_leaf_splits[leaf_index_in_level]->num_data_in_leaf(),
        parent_leaf_splits[leaf_index_in_level]->weight()) + meta.config->min_gain_to_split;
    }
  }
  if (meta.num_bin > 2 && meta.missing_type != MissingType::None) {
    if (meta.missing_type == MissingType::Zero) {
      if (use_l1) {
        FindBestThresholdFromLevelHistogramsInner<true, true, true, false>(FindBestThresholdFromLevelHistogramsInner_ARGS);
        FindBestThresholdFromLevelHistogramsInner<true, false, true, false>(FindBestThresholdFromLevelHistogramsInner_ARGS);
      } else {
        FindBestThresholdFromLevelHistogramsInner<false, true, true, false>(FindBestThresholdFromLevelHistogramsInner_ARGS);
        FindBestThresholdFromLevelHistogramsInner<false, false, true, false>(FindBestThresholdFromLevelHistogramsInner_ARGS);
      }
    } else {
      if (use_l1) {
        FindBestThresholdFromLevelHistogramsInner<true, true, false, true>(FindBestThresholdFromLevelHistogramsInner_ARGS);
        FindBestThresholdFromLevelHistogramsInner<true, false, false, true>(FindBestThresholdFromLevelHistogramsInner_ARGS);
      } else {
        FindBestThresholdFromLevelHistogramsInner<false, true, false, true>(FindBestThresholdFromLevelHistogramsInner_ARGS);
        FindBestThresholdFromLevelHistogramsInner<false, false, false, true>(FindBestThresholdFromLevelHistogramsInner_ARGS);
      }
    }
  } else {
    if (use_l1) {
      FindBestThresholdFromLevelHistogramsInner<true, true, false, false>(FindBestThresholdFromLevelHistogramsInner_ARGS);
    } else {
      FindBestThresholdFromLevelHistogramsInner<false, true, false, false>(FindBestThresholdFromLevelHistogramsInner_ARGS);
    }
    if (meta.missing_type == MissingType::NaN) {
      *default_left = 0;
    }
  }
}

#define GET_GRAD(hist, i) hist[(i) << 1]
#define GET_HESS(hist, i) hist[((i) << 1) + 1]

template <bool USE_L1, bool REVERSE, bool SKIP_DEFAULT_BIN, bool NA_AS_MISSING>
void SymmetricHistogramPool::FindBestThresholdFromLevelHistogramsInner(
  const std::vector<std::unique_ptr<LeafSplits>>& parent_leaf_splits,
  const std::vector<double>& min_gain_shift,
  const std::vector<hist_t*>& level_histogram_ptr,
  const int num_leaves_in_cur_level,
  const int inner_feature_index,
  const int thread_id,
  int* best_inner_feature_index,
  int* best_threshold,
  double* best_gain,
  int8_t* best_default_left) {

  const auto& meta = feature_metas_[inner_feature_index];
  const int8_t offset = meta.offset;
  std::vector<double> cnt_factors(num_leaves_in_cur_level, 0.0f);
  for (int i = 0; i < num_leaves_in_cur_level; ++i) {
    cnt_factors[i] = parent_leaf_splits[i]->num_data_in_leaf() / parent_leaf_splits[i]->sum_hessians();
  }

  std::vector<double>& left_sum_gradient_ref = thread_left_sum_gradient_[thread_id];
  std::vector<double>& left_sum_hessian_ref = thread_left_sum_hessian_[thread_id];
  std::vector<data_size_t>& left_count_ref = thread_left_count_[thread_id];
  std::vector<double>& right_sum_gradient_ref = thread_right_sum_gradient_[thread_id];
  std::vector<double>& right_sum_hessian_ref = thread_right_sum_hessian_[thread_id];
  std::vector<data_size_t>& right_count_ref = thread_right_count_[thread_id];

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
            static_cast<data_size_t>(Common::RoundInt(hess * cnt_factors[leaf_index_in_level]));
        right_sum_gradient_ref[leaf_index_in_level] += grad;
        right_sum_hessian_ref[leaf_index_in_level] += hess;
        right_count_ref[leaf_index_in_level] += cnt;
        if (right_count_ref[leaf_index_in_level] < meta.config->min_data_in_leaf ||
          right_sum_hessian_ref[leaf_index_in_level] < meta.config->min_sum_hessian_in_leaf) {
          has_leaf_to_split = true;
          continue;
        }
        left_count_ref[leaf_index_in_level] = parent_leaf_splits[leaf_index_in_level]->num_data_in_leaf() - right_count_ref[leaf_index_in_level];
        if (left_count_ref[leaf_index_in_level] < meta.config->min_data_in_leaf) {
          continue;
        }
        left_sum_hessian_ref[leaf_index_in_level] = parent_leaf_splits[leaf_index_in_level]->sum_hessians() - right_sum_hessian_ref[leaf_index_in_level];
        if (left_sum_hessian_ref[leaf_index_in_level] < meta.config->min_sum_hessian_in_leaf) {
          continue;
        }
        has_leaf_to_split = true;
        left_sum_gradient_ref[leaf_index_in_level] = parent_leaf_splits[leaf_index_in_level]->sum_gradients() - right_sum_gradient_ref[leaf_index_in_level];
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
            parent_leaf_splits[leaf_index_in_level]->weight());
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
          *best_inner_feature_index = inner_feature_index;
          *best_gain = threshold_gain;
          *best_threshold = t - 1 + offset;
          *best_default_left = REVERSE;
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
            left_sum_gradient_ref[leaf_index_in_level] = parent_leaf_splits[leaf_index_in_level]->sum_gradients();
            left_sum_hessian_ref[leaf_index_in_level] = parent_leaf_splits[leaf_index_in_level]->sum_hessians();
            left_count_ref[leaf_index_in_level] = parent_leaf_splits[leaf_index_in_level]->num_data_in_leaf();
            for (int i = 0; i < meta.num_bin - 1; ++i) {
              const auto grad = GET_GRAD(level_histogram_ptr[leaf_index_in_level], i);
              const auto hess = GET_HESS(level_histogram_ptr[leaf_index_in_level], i);
              const data_size_t cnt =
                static_cast<data_size_t>(Common::RoundInt(hess * cnt_factors[leaf_index_in_level]));
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
                static_cast<data_size_t>(Common::RoundInt(hess * cnt_factors[leaf_index_in_level]));
            left_sum_gradient_ref[leaf_index_in_level] += grad;
            left_sum_hessian_ref[leaf_index_in_level] += hess;
            left_count_ref[leaf_index_in_level] += cnt;

            if (left_count_ref[leaf_index_in_level] < meta.config->min_data_in_leaf ||
              left_sum_hessian_ref[leaf_index_in_level] < meta.config->min_sum_hessian_in_leaf) {
              has_leaf_to_split = true;
              continue;
            }

            right_count_ref[leaf_index_in_level] = parent_leaf_splits[leaf_index_in_level]->num_data_in_leaf() - left_count_ref[leaf_index_in_level];
            if (right_count_ref[leaf_index_in_level] < meta.config->min_data_in_leaf) {
              continue;
            }

            right_sum_hessian_ref[leaf_index_in_level] = parent_leaf_splits[leaf_index_in_level]->sum_hessians() - left_sum_hessian_ref[leaf_index_in_level];
            if (right_sum_hessian_ref[leaf_index_in_level] < meta.config->min_sum_hessian_in_leaf) {
              continue;
            }
            has_leaf_to_split = true;
            right_sum_gradient_ref[leaf_index_in_level] = parent_leaf_splits[leaf_index_in_level]->sum_gradients() - left_sum_gradient_ref[leaf_index_in_level];

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
              parent_leaf_splits[leaf_index_in_level]->weight());
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
              *best_inner_feature_index = inner_feature_index;
              *best_gain = threshold_gain;
              *best_threshold = t + offset;
              *best_default_left = REVERSE;
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

#endif  // LIGHTGBM_TREELEARNER_SYMMETRIC_FEATURE_HISTOGRAM_HPP_
