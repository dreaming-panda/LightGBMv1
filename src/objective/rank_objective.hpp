/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace LightGBM {

/*!
 * \brief Objective function for Ranking
 */
class RankingObjective : public ObjectiveFunction {
 public:
  explicit RankingObjective(const Config& config)
      : seed_(config.objective_seed) {}

  explicit RankingObjective(const std::vector<std::string>&) : seed_(0) {}

  ~RankingObjective() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get weights
    weights_ = metadata.weights();
    // get boundries
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("Ranking tasks require query information");
    }
    num_queries_ = metadata.num_queries();
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
#pragma omp parallel for schedule(guided)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      const data_size_t start = query_boundaries_[i];
      const data_size_t cnt = query_boundaries_[i + 1] - query_boundaries_[i];
      GetGradientsForOneQuery(i, cnt, label_ + start, score + start,
                              gradients + start, hessians + start);
      if (weights_ != nullptr) {
        for (data_size_t j = 0; j < cnt; ++j) {
          gradients[start + j] =
              static_cast<score_t>(gradients[start + j] * weights_[start + j]);
          hessians[start + j] =
              static_cast<score_t>(hessians[start + j] * weights_[start + j]);
        }
      }
    }
  }

  void GetIntGradients(const double* score,
    score_t* gradients, score_t* hessians,
    int_score_t* int_gradients, int_score_t* int_hessians,
    double* grad_scale, double* hess_scale) override {
    GetGradients(score, gradients, hessians);
    UniformDiscretizeGradients(gradients, hessians, int_gradients, int_hessians, num_data_,
      grad_scale, hess_scale);
  }
  
  void ClipAndDiscretizeGradients(score_t* gradients, score_t* hessians,
    int_score_t* int_gradients, int_score_t* int_hessians,
    double* grad_scale, double* hess_scale) const {
    double sum_gradients = 0.0f, sum_sq_gradients = 0.0f;
    const int num_threads = OMP_NUM_THREADS();
    static int iter = 0;
    std::vector<double> thread_max_gradient(num_threads, std::fabs(gradients[0]));
    std::vector<double> thread_max_hessian(num_threads, std::fabs(hessians[0]));
    #pragma omp parallel for schedule(static) reduction(+:sum_gradients,sum_sq_gradients) num_threads(num_threads)
    for (int i = 0; i < num_data_; ++i) {
      const int thread_id = omp_get_thread_num();
      const double gradient = gradients[i];
      const double hessian = hessians[i];
      sum_gradients += gradient;
      sum_sq_gradients += gradient * gradient;
      const double gradient_abs = std::fabs(gradient);
      const double hessian_abs = std::fabs(hessian);
      if (gradient_abs > thread_max_gradient[thread_id]) {
        thread_max_gradient[thread_id] = gradient_abs;
      }
      if (hessian_abs > thread_max_hessian[thread_id]) {
        thread_max_hessian[thread_id] = hessian_abs;
      }
    }
    double max_gradient_abs = thread_max_gradient[0];
    double max_hessian_abs = thread_max_hessian[0];
    for (int thread_id = 1; thread_id < num_threads; ++thread_id) {
      if (thread_max_gradient[thread_id] > max_gradient_abs) {
        max_gradient_abs = thread_max_gradient[thread_id];
      }
      if (thread_max_hessian[thread_id] > max_hessian_abs) {
        max_hessian_abs = thread_max_hessian[thread_id];
      }
    }
    const double mean_gradient = sum_gradients / num_data_;
    const double std_gradient = std::sqrt(sum_sq_gradients / num_data_ - mean_gradient * mean_gradient);
    const double gradient_upper_bound = mean_gradient + 2 * std_gradient;
    const double gradient_lower_bound = mean_gradient - 2 * std_gradient;
    Log::Warning("gradient_upper_bound = %f, gradient_lower_bound = %f", gradient_upper_bound, gradient_lower_bound);
    Log::Warning("std_gradient = %f", std_gradient);
    const double gradient_bound = std::max(std::fabs(gradient_upper_bound), std::fabs(gradient_lower_bound));
    max_gradient_abs = std::min(max_gradient_abs, gradient_bound);

    Log::Warning("max_gradient_abs = %f, max_hessian_abs = %f", max_gradient_abs, max_hessian_abs);
    *grad_scale = max_gradient_abs / static_cast<double>(kIntGradBins / 2);
    *hess_scale = max_hessian_abs / static_cast<double>(kIntGradBins);
    Log::Warning("grad_scale = %.20f, hess_scale = %.20f", *grad_scale, *hess_scale);
    const double g_inverse_scale = 1.0f / (*grad_scale);
    const double h_inverse_scale = 1.0f / (*hess_scale);
    const double gs = *grad_scale;
    const double hs = *hess_scale;
    std::vector<std::mt19937> mt_generators;
    for (int i = 0; i < OMP_NUM_THREADS(); ++i) {
      mt_generators.emplace_back(i + iter);
    }
    std::vector<std::uniform_real_distribution<double>> dist;
    for (int i = 0; i < OMP_NUM_THREADS(); ++i) {
      dist.push_back(std::uniform_real_distribution<double>(0.0f, 1.0f));
    }
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_data_; ++i) {
      const int thread_id = omp_get_thread_num();
      score_t gradient = gradients[i];
      if (std::fabs(gradient) > max_gradient_abs) {
        gradient = gradient >= 0 ? max_gradient_abs : -max_gradient_abs;
      }
      const score_t hessian = hessians[i];
      const int_score_t int_grad = static_cast<int_score_t>(std::lround(gradient * g_inverse_scale));
      const int_score_t int_hess = static_cast<int_score_t>(std::lround(hessian * h_inverse_scale));
      const score_t gradient_low = int_grad * gs;
      const score_t gradient_high = gradient >= 0.0f ? (int_grad + 1) * gs : (int_grad - 1) * gs;
      const score_t hessian_low = int_hess * hs;
      const score_t hessian_high = (int_hess + 1) * hs;
      const score_t gradient_bias = (gradient - gradient_low) / (gradient_high - gradient_low);
      const score_t hessian_bias = (hessian - hessian_low) / (hessian_high - hessian_low);
      if (dist[thread_id](mt_generators[thread_id]) > gradient_bias) {
        int_gradients[i] = int_grad;
      } else {
        if (gradient < 0.0f) {
          CHECK(int_grad <= 0);
          int_gradients[i] = int_grad - 1;
        } else {
          CHECK(int_grad >= 0);
          int_gradients[i] = int_grad + 1;
        }
      }
      if (dist[thread_id](mt_generators[thread_id]) > hessian_bias) {
        int_hessians[i] = int_hess;
      } else {
        int_hessians[i] = int_hess + 1;
      }
    }
    ++iter;
  }

  void DiscretizeGradients(score_t* gradients, score_t* hessians,
    int_score_t* int_gradients, int_score_t* int_hessians,
    double* grad_scale, double* hess_scale) const override {
    double max_gradient = std::fabs(gradients[0]);
    double max_hessian = std::fabs(hessians[0]);
    int num_threads = OMP_NUM_THREADS();
    std::vector<double> thread_max_gradient(num_threads, max_gradient);
    std::vector<double> thread_max_hessian(num_threads, max_hessian);
    std::vector<double> thread_min_gradient(num_threads, max_gradient);
    std::vector<double> thread_min_hessian(num_threads, max_hessian);
    static int iter = 0;
    Threading::For<data_size_t>(0, num_data_, 1024,
      [gradients, hessians, &thread_max_gradient, &thread_max_hessian,
        &thread_min_gradient, &thread_min_hessian]
      (int, data_size_t start, data_size_t end) {
        int thread_id = omp_get_thread_num();
        for (data_size_t i = start; i < end; ++i) {
          double fabs_grad = std::fabs(gradients[i]);
          double fabs_hess = std::fabs(hessians[i]);
          if (fabs_grad > thread_max_gradient[thread_id]) {
            thread_max_gradient[thread_id] = fabs_grad;
          }
          if (fabs_hess > thread_max_hessian[thread_id]) {
            thread_max_hessian[thread_id] = fabs_hess;
          }
          if (fabs_grad < thread_min_gradient[thread_id]) {
            thread_min_gradient[thread_id] = fabs_grad;
          }
          if (fabs_hess < thread_min_hessian[thread_id]) {
            thread_min_hessian[thread_id] = fabs_hess;
          }
        }});
    max_gradient = thread_max_gradient[0];
    max_hessian = thread_max_hessian[0];
    double min_gradient = thread_min_gradient[0];
    double min_hessian = thread_min_hessian[0];
    for (int thread_id = 1; thread_id < num_threads; ++thread_id) {
      if (max_gradient < thread_max_gradient[thread_id]) {
        max_gradient = thread_max_gradient[thread_id];
      }
      if (max_hessian < thread_max_hessian[thread_id]) {
        max_hessian = thread_max_hessian[thread_id];
      }
    }
    Log::Warning("max_gradient = %f, max_hessian = %f", max_gradient, max_hessian);
    Log::Warning("min_gradient = %f, min_hessian = %f", min_gradient, min_hessian);
    *grad_scale = max_gradient / static_cast<double>(kIntGradBins / 2);
    *hess_scale = max_hessian / static_cast<double>(kIntGradBins);
    Log::Warning("grad_scale = %.20f, hess_scale = %.20f", *grad_scale, *hess_scale);
    const double g_inverse_scale = 1.0f / (*grad_scale);
    const double h_inverse_scale = 1.0f / (*hess_scale);
    const double gs = *grad_scale;
    const double hs = *hess_scale;
    std::vector<std::mt19937> mt_generators;
    for (int i = 0; i < OMP_NUM_THREADS(); ++i) {
      mt_generators.emplace_back(i + iter);
    }
    std::vector<std::uniform_real_distribution<double>> dist;
    for (int i = 0; i < OMP_NUM_THREADS(); ++i) {
      dist.push_back(std::uniform_real_distribution<double>(0.0f, 1.0f));
    }
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_data_; ++i) {
      const int thread_id = omp_get_thread_num();
      const score_t gradient = gradients[i];
      const score_t hessian = hessians[i];
      const int_score_t int_grad = static_cast<int_score_t>(std::lround(gradient * g_inverse_scale));
      const int_score_t int_hess = static_cast<int_score_t>(std::lround(hessian * h_inverse_scale));
      const score_t gradient_low = int_grad * gs;
      const score_t gradient_high = gradient >= 0.0f ? (int_grad + 1) * gs : (int_grad - 1) * gs;
      const score_t hessian_low = int_hess * hs;
      const score_t hessian_high = (int_hess + 1) * hs;
      const score_t gradient_bias = (gradient - gradient_low) / (gradient_high - gradient_low);
      const score_t hessian_bias = (hessian - hessian_low) / (hessian_high - hessian_low);
      if (dist[thread_id](mt_generators[thread_id]) > gradient_bias) {
        int_gradients[i] = int_grad;
      } else {
        if (gradient < 0.0f) {
          CHECK(int_grad <= 0);
          int_gradients[i] = int_grad - 1;
        } else {
          CHECK(int_grad >= 0);
          int_gradients[i] = int_grad + 1;
        }
      }
      if (dist[thread_id](mt_generators[thread_id]) > hessian_bias) {
        int_hessians[i] = int_hess;
      } else {
        int_hessians[i] = int_hess + 1;
      }
    }
    ++iter;
  }

  void NonUniformDiscretizeGradients(score_t* gradients, score_t* hessians,
    int_score_t* int_gradients, int_score_t* int_hessians,
    double* grad_scale, double* hess_scale) const {
    double max_gradient = std::fabs(gradients[0]);
    double max_hessian = std::fabs(hessians[0]);
    int num_threads = OMP_NUM_THREADS();
    std::vector<double> thread_max_gradient(num_threads, max_gradient);
    std::vector<double> thread_max_hessian(num_threads, max_hessian);
    std::vector<double> thread_min_gradient(num_threads, max_gradient);
    std::vector<double> thread_min_hessian(num_threads, max_hessian);
    static int iter = 0;
    //DumpGradientToFile(gradients, hessians, iter, num_data_, 1, "non_uniform", 0.0f, 0.0f, false);
    Threading::For<data_size_t>(0, num_data_, 1024,
      [gradients, hessians, &thread_max_gradient, &thread_max_hessian,
        &thread_min_gradient, &thread_min_hessian]
      (int, data_size_t start, data_size_t end) {
        int thread_id = omp_get_thread_num();
        for (data_size_t i = start; i < end; ++i) {
          double fabs_grad = std::fabs(gradients[i]);
          double fabs_hess = std::fabs(hessians[i]);
          if (fabs_grad > thread_max_gradient[thread_id]) {
            thread_max_gradient[thread_id] = fabs_grad;
          }
          if (fabs_hess > thread_max_hessian[thread_id]) {
            thread_max_hessian[thread_id] = fabs_hess;
          }
          if (fabs_grad < thread_min_gradient[thread_id]) {
            thread_min_gradient[thread_id] = fabs_grad;
          }
          if (fabs_hess < thread_min_hessian[thread_id]) {
            thread_min_hessian[thread_id] = fabs_hess;
          }
        }});
    max_gradient = thread_max_gradient[0];
    max_hessian = thread_max_hessian[0];
    double min_gradient = thread_min_gradient[0];
    double min_hessian = thread_min_hessian[0];
    for (int thread_id = 1; thread_id < num_threads; ++thread_id) {
      if (max_gradient < thread_max_gradient[thread_id]) {
        max_gradient = thread_max_gradient[thread_id];
      }
      if (max_hessian < thread_max_hessian[thread_id]) {
        max_hessian = thread_max_hessian[thread_id];
      }
      if (min_gradient > thread_min_gradient[thread_id]) {
        min_gradient = thread_min_gradient[thread_id];
      }
      if (min_hessian > thread_min_hessian[thread_id]) {
        min_hessian = thread_min_hessian[thread_id];
      }
    }
    Log::Warning("max_gradient = %f, max_hessian = %f", max_gradient, max_hessian);
    Log::Warning("min_gradient = %f, min_hessian = %f", min_gradient, min_hessian);
    *grad_scale = max_gradient / static_cast<double>(kIntGradBins / 2 / 2 * 3);
    *hess_scale = max_hessian / static_cast<double>(kIntGradBins);
    Log::Warning("grad_scale = %.20f, hess_scale = %.20f", *grad_scale, *hess_scale);
    const double g_inverse_scale = 1.0f / (*grad_scale);
    const double h_inverse_scale = 1.0f / (*hess_scale);
    const double gs = *grad_scale;
    const double hs = *hess_scale;
    std::vector<std::mt19937> mt_generators;
    for (int i = 0; i < OMP_NUM_THREADS(); ++i) {
      mt_generators.emplace_back(i + iter);
    }
    std::vector<std::uniform_real_distribution<double>> dist;
    for (int i = 0; i < OMP_NUM_THREADS(); ++i) {
      dist.push_back(std::uniform_real_distribution<double>(0.0f, 1.0f));
    }
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_data_; ++i) {
      const int thread_id = omp_get_thread_num();
      const score_t gradient = gradients[i];
      const score_t hessian = hessians[i];
      const int_score_t int_grad = static_cast<int_score_t>(std::lround(gradient * g_inverse_scale));
      const int_score_t int_hess = static_cast<int_score_t>(std::lround(hessian * h_inverse_scale));
      const score_t gradient_low = std::abs(int_grad) >= 2 ? int_grad * gs : 0.0f;
      const score_t gradient_high = std::abs(int_grad) >= 2 ?
        (gradient >= 0.0f ? (int_grad + 1) * gs : (int_grad - 1) * gs) :
        (gradient >= 0.0f ? 2 * gs : -2 * gs);
      const score_t hessian_low = int_hess * hs;
      const score_t hessian_high = (int_hess + 1) * hs;
      const score_t gradient_bias = (gradient - gradient_low) / (gradient_high - gradient_low);
      const score_t hessian_bias = (hessian - hessian_low) / (hessian_high - hessian_low);
      if (dist[thread_id](mt_generators[thread_id]) > gradient_bias) {
        int_gradients[i] = std::abs(int_grad) >= 2 ? int_grad : 0;
      } else {
        if (gradient < 0.0f) {
          CHECK(int_grad <= 0);
          int_gradients[i] = int_grad <= -2 ? int_grad - 1 : -2;
        } else {
          CHECK(int_grad >= 0);
          int_gradients[i] = int_grad >= 2 ? int_grad + 1 : 2;
        }
      }
      if (dist[thread_id](mt_generators[thread_id]) > hessian_bias) {
        int_hessians[i] = int_hess;
      } else {
        int_hessians[i] = int_hess + 1;
      }
    }
    //DumpGradientToFile(int_gradients, int_hessians, iter, num_data_, 1, "non_uniform_discretized", *grad_scale, *hess_scale, true);
    ++iter;
  }

  virtual void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                       const label_t* label,
                                       const double* score, score_t* lambdas,
                                       score_t* hessians) const = 0;

  const char* GetName() const override = 0;

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  bool NeedAccuratePrediction() const override { return false; }

 protected:
  int seed_;
  data_size_t num_queries_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weights */
  const label_t* weights_;
  /*! \brief Query boundries */
  const data_size_t* query_boundaries_;
};

/*!
 * \brief Objective function for Lambdrank with NDCG
 */
class LambdarankNDCG : public RankingObjective {
 public:
  explicit LambdarankNDCG(const Config& config)
      : RankingObjective(config),
        sigmoid_(config.sigmoid),
        norm_(config.lambdarank_norm),
        truncation_level_(config.lambdarank_truncation_level) {
    label_gain_ = config.label_gain;
    // initialize DCG calculator
    DCGCalculator::DefaultLabelGain(&label_gain_);
    DCGCalculator::Init(label_gain_);
    sigmoid_table_.clear();
    inverse_max_dcgs_.clear();
    if (sigmoid_ <= 0.0) {
      Log::Fatal("Sigmoid param %f should be greater than zero", sigmoid_);
    }
  }

  explicit LambdarankNDCG(const std::vector<std::string>& strs)
      : RankingObjective(strs) {}

  ~LambdarankNDCG() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankingObjective::Init(metadata, num_data);
    DCGCalculator::CheckLabel(label_, num_data_);
    inverse_max_dcgs_.resize(num_queries_);
#pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(
          truncation_level_, label_ + query_boundaries_[i],
          query_boundaries_[i + 1] - query_boundaries_[i]);

      if (inverse_max_dcgs_[i] > 0.0) {
        inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
      }
    }
    // construct sigmoid table to speed up sigmoid transform
    ConstructSigmoidTable();
    const int num_threads = OMP_NUM_THREADS();
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
      rand_generators_.emplace_back(thread_id);
    }
    uniform_dists_.resize(num_threads);
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                      const label_t* label, const double* score,
                                      score_t* lambdas,
                                      score_t* hessians) const override {
    // get max DCG on current query
    const double inverse_max_dcg = inverse_max_dcgs_[query_id];
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
        sorted_idx.begin(), sorted_idx.end(),
        [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;
    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        } else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_label_gain = label_gain_[high_label];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;

        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }
        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;
      }
    }
    if (norm_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }
  }

  inline double GetSigmoid(double score) const {
    if (score <= min_sigmoid_input_) {
      // too small, use lower bound
      return sigmoid_table_[0];
    } else if (score >= max_sigmoid_input_) {
      // too large, use upper bound
      return sigmoid_table_[_sigmoid_bins - 1];
    } else {
      return sigmoid_table_[static_cast<size_t>((score - min_sigmoid_input_) *
                                                sigmoid_table_idx_factor_)];
    }
  }

  void ConstructSigmoidTable() {
    // get boundary
    min_sigmoid_input_ = min_sigmoid_input_ / sigmoid_ / 2;
    max_sigmoid_input_ = -min_sigmoid_input_;
    sigmoid_table_.resize(_sigmoid_bins);
    // get score to bin factor
    sigmoid_table_idx_factor_ =
        _sigmoid_bins / (max_sigmoid_input_ - min_sigmoid_input_);
    // cache
    for (size_t i = 0; i < _sigmoid_bins; ++i) {
      const double score = i / sigmoid_table_idx_factor_ + min_sigmoid_input_;
      sigmoid_table_[i] = 1.0f / (1.0f + std::exp(score * sigmoid_));
    }
  }

  const char* GetName() const override { return "lambdarank"; }

 private:
  /*! \brief Simgoid param */
  double sigmoid_;
  /*! \brief Normalize the lambdas or not */
  bool norm_;
  /*! \brief Truncation position for max DCG */
  int truncation_level_;
  /*! \brief Cache inverse max DCG, speed up calculation */
  std::vector<double> inverse_max_dcgs_;
  /*! \brief Cache result for sigmoid transform to speed up */
  std::vector<double> sigmoid_table_;
  /*! \brief Gains for labels */
  std::vector<double> label_gain_;
  /*! \brief Number of bins in simoid table */
  size_t _sigmoid_bins = 1024 * 1024;
  /*! \brief Minimal input of sigmoid table */
  double min_sigmoid_input_ = -50;
  /*! \brief Maximal input of sigmoid table */
  double max_sigmoid_input_ = 50;
  /*! \brief Factor that covert score to bin in sigmoid table */
  double sigmoid_table_idx_factor_;
};

/*!
 * \brief Implementation of the learning-to-rank objective function, XE_NDCG
 * [arxiv.org/abs/1911.09798].
 */
class RankXENDCG : public RankingObjective {
 public:
  explicit RankXENDCG(const Config& config) : RankingObjective(config) {}

  explicit RankXENDCG(const std::vector<std::string>& strs)
      : RankingObjective(strs) {}

  ~RankXENDCG() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankingObjective::Init(metadata, num_data);
    for (data_size_t i = 0; i < num_queries_; ++i) {
      rands_.emplace_back(seed_ + i);
    }
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                      const label_t* label, const double* score,
                                      score_t* lambdas,
                                      score_t* hessians) const override {
    // Skip groups with too few items.
    if (cnt <= 1) {
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = 0.0f;
        hessians[i] = 0.0f;
      }
      return;
    }

    // Turn scores into a probability distribution using Softmax.
    std::vector<double> rho(cnt, 0.0);
    Common::Softmax(score, rho.data(), cnt);

    // An auxiliary buffer of parameters used to form the ground-truth
    // distribution and compute the loss.
    std::vector<double> params(cnt);

    double inv_denominator = 0;
    for (data_size_t i = 0; i < cnt; ++i) {
      params[i] = Phi(label[i], rands_[query_id].NextFloat());
      inv_denominator += params[i];
    }
    // sum_labels will always be positive number
    inv_denominator = 1. / std::max<double>(kEpsilon, inv_denominator);

    // Approximate gradients and inverse Hessian.
    // First order terms.
    double sum_l1 = 0.0;
    for (data_size_t i = 0; i < cnt; ++i) {
      double term = -params[i] * inv_denominator + rho[i];
      lambdas[i] = static_cast<score_t>(term);
      // Params will now store terms needed to compute second-order terms.
      params[i] = term / (1. - rho[i]);
      sum_l1 += params[i];
    }
    // Second order terms.
    double sum_l2 = 0.0;
    for (data_size_t i = 0; i < cnt; ++i) {
      double term = rho[i] * (sum_l1 - params[i]);
      lambdas[i] += static_cast<score_t>(term);
      // Params will now store terms needed to compute third-order terms.
      params[i] = term / (1. - rho[i]);
      sum_l2 += params[i];
    }
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] += static_cast<score_t>(rho[i] * (sum_l2 - params[i]));
      hessians[i] = static_cast<score_t>(rho[i] * (1.0 - rho[i]));
    }
  }

  double Phi(const label_t l, double g) const {
    return Common::Pow(2, static_cast<int>(l)) - g;
  }

  const char* GetName() const override { return "rank_xendcg"; }

 private:
  mutable std::vector<Random> rands_;
};

}  // namespace LightGBM
#endif  // LightGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
