/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_

#include <LightGBM/meta.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/utils/array_args.h>

#include <string>
#include <algorithm>
#include <vector>

namespace LightGBM {

#define PercentileFun(T, data_reader, cnt_data, alpha)                    \
  {                                                                       \
    if (cnt_data <= 1) {                                                  \
      return data_reader(0);                                              \
    }                                                                     \
    std::vector<T> ref_data(cnt_data);                                    \
    for (data_size_t i = 0; i < cnt_data; ++i) {                          \
      ref_data[i] = data_reader(i);                                       \
    }                                                                     \
    const double float_pos = (1.0f - alpha) * cnt_data;                   \
    const data_size_t pos = static_cast<data_size_t>(float_pos);          \
    if (pos < 1) {                                                        \
      return ref_data[ArrayArgs<T>::ArgMax(ref_data)];                    \
    } else if (pos >= cnt_data) {                                         \
      return ref_data[ArrayArgs<T>::ArgMin(ref_data)];                    \
    } else {                                                              \
      const double bias = float_pos - pos;                                \
      if (pos > cnt_data / 2) {                                           \
        ArrayArgs<T>::ArgMaxAtK(&ref_data, 0, cnt_data, pos - 1);         \
        T v1 = ref_data[pos - 1];                                         \
        T v2 = ref_data[pos + ArrayArgs<T>::ArgMax(ref_data.data() + pos, \
                                                   cnt_data - pos)];      \
        return static_cast<T>(v1 - (v1 - v2) * bias);                     \
      } else {                                                            \
        ArrayArgs<T>::ArgMaxAtK(&ref_data, 0, cnt_data, pos);             \
        T v2 = ref_data[pos];                                             \
        T v1 = ref_data[ArrayArgs<T>::ArgMin(ref_data.data(), pos)];      \
        return static_cast<T>(v1 - (v1 - v2) * bias);                     \
      }                                                                   \
    }                                                                     \
  }\

#define WeightedPercentileFun(T, data_reader, weight_reader, cnt_data, alpha) \
  {                                                                           \
    if (cnt_data <= 1) {                                                      \
      return data_reader(0);                                                  \
    }                                                                         \
    std::vector<data_size_t> sorted_idx(cnt_data);                            \
    for (data_size_t i = 0; i < cnt_data; ++i) {                              \
      sorted_idx[i] = i;                                                      \
    }                                                                         \
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(),                    \
                     [&](data_size_t a, data_size_t b) {                      \
                       return data_reader(a) < data_reader(b);                \
                     });                                                      \
    std::vector<double> weighted_cdf(cnt_data);                               \
    weighted_cdf[0] = weight_reader(sorted_idx[0]);                           \
    for (data_size_t i = 1; i < cnt_data; ++i) {                              \
      weighted_cdf[i] = weighted_cdf[i - 1] + weight_reader(sorted_idx[i]);   \
    }                                                                         \
    double threshold = weighted_cdf[cnt_data - 1] * alpha;                    \
    size_t pos = std::upper_bound(weighted_cdf.begin(), weighted_cdf.end(),   \
                                  threshold) -                                \
                 weighted_cdf.begin();                                        \
    pos = std::min(pos, static_cast<size_t>(cnt_data - 1));                   \
    if (pos == 0 || pos == static_cast<size_t>(cnt_data - 1)) {               \
      return data_reader(sorted_idx[pos]);                                    \
    }                                                                         \
    CHECK_GE(threshold, weighted_cdf[pos - 1]);                               \
    CHECK_LT(threshold, weighted_cdf[pos]);                                   \
    T v1 = data_reader(sorted_idx[pos - 1]);                                  \
    T v2 = data_reader(sorted_idx[pos]);                                      \
    if (weighted_cdf[pos + 1] - weighted_cdf[pos] >= 1.0f) {                  \
      return static_cast<T>((threshold - weighted_cdf[pos]) /                 \
                                (weighted_cdf[pos + 1] - weighted_cdf[pos]) * \
                                (v2 - v1) +                                   \
                            v1);                                              \
    } else {                                                                  \
      return static_cast<T>(v2);                                              \
    }                                                                         \
  }\

/*!
* \brief Objective function for regression
*/
class RegressionL2loss: public ObjectiveFunction {
 public:
  explicit RegressionL2loss(const Config& config) {
    sqrt_ = config.reg_sqrt;
  }

  explicit RegressionL2loss(const std::vector<std::string>& strs) {
    sqrt_ = false;
    for (auto str : strs) {
      if (str == std::string("sqrt")) {
        sqrt_ = true;
      }
    }
  }

  ~RegressionL2loss() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    if (sqrt_) {
      trans_label_.resize(num_data_);
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data; ++i) {
        trans_label_[i] = Common::Sign(label_[i]) * std::sqrt(std::fabs(label_[i]));
      }
      label_ = trans_label_.data();
    }
    weights_ = metadata.weights();
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = static_cast<score_t>(score[i] - label_[i]);
        hessians[i] = 1.0f;
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = static_cast<score_t>((score[i] - label_[i]) * weights_[i]);
        hessians[i] = static_cast<score_t>(weights_[i]);
      }
    }
  }

  void GetIntGradients(const double* score,
    score_t* gradients, score_t* hessians,
    int_score_t* int_gradients, int_score_t* int_hessians,
    double* grad_scale, double* hess_scale) const override {
    GetGradients(score, gradients, hessians);
    NonUniformDiscretizeGradients(gradients, hessians, int_gradients, int_hessians,
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
    int num_threads = OMP_NUM_THREADS();
    std::vector<double> thread_max_gradient(num_threads, max_gradient);
    std::vector<double> thread_min_gradient(num_threads, max_gradient);
    static int iter = 0;
    Threading::For<data_size_t>(0, num_data_, 1024,
      [gradients, hessians, &thread_max_gradient, &thread_min_gradient]
      (int, data_size_t start, data_size_t end) {
        int thread_id = omp_get_thread_num();
        for (data_size_t i = start; i < end; ++i) {
          double fabs_grad = std::fabs(gradients[i]);
          if (fabs_grad > thread_max_gradient[thread_id]) {
            thread_max_gradient[thread_id] = fabs_grad;
          }
          if (fabs_grad < thread_min_gradient[thread_id]) {
            thread_min_gradient[thread_id] = fabs_grad;
          }
        }});
    max_gradient = thread_max_gradient[0];
    double min_gradient = thread_min_gradient[0];
    for (int thread_id = 1; thread_id < num_threads; ++thread_id) {
      if (max_gradient < thread_max_gradient[thread_id]) {
        max_gradient = thread_max_gradient[thread_id];
      }
    }
    Log::Warning("max_gradient = %f", max_gradient);
    Log::Warning("min_gradient = %f", min_gradient);
    *grad_scale = max_gradient / static_cast<double>(kIntGradBins / 2);
    *hess_scale = hessians[0];
    Log::Warning("grad_scale = %.20f, hess_scale = %.20f", *grad_scale, *hess_scale);
    const double g_inverse_scale = 1.0f / (*grad_scale);
    const double gs = *grad_scale;
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
      const int_score_t int_grad = static_cast<int_score_t>(std::lround(gradient * g_inverse_scale));
      const int_score_t int_hess = 1;
      const score_t gradient_low = int_grad * gs;
      const score_t gradient_high = gradient >= 0.0f ? (int_grad + 1) * gs : (int_grad - 1) * gs;
      const score_t gradient_bias = (gradient - gradient_low) / (gradient_high - gradient_low);
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
      int_hessians[i] = int_hess;
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

  const char* GetName() const override {
    return "regression";
  }

  void ConvertOutput(const double* input, double* output) const override {
    if (sqrt_) {
      output[0] = Common::Sign(input[0]) * input[0] * input[0];
    } else {
      output[0] = input[0];
    }
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    if (sqrt_) {
      str_buf << " sqrt";
    }
    return str_buf.str();
  }

  bool IsConstantHessian() const override {
    if (weights_ == nullptr) {
      return true;
    } else {
      return false;
    }
  }

  double BoostFromScore(int) const override {
    double suml = 0.0f;
    double sumw = 0.0f;
    if (weights_ != nullptr) {
      #pragma omp parallel for schedule(static) reduction(+:suml, sumw)
      for (data_size_t i = 0; i < num_data_; ++i) {
        suml += label_[i] * weights_[i];
        sumw += weights_[i];
      }
    } else {
      sumw = static_cast<double>(num_data_);
      #pragma omp parallel for schedule(static) reduction(+:suml)
      for (data_size_t i = 0; i < num_data_; ++i) {
        suml += label_[i];
      }
    }
    return suml / sumw;
  }

 protected:
  bool sqrt_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weights */
  const label_t* weights_;
  std::vector<label_t> trans_label_;
};

/*!
* \brief L1 regression loss
*/
class RegressionL1loss: public RegressionL2loss {
 public:
  explicit RegressionL1loss(const Config& config): RegressionL2loss(config) {
  }

  explicit RegressionL1loss(const std::vector<std::string>& strs): RegressionL2loss(strs) {
  }

  ~RegressionL1loss() {}

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double diff = score[i] - label_[i];
        gradients[i] = static_cast<score_t>(Common::Sign(diff));
        hessians[i] = 1.0f;
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double diff = score[i] - label_[i];
        gradients[i] = static_cast<score_t>(Common::Sign(diff) * weights_[i]);
        hessians[i] = weights_[i];
      }
    }
  }

  double BoostFromScore(int) const override {
    const double alpha = 0.5;
    if (weights_ != nullptr) {
      #define data_reader(i) (label_[i])
      #define weight_reader(i) (weights_[i])
      WeightedPercentileFun(label_t, data_reader, weight_reader, num_data_, alpha);
      #undef data_reader
      #undef weight_reader
    } else {
      #define data_reader(i) (label_[i])
      PercentileFun(label_t, data_reader, num_data_, alpha);
      #undef data_reader
    }
  }

  bool IsRenewTreeOutput() const override { return true; }

  double RenewTreeOutput(double, std::function<double(const label_t*, int)> residual_getter,
                         const data_size_t* index_mapper,
                         const data_size_t* bagging_mapper,
                         data_size_t num_data_in_leaf) const override {
    const double alpha = 0.5;
    if (weights_ == nullptr) {
      if (bagging_mapper == nullptr) {
        #define data_reader(i) (residual_getter(label_, index_mapper[i]))
        PercentileFun(double, data_reader, num_data_in_leaf, alpha);
        #undef data_reader
      } else {
        #define data_reader(i) (residual_getter(label_, bagging_mapper[index_mapper[i]]))
        PercentileFun(double, data_reader, num_data_in_leaf, alpha);
        #undef data_reader
      }
    } else {
      if (bagging_mapper == nullptr) {
        #define data_reader(i) (residual_getter(label_, index_mapper[i]))
        #define weight_reader(i) (weights_[index_mapper[i]])
        WeightedPercentileFun(double, data_reader, weight_reader, num_data_in_leaf, alpha);
        #undef data_reader
        #undef weight_reader
      } else {
        #define data_reader(i) (residual_getter(label_, bagging_mapper[index_mapper[i]]))
        #define weight_reader(i) (weights_[bagging_mapper[index_mapper[i]]])
        WeightedPercentileFun(double, data_reader, weight_reader, num_data_in_leaf, alpha);
        #undef data_reader
        #undef weight_reader
      }
    }
  }

  const char* GetName() const override {
    return "regression_l1";
  }
};

/*!
* \brief Huber regression loss
*/
class RegressionHuberLoss: public RegressionL2loss {
 public:
  explicit RegressionHuberLoss(const Config& config): RegressionL2loss(config) {
    alpha_ = static_cast<double>(config.alpha);
    if (sqrt_) {
      Log::Warning("Cannot use sqrt transform in %s Regression, will auto disable it", GetName());
      sqrt_ = false;
    }
  }

  explicit RegressionHuberLoss(const std::vector<std::string>& strs): RegressionL2loss(strs) {
    if (sqrt_) {
      Log::Warning("Cannot use sqrt transform in %s Regression, will auto disable it", GetName());
      sqrt_ = false;
    }
  }

  ~RegressionHuberLoss() {
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double diff = score[i] - label_[i];
        if (std::abs(diff) <= alpha_) {
          gradients[i] = static_cast<score_t>(diff);
        } else {
          gradients[i] = static_cast<score_t>(Common::Sign(diff) * alpha_);
        }
        hessians[i] = 1.0f;
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double diff = score[i] - label_[i];
        if (std::abs(diff) <= alpha_) {
          gradients[i] = static_cast<score_t>(diff * weights_[i]);
        } else {
          gradients[i] = static_cast<score_t>(Common::Sign(diff) * weights_[i] * alpha_);
        }
        hessians[i] = static_cast<score_t>(weights_[i]);
      }
    }
  }

  const char* GetName() const override {
    return "huber";
  }

  bool IsConstantHessian() const override {
    return false;
  }

 private:
  /*! \brief delta for Huber loss */
  double alpha_;
};


// http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html
class RegressionFairLoss: public RegressionL2loss {
 public:
  explicit RegressionFairLoss(const Config& config): RegressionL2loss(config) {
    c_ = static_cast<double>(config.fair_c);
  }

  explicit RegressionFairLoss(const std::vector<std::string>& strs): RegressionL2loss(strs) {
  }

  ~RegressionFairLoss() {}

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double x = score[i] - label_[i];
        gradients[i] = static_cast<score_t>(c_ * x / (std::fabs(x) + c_));
        hessians[i] = static_cast<score_t>(c_ * c_ / ((std::fabs(x) + c_) * (std::fabs(x) + c_)));
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double x = score[i] - label_[i];
        gradients[i] = static_cast<score_t>(c_ * x / (std::fabs(x) + c_) * weights_[i]);
        hessians[i] = static_cast<score_t>(c_ * c_ / ((std::fabs(x) + c_) * (std::fabs(x) + c_)) * weights_[i]);
      }
    }
  }

  const char* GetName() const override {
    return "fair";
  }

  bool IsConstantHessian() const override {
    return false;
  }

 private:
  /*! \brief c for Fair loss */
  double c_;
};


/*!
* \brief Objective function for Poisson regression
*/
class RegressionPoissonLoss: public RegressionL2loss {
 public:
  explicit RegressionPoissonLoss(const Config& config): RegressionL2loss(config) {
    max_delta_step_ = static_cast<double>(config.poisson_max_delta_step);
    if (sqrt_) {
      Log::Warning("Cannot use sqrt transform in %s Regression, will auto disable it", GetName());
      sqrt_ = false;
    }
  }

  explicit RegressionPoissonLoss(const std::vector<std::string>& strs): RegressionL2loss(strs) {
  }

  ~RegressionPoissonLoss() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    if (sqrt_) {
      Log::Warning("Cannot use sqrt transform in %s Regression, will auto disable it", GetName());
      sqrt_ = false;
    }
    RegressionL2loss::Init(metadata, num_data);
    // Safety check of labels
    label_t miny;
    double sumy;
    Common::ObtainMinMaxSum(label_, num_data_, &miny, static_cast<label_t*>(nullptr), &sumy);
    if (miny < 0.0f) {
      Log::Fatal("[%s]: at least one target label is negative", GetName());
    }
    if (sumy == 0.0f) {
      Log::Fatal("[%s]: sum of labels is zero", GetName());
    }
  }

  /* Parametrize with unbounded internal score "f"; then
   *  loss = exp(f) - label * f
   *  grad = exp(f) - label
   *  hess = exp(f)
   *
   * And the output is exp(f); so the associated metric get s=exp(f)
   * so that its loss = s - label * log(s); a little awkward maybe.
   *
   */
  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = static_cast<score_t>(std::exp(score[i]) - label_[i]);
        hessians[i] = static_cast<score_t>(std::exp(score[i] + max_delta_step_));
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = static_cast<score_t>((std::exp(score[i]) - label_[i]) * weights_[i]);
        hessians[i] = static_cast<score_t>(std::exp(score[i] + max_delta_step_) * weights_[i]);
      }
    }
  }

  void ConvertOutput(const double* input, double* output) const override {
    output[0] = std::exp(input[0]);
  }

  const char* GetName() const override {
    return "poisson";
  }

  double BoostFromScore(int) const override {
    return Common::SafeLog(RegressionL2loss::BoostFromScore(0));
  }

  bool IsConstantHessian() const override {
    return false;
  }

 private:
  /*! \brief used to safeguard optimization */
  double max_delta_step_;
};

class RegressionQuantileloss : public RegressionL2loss {
 public:
  explicit RegressionQuantileloss(const Config& config): RegressionL2loss(config) {
    alpha_ = static_cast<score_t>(config.alpha);
    CHECK(alpha_ > 0 && alpha_ < 1);
  }

  explicit RegressionQuantileloss(const std::vector<std::string>& strs): RegressionL2loss(strs) {
  }

  ~RegressionQuantileloss() {}

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        score_t delta = static_cast<score_t>(score[i] - label_[i]);
        if (delta >= 0) {
          gradients[i] = (1.0f - alpha_);
        } else {
          gradients[i] = -alpha_;
        }
        hessians[i] = 1.0f;
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        score_t delta = static_cast<score_t>(score[i] - label_[i]);
        if (delta >= 0) {
          gradients[i] = static_cast<score_t>((1.0f - alpha_) * weights_[i]);
        } else {
          gradients[i] = static_cast<score_t>(-alpha_ * weights_[i]);
        }
        hessians[i] = static_cast<score_t>(weights_[i]);
      }
    }
  }

  const char* GetName() const override {
    return "quantile";
  }

  double BoostFromScore(int) const override {
    if (weights_ != nullptr) {
      #define data_reader(i) (label_[i])
      #define weight_reader(i) (weights_[i])
      WeightedPercentileFun(label_t, data_reader, weight_reader, num_data_, alpha_);
      #undef data_reader
      #undef weight_reader
    } else {
      #define data_reader(i) (label_[i])
      PercentileFun(label_t, data_reader, num_data_, alpha_);
      #undef data_reader
    }
  }

  bool IsRenewTreeOutput() const override { return true; }

  double RenewTreeOutput(double, std::function<double(const label_t*, int)> residual_getter,
                         const data_size_t* index_mapper,
                         const data_size_t* bagging_mapper,
                         data_size_t num_data_in_leaf) const override {
    if (weights_ == nullptr) {
      if (bagging_mapper == nullptr) {
        #define data_reader(i) (residual_getter(label_, index_mapper[i]))
        PercentileFun(double, data_reader, num_data_in_leaf, alpha_);
        #undef data_reader
      } else {
        #define data_reader(i) (residual_getter(label_, bagging_mapper[index_mapper[i]]))
        PercentileFun(double, data_reader, num_data_in_leaf, alpha_);
        #undef data_reader
      }
    } else {
      if (bagging_mapper == nullptr) {
        #define data_reader(i) (residual_getter(label_, index_mapper[i]))
        #define weight_reader(i) (weights_[index_mapper[i]])
        WeightedPercentileFun(double, data_reader, weight_reader, num_data_in_leaf, alpha_);
        #undef data_reader
        #undef weight_reader
      } else {
        #define data_reader(i) (residual_getter(label_, bagging_mapper[index_mapper[i]]))
        #define weight_reader(i) (weights_[bagging_mapper[index_mapper[i]]])
        WeightedPercentileFun(double, data_reader, weight_reader, num_data_in_leaf, alpha_);
        #undef data_reader
        #undef weight_reader
      }
    }
  }

 private:
  score_t alpha_;
};


/*!
* \brief Mape Regression Loss
*/
class RegressionMAPELOSS : public RegressionL1loss {
 public:
  explicit RegressionMAPELOSS(const Config& config) : RegressionL1loss(config) {
  }

  explicit RegressionMAPELOSS(const std::vector<std::string>& strs) : RegressionL1loss(strs) {
  }

  ~RegressionMAPELOSS() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RegressionL2loss::Init(metadata, num_data);
    for (data_size_t i = 0; i < num_data_; ++i) {
      if (std::fabs(label_[i]) < 1) {
        Log::Warning("Met 'abs(label) < 1', will convert them to '1' in MAPE objective and metric");
        break;
      }
    }
    label_weight_.resize(num_data);
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        label_weight_[i] = 1.0f / std::max(1.0f, std::fabs(label_[i]));
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        label_weight_[i] = 1.0f / std::max(1.0f, std::fabs(label_[i])) * weights_[i];
      }
    }
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double diff = score[i] - label_[i];
        gradients[i] = static_cast<score_t>(Common::Sign(diff) * label_weight_[i]);
        hessians[i] = 1.0f;
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        const double diff = score[i] - label_[i];
        gradients[i] = static_cast<score_t>(Common::Sign(diff) * label_weight_[i]);
        hessians[i] = weights_[i];
      }
    }
  }

  double BoostFromScore(int) const override {
    const double alpha = 0.5;
    #define data_reader(i) (label_[i])
    #define weight_reader(i) (label_weight_[i])
    WeightedPercentileFun(label_t, data_reader, weight_reader, num_data_, alpha);
    #undef data_reader
    #undef weight_reader
  }

  bool IsRenewTreeOutput() const override { return true; }

  double RenewTreeOutput(double, std::function<double(const label_t*, int)> residual_getter,
                         const data_size_t* index_mapper,
                         const data_size_t* bagging_mapper,
                         data_size_t num_data_in_leaf) const override {
    const double alpha = 0.5;
    if (bagging_mapper == nullptr) {
      #define data_reader(i) (residual_getter(label_, index_mapper[i]))
      #define weight_reader(i) (label_weight_[index_mapper[i]])
      WeightedPercentileFun(double, data_reader, weight_reader, num_data_in_leaf, alpha);
      #undef data_reader
      #undef weight_reader
    } else {
      #define data_reader(i) (residual_getter(label_, bagging_mapper[index_mapper[i]]))
      #define weight_reader(i) (label_weight_[bagging_mapper[index_mapper[i]]])
      WeightedPercentileFun(double, data_reader, weight_reader, num_data_in_leaf, alpha);
      #undef data_reader
      #undef weight_reader
    }
  }

  const char* GetName() const override {
    return "mape";
  }

  bool IsConstantHessian() const override {
    return true;
  }

 private:
  std::vector<label_t> label_weight_;
};



/*!
* \brief Objective function for Gamma regression
*/
class RegressionGammaLoss : public RegressionPoissonLoss {
 public:
  explicit RegressionGammaLoss(const Config& config) : RegressionPoissonLoss(config) {
  }

  explicit RegressionGammaLoss(const std::vector<std::string>& strs) : RegressionPoissonLoss(strs) {
  }

  ~RegressionGammaLoss() {}

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = static_cast<score_t>(1.0 - label_[i] / std::exp(score[i]));
        hessians[i] = static_cast<score_t>(label_[i] / std::exp(score[i]));
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = static_cast<score_t>(1.0 - label_[i] / std::exp(score[i]) * weights_[i]);
        hessians[i] = static_cast<score_t>(label_[i] / std::exp(score[i]) * weights_[i]);
      }
    }
  }

  const char* GetName() const override {
    return "gamma";
  }
};

/*!
* \brief Objective function for Tweedie regression
*/
class RegressionTweedieLoss: public RegressionPoissonLoss {
 public:
  explicit RegressionTweedieLoss(const Config& config) : RegressionPoissonLoss(config) {
    rho_ = config.tweedie_variance_power;
  }

  explicit RegressionTweedieLoss(const std::vector<std::string>& strs) : RegressionPoissonLoss(strs) {
  }

  ~RegressionTweedieLoss() {}

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = static_cast<score_t>(-label_[i] * std::exp((1 - rho_) * score[i]) + std::exp((2 - rho_) * score[i]));
        hessians[i] = static_cast<score_t>(-label_[i] * (1 - rho_) * std::exp((1 - rho_) * score[i]) +
          (2 - rho_) * std::exp((2 - rho_) * score[i]));
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = static_cast<score_t>((-label_[i] * std::exp((1 - rho_) * score[i]) + std::exp((2 - rho_) * score[i])) * weights_[i]);
        hessians[i] = static_cast<score_t>((-label_[i] * (1 - rho_) * std::exp((1 - rho_) * score[i]) +
          (2 - rho_) * std::exp((2 - rho_) * score[i])) * weights_[i]);
      }
    }
  }

  const char* GetName() const override {
    return "tweedie";
  }

 private:
  double rho_;
};

#undef PercentileFun
#undef WeightedPercentileFun

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_
