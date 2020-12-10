/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_OBJECTIVE_BINARY_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_BINARY_OBJECTIVE_HPP_

#include <LightGBM/network.h>
#include <LightGBM/objective_function.h>

#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace LightGBM {
/*!
* \brief Objective function for binary classification
*/
class BinaryLogloss: public ObjectiveFunction {
 public:
  explicit BinaryLogloss(const Config& config, std::function<bool(label_t)> is_pos = nullptr) {
    sigmoid_ = static_cast<double>(config.sigmoid);
    if (sigmoid_ <= 0.0) {
      Log::Fatal("Sigmoid parameter %f should be greater than zero", sigmoid_);
    }
    is_unbalance_ = config.is_unbalance;
    scale_pos_weight_ = static_cast<double>(config.scale_pos_weight);
    if (is_unbalance_ && std::fabs(scale_pos_weight_ - 1.0f) > 1e-6) {
      Log::Fatal("Cannot set is_unbalance and scale_pos_weight at the same time");
    }
    is_pos_ = is_pos;
    if (is_pos_ == nullptr) {
      is_pos_ = [](label_t label) {return label > 0; };
    }
  }

  explicit BinaryLogloss(const std::vector<std::string>& strs) {
    sigmoid_ = -1;
    for (auto str : strs) {
      auto tokens = Common::Split(str.c_str(), ':');
      if (tokens.size() == 2) {
        if (tokens[0] == std::string("sigmoid")) {
          Common::Atof(tokens[1].c_str(), &sigmoid_);
        }
      }
    }
    if (sigmoid_ <= 0.0) {
      Log::Fatal("Sigmoid parameter %f should be greater than zero", sigmoid_);
    }
  }

  ~BinaryLogloss() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
    data_size_t cnt_positive = 0;
    data_size_t cnt_negative = 0;
    // count for positive and negative samples
    #pragma omp parallel for schedule(static) reduction(+:cnt_positive, cnt_negative)
    for (data_size_t i = 0; i < num_data_; ++i) {
      if (is_pos_(label_[i])) {
        ++cnt_positive;
      } else {
        ++cnt_negative;
      }
    }
    num_pos_data_ = cnt_positive;
    if (Network::num_machines() > 1) {
      cnt_positive = Network::GlobalSyncUpBySum(cnt_positive);
      cnt_negative = Network::GlobalSyncUpBySum(cnt_negative);
    }
    need_train_ = true;
    if (cnt_negative == 0 || cnt_positive == 0) {
      Log::Warning("Contains only one class");
      // not need to boost.
      need_train_ = false;
    }
    Log::Info("Number of positive: %d, number of negative: %d", cnt_positive, cnt_negative);
    // use -1 for negative class, and 1 for positive class
    label_val_[0] = -1;
    label_val_[1] = 1;
    // weight for label
    label_weights_[0] = 1.0f;
    label_weights_[1] = 1.0f;
    // if using unbalance, change the labels weight
    if (is_unbalance_ && cnt_positive > 0 && cnt_negative > 0) {
      if (cnt_positive > cnt_negative) {
        label_weights_[1] = 1.0f;
        label_weights_[0] = static_cast<double>(cnt_positive) / cnt_negative;
      } else {
        label_weights_[1] = static_cast<double>(cnt_negative) / cnt_positive;
        label_weights_[0] = 1.0f;
      }
    }
    label_weights_[1] *= scale_pos_weight_;
  }

  void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
    if (!need_train_) {
      return;
    }
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        // get label and label weights
        const int is_pos = is_pos_(label_[i]);
        const int label = label_val_[is_pos];
        const double label_weight = label_weights_[is_pos];
        // calculate gradients and hessians
        const double response = -label * sigmoid_ / (1.0f + std::exp(label * sigmoid_ * score[i]));
        const double abs_response = fabs(response);
        gradients[i] = static_cast<score_t>(response * label_weight);
        hessians[i] = static_cast<score_t>(abs_response * (sigmoid_ - abs_response) * label_weight);
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        // get label and label weights
        const int is_pos = is_pos_(label_[i]);
        const int label = label_val_[is_pos];
        const double label_weight = label_weights_[is_pos];
        // calculate gradients and hessians
        const double response = -label * sigmoid_ / (1.0f + std::exp(label * sigmoid_ * score[i]));
        const double abs_response = fabs(response);
        gradients[i] = static_cast<score_t>(response * label_weight  * weights_[i]);
        hessians[i] = static_cast<score_t>(abs_response * (sigmoid_ - abs_response) * label_weight * weights_[i]);
      }
    }
  }

  void GetIntGradients(const double* score,
    score_t* gradients, score_t* hessians,
    int_score_t* int_gradients, int_score_t* int_hessians,
    double* grad_scale, double* hess_scale) const override {
    if (!need_train_) {
      return;
    }
    GetGradients(score, gradients, hessians);
    DiscretizeGradients(gradients, hessians, int_gradients, int_hessians,
      grad_scale, hess_scale);
  }

  void DiscretizeGradients(score_t* gradients, score_t* hessians,
    int_score_t* int_gradients, int_score_t* /*int_hessians*/,
    double* grad_scale, double* hess_scale) const override {
    double max_gradient = std::fabs(gradients[0]);
    double max_hessian = std::fabs(hessians[0]);
    int num_threads = OMP_NUM_THREADS();
    std::vector<double> thread_max_gradient(num_threads, max_gradient);
    std::vector<double> thread_max_hessian(num_threads, max_hessian);
    std::vector<double> thread_min_gradient(num_threads, max_gradient);
    std::vector<double> thread_min_hessian(num_threads, max_hessian);
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
    *hess_scale = max_hessian / static_cast<double>(kIntGradBins / 2);
    Log::Warning("grad_scale = %.20f, hess_scale = %.20f", *grad_scale, *hess_scale);
    const double g_inverse_scale = 1.0f / (*grad_scale);
    const double h_inverse_scale = 1.0f / (*hess_scale);
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_data_; ++i) {
      int_gradients[2 * i + 1] = static_cast<int_score_t>(gradients[i] * g_inverse_scale);
      int_gradients[2 * i] = static_cast<int_score_t>(hessians[i] * h_inverse_scale);
      //int_hessians[i] = static_cast<int_score_t>(hessians[i] * h_inverse_scale);
    }
  }

  // implement custom average to boost from (if enabled among options)
  double BoostFromScore(int) const override {
    double suml = 0.0f;
    double sumw = 0.0f;
    if (weights_ != nullptr) {
      #pragma omp parallel for schedule(static) reduction(+:suml, sumw)
      for (data_size_t i = 0; i < num_data_; ++i) {
        suml += is_pos_(label_[i]) * weights_[i];
        sumw += weights_[i];
      }
    } else {
      sumw = static_cast<double>(num_data_);
      #pragma omp parallel for schedule(static) reduction(+:suml)
      for (data_size_t i = 0; i < num_data_; ++i) {
        suml += is_pos_(label_[i]);
      }
    }
    double pavg = suml / sumw;
    pavg = std::min(pavg, 1.0 - kEpsilon);
    pavg = std::max<double>(pavg, kEpsilon);
    double initscore = std::log(pavg / (1.0f - pavg)) / sigmoid_;
    Log::Info("[%s:%s]: pavg=%f -> initscore=%f",  GetName(), __func__, pavg, initscore);
    return initscore;
  }

  bool ClassNeedTrain(int /*class_id*/) const override {
    return need_train_;
  }

  const char* GetName() const override {
    return "binary";
  }

  void ConvertOutput(const double* input, double* output) const override {
    output[0] = 1.0f / (1.0f + std::exp(-sigmoid_ * input[0]));
  }

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName() << " ";
    str_buf << "sigmoid:" << sigmoid_;
    return str_buf.str();
  }

  bool SkipEmptyClass() const override { return true; }

  bool NeedAccuratePrediction() const override { return false; }

  data_size_t NumPositiveData() const override { return num_pos_data_; }

 private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Number of positive samples */
  data_size_t num_pos_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief True if using unbalance training */
  bool is_unbalance_;
  /*! \brief Sigmoid parameter */
  double sigmoid_;
  /*! \brief Values for positive and negative labels */
  int label_val_[2];
  /*! \brief Weights for positive and negative labels */
  double label_weights_[2];
  /*! \brief Weights for data */
  const label_t* weights_;
  double scale_pos_weight_;
  std::function<bool(label_t)> is_pos_;
  bool need_train_;
};

}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_BINARY_OBJECTIVE_HPP_
