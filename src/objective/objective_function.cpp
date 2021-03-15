/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/objective_function.h>

#include "binary_objective.hpp"
#include "multiclass_objective.hpp"
#include "rank_objective.hpp"
#include "regression_objective.hpp"
#include "xentropy_objective.hpp"

namespace LightGBM {

ObjectiveFunction* ObjectiveFunction::CreateObjectiveFunction(const std::string& type, const Config& config) {
  if (type == std::string("regression")) {
    return new RegressionL2loss(config);
  } else if (type == std::string("regression_l1")) {
    return new RegressionL1loss(config);
  } else if (type == std::string("quantile")) {
    return new RegressionQuantileloss(config);
  } else if (type == std::string("huber")) {
    return new RegressionHuberLoss(config);
  } else if (type == std::string("fair")) {
    return new RegressionFairLoss(config);
  } else if (type == std::string("poisson")) {
    return new RegressionPoissonLoss(config);
  } else if (type == std::string("binary")) {
    return new BinaryLogloss(config);
  } else if (type == std::string("lambdarank")) {
    return new LambdarankNDCG(config);
  } else if (type == std::string("rank_xendcg")) {
    return new RankXENDCG(config);
  } else if (type == std::string("multiclass")) {
    return new MulticlassSoftmax(config);
  } else if (type == std::string("multiclassova")) {
    return new MulticlassOVA(config);
  } else if (type == std::string("cross_entropy")) {
    return new CrossEntropy(config);
  } else if (type == std::string("cross_entropy_lambda")) {
    return new CrossEntropyLambda(config);
  } else if (type == std::string("mape")) {
    return new RegressionMAPELOSS(config);
  } else if (type == std::string("gamma")) {
    return new RegressionGammaLoss(config);
  } else if (type == std::string("tweedie")) {
    return new RegressionTweedieLoss(config);
  } else if (type == std::string("custom")) {
    return nullptr;
  }
  Log::Fatal("Unknown objective type name: %s", type.c_str());
  return nullptr;
}

ObjectiveFunction* ObjectiveFunction::CreateObjectiveFunction(const std::string& str) {
  auto strs = Common::Split(str.c_str(), ' ');
  auto type = strs[0];
  if (type == std::string("regression")) {
    return new RegressionL2loss(strs);
  } else if (type == std::string("regression_l1")) {
    return new RegressionL1loss(strs);
  } else if (type == std::string("quantile")) {
    return new RegressionQuantileloss(strs);
  } else if (type == std::string("huber")) {
    return new RegressionHuberLoss(strs);
  } else if (type == std::string("fair")) {
    return new RegressionFairLoss(strs);
  } else if (type == std::string("poisson")) {
    return new RegressionPoissonLoss(strs);
  } else if (type == std::string("binary")) {
    return new BinaryLogloss(strs);
  } else if (type == std::string("lambdarank")) {
    return new LambdarankNDCG(strs);
  } else if (type == std::string("rank_xendcg")) {
    return new RankXENDCG(strs);
  } else if (type == std::string("multiclass")) {
    return new MulticlassSoftmax(strs);
  } else if (type == std::string("multiclassova")) {
    return new MulticlassOVA(strs);
  } else if (type == std::string("cross_entropy")) {
    return new CrossEntropy(strs);
  } else if (type == std::string("cross_entropy_lambda")) {
    return new CrossEntropyLambda(strs);
  } else if (type == std::string("mape")) {
    return new RegressionMAPELOSS(strs);
  } else if (type == std::string("gamma")) {
    return new RegressionGammaLoss(strs);
  } else if (type == std::string("tweedie")) {
    return new RegressionTweedieLoss(strs);
  } else if (type == std::string("custom")) {
    return nullptr;
  }
  Log::Fatal("Unknown objective type name: %s", type.c_str());
  return nullptr;
}

bool ObjectiveFunction::GetQuantile(const score_t* gradients, const score_t* hessians,
    const int num_quantiles, const data_size_t num_data,
    score_t* max_gradient, score_t* min_gradient,
    score_t* max_hessian, score_t* min_hessian) {
  const int num_threads = OMP_NUM_THREADS();
  std::vector<score_t> thread_max_gradient(num_threads, gradients[0]), thread_min_gradient(num_threads, gradients[0]);
  std::vector<score_t> thread_max_hessian(num_threads, hessians[0]), thread_min_hessian(num_threads, hessians[0]);
  Threading::For<data_size_t>(0, num_data, 512,
    [&thread_max_gradient, &thread_min_gradient, &thread_max_hessian, &thread_min_hessian, gradients, hessians]
    (int thread_id, data_size_t start, data_size_t end) {
      for (data_size_t i = start; i < end; ++i) {
        const score_t gradient = gradients[i];
        const score_t hessian = hessians[i];
        if (gradient > thread_max_gradient[thread_id]) {
          thread_max_gradient[thread_id] = gradient;
        }
        if (gradient < thread_min_gradient[thread_id]) {
          thread_min_gradient[thread_id] = gradient;
        }
        if (hessian > thread_max_hessian[thread_id]) {
          thread_max_hessian[thread_id] = hessian;
        }
        if (hessian < thread_min_hessian[thread_id]) {
          thread_min_hessian[thread_id] = hessian;
        }
      }
    });
  *max_gradient = thread_max_gradient[0];
  *min_gradient = thread_min_gradient[0];
  *max_hessian = thread_max_hessian[0];
  *min_hessian = thread_min_hessian[0];
  for (int thread_id = 1; thread_id < num_threads; ++thread_id) {
    if (thread_max_gradient[thread_id] > *max_gradient) {
      *max_gradient = thread_max_gradient[thread_id];
    }
    if (thread_min_gradient[thread_id] < *min_gradient) {
      *min_gradient = thread_min_gradient[thread_id];
    }
    if (thread_max_hessian[thread_id] > *max_hessian) {
      *max_hessian = thread_max_hessian[thread_id];
    }
    if (thread_min_hessian[thread_id] < *min_hessian) {
      *min_hessian = thread_min_hessian[thread_id];
    }
  }
  const int num_pre_quantiles = 1000;
  const double grad_pre_scale = (*max_gradient - *min_gradient) / num_pre_quantiles;
  const double hess_pre_scale = (*max_hessian) / num_pre_quantiles;
  std::vector<std::vector<int>> thread_grad_cnt(num_threads), thread_hess_cnt(num_threads);
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    thread_grad_cnt[thread_id].resize(num_pre_quantiles + 1, 0);
    thread_hess_cnt[thread_id].resize(num_pre_quantiles + 1, 0);
  }
  int neg_grad_cnt = 0;
  #pragma omp parallel for schedule(static) num_threads(num_threads) reduction(+:neg_grad_cnt)
  for (int i = 0; i < num_data; ++i) {
    if (gradients[i] < 0.0f) {
      ++neg_grad_cnt;
    }
  }
  const int pos_grad_cnt = num_data - neg_grad_cnt;
  std::vector<int> grad_cnt(num_pre_quantiles + 1, 0), hess_cnt(num_pre_quantiles + 1, 0);
  Threading::For<data_size_t>(0, num_data, 512,
    [gradients, hessians, grad_pre_scale, hess_pre_scale, max_gradient, min_gradient, max_hessian,
      &thread_grad_cnt, &thread_hess_cnt]
    (int thread_id, data_size_t start, data_size_t end) {
      for (int i = start; i < end; ++i) {
        const score_t gradient = gradients[i];
        const score_t hessian = hessians[i];
        const int grad_int = std::lround((gradient - *min_gradient) / grad_pre_scale);
        const int hess_int = std::lround((hessian) / hess_pre_scale);
        ++thread_grad_cnt[thread_id][grad_int];
        ++thread_hess_cnt[thread_id][hess_int];
      }
    });
  #pragma omp paralell for schedule(static)
  for (int i = 0; i < num_pre_quantiles + 1; ++i) {
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
      grad_cnt[i] += thread_grad_cnt[thread_id][i];
      hess_cnt[i] += thread_hess_cnt[thread_id][i];
    }
  }
  const data_size_t num_data_per_quantile = (num_data + num_quantiles - 1) / num_quantiles;
  const int num_grad_quantile_per_side = num_quantiles / 2;
  const data_size_t num_data_neg_grad_quantile = (neg_grad_cnt + num_grad_quantile_per_side - 1) / num_grad_quantile_per_side;
  const data_size_t num_data_pos_grad_quantile = (pos_grad_cnt + num_grad_quantile_per_side - 1) / num_grad_quantile_per_side;

  int cur_data_cnt = 0;
  const int zero_grad_quantile = std::lround((-*min_gradient) / grad_pre_scale);
  grad_quantiles_.clear();
  grad_int_map_.clear();
  grad_quantiles_.emplace_back(*min_gradient);
  grad_int_map_.emplace_back(-num_quantiles / 2);
  int cur_int_grad = -num_quantiles / 2;
  for (int i = 0; i < zero_grad_quantile; ++i) {
    cur_data_cnt += grad_cnt[i];
    if (cur_data_cnt == neg_grad_cnt) {
      break;
    }
    if (cur_data_cnt >= num_data_neg_grad_quantile || i == zero_grad_quantile - 1) {
      grad_quantiles_.emplace_back(*min_gradient + i * grad_pre_scale);
      cur_data_cnt = 0;
      ++cur_int_grad;
      grad_int_map_.emplace_back(cur_int_grad);
      if (static_cast<int>(grad_quantiles_.size()) == num_grad_quantile_per_side) {
        break;
      }
    }
  }
  grad_quantiles_.emplace_back(0.0f);
  grad_int_map_.emplace_back(0);
  cur_data_cnt = 0;
  cur_int_grad = 0;
  for (int i = zero_grad_quantile; i < num_pre_quantiles + 1; ++i) {
    cur_data_cnt += grad_cnt[i];
    if (cur_data_cnt == pos_grad_cnt) { 
      break;
    }
    if (cur_data_cnt >= num_data_pos_grad_quantile) {
      grad_quantiles_.emplace_back(*min_gradient + i * grad_pre_scale);
      ++cur_int_grad;
      grad_int_map_.emplace_back(cur_int_grad);
      cur_data_cnt = 0;
      if (static_cast<int>(grad_quantiles_.size()) == num_quantiles) {
        break;
      }
    }
  }
  grad_int_map_.emplace_back(num_quantiles / 2);
  grad_quantiles_.emplace_back(*max_gradient);
  //CHECK(grad_int_map_.size() == static_cast<size_t>(num_quantiles) + 1);


  hess_quantiles_.clear();
  hess_int_map_.clear();
  hess_quantiles_.emplace_back(0.0f);
  hess_int_map_.emplace_back(0);
  cur_data_cnt = 0;
  int all_data_cnt = 0;
  int cur_int_hess = 0;
  for (int i = 0; i < num_pre_quantiles + 1; ++i) {
    cur_data_cnt += hess_cnt[i];
    if (cur_data_cnt == num_data) {
      break;
    }
    all_data_cnt += hess_cnt[i];
    if (cur_data_cnt >= num_data_per_quantile || all_data_cnt == num_data) {
      hess_quantiles_.emplace_back(i * hess_pre_scale);
      ++cur_int_hess;
      hess_int_map_.emplace_back(cur_int_hess);
      cur_data_cnt = 0;
      if (static_cast<int>(hess_quantiles_.size()) == num_quantiles) {
        break;
      }
    }
  }
  hess_quantiles_.emplace_back(*max_hessian);
  hess_int_map_.emplace_back(num_quantiles);
  //CHECK(hess_int_map_.size() == static_cast<size_t>(num_quantiles) + 1);
  double max_length = 0.0f;
  size_t max_pos = 0;
  for (int i = 0; i < static_cast<int>(grad_quantiles_.size()) - 1; ++i) {
    const double length = grad_quantiles_[i + 1] - grad_quantiles_[i];
    if (length > max_length) {
      max_pos = i;
    }
  }
  if (grad_quantiles_.size() > 3 && (max_pos == 0 ||
    static_cast<int>(max_pos) == static_cast<int>(grad_quantiles_.size()) - 1)) {
    return false;
  } else {
    const int num_quantiles = static_cast<int>(grad_quantiles_.size());
    for (int i = 0; i < num_quantiles; ++i) {
      Log::Warning("grad_quantile_[%d] = %f", i, grad_quantiles_[i]);
    }
    for (size_t i = 0; i < hess_quantiles_.size(); ++i) {
      Log::Warning("hess_quantile_[%d] = %f", i, hess_quantiles_[i]);
    }
    if (grad_quantiles_.size() > 3 &&
      std::fabs(std::fabs(grad_quantiles_[num_quantiles - 1]) - std::fabs(grad_quantiles_[1])) >= 0.1) {
      return false;
    } else {
      return true;
    }
  }
}

void InnerQuantize(const score_t value, int_score_t* out_value_int, double scale_inverse,
  const std::vector<int>& int_values, size_t i_found, const double rand_val) {
  const double scaled_value = value * scale_inverse;
  if (scaled_value >= static_cast<double>(int_values[i_found]) &&
    scaled_value <= static_cast<double>(int_values[i_found + 1])) {
    const double prob = (scaled_value - static_cast<double>(int_values[i_found])) /
      static_cast<double>(int_values[i_found + 1] - int_values[i_found]);
    if (rand_val <= prob) {
      *out_value_int = int_values[i_found + 1];
    } else {
      *out_value_int = int_values[i_found];
    }
  } else if (scaled_value > static_cast<double>(int_values[i_found + 1])) {
    size_t i_upper = i_found + 2;
    CHECK_LE(i_upper, int_values.size() - 1);
    while (scaled_value > static_cast<double>(int_values[i_upper])) {
      CHECK_LT(i_upper, int_values.size() - 1);
      ++i_upper;
    }
    //CHECK_LT(i_found, int_values.size() - 2);
    CHECK_LE(scaled_value, int_values[i_upper]);
    CHECK_GE(scaled_value, int_values[i_upper - 1]);
    const double prob = (scaled_value - static_cast<double>(int_values[i_upper - 1])) /
      static_cast<double>(int_values[i_upper] - int_values[i_upper - 1]);
    if (rand_val <= prob) {
      *out_value_int = int_values[i_upper];
    } else {
      *out_value_int = int_values[i_upper - 1];
    }
  } else {
    int i_lower = static_cast<int>(i_found) - 1;
    CHECK_GE(i_lower, 0);
    while (scaled_value < int_values[i_lower]) {
      CHECK_GT(i_lower, 0);
      --i_lower;
    }
    //CHECK_GT(i_found, 0);
    CHECK_GE(scaled_value, int_values[i_lower]);
    CHECK_LE(scaled_value, int_values[i_lower + 1]);
    const double prob = (scaled_value - static_cast<double>(int_values[i_lower])) /
      static_cast<double>(int_values[i_lower + 1] - int_values[i_lower]);
    if (rand_val <= prob) {
      *out_value_int = int_values[i_lower + 1];
    } else {
      *out_value_int = int_values[i_lower];
    }
  }
}

void ObjectiveFunction::Quantize(const score_t gradient, const score_t hessian,
  int_score_t* grad_int, int_score_t* hess_int, const int thread_id,
  double grad_scale_inverse, double hess_scale_inverse) {
  auto& engine = rand_generators_[thread_id];
  auto& dist = uniform_dists_[thread_id];
  bool grad_found = false, hess_found = false;
  const double rand_val_grad = dist(engine);
  const double rand_val_hess = dist(engine);
  for (int i = 0; i < static_cast<int>(grad_quantiles_.size()) - 1; ++i) {
    if (grad_quantiles_[i] <= gradient && grad_quantiles_[i + 1] >= gradient) {
      InnerQuantize(gradient, grad_int, grad_scale_inverse, grad_int_map_, i, rand_val_grad);
      grad_found = true;
      break;
    }
  }
  for (int i = 0; i < static_cast<int>(hess_quantiles_.size()) - 1; ++i) {
    if (hess_quantiles_[i] <= hessian && hess_quantiles_[i + 1] >= hessian) {
      InnerQuantize(hessian, hess_int, hess_scale_inverse, hess_int_map_, i, rand_val_hess);
      hess_found = true;
      break;
    }
  }
  CHECK(hess_found && grad_found);
}

void ObjectiveFunction::UniformDiscretizeGradients(score_t* gradients, score_t* hessians,
    int_score_t* int_gradients, int_score_t* int_hessians,
    data_size_t num_data, double* grad_scale, double* hess_scale) {
  score_t max_gradient = 0.0f, min_gradient = 0.0f, max_hessian = 0.0f, min_hessian = 0.0f;
  bool use_non_uniform_quantile = GetQuantile(gradients, hessians, kIntGradBins, num_data,
    &max_gradient, &min_gradient, &max_hessian, &min_hessian);
  if (use_non_uniform_quantile) {
    double grad_scale_inverse = 1.0f / (grad_quantiles_[1] - grad_quantiles_[0]);
    double hess_scale_inverse = 1.0f / (hess_quantiles_[1] - hess_quantiles_[0]);
    *grad_scale = (grad_quantiles_[1] - grad_quantiles_[0]);
    *hess_scale = (hess_quantiles_[1] - hess_quantiles_[0]);
    for (size_t i = 1; i < grad_quantiles_.size() - 1; ++i) {
      const double scale = (grad_quantiles_[i + 1] - grad_quantiles_[i]);
      if (scale > *grad_scale) {
        *grad_scale = scale;
        grad_scale_inverse = 1.0f / scale;
      }
    }
    for (size_t i = 1; i < hess_quantiles_.size() - 1; ++i) {
      const double scale = (hess_quantiles_[i + 1] - hess_quantiles_[i]);
      if (scale > *hess_scale) {
        *hess_scale = scale;
        hess_scale_inverse = 1.0f / scale;
      }
    }
    Threading::For<data_size_t>(0, num_data, 512,
      [this, gradients, hessians, int_gradients, int_hessians, grad_scale_inverse, hess_scale_inverse]
      (int thread_id, data_size_t start, data_size_t end) {
      for (data_size_t i = start; i < end; ++i) {
        Quantize(gradients[i], hessians[i], int_gradients + i, int_hessians + i, thread_id, grad_scale_inverse, hess_scale_inverse);
      }
    });

    CHECK_GT(max_hessian, 0.0f);
    const double hess_scale = max_hessian / (kIntGradBins);
    const double h_inverse_scale = 1.0f / hess_scale;
    Threading::For<data_size_t>(0, num_data, 512,
      [this, hessians, h_inverse_scale, int_hessians, hess_scale]
      (int thread_id, data_size_t start, data_size_t end) {
        for (data_size_t i = start; i < end; ++i) {
          const score_t hessian = hessians[i];
          const int low_int_hess = static_cast<int>(hessian * h_inverse_scale);
          const int high_int_hess = low_int_hess + 1;
          const score_t low_hessian = low_int_hess * hess_scale;
          const score_t high_hessian = high_int_hess * hess_scale;
          const double hess_prob = (hessian - low_hessian) / (high_hessian - low_hessian);
          const double rand_val_hess = uniform_dists_[thread_id](rand_generators_[thread_id]);
          int_hessians[i] = rand_val_hess <= hess_prob ? high_int_hess : low_int_hess;
        }
      });
  } else {
    const double max_gradient_abs = std::max(std::fabs(max_gradient), std::fabs(min_gradient));
    CHECK_GT(max_hessian, 0.0f);
    const double grad_scale = max_gradient_abs / (kIntGradBins / 2);
    const double hess_scale = max_hessian / (kIntGradBins);
    const double g_inverse_scale = 1.0f / grad_scale;
    const double h_inverse_scale = 1.0f / hess_scale;
    Threading::For<data_size_t>(0, num_data, 512,
      [this, gradients, hessians, g_inverse_scale, h_inverse_scale, int_gradients, int_hessians, grad_scale, hess_scale]
      (int thread_id, data_size_t start, data_size_t end) {
        for (data_size_t i = start; i < end; ++i) {
          const score_t gradient = gradients[i];
          const int low_int_grad = static_cast<int>(gradient * g_inverse_scale);
          const int high_int_grad = gradient >= 0.0f ? low_int_grad + 1 : low_int_grad - 1;
          const score_t low_gradient = low_int_grad * grad_scale;
          const score_t high_gradient = high_int_grad * grad_scale;
          const double grad_prob = (gradient - low_gradient) / (high_gradient - low_gradient);
          const double rand_val_grad = uniform_dists_[thread_id](rand_generators_[thread_id]);
          int_gradients[i] = rand_val_grad <= grad_prob ? high_int_grad : low_int_grad;

          const score_t hessian = hessians[i];
          const int low_int_hess = static_cast<int>(hessian * h_inverse_scale);
          const int high_int_hess = low_int_hess + 1;
          const score_t low_hessian = low_int_hess * hess_scale;
          const score_t high_hessian = high_int_hess * hess_scale;
          const double hess_prob = (hessian - low_hessian) / (high_hessian - low_hessian);
          const double rand_val_hess = uniform_dists_[thread_id](rand_generators_[thread_id]);
          int_hessians[i] = rand_val_hess <= hess_prob ? high_int_hess : low_int_hess;
        }
      });
  }
}

}  // namespace LightGBM
