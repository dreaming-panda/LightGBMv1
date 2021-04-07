/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_OBJECTIVE_FUNCTION_H_
#define LIGHTGBM_OBJECTIVE_FUNCTION_H_

#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/meta.h>

#include <string>
#include <functional>

namespace LightGBM {

class ObjectiveRandomStates {
 public:
  ObjectiveRandomStates(const data_size_t num_data, const int random_seed) {
    num_data_ = num_data;
    num_threads_ = OMP_NUM_THREADS();
    random_seed_ = random_seed;

    gradient_random_values_.resize(num_data_, 0.0f);
    hessian_random_values_.resize(num_data_, 0.0f);

    random_values_use_start_eng_ = std::mt19937(random_seed_);
    random_values_use_start_dist_ = std::uniform_int_distribution<data_size_t>(0, num_data_);
    int num_blocks = 0;
    data_size_t block_size = 0;
    Threading::BlockInfo<data_size_t>(num_data_, 512, &num_blocks, &block_size);
    #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
    for (int thread_id = 0; thread_id < num_blocks; ++thread_id) {
      const data_size_t start = thread_id * block_size;
      const data_size_t end = std::min(start + block_size, num_data_);
      std::mt19937 gradient_random_values_eng(random_seed_ + thread_id);
      std::uniform_real_distribution<double> gradient_random_values_dist(0.0f, 1.0f);
      std::mt19937 hessian_random_values_eng(random_seed_ + thread_id + num_threads_);
      std::uniform_real_distribution<double> hessian_random_values_dist(0.0f, 1.0f);
      for (data_size_t i = start; i < end; ++i) {
        gradient_random_values_[i] = gradient_random_values_dist(gradient_random_values_eng);
        hessian_random_values_[i] = hessian_random_values_dist(hessian_random_values_eng);
      }
    }

    max_gradient_abs_ = 0.0f;
    max_hessian_abs_ = 0.0f;

    gradient_scale_ = 0.0f;
    hessian_scale_ = 0.0f;
    inverse_gradient_scale_ = 0.0f;
    inverse_hessian_scale_ = 0.0f;

    boundary_locked_ = false;
  }

  const double* gradient_random_values() const { return gradient_random_values_.data(); }

  const double* hessian_random_values() const { return hessian_random_values_.data(); }

  double max_gradient_abs() const { return max_gradient_abs_; }

  double max_hessian_abs() const { return max_hessian_abs_; }

  void SetGradientInfo(const double max_gradient_abs, const double max_hessian_abs) {
    if (!boundary_locked_) {
      if (max_gradient_abs >= 0.99f && max_hessian_abs >= 0.248f) {
        boundary_locked_ = true;
        max_gradient_abs_ = 1.0f;
        max_hessian_abs_ = 0.25f;
      } else {
        max_gradient_abs_ = max_gradient_abs;
        max_hessian_abs_ = max_hessian_abs;
      }
      gradient_scale_ = max_gradient_abs_ / static_cast<double>(kIntGradBins / 2);
      hessian_scale_ = max_hessian_abs_ / static_cast<double>(kIntGradBins);
      inverse_gradient_scale_ = 1.0f / gradient_scale_;
      inverse_hessian_scale_ = 1.0f / hessian_scale_;
    }
  }

  bool boundary_locked() const { return boundary_locked_; }

  double gradient_scale() const { return gradient_scale_; }

  double hessian_scale() const { return hessian_scale_; }

  double inverse_gradient_scale() const { return inverse_gradient_scale_; }

  double inverse_hessian_scale() const { return inverse_hessian_scale_; }

  int GetNextRandomValueUseStart() { return random_values_use_start_dist_(random_values_use_start_eng_); }

  void SetGradientValueSign(const data_size_t index, const double sign) {
    gradient_random_values_[index] *= sign;
  }

 private:
  data_size_t num_data_;
  int num_threads_;

  std::vector<double> gradient_random_values_;
  std::vector<double> hessian_random_values_;
  std::mt19937 random_values_use_start_eng_;
  std::uniform_int_distribution<data_size_t> random_values_use_start_dist_;
  int random_seed_;

  double max_gradient_abs_;
  double max_hessian_abs_;

  double gradient_scale_;
  double hessian_scale_;
  double inverse_gradient_scale_;
  double inverse_hessian_scale_;

  bool boundary_locked_;
};

/*!
* \brief The interface of Objective Function.
*/
class ObjectiveFunction {
 public:
  /*! \brief virtual destructor */
  virtual ~ObjectiveFunction() {}

  /*!
  * \brief Initialize
  * \param metadata Label data
  * \param num_data Number of data
  */
  virtual void Init(const Metadata& metadata, data_size_t num_data) = 0;

  /*!
  * \brief calculating first order derivative of loss function
  * \param score prediction score in this round
  * \param gradients Output gradients
  * \param hessians Output hessians
  */
  virtual void GetGradients(const double* score,
    score_t* gradients, score_t* hessians) const = 0;

  /*!
  * \brief calculating first order derivative of loss function
  * \param score prediction score in this round
  * \param gradients Output gradients
  * \param hessians Output hessians
  * \param int_gradients Output gradients
  * \param int_hessians Output hessians
  * \param grad_scale Output scaling factor of gradient
  * \param grad_bias Output bias of gradient
  * \param hess_scale Output scaling factor of hessian
  * \param hess_bias Output bias of hessian
  */
  virtual void GetIntGradients(const double* /*score*/,
    score_t* /*gradients*/, score_t* /*hessians*/,
    int_score_t* /*int_gradients*/, int_score_t* /*int_hessians*/,
    double* /*grad_scale*/, double* /*hess_scale*/,
    ObjectiveRandomStates* /*obj_rand_state*/) const {}

  virtual void DiscretizeGradients(score_t* /*gradients*/, score_t* /*hessians*/,
    int_score_t* /*int_gradients*/, int_score_t* /*int_hessians*/,
    double* /*grad_scale*/, double* /*hess_scale*/,
    ObjectiveRandomStates* /*obj_rand_state*/) const {}

  virtual void InitObjectiveRandomStates(ObjectiveRandomStates* /*obj_rand_state*/) const {}

  virtual const char* GetName() const = 0;

  virtual bool IsConstantHessian() const { return false; }

  virtual bool IsRenewTreeOutput() const { return false; }

  virtual double RenewTreeOutput(double ori_output, std::function<double(const label_t*, int)>,
                                 const data_size_t*,
                                 const data_size_t*,
                                 data_size_t) const { return ori_output; }

  virtual double BoostFromScore(int /*class_id*/) const { return 0.0; }

  virtual bool ClassNeedTrain(int /*class_id*/) const { return true; }

  virtual bool SkipEmptyClass() const { return false; }

  virtual int NumModelPerIteration() const { return 1; }

  virtual int NumPredictOneRow() const { return 1; }

  /*! \brief The prediction should be accurate or not. True will disable early stopping for prediction. */
  virtual bool NeedAccuratePrediction() const { return true; }

  /*! \brief Return the number of positive samples. Return 0 if no binary classification tasks.*/
  virtual data_size_t NumPositiveData() const { return 0; }

  virtual void ConvertOutput(const double* input, double* output) const {
    output[0] = input[0];
  }

  virtual std::string ToString() const = 0;

  ObjectiveFunction() = default;
  /*! \brief Disable copy */
  ObjectiveFunction& operator=(const ObjectiveFunction&) = delete;
  /*! \brief Disable copy */
  ObjectiveFunction(const ObjectiveFunction&) = delete;

  /*!
  * \brief Create object of objective function
  * \param type Specific type of objective function
  * \param config Config for objective function
  */
  LIGHTGBM_EXPORT static ObjectiveFunction* CreateObjectiveFunction(const std::string& type,
    const Config& config);

  /*!
  * \brief Load objective function from string object
  */
  LIGHTGBM_EXPORT static ObjectiveFunction* CreateObjectiveFunction(const std::string& str);
};

}  // namespace LightGBM

#endif   // LightGBM_OBJECTIVE_FUNCTION_H_
