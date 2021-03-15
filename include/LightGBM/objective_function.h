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

#include <fstream>

namespace LightGBM {
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
    double* /*grad_scale*/, double* /*hess_scale*/) {}

  virtual void DiscretizeGradients(score_t* /*gradients*/, score_t* /*hessians*/,
    int_score_t* /*int_gradients*/, int_score_t* /*int_hessians*/,
    double* /*grad_scale*/, double* /*hess_scale*/) const {}

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

  template <typename SCORE_T>
  void DumpGradientToFile(const SCORE_T* gradients, const SCORE_T* hessians, const int iter,
    const int num_data, const int num_class, const std::string suffix, const double grad_scale,
    const double hess_scale, const bool is_int) const {
    static std::vector<int> sample_indices;
    if (sample_indices.size() == 0) {
      std::mt19937 eng;
      std::uniform_real_distribution<double> dist;
      for (int i = 0; i < num_data; ++i) {
        const double prob = dist(eng);
        if (prob <= 0.1f) {
          sample_indices.emplace_back(i);
        }
      }
    }
    std::vector<char> buffer(100);
    Common::Int32ToStr(iter, buffer.data());
    std::ofstream fout(std::string("gradient_" + std::string(buffer.data()) + "_" + suffix + std::string(".csv")));
    fout << "gradient,hessian" << std::endl;
    if (!is_int) {
      for (int i = 0; i < num_class; ++i) {
        const int base = i * num_data;
        for (int j = 0; j < static_cast<int>(sample_indices.size()); ++j) {
          const int index = sample_indices[j];
          fout << gradients[base + index] << "," << hessians[base + index] << std::endl;
        }
      }
    } else {
      for (int i = 0; i < num_class; ++i) {
        const int base = i * num_data;
        for (int j = 0; j < static_cast<int>(sample_indices.size()); ++j) {
          const int index = sample_indices[j];
          fout << gradients[base + index] * grad_scale << "," << hessians[base + index] * hess_scale << std::endl;
        }
      }
    }
  }

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

 protected:
  bool GetQuantile(const score_t* gradients, const score_t* hessians,
    const int num_quantiles, const data_size_t num_data);

  void Quantize(const score_t gradient, const score_t hessian,
    int_score_t* grad_int, int_score_t* hess_int, const int thread_id,
    double grad_scale_inverse, double hess_scale_inverse);

  void UniformDiscretizeGradients(score_t* gradients, score_t* hessians,
    int_score_t* int_gradients, int_score_t* int_hessians,
    data_size_t num_data, double* grad_scale, double* hess_scale);

  std::vector<std::mt19937> rand_generators_;
  std::vector<std::uniform_real_distribution<double>> uniform_dists_;
  std::vector<double> grad_quantiles_;
  std::vector<double> hess_quantiles_;
  std::vector<int> grad_int_map_;
  std::vector<int> hess_int_map_;
};

}  // namespace LightGBM

#endif   // LightGBM_OBJECTIVE_FUNCTION_H_
