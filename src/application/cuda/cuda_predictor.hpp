/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifndef LIGHTGBM_CUDA_PREDICTOR_HPP_
#define LIGHTGBM_CUDA_PREDICTOR_HPP_

#include <LightGBM/cuda/cuda_tree.hpp>

#include "../predictor.hpp"

#define CDUA_PREDICTOR_MAX_TREE_SIZE (1024)

namespace LightGBM {

class CUDAPredictor : public Predictor {
 public:
  /*!
  * \brief Constructor
  * \param boosting Input boosting model
  * \param start_iteration Start index of the iteration to predict
  * \param num_iteration Number of boosting round
  * \param is_raw_score True if need to predict result with raw score
  * \param predict_leaf_index True to output leaf index instead of prediction score
  * \param predict_contrib True to output feature contributions instead of prediction score
  * \param num_data Number of data in the test set
  */
  CUDAPredictor(Boosting* boosting, int start_iteration, int num_iteration, bool is_raw_score,
            bool predict_leaf_index, bool predict_contrib, bool early_stop,
            int early_stop_freq, double early_stop_margin);

  void Predict(const char* data_filename, const char* result_filename, bool header, bool disable_shape_check) override;

  void Predict(std::function<std::vector<std::pair<int, double>>(data_size_t)> get_row_fun,
               const data_size_t num_row, const int num_col);

  /*!
  * \brief Destructor
  */
  ~CUDAPredictor();

 private:
  void InitCUDAModel();

  void InitCUDAData(const char* data_filename, const bool header, const bool diable_shape_check);

  void InitCUDAData(std::function<std::vector<std::pair<int, double>>(int row_idx)> get_row_fun,
                    const data_size_t num_row, const int num_col);

  void PredictForMap();

  void LaunchPredictForMapKernel(double* score);

  BatchPredictFunction batch_predict_fun_;
  data_size_t num_data_;
  std::vector<std::unique_ptr<CUDATree>> cuda_models_;

  int* cuda_all_tree_num_leaves_;
  const int** cuda_all_tree_left_child_;
  const int** cuda_all_tree_right_child_;
  const int** cuda_all_tree_split_feature_inner_;
  const int** cuda_all_tree_split_feature_;
  const uint32_t** cuda_all_tree_threshold_in_bin_;
  const double** cuda_all_tree_threshold_;
  const int8_t** cuda_all_tree_decision_type_;
  const double** cuda_all_tree_leaf_value_;

  std::vector<int> row_ptr_;
  std::vector<int> feature_index_;
  std::vector<double> feature_values_;
  std::vector<double> tmp_score_;

  int* cuda_row_ptr_;
  int* cuda_feature_index_per_row_;
  double* cuda_feature_values_;
  double* cuda_tmp_score_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_CUDA_PREDICTOR_HPP_
