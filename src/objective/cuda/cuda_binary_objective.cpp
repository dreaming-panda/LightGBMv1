/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#ifdef USE_CUDA

#include "cuda_binary_objective.hpp"

namespace LightGBM {

CUDABinaryLogloss::CUDABinaryLogloss(const Config& config):
BinaryLogloss(config), ova_class_id_(-1) {}

CUDABinaryLogloss::CUDABinaryLogloss(const Config& config, const int ova_class_id):
BinaryLogloss(config, [ova_class_id](label_t label) { return static_cast<int>(label) == ova_class_id; }), ova_class_id_(ova_class_id) {}

CUDABinaryLogloss::CUDABinaryLogloss(const std::vector<std::string>& strs): BinaryLogloss(strs) {}

CUDABinaryLogloss::~CUDABinaryLogloss() {}

void CUDABinaryLogloss::Init(const Metadata& metadata, data_size_t num_data) {
  BinaryLogloss::Init(metadata, num_data);
  if (ova_class_id_ == -1) {
    cuda_label_ = metadata.cuda_metadata()->cuda_label();
    cuda_ova_label_ = nullptr;
  } else {
    Log::Warning("converting cuda labels with ova_class_id_ = %d", ova_class_id_);
    InitCUDAMemoryFromHostMemory<label_t>(&cuda_ova_label_, metadata.cuda_metadata()->cuda_label(), static_cast<size_t>(num_data), __FILE__, __LINE__);
    LaunchResetOVACUDALableKernel();
    cuda_label_ = cuda_ova_label_;
  }
  cuda_weights_ = metadata.cuda_metadata()->cuda_weights();
  AllocateCUDAMemory<double>(&cuda_boost_from_score_, 1, __FILE__, __LINE__);
  SetCUDAMemory<double>(cuda_boost_from_score_, 0, 1, __FILE__, __LINE__);
  if (label_weights_[0] != 1.0f || label_weights_[1] != 1.0f) {
    InitCUDAMemoryFromHostMemory<double>(&cuda_label_weights_, label_weights_, 2, __FILE__, __LINE__);
  } else {
    cuda_label_weights_ = nullptr;
  }
}

void CUDABinaryLogloss::GetGradients(const double* scores, score_t* gradients, score_t* hessians) const {
  LaunchGetGradientsKernel(scores, gradients, hessians);
  SynchronizeCUDADevice(__FILE__, __LINE__);
}

double CUDABinaryLogloss::BoostFromScore(int) const {
  LaunchBoostFromScoreKernel();
  SynchronizeCUDADevice(__FILE__, __LINE__);
  double boost_from_score = 0.0f;
  CopyFromCUDADeviceToHost<double>(&boost_from_score, cuda_boost_from_score_, 1, __FILE__, __LINE__);
  Log::Warning("boost_from_score = %f", boost_from_score);
  return boost_from_score;
}

void CUDABinaryLogloss::ConvertOutputCUDA(const data_size_t num_data, const double* input, double* output) const {
  LaunchConvertOutputCUDAKernel(num_data, input, output);
}

}  // namespace LightGBM

#endif  // USE_CUDA
