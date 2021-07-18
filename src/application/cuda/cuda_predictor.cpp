/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_predictor.hpp"

namespace LightGBM {

CUDAPredictor::CUDAPredictor(Boosting* boosting, int start_iteration, int num_iteration, bool is_raw_score,
            bool predict_leaf_index, bool predict_contrib, bool early_stop,
            int early_stop_freq, double early_stop_margin):
Predictor(boosting, start_iteration, num_iteration, is_raw_score,
          predict_leaf_index, predict_contrib, early_stop,
          early_stop_freq, early_stop_margin) {
  InitCUDAModel();
}

void CUDAPredictor::InitCUDAModel() {
  boosting_->GetCUDAModel(&cuda_models_);
  const size_t num_tree_models = cuda_models_.size();
  std::vector<const int*> all_tree_left_child(num_tree_models, nullptr);
  std::vector<const int*> all_tree_right_child(num_tree_models, nullptr);
  std::vector<const int*> all_tree_split_feature_inner(num_tree_models, nullptr);
  std::vector<const int*> all_tree_split_feature(num_tree_models, nullptr);
  std::vector<const uint32_t*> all_tree_threshold_in_bin(num_tree_models, nullptr);
  std::vector<const double*> all_tree_threshold(num_tree_models, nullptr);
  std::vector<const int8_t*> all_tree_decision_type(num_tree_models, nullptr);
  std::vector<const double*> all_tree_leaf_value(num_tree_models, nullptr);
  std::vector<int> all_tree_num_leaves(num_tree_models, 0);

  for (size_t i = 0; i < cuda_models_.size(); ++i) {
    all_tree_left_child[i] = cuda_models_[i]->cuda_left_child();
    all_tree_right_child[i] = cuda_models_[i]->cuda_right_child();
    all_tree_split_feature_inner[i] = cuda_models_[i]->cuda_split_feature_inner();
    all_tree_split_feature[i] = cuda_models_[i]->cuda_split_feature();
    all_tree_threshold_in_bin[i] = cuda_models_[i]->cuda_threshold_in_bin();
    all_tree_threshold[i] = cuda_models_[i]->cuda_threshold();
    all_tree_decision_type[i] = cuda_models_[i]->cuda_decision_type();
    all_tree_leaf_value[i] = cuda_models_[i]->cuda_leaf_value();
    all_tree_num_leaves[i] = cuda_models_[i]->num_leaves();
  }

  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_all_tree_num_leaves_,
                                    all_tree_num_leaves.data(),
                                    all_tree_num_leaves.size(),
                                    __FILE__,
                                    __LINE__);

  InitCUDAMemoryFromHostMemoryOuter<const int*>(&cuda_all_tree_left_child_,
                                           all_tree_left_child.data(),
                                           all_tree_left_child.size(),
                                           __FILE__,
                                           __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const int*>(&cuda_all_tree_right_child_,
                                           all_tree_right_child.data(),
                                           all_tree_right_child.size(),
                                           __FILE__,
                                           __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const int*>(&cuda_all_tree_split_feature_inner_,
                                           all_tree_split_feature_inner.data(),
                                           all_tree_split_feature_inner.size(),
                                           __FILE__,
                                           __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const int*>(&cuda_all_tree_split_feature_,
                                           all_tree_split_feature.data(),
                                           all_tree_split_feature.size(),
                                           __FILE__,
                                           __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const uint32_t*>(&cuda_all_tree_threshold_in_bin_,
                                           all_tree_threshold_in_bin.data(),
                                           all_tree_threshold_in_bin.size(),
                                           __FILE__,
                                           __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const double*>(&cuda_all_tree_threshold_,
                                           all_tree_threshold.data(),
                                           all_tree_threshold.size(),
                                           __FILE__,
                                           __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const int8_t*>(&cuda_all_tree_decision_type_,
                                           all_tree_decision_type.data(),
                                           all_tree_decision_type.size(),
                                           __FILE__,
                                           __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const double*>(&cuda_all_tree_leaf_value_,
                                           all_tree_leaf_value.data(),
                                           all_tree_leaf_value.size(),
                                           __FILE__,
                                           __LINE__);
}

CUDAPredictor::~CUDAPredictor() {}

void CUDAPredictor::InitCUDAData(const char* data_filename, const bool header, const bool disable_shape_check) {
  auto label_idx = header ? -1 : boosting_->LabelIdx();
  auto parser = std::unique_ptr<Parser>(Parser::CreateParser(data_filename, header, boosting_->MaxFeatureIdx() + 1, label_idx));

  if (parser == nullptr) {
    Log::Fatal("Could not recognize the data format of data file %s", data_filename);
  }
  if (!header && !disable_shape_check && parser->NumFeatures() != boosting_->MaxFeatureIdx() + 1) {
    Log::Fatal("The number of features in data (%d) is not the same as it was in training data (%d).\n" \
                "You can set ``predict_disable_shape_check=true`` to discard this error, but please be aware what you are doing.", parser->NumFeatures(), boosting_->MaxFeatureIdx() + 1);
  }
  TextReader<data_size_t> predict_data_reader(data_filename, header);
  std::vector<int> feature_remapper(parser->NumFeatures(), -1);
  bool need_adjust = false;
  if (header) {
    std::string first_line = predict_data_reader.first_line();
    std::vector<std::string> header_words = Common::Split(first_line.c_str(), "\t,");
    std::unordered_map<std::string, int> header_mapper;
    for (int i = 0; i < static_cast<int>(header_words.size()); ++i) {
      if (header_mapper.count(header_words[i]) > 0) {
        Log::Fatal("Feature (%s) appears more than one time.", header_words[i].c_str());
      }
      header_mapper[header_words[i]] = i;
    }
    const auto& fnames = boosting_->FeatureNames();
    for (int i = 0; i < static_cast<int>(fnames.size()); ++i) {
      if (header_mapper.count(fnames[i]) <= 0) {
        Log::Warning("Feature (%s) is missed in data file. If it is weight/query/group/ignore_column, you can ignore this warning.", fnames[i].c_str());
      } else {
        feature_remapper[header_mapper.at(fnames[i])] = i;
      }
    }
    for (int i = 0; i < static_cast<int>(feature_remapper.size()); ++i) {
      if (feature_remapper[i] >= 0 && i != feature_remapper[i]) {
        need_adjust = true;
        break;
      }
    }
  }
  // function for parse data
  std::function<void(const char*, std::vector<std::pair<int, double>>*)> parser_fun;
  double tmp_label;
  parser_fun = [&parser, &feature_remapper, &tmp_label, need_adjust]
  (const char* buffer, std::vector<std::pair<int, double>>* feature) {
    parser->ParseOneLine(buffer, feature, &tmp_label);
    if (need_adjust) {
      int i = 0, j = static_cast<int>(feature->size());
      while (i < j) {
        if (feature_remapper[(*feature)[i].first] >= 0) {
          (*feature)[i].first = feature_remapper[(*feature)[i].first];
          ++i;
        } else {
          // move the non-used features to the end of the feature vector
          std::swap((*feature)[i], (*feature)[--j]);
        }
      }
      feature->resize(i);
    }
  };

  row_ptr_.clear();
  feature_index_.clear();
  feature_values_.clear();
  row_ptr_.emplace_back(0);
  num_data_ = 0;
  std::function<void(data_size_t, const std::vector<std::string>&)> process_func = 
    [&parser_fun, this] (data_size_t, const std::vector<std::string>& lines) {
    std::vector<std::pair<int, double>> oneline_features;
    for (data_size_t i = 0; i < static_cast<data_size_t>(lines.size()); ++i) {
      oneline_features.clear();
      // parser
      parser_fun(lines[i].c_str(), &oneline_features);
      ++num_data_;
      row_ptr_.emplace_back(static_cast<int>(oneline_features.size()));
      for (const auto& pair : oneline_features) {
        feature_index_.emplace_back(pair.first);
        feature_values_.emplace_back(pair.second);
      }
    }
  };
  predict_data_reader.ReadAllAndProcessParallel(process_func);
  CHECK_EQ(row_ptr_.size(), static_cast<size_t>(num_data_) + 1);

  const int num_threads = OMP_NUM_THREADS();
  std::vector<int> thread_row_ptr_offset(num_threads + 1, 0);
  Threading::For<data_size_t>(0, num_data_ + 1, 512,
    [&thread_row_ptr_offset, this] (int thread_index, data_size_t start, data_size_t end) {
      int row_ptr_offset = 0;
      for (data_size_t i = start; i < end; ++i) {
        row_ptr_offset += row_ptr_[i];
      }
      thread_row_ptr_offset[thread_index + 1] = row_ptr_offset;
    });
  Threading::For<data_size_t>(0, num_data_ + 1, 512,
    [&thread_row_ptr_offset, this] (int thread_index, data_size_t start, data_size_t end) {
      for (data_size_t i = start; i < end; ++i) {
        row_ptr_[i] += thread_row_ptr_offset[thread_index];
      }
    });
  CHECK_EQ(feature_index_.size(), static_cast<size_t>(row_ptr_.back()));
  CHECK_EQ(feature_values_.size(), static_cast<size_t>(row_ptr_.back()));
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_row_ptr_,
                                    row_ptr_.data(),
                                    row_ptr_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_feature_index_per_row_,
                                    feature_index_.data(),
                                    feature_index_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_feature_values_,
                                    feature_values_.data(),
                                    feature_values_.size(),
                                    __FILE__,
                                    __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_tmp_score_,
                             static_cast<size_t>(num_data_),
                             __FILE__,
                             __LINE__);
}

void CUDAPredictor::Predict(const char* data_filename, const char* result_filename, bool header, bool disable_shape_check) {
  auto writer = VirtualFileWriter::Make(result_filename);
  if (!writer->Init()) {
    Log::Fatal("Prediction results file %s cannot be found", result_filename);
  }
  InitCUDAData(data_filename, header, disable_shape_check);
  LaunchPredictForMapKernel(cuda_tmp_score_);
  CopyFromCUDADeviceToHostOuter<double>(tmp_score_.data(),
                                   cuda_tmp_score_,
                                   static_cast<size_t>(num_data_),
                                   __FILE__,
                                   __LINE__);
  for (data_size_t i = 0; i < num_data_; ++i) {
    auto str_result = Common::Join<double>(std::vector<double>{tmp_score_[i]}, "\t");
    writer->Write(str_result.c_str(), str_result.size());
    writer->Write("\n", 1);
  }
}

void CUDAPredictor::InitCUDAData(std::function<std::vector<std::pair<int, double>>(int row_idx)> get_row_fun,
                    const data_size_t num_row, const int num_col) {
  row_ptr_.clear();
  feature_index_.clear();
  feature_values_.clear();
  row_ptr_.emplace_back(0);
  num_data_ = num_row;
  std::function<void(data_size_t, const std::vector<std::string>&)> process_func = 
    [&get_row_fun, this] (data_size_t data_index, const std::vector<std::string>& lines) {
    std::vector<std::pair<int, double>> oneline_features;
    for (data_size_t i = 0; i < static_cast<data_size_t>(lines.size()); ++i) {
      oneline_features = get_row_fun(data_index);
      row_ptr_.emplace_back(static_cast<int>(oneline_features.size()));
      for (const auto& pair : oneline_features) {
        feature_index_.emplace_back(pair.first);
        feature_values_.emplace_back(pair.second);
      }
    }
  };
  // TODO(shiyu1994): fix read from get_row_fun directly.
  
  CHECK_EQ(row_ptr_.size(), static_cast<size_t>(num_data_) + 1);

  const int num_threads = OMP_NUM_THREADS();
  std::vector<int> thread_row_ptr_offset(num_threads + 1, 0);
  Threading::For<data_size_t>(0, num_data_ + 1, 512,
    [&thread_row_ptr_offset, this] (int thread_index, data_size_t start, data_size_t end) {
      int row_ptr_offset = 0;
      for (data_size_t i = start; i < end; ++i) {
        row_ptr_offset += row_ptr_[i];
      }
      thread_row_ptr_offset[thread_index + 1] = row_ptr_offset;
    });
  Threading::For<data_size_t>(0, num_data_ + 1, 512,
    [&thread_row_ptr_offset, this] (int thread_index, data_size_t start, data_size_t end) {
      for (data_size_t i = start; i < end; ++i) {
        row_ptr_[i] += thread_row_ptr_offset[thread_index];
      }
    });
  CHECK_EQ(feature_index_.size(), static_cast<size_t>(row_ptr_.back()));
  CHECK_EQ(feature_values_.size(), static_cast<size_t>(row_ptr_.back()));
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_row_ptr_,
                                    row_ptr_.data(),
                                    row_ptr_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_feature_index_per_row_,
                                    feature_index_.data(),
                                    feature_index_.size(),
                                    __FILE__,
                                    __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_feature_values_,
                                    feature_values_.data(),
                                    feature_values_.size(),
                                    __FILE__,
                                    __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_tmp_score_,
                             static_cast<size_t>(num_data_) * static_cast<size_t>(num_pred_one_row_),
                             __FILE__,
                             __LINE__);
}

void CUDAPredictor::Predict(std::function<std::vector<std::pair<int, double>>(data_size_t)> get_row_fun,
                            const data_size_t num_row, const int num_col) {
  InitCUDAData(get_row_fun, num_row, num_col);
  LaunchPredictForMapKernel(cuda_tmp_score_);
  CopyFromCUDADeviceToHostOuter<double>(tmp_score_.data(),
                                   cuda_tmp_score_,
                                   static_cast<size_t>(num_data_),
                                   __FILE__,
                                   __LINE__);
}

}  // namespace LightGBM
