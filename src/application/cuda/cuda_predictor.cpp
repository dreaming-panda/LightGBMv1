/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "cuda_predictor.hpp"

namespace LightGBM {

CUDAPredictor::CUDAPredictor(Boosting* boosting, int start_iteration, int num_iteration, bool is_raw_score,
  bool predict_leaf_index, bool predict_contrib, bool early_stop,
  int early_stop_freq, double early_stop_margin):
  Predictor(boosting, start_iteration, num_iteration, is_raw_score, predict_leaf_index, predict_contrib, early_stop, early_stop_freq, early_stop_margin) {
  InitCUDAModel();
}


void CUDAPredictor::Predict(const char* data_filename, const char* result_filename, bool header, bool disable_shape_check) {
  const data_size_t num_data = ReadDataToCUDADevice(data_filename, header, disable_shape_check);
  result_buffer_.resize(num_data, 0.0f);
  // TODO(shiyu1994): free memory when prediction is finished
  AllocateCUDAMemoryOuter<double>(&cuda_data_, static_cast<size_t>(num_data * num_feature_), __FILE__, __LINE__);
  AllocateCUDAMemoryOuter<double>(&cuda_result_buffer_, static_cast<size_t>(num_data), __FILE__, __LINE__);
  LaunchPredictKernel(num_data);
}

int CUDAPredictor::ReadDataToCUDADevice(const char* data_filename, const bool header, const bool disable_shape_check) {
  auto label_idx = header ? -1 : boosting_->LabelIdx();
  auto parser = std::unique_ptr<Parser>(Parser::CreateParser(data_filename, header, boosting_->MaxFeatureIdx() + 1, label_idx));
  if (parser == nullptr) {
    Log::Fatal("Could not recognize the data format of data file %s", data_filename);
  }
  if (!header && !disable_shape_check && parser->NumFeatures() != boosting_->MaxFeatureIdx() + 1) {
    Log::Fatal("The number of features in data (%d) is not the same as it was in training data (%d).\n" \
                "You can set ``predict_disable_shape_check=true`` to discard this error, but please be aware what you are doing.",
                parser->NumFeatures(), boosting_->MaxFeatureIdx() + 1);
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
  std::vector<int> feature_index_buffer;
  std::vector<double> feature_value_buffer;
  std::vector<data_size_t> feature_value_num_buffer;
  predict_feature_index_.clear();
  predict_feature_value_.clear();
  std::function<void(data_size_t, const std::vector<std::string>&)>
      process_fun = [&parser_fun, this, &feature_index_buffer, &feature_value_buffer, &feature_value_num_buffer](
                        data_size_t /*start_index*/, const std::vector<std::string>& lines) {
    std::vector<std::pair<int, double>> oneline_features;
    feature_index_buffer.clear();
    feature_value_buffer.clear();
    feature_value_num_buffer.clear();
    OMP_INIT_EX();
    #pragma omp parallel for schedule(static) firstprivate(oneline_features)
    for (data_size_t i = 0; i < static_cast<data_size_t>(lines.size()); ++i) {
      OMP_LOOP_EX_BEGIN();
      oneline_features.clear();
      // parser
      parser_fun(lines[i].c_str(), &oneline_features);
      for (const auto& pair : oneline_features) {
        feature_index_buffer.emplace_back(pair.first);
        feature_value_buffer.emplace_back(pair.second);
      }
      feature_value_num_buffer.emplace_back(static_cast<data_size_t>(oneline_features.size()));
      OMP_LOOP_EX_END();
    }
    OMP_THROW_EX();
    for (size_t i = 0; i < feature_index_buffer.size(); ++i) {
      predict_feature_index_.emplace_back(feature_index_buffer[i]);
      predict_feature_value_.emplace_back(feature_value_buffer[i]);
    }
    for (size_t i = 0; i < feature_value_num_buffer.size(); ++i) {
      predict_row_ptr_.emplace_back(feature_value_num_buffer[i]);
    }
  };
  const data_size_t num_data = static_cast<data_size_t>(predict_row_ptr_.size());
  predict_data_reader.ReadAllAndProcessParallel(process_fun);
  GetPredictRowPtr();
  InitCUDAMemoryFromHostMemoryOuter<double>(&cuda_predict_feature_value_,
                                            predict_feature_value_.data(),
                                            predict_feature_value_.size(),
                                            __FILE__,
                                            __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_predict_feature_index_,
                                            predict_feature_index_.data(),
                                            predict_feature_index_.size(),
                                            __FILE__,
                                            __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<data_size_t>(&cuda_predict_row_ptr_,
                                            predict_row_ptr_.data(),
                                            predict_row_ptr_.size(),
                                            __FILE__,
                                            __LINE__);
  return num_data;
}

void CUDAPredictor::GetPredictRowPtr() {
  const int num_threads = OMP_NUM_THREADS();
  std::vector<data_size_t> thread_offset(num_threads + 1, 0);
  const data_size_t num_data = static_cast<data_size_t>(predict_row_ptr_.size());
  Threading::For<data_size_t>(0, num_data, 512,
    [this, &thread_offset] (int thread_index, data_size_t start, data_size_t end) {
      int num_value_in_thread = 0;
      for (data_size_t i = start; i < end; ++i) {
        num_value_in_thread += predict_row_ptr_[i];
      }
      thread_offset[thread_index + 1] = num_value_in_thread;
    });
  for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
    thread_offset[thread_index + 1] += thread_offset[thread_index];
  }
  Threading::For<data_size_t>(0, num_data, 512,
    [this, &thread_offset] (int thread_index, data_size_t start, data_size_t end) {
      int offset = thread_offset[thread_index];
      for (data_size_t i = start; i < end; ++i) {
        const data_size_t num_feature_values = predict_row_ptr_[i];
        predict_row_ptr_[i] += offset;
        offset += num_feature_values;
      }
      CHECK_EQ(offset, thread_offset[thread_index + 1]);
    });
  predict_row_ptr_.emplace_back(thread_offset.back());
}

void CUDAPredictor::InitCUDAModel() {
  const std::vector<std::unique_ptr<Tree>>& models = boosting_->models();
  const int num_trees = static_cast<int>(models.size());
  num_trees_ = num_trees;
  std::vector<int> tree_num_leaves(num_trees, 0);
  std::vector<const int*> tree_left_child(num_trees, nullptr);
  std::vector<const int*> tree_right_child(num_trees, nullptr);
  std::vector<const double*> tree_leaf_value(num_trees, nullptr);
  std::vector<const double*> tree_threshold(num_trees, nullptr);
  std::vector<const int8_t*> tree_decision_type(num_trees, nullptr);
  std::vector<const int*> tree_split_feature_index(num_trees, nullptr);
  const int num_threads = OMP_NUM_THREADS();
  #pragma omp parallel for schedule(static) num_threads(num_threads) if (num_trees >= 1024)
  for (int tree_index = 0; tree_index < num_trees; ++tree_index) {
    tree_num_leaves[tree_index] = models[tree_index]->num_leaves();
    CHECK(models[tree_index]->is_cuda_tree());
    const CUDATree* cuda_tree = reinterpret_cast<const CUDATree*>(models[tree_index].get());
    tree_left_child[tree_index] = cuda_tree->cuda_left_child();
    tree_right_child[tree_index] = cuda_tree->cuda_right_child();
    tree_leaf_value[tree_index] = cuda_tree->cuda_leaf_value();
    tree_threshold[tree_index] = cuda_tree->cuda_threshold();
    tree_decision_type[tree_index] = cuda_tree->cuda_decision_type();
    tree_split_feature_index[tree_index] = cuda_tree->cuda_split_feature();
  }
  InitCUDAMemoryFromHostMemoryOuter<int>(&cuda_tree_num_leaves_,
                                         tree_num_leaves.data(),
                                         tree_num_leaves.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const int*>(&cuda_left_child_,
                                         tree_left_child.data(),
                                         tree_left_child.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const int*>(&cuda_right_child_,
                                         tree_right_child.data(),
                                         tree_right_child.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const double*>(&cuda_leaf_value_,
                                         tree_leaf_value.data(),
                                         tree_leaf_value.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const double*>(&cuda_threshold_,
                                         tree_threshold.data(),
                                         tree_threshold.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const int8_t*>(&cuda_decision_type_,
                                         tree_decision_type.data(),
                                         tree_decision_type.size(),
                                         __FILE__,
                                         __LINE__);
  InitCUDAMemoryFromHostMemoryOuter<const int*>(&cuda_split_feature_index_,
                                                tree_split_feature_index.data(),
                                                tree_split_feature_index.size(),
                                                __FILE__,
                                                __LINE__);
}

}  // namespace LightGBM
