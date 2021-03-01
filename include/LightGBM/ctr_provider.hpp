/*!
  * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
  * Licensed under the MIT License. See LICENSE file in the project root for license information.
  */
#ifndef LIGHTGBM_CTR_PROVIDER_H_
#define LIGHTGBM_CTR_PROVIDER_H_

#include <LightGBM/config.h>
#include <LightGBM/network.h>
#include <LightGBM/utils/text_reader.h>
#include <LightGBM/utils/threading.h>
#include <LightGBM/parser_base.h>
#include <LightGBM/bin.h>
#include <LightGBM/tree.h>

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace LightGBM {

// transform categorical features to ctr values before the bin construction process
class CTRProvider {
 public:
  class CatConverter {
   protected:
    std::unordered_map<int, int> cat_fid_to_convert_fid_;

   public:
    virtual ~CatConverter() {}

    virtual double CalcValue(const double sum_label, const double sum_count,
      const double all_fold_sum_count, const double prior) const = 0;

    virtual double CalcValue(const double sum_label, const double sum_count,
      const double all_fold_sum_count) const = 0;

    virtual std::string DumpToString() const = 0;

    virtual std::string Name() const = 0;

    virtual void SetPrior(const double /*prior*/, const double /*prior_weight*/) {}

    void SetCatFidToConvertFid(const std::unordered_map<int, int>& cat_fid_to_convert_fid) {
      cat_fid_to_convert_fid_ = cat_fid_to_convert_fid;
    }

    void RegisterConvertFid(const int cat_fid, const int convert_fid) {
      cat_fid_to_convert_fid_[cat_fid] = convert_fid;
    }

    int GetConvertFid(const int cat_fid) const {
      return cat_fid_to_convert_fid_.at(cat_fid);
    }

    static CatConverter* CreateFromString(const std::string& model_string, const double prior_weight) {
      std::vector<std::string> split_model_string = Common::Split(model_string.c_str(), ",");
      if (split_model_string.size() != 2) {
        Log::Fatal("Invalid CatConverter model string %s", model_string.c_str());
      }
      const std::string& cat_converter_name = split_model_string[0];
      CatConverter* cat_converter = nullptr;
      if (Common::StartsWith(cat_converter_name, std::string("label_mean_prior_ctr"))) {
        double prior = 0.0f;
        Common::Atof(Common::Split(cat_converter_name.c_str(), ':')[1].c_str(), &prior);
        cat_converter = new CTRConverterLabelMean();
        cat_converter->SetPrior(prior, prior_weight);
      } else if (Common::StartsWith(cat_converter_name, std::string("ctr"))) {
        double prior = 0.0f;
        Common::Atof(Common::Split(cat_converter_name.c_str(), ':')[1].c_str(), &prior);
        cat_converter = new CTRConverter(prior);
        cat_converter->SetPrior(prior, prior_weight);
      } else if (cat_converter_name == std::string("count")) {
        cat_converter = new CountConverter();
      } else {
        Log::Fatal("Invalid CatConverter model string %s", model_string.c_str());
      }
      cat_converter->cat_fid_to_convert_fid_.clear();

      const std::string& feature_map = split_model_string[1];
      std::stringstream feature_map_stream(feature_map);
      int key = 0, val = 0;
      while (feature_map_stream >> key) {
        CHECK_EQ(feature_map_stream.get(), ':');
        feature_map_stream >> val;
        cat_converter->cat_fid_to_convert_fid_[key] = val;
        feature_map_stream.get();
      }

      return cat_converter;
    }
  };

  class CTRConverter: public CatConverter {
   public:
    explicit CTRConverter(const double prior): prior_(prior) {}
    inline double CalcValue(const double sum_label, const double sum_count,
      const double /*all_fold_sum_count*/) const override {
      return (sum_label + prior_ * prior_weight_) / (sum_count + prior_weight_);
    }

    inline double CalcValue(const double sum_label, const double sum_count,
      const double /*all_fold_sum_count*/, const double /*prior*/) const override {
      return (sum_label + prior_ * prior_weight_) / (sum_count + prior_weight_);
    }

    void SetPrior(const double /*prior*/, const double prior_weight) override {
      prior_weight_ = prior_weight;
    }

    std::string Name() const override {
      std::stringstream str_stream;
      str_stream << "ctr:" << prior_;
      return str_stream.str();
    }

    std::string DumpToString() const override {
      std::stringstream str_stream;
      str_stream << Name() << "," << DumpDictToString(cat_fid_to_convert_fid_, '#');
      return str_stream.str();
    }

   private:
    const double prior_;
    double prior_weight_;
  };

  class CountConverter: public CatConverter {
   public:
    CountConverter() {}

   private:
    inline double CalcValue(const double /*sum_label*/, const double /*sum_count*/,
      const double all_fold_sum_count) const override {
      return all_fold_sum_count;
    }

    inline double CalcValue(const double /*sum_label*/, const double /*sum_count*/,
      const double all_fold_sum_count, const double /*prior*/) const override {
      return all_fold_sum_count;
    }

    std::string Name() const override {
      return std::string("count");
    }

    std::string DumpToString() const override {
      std::stringstream str_stream;
      str_stream << Name() << "," << DumpDictToString(cat_fid_to_convert_fid_, '#');
      return str_stream.str();
    }
  };

  class CTRConverterLabelMean: public CatConverter {
   public:
    CTRConverterLabelMean() { prior_set_ = false; }

    void SetPrior(const double prior, const double prior_weight) override {
      prior_ = prior;
      prior_weight_ = prior_weight;
      prior_set_ = true;
    }

    inline double CalcValue(const double sum_label, const double sum_count,
      const double /*all_fold_sum_count*/) const override {
      if (!prior_set_) {
        Log::Fatal("CTRConverterLabelMean is not ready since the prior value is not set.");
      }
      return (sum_label + prior_weight_ * prior_) / (sum_count + prior_weight_);
    }

    inline double CalcValue(const double sum_label, const double sum_count,
      const double /*all_fold_sum_count*/, const double prior) const override {
      if (!prior_set_) {
        Log::Fatal("CTRConverterLabelMean is not ready since the prior value is not set.");
      }
      return (sum_label + prior * prior_weight_) / (sum_count + prior_weight_);
    }

    std::string Name() const override {
      std::stringstream str_stream;
      str_stream << "label_mean_prior_ctr:" << prior_;
      return str_stream.str();
    }

    std::string DumpToString() const override {
      std::stringstream str_stream;
      str_stream << Name() << "," << DumpDictToString(cat_fid_to_convert_fid_, '#');
      return str_stream.str();
    }

   private:
    double prior_;
    double prior_weight_;
    bool prior_set_;
  };

  ~CTRProvider() {
    training_data_fold_id_.clear();
    training_data_fold_id_.shrink_to_fit();
    fold_prior_.clear();
    fold_prior_.shrink_to_fit();
    is_categorical_feature_.clear();
    is_categorical_feature_.shrink_to_fit();
    count_info_.clear();
    label_info_.clear();
    cat_converters_.clear();
    cat_converters_.shrink_to_fit();
  }

  // for file data input and accumulating statistics when sampling from file
  static CTRProvider* CreateCTRProvider(Config* config) {
    std::unique_ptr<CTRProvider> ctr_provider(new CTRProvider(config));
    if (ctr_provider->GetNumCatConverters() == 0) {
      return nullptr;
    } else {
      return ctr_provider.release();
    }
  }

  // for pandas/numpy array data input
  static CTRProvider* CreateCTRProvider(Config* config,
    const std::vector<std::function<std::vector<double>(int row_idx)>>& get_row_fun,
    const std::function<double(int row_idx)>& get_label_fun,
    int32_t nmat, int32_t* nrow, int32_t ncol) {
    std::unique_ptr<CTRProvider> ctr_provider(new CTRProvider(config, get_row_fun, get_label_fun, nmat, nrow, ncol));
    if (ctr_provider->GetNumCatConverters() == 0) {
      return nullptr;
    } else {
      return ctr_provider.release();
    }
  }

  // for csr sparse matrix data input
  static CTRProvider* CreateCTRProvider(Config* config,
    const std::function<std::vector<std::pair<int, double>>(int row_idx)>& get_row_fun,
    const std::function<double(int row_idx)>& get_label_fun,
    int64_t nrow, int64_t ncol) {
    std::unique_ptr<CTRProvider> ctr_provider(new CTRProvider(config, get_row_fun, get_label_fun, nrow, ncol));
    if (ctr_provider->GetNumCatConverters() == 0) {
      return nullptr;
    } else {
      return ctr_provider.release();
    }
  }

  // for csc sparse matrix data input
  static CTRProvider* CreateCTRProvider(Config* config,
    const std::vector<std::unique_ptr<CSC_RowIterator>>& csc_func,
    const std::function<double(int row_idx)>& get_label_fun,
    int64_t nrow, int64_t ncol) {
    std::unique_ptr<CTRProvider> ctr_provider(new CTRProvider(config, csc_func, get_label_fun, nrow, ncol));
    if (ctr_provider->GetNumCatConverters() == 0) {
      return nullptr;
    } else {
      return ctr_provider.release();
    }
  }

  void PrepareCTRStatVectors();

  void ProcessOneLine(const std::vector<double>& one_line, double label,
    int line_idx, int thread_id, const std::vector<int>& fold_ids);

  void ProcessOneLine(const std::vector<std::pair<int, double>>& one_line, double label,
    int line_idx, std::vector<bool>* is_feature_processed, int thread_id, const std::vector<int>& fold_ids);

  void ProcessOneLine(const std::vector<std::pair<int, double>>& one_line, double label,
    int line_idx, std::vector<bool>* is_feature_processed, const std::vector<int>& fold_ids);

  std::string DumpModelInfo() const;

  static CTRProvider* RecoverFromModelString(const std::string model_string);

  bool IsCategorical(const int fid) const {
    if (fid < num_original_features_) {
      return is_categorical_feature_[fid];
    } else {
      return false;
    }
  }

  inline int GetNumOriginalFeatures() const {
    return num_original_features_;
  }

  inline int GetNumTotalFeatures() const {
    return num_total_features_;
  }

  inline int GetNumCatConverters() const {
    return static_cast<int>(cat_converters_.size());
  }

  void IterateOverCatConverters(int fid, double fval, int line_idx,
    const std::function<void(int convert_fid, int fid, double convert_value)>& write_func,
    const std::function<void(int fid)>& post_process_func) const;

  void IterateOverCatConverters(int fid, double fval,
    const std::function<void(int convert_fid, int fid, double convert_value)>& write_func,
    const std::function<void(int fid)>& post_process_func) const;

  inline void GetCTRStatForOneCatValue(int fid, double fval, const std::vector<int>& fold_ids,
    std::vector<double>* out_label_sum, std::vector<double>* out_total_count,
    std::vector<double>* out_all_fold_total_count) const {
    auto& out_label_sum_ref = *out_label_sum;
    auto& out_total_count_ref = *out_total_count;
    auto& out_all_fold_total_count_ref = *out_all_fold_total_count;
    for (int i = 0; i < num_ctr_partitions_; ++i) {
      out_label_sum_ref[i] = 0.0f;
      out_total_count_ref[i] = 0.0f;
      out_all_fold_total_count_ref[i] = 0.0f;
    }
    const int int_fval = static_cast<int>(fval);
    for (int i = 0; i < num_ctr_partitions_; ++i) {
      const auto& fold_label_info =
        label_info_.at(fid)[i].at(fold_ids[i]);
      const auto& fold_count_info =
        count_info_.at(fid)[i].at(fold_ids[i]);
      if (fold_count_info.count(int_fval) > 0) {
        out_label_sum_ref[i] = fold_label_info.at(int_fval);
        out_total_count_ref[i] = fold_count_info.at(int_fval);
      }
      const auto& all_fold_count_info = count_info_.at(fid)[i].back();
      if (all_fold_count_info.count(int_fval) > 0) {
        out_all_fold_total_count_ref[i] = all_fold_count_info.at(int_fval);
      }
    }
  }

  inline void GetCTRStatForOneCatValue(int fid, double fval, double* out_label_sum,
    double* out_total_count, double* out_all_fold_total_count) const {
    auto& out_label_sum_ref = *out_label_sum;
    auto& out_total_count_ref = *out_total_count;
    auto& out_all_fold_total_count_ref = *out_all_fold_total_count;
    out_label_sum_ref = 0.0f;
    out_total_count_ref = 0.0f;
    out_all_fold_total_count_ref = 0.0f;
    const int int_fval = static_cast<int>(fval);
    const auto& fold_label_info = label_info_.at(fid)[0].back();
    const auto& fold_count_info = count_info_.at(fid)[0].back();
    if (fold_count_info.count(int_fval) > 0) {
      out_label_sum_ref = fold_label_info.at(int_fval);
      out_total_count_ref = fold_count_info.at(int_fval);
    }
    out_all_fold_total_count_ref = out_total_count_ref;
  }

  template <bool IS_TRAIN>
  void IterateOverCatConvertersInner(int fid, double fval, const std::vector<int>& fold_ids,
    const std::function<void(int convert_fid, int fid, double convert_value)>& write_func,
    const std::function<void(int fid)>& post_process_func) const {
    if (IS_TRAIN) {
      std::vector<double> label_sum(num_ctr_partitions_, 0.0f);
      std::vector<double> total_count(num_ctr_partitions_, 0.0f);
      std::vector<double> all_fold_total_count(num_ctr_partitions_, 0.0f);
      GetCTRStatForOneCatValue(fid, fval, fold_ids, &label_sum, &total_count, &all_fold_total_count);
      for (const auto& cat_converter : cat_converters_) {
        double convert_value = 0.0f;
        for (int i = 0; i < num_ctr_partitions_; ++i) {
          convert_value += 
            cat_converter->CalcValue(label_sum[i], total_count[i], all_fold_total_count[i], fold_prior_[i][fold_ids[i]]);
        }
        convert_value /= num_ctr_partitions_;
        const int convert_fid = cat_converter->GetConvertFid(fid);
        write_func(convert_fid, fid, convert_value);
      }
      post_process_func(fid);
    } else {
      double label_sum = 0.0f, total_count = 0.0f, all_fold_total_count = 0.0f;
      GetCTRStatForOneCatValue(fid, fval, &label_sum, &total_count, &all_fold_total_count);
      for (const auto& cat_converter : cat_converters_) {
        const double convert_value = cat_converter->CalcValue(label_sum, total_count, all_fold_total_count);
        const int convert_fid = cat_converter->GetConvertFid(fid);
        write_func(convert_fid, fid, convert_value);
      }
      post_process_func(fid);
    }
  }

  template <bool IS_TRAIN>
  double HandleOneCatConverter(int fid, double fval, const std::vector<int>& fold_ids,
    const CTRProvider::CatConverter* cat_converter) const {
    if (IS_TRAIN) {
      std::vector<double> label_sum(num_ctr_partitions_, 0.0f);
      std::vector<double> total_count(num_ctr_partitions_, 0.0f);
      std::vector<double> all_fold_total_count(num_ctr_partitions_, 0.0f);
      GetCTRStatForOneCatValue(fid, fval, fold_ids, &label_sum, &total_count, &all_fold_total_count);
      double result = 0.0f;
      for (int i = 0; i < num_ctr_partitions_; ++i) {
        result += cat_converter->CalcValue(label_sum[i], total_count[i], all_fold_total_count[i], fold_prior_[i][fold_ids[i]]);   
      }
      result /= num_ctr_partitions_;
      return result;
    } else {
      double label_sum = 0.0f, total_count = 0.0f, all_fold_total_count = 0.0f;
      GetCTRStatForOneCatValue(fid, fval, &label_sum, &total_count, &all_fold_total_count);
      return cat_converter->CalcValue(label_sum, total_count, all_fold_total_count);
    }
  }

  void ConvertCatToCTR(std::vector<double>* features, int line_idx) const;

  void ConvertCatToCTR(std::vector<double>* features) const;

  void ConvertCatToCTR(std::vector<std::pair<int, double>>* features_ptr, const int fold_id) const;

  void ConvertCatToCTR(std::vector<std::pair<int, double>>* features_ptr) const;

  double ConvertCatToCTR(double fval, const CTRProvider::CatConverter* cat_converter,
    int col_idx, int line_idx) const;

  double ConvertCatToCTR(double fval, const CTRProvider::CatConverter* cat_converter,
    int col_idx) const;

  void ExtendFeatureNames(std::vector<std::string>* feature_names_ptr) const;

  template <typename INDEX_T>
  void WrapRowFunctions(
    std::vector<std::function<std::vector<double>(INDEX_T row_idx)>>* get_row_fun,
    int32_t* ncol, bool is_valid) const {
    const std::vector<std::function<std::vector<double>(INDEX_T row_idx)>> old_get_row_fun = *get_row_fun;
    get_row_fun->clear();
    for (size_t i = 0; i < old_get_row_fun.size(); ++i) {
      get_row_fun->push_back(WrapRowFunctionInner<double, INDEX_T>(&old_get_row_fun[i], is_valid));
    }
    *ncol = static_cast<int32_t>(num_total_features_);
  }

  template <typename INDEX_T>
  void WrapRowFunction(
    std::function<std::vector<std::pair<int, double>>(INDEX_T row_idx)>* get_row_fun,
    int64_t* ncol, bool is_valid) const {
    *get_row_fun = WrapRowFunctionInner<std::pair<int, double>, INDEX_T>(get_row_fun, is_valid);
    *ncol = static_cast<int64_t>(num_total_features_);
  }

  template <typename T, typename INDEX_T>
  std::function<std::vector<T>(INDEX_T row_idx)> WrapRowFunctionInner(
    const std::function<std::vector<T>(INDEX_T row_idx)>* get_row_fun, bool is_valid) const {
  std::function<std::vector<T>(INDEX_T row_idx)> old_get_row_fun = *get_row_fun;
    if (is_valid) {
      return [old_get_row_fun, this] (INDEX_T row_idx) {
        std::vector<T> row = old_get_row_fun(row_idx);
        ConvertCatToCTR(&row);
        return row;
      };
    } else {
      return [old_get_row_fun, this] (INDEX_T row_idx) {
        std::vector<T> row = old_get_row_fun(row_idx);
        ConvertCatToCTR(&row, row_idx);
        return row;
      };
    }
  }

  void WrapColIters(
    std::vector<std::unique_ptr<CSC_RowIterator>>* col_iters,
    int64_t* ncol_ptr, bool is_valid, int64_t num_row) const;

  Parser* FinishProcess(const int num_machines, Config* config);

  void InitFromParser(Config* config_from_loader, Parser* parser, const int num_machines,
    std::unordered_set<int>* categorical_features_from_loader);

  void AccumulateOneLineStat(const char* buffer, const size_t size, const data_size_t row_idx);

  void CreateCatShadowFeatureSet(const std::vector<const BinMapper*>& feature_bin_mappers);

 private:
  void SetConfig(const Config* config);

  explicit CTRProvider(const std::string model_string);

  explicit CTRProvider(Config* config);

  CTRProvider(Config* config,
    const std::vector<std::function<std::vector<double>(int row_idx)>>& get_row_fun,
    const std::function<double(int row_idx)>& get_label_fun, const int32_t nmat,
    const int32_t* nrow, const int32_t ncol);

  CTRProvider(Config* config,
    const std::function<std::vector<std::pair<int, double>>(int row_idx)>& get_row_fun,
    const std::function<double(int row_idx)>& get_label_fun,
    const int64_t nrow, const int64_t ncol);

  CTRProvider(Config* config,
    const std::vector<std::unique_ptr<CSC_RowIterator>>& csc_iters,
    const std::function<double(int row_idx)>& get_label_fun,
    const int64_t nrow, const int64_t ncol);

  template <bool ACCUMULATE_FROM_FILE>
  void ProcessOneLineInner(const std::vector<std::pair<int, double>>& one_line,
    double label, int line_idx, std::vector<bool>* is_feature_processed_ptr,
    std::unordered_map<int, std::vector<std::vector<std::unordered_map<int, int>>>>* count_info_ptr,
    std::unordered_map<int, std::vector<std::vector<std::unordered_map<int, label_t>>>>* label_info_ptr,
    std::vector<std::vector<label_t>>* label_sum_ptr, std::vector<std::vector<int>>* num_data_ptr, const std::vector<int>& fold_ids);

  // sync up ctr values by gathering statistics from all machines in distributed scenario
  void SyncCTRStat(std::vector<std::unordered_map<int, label_t>>* fold_label_sum_ptr,
    std::vector<std::unordered_map<int, int>>* fold_total_count_ptr, const int num_machines) const;

  // sync up statistics to calculate the ctr prior by gathering statistics from all machines in distributed scenario
  void SyncCTRPrior(const double label_sum, const int local_num_data, double* all_label_sum_ptr,
    int* all_num_data_ptr, int num_machines) const;

  // dump a dictionary to string
  template <typename T>
  static std::string DumpDictToString(const std::unordered_map<int, T>& dict, const char delimiter) {
    std::stringstream str_buf;
    if (dict.empty()) {
      return str_buf.str();
    }
    auto iter = dict.begin();
    str_buf << iter->first << ":" << iter->second;
    ++iter;
    for (; iter != dict.end(); ++iter) {
      str_buf << delimiter << iter->first << ":" << iter->second;
    }
    return str_buf.str();
  }

  int ParseMetaInfo(const char* filename, Config* config);

  void ExpandNumFeatureWhileAccumulate(const int new_largest_fid);

  inline void AddCountAndLabel(std::unordered_map<int, int>* count_map,
    std::unordered_map<int, label_t>* label_map,
    const int cat_value, const int count_value, const label_t label_value) {
    if (count_map->count(cat_value) == 0) {
      count_map->operator[](cat_value) = count_value;
      label_map->operator[](cat_value) = label_value;
    } else {
      count_map->operator[](cat_value) += count_value;
      label_map->operator[](cat_value) += label_value;
    }
  }

  inline void AddCountAndLabel(std::vector<std::vector<std::unordered_map<int, int>>>* count_map,
    std::vector<std::vector<std::unordered_map<int, label_t>>>* label_map,
    const int cat_value, const int count_value, const label_t label_value,
    const std::vector<int>& fold_ids) {
    for (int i = 0; i < num_ctr_partitions_; ++i) {
      if (count_map->at(i).at(fold_ids[i]).count(cat_value) == 0) {
        count_map->at(i).at(fold_ids[i])[cat_value] = count_value;
        label_map->at(i).at(fold_ids[i])[cat_value] = label_value;
      } else {
        count_map->at(i).at(fold_ids[i])[cat_value] += count_value;
        label_map->at(i).at(fold_ids[i])[cat_value] += label_value;
      }
    }
  }

  // parameter configuration
  Config config_;

  // size of training data
  data_size_t num_data_;
  // list of categorical feature indices (real index, not inner index of Dataset)
  std::vector<int> categorical_features_;

  // maps training data index to fold index
  std::vector<std::vector<int>> training_data_fold_id_;
  // prior used by per fold
  std::vector<std::vector<double>> fold_prior_;
  // weight of the prior in ctr calculation
  double prior_weight_;
  // record whether a feature is categorical in the original data
  std::vector<bool> is_categorical_feature_;

  // number of features in the original dataset, without adding count features
  int num_original_features_;
  // number of features after converting categorical features
  int num_total_features_;

  // number of threads used for ctr encoding
  int num_threads_;

  // the accumulated count information for ctr
  std::unordered_map<int, std::vector<std::vector<std::unordered_map<int, int>>>> count_info_;
  // the accumulated label sum information for ctr
  std::unordered_map<int, std::vector<std::vector<std::unordered_map<int, label_t>>>> label_info_;
  // the accumulated count information for ctr per thread
  std::vector<std::unordered_map<int, std::vector<std::vector<std::unordered_map<int, int>>>>> thread_count_info_;
  // the accumulated label sum information for ctr per thread
  std::vector<std::unordered_map<int, std::vector<std::vector<std::unordered_map<int, label_t>>>>> thread_label_info_;
  // the accumulated label sum per fold
  std::vector<std::vector<label_t>> fold_label_sum_;
  // the accumulated label sum per thread per fold
  std::vector<std::vector<std::vector<label_t>>> thread_fold_label_sum_;
  // the accumulated number of data per fold per thread
  std::vector<std::vector<std::vector<data_size_t>>> thread_fold_num_data_;
  // number of data per fold
  std::vector<std::vector<data_size_t>> fold_num_data_;
  // categorical value converters
  std::vector<std::unique_ptr<CatConverter>> cat_converters_;
  // whether the old categorical handling method is used
  bool keep_raw_cat_method_;

  // temporary parser used when accumulating statistics from file
  std::unique_ptr<Parser> tmp_parser_;
  // temporary oneline_features used when accumulating statistics from file
  std::vector<std::pair<int, double>> tmp_oneline_features_;
  // temporary random generator used when accumulating statistics from file,
  // used to generate training data folds for CTR calculations
  std::mt19937 tmp_mt_generator_;
  // temporary fold distribution probability when accumulating statistics from file
  std::vector<double> tmp_fold_probs_;
  // temporary fold distribution when accumulating statistics from file
  std::discrete_distribution<int> tmp_fold_distribution_;
  // temporary feature read mask when accumulating statistics from files
  std::vector<bool> tmp_is_feature_processed_;
  // mark whether the ctr statistics is accumulated from file
  bool accumulated_from_file_;
  // number of partitions to calculate ctr and count
  int num_ctr_partitions_;
  // stores the shadow ctr feature with partition id's > 0
  std::unique_ptr<CatShadowFeatureSet> cat_shadow_feature_set_ = nullptr;
  // raw categorical values
  std::vector<std::vector<int>> raw_cat_values_;
  // caategorical feature index order
  std::vector<int> cat_feature_order_;
};

class CTRParser : public Parser {
 public:
    explicit CTRParser(const Parser* inner_parser,
      const CTRProvider* ctr_provider, const bool is_valid):
      inner_parser_(inner_parser), ctr_provider_(ctr_provider), is_valid_(is_valid) {}

    inline void ParseOneLine(const char* str,
      std::vector<std::pair<int, double>>* out_features,
      double* out_label, const int line_idx = -1) const override {
      inner_parser_->ParseOneLine(str, out_features, out_label);
      if (is_valid_) {
        ctr_provider_->ConvertCatToCTR(out_features);
      } else {
        ctr_provider_->ConvertCatToCTR(out_features, line_idx);
      }
    }

    inline int NumFeatures() const override {
      return ctr_provider_->GetNumTotalFeatures();
    }

 private:
    std::unique_ptr<const Parser> inner_parser_;
    const CTRProvider* ctr_provider_;
    const bool is_valid_;
};

class CTR_CSC_RowIterator: public CSC_RowIterator {
 public:
  CTR_CSC_RowIterator(const void* col_ptr, int col_ptr_type, const int32_t* indices,
                  const void* data, int data_type, int64_t ncol_ptr, int64_t nelem, int col_idx,
                  const CTRProvider::CatConverter* cat_converter,
                  const CTRProvider* ctr_provider, bool is_valid, int64_t num_row):
    CSC_RowIterator(col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem, col_idx),
    col_idx_(col_idx),
    is_valid_(is_valid),
    num_row_(num_row) {
    cat_converter_ = cat_converter;
    ctr_provider_ = ctr_provider;
  }

  CTR_CSC_RowIterator(CSC_RowIterator* csc_iter, const int col_idx,
                  const CTRProvider::CatConverter* cat_converter,
                  const CTRProvider* ctr_provider, bool is_valid, int64_t num_row):
    CSC_RowIterator(*csc_iter),
    col_idx_(col_idx),
    is_valid_(is_valid),
    num_row_(num_row) {
    cat_converter_ = cat_converter;
    ctr_provider_ = ctr_provider;
  }

  double Get(int row_idx) override {
    const double value = CSC_RowIterator::Get(row_idx);
    if (is_valid_) {
      return ctr_provider_->ConvertCatToCTR(value, cat_converter_, col_idx_);
    } else {
      return ctr_provider_->ConvertCatToCTR(value, cat_converter_, col_idx_, row_idx);
    }
  }

  std::pair<int, double> NextNonZero() override {
    if (cur_row_idx_ + 1 < static_cast<int>(num_row_)) {
      auto pair = cached_pair_;
      if (cur_row_idx_ == cached_pair_.first) {
        cached_pair_ = CSC_RowIterator::NextNonZero();
      }
      if ((cur_row_idx_ + 1 < cached_pair_.first) || is_end_) {
        pair = std::make_pair(cur_row_idx_ + 1, 0.0f);
      } else {
        pair = cached_pair_;
      }
      ++cur_row_idx_;
      double value = 0.0f;
      if (is_valid_) {
        value = ctr_provider_->ConvertCatToCTR(pair.second, cat_converter_, col_idx_);
      } else {
        value = ctr_provider_->ConvertCatToCTR(pair.second, cat_converter_, col_idx_, pair.first);
      }
      pair.second = value;
      return pair;
    } else {
      return std::make_pair(-1, 0.0f);
    }
  }

  void Reset() override {
    CSC_RowIterator::Reset();
    cur_row_idx_ = -1;
    cached_pair_ = std::make_pair(-1, 0.0f);
  }

 private:
  const CTRProvider::CatConverter* cat_converter_;
  const CTRProvider* ctr_provider_;
  const int col_idx_;
  const bool is_valid_;
  const int64_t num_row_;

  int cur_row_idx_ = -1;
  std::pair<int, double> cached_pair_ = std::make_pair(-1, 0.0f);
};

class CatShadowFeatureSet {
 public:
  CatShadowFeatureSet(const int num_ctr_partitions, const int num_data,
    const std::vector<int>& categorical_features,
    const std::vector<const BinMapper*>& cat_feature_bin_mappers);

  void PushData(const int cat_feature_index, const double value,
                const int partition_id, const data_size_t row_idx,
                const int thread_id);

  void FinishLoad();

  BinIterator* GetShadowIterator(const int real_cat_feature_index, const int partition_id) const;

  void InitBoostingInfo(const int num_classes);

  void Boosting(const std::function<void(const double* pred_score,
    score_t* gradients, score_t* hessians)>& boosting_func);

  void AddPredictionToScore(const Tree* tree);

 private:
  std::vector<std::vector<std::unique_ptr<Bin>>> cat_feature_shadow_bins_;
  std::unordered_map<int, size_t> cat_feature_map_;
  const std::vector<const BinMapper*> cat_feature_bin_mappers_;
  const int num_ctr_partitions_;
  std::vector<std::vector<double>> partition_pred_scores_;
  std::vector<std::vector<score_t>> partition_gradients_;
  std::vector<std::vector<score_t>> partition_hessians_;
  const int num_data_;
  int num_classes_;
};

}  // namespace LightGBM
#endif  // LightGBM_CTR_PROVIDER_H_
