/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TRAIN_SHARE_STATES_H_
#define LIGHTGBM_TRAIN_SHARE_STATES_H_

#include <LightGBM/bin.h>
#include <LightGBM/feature_group.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/threading.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace LightGBM {

template <typename HIST_BUF_T>
using AlignedVector = std::vector<HIST_BUF_T, Common::AlignmentAllocator<HIST_BUF_T, kAlignedSize>>;

class MultiValBinWrapperBase {
 public:
  MultiValBinWrapperBase(const std::vector<int> feature_groups_contained):
    feature_groups_contained_(feature_groups_contained) {
      is_distributed_ = false;
    }

  bool IsSparse() {
    if (multi_val_bin_ != nullptr) {
      return multi_val_bin_->IsSparse();
    }
    return false;
  }

  void InitTrain(const std::vector<int>& group_feature_start,
    const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
    const std::vector<int8_t>& is_feature_used,
    const data_size_t* bagging_use_indices,
    data_size_t bagging_indices_cnt);

  void SetUseSubrow(bool is_use_subrow) {
    is_use_subrow_ = is_use_subrow;
  }

  void SetSubrowCopied(bool is_subrow_copied) {
    is_subrow_copied_ = is_subrow_copied;
  }

  void SetGradScale(double g_scale, double h_scale) {
    grad_scale_ = g_scale;
    hess_scale_ = h_scale;
  }

  int GetNumBin() const {
    return num_bin_;
  }

  void SetIsDistributed(const bool is_distributed) {
    is_distributed_ = is_distributed;
  }

  virtual void ConstructHistograms(const data_size_t* data_indices,
    data_size_t num_data,
    const score_t* gradients,
    const score_t* hessians,
    const int_score_t* int_gradients,
    const int_score_t* int_hessians,
    AlignedVector<hist_t>* hist_buf,
    AlignedVector<int_hist_t>* int_hist_buf,
    hist_t* out_hist_data,
    const bool use_indices,
    const bool ordered) = 0;

 protected:

  void CopyMultiValBinSubset(const std::vector<int>& group_feature_start,
    const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
    const std::vector<int8_t>& is_feature_used,
    const data_size_t* bagging_use_indices,
    data_size_t bagging_indices_cnt);

  bool is_use_subcol_ = false;
  bool is_use_subrow_ = false;
  bool is_subrow_copied_ = false;
  std::unique_ptr<MultiValBin> multi_val_bin_;
  std::unique_ptr<MultiValBin> multi_val_bin_subset_;
  std::vector<uint32_t> hist_move_src_;
  std::vector<uint32_t> hist_move_dest_;
  std::vector<uint32_t> hist_move_size_;
  const std::vector<int> feature_groups_contained_;

  int num_threads_;
  int num_bin_;
  int num_bin_aligned_;
  int n_data_block_;
  int data_block_size_;
  int min_block_size_;
  int num_data_;

  double grad_scale_;
  double hess_scale_;
  bool is_distributed_;
};

template <typename SCORE_T, typename HIST_BUF_T>
class MultiValBinWrapper : public MultiValBinWrapperBase {
 public:
  MultiValBinWrapper(MultiValBin* bin, data_size_t num_data,
    const std::vector<int>& feature_groups_contained);

  void ConstructHistograms(const data_size_t* data_indices,
    data_size_t num_data,
    const score_t* gradients,
    const score_t* hessians,
    const int_score_t* int_gradients,
    const int_score_t* int_hessians,
    AlignedVector<hist_t>* hist_buf,
    AlignedVector<int_hist_t>* int_hist_buf,
    hist_t* out_hist_data,
    const bool use_indices,
    const bool ordered) override;

  template <bool IS_INT_GRAD>
  void ConstructHistogramsInner(const data_size_t* data_indices,
    data_size_t num_data,
    const SCORE_T* gradients,
    const SCORE_T* hessians,
    const bool use_indices,
    const bool ordered,
    AlignedVector<HIST_BUF_T>* hist_buf,
    hist_t* out_hist_data);

  template <bool USE_INDICES, bool ORDERED, bool IS_INT_GRAD>
  void ConstructHistogramsForBlocks(const data_size_t* data_indices,
      data_size_t num_data,
      const SCORE_T* gradients,
      const SCORE_T* hessians,
      AlignedVector<HIST_BUF_T>* hist_buf,
      hist_t* out_hist_data);

  template <bool USE_INDICES, bool ORDERED, bool IS_INT_GRAD>
  void ConstructHistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const SCORE_T* gradients, const SCORE_T* hessians, int block_id,
    AlignedVector<HIST_BUF_T>* hist_buf,
    hist_t* out_hist_data);

  template <bool USE_INDICES, bool ORDERED, bool IS_INT_GRAD>
  void ConstructHistogramsForBlockInner(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const SCORE_T* gradients, const SCORE_T* hessians,
    HIST_BUF_T* out_data) {
    std::memset(reinterpret_cast<void*>(out_data), 0, num_bin_ * 2 * sizeof(HIST_BUF_T));
    if (IS_INT_GRAD) {
      const int_score_t* gradients_and_hessians_ptr = reinterpret_cast<const int_score_t*>(gradients);
      int_hist_t* out_data_ptr = reinterpret_cast<int_hist_t*>(out_data); 
      if (USE_INDICES) {
        if (ORDERED) {
          sub_multi_val_bin->ConstructIntHistogramOrdered(data_indices, start, end,
                                                  gradients_and_hessians_ptr, out_data_ptr);
        } else {
          sub_multi_val_bin->ConstructIntHistogram(data_indices, start, end, gradients_and_hessians_ptr,
                                            out_data_ptr);
        }
      } else {
        sub_multi_val_bin->ConstructIntHistogram(start, end, gradients_and_hessians_ptr,
                                          out_data_ptr);
      }
    } else {
      const score_t* gradients_ptr = reinterpret_cast<const score_t*>(gradients);
      const score_t* hessians_ptr = reinterpret_cast<const score_t*>(hessians);
      hist_t* out_data_ptr = reinterpret_cast<hist_t*>(out_data); 
      if (USE_INDICES) {
        if (ORDERED) {
          sub_multi_val_bin->ConstructHistogramOrdered(data_indices, start, end,
                                                  gradients_ptr, hessians_ptr, out_data_ptr);
        } else {
          sub_multi_val_bin->ConstructHistogram(data_indices, start, end, gradients_ptr,
                                            hessians_ptr, out_data_ptr);
        }
      } else {
        sub_multi_val_bin->ConstructHistogram(start, end, gradients_ptr, hessians_ptr,
                                          out_data_ptr);
      }
    }
  }

 private:
  void HistMove(const AlignedVector<HIST_BUF_T>& hist_buf, hist_t* out_hist_data);

  void HistMerge(AlignedVector<HIST_BUF_T>* hist_buf, hist_t* out_hist_data);

  void HistMergeInner(AlignedVector<HIST_BUF_T>* hist_buf, HIST_BUF_T* dst, const int offset);

  void ResizeHistBuf(AlignedVector<HIST_BUF_T>* hist_buf, MultiValBin* sub_multi_val_bin);
};

struct TrainingShareStates {
  int num_threads = 0;
  bool is_col_wise = true;
  bool is_constant_hessian = true;
  const data_size_t* bagging_use_indices;
  data_size_t bagging_indices_cnt;
  bool is_int_gradient;

  TrainingShareStates() {
    multi_val_bin_wrapper_.reset(nullptr);
  }

  uint64_t num_hist_total_bin() { return num_hist_total_bin_; }

  const std::vector<uint32_t>& feature_hist_offsets() { return feature_hist_offsets_; }

  bool IsSparseRowwise() {
    return (multi_val_bin_wrapper_ != nullptr && multi_val_bin_wrapper_->IsSparse());
  }

  void SetMultiValBin(MultiValBin* bin, data_size_t num_data,
    const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
    bool dense_only, bool sparse_only, bool use_gradient_discretization);

  void CalcBinOffsets(const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
    std::vector<uint32_t>* offsets, bool is_col_wise);

  void InitTrain(const std::vector<int>& group_feature_start,
        const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
        const std::vector<int8_t>& is_feature_used) {
    if (multi_val_bin_wrapper_ != nullptr) {
      multi_val_bin_wrapper_->InitTrain(group_feature_start,
        feature_groups,
        is_feature_used,
        bagging_use_indices,
        bagging_indices_cnt);
    }
    if (is_col_wise) {
      int offset = 0;
      group_bin_boundaries_.clear();
      for (int group = 0; group < static_cast<int>(feature_groups.size()); ++group) {
        const auto& fg = feature_groups[group];
        group_bin_boundaries_.push_back(offset);
        offset += fg->num_total_bin_;
      }
      group_bin_boundaries_.push_back(offset);
      int_hist_buf_for_col_wise_.resize(2 * group_bin_boundaries_.back());
    }
  }

#define MultiValBinWraperConstructHistograms_ARGS data_indices, \
  num_data, gradients_ptr, hessians_ptr, int_gradients_ptr, int_hessians_ptr, \
  hist_buf_ptr, int_hist_buf_ptr, out_hist_data

  template <bool USE_INDICES, bool ORDERED>
  void ConstructHistogramsInner(const data_size_t* data_indices,
    data_size_t num_data,
    const score_t* gradients_ptr,
    const score_t* hessians_ptr,
    const int_score_t* int_gradients_ptr,
    const int_score_t* int_hessians_ptr,
    AlignedVector<hist_t>* hist_buf_ptr,
    AlignedVector<int_hist_t>* int_hist_buf_ptr,
    hist_t* out_hist_data) {
    if (USE_INDICES) {
      if (ORDERED) {
        multi_val_bin_wrapper_->ConstructHistograms(MultiValBinWraperConstructHistograms_ARGS, true, true);
      } else {
        multi_val_bin_wrapper_->ConstructHistograms(MultiValBinWraperConstructHistograms_ARGS, true, false);
      }
    } else {
      if (ORDERED) {
        multi_val_bin_wrapper_->ConstructHistograms(MultiValBinWraperConstructHistograms_ARGS, false, true);
      } else {
        multi_val_bin_wrapper_->ConstructHistograms(MultiValBinWraperConstructHistograms_ARGS, false, false);
      }
    }
  }

  template <bool USE_INDICES, bool ORDERED, typename SCORE_T, bool IS_INT_GRAD>
  void ConstructHistograms(const data_size_t* data_indices,
                          data_size_t num_data,
                          const SCORE_T* gradients,
                          const SCORE_T* hessians,
                          hist_t* out_hist_data) {
    if (multi_val_bin_wrapper_ != nullptr) {
      if (IS_INT_GRAD) {
        const score_t* gradients_ptr = nullptr;
        const score_t* hessians_ptr = nullptr;
        const int_score_t* int_gradients_ptr = reinterpret_cast<const int_score_t*>(gradients);
        const int_score_t* int_hessians_ptr = nullptr;
        AlignedVector<hist_t>* hist_buf_ptr = nullptr;
        AlignedVector<int_hist_t>* int_hist_buf_ptr = &int_hist_buf_;
        ConstructHistogramsInner<USE_INDICES, ORDERED>(MultiValBinWraperConstructHistograms_ARGS);
      } else {
        const score_t* gradients_ptr = reinterpret_cast<const score_t*>(gradients);
        const score_t* hessians_ptr = reinterpret_cast<const score_t*>(hessians);
        const int_score_t* int_gradients_ptr = nullptr;
        const int_score_t* int_hessians_ptr = nullptr;
        AlignedVector<hist_t>* hist_buf_ptr = &hist_buf_;
        AlignedVector<int_hist_t>* int_hist_buf_ptr = nullptr;
        ConstructHistogramsInner<USE_INDICES, ORDERED>(MultiValBinWraperConstructHistograms_ARGS);
      }
    }
  }

#undef MultiValBinWraperConstructHistograms_ARGS

  void SetUseSubrow(bool is_use_subrow) {
    if (multi_val_bin_wrapper_ != nullptr) {
      multi_val_bin_wrapper_->SetUseSubrow(is_use_subrow);
    }
  }

  void SetSubrowCopied(bool is_subrow_copied) {
    if (multi_val_bin_wrapper_ != nullptr) {
      multi_val_bin_wrapper_->SetSubrowCopied(is_subrow_copied);
    }
  }

  void RecoverHistogramsFromInteger(hist_t* hist);

  void RecoverHistogramsFromIntegerDistributed(hist_t* hist);

  int_hist_t* GetIntegerHistogram(int group_id);

  void SetGradScale(double grad_scale, double hess_scale) {
    if (multi_val_bin_wrapper_ != nullptr) {
      multi_val_bin_wrapper_->SetGradScale(grad_scale, hess_scale);
    }
    grad_scale_ = grad_scale;
    hess_scale_ = hess_scale;
  }

  double grad_scale() const { return grad_scale_; }

  double hess_scale() const { return hess_scale_; }

  void SetIsDistributed() {
    multi_val_bin_wrapper_->SetIsDistributed(true);
    is_distributed_ = true;
  }

 private:
  std::vector<uint32_t> feature_hist_offsets_;
  uint64_t num_hist_total_bin_ = 0;
  std::unique_ptr<MultiValBinWrapperBase> multi_val_bin_wrapper_;
  AlignedVector<hist_t> hist_buf_;
  AlignedVector<int_hist_t> int_hist_buf_;
  AlignedVector<int_hist_t> int_hist_buf_for_col_wise_;
  int num_total_bin_ = 0;
  double num_elements_per_row_ = 0.0f;
  std::vector<int> group_bin_boundaries_;
  double grad_scale_;
  double hess_scale_;
  bool is_distributed_ = false;

  data_size_t total_num_data_;
};

}  // namespace LightGBM

#endif   // LightGBM_TRAIN_SHARE_STATES_H_
