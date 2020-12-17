/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TRAIN_SHARE_STATES_H_
#define LIGHTGBM_TRAIN_SHARE_STATES_H_

#include <LightGBM/bin.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/threading.h>
#include <LightGBM/feature_group.h>

#include <memory>
#include <vector>
#include <algorithm>

namespace LightGBM {

class MultiValBinWrapper {
 public:
  MultiValBinWrapper(MultiValBin* bin, data_size_t num_data,
    const std::vector<int>& feature_groups_contained);

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

  void HistMove(const std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>& hist_buf);

  void IntHistMove(const std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>& hist_buf);

  void HistMerge(std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf);

  void IntHistMerge(std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* hist_buf);

  void Int32HistMerge(std::vector<int_buf_hist_t, Common::AlignmentAllocator<int_buf_hist_t, kAlignedSize>>* hist_buf,
    std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* merge_hist_buf);

  void Int48HistMerge(std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* hist_buf,
    std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* merge_hist_buf);

  void ResizeHistBuf(std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf,
    MultiValBin* sub_multi_val_bin,
    hist_t* origin_hist_data);

  void ResizeIntHistBuf(std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* hist_buf,
    MultiValBin* sub_multi_val_bin,
    hist_t* origin_hist_data);

  void Resize32IntHistBuf(std::vector<int_buf_hist_t, Common::AlignmentAllocator<int_buf_hist_t, kAlignedSize>>* hist_buf,
    std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* merged_hist_buf,
    MultiValBin* sub_multi_val_bin,
    hist_t* origin_hist_data);

  void Resize48IntHistBuf(std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* hist_buf,
    std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* merged_hist_buf,
    MultiValBin* sub_multi_val_bin,
    hist_t* origin_hist_data);

  template <bool USE_INDICES, bool ORDERED, typename SCORE_T, typename HIST_BUF_T, typename HIST_T,  bool IS_INT_GRAD>
  void ConstructHistograms(const data_size_t* data_indices,
      data_size_t num_data,
      const SCORE_T* gradients,
      const SCORE_T* hessians,
      std::vector<HIST_BUF_T, Common::AlignmentAllocator<HIST_BUF_T, kAlignedSize>>* hist_buf,
      std::vector<HIST_T, Common::AlignmentAllocator<HIST_T, kAlignedSize>>* merged_hist_buf,
      hist_t* origin_hist_data) {
    const auto cur_multi_val_bin = (is_use_subcol_ || is_use_subrow_)
          ? multi_val_bin_subset_.get()
          : multi_val_bin_.get();
    if (cur_multi_val_bin != nullptr) {
      global_timer.Start("Dataset::sparse_bin_histogram");
      n_data_block_ = 1;
      data_block_size_ = num_data;
      Threading::BlockInfo<data_size_t>(num_threads_, num_data, min_block_size_,
                                        max_block_size_, &n_data_block_, &data_block_size_);
      if (IS_INT_GRAD) {
        Resize48IntHistBuf(
          reinterpret_cast<std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>*>(hist_buf),
          reinterpret_cast<std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>*>(merged_hist_buf),
          cur_multi_val_bin, origin_hist_data);
      } else {
        ResizeHistBuf(
          reinterpret_cast<std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>*>(hist_buf),
          cur_multi_val_bin, origin_hist_data);
      }
      OMP_INIT_EX();
      #pragma omp parallel for schedule(static) num_threads(num_threads_)
      for (int block_id = 0; block_id < n_data_block_; ++block_id) {
        OMP_LOOP_EX_BEGIN();
        data_size_t start = block_id * data_block_size_;
        data_size_t end = std::min<data_size_t>(start + data_block_size_, num_data);
        if (IS_INT_GRAD) {
          ConstructInt48HistogramsForBlock<USE_INDICES, ORDERED>(
            cur_multi_val_bin, start, end, data_indices,
            reinterpret_cast<const int_score_t*>(gradients),
            reinterpret_cast<const int_score_t*>(hessians),
            block_id,
            reinterpret_cast<std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>*>(hist_buf));
        } else {
          ConstructHistogramsForBlock<USE_INDICES, ORDERED>(
            cur_multi_val_bin, start, end, data_indices,
            reinterpret_cast<const score_t*>(gradients),
            reinterpret_cast<const score_t*>(hessians),
            block_id,
            reinterpret_cast<std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>*>(hist_buf));
        }
        OMP_LOOP_EX_END();
      }
      OMP_THROW_EX();
      global_timer.Stop("Dataset::sparse_bin_histogram");

      global_timer.Start("Dataset::sparse_bin_histogram_merge");
      if (IS_INT_GRAD) {
        Int48HistMerge(
          reinterpret_cast<std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>*>(hist_buf),
          reinterpret_cast<std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>*>(merged_hist_buf));
      } else {
        HistMerge(
          reinterpret_cast<std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>*>(hist_buf));
      }
      global_timer.Stop("Dataset::sparse_bin_histogram_merge");
      global_timer.Start("Dataset::sparse_bin_histogram_move");
      if (IS_INT_GRAD) {
        IntHistMove(
          *reinterpret_cast<std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>*>(merged_hist_buf));
      } else {
        HistMove(
          *reinterpret_cast<std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>*>(hist_buf));
      }
      global_timer.Stop("Dataset::sparse_bin_histogram_move");
    }
  }

  template <bool USE_INDICES, bool ORDERED>
  void ConstructHistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const score_t* gradients, const score_t* hessians, int block_id,
    std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf) {
    hist_t* data_ptr = origin_hist_data_;
    if (block_id == 0) {
      if (is_use_subcol_) {
        data_ptr = hist_buf->data() + hist_buf->size() - 2 * static_cast<size_t>(num_bin_aligned_);
      }
    } else {
      data_ptr = hist_buf->data() +
        static_cast<size_t>(num_bin_aligned_) * (block_id - 1) * 2;
    }
    std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin_ * kHistBufferEntrySize);
    if (USE_INDICES) {
      if (ORDERED) {
        sub_multi_val_bin->ConstructHistogramOrdered(data_indices, start, end,
                                                gradients, hessians, data_ptr);
      } else {
        sub_multi_val_bin->ConstructHistogram(data_indices, start, end, gradients,
                                          hessians, data_ptr);
      }
    } else {
      sub_multi_val_bin->ConstructHistogram(start, end, gradients, hessians,
                                        data_ptr);
    }
  }

  template <bool USE_INDICES, bool ORDERED>
  void ConstructIntHistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const int_score_t* gradients, const int_score_t* hessians, int block_id,
    std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* hist_buf) {
    int_hist_t* data_ptr = hist_buf->data() + static_cast<size_t>(num_bin_aligned_) * block_id * 2;
    std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin_ * kHistIntBufferEntrySize);
    if (USE_INDICES) {
      if (ORDERED) {
        sub_multi_val_bin->ConstructIntHistogramOrdered(data_indices, start, end,
                                                gradients, hessians, data_ptr);
      } else {
        sub_multi_val_bin->ConstructIntHistogram(data_indices, start, end, gradients,
                                          hessians, data_ptr);
      }
    } else {
      sub_multi_val_bin->ConstructIntHistogram(start, end, gradients, hessians,
                                        data_ptr);
    }
  }

  template <bool USE_INDICES, bool ORDERED>
  void ConstructInt32HistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const int_score_t* gradients, const int_score_t* hessians, int block_id,
    std::vector<int_buf_hist_t, Common::AlignmentAllocator<int_buf_hist_t, kAlignedSize>>* hist_buf) {
    int_buf_hist_t* data_ptr = hist_buf->data() + static_cast<size_t>(num_bin_aligned_) * block_id * 2;
    std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin_ * kHist32IntBufferEntrySize);
    if (USE_INDICES) {
      if (ORDERED) {
        sub_multi_val_bin->ConstructInt32HistogramOrdered(data_indices, start, end,
                                                gradients, hessians, data_ptr);
      } else {
        sub_multi_val_bin->ConstructInt32Histogram(data_indices, start, end, gradients,
                                          hessians, data_ptr);
      }
    } else {
      sub_multi_val_bin->ConstructInt32Histogram(start, end, gradients, hessians,
                                        data_ptr);
    }
  }

  template <bool USE_INDICES, bool ORDERED>
  void ConstructInt48HistogramsForBlock(const MultiValBin* sub_multi_val_bin,
    data_size_t start, data_size_t end, const data_size_t* data_indices,
    const int_score_t* gradients, const int_score_t* hessians, int block_id,
    std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* hist_buf) {
    int_hist_t* data_ptr = reinterpret_cast<int_hist_t*>( 
      (reinterpret_cast<int16_t*>(hist_buf->data()) + static_cast<size_t>(num_bin_aligned_) * block_id * 3));
    std::memset(reinterpret_cast<void*>(data_ptr), 0, num_bin_ * kHist48IntBufferEntrySize);
    if (USE_INDICES) {
      if (ORDERED) {
        sub_multi_val_bin->ConstructInt48HistogramOrdered(data_indices, start, end,
                                                gradients, hessians, data_ptr);
      } else {
        sub_multi_val_bin->ConstructInt48Histogram(data_indices, start, end, gradients,
                                          hessians, data_ptr);
      }
    } else {
      sub_multi_val_bin->ConstructInt48Histogram(start, end, gradients, hessians,
                                        data_ptr);
    }
  }

  void CopyMultiValBinSubset(const std::vector<int>& group_feature_start,
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
    grad_scale = g_scale;
    hess_scale = h_scale;
  }

 private:
  bool is_use_subcol_ = false;
  bool is_use_subrow_ = false;
  bool is_subrow_copied_ = false;
  std::unique_ptr<MultiValBin> multi_val_bin_;
  std::unique_ptr<MultiValBin> multi_val_bin_subset_;
  MultiValBin* cur_multi_val_bin_;
  std::vector<uint32_t> hist_move_src_;
  std::vector<uint32_t> hist_move_dest_;
  std::vector<uint32_t> hist_move_size_;
  const std::vector<int> feature_groups_contained_;

  int num_threads_;
  int max_block_size_;
  int num_bin_;
  int num_bin_aligned_;
  int n_data_block_;
  int data_block_size_;
  int min_block_size_;
  int num_data_;

  hist_t* origin_hist_data_;

  const size_t kHistBufferEntrySize = 2 * sizeof(hist_t);
  const size_t kHistIntBufferEntrySize = 2 * sizeof(int_hist_t);
  const size_t kHist32IntBufferEntrySize = 2 * sizeof(int_buf_hist_t);
  const size_t kHist48IntBufferEntrySize = 3 * sizeof(int_buf_hist_t);

  double grad_scale, hess_scale;
};

struct TrainingShareStates {
  int num_threads = 0;
  bool is_col_wise = true;
  bool is_constant_hessian = true;
  const data_size_t* bagging_use_indices;
  data_size_t bagging_indices_cnt;

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
    bool dense_only, bool sparse_only);

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
      int_hist_buf_.resize(2 * group_bin_boundaries_.back());
    }
  }

  template <bool USE_INDICES, bool ORDERED, typename SCORE_T, bool IS_INT_GRAD>
  void ConstructHistograms(const data_size_t* data_indices,
                          data_size_t num_data,
                          const SCORE_T* gradients,
                          const SCORE_T* hessians,
                          hist_t* hist_data) {
    if (multi_val_bin_wrapper_ != nullptr) {
      if (IS_INT_GRAD) {
        multi_val_bin_wrapper_->ConstructHistograms<
          USE_INDICES, ORDERED, SCORE_T, int_hist_t, int_hist_t, IS_INT_GRAD>(
          data_indices, num_data, gradients, hessians, &int_48_hist_buf_, &merged_int_hist_buf_, hist_data);
      } else {
        multi_val_bin_wrapper_->ConstructHistograms<
          USE_INDICES, ORDERED, SCORE_T, hist_t, hist_t, IS_INT_GRAD>(
          data_indices, num_data, gradients, hessians, &hist_buf_, &hist_buf_, hist_data);
      }
    }
  }

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

 private:
  std::vector<uint32_t> feature_hist_offsets_;
  uint64_t num_hist_total_bin_ = 0;
  std::unique_ptr<MultiValBinWrapper> multi_val_bin_wrapper_;
  std::vector<hist_t, Common::AlignmentAllocator<hist_t, kAlignedSize>> hist_buf_;
  std::vector<int_buf_hist_t, Common::AlignmentAllocator<int_buf_hist_t, kAlignedSize>> int_hist_buf_;
  std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>> int_48_hist_buf_;
  std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>> merged_int_hist_buf_;
  int num_total_bin_ = 0;
  double num_elements_per_row_ = 0.0f;
  std::vector<int> group_bin_boundaries_;
  double grad_scale_;
  double hess_scale_;
};

}  // namespace LightGBM

#endif   // LightGBM_TRAIN_SHARE_STATES_H_
