/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_IO_MULTI_VAL_MIX_BIN_HPP_
#define LIGHTGBM_IO_MULTI_VAL_MIX_BIN_HPP_

#include <LightGBM/bin.h>
#include <LightGBM/utils/openmp_wrapper.h>
#include <LightGBM/utils/threading.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace LightGBM {

template <typename VAL_T>
class MultiValMixBin : public MultiValBin {
 public:
  explicit MultiValMixBin(data_size_t num_data, int num_bin, int num_feature,
    const std::vector<uint32_t>& offsets, const int num_single_val_groups,
    const double estimate_element_per_row)
    : num_data_(num_data), num_bin_(num_bin), num_feature_(num_feature),
      offsets_(offsets), estimate_element_per_row_(estimate_element_per_row) {
    bit_states_.clear();
    num_bit_state_groups_.clear();
    BIT_STATE cur_bit_state = BIT_STATE_4;
    int num_groups_in_cur_bit_state = 0;
    for (int group_index = 0; group_index < num_single_val_groups; ++group_index) {
      const int num_bin_in_group = offsets_[group_index + 1] - offsets_[group_index];
      if (num_bin_in_group <= 16) {
        ++num_groups_in_cur_bit_state;
      } else if (num_bin_in_group <= 256) {
        if (cur_bit_state == BIT_STATE_4) {
          if (num_groups_in_cur_bit_state >= num_single_val_groups / 3) {
            bit_states_.push_back(BIT_STATE_4);
            num_bit_state_groups_.push_back(num_groups_in_cur_bit_state);
            num_groups_in_cur_bit_state = 0;
            has_bit_state_4_ = true;
          }
        }
        cur_bit_state = BIT_STATE_8;
        ++num_groups_in_cur_bit_state;
      } else if (num_bin_in_group <= 65536) {
        if (cur_bit_state == BIT_STATE_8) {
          if (num_groups_in_cur_bit_state >= num_single_val_groups / 3) {
            bit_states_.push_back(BIT_STATE_8);
            num_bit_state_groups_.push_back(num_groups_in_cur_bit_state);
            num_groups_in_cur_bit_state = 0;
            has_bit_state_8_ = true;
          }
        }
        cur_bit_state = BIT_STATE_16;
        ++num_groups_in_cur_bit_state;
      } else {
        if (cur_bit_state == BIT_STATE_16) {
          if (num_groups_in_cur_bit_state >= num_single_val_groups / 3) {
            bit_states_.push_back(BIT_STATE_16);
            num_bit_state_groups_.push_back(num_groups_in_cur_bit_state);
            num_groups_in_cur_bit_state = 0;
            has_bit_state_16_ = true;
          }
        }
        cur_bit_state = BIT_STATE_32;
        ++num_groups_in_cur_bit_state;
      }
    }
    if (bit_states_.empty() || bit_states_.back() != cur_bit_state) {
      bit_states_.push_back(cur_bit_state);
      num_bit_state_groups_.push_back(num_groups_in_cur_bit_state);
      if (cur_bit_state == BIT_STATE_4) {
        has_bit_state_4_ = true;
      } else if (cur_bit_state == BIT_STATE_8) {
        has_bit_state_8_ = true;
      } else if (cur_bit_state == BIT_STATE_16) {
        has_bit_state_16_ = true;
      } else if (cur_bit_state == BIT_STATE_32) {
        has_bit_state_32_ = true;
      }
    }
    if (num_single_val_groups < num_feature_) {
      // has multi val bin
      const int num_multi_val_group_bin = offsets.back() - offsets[num_single_val_groups];
      if (num_multi_val_group_bin <= 16) {
        bit_states_.push_back(BIT_STATE_4);
        mvg_bit_state_4_ = true;
      } else if (num_multi_val_group_bin <= 256) {
        bit_states_.push_back(BIT_STATE_8);
        mvg_bit_state_8_ = true;
      } else if (num_multi_val_group_bin <= 65536) {
        bit_states_.push_back(BIT_STATE_16);
        mvg_bit_state_16_ = true;
      } else {
        bit_states_.push_back(BIT_STATE_32);
        mvg_bit_state_32_ = true;
      }
    }
    CalcBitBoundaries();
    row_ptr_.resize(num_data_ + 1, 0);
    //sparse_row_ptr_.resize(num_data_ + 1, 0);
    const int num_threads = OMP_NUM_THREADS();
    t_size_.resize(num_threads, 0);
    uint32_t estimate_num_data = static_cast<uint32_t>(estimate_element_per_row_ * 1.1 * num_data_);
    if (num_threads > 1) {
      t_data_.resize(num_threads - 1);
      #pragma omp parallel for schedule(static) num_threads(num_threads - 1)
      for (int i = 0; i < num_threads - 1; ++i) {
        t_data_[i].resize(estimate_num_data / num_threads);
      }
    }
    data_.resize(estimate_num_data / num_threads);
    thread_compressed_values_.resize(num_threads);
    mvg_offset_ = offsets_[num_single_val_groups];
    num_dense_feature_groups_ = num_single_val_groups;
  }

  ~MultiValMixBin() {
  }

  data_size_t num_data() const override {
    return num_data_;
  }

  int num_bin() const override {
    return num_bin_;
  }

  double num_element_per_row() const override { return num_feature_; }

  const std::vector<uint32_t>& offsets() const override { return offsets_; }

  void CompressOneRow(const std::vector<uint32_t>& values, std::vector<VAL_T>* compressed_values);

  void PushOneRow(int tid, data_size_t idx, const std::vector<uint32_t>& values) override {
    const int pre_alloc_size = 50;
    auto& compressed_values = thread_compressed_values_[tid];
    CompressOneRow(values, &compressed_values);
    row_ptr_[idx + 1] = static_cast<uint32_t>(compressed_values.size());
    if (tid == 0) {
      if (t_size_[tid] + row_ptr_[idx + 1] >
          static_cast<uint32_t>(data_.size())) {
        data_.resize(t_size_[tid] + row_ptr_[idx + 1] * pre_alloc_size);
      }
      for (auto val : compressed_values) {
        data_[t_size_[tid]++] = static_cast<VAL_T>(val);
      }
    } else {
      if (t_size_[tid] + row_ptr_[idx + 1] >
          static_cast<uint32_t>(t_data_[tid - 1].size())) {
        t_data_[tid - 1].resize(t_size_[tid] +
                                row_ptr_[idx + 1] * pre_alloc_size);
      }
      for (auto val : compressed_values) {
        t_data_[tid - 1][t_size_[tid]++] = static_cast<VAL_T>(val);
      }
    }
  }

  void MergeData(const uint32_t* sizes) {
    Common::FunctionTimer fun_time("MultiValMixBin::MergeData", global_timer);
    for (data_size_t i = 0; i < num_data_; ++i) {
      row_ptr_[i + 1] += row_ptr_[i];
    }
    if (t_data_.size() > 0) {
      std::vector<uint32_t> offsets(1 + t_data_.size());
      offsets[0] = sizes[0];
      for (size_t tid = 0; tid < t_data_.size() - 1; ++tid) {
        offsets[tid + 1] = offsets[tid] + sizes[tid + 1];
      }
      data_.resize(row_ptr_[num_data_]);
#pragma omp parallel for schedule(static, 1)
      for (int tid = 0; tid < static_cast<int>(t_data_.size()); ++tid) {
        std::copy_n(t_data_[tid].data(), sizes[tid + 1],
                    data_.data() + offsets[tid]);
      }
    } else {
      data_.resize(row_ptr_[num_data_]);
    }
  }

  void FinishLoad() override {
    MergeData(t_size_.data());
    t_size_.clear();
    row_ptr_.shrink_to_fit();
    data_.shrink_to_fit();
    t_data_.clear();
    t_data_.shrink_to_fit();
    // update estimate_element_per_row_ by all data
    estimate_element_per_row_ =
        static_cast<double>(row_ptr_[num_data_]) / num_data_;
    for (size_t thread_index = 0; thread_index < thread_compressed_values_.size(); ++thread_index) {
      thread_compressed_values_[thread_index].reserve(estimate_element_per_row_);
    }
  }

  bool IsSparse() override {
    return false;
  }

  bool IsMix() override {
    return true;
  }

  template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
    bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32,
    bool MVG_BIT_STATE_4, bool MVG_BIT_STATE_8, bool MVG_BIT_STATE_16, bool MVG_BIT_STATE_32>
  inline void ConstructHistogramInnerMost(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const;

  template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T>
  inline void ConstructHistogramInner(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const;

  template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
    bool HAS_BIT_STATE_4>
  void ConstructHistogramInnerUnroll1(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const;

  template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
    bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8>
  inline void ConstructHistogramInnerUnroll2(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const;

  template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
    bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16>
  inline void ConstructHistogramInnerUnroll3(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const;

  template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
    bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32>
  inline void ConstructHistogramInnerUnroll4(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const;

  template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
    bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32,
    bool MVG_BIT_STATE_4>
  inline void ConstructHistogramInnerUnroll5(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const;

  template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
    bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32,
    bool MVG_BIT_STATE_4, bool MVG_BIT_STATE_8>
  inline void ConstructHistogramInnerUnroll6(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const;

  template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
    bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32,
    bool MVG_BIT_STATE_4, bool MVG_BIT_STATE_8, bool MVG_BIT_STATE_16>
  inline void ConstructHistogramInnerUnroll7(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const;

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const score_t* gradients,
                          const score_t* hessians, hist_t* out) const override {
    ConstructHistogramInner<true, true, false, hist_t, score_t>(data_indices, start, end,
                                               gradients, hessians, out);
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
                          const score_t* gradients, const score_t* hessians,
                          hist_t* out) const override {
    ConstructHistogramInner<false, false, false, hist_t, score_t>(
        nullptr, start, end, gradients, hessians, out);
  }

  void ConstructHistogramOrdered(const data_size_t* data_indices,
                                 data_size_t start, data_size_t end,
                                 const score_t* gradients,
                                 const score_t* hessians,
                                 hist_t* out) const override {
    ConstructHistogramInner<true, true, true, hist_t, score_t>(data_indices, start, end,
                                              gradients, hessians, out);
  }

  void ConstructIntHistogram(const data_size_t* data_indices, data_size_t start,
                          data_size_t end, const int_score_t* gradients,
                          const int_score_t* hessians, int_hist_t* out) const override {
    ConstructHistogramInner<true, true, false, int_hist_t, int_score_t>(data_indices, start, end,
                                               gradients, hessians, out);
  }

  void ConstructIntHistogram(data_size_t start, data_size_t end,
                          const int_score_t* gradients, const int_score_t* hessians,
                          int_hist_t* out) const override {
    ConstructHistogramInner<false, false, false, int_hist_t, int_score_t>(
        nullptr, start, end, gradients, hessians, out);
  }

  void ConstructIntHistogramOrdered(const data_size_t* data_indices,
                                 data_size_t start, data_size_t end,
                                 const int_score_t* gradients,
                                 const int_score_t* hessians,
                                 int_hist_t* out) const override {
    ConstructHistogramInner<true, true, true, int_hist_t, int_score_t>(data_indices, start, end,
                                              gradients, hessians, out);
  }

  MultiValBin* CreateLike(data_size_t /*num_data*/, int /*num_bin*/, int /*num_feature*/, double /*estimtate_elements_per_row*/,
    const std::vector<uint32_t>& /*offsets*/, const int /*num_dense_feature_groups*/) const override {
    //return new MultiValMixBin<VAL_T>(num_data, num_bin, num_feature, offsets, num_dense_feature_groups, estimtate_elements_per_row);
    Log::Fatal("unsupported");
    return nullptr;
  }

  void ReSize(data_size_t /*num_data*/, int /*num_bin*/, int /*num_feature*/,
              double /*estimate_elements_per_row*/, const std::vector<uint32_t>& /*offsets*/,
              const int /*num_dense_feature_groups*/) override {
    Log::Fatal("unsupported");
  }

  template <bool SUBROW, bool SUBCOL>
  void CopyInner(const MultiValBin* /*full_bin*/, const data_size_t* /*used_indices*/,
                 data_size_t /*num_used_indices*/,
                 const std::vector<int>& /*used_feature_index*/) {
    Log::Fatal("unsupported");
  }


  void CopySubrow(const MultiValBin* full_bin, const data_size_t* used_indices,
                  data_size_t num_used_indices) override {
    CopyInner<true, false>(full_bin, used_indices, num_used_indices,
                           std::vector<int>());
  }

  void CopySubcol(const MultiValBin* full_bin,
                  const std::vector<int>& used_feature_index,
                  const std::vector<uint32_t>&,
                  const std::vector<uint32_t>&,
                  const std::vector<uint32_t>&) override {
    CopyInner<false, true>(full_bin, nullptr, num_data_, used_feature_index);
  }

  void CopySubrowAndSubcol(const MultiValBin* full_bin,
                           const data_size_t* used_indices,
                           data_size_t num_used_indices,
                           const std::vector<int>& used_feature_index,
                           const std::vector<uint32_t>&,
                           const std::vector<uint32_t>&,
                           const std::vector<uint32_t>&) override {
    CopyInner<true, true>(full_bin, used_indices, num_used_indices,
                          used_feature_index);
  }

  inline size_t RowPtr(data_size_t idx) const {
    return row_ptr_[idx];
  }

  MultiValMixBin<VAL_T>* Clone() override {
    Log::Fatal("unsupported");
    return nullptr;
  }

 private:
  enum BIT_STATE {
    BIT_STATE_4,
    BIT_STATE_8,
    BIT_STATE_16,
    BIT_STATE_32
  };
  
  data_size_t num_data_;
  int num_bin_;
  int num_feature_;
  int num_dense_units_per_row_;
  std::vector<uint32_t> offsets_;
  std::vector<VAL_T, Common::AlignmentAllocator<VAL_T, 32>> data_;
  std::vector<BIT_STATE> bit_states_;
  std::vector<int> num_bit_state_groups_;
  std::vector<uint32_t> row_ptr_;
  std::vector<uint16_t> sparse_row_ptr_;
  std::vector<uint32_t> t_size_;
  std::vector<std::vector<VAL_T, Common::AlignmentAllocator<VAL_T, 32>>> t_data_;
  std::vector<std::vector<VAL_T>> thread_compressed_values_;
  double estimate_element_per_row_;

  int bit4_start_ = 0;
  int bit4_end_res_ = 0;
  int bit4_end_ = 0;
  int bit8_start_ = 0;
  int bit8_end_res_ = 0;
  int bit8_end_ = 0;
  int bit16_start_ = 0;
  int bit16_end_res_ = 0;
  int bit16_end_ = 0;
  int bit32_start_ = 0;
  int bit32_end_res_ = 0;
  int bit32_end_ = 0;

  bool has_bit_state_4_ = false;
  bool has_bit_state_8_ = false;
  bool has_bit_state_16_ = false;
  bool has_bit_state_32_ = false;
  bool mvg_bit_state_4_ = false;
  bool mvg_bit_state_8_ = false;
  bool mvg_bit_state_16_ = false;
  bool mvg_bit_state_32_ = false;

  int num_dense_feature_groups_;

  int mvg_offset_ = 0;

  MultiValMixBin<VAL_T>(const MultiValMixBin<VAL_T>& other)
    : num_data_(other.num_data_), num_bin_(other.num_bin_), num_feature_(other.num_feature_),
      offsets_(other.offsets_), data_(other.data_) {
    Log::Fatal("unsupported");
  }

  void CalcBitBoundaries();
};

template <>
template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
    bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32,
    bool MVG_BIT_STATE_4, bool MVG_BIT_STATE_8, bool MVG_BIT_STATE_16, bool MVG_BIT_STATE_32>
void MultiValMixBin<uint8_t>::ConstructHistogramInnerMost(const data_size_t* data_indices, data_size_t start, data_size_t end,
  const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const {
  data_size_t i = start;
  HIST_T* grad = out;
  HIST_T* hess = out + 1;
  const uint8_t* data_ptr_base = data_.data();
  if (USE_PREFETCH) {
    const data_size_t pf_offset = 32 / sizeof(uint8_t);
    const data_size_t pf_end = end - pf_offset;

    for (; i < pf_end; ++i) {
      const auto idx = USE_INDICES ? data_indices[i] : i;
      const auto pf_idx = USE_INDICES ? data_indices[i + pf_offset] : i + pf_offset;
      if (!ORDERED) {
        PREFETCH_T0(gradients + pf_idx);
        PREFETCH_T0(hessians + pf_idx);
      }
      PREFETCH_T0(data_ptr_base + RowPtr(pf_idx));
      const auto j_start = RowPtr(idx);
      const uint8_t* data_ptr = data_ptr_base + j_start;
      const SCORE_T gradient = ORDERED ? gradients[i] : gradients[idx];
      const SCORE_T hessian = ORDERED ? hessians[i] : hessians[idx];
      int j = 0;
      int i = 0;
      if (HAS_BIT_STATE_4) { 
        for (; j < bit4_end_; ++j) {
          const uint8_t packed_bin = data_ptr[j];
          const uint32_t bin0 = static_cast<uint32_t>(packed_bin & 0xf);
          const uint32_t bin1 = static_cast<uint32_t>((packed_bin >> 4) & 0xf);
          const auto ti0 = (bin0 + offsets_[i]) << 1;
          const auto ti1 = (bin1 + offsets_[i + 1]) << 1;
          grad[ti0] += gradient;
          hess[ti0] += hessian;
          grad[ti1] += gradient;
          hess[ti1] += hessian;
          i += 2;
        }
        if (bit4_end_res_ > 0) {
          const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
          const auto ti = (bin + offsets_[i]) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
          ++j;
          ++i;
        }
      }
      if (HAS_BIT_STATE_8) {
        for (; j < bit8_end_; ++j) {
          const uint8_t bin = data_ptr[j];
          const auto ti = (bin + offsets_[i]) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
          ++i;
        }
      }
      if (MVG_BIT_STATE_4) {
        const auto num_values = static_cast<int>(RowPtr(idx + 1) - j_start);
        for (; j < num_values; ++j) {
          const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
          const auto ti = (bin + mvg_offset_) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
      if (MVG_BIT_STATE_8) {
        const auto num_values = static_cast<int>(RowPtr(idx + 1) - j_start);
        for (; j < num_values; ++j) {
          const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
          const auto ti = (bin + mvg_offset_) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
    }
  }
  for (; i < end; ++i) {
    const auto idx = USE_INDICES ? data_indices[i] : i;
    const auto j_start = RowPtr(idx);
    const uint8_t* data_ptr = data_ptr_base + j_start;
    const SCORE_T gradient = ORDERED ? gradients[i] : gradients[idx];
    const SCORE_T hessian = ORDERED ? hessians[i] : hessians[idx];
    int j = 0;
    int i = 0;
    if (HAS_BIT_STATE_4) { 
      for (; j < bit4_end_; ++j) {
        const uint8_t packed_bin = data_ptr[j];
        const uint32_t bin0 = static_cast<uint32_t>(packed_bin & 0xf);
        const uint32_t bin1 = static_cast<uint32_t>((packed_bin >> 4) & 0xf);
        const auto ti0 = (bin0 + offsets_[i]) << 1;
        const auto ti1 = (bin1 + offsets_[i + 1]) << 1;
        grad[ti0] += gradient;
        hess[ti0] += hessian;
        grad[ti1] += gradient;
        hess[ti1] += hessian;
        i += 2;
      }
      if (bit4_end_res_ > 0) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + offsets_[i]) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
        ++j;
        ++i;
      }
    }
    if (HAS_BIT_STATE_8) {
      for (; j < bit8_end_; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + offsets_[i]) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
        ++i;
      }
    }
    if (MVG_BIT_STATE_4) {
      const auto num_values = static_cast<int>(RowPtr(idx + 1) - j_start);
      for (; j < num_values; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + mvg_offset_) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
    if (MVG_BIT_STATE_8) {
      const auto num_values = static_cast<int>(RowPtr(idx + 1) - j_start);
      for (; j < num_values; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + mvg_offset_) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
  }
}

template <>
template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
    bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32,
    bool MVG_BIT_STATE_4, bool MVG_BIT_STATE_8, bool MVG_BIT_STATE_16, bool MVG_BIT_STATE_32>
void MultiValMixBin<uint16_t>::ConstructHistogramInnerMost(const data_size_t* data_indices, data_size_t start, data_size_t end,
  const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const {
  data_size_t i = start;
  HIST_T* grad = out;
  HIST_T* hess = out + 1;
  const uint16_t* data_ptr_base = data_.data();
  if (USE_PREFETCH) {
    const data_size_t pf_offset = 32 / sizeof(uint16_t);
    const data_size_t pf_end = end - pf_offset;

    for (; i < pf_end; ++i) {
      const auto idx = USE_INDICES ? data_indices[i] : i;
      const auto pf_idx = USE_INDICES ? data_indices[i + pf_offset] : i + pf_offset;
      if (!ORDERED) {
        PREFETCH_T0(gradients + pf_idx);
        PREFETCH_T0(hessians + pf_idx);
      }
      PREFETCH_T0(data_ptr_base + RowPtr(pf_idx));
      const auto j_start = RowPtr(idx);
      const uint16_t* data_ptr = data_ptr_base + j_start;
      const SCORE_T gradient = ORDERED ? gradients[i] : gradients[idx];
      const SCORE_T hessian = ORDERED ? hessians[i] : hessians[idx];
      int j = 0;
      int i = 0;
      if (HAS_BIT_STATE_4) { 
        for (; j < bit4_end_; ++j) {
          uint16_t packed_bin = data_ptr[j];
          const uint32_t bin0 = static_cast<uint32_t>(packed_bin & 0xf);
          packed_bin >>= 4;
          const uint32_t bin1 = static_cast<uint32_t>(packed_bin & 0xf);
          packed_bin >>= 4;
          const uint32_t bin2 = static_cast<uint32_t>(packed_bin & 0xf);
          packed_bin >>= 4;
          const uint32_t bin3 = static_cast<uint32_t>(packed_bin & 0xf);
          const auto ti0 = (bin0 + offsets_[i]) << 1;
          const auto ti1 = (bin1 + offsets_[i + 1]) << 1;
          const auto ti2 = (bin2 + offsets_[i + 2]) << 1;
          const auto ti3 = (bin3 + offsets_[i + 3]) << 1;
          //CHECK_LT(ti0, 5459 * 2 - 1);
          //CHECK_LT(ti1, 5459 * 2 - 1);
          //CHECK_LT(ti2, 5459 * 2 - 1);
          //CHECK_LT(ti3, 5459 * 2 - 1);
          grad[ti0] += gradient;
          hess[ti0] += hessian;
          grad[ti1] += gradient;
          hess[ti1] += hessian;
          grad[ti2] += gradient;
          hess[ti2] += hessian;
          grad[ti3] += gradient;
          hess[ti3] += hessian;
          i += 4;
        }
        if (bit4_end_res_ > 0) {
          uint16_t packed_bin = data_ptr[j];
          for (int k = 0; k < bit4_end_res_; ++k) {
            const uint32_t bin = static_cast<uint32_t>(packed_bin & 0xf);
            packed_bin >>= 4;
            const auto ti = (bin + offsets_[i]) << 1;
            //CHECK_LT(ti, 5459 * 2 - 1);
            grad[ti] += gradient;
            hess[ti] += hessian;
            ++i;
          }
          ++j;
        }
      }
      if (HAS_BIT_STATE_8) {
        for (; j < bit8_end_; ++j) {
          uint16_t packed_bin = data_ptr[j];
          const uint32_t bin0 = static_cast<uint32_t>(packed_bin & 0xff);
          const uint32_t bin1 = static_cast<uint32_t>((packed_bin >> 8) & 0xff);
          const auto ti0 = (bin0 + offsets_[i]) << 1;
          const auto ti1 = (bin1 + offsets_[i + 1]) << 1;
          //CHECK_LT(ti0, 5459 * 2 - 1);
          //CHECK_LT(ti1, 5459 * 2 - 1);
          grad[ti0] += gradient;
          hess[ti0] += hessian;
          grad[ti1] += gradient;
          hess[ti1] += hessian;
          i += 2;
        }
        if (bit8_end_res_ > 0) {
          const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
          const auto ti = (bin + offsets_[i]) << 1;
          //CHECK_LT(ti, 5459 * 2 - 1);
          grad[ti] += gradient;
          hess[ti] += hessian;
          ++i;
          ++j;
        }
      }
      if (HAS_BIT_STATE_16) {
        for (; j < bit16_end_; ++j) {
          uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
          const auto ti = (bin + offsets_[i]) << 1;
          //CHECK_LT(ti, 5459 * 2 - 1);
          grad[ti] += gradient;
          hess[ti] += hessian;
          ++i;
        }
      }
      if (MVG_BIT_STATE_4) {
        const auto num_values = static_cast<int>(RowPtr(idx + 1) - j_start);
        for (; j < num_values; ++j) {
          const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
          const auto ti = (bin + mvg_offset_) << 1;
          //CHECK_LT(ti, 5459 * 2 - 1);
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
      if (MVG_BIT_STATE_8) {
        const auto num_values = static_cast<int>(RowPtr(idx + 1) - j_start);
        for (; j < num_values; ++j) {
          const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
          const auto ti = (bin + mvg_offset_) << 1;
          //CHECK_LT(ti, 5459 * 2 - 1);
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
      if (MVG_BIT_STATE_16) {
        const auto num_values = static_cast<int>(RowPtr(idx + 1) - j_start);
        for (; j < num_values; ++j) {
          const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
          const auto ti = (bin + mvg_offset_) << 1;
          //CHECK_LT(ti, 5459 * 2 - 1);
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
      //CHECK_EQ(i, num_dense_feature_groups_);
      //CHECK_EQ(j, static_cast<int>(RowPtr(idx + 1) - j_start));
    }
  }
  for (; i < end; ++i) {
    const auto idx = USE_INDICES ? data_indices[i] : i;
    const auto j_start = RowPtr(idx);
    const uint16_t* data_ptr = data_ptr_base + j_start;
    const SCORE_T gradient = ORDERED ? gradients[i] : gradients[idx];
    const SCORE_T hessian = ORDERED ? hessians[i] : hessians[idx];
    int j = 0;
    int i = 0;
    if (HAS_BIT_STATE_4) { 
      for (; j < bit4_end_; ++j) {
        uint16_t packed_bin = data_ptr[j];
        const uint32_t bin0 = static_cast<uint32_t>(packed_bin & 0xf);
        packed_bin >>= 4;
        const uint32_t bin1 = static_cast<uint32_t>(packed_bin & 0xf);
        packed_bin >>= 4;
        const uint32_t bin2 = static_cast<uint32_t>(packed_bin & 0xf);
        packed_bin >>= 4;
        const uint32_t bin3 = static_cast<uint32_t>(packed_bin & 0xf);
        const auto ti0 = (bin0 + offsets_[i]) << 1;
        const auto ti1 = (bin1 + offsets_[i + 1]) << 1;
        const auto ti2 = (bin2 + offsets_[i + 2]) << 1;
        const auto ti3 = (bin3 + offsets_[i + 3]) << 1;
        //CHECK_LT(ti0, 5459 * 2 - 1);
        //CHECK_LT(ti1, 5459 * 2 - 1);
        //CHECK_LT(ti2, 5459 * 2 - 1);
        //CHECK_LT(ti3, 5459 * 2 - 1);
        grad[ti0] += gradient;
        hess[ti0] += hessian;
        grad[ti1] += gradient;
        hess[ti1] += hessian;
        grad[ti2] += gradient;
        hess[ti2] += hessian;
        grad[ti3] += gradient;
        hess[ti3] += hessian;
        i += 4;
      }
      if (bit4_end_res_ > 0) {
        uint16_t packed_bin = data_ptr[j];
        for (int k = 0; k < bit4_end_res_; ++k) {
          const uint32_t bin = static_cast<uint32_t>(packed_bin & 0xf);
          packed_bin >>= 4;
          const auto ti = (bin + offsets_[i]) << 1;
          //CHECK_LT(ti, 5459 * 2 - 1);
          grad[ti] += gradient;
          hess[ti] += hessian;
          ++i;
        }
        ++j;
      }
    }
    if (HAS_BIT_STATE_8) {
      for (; j < bit8_end_; ++j) {
        uint16_t packed_bin = data_ptr[j];
        const uint32_t bin0 = static_cast<uint32_t>(packed_bin & 0xff);
        const uint32_t bin1 = static_cast<uint32_t>((packed_bin >> 8) & 0xff);
        const auto ti0 = (bin0 + offsets_[i]) << 1;
        const auto ti1 = (bin1 + offsets_[i + 1]) << 1;
        //CHECK_LT(ti0, 5459 * 2 - 1);
        //CHECK_LT(ti1, 5459 * 2 - 1);
        grad[ti0] += gradient;
        hess[ti0] += hessian;
        grad[ti1] += gradient;
        hess[ti1] += hessian;
        i += 2;
      }
      if (bit8_end_res_ > 0) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + offsets_[i]) << 1;
        //CHECK_LT(ti, 5459 * 2 - 1);
        grad[ti] += gradient;
        hess[ti] += hessian;
        ++i;
        ++j;
      }
    }
    if (HAS_BIT_STATE_16) {
      for (; j < bit16_end_; ++j) {
        uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + offsets_[i]) << 1;
        //CHECK_LT(ti, 5459 * 2 - 1);
        grad[ti] += gradient;
        hess[ti] += hessian;
        ++i;
      }
    }
    if (MVG_BIT_STATE_4) {
      const auto num_values = static_cast<int>(RowPtr(idx + 1) - j_start);
      for (; j < num_values; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + mvg_offset_) << 1;
        //CHECK_LT(ti, 5459 * 2 - 1);
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
    if (MVG_BIT_STATE_8) {
      const auto num_values = static_cast<int>(RowPtr(idx + 1) - j_start);
      for (; j < num_values; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + mvg_offset_) << 1;
        //CHECK_LT(ti, 5459 * 2 - 1);
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
    if (MVG_BIT_STATE_16) {
      const auto num_values = static_cast<int>(RowPtr(idx + 1) - j_start);
      for (; j < num_values; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + mvg_offset_) << 1;
        //CHECK_LT(ti, 5459 * 2 - 1);
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
    //CHECK_EQ(i, num_dense_feature_groups_);
    //CHECK_EQ(j, static_cast<int>(RowPtr(idx + 1) - j_start));
  }
}

template <typename VAL_T>
template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T>
void MultiValMixBin<VAL_T>::ConstructHistogramInner(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const {
  if (has_bit_state_4_) {
    ConstructHistogramInnerUnroll1<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, true>(
      data_indices, start, end, gradients, hessians, out);
  } else {
    ConstructHistogramInnerUnroll1<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, false>(
      data_indices, start, end, gradients, hessians, out);
  }
}

template <typename VAL_T>
template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
  bool HAS_BIT_STATE_4>
void MultiValMixBin<VAL_T>::ConstructHistogramInnerUnroll1(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const {
  if (has_bit_state_8_) {
    ConstructHistogramInnerUnroll2<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, true>(
      data_indices, start, end, gradients, hessians, out);
  } else {
    ConstructHistogramInnerUnroll2<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, false>(
      data_indices, start, end, gradients, hessians, out);
  }
}

template <typename VAL_T>
template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
  bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8>
void MultiValMixBin<VAL_T>::ConstructHistogramInnerUnroll2(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const {
  if (has_bit_state_16_) {
    ConstructHistogramInnerUnroll3<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, HAS_BIT_STATE_8, true>(
      data_indices, start, end, gradients, hessians, out);
  } else {
    ConstructHistogramInnerUnroll3<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, HAS_BIT_STATE_8, false>(
      data_indices, start, end, gradients, hessians, out);
  }
}

template <typename VAL_T>
template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
  bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16>
void MultiValMixBin<VAL_T>::ConstructHistogramInnerUnroll3(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const {
  if (has_bit_state_32_) {
    ConstructHistogramInnerUnroll4<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, true>(
      data_indices, start, end, gradients, hessians, out);
  } else {
    ConstructHistogramInnerUnroll4<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, false>(
      data_indices, start, end, gradients, hessians, out);
  }
}

template <typename VAL_T>
template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
  bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32>
void MultiValMixBin<VAL_T>::ConstructHistogramInnerUnroll4(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const {
  if (mvg_bit_state_4_) {
    ConstructHistogramInnerUnroll5<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, HAS_BIT_STATE_32, true>(
      data_indices, start, end, gradients, hessians, out);
  } else {
    ConstructHistogramInnerUnroll5<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, HAS_BIT_STATE_32, false>(
      data_indices, start, end, gradients, hessians, out);
  }
}

template <typename VAL_T>
template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
  bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32, bool MVG_BIT_STATE_4>
void MultiValMixBin<VAL_T>::ConstructHistogramInnerUnroll5(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const {
  if (mvg_bit_state_8_) {
    ConstructHistogramInnerUnroll6<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, HAS_BIT_STATE_32, MVG_BIT_STATE_4, true>(
      data_indices, start, end, gradients, hessians, out);
  } else {
    ConstructHistogramInnerUnroll6<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, HAS_BIT_STATE_32, MVG_BIT_STATE_4, false>(
      data_indices, start, end, gradients, hessians, out);
  }
}

template <typename VAL_T>
template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
  bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32, bool MVG_BIT_STATE_4, bool MVG_BIT_STATE_8>
void MultiValMixBin<VAL_T>::ConstructHistogramInnerUnroll6(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const {
  if (mvg_bit_state_16_) {
    ConstructHistogramInnerUnroll7<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, HAS_BIT_STATE_32, MVG_BIT_STATE_4, MVG_BIT_STATE_8, true>(
      data_indices, start, end, gradients, hessians, out);
  } else {
    ConstructHistogramInnerUnroll7<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, HAS_BIT_STATE_32, MVG_BIT_STATE_4, MVG_BIT_STATE_8, false>(
      data_indices, start, end, gradients, hessians, out);
  }
}

template <typename VAL_T>
template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
  bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32, bool MVG_BIT_STATE_4, bool MVG_BIT_STATE_8, bool MVG_BIT_STATE_16>
void MultiValMixBin<VAL_T>::ConstructHistogramInnerUnroll7(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const {
  if (mvg_bit_state_32_) {
    ConstructHistogramInnerMost<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, HAS_BIT_STATE_32, MVG_BIT_STATE_4, MVG_BIT_STATE_8, MVG_BIT_STATE_16, true>(
      data_indices, start, end, gradients, hessians, out);
  } else {
    ConstructHistogramInnerMost<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, HAS_BIT_STATE_32, MVG_BIT_STATE_4, MVG_BIT_STATE_8, MVG_BIT_STATE_16, false>(
      data_indices, start, end, gradients, hessians, out);
  }
}

}  // namespace LightGBM
#endif   // LIGHTGBM_IO_MULTI_VAL_MIX_BIN_HPP_
