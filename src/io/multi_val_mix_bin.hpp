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
    const std::vector<uint32_t>& offsets, const double estimate_element_per_row,
    const int num_dense_col);

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
        static_cast<double>(row_ptr_[num_data_]) / num_data_ - bit16_start_;
    if (!has_multi_val_) {
      size_dense_col_ = row_ptr_[1];
    } else {
      size_dense_col_ = 0;
    }
    offsets_uint16_.resize(num_dense_col_);
    for (int i = 0; i < num_dense_col_; ++i) {
      offsets_uint16_[i] = static_cast<uint16_t>(offsets_[i]);
    }
    offsets_.resize(num_dense_col_);
    offsets_.shrink_to_fit();
  }

  bool IsSparse() override {
    return false;
  }

  bool IsMix() override {
    return true;
  }

  template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
    bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32,
    bool HAS_MULTI_VAL, bool NEED_OFFSET>
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
    bool HAS_MULTI_VAL>
  inline void ConstructHistogramInnerUnroll5(const data_size_t* data_indices, data_size_t start, data_size_t end,
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

  MultiValBin* CreateLike(data_size_t /*num_data*/, int /*num_bin*/, int /*num_feature*/, double /*estimtate_elements_per_row*/,
    const std::vector<uint32_t>& /*offsets*/) const override {
    Log::Fatal("unsupported");
    return nullptr;
  }

  void ReSize(data_size_t /*num_data*/, int /*num_bin*/, int /*num_feature*/,
              double /*estimate_elements_per_row*/, const std::vector<uint32_t>& /*offsets*/) override {
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
  int num_dense_col_;
  int size_dense_col_;
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
  std::vector<uint16_t> offsets_uint16_;

  int bit4_start_ = 0;
  int bit4_end_ = 0;
  int bit8_start_ = 0;
  int bit8_end_ = 0;
  int bit16_start_ = 0;
  int bit16_end_ = 0;
  int bit32_start_ = 0;
  int bit32_end_ = 0;

  bool has_bit_state_4_ = false;
  bool has_bit_state_8_ = false;
  bool has_bit_state_16_ = false;
  bool has_bit_state_32_ = false;

  bool has_multi_val_ = false;
  int multi_val_offset_ = 0;

  bool need_offset_ = false;

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
    bool HAS_MULTI_VAL, bool NEED_OFFSET>
void MultiValMixBin<uint16_t>::ConstructHistogramInnerMost(const data_size_t* data_indices, data_size_t start, data_size_t end,
  const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const {
  data_size_t i = start;
  HIST_T* grad = out;
  HIST_T* hess = out + 1;
  const uint16_t* data_ptr_base = data_.data();
  const uint16_t* data_ptr_uint8 = nullptr;
  const uint16_t* data_ptr_uint16 = nullptr;
  int num_values = num_dense_col_;
  const uint32_t* offsets_ptr = offsets_.data();
  const int local_bit4_end = bit4_end_;
  const int local_bit8_end = bit8_end_;
  const int local_bit16_end = bit16_end_;
  const int local_bit8_start = bit8_start_;
  const int local_bit16_start = bit16_start_;
  const int local_size_dense_col = size_dense_col_;
  const int local_multi_val_offset = multi_val_offset_;
  const uint32_t* row_ptr_base = row_ptr_.data();
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
      if (!HAS_MULTI_VAL) {
        PREFETCH_T0(data_ptr_base + pf_idx * local_size_dense_col);
      } else {
        PREFETCH_T0(data_ptr_base + row_ptr_base[pf_idx]);
      }
      const auto j_start = HAS_MULTI_VAL ? row_ptr_base[idx] : idx * local_size_dense_col;
      if (HAS_MULTI_VAL) {
        num_values = static_cast<int>(row_ptr_base[idx + 1] - j_start - local_bit16_start);
      }
      const uint16_t* data_ptr = data_ptr_base + j_start;
      if (HAS_BIT_STATE_4) {
        data_ptr_uint8 = data_ptr + local_bit8_start;
      }
      if (HAS_BIT_STATE_4 || HAS_BIT_STATE_8) {
        data_ptr_uint16 = data_ptr + local_bit16_start;
      }
      const SCORE_T gradient = ORDERED ? gradients[i] : gradients[idx];
      const SCORE_T hessian = ORDERED ? hessians[i] : hessians[idx];
      int j = 0;
      if (HAS_BIT_STATE_4) {
        for (; j < local_bit4_end; j += 4) {
          uint16_t packed_bin = data_ptr[j >> 2];
          const uint32_t bin0 = static_cast<uint32_t>(packed_bin & 0xf);
          const auto ti0 = (bin0 + offsets_ptr[j]) << 1;
          grad[ti0] += gradient;
          hess[ti0] += hessian;

          const uint32_t bin1 = static_cast<uint32_t>((packed_bin >> 4) & 0xf);
          const auto ti1 = (bin1 + offsets_ptr[j + 1]) << 1;
          grad[ti1] += gradient;
          hess[ti1] += hessian;

          const uint32_t bin2 = static_cast<uint32_t>((packed_bin >> 8) & 0xf);
          const auto ti2 = (bin2 + offsets_ptr[j + 2]) << 1;
          grad[ti2] += gradient;
          hess[ti2] += hessian;

          const uint32_t bin3 = static_cast<uint32_t>((packed_bin >> 12) & 0xf);
          const auto ti3 = (bin3 + offsets_ptr[j + 3]) << 1;
          grad[ti3] += gradient;
          hess[ti3] += hessian;
        }
      }
      if (HAS_BIT_STATE_8) {
        for (; j < local_bit8_end; j += 2) {
          const uint16_t packed_bin = HAS_BIT_STATE_4 ?
            static_cast<uint16_t>(data_ptr_uint8[j >> 1]) :
            static_cast<uint16_t>(data_ptr[j >> 1]);
          const uint32_t bin0 = static_cast<uint32_t>(packed_bin & 0xff);
          const auto ti0 = (bin0 + offsets_ptr[j]) << 1;
          grad[ti0] += gradient;
          hess[ti0] += hessian;

          const uint32_t bin1 = static_cast<uint32_t>((packed_bin >> 8) & 0xff);
          const auto ti1 = (bin1 + offsets_ptr[j + 1]) << 1;
          grad[ti1] += gradient;
          hess[ti1] += hessian;
        }
      }
      if (HAS_BIT_STATE_16 || HAS_MULTI_VAL) {
        if (NEED_OFFSET) {
          if (HAS_BIT_STATE_16) {
            for (; j < local_bit16_end; ++j) {
              const uint32_t bin = (HAS_BIT_STATE_4 || HAS_BIT_STATE_8) ?
                static_cast<uint32_t>(data_ptr_uint16[j]):
                static_cast<uint32_t>(data_ptr[j]);
              const auto ti = (bin + offsets_ptr[j]) << 1;
              grad[ti] += gradient;
              hess[ti] += hessian;
            }
          }
          if (HAS_MULTI_VAL) {
            for (; j < num_values; ++j) {
              const uint32_t bin = (HAS_BIT_STATE_4 || HAS_BIT_STATE_8) ?
                static_cast<uint32_t>(data_ptr_uint16[j]):
                static_cast<uint32_t>(data_ptr[j]);
              const auto ti = (bin + local_multi_val_offset) << 1;
              grad[ti] += gradient;
              hess[ti] += hessian;
            }
          }
        } else {
          for (; j < num_values; ++j) {
            const auto ti = (HAS_BIT_STATE_4 || HAS_BIT_STATE_8) ?
              static_cast<uint32_t>(data_ptr_uint16[j]) << 1 :
              static_cast<uint32_t>(data_ptr[j]) << 1;
            grad[ti] += gradient;
            hess[ti] += hessian;
          }
        }
      }
    }
  }
  for (; i < end; ++i) {
    const auto idx = USE_INDICES ? data_indices[i] : i;
    const auto j_start = HAS_MULTI_VAL ? row_ptr_base[idx] : idx * local_size_dense_col;
    const uint16_t* data_ptr = data_ptr_base + j_start;
    if (HAS_BIT_STATE_4) {
      data_ptr_uint8 = data_ptr + local_bit8_start;
    }
    if (HAS_BIT_STATE_4 || HAS_BIT_STATE_8) {
      data_ptr_uint16 = data_ptr + local_bit16_start;
    }
    const SCORE_T gradient = ORDERED ? gradients[i] : gradients[idx];
    const SCORE_T hessian = ORDERED ? hessians[i] : hessians[idx];
    if (HAS_MULTI_VAL) {
      num_values = static_cast<int>(row_ptr_base[idx + 1] - j_start - local_bit16_start);
    }
    int j = 0;
    if (HAS_BIT_STATE_4) { 
      for (; j < local_bit4_end; j += 4) {
        uint16_t packed_bin = data_ptr[j >> 2];
        const uint32_t bin0 = static_cast<uint32_t>(packed_bin & 0xf);
        const auto ti0 = (bin0 + offsets_ptr[j]) << 1;
        grad[ti0] += gradient;
        hess[ti0] += hessian;

        const uint32_t bin1 = static_cast<uint32_t>((packed_bin >> 4) & 0xf);
        const auto ti1 = (bin1 + offsets_ptr[j + 1]) << 1;
        grad[ti1] += gradient;
        hess[ti1] += hessian;

        const uint32_t bin2 = static_cast<uint32_t>((packed_bin >> 8) & 0xf);
        const auto ti2 = (bin2 + offsets_ptr[j + 2]) << 1;
        grad[ti2] += gradient;
        hess[ti2] += hessian;

        const uint32_t bin3 = static_cast<uint32_t>((packed_bin >> 12) & 0xf);
        const auto ti3 = (bin3 + offsets_ptr[j + 3]) << 1;
        grad[ti3] += gradient;
        hess[ti3] += hessian;
      }
    }
    if (HAS_BIT_STATE_8) {
      for (; j < local_bit8_end; j += 2) {
        const uint16_t packed_bin = HAS_BIT_STATE_4 ?
            static_cast<uint16_t>(data_ptr_uint8[j >> 1]) :
            static_cast<uint16_t>(data_ptr[j >> 1]);
        const uint32_t bin0 = static_cast<uint32_t>(packed_bin & 0xff);
        const auto ti0 = (bin0 + offsets_ptr[j]) << 1;
        grad[ti0] += gradient;
        hess[ti0] += hessian;

        const uint32_t bin1 = static_cast<uint32_t>((packed_bin >> 8) & 0xff);
        const auto ti1 = (bin1 + offsets_ptr[j + 1]) << 1;
        grad[ti1] += gradient;
        hess[ti1] += hessian;
      }
    }
    if (HAS_BIT_STATE_16 || HAS_MULTI_VAL) {
      if (NEED_OFFSET) {
        if (HAS_BIT_STATE_16) {
          for (; j < local_bit16_end; ++j) {
            const uint32_t bin = (HAS_BIT_STATE_4 || HAS_BIT_STATE_8) ?
              static_cast<uint32_t>(data_ptr_uint16[j]) :
              static_cast<uint32_t>(data_ptr[j]);
            const auto ti = (bin + offsets_ptr[j]) << 1;
            grad[ti] += gradient;
            hess[ti] += hessian;
          }
        }
        if (HAS_MULTI_VAL) {
          for (; j < num_values; ++j) {
            const uint32_t bin = (HAS_BIT_STATE_4 || HAS_BIT_STATE_8) ?
              static_cast<uint32_t>(data_ptr_uint16[j]) :
              static_cast<uint32_t>(data_ptr[j]);
            const auto ti = (bin + local_multi_val_offset) << 1;
            grad[ti] += gradient;
            hess[ti] += hessian;
          }
        }
      } else {
        for (; j < num_values; ++j) {
          const auto ti = (HAS_BIT_STATE_4 || HAS_BIT_STATE_8) ?
            static_cast<uint32_t>(data_ptr_uint16[j]) << 1 :
            static_cast<uint32_t>(data_ptr[j]) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
    }
  }
}

template <>
template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
    bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool /*HAS_BIT_STATE_16*/, bool /*HAS_BIT_STATE_32*/,
    bool HAS_MULTI_VAL, bool NEED_OFFSET>
void MultiValMixBin<uint8_t>::ConstructHistogramInnerMost(const data_size_t* data_indices, data_size_t start, data_size_t end,
  const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const {
  data_size_t i = start;
  HIST_T* grad = out;
  HIST_T* hess = out + 1;
  const uint8_t* data_ptr_base = data_.data();
  const uint8_t* data_ptr_uint8 = nullptr;
  int num_values = num_dense_col_;
  const uint32_t* offsets_ptr = offsets_.data();
  const int local_bit4_end = bit4_end_;
  const int local_bit8_end = bit8_end_;
  const int local_bit8_start = bit8_start_;
  const int local_size_dense_col = size_dense_col_;
  const int local_multi_val_offset = multi_val_offset_;
  const uint32_t* row_ptr_base = row_ptr_.data();
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
      if (!HAS_MULTI_VAL) {
        PREFETCH_T0(data_ptr_base + pf_idx * local_size_dense_col);
      } else {
        PREFETCH_T0(data_ptr_base + row_ptr_base[pf_idx]);
      }
      const auto j_start = HAS_MULTI_VAL ? row_ptr_base[idx] : idx * local_size_dense_col;
      if (HAS_MULTI_VAL) {
        num_values = static_cast<int>(row_ptr_base[idx + 1] - j_start - local_bit8_start);
      }
      const uint8_t* data_ptr = data_ptr_base + j_start;
      if (HAS_BIT_STATE_4) {
        data_ptr_uint8 = data_ptr + local_bit8_start;
      }
      const SCORE_T gradient = ORDERED ? gradients[i] : gradients[idx];
      const SCORE_T hessian = ORDERED ? hessians[i] : hessians[idx];
      int j = 0;
      if (HAS_BIT_STATE_4) {
        for (; j < local_bit4_end; j += 2) {
          uint8_t packed_bin = data_ptr[j >> 1];
          const uint32_t bin0 = static_cast<uint32_t>(packed_bin & 0xf);
          const auto ti0 = (bin0 + offsets_ptr[j]) << 1;
          grad[ti0] += gradient;
          hess[ti0] += hessian;

          const uint32_t bin1 = static_cast<uint32_t>((packed_bin >> 4) & 0xf);
          const auto ti1 = (bin1 + offsets_ptr[j + 1]) << 1;
          grad[ti1] += gradient;
          hess[ti1] += hessian;
        }
      }
      if (HAS_BIT_STATE_8 || HAS_MULTI_VAL) {
        if (NEED_OFFSET) {
          if (HAS_BIT_STATE_8) {
            for (; j < local_bit8_end; ++j) {
              const uint32_t bin = HAS_BIT_STATE_4 ?
                static_cast<uint32_t>(data_ptr_uint8[j]):
                static_cast<uint32_t>(data_ptr[j]);
              const auto ti = (bin + offsets_ptr[j]) << 1;
              grad[ti] += gradient;
              hess[ti] += hessian;
            }
          }
          if (HAS_MULTI_VAL) {
            for (; j < num_values; ++j) {
              const uint32_t bin = HAS_BIT_STATE_4 ?
                static_cast<uint32_t>(data_ptr_uint8[j]):
                static_cast<uint32_t>(data_ptr[j]);
              const auto ti = (bin + local_multi_val_offset) << 1;
              grad[ti] += gradient;
              hess[ti] += hessian;
            }
          }
        } else {
          for (; j < num_values; ++j) {
            const auto ti = HAS_BIT_STATE_4 ?
              static_cast<uint32_t>(data_ptr_uint8[j]) << 1 :
              static_cast<uint32_t>(data_ptr[j]) << 1;
            grad[ti] += gradient;
            hess[ti] += hessian;
          }
        }
      }
    }
  }
  for (; i < end; ++i) {
    const auto idx = USE_INDICES ? data_indices[i] : i;
    const auto j_start = HAS_MULTI_VAL ? row_ptr_base[idx] : idx * local_size_dense_col;
    const uint8_t* data_ptr = data_ptr_base + j_start;
    if (HAS_BIT_STATE_4) {
      data_ptr_uint8 = data_ptr + local_bit8_start;
    }
    const SCORE_T gradient = ORDERED ? gradients[i] : gradients[idx];
    const SCORE_T hessian = ORDERED ? hessians[i] : hessians[idx];
    if (HAS_MULTI_VAL) {
      num_values = static_cast<int>(row_ptr_base[idx + 1] - j_start - local_bit8_start);
    }
    int j = 0;
    if (HAS_BIT_STATE_4) {
      for (; j < local_bit4_end; j += 2) {
        uint8_t packed_bin = data_ptr[j >> 1];
        const uint32_t bin0 = static_cast<uint32_t>(packed_bin & 0xf);
        const auto ti0 = (bin0 + offsets_ptr[j]) << 1;
        grad[ti0] += gradient;
        hess[ti0] += hessian;

        const uint32_t bin1 = static_cast<uint32_t>((packed_bin >> 4) & 0xf);
        const auto ti1 = (bin1 + offsets_ptr[j + 1]) << 1;
        grad[ti1] += gradient;
        hess[ti1] += hessian;
      }
    }
    if (HAS_BIT_STATE_8 || HAS_MULTI_VAL) {
      if (NEED_OFFSET) {
        if (HAS_BIT_STATE_8) {
          for (; j < local_bit8_end; ++j) {
            const uint32_t bin = HAS_BIT_STATE_4 ?
              static_cast<uint32_t>(data_ptr_uint8[j]):
              static_cast<uint32_t>(data_ptr[j]);
            const auto ti = (bin + offsets_ptr[j]) << 1;
            grad[ti] += gradient;
            hess[ti] += hessian;
          }
        }
        if (HAS_MULTI_VAL) {
          for (; j < num_values; ++j) {
            const uint32_t bin = HAS_BIT_STATE_4 ?
              static_cast<uint32_t>(data_ptr_uint8[j]):
              static_cast<uint32_t>(data_ptr[j]);
            const auto ti = (bin + local_multi_val_offset) << 1;
            grad[ti] += gradient;
            hess[ti] += hessian;
          }
        }
      } else {
        for (; j < num_values; ++j) {
          const auto ti = HAS_BIT_STATE_4 ?
            static_cast<uint32_t>(data_ptr_uint8[j]) << 1 :
            static_cast<uint32_t>(data_ptr[j]) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
    }
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
  if (has_multi_val_) {
    ConstructHistogramInnerUnroll5<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, HAS_BIT_STATE_32, true>(
      data_indices, start, end, gradients, hessians, out);
  } else {
    ConstructHistogramInnerUnroll5<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, HAS_BIT_STATE_32, false>(
      data_indices, start, end, gradients, hessians, out);
  }
}

template <typename VAL_T>
template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
  bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32, bool HAS_MULTI_VAL>
void MultiValMixBin<VAL_T>::ConstructHistogramInnerUnroll5(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const {
  if (need_offset_) {
    ConstructHistogramInnerMost<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, HAS_BIT_STATE_32, HAS_MULTI_VAL, true>(
      data_indices, start, end, gradients, hessians, out);
  } else {
    ConstructHistogramInnerMost<USE_INDICES, USE_PREFETCH, ORDERED, HIST_T, SCORE_T, HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, HAS_BIT_STATE_32, HAS_MULTI_VAL, false>(
      data_indices, start, end, gradients, hessians, out);
  }
}

}  // namespace LightGBM
#endif   // LIGHTGBM_IO_MULTI_VAL_MIX_BIN_HPP_
