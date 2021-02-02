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
    const int num_multi_val_group_bin)
    : num_data_(num_data), num_bin_(num_bin), num_feature_(num_feature),
      offsets_(offsets) {
    data_.resize(static_cast<size_t>(num_data_) * num_feature_, static_cast<VAL_T>(0));
    bit_states_.clear();
    num_bit_state_groups_.clear();
    BIT_STATE cur_bit_state = BIT_STATE_4;
    bool has_bit_state_4 = false, has_bit_state_8 = false, has_bit_state_16 = false, has_bit_state_32 = false;
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
            has_bit_state_4 = true;
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
            has_bit_state_8 = true;
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
            has_bit_state_16 = true;
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
        has_bit_state_4 = true;
      } else if (cur_bit_state == BIT_STATE_8) {
        has_bit_state_8 = true;
      } else if (cur_bit_state == BIT_STATE_16) {
        has_bit_state_16 = true;
      } else if (cur_bit_state == BIT_STATE_32) {
        has_bit_state_32 = true;
      }
    }
    if (num_single_val_groups < num_feature_) {
      // has multi val bin
      if (num_multi_val_group_bin <= 16) {
        bit_states_.push_back(BIT_STATE_4);
      } else if (num_multi_val_group_bin <= 256) {
        bit_states_.push_back(BIT_STATE_8);
      } else if (num_multi_val_group_bin <= 65536) {
        bit_states_.push_back(BIT_STATE_16);
      } else {
        bit_states_.push_back(BIT_STATE_32);
      }
    }
    if (has_bit_state_4 && has_bit_state_8 && has_bit_state_16 && has_bit_state_32) {
      CalcBitBoundaries<true, true, true, true>();
    } else if (has_bit_state_4 && has_bit_state_8 && has_bit_state_16 && !has_bit_state_32) {
      CalcBitBoundaries<true, true, true, false>();
    } else if (has_bit_state_4 && has_bit_state_8 && !has_bit_state_16 && has_bit_state_32) {
      CalcBitBoundaries<true, true, false, true>();
    } else if (has_bit_state_4 && has_bit_state_8 && !has_bit_state_16 && !has_bit_state_32) {
      CalcBitBoundaries<true, true, false, false>();
    } else if (has_bit_state_4 && !has_bit_state_8 && has_bit_state_16 && has_bit_state_32) {
      CalcBitBoundaries<true, false, true, true>();
    } else if (has_bit_state_4 && !has_bit_state_8 && has_bit_state_16 && !has_bit_state_32) {
      CalcBitBoundaries<true, false, true, false>();
    } else if (has_bit_state_4 && !has_bit_state_8 && !has_bit_state_16 && has_bit_state_32) {
      CalcBitBoundaries<true, false, false, true>();
    } else if (has_bit_state_4 && !has_bit_state_8 && !has_bit_state_16 && !has_bit_state_32) {
      CalcBitBoundaries<true, false, false, false>();
    } else if (!has_bit_state_4 && has_bit_state_8 && has_bit_state_16 && has_bit_state_32) {
      CalcBitBoundaries<false, true, true, true>();
    } else if (!has_bit_state_4 && has_bit_state_8 && has_bit_state_16 && !has_bit_state_32) {
      CalcBitBoundaries<false, true, true, false>();
    } else if (!has_bit_state_4 && has_bit_state_8 && !has_bit_state_16 && has_bit_state_32) {
      CalcBitBoundaries<false, true, false, true>();
    } else if (!has_bit_state_4 && has_bit_state_8 && !has_bit_state_16 && !has_bit_state_32) {
      CalcBitBoundaries<false, true, false, false>();
    } else if (!has_bit_state_4 && !has_bit_state_8 && has_bit_state_16 && has_bit_state_32) {
      CalcBitBoundaries<false, false, true, true>();
    } else if (!has_bit_state_4 && !has_bit_state_8 && has_bit_state_16 && !has_bit_state_32) {
      CalcBitBoundaries<false, false, true, false>();
    } else if (!has_bit_state_4 && !has_bit_state_8 && !has_bit_state_16 && has_bit_state_32) {
      CalcBitBoundaries<false, false, false, true>();
    } else {
      Log::Fatal("invalid bit state.");
    }
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

  void PushOneRow(int , data_size_t idx, const std::vector<uint32_t>& values) override {
    auto start = RowPtr(idx);
    for (auto i = 0; i < num_feature_; ++i) {
      data_[start + i] = static_cast<VAL_T>(values[i]);
    }
  }

  void FinishLoad() override {
  }

  bool IsSparse() override {
    return false;
  }

  template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
    bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32,
    bool MVG_BIT_STATE_4, bool MVG_BIT_STATE_8, bool MVG_BIT_STATE_16, bool MVG_BIT_STATE_32>
  void ConstructHistogramInner(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const;

  /*template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T>
  void ConstructHistogramInner(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const SCORE_T* gradients, const SCORE_T* hessians, HIST_T* out) const {
    data_size_t i = start;
    HIST_T* grad = out;
    HIST_T* hess = out + 1;
    const VAL_T* data_ptr_base = data_.data();

    if (USE_PREFETCH) {
      const data_size_t pf_offset = 32 / sizeof(VAL_T);
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
        const VAL_T* data_ptr = data_ptr_base + j_start;
        const SCORE_T gradient = ORDERED ? gradients[i] : gradients[idx];
        const SCORE_T hessian = ORDERED ? hessians[i] : hessians[idx];
        for (int j = 0; j < num_feature_; ++j) {
          const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
          const auto ti = (bin + offsets_[j]) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
    }
    for (; i < end; ++i) {
      const auto idx = USE_INDICES ? data_indices[i] : i;
      const auto j_start = RowPtr(idx);
      const VAL_T* data_ptr = data_ptr_base + j_start;
      const SCORE_T gradient = ORDERED ? gradients[i] : gradients[idx];
      const SCORE_T hessian = ORDERED ? hessians[i] : hessians[idx];
      for (int j = 0; j < num_feature_; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + offsets_[j]) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
  }*/

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

  MultiValBin* CreateLike(data_size_t num_data, int num_bin, int num_feature, double,
    const std::vector<uint32_t>& offsets) const override {
    return new MultiValMixBin<VAL_T>(num_data, num_bin, num_feature, offsets);
  }

  void ReSize(data_size_t num_data, int num_bin, int num_feature,
              double, const std::vector<uint32_t>& offsets) override {
    num_data_ = num_data;
    num_bin_ = num_bin;
    num_feature_ = num_feature;
    offsets_ = offsets;
    size_t new_size = static_cast<size_t>(num_feature_) * num_data_;
    if (data_.size() < new_size) {
      data_.resize(new_size, 0);
    }
  }

  template <bool SUBROW, bool SUBCOL>
  void CopyInner(const MultiValBin* full_bin, const data_size_t* used_indices,
                 data_size_t num_used_indices,
                 const std::vector<int>& used_feature_index) {
    const auto other_bin =
        reinterpret_cast<const MultiValMixBin<VAL_T>*>(full_bin);
    if (SUBROW) {
      CHECK_EQ(num_data_, num_used_indices);
    }
    int n_block = 1;
    data_size_t block_size = num_data_;
    Threading::BlockInfo<data_size_t>(num_data_, 1024, &n_block,
                                      &block_size);
#pragma omp parallel for schedule(static, 1)
    for (int tid = 0; tid < n_block; ++tid) {
      data_size_t start = tid * block_size;
      data_size_t end = std::min(num_data_, start + block_size);
      for (data_size_t i = start; i < end; ++i) {
        const auto j_start = RowPtr(i);
        const auto other_j_start =
            SUBROW ? other_bin->RowPtr(used_indices[i]) : other_bin->RowPtr(i);
        for (int j = 0; j < num_feature_; ++j) {
          if (SUBCOL) {
            if (other_bin->data_[other_j_start + used_feature_index[j]] > 0) {
              data_[j_start + j] = static_cast<VAL_T>(
                  other_bin->data_[other_j_start + used_feature_index[j]]);
            } else {
              data_[j_start + j] = 0;
            }
          } else {
            data_[j_start + j] =
                static_cast<VAL_T>(other_bin->data_[other_j_start + j]);
          }
        }
      }
    }
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
    return static_cast<size_t>(idx) * num_feature_;
  }

  MultiValMixBin<VAL_T>* Clone() override;

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
  std::vector<uint32_t> offsets_;
  std::vector<VAL_T, Common::AlignmentAllocator<VAL_T, 32>> data_;
  std::vector<BIT_STATE> bit_states_;
  std::vector<int> num_bit_state_groups_;

  int bit4_start_ = 0;
  int bit4_end_trim_ = 0;
  int bit4_end_ = 0;
  int bit8_start_ = 0;
  int bit8_end_trim_ = 0;
  int bit8_end_ = 0;
  int bit16_start_ = 0;
  int bit16_end_trim_ = 0;
  int bit16_end_ = 0;
  int bit32_start_ = 0;
  int bit32_end_trim_ = 0;
  int bit32_end_ = 0;

  MultiValMixBin<VAL_T>(const MultiValMixBin<VAL_T>& other)
    : num_data_(other.num_data_), num_bin_(other.num_bin_), num_feature_(other.num_feature_),
      offsets_(other.offsets_), data_(other.data_) {
  }

  template <bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32>
  void CalcBitBoundaries();

  template <bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32>
  void CalcBitEnds();
};

template<typename VAL_T>
MultiValMixBin<VAL_T>* MultiValMixBin<VAL_T>::Clone() {
  return new MultiValMixBin<VAL_T>(*this);
}

template <>
template<bool USE_INDICES, bool USE_PREFETCH, bool ORDERED, typename HIST_T, typename SCORE_T,
    bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32,
    bool MVG_BIT_STATE_4, bool MVG_BIT_STATE_8, bool MVG_BIT_STATE_16, bool MVG_BIT_STATE_32>
  void MultiValMixBin<uint8_t>::ConstructHistogramInner(const data_size_t* data_indices, data_size_t start, data_size_t end,
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
        if (HAS_BIT_STATE_4) { 
          for (; j < bit4_end_; ++j) {

          }
        }
        for (int j = 0; j < num_feature_; ++j) {
          const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
          const auto ti = (bin + offsets_[j]) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
    }
    for (; i < end; ++i) {
      const auto idx = USE_INDICES ? data_indices[i] : i;
      const auto j_start = RowPtr(idx);
      const uint8_t* data_ptr = data_ptr_base + j_start;
      const SCORE_T gradient = ORDERED ? gradients[i] : gradients[idx];
      const SCORE_T hessian = ORDERED ? hessians[i] : hessians[idx];
      for (int j = 0; j < num_feature_; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + offsets_[j]) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
  }

}  // namespace LightGBM
#endif   // LIGHTGBM_IO_MULTI_VAL_MIX_BIN_HPP_
