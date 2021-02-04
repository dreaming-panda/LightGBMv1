/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "multi_val_mix_bin.hpp"

namespace LightGBM {

/*template <>
template <bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32>
void MultiValMixBin<uint8_t>::CalcBitBoundaries() {
  int cur_end = 0;
  for (int i = 0; i < static_cast<int>(num_bit_state_groups_.size()); ++i) {
    const int num_groups_in_bit_state = num_bit_state_groups_[i];
    const auto bit_state = bit_states_[i];
    const int cur_num_groups = num_bit_state_groups_[i];
    if (bit_state == BIT_STATE_4) {
      bit4_start_ = 0;
      bit4_end_ = cur_num_groups / 2;
      cur_end = (cur_num_groups + 1) / 2;
      bit4_end_res_ = cur_num_groups % 2;
    } else if (bit_state == BIT_STATE_8) {
			bit8_start_ = cur_end;
      cur_end += cur_num_groups;
      bit8_end_ = cur_end;
      bit8_end_res_ = 0;
    } else {
      Log::Fatal("Too large bit_state encountered in 8 bit multi val mix bin.");
    }
  }
  num_dense_units_per_row_ = cur_end;
}

template <>
template <bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32>
void MultiValMixBin<uint16_t>::CalcBitBoundaries() {
  int cur_end = 0;
  for (int i = 0; i < static_cast<int>(num_bit_state_groups_.size()); ++i) {
    const int num_groups_in_bit_state = num_bit_state_groups_[i];
    const auto bit_state = bit_states_[i];
    const int cur_num_groups = num_bit_state_groups_[i];
    if (bit_state == BIT_STATE_4) {
      bit4_start_ = 0;
      bit4_end_ = cur_num_groups / 4;
			bit4_end_res_ = cur_num_groups % 4;
      cur_end = (cur_num_groups + 3) / 4;
    } else if (bit_state == BIT_STATE_8) {
			bit8_start_ = cur_end;
      bit8_end_ = cur_end + (cur_num_groups / 2);
      bit8_end_res_ = cur_num_groups % 2;
      cur_end += (cur_num_groups + 1) / 2;
    } else if (bit_state == BIT_STATE_16) {
      bit16_start_ = cur_end;
      cur_end += cur_num_groups;
      bit16_end_ = cur_end;
      bit16_end_res_ = 0;
    } else {
      Log::Fatal("Too large bit_state encountered in 16 bit multi val mix bin.");
    }
  }
  num_dense_units_per_row_ = cur_end;
}

template <>
template <bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32>
void MultiValMixBin<uint32_t>::CalcBitBoundaries() {
  int cur_end = 0;
  for (int i = 0; i < static_cast<int>(num_bit_state_groups_.size()); ++i) {
    const int num_groups_in_bit_state = num_bit_state_groups_[i];
    const auto bit_state = bit_states_[i];
    const int cur_num_groups = num_bit_state_groups_[i];
    if (bit_state == BIT_STATE_4) {
			bit4_start_ = 0;
      bit4_end_ = cur_num_groups / 8;
			bit4_end_res_ = cur_num_groups % 8;
      cur_end = (cur_num_groups + 7) / 8;
    } else if (bit_state == BIT_STATE_8) {
      bit8_start_ = cur_end;
      bit8_end_ = cur_end + (cur_num_groups / 4);
      bit8_end_res_ = cur_num_groups % 4;
      cur_end += (cur_num_groups + 3) / 4;
    } else if (bit_state == BIT_STATE_16) {
      bit16_start_ = cur_end;
      bit16_end_ = cur_end + (cur_num_groups / 2);
      bit16_end_res_ = cur_num_groups % 2;
      cur_end += (cur_num_groups + 1) / 2;
    } else if (bit_state == BIT_STATE_32) {
      bit32_start_ = cur_end;
      cur_end += cur_num_groups;
      bit32_end_ = cur_end;
      bit32_end_res_ = 0;
    } else {
      Log::Fatal("Too large bit_state encountered in 32 bit multi val mix bin.");
    }
  }
  num_dense_units_per_row_ = cur_end;
}

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
        const auto num_values = RowPtr(idx + 1) - j_start;
        for (; j < num_values; ++j) {
          const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
          const auto ti = (bin + mvg_offset_) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
      if (MVG_BIT_STATE_8) {
        const auto num_values = RowPtr(idx + 1) - j_start;
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
      const auto num_values = RowPtr(idx + 1) - j_start;
      for (; j < num_values; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + mvg_offset_) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
    if (MVG_BIT_STATE_8) {
      const auto num_values = RowPtr(idx + 1) - j_start;
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
          const auto ti2 = (bin0 + offsets_[i + 2]) << 1;
          const auto ti3 = (bin1 + offsets_[i + 3]) << 1;
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
          const uint32_t bin1 = static_cast<uint32_t>((packed_bin >> 4) & 0xff);
          const auto ti0 = (bin0 + offsets_[i]) << 1;
          const auto ti1 = (bin1 + offsets_[i + 1]) << 1; 
          grad[ti0] += gradient;
          hess[ti0] += hessian;
          grad[ti1] += gradient;
          hess[ti1] += hessian;
          i += 2;
        }
        if (bit8_end_res_ > 0) {
          const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
          const auto ti = (bin + offsets_[i]) << 1;
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
          grad[ti] += gradient;
          hess[ti] += hessian;
          ++i;
        }
      }
      if (MVG_BIT_STATE_4) {
        const auto num_values = RowPtr(idx + 1) - j_start;
        for (; j < num_values; ++j) {
          const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
          const auto ti = (bin + mvg_offset_) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
      if (MVG_BIT_STATE_8) {
        const auto num_values = RowPtr(idx + 1) - j_start;
        for (; j < num_values; ++j) {
          const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
          const auto ti = (bin + mvg_offset_) << 1;
          grad[ti] += gradient;
          hess[ti] += hessian;
        }
      }
      if (MVG_BIT_STATE_16) {
        const auto num_values = RowPtr(idx + 1) - j_start;
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
        const auto ti2 = (bin0 + offsets_[i + 2]) << 1;
        const auto ti3 = (bin1 + offsets_[i + 3]) << 1;
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
        const uint32_t bin1 = static_cast<uint32_t>((packed_bin >> 4) & 0xff);
        const auto ti0 = (bin0 + offsets_[i]) << 1;
        const auto ti1 = (bin1 + offsets_[i + 1]) << 1; 
        grad[ti0] += gradient;
        hess[ti0] += hessian;
        grad[ti1] += gradient;
        hess[ti1] += hessian;
        i += 2;
      }
      if (bit8_end_res_ > 0) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + offsets_[i]) << 1;
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
        grad[ti] += gradient;
        hess[ti] += hessian;
        ++i;
      }
    }
    if (MVG_BIT_STATE_4) {
      const auto num_values = RowPtr(idx + 1) - j_start;
      for (; j < num_values; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + mvg_offset_) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
    if (MVG_BIT_STATE_8) {
      const auto num_values = RowPtr(idx + 1) - j_start;
      for (; j < num_values; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + mvg_offset_) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
    if (MVG_BIT_STATE_16) {
      const auto num_values = RowPtr(idx + 1) - j_start;
      for (; j < num_values; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + mvg_offset_) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
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
}*/

template <>
void MultiValMixBin<uint8_t>::CalcBitBoundaries() {
  int cur_end = 0;
  for (int i = 0; i < static_cast<int>(num_bit_state_groups_.size()); ++i) {
    const auto bit_state = bit_states_[i];
    const int cur_num_groups = num_bit_state_groups_[i];
    if (bit_state == BIT_STATE_4) {
      bit4_start_ = 0;
      bit4_end_ = cur_num_groups / 2;
      cur_end = (cur_num_groups + 1) / 2;
      bit4_end_res_ = cur_num_groups % 2;
    } else if (bit_state == BIT_STATE_8) {
			bit8_start_ = cur_end;
      cur_end += cur_num_groups;
      bit8_end_ = cur_end;
      bit8_end_res_ = 0;
    } else {
      Log::Fatal("Too large bit_state encountered in 8 bit multi val mix bin.");
    }
  }
  num_dense_units_per_row_ = cur_end;
}

template <>
void MultiValMixBin<uint16_t>::CalcBitBoundaries() {
  int cur_end = 0;
  for (int i = 0; i < static_cast<int>(num_bit_state_groups_.size()); ++i) {
    const auto bit_state = bit_states_[i];
    const int cur_num_groups = num_bit_state_groups_[i];
    if (bit_state == BIT_STATE_4) {
      bit4_start_ = 0;
      bit4_end_ = cur_num_groups / 4;
			bit4_end_res_ = cur_num_groups % 4;
      cur_end = (cur_num_groups + 3) / 4;
    } else if (bit_state == BIT_STATE_8) {
			bit8_start_ = cur_end;
      bit8_end_ = cur_end + (cur_num_groups / 2);
      bit8_end_res_ = cur_num_groups % 2;
      cur_end += (cur_num_groups + 1) / 2;
    } else if (bit_state == BIT_STATE_16) {
      bit16_start_ = cur_end;
      cur_end += cur_num_groups;
      bit16_end_ = cur_end;
      bit16_end_res_ = 0;
    } else {
      Log::Fatal("Too large bit_state encountered in 16 bit multi val mix bin.");
    }
  }
  num_dense_units_per_row_ = cur_end;
}

template <>
void MultiValMixBin<uint8_t>::CompressOneRow(const std::vector<uint32_t>& values, std::vector<uint8_t>* compressed_values) {
  compressed_values->clear();
  int j = 0, i = 0;
  if (has_bit_state_4_) {
    for (; j < bit4_end_; ++j) {
      const uint32_t bin0 = values[i];
      const uint32_t bin1 = values[i + 1];
      const uint8_t compressed_bin = static_cast<uint8_t>((bin0 & 0xf) | ((bin1 << 4) & 0xf0));
      compressed_values->push_back(compressed_bin);
      i += 2;
    }
    if (bit4_end_res_ > 0) {
      const uint8_t bin = static_cast<uint8_t>(values[i]);
      compressed_values->push_back(bin);
      ++i;
      ++j;
    }
  }
  if (has_bit_state_8_) {
    for (; j < bit8_end_; ++j) {
      const uint8_t bin = static_cast<uint8_t>(values[i]);
      compressed_values->push_back(bin);
      ++i;
    }
  }
  const int num_values = static_cast<int>(values.size());
  if (mvg_bit_state_4_) {
    for (; i < num_values; ++i) {
      compressed_values->push_back(static_cast<uint8_t>(values[i] - mvg_offset_));
    }
  }
  if (mvg_bit_state_8_) {
    for (; i < num_values; ++i) {
      compressed_values->push_back(static_cast<uint8_t>(values[i] - mvg_offset_));
    }
  }
}

template <>
void MultiValMixBin<uint16_t>::CompressOneRow(const std::vector<uint32_t>& values, std::vector<uint16_t>* compressed_values) {
  int j = 0, i = 0;
  compressed_values->clear();
  if (has_bit_state_4_) {
    for (; j < bit4_end_; ++j) {
      const uint32_t bin0 = values[i];
      const uint32_t bin1 = values[i + 1];
      const uint32_t bin2 = values[i + 2];
      const uint32_t bin3 = values[i + 3];
      const uint16_t compressed_bin = static_cast<uint16_t>((bin0 & 0xf) | ((bin1 << 4) & 0xf0) |
        ((bin2 << 8) & 0xf00) | ((bin3 << 12) & 0xf000));
      compressed_values->push_back(compressed_bin);
      i += 4;
    }
    if (bit4_end_res_ > 0) {
      uint16_t compressed_bin = 0;
      for (int k = 0; k < bit4_end_res_; ++k) {
        compressed_bin <<= 4;
        compressed_bin |= values[i];
        ++i;
      }
      compressed_values->push_back(compressed_bin);
      ++j;
    }
  }
  if (has_bit_state_8_) {
    for (; j < bit8_end_; ++j) {
      const uint32_t bin0 = values[i];
      const uint32_t bin1 = values[i + 1];
      const uint16_t compressed_bin = static_cast<uint16_t>((bin0 & 0xff) | ((bin1 & 0xff) << 8));
      compressed_values->push_back(compressed_bin);
      i += 2;
    }
    if (bit8_end_res_ > 0) {
      compressed_values->push_back(static_cast<uint16_t>(values[i]));
      ++i;
      ++j;
    }
  }
  if (has_bit_state_16_) {
    for (; j < bit16_end_; ++j) {
      compressed_values->push_back(static_cast<uint16_t>(values[i]));
      ++i;
    }
  }
  const int num_values = static_cast<int>(values.size());
  if (mvg_bit_state_4_) {
    for (; i < num_values; ++i) {
      compressed_values->push_back(static_cast<uint16_t>(values[i] - mvg_offset_));
    }
  }
  if (mvg_bit_state_8_) {
    for (; i < num_values; ++i) {
      CHECK_GE(static_cast<int>(values[i]), mvg_offset_);
      compressed_values->push_back(static_cast<uint16_t>(values[i] - mvg_offset_));
    }
  }
  if (mvg_bit_state_16_) {
    for (; i < num_values; ++i) {
      compressed_values->push_back(static_cast<uint16_t>(values[i] - mvg_offset_));
    }
  }
}

} //  namespace LightGBM
