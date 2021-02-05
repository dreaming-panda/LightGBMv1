/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "multi_val_mix_bin.hpp"

namespace LightGBM {

template <>
void MultiValMixBin<uint8_t>::CalcBitBoundaries() {
  int cur_end = 0;
  for (int i = 0; i < static_cast<int>(num_bit_state_groups_.size()); ++i) {
    const auto bit_state = bit_states_[i];
    const int cur_num_groups = num_bit_state_groups_[i];
    cur_end += cur_num_groups;
    if (bit_state == BIT_STATE_4) {
      bit4_start_ = 0;
      bit4_end_ = cur_end;
    } else if (bit_state == BIT_STATE_8) {
			bit8_start_ = -(bit4_end_ / 2);
      bit8_end_ = cur_end;
    } else {
      Log::Fatal("Too large bit_state encountered in 8 bit multi val mix bin.");
    }
  }
  if (has_multi_val_ && !has_bit_state_8_) {
    bit8_start_ = -(bit4_end_ / 2);
  }
}

template <>
void MultiValMixBin<uint16_t>::CalcBitBoundaries() {
  int cur_end = 0;
  for (int i = 0; i < static_cast<int>(num_bit_state_groups_.size()); ++i) {
    const auto bit_state = bit_states_[i];
    const int cur_num_groups = num_bit_state_groups_[i];
    cur_end += cur_num_groups;
    if (bit_state == BIT_STATE_4) {
      bit4_start_ = 0;
      bit4_end_ = cur_end;
    } else if (bit_state == BIT_STATE_8) {
			bit8_start_ = -(bit4_end_ / 4);
      bit8_end_ = cur_end;
    } else if (bit_state == BIT_STATE_16) {
      bit16_start_ = (bit4_end_ * 4 + (bit8_end_ - bit4_end_) * 8) / 16 - bit8_end_;
      bit16_end_ = cur_end;
    } else {
      Log::Fatal("Too large bit_state encountered in 16 bit multi val mix bin.");
    }
  }
  if (has_multi_val_ && !has_bit_state_16_) {
    bit16_start_ = (bit4_end_ * 4 + (bit8_end_ - bit4_end_) * 8) / 16 - bit8_end_;
  }
}

template <>
void MultiValMixBin<uint8_t>::CompressOneRow(const std::vector<uint32_t>& values, std::vector<uint8_t>* compressed_values) {
  int i = 0;
  compressed_values->clear();
  if (has_bit_state_4_) {
    for (; i < bit4_end_; i += 2) {
      const uint32_t bin0 = values[i];
      const uint32_t bin1 = values[i + 1];
      const uint8_t compressed_bin = static_cast<uint8_t>((bin0 & 0xf) | ((bin1 << 4) & 0xf0));
      compressed_values->push_back(compressed_bin);
    }
  }
  if (has_bit_state_8_) {
    if (need_offset_) {
      for (; i < bit8_end_; ++i) {
        compressed_values->push_back(static_cast<uint8_t>(values[i]));
      }
    } else {
      for (; i < bit8_end_; ++i) {
        compressed_values->push_back(static_cast<uint8_t>(values[i] + offsets_[i]));
      }
    }
  }
  if (has_multi_val_) {
    if (need_offset_) {
      for (; i < static_cast<int>(values.size()); ++i) {
        compressed_values->push_back(static_cast<uint8_t>(values[i]));
      }
    } else {
      for (; i < static_cast<int>(values.size()); ++i) {
        compressed_values->push_back(static_cast<uint8_t>(values[i] + multi_val_offset_));
      }
    }
  }
}

template <>
void MultiValMixBin<uint16_t>::CompressOneRow(const std::vector<uint32_t>& values, std::vector<uint16_t>* compressed_values) {
  int i = 0;
  compressed_values->clear();
  if (has_bit_state_4_) {
    for (; i < bit4_end_; i += 4) {
      const uint32_t bin0 = values[i];
      const uint32_t bin1 = values[i + 1];
      const uint32_t bin2 = values[i + 2];
      const uint32_t bin3 = values[i + 3];
      const uint16_t compressed_bin = static_cast<uint16_t>((bin0 & 0xf) | ((bin1 << 4) & 0xf0) |
        ((bin2 << 8) & 0xf00) | ((bin3 << 12) & 0xf000));
      compressed_values->push_back(compressed_bin);
    }
  }
  if (has_bit_state_8_) {
    for (; i < bit8_end_; i += 2) {
      const uint32_t bin0 = values[i];
      const uint32_t bin1 = values[i + 1];
      const uint16_t compressed_bin = static_cast<uint16_t>((bin0 & 0xff) | ((bin1 & 0xff) << 8));
      compressed_values->push_back(compressed_bin);
    }
  }
  if (has_bit_state_16_) {
    if (need_offset_) {
      for (; i < bit16_end_; ++i) {
        compressed_values->push_back(static_cast<uint16_t>(values[i]));
      }
    } else {
      for (; i < bit16_end_; ++i) {
        compressed_values->push_back(static_cast<uint16_t>(values[i] + offsets_[i]));
      }
    }
  }
  if (has_multi_val_) {
    if (need_offset_) {
      for (; i < static_cast<int>(values.size()); ++i) {
        compressed_values->push_back(static_cast<uint16_t>(values[i]));
      }
    } else {
      for (; i < static_cast<int>(values.size()); ++i) {
        compressed_values->push_back(static_cast<uint16_t>(values[i] + multi_val_offset_));
      }
    }
  }
}

template <>
MultiValMixBin<uint8_t>::MultiValMixBin(data_size_t num_data, int num_bin, int num_feature,
  const std::vector<uint32_t>& offsets, const double estimate_element_per_row,
  const int num_dense_col)
  : num_data_(num_data), num_bin_(num_bin), num_feature_(num_feature), num_dense_col_(num_dense_col),
    offsets_(offsets), estimate_element_per_row_(estimate_element_per_row) {
  bit_states_.clear();
  num_bit_state_groups_.clear();
  BIT_STATE cur_bit_state = BIT_STATE_4;
  int num_groups_in_cur_bit_state = 0;
  for (int group_index = 0; group_index < num_dense_col_; ++group_index) {
    const int num_bin_in_group = offsets_[group_index + 1] - offsets_[group_index];
    if (num_bin_in_group <= 16) {
      ++num_groups_in_cur_bit_state;
    } else if (num_bin_in_group <= 256) {
      if (cur_bit_state == BIT_STATE_4) {
        if (num_groups_in_cur_bit_state >= num_dense_col_ / 3) {
          const int res = num_groups_in_cur_bit_state % 2;
          if (res < num_groups_in_cur_bit_state) {
            bit_states_.push_back(BIT_STATE_4);
            num_groups_in_cur_bit_state -= res;
            num_bit_state_groups_.push_back(num_groups_in_cur_bit_state);
            num_groups_in_cur_bit_state = res;
            has_bit_state_4_ = true;
          }
        }
      }
      cur_bit_state = BIT_STATE_8;
      ++num_groups_in_cur_bit_state;
    } else {
      Log::Fatal("invalid bin size for feature group");
    }
  }
  if (bit_states_.empty() || bit_states_.back() != cur_bit_state) {
    if (cur_bit_state == BIT_STATE_4) {
      const int res = num_groups_in_cur_bit_state % 2;
      if (res < num_groups_in_cur_bit_state) {
        bit_states_.push_back(cur_bit_state);
        num_groups_in_cur_bit_state -= res;
        num_bit_state_groups_.push_back(num_groups_in_cur_bit_state);
        num_groups_in_cur_bit_state = res;
        has_bit_state_4_ = true;
      }
      if (res > 0) {
        cur_bit_state = BIT_STATE_8;
      }
    }
    if (num_groups_in_cur_bit_state > 0) {
      bit_states_.push_back(cur_bit_state);
      num_bit_state_groups_.push_back(num_groups_in_cur_bit_state);
      CHECK_EQ(cur_bit_state, BIT_STATE_8);
      has_bit_state_8_ = true;
    }
  }
  if (num_dense_col_ < num_feature_) {
    has_multi_val_ = true;
    multi_val_offset_ = offsets_[num_dense_col_];
  }
  if (num_bin_ > 256) {
    need_offset_ = true;
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
}

template <>
MultiValMixBin<uint16_t>::MultiValMixBin(data_size_t num_data, int num_bin, int num_feature,
  const std::vector<uint32_t>& offsets, const double estimate_element_per_row,
  const int num_dense_col)
  : num_data_(num_data), num_bin_(num_bin), num_feature_(num_feature), num_dense_col_(num_dense_col),
    offsets_(offsets), estimate_element_per_row_(estimate_element_per_row) {
  bit_states_.clear();
  num_bit_state_groups_.clear();
  BIT_STATE cur_bit_state = BIT_STATE_4;
  int num_groups_in_cur_bit_state = 0;
  for (int group_index = 0; group_index < num_dense_col_; ++group_index) {
    const int num_bin_in_group = offsets_[group_index + 1] - offsets_[group_index];
    if (num_bin_in_group <= 16) {
      ++num_groups_in_cur_bit_state;
    } else if (num_bin_in_group <= 256) {
      if (cur_bit_state == BIT_STATE_4) {
        if (num_groups_in_cur_bit_state >= num_dense_col_ / 3) {
          const int res = num_groups_in_cur_bit_state % 4;
          if (res < num_groups_in_cur_bit_state) {
            bit_states_.push_back(BIT_STATE_4);
            num_groups_in_cur_bit_state -= res;
            num_bit_state_groups_.push_back(num_groups_in_cur_bit_state);
            num_groups_in_cur_bit_state = res;
            has_bit_state_4_ = true;
          }
        }
      }
      cur_bit_state = BIT_STATE_8;
      ++num_groups_in_cur_bit_state;
    } else if (num_bin_in_group <= 65536) {
      if (cur_bit_state == BIT_STATE_8) {
        if (num_groups_in_cur_bit_state >= num_dense_col_ / 3) {
          const int res = num_groups_in_cur_bit_state % 2;
          if (res < num_groups_in_cur_bit_state) {
            bit_states_.push_back(BIT_STATE_8);
            num_groups_in_cur_bit_state -= res;
            num_bit_state_groups_.push_back(num_groups_in_cur_bit_state);
            num_groups_in_cur_bit_state = res;
            has_bit_state_8_ = true;
          }
        }
      } else if (cur_bit_state == BIT_STATE_4) {
        if (num_groups_in_cur_bit_state >= num_dense_col_ / 3) {
          const int res = num_groups_in_cur_bit_state % 4;
          if (res < num_groups_in_cur_bit_state) {
            bit_states_.push_back(BIT_STATE_4);
            num_groups_in_cur_bit_state -= res;
            num_bit_state_groups_.push_back(num_groups_in_cur_bit_state);
            num_groups_in_cur_bit_state = res;
            has_bit_state_4_ = true;
          }
        }
      }
      cur_bit_state = BIT_STATE_16;
      ++num_groups_in_cur_bit_state;
    } else {
      Log::Fatal("invalid bin size for feature group");
    }
  }
  if (bit_states_.empty() || bit_states_.back() != cur_bit_state) {
    if (cur_bit_state == BIT_STATE_4) {
      const int res = num_groups_in_cur_bit_state % 4;
      if (res < num_groups_in_cur_bit_state) {
        bit_states_.push_back(cur_bit_state);
        num_groups_in_cur_bit_state -= res;
        num_bit_state_groups_.push_back(num_groups_in_cur_bit_state);
        num_groups_in_cur_bit_state = res;
        has_bit_state_4_ = true;
      }
      if (res > 0) {
        cur_bit_state = BIT_STATE_16;
      }
    } else if (cur_bit_state == BIT_STATE_8) {
      const int res = num_groups_in_cur_bit_state % 2;
      if (res < num_groups_in_cur_bit_state) {
        bit_states_.push_back(cur_bit_state);
        num_groups_in_cur_bit_state -= res;
        num_bit_state_groups_.push_back(num_groups_in_cur_bit_state);
        num_groups_in_cur_bit_state = res;
        has_bit_state_8_ = true;
      }
      if (res > 0) {
        cur_bit_state = BIT_STATE_16;
      }
    }
    if (num_groups_in_cur_bit_state > 0) {
      bit_states_.push_back(cur_bit_state);
      num_bit_state_groups_.push_back(num_groups_in_cur_bit_state);
      CHECK_EQ(cur_bit_state, BIT_STATE_16);
      has_bit_state_16_ = true;
    }
  }
  if (num_dense_col_ < num_feature_) {
    has_multi_val_ = true;
    multi_val_offset_ = offsets_[num_dense_col_];
  }
  if (num_bin_ > 65536) {
    need_offset_ = true;
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
}

} //  namespace LightGBM
