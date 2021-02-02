/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "multi_val_mix_bin.hpp"

namespace LightGBM {

template <typename VAL_T>
template <bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32>
void MultiValMixBin<VAL_T>::CalcBitEnds() {
	int cur_bit_end = 0;
	int cur_num_state_pos = 0;
	bit4_end_ = 0;
	bit8_end_ = 0;
	bit16_end_ = 0;
	bit32_end_ = 0;
	if (HAS_BIT_STATE_4) {
		bit4_end_ = num_bit_state_groups_[cur_num_state_pos];
		cur_bit_end += bit4_end_;
		++cur_num_state_pos;
	}
	if (HAS_BIT_STATE_8) {
		bit8_end_ = num_bit_state_groups_[cur_num_state_pos] + cur_bit_end;
		cur_bit_end += bit8_end_;
		++cur_num_state_pos;
	}
	if (HAS_BIT_STATE_16) {
		bit16_end_ = num_bit_state_groups_[cur_num_state_pos] + cur_bit_end;
		cur_bit_end += bit16_end_;
		++cur_num_state_pos;
	}
	if (HAS_BIT_STATE_32) {
		bit32_end_ = num_bit_state_groups_[cur_num_state_pos] + cur_bit_end;
		cur_bit_end += bit32_end_;
		++cur_num_state_pos;
	}
}

template <>
template <bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32>
void MultiValMixBin<uint8_t>::CalcBitBoundaries() {
	CalcBitEnds<HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, HAS_BIT_STATE_32>();
  for (int i = 0; i < static_cast<int>(num_bit_state_groups_.size()); ++i) {
    const int num_groups_in_bit_state = num_bit_state_groups_[i];
    const auto bit_state = bit_states_[i];
    if (bit_state == BIT_STATE_4) {
			bit4_start_ = 0;
			bit4_end_trim_ = bit4_end_ / 2 * 2;
    } else if (bit_state == BIT_STATE_8) {
			bit8_end_trim_ = bit8_end_;
    } else {
      Log::Fatal("Too large bit_state encountered in 8 bit multi val mix bin.");
    }
  }
}

template <>
template <bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32>
void MultiValMixBin<uint16_t>::CalcBitBoundaries() {
  for (int i = 0; i < static_cast<int>(num_bit_state_groups_.size()); ++i) {
    const int num_groups_in_bit_state = num_bit_state_groups_[i];
    const auto bit_state = bit_states_[i];
    if (bit_state == BIT_STATE_4) {
			bit4_end_trim_ = bit4_end_ / 4 * 4;
    } else if (bit_state == BIT_STATE_8) {
			bit8_end_trim_ = bit8_end_ / 2 * 2;
    } else if (bit_state == BIT_STATE_16) {
			bit16_end_trim_ = bit16_end_;
    } else {
      Log::Fatal("Too large bit_state encountered in 16 bit multi val mix bin.");
    }
  }
	CalcBitEnds<HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, HAS_BIT_STATE_32>();
}

template <>
template <bool HAS_BIT_STATE_4, bool HAS_BIT_STATE_8, bool HAS_BIT_STATE_16, bool HAS_BIT_STATE_32>
void MultiValMixBin<uint32_t>::CalcBitBoundaries() {
  for (int i = 0; i < static_cast<int>(num_bit_state_groups_.size()); ++i) {
    const int num_groups_in_bit_state = num_bit_state_groups_[i];
    const auto bit_state = bit_states_[i];
    if (bit_state == BIT_STATE_4) {
			bit4_end_trim_ = bit4_end_ / 8 * 8;
    } else if (bit_state == BIT_STATE_8) {

    } else if (bit_state == BIT_STATE_16) {
    } else if (bit_state == BIT_STATE_32) {
    } else {
      Log::Fatal("Too large bit_state encountered in 32 bit multi val mix bin.");
    }
  }
	CalcBitEnds<HAS_BIT_STATE_4, HAS_BIT_STATE_8, HAS_BIT_STATE_16, HAS_BIT_STATE_32>();
}

} //  namespace LightGBM