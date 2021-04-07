/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */

#include <LightGBM/train_share_states.h>

namespace LightGBM {

MultiValBinWrapper::MultiValBinWrapper(MultiValBin* bin, data_size_t num_data,
  const std::vector<int>& feature_groups_contained):
    feature_groups_contained_(feature_groups_contained) {
  num_threads_ = OMP_NUM_THREADS();
  max_block_size_ = num_data;
  num_data_ = num_data;
  multi_val_bin_.reset(bin);
  if (bin == nullptr) {
    return;
  }
  num_bin_ = bin->num_bin();
  num_bin_aligned_ = (num_bin_ + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
}

void MultiValBinWrapper::InitTrain(const std::vector<int>& group_feature_start,
  const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
  const std::vector<int8_t>& is_feature_used,
  const data_size_t* bagging_use_indices,
  data_size_t bagging_indices_cnt) {
  is_use_subcol_ = false;
  if (multi_val_bin_ == nullptr) {
    return;
  }
  CopyMultiValBinSubset(group_feature_start, feature_groups,
    is_feature_used, bagging_use_indices, bagging_indices_cnt);
  const auto cur_multi_val_bin = (is_use_subcol_ || is_use_subrow_)
        ? multi_val_bin_subset_.get()
        : multi_val_bin_.get();
  if (cur_multi_val_bin != nullptr) {
    num_bin_ = cur_multi_val_bin->num_bin();
    num_bin_aligned_ = (num_bin_ + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
    min_block_size_ = std::min<int>(static_cast<int>(0.3f * num_bin_ /
      cur_multi_val_bin->num_element_per_row()) + 1, 1024);
  }
}

void MultiValBinWrapper::HistMove(const std::vector<hist_t,
  Common::AlignmentAllocator<hist_t, kAlignedSize>>& hist_buf) {
  if (!is_use_subcol_) {
    return;
  }
  const hist_t* src = hist_buf.data() + hist_buf.size() -
    2 * static_cast<size_t>(num_bin_aligned_);
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < static_cast<int>(hist_move_src_.size()); ++i) {
    std::copy_n(src + hist_move_src_[i], hist_move_size_[i],
                origin_hist_data_ + hist_move_dest_[i]);
  }
}

void MultiValBinWrapper::IntHistMove(const std::vector<int_hist_t,
  Common::AlignmentAllocator<int_hist_t, kAlignedSize>>& hist_buf) {
  if (!is_use_subcol_) {
    const int_hist_t* src = hist_buf.data();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_bin_; ++i) {
      origin_hist_data_[2 * i] = src[2 * i + 1] * grad_scale;
      origin_hist_data_[2 * i + 1] = src[2 * i] * hess_scale;
    }
  } else {
    const int_hist_t* src = hist_buf.data();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(hist_move_src_.size()); ++i) {
      const int_hist_t* src_ptr = src + hist_move_src_[i];
      hist_t* dst_ptr = origin_hist_data_ + hist_move_dest_[i];
      for (int j = 0; j < static_cast<int>(hist_move_size_[i]) / 2; ++j) {
        dst_ptr[2 * j] = grad_scale * src_ptr[2 * j + 1];
        dst_ptr[2 * j + 1] = hess_scale * src_ptr[2 * j];
      }
    }
  }
}

void MultiValBinWrapper::HistMerge(std::vector<hist_t,
  Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf) {
  int n_bin_block = 1;
  int bin_block_size = num_bin_;
  Threading::BlockInfo<data_size_t>(num_threads_, num_bin_, 512, &n_bin_block,
                                  &bin_block_size);
  hist_t* dst = origin_hist_data_;
  if (is_use_subcol_) {
    dst = hist_buf->data() + hist_buf->size() - 2 * static_cast<size_t>(num_bin_aligned_);
  }
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
  for (int t = 0; t < n_bin_block; ++t) {
    const int start = t * bin_block_size;
    const int end = std::min(start + bin_block_size, num_bin_);
    for (int tid = 1; tid < n_data_block_; ++tid) {
      auto src_ptr = hist_buf->data() + static_cast<size_t>(num_bin_aligned_) * 2 * (tid - 1);
      for (int i = start * 2; i < end * 2; ++i) {
        dst[i] += src_ptr[i];
      }
    }
  }
}

void MultiValBinWrapper::IntHistMerge(std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* hist_buf) {
  int n_bin_block = 1;
  int bin_block_size = num_bin_;
  Threading::BlockInfo<data_size_t>(num_threads_, num_bin_, 512, &n_bin_block,
                                  &bin_block_size);
  int_hist_t* dst = hist_buf->data();
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
  for (int t = 0; t < n_bin_block; ++t) {
    const int start = t * bin_block_size;
    const int end = std::min(start + bin_block_size, num_bin_);
    for (int tid = 1; tid < n_data_block_; ++tid) {
      auto src_ptr = hist_buf->data() + static_cast<size_t>(num_bin_aligned_) * 2 * tid;
      for (int i = start * 2; i < end * 2; ++i) {
        dst[i] += src_ptr[i];
      }
    }
  }
}

void MultiValBinWrapper::Int32HistMerge(std::vector<int_buf_hist_t, Common::AlignmentAllocator<int_buf_hist_t, kAlignedSize>>* hist_buf,
  std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* merge_hist_buf) {
  int n_bin_block = 1;
  int bin_block_size = num_bin_;
  Threading::BlockInfo<data_size_t>(num_threads_, num_bin_, 512, &n_bin_block,
                                  &bin_block_size);
  int_hist_t* dst = merge_hist_buf->data();
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int i = 0; i < num_bin_aligned_ * 2; ++i) {
    dst[i] = 0;
  }
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
  for (int t = 0; t < n_bin_block; ++t) {
    const int start = t * bin_block_size;
    const int end = std::min(start + bin_block_size, num_bin_);
    for (int tid = 0; tid < n_data_block_; ++tid) {
      auto src_ptr = hist_buf->data() + static_cast<size_t>(num_bin_aligned_) * 2 * tid;
      for (int i = start * 2; i < end * 2; ++i) {
        dst[i] += src_ptr[i];
      }
    }
  }
}

void MultiValBinWrapper::Int48HistMerge(std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* hist_buf,
  std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* merge_hist_buf) {
  int n_bin_block = 1;
  int bin_block_size = num_bin_;
  Threading::BlockInfo<data_size_t>(num_threads_, num_bin_, 512, &n_bin_block,
                                  &bin_block_size);
  int32_t* dst = reinterpret_cast<int32_t*>(merge_hist_buf->data());
  int16_t* hist_buf_ptr = reinterpret_cast<int16_t*>(hist_buf->data()) + 1;
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int i = 0; i < num_bin_aligned_ * 2; ++i) {
    dst[i] = 0;
  }
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
  for (int t = 0; t < n_bin_block; ++t) {
    const int start = t * bin_block_size;
    const int end = std::min(start + bin_block_size, num_bin_);
    for (int tid = 0; tid < n_data_block_; ++tid) {
      auto src_ptr = hist_buf_ptr + static_cast<size_t>(int48_hist_block_size_) * tid + 3 * start;
      for (int i = start * 2; i < end * 2; i += 2) {
        int64_t gh = *reinterpret_cast<int64_t*>(src_ptr);
        int32_t hess_val = static_cast<int32_t>(gh & 0x00ffffff);
        dst[i] += hess_val;
        int32_t grad_val = static_cast<int32_t>((gh >> 24) & 0x00ffffff);
        const bool should_mask = (grad_val & 0x00800000) > 0;
        if (should_mask) {
          grad_val |= 0xff000000;
        }
        dst[i + 1] += grad_val;
        src_ptr += 3;
      }
    }
  }
}

void MultiValBinWrapper::ResizeHistBuf(std::vector<hist_t,
  Common::AlignmentAllocator<hist_t, kAlignedSize>>* hist_buf,
  MultiValBin* sub_multi_val_bin,
  hist_t* origin_hist_data) {
  num_bin_ = sub_multi_val_bin->num_bin();
  num_bin_aligned_ = (num_bin_ + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
  origin_hist_data_ = origin_hist_data;
  size_t new_buf_size = static_cast<size_t>(n_data_block_) * static_cast<size_t>(num_bin_aligned_) * 2;
  if (hist_buf->size() < new_buf_size) {
    hist_buf->resize(new_buf_size);
  }
}

void MultiValBinWrapper::ResizeIntHistBuf(std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* hist_buf,
  MultiValBin* sub_multi_val_bin,
  hist_t* origin_hist_data) {
  num_bin_ = sub_multi_val_bin->num_bin();
  num_bin_aligned_ = (num_bin_ + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
  origin_hist_data_ = origin_hist_data;
  size_t block_hist_size = static_cast<size_t>(num_bin_aligned_) * 2;
  size_t new_buf_size = static_cast<size_t>(n_data_block_) * block_hist_size;
  if (hist_buf->size() < new_buf_size) {
    hist_buf->resize(new_buf_size);
  }
}

void MultiValBinWrapper::Resize32IntHistBuf(std::vector<int_buf_hist_t, Common::AlignmentAllocator<int_buf_hist_t, kAlignedSize>>* hist_buf,
  std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* merged_hist_buf,
  MultiValBin* sub_multi_val_bin,
  hist_t* origin_hist_data) {
  num_bin_ = sub_multi_val_bin->num_bin();
  num_bin_aligned_ = (num_bin_ + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
  origin_hist_data_ = origin_hist_data;
  size_t block_hist_size = static_cast<size_t>(num_bin_aligned_) * 2;
  size_t new_buf_size = static_cast<size_t>(n_data_block_) * block_hist_size;
  if (hist_buf->size() < new_buf_size) {
    hist_buf->resize(new_buf_size);
  }
  if (merged_hist_buf->size() < block_hist_size) {
    merged_hist_buf->resize(block_hist_size);
  }
}

void MultiValBinWrapper::Resize48IntHistBuf(std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* hist_buf,
  std::vector<int_hist_t, Common::AlignmentAllocator<int_hist_t, kAlignedSize>>* merged_hist_buf,
  MultiValBin* sub_multi_val_bin,
  hist_t* origin_hist_data) {
  num_bin_ = sub_multi_val_bin->num_bin();
  num_bin_aligned_ = (num_bin_ + kAlignedSize - 1) / kAlignedSize * kAlignedSize;
  origin_hist_data_ = origin_hist_data;
  size_t block_hist_size = static_cast<size_t>(num_bin_aligned_) / 2 * 3 + 3;
  if (!block_hist_size % kAlignedSize == 0) {
    block_hist_size += (block_hist_size - (block_hist_size % kAlignedSize));
  }
  int48_hist_block_size_ = block_hist_size;
  size_t new_buf_size = static_cast<size_t>(n_data_block_) * block_hist_size;
  if (hist_buf->size() < new_buf_size) {
    hist_buf->resize(new_buf_size);
  }
  size_t merged_hist_buf_size = num_bin_aligned_ * 2;
  if (merged_hist_buf->size() < merged_hist_buf_size) {
    merged_hist_buf->resize(merged_hist_buf_size);
  }
}

void MultiValBinWrapper::CopyMultiValBinSubset(
  const std::vector<int>& group_feature_start,
  const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
  const std::vector<int8_t>& is_feature_used,
  const data_size_t* bagging_use_indices,
  data_size_t bagging_indices_cnt) {
  double sum_used_dense_ratio = 0.0;
  double sum_dense_ratio = 0.0;
  int num_used = 0;
  int total = 0;
  std::vector<int> used_feature_index;
  for (int i : feature_groups_contained_) {
    int f_start = group_feature_start[i];
    if (feature_groups[i]->is_multi_val_) {
      for (int j = 0; j < feature_groups[i]->num_feature_; ++j) {
        const auto dense_rate =
            1.0 - feature_groups[i]->bin_mappers_[j]->sparse_rate();
        if (is_feature_used[f_start + j]) {
          ++num_used;
          used_feature_index.push_back(total);
          sum_used_dense_ratio += dense_rate;
        }
        sum_dense_ratio += dense_rate;
        ++total;
      }
    } else {
      bool is_group_used = false;
      double dense_rate = 0;
      for (int j = 0; j < feature_groups[i]->num_feature_; ++j) {
        if (is_feature_used[f_start + j]) {
          is_group_used = true;
        }
        dense_rate += 1.0 - feature_groups[i]->bin_mappers_[j]->sparse_rate();
      }
      if (is_group_used) {
        ++num_used;
        used_feature_index.push_back(total);
        sum_used_dense_ratio += dense_rate;
      }
      sum_dense_ratio += dense_rate;
      ++total;
    }
  }
  const double k_subfeature_threshold = 0.6;
  if (sum_used_dense_ratio >= sum_dense_ratio * k_subfeature_threshold) {
    // only need to copy subset
    if (is_use_subrow_ && !is_subrow_copied_) {
      if (multi_val_bin_subset_ == nullptr) {
        multi_val_bin_subset_.reset(multi_val_bin_->CreateLike(
            bagging_indices_cnt, multi_val_bin_->num_bin(), total,
            multi_val_bin_->num_element_per_row(), multi_val_bin_->offsets()));
      } else {
        multi_val_bin_subset_->ReSize(
            bagging_indices_cnt, multi_val_bin_->num_bin(), total,
            multi_val_bin_->num_element_per_row(), multi_val_bin_->offsets());
      }
      multi_val_bin_subset_->CopySubrow(
          multi_val_bin_.get(), bagging_use_indices,
          bagging_indices_cnt);
      // avoid to copy subset many times
      is_subrow_copied_ = true;
    }
  } else {
    is_use_subcol_ = true;
    std::vector<uint32_t> upper_bound;
    std::vector<uint32_t> lower_bound;
    std::vector<uint32_t> delta;
    std::vector<uint32_t> offsets;
    hist_move_src_.clear();
    hist_move_dest_.clear();
    hist_move_size_.clear();

    const int offset = multi_val_bin_->IsSparse() ? 1 : 0;
    int num_total_bin = offset;
    int new_num_total_bin = offset;
    offsets.push_back(static_cast<uint32_t>(new_num_total_bin));
    for (int i : feature_groups_contained_) {
      int f_start = group_feature_start[i];
      if (feature_groups[i]->is_multi_val_) {
        for (int j = 0; j < feature_groups[i]->num_feature_; ++j) {
          const auto& bin_mapper = feature_groups[i]->bin_mappers_[j];
          int cur_num_bin = bin_mapper->num_bin();
          if (bin_mapper->GetMostFreqBin() == 0) {
            cur_num_bin -= offset;
          }
          num_total_bin += cur_num_bin;
          if (is_feature_used[f_start + j]) {
            new_num_total_bin += cur_num_bin;
            offsets.push_back(static_cast<uint32_t>(new_num_total_bin));
            lower_bound.push_back(num_total_bin - cur_num_bin);
            upper_bound.push_back(num_total_bin);

            hist_move_src_.push_back(
                (new_num_total_bin - cur_num_bin) * 2);
            hist_move_dest_.push_back((num_total_bin - cur_num_bin) *
                                                2);
            hist_move_size_.push_back(cur_num_bin * 2);
            delta.push_back(num_total_bin - new_num_total_bin);
          }
        }
      } else {
        bool is_group_used = false;
        for (int j = 0; j < feature_groups[i]->num_feature_; ++j) {
          if (is_feature_used[f_start + j]) {
            is_group_used = true;
            break;
          }
        }
        int cur_num_bin = feature_groups[i]->bin_offsets_.back() - offset;
        num_total_bin += cur_num_bin;
        if (is_group_used) {
          new_num_total_bin += cur_num_bin;
          offsets.push_back(static_cast<uint32_t>(new_num_total_bin));
          lower_bound.push_back(num_total_bin - cur_num_bin);
          upper_bound.push_back(num_total_bin);

          hist_move_src_.push_back(
              (new_num_total_bin - cur_num_bin) * 2);
          hist_move_dest_.push_back((num_total_bin - cur_num_bin) *
                                              2);
          hist_move_size_.push_back(cur_num_bin * 2);
          delta.push_back(num_total_bin - new_num_total_bin);
        }
      }
    }
    // avoid out of range
    lower_bound.push_back(num_total_bin);
    upper_bound.push_back(num_total_bin);
    data_size_t num_data = is_use_subrow_ ? bagging_indices_cnt : num_data_;
    if (multi_val_bin_subset_ == nullptr) {
      multi_val_bin_subset_.reset(multi_val_bin_->CreateLike(
          num_data, new_num_total_bin, num_used, sum_used_dense_ratio, offsets));
    } else {
      multi_val_bin_subset_->ReSize(num_data, new_num_total_bin,
                                              num_used, sum_used_dense_ratio, offsets);
    }
    if (is_use_subrow_) {
      multi_val_bin_subset_->CopySubrowAndSubcol(
          multi_val_bin_.get(), bagging_use_indices,
          bagging_indices_cnt, used_feature_index, lower_bound,
          upper_bound, delta);
      // may need to recopy subset
      is_subrow_copied_ = false;
    } else {
      multi_val_bin_subset_->CopySubcol(
          multi_val_bin_.get(), used_feature_index, lower_bound, upper_bound, delta);
    }
  }
}

void TrainingShareStates::CalcBinOffsets(const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
  std::vector<uint32_t>* offsets, bool is_col_wise) {
  offsets->clear();
  feature_hist_offsets_.clear();
  if (is_col_wise) {
    uint32_t cur_num_bin = 0;
    uint32_t hist_cur_num_bin = 0;
    for (int group = 0; group < static_cast<int>(feature_groups.size()); ++group) {
      const std::unique_ptr<FeatureGroup>& feature_group = feature_groups[group];
      if (feature_group->is_multi_val_) {
        if (feature_group->is_dense_multi_val_) {
          for (int i = 0; i < feature_group->num_feature_; ++i) {
            const std::unique_ptr<BinMapper>& bin_mapper = feature_group->bin_mappers_[i];
            if (group == 0 && i == 0 && bin_mapper->GetMostFreqBin() > 0) {
              cur_num_bin += 1;
              hist_cur_num_bin += 1;
            }
            offsets->push_back(cur_num_bin);
            feature_hist_offsets_.push_back(hist_cur_num_bin);
            int num_bin = bin_mapper->num_bin();
            hist_cur_num_bin += num_bin;
            if (bin_mapper->GetMostFreqBin() == 0) {
              feature_hist_offsets_.back() += 1;
            }
            cur_num_bin += num_bin;
          }
          offsets->push_back(cur_num_bin);
          CHECK(cur_num_bin == feature_group->bin_offsets_.back());
        } else {
          cur_num_bin += 1;
          hist_cur_num_bin += 1;
          for (int i = 0; i < feature_group->num_feature_; ++i) {
            offsets->push_back(cur_num_bin);
            feature_hist_offsets_.push_back(hist_cur_num_bin);
            const std::unique_ptr<BinMapper>& bin_mapper = feature_group->bin_mappers_[i];
            int num_bin = bin_mapper->num_bin();
            if (bin_mapper->GetMostFreqBin() == 0) {
              num_bin -= 1;
            }
            hist_cur_num_bin += num_bin;
            cur_num_bin += num_bin;
          }
          offsets->push_back(cur_num_bin);
          CHECK(cur_num_bin == feature_group->bin_offsets_.back());
        }
      } else {
        for (int i = 0; i < feature_group->num_feature_; ++i) {
          feature_hist_offsets_.push_back(hist_cur_num_bin + feature_group->bin_offsets_[i]);
        }
        hist_cur_num_bin += feature_group->bin_offsets_.back();
      }
    }
    feature_hist_offsets_.push_back(hist_cur_num_bin);
    num_hist_total_bin_ = static_cast<uint64_t>(feature_hist_offsets_.back());
  } else {
    double sum_dense_ratio = 0.0f;
    int ncol = 0;
    for (int gid = 0; gid < static_cast<int>(feature_groups.size()); ++gid) {
      if (feature_groups[gid]->is_multi_val_) {
        ncol += feature_groups[gid]->num_feature_;
      } else {
        ++ncol;
      }
      for (int fid = 0; fid < feature_groups[gid]->num_feature_; ++fid) {
        const auto& bin_mapper = feature_groups[gid]->bin_mappers_[fid];
        sum_dense_ratio += 1.0f - bin_mapper->sparse_rate();
      }
    }
    sum_dense_ratio /= ncol;
    const bool is_sparse_row_wise = (1.0f - sum_dense_ratio) >=
      MultiValBin::multi_val_bin_sparse_threshold ? 1 : 0;
    if (is_sparse_row_wise) {
      int cur_num_bin = 1;
      uint32_t hist_cur_num_bin = 1;
      for (int group = 0; group < static_cast<int>(feature_groups.size()); ++group) {
        const std::unique_ptr<FeatureGroup>& feature_group = feature_groups[group];
        if (feature_group->is_multi_val_) {
          for (int i = 0; i < feature_group->num_feature_; ++i) {
            offsets->push_back(cur_num_bin);
            feature_hist_offsets_.push_back(hist_cur_num_bin);
            const std::unique_ptr<BinMapper>& bin_mapper = feature_group->bin_mappers_[i];
            int num_bin = bin_mapper->num_bin();
            if (bin_mapper->GetMostFreqBin() == 0) {
              num_bin -= 1;
            }
            cur_num_bin += num_bin;
            hist_cur_num_bin += num_bin;
          }
        } else {
          offsets->push_back(cur_num_bin);
          cur_num_bin += feature_group->bin_offsets_.back() - 1;
          for (int i = 0; i < feature_group->num_feature_; ++i) {
            feature_hist_offsets_.push_back(hist_cur_num_bin + feature_group->bin_offsets_[i] - 1);
          }
          hist_cur_num_bin += feature_group->bin_offsets_.back() - 1;
        }
      }
      offsets->push_back(cur_num_bin);
      feature_hist_offsets_.push_back(hist_cur_num_bin);
    } else {
      int cur_num_bin = 0;
      uint32_t hist_cur_num_bin = 0;
      for (int group = 0; group < static_cast<int>(feature_groups.size()); ++group) {
        const std::unique_ptr<FeatureGroup>& feature_group = feature_groups[group];
        if (feature_group->is_multi_val_) {
          for (int i = 0; i < feature_group->num_feature_; ++i) {
            const std::unique_ptr<BinMapper>& bin_mapper = feature_group->bin_mappers_[i];
            if (group == 0 && i == 0 && bin_mapper->GetMostFreqBin() > 0) {
              cur_num_bin += 1;
              hist_cur_num_bin += 1;
            }
            offsets->push_back(cur_num_bin);
            feature_hist_offsets_.push_back(hist_cur_num_bin);
            int num_bin = bin_mapper->num_bin();
            cur_num_bin += num_bin;
            hist_cur_num_bin += num_bin;
            if (bin_mapper->GetMostFreqBin() == 0) {
              feature_hist_offsets_.back() += 1;
            }
          }
        } else {
          offsets->push_back(cur_num_bin);
          cur_num_bin += feature_group->bin_offsets_.back();
          for (int i = 0; i < feature_group->num_feature_; ++i) {
            feature_hist_offsets_.push_back(hist_cur_num_bin + feature_group->bin_offsets_[i]);
          }
          hist_cur_num_bin += feature_group->bin_offsets_.back();
        }
      }
      offsets->push_back(cur_num_bin);
      feature_hist_offsets_.push_back(hist_cur_num_bin);
    }
    num_hist_total_bin_ = static_cast<uint64_t>(feature_hist_offsets_.back());
  }
}

void TrainingShareStates::SetMultiValBin(MultiValBin* bin, data_size_t num_data,
  const std::vector<std::unique_ptr<FeatureGroup>>& feature_groups,
  bool dense_only, bool sparse_only) {
  num_threads = OMP_NUM_THREADS();
  if (bin == nullptr) {
    return;
  }
  std::vector<int> feature_groups_contained;
  for (int group = 0; group < static_cast<int>(feature_groups.size()); ++group) {
    const auto& feature_group = feature_groups[group];
    if (feature_group->is_multi_val_) {
      if (!dense_only) {
        feature_groups_contained.push_back(group);
      }
    } else if (!sparse_only) {
      feature_groups_contained.push_back(group);
    }
  }
  num_total_bin_ += bin->num_bin();
  num_elements_per_row_ += bin->num_element_per_row();
  multi_val_bin_wrapper_.reset(new MultiValBinWrapper(
    bin, num_data, feature_groups_contained));
  total_num_data_ = num_data;
}

void TrainingShareStates::RecoverHistogramsFromInteger(hist_t* hist) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < group_bin_boundaries_.back(); ++i) {
    hist[2 * i] = int_hist_buf_[2 * i + 1] * grad_scale_;
    hist[2 * i + 1] = int_hist_buf_[2 * i] * hess_scale_;
  }
}

int_hist_t* TrainingShareStates::GetIntegerHistogram(int group_id) {
  return merged_int_hist_buf_.data() + group_bin_boundaries_[group_id] * 2;
}

void TrainingShareStates::CalcHistBit(const std::vector<const hist_t*>& parent_hist, 
  const std::vector<const BinMapper*>& feature_bin_mappers,
  double /*small_leaf_sum_gradient*/,
  double small_leaf_sum_hessian, data_size_t small_leaf_num_data,
  double /*large_leaf_sum_gradient*/, double large_leaf_sum_hessian,
  data_size_t large_leaf_num_data, bool is_small_leaf) {
  global_timer.Start("TrainingShareStates::CalcHistBit");
  /*data_size_t max_cnt_per_bin = 0;
  if (is_small_leaf) {
    double sum_hessian = small_leaf_sum_hessian + large_leaf_sum_hessian;
    data_size_t num_data = small_leaf_num_data + large_leaf_num_data;
    double cnt_hess = num_data / sum_hessian;
    std::vector<data_size_t> thread_max_cnt_per_bin(num_threads, 0);
    std::vector<int> thread_max_cnt_feature(num_threads, -1);
    int num_features = static_cast<int>(feature_bin_mappers.size());
    int block_size = (num_features + num_threads - 1) / num_threads;
    #pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for (int tid = 0; tid < num_threads; ++tid) {
      int start = tid * block_size;
      int end = std::min<int>(start + block_size, num_features);
      for (int i = start; i < end; ++i) {
        const hist_t* feature_hist = parent_hist[i];
        const int num_bin = feature_bin_mappers[i]->num_bin();
        const int most_freq_bin = feature_bin_mappers[i]->GetMostFreqBin();
        const int offset = static_cast<int>(most_freq_bin == 0);
        for (int j = 0; j < num_bin - offset; ++j) {
          const hist_t hess = feature_hist[2 * j + 1];
          if (offset == 0 && j == most_freq_bin) continue;
          const data_size_t est_cnt = static_cast<data_size_t>(hess * cnt_hess);
          if (est_cnt > thread_max_cnt_per_bin[tid]) {
            thread_max_cnt_per_bin[tid] = est_cnt;
            thread_max_cnt_feature[tid] = i;
          }
        }
      }
    }
    for (int tid = 0; tid < num_threads; ++tid) {
      if (thread_max_cnt_per_bin[tid] > max_cnt_per_bin) {
        max_cnt_per_bin = thread_max_cnt_per_bin[tid];
      }
    }
    max_cnt_per_bin_est_ = max_cnt_per_bin;
    if (max_cnt_per_bin <= 200) {
      hist_bit_ = BIT16_HIST;
    } else if (max_cnt_per_bin <= 60000) {
      hist_bit_ = BIT24_HIST;
    } else {
      hist_bit_ = BIT32_HIST;
    }
  } else {*/
    hist_bit_ = BIT32_HIST;
  //}
  global_timer.Stop("TrainingShareStates::CalcHistBit");
}

void MultiValBinWrapper::RefreshHistBit() {
  if (data_block_size_ <= 200) {
    hist_bit_ = std::min(hist_bit_, BIT16_HIST);
  } else if (data_block_size_ <= 60000) {
    hist_bit_ = std::min(hist_bit_, BIT24_HIST);
  }

  int max_cnt_per_bin_per_block_est_ = (max_cnt_per_bin_est_ + n_data_block_ - 1) / n_data_block_;
  if (max_cnt_per_bin_per_block_est_ <= 200) {
    hist_bit_ = std::min(hist_bit_, BIT16_HIST);
  } else if (max_cnt_per_bin_per_block_est_ <= 60000) {
    hist_bit_ = std::min(hist_bit_, BIT24_HIST);
  }
  if (hist_bit_ == BIT32_HIST) {
    //Log::Warning("BIT32_HIST");
  } else if (hist_bit_ == BIT24_HIST) {
    //Log::Warning("BIT24_HIST");
  } else if (hist_bit_ == BIT16_HIST) {
    //Log::Warning("BIT16_HIST");
  } else {
    Log::Fatal("Unknown hist_bit_");
  }
}

void MultiValBinWrapper::SetHistBitInfo(HIST_BIT hist_bit, int max_cnt_per_bin_est) {
  hist_bit_ = hist_bit;
  max_cnt_per_bin_est_ = max_cnt_per_bin_est;
}

}  // namespace LightGBM
