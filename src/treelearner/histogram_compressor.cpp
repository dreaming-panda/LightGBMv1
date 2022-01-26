/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include "histogram_compressor.hpp"
#include <random>

namespace LightGBM {

HistogramCompressor::HistogramCompressor(const int num_threads) {
  num_threads_ = num_threads > 0 ? num_threads : OMP_NUM_THREADS();
  thread_first_bits_.resize(num_threads_, 0);
  thread_first_.resize(num_threads_, 0);
}

template <typename S_HIST_T, typename U_HIST_T>
void HistogramCompressor::Compress(const S_HIST_T* in_buffer, uint8_t* out_buffer, uint8_t* out_bits_buffer, data_size_t num_bin, uint64_t* thread_total_half_bytes_offset) {
  const data_size_t block_size = (num_bin + num_threads_ - 1) / num_threads_;
  const uint64_t total_size_out_bits_buffer = (num_bin * 2 + 3) / 4;
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (uint64_t i = 0; i < total_size_out_bits_buffer; ++i) {
    out_bits_buffer[i] = 0;
  }
  thread_total_half_bytes_offset[0] = 0;
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const data_size_t start = thread_index * block_size;
    const data_size_t end = std::min(start + block_size, num_bin);
    const uint64_t thread_total_bytes = ComputeThreadHalfBytes<S_HIST_T, U_HIST_T>(
      in_buffer,
      out_bits_buffer,
      &thread_first_bits_[thread_index],
      start, end);
    thread_total_half_bytes_offset[thread_index + 1] = thread_total_bytes;
  }
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    Log::Warning("thread_total_half_bytes_offset[%d] = %ld", thread_index, thread_total_half_bytes_offset[thread_index]);
  }
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const data_size_t start = thread_index * block_size;
    if (start % 2 != 0) {
      out_bits_buffer[start / 2] |= thread_first_bits_[thread_index];
    }
    thread_total_half_bytes_offset[thread_index + 1] += thread_total_half_bytes_offset[thread_index];
  }
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    Log::Warning("before write compress thread %d, thread_total_half_bytes_offset = %ld", thread_index, thread_total_half_bytes_offset[thread_index]);
  }
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const data_size_t start = thread_index * block_size;
    const data_size_t end = std::min(start + block_size, num_bin);
    WriteThreadCompressedData<S_HIST_T, U_HIST_T>(in_buffer,
      out_bits_buffer, out_buffer,
      &thread_first_[thread_index],
      start, end, thread_total_half_bytes_offset[thread_index]);
  }
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    Log::Warning("after write compress thread %d, thread_total_half_bytes_offset = %ld", thread_index, thread_total_half_bytes_offset[thread_index]);
  }
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const uint64_t cur_half_bytes = thread_total_half_bytes_offset[thread_index];
    const uint64_t pos = cur_half_bytes / 2;
    const uint8_t offset = cur_half_bytes % 2;
    if (offset == 1) {
      out_buffer[pos] |= thread_first_[thread_index];
    }
  }
}

template <typename S_HIST_T, typename U_HIST_T>
uint64_t HistogramCompressor::ComputeThreadHalfBytes(
  const S_HIST_T* in_buffer,
  uint8_t* out_bits_buffer,
  uint8_t* thread_first_bits_buffer,
  data_size_t start_bin,
  data_size_t end_bin) {
  int64_t prev_hess =  static_cast<int64_t>(start_bin == 0 ? 0 : static_cast<U_HIST_T>(in_buffer[(start_bin - 1) << 1]));
  int64_t prev_grad =  static_cast<int64_t>(start_bin == 0 ? 0 : in_buffer[((start_bin - 1) << 1) + 1]);
  uint64_t total_half_bytes = 0;
  data_size_t bin = start_bin;
  *thread_first_bits_buffer = 0;
  for (; bin < (start_bin + 1) / 2 * 2; ++bin) {
    const data_size_t bin_offset = (bin << 1);
    const int64_t hess = static_cast<int64_t>(static_cast<U_HIST_T>(in_buffer[bin_offset]));
    const int64_t hess_diff = hess - prev_hess;
    const uint8_t hess_offset = (bin_offset % 4) << 1;
    const int64_t grad = static_cast<int64_t>(in_buffer[bin_offset + 1]);
    const int64_t grad_diff = grad - prev_grad;
    const uint8_t grad_offset = ((bin_offset + 1) % 4) << 1;
    if (hess_diff >= -8 && hess_diff < 8) {
      total_half_bytes += 1;
      Log::Warning("0 bin %d, writing hess_diff = %ld, hess_bits = %d", bin, hess_diff, 0);
    } else if (hess_diff >= -128 && hess_diff < 128) {
      total_half_bytes += 2;
      (*thread_first_bits_buffer) |= (0x01 << hess_offset);
      Log::Warning("1 bin %d, writing hess_diff = %ld, hess_bits = %d, pos = %d, offset = %d", bin, hess_diff, 1, -1, hess_offset);
    } else if (hess_diff >= -32768 && hess_diff < 32768) {
      total_half_bytes += 4;
      (*thread_first_bits_buffer) |= (0x02 << hess_offset);
      Log::Warning("2 bin %d, writing hess_diff = %ld, hess_bits = %d, pos = %d, offset = %d", bin, hess_diff, 2, -1, hess_offset);
    } else {
      total_half_bytes += 8;
      (*thread_first_bits_buffer) |= (0x03 << hess_offset);
      Log::Warning("3 bin %d, writing hess_diff = %ld, hess_bits = %d, pos = %d, offset = %d", bin, hess_diff, 3, -1, hess_offset);
    }
    if (grad_diff >= -8 && grad_diff < 8) {
      total_half_bytes += 1;
      Log::Warning("4 bin %d, writing grad_diff = %ld, grad_bits = %d", bin, grad_diff, 0);
    } else if (grad_diff >= -128 && grad_diff < 128) {
      total_half_bytes += 2;
      (*thread_first_bits_buffer) |= (0x01 << grad_offset);
      Log::Warning("5 bin %d, writing grad_diff = %ld, grad_bits = %d, pos = %d, offset = %d", bin, grad_diff, 1, -1, grad_offset);
    } else if (grad_diff >= -32768 && grad_diff < 32768) {
      total_half_bytes += 4;
      (*thread_first_bits_buffer) |= (0x02 << grad_offset);
      Log::Warning("6 bin %d, writing grad_diff = %ld, grad_bits = %d, pos = %d, offset = %d", bin, grad_diff, 2, -1, grad_offset);
    } else {
      total_half_bytes += 8;
      (*thread_first_bits_buffer) |= (0x03 << grad_offset);
      Log::Warning("7 bin %d, writing grad_diff = %ld, grad_bits = %d, pos = %d, offset = %d", bin, grad_diff, 3, -1, grad_offset);
    }
    prev_grad = grad;
    prev_hess = hess;
  }
  for (; bin < end_bin; ++bin) {
    const data_size_t bin_offset = (bin << 1);
    const int64_t hess = static_cast<int64_t>(static_cast<U_HIST_T>(in_buffer[bin_offset]));
    const int64_t hess_diff = hess - prev_hess;
    const uint64_t hess_pos = (bin_offset / 4);
    const uint8_t hess_offset = (bin_offset % 4) << 1;
    const int64_t grad = static_cast<int64_t>(in_buffer[bin_offset + 1]);
    const int64_t grad_diff = grad - prev_grad;
    const uint64_t grad_pos = ((bin_offset + 1) / 4);
    const uint8_t grad_offset = ((bin_offset + 1) % 4) << 1;
    if (hess_diff >= -8 && hess_diff < 8) {
      total_half_bytes += 1;
      Log::Warning("8 bin %d, writing hess_diff = %ld, hess_bits = %d", bin, hess_diff, 0);
    } else if (hess_diff >= -128 && hess_diff < 128) {
      total_half_bytes += 2;
      out_bits_buffer[hess_pos] |= (0x01 << hess_offset);
      Log::Warning("9 bin %d, writing hess_diff = %ld, hess_bits = %d, pos = %d, offset = %d", bin, hess_diff, 1, hess_pos, hess_offset);
    } else if (hess_diff >= -32768 && hess_diff < 32768) {
      total_half_bytes += 4;
      out_bits_buffer[hess_pos] |= (0x02 << hess_offset);
      Log::Warning("10 bin %d, writing hess_diff = %ld, hess_bits = %d, pos = %d, offset = %d", bin, hess_diff, 2, hess_pos, hess_offset);
    } else {
      total_half_bytes += 8;
      out_bits_buffer[hess_pos] |= (0x03 << hess_offset);
      Log::Warning("11 bin %d, writing hess_diff = %ld, hess_bits = %d, pos = %d, offset = %d", bin, hess_diff, 3, hess_pos, hess_offset);
    }
    if (grad_diff >= -8 && grad_diff < 8) {
      total_half_bytes += 1;
      Log::Warning("12 bin %d, writing grad_diff = %ld, grad_bits = %d", bin, grad_diff, 0);
    } else if (grad_diff >= -128 && grad_diff < 128) {
      total_half_bytes += 2;
      out_bits_buffer[grad_pos] |= (0x01 << grad_offset);
      Log::Warning("13 bin %d, writing grad_diff = %ld, grad_bits = %d, pos = %d, offset = %d", bin, grad_diff, 1, grad_pos, grad_offset);
    } else if (grad_diff >= -32768 && grad_diff < 32768) {
      total_half_bytes += 4;
      out_bits_buffer[grad_pos] |= (0x02 << grad_offset);
      Log::Warning("14 bin %d, writing grad_diff = %ld, grad_bits = %d, pos = %d, offset = %d", bin, grad_diff, 2, grad_pos, grad_offset);
    } else {
      total_half_bytes += 8;
      out_bits_buffer[grad_pos] |= (0x03 << grad_offset);
      Log::Warning("15 bin %d, writing grad_diff = %ld, grad_bits = %d, pos = %d, offset = %d", bin, grad_diff, 3, grad_pos, grad_offset);
    }
    prev_grad = grad;
    prev_hess = hess;
  }
  return total_half_bytes;
}

template <typename S_HIST_T, typename U_HIST_T>
void HistogramCompressor::WriteThreadCompressedData(const S_HIST_T* in_buffer, const uint8_t* bits_buffer,
  uint8_t* out_buffer,
  uint8_t* thread_first_buffer,
  data_size_t start_bin,
  data_size_t end_bin,
  uint64_t thread_start_half_bytes) {
  int64_t prev_hess = (start_bin == 0 ? 0 : static_cast<U_HIST_T>(in_buffer[((start_bin - 1) << 1)]));
  int64_t prev_grad = (start_bin == 0 ? 0 : in_buffer[((start_bin - 1) << 1) + 1]);

  uint64_t cur_half_bytes = thread_start_half_bytes;
  Log::Warning("thread_start_half_bytes = %ld", thread_start_half_bytes);
  *thread_first_buffer = 0;
  data_size_t bin = start_bin;
  if (thread_start_half_bytes % 2 == 1) {
    const data_size_t bin_offset = (bin << 1);
    const uint64_t hess_bits_pos = bin_offset / 4;
    const uint8_t hess_bits_offset = (bin_offset % 4) << 1;
    const uint8_t hess_bits = (bits_buffer[hess_bits_pos] >> hess_bits_offset) & 0x03;
    const int64_t hess = static_cast<int64_t>(static_cast<U_HIST_T>(in_buffer[bin_offset]));
    const int64_t hess_diff = hess - prev_hess;
    const uint64_t grad_bits_pos = (bin_offset + 1) / 4;
    const uint8_t grad_bits_offset = ((bin_offset + 1) % 4) << 1;
    const uint8_t grad_bits = (bits_buffer[grad_bits_pos] >> grad_bits_offset) & 0x03;
    const int64_t grad = static_cast<int64_t>(in_buffer[bin_offset + 1]);
    const int64_t grad_diff = grad - prev_grad;
    prev_grad = grad;
    prev_hess = hess;
    const uint64_t hess_pos = cur_half_bytes / 2;
    Log::Warning("bin %d, bits_buffer = %d, grad_bits_pos = %ld, grad_bits_offset = %d, hess_bits_pos = %ld, hess_bits_offset = %d", bin, bits_buffer[grad_bits_pos], grad_bits_pos, grad_bits_offset, hess_bits_pos, hess_bits_offset);
    if (hess_bits == 0) {
      Log::Warning("0 bin %d, hess_diff = %ld, hess_bits = %d", bin, hess_diff, hess_bits);
      (*thread_first_buffer) |= static_cast<uint8_t>(hess_diff << 4);
      Log::Warning("2 compressed hess = %d", (static_cast<uint8_t>(hess_diff << 4)));
      ++cur_half_bytes;
    } else if (hess_bits == 1) {
      Log::Warning("1 bin %d, hess_diff = %ld, hess_bits = %d", bin, hess_diff, hess_bits);
      (*thread_first_buffer) |= static_cast<uint8_t>(hess_diff << 4);
      out_buffer[hess_pos + 1] |= (static_cast<uint8_t>(hess_diff >> 4) & 0x0f);
      cur_half_bytes += 2;
    } else if (hess_bits == 2) {
      Log::Warning("2 bin %d, hess_diff = %ld, hess_bits = %d", bin, hess_diff, hess_bits);
      (*thread_first_buffer) |= static_cast<uint8_t>(hess_diff << 4);
      out_buffer[hess_pos + 1] = static_cast<uint8_t>(hess_diff >> 4);
      out_buffer[hess_pos + 2] |= (static_cast<uint8_t>(hess_diff >> 12) & 0x0f);
      cur_half_bytes += 4;
    } else {
      Log::Warning("3 bin %d, hess_diff = %ld, hess_bits = %d", bin, hess_diff, hess_bits);
      (*thread_first_buffer) |= static_cast<uint8_t>(hess_diff << 4);
      out_buffer[hess_pos + 1] = static_cast<uint8_t>(hess_diff >> 4);
      out_buffer[hess_pos + 2] = static_cast<uint8_t>(hess_diff >> 12);
      out_buffer[hess_pos + 3] = static_cast<uint8_t>(hess_diff >> 20);
      out_buffer[hess_pos + 4] |= (static_cast<uint8_t>(hess_diff >> 28) & 0x0f);
      cur_half_bytes += 8;
    }
    const uint64_t grad_pos = cur_half_bytes / 2;
    const uint8_t grad_offset = cur_half_bytes % 2;
    if (grad_offset == 0) {
      if (grad_bits == 0) {
      Log::Warning("bin %d, grad_diff = %ld, grad_bits = %d", bin, grad_diff, grad_bits);
        out_buffer[grad_pos] |= (static_cast<uint8_t>(grad_diff) & 0x0f);
        Log::Warning("4 bin %d, grad diff compressed = %d", static_cast<uint8_t>(grad_diff) & 0x0f);
        ++cur_half_bytes;
      } else if (grad_bits == 1) {
      Log::Warning("5 bin %d, grad_diff = %ld, grad_bits = %d", bin, grad_diff, grad_bits);
        out_buffer[grad_pos] = static_cast<uint8_t>(grad_diff);
        cur_half_bytes += 2;
      } else if (grad_bits == 2) {
      Log::Warning("6 bin %d, grad_diff = %ld, grad_bits = %d", bin, grad_diff, grad_bits);
        out_buffer[grad_pos] = static_cast<uint8_t>(grad_diff);
        out_buffer[grad_pos + 1] = static_cast<uint8_t>(grad_diff >> 8);
        cur_half_bytes += 4;
      } else if (grad_bits == 3) {
      Log::Warning("7 bin %d, grad_diff = %ld, grad_bits = %d", bin, grad_diff, grad_bits);
        out_buffer[grad_pos] = static_cast<uint8_t>(grad_diff);
        out_buffer[grad_pos + 1] = static_cast<uint8_t>(grad_diff >> 8);
        out_buffer[grad_pos + 2] = static_cast<uint8_t>(grad_diff >> 16);
        out_buffer[grad_pos + 3] = static_cast<uint8_t>(grad_diff >> 24);
        cur_half_bytes += 8;
      }
    } else {
      if (grad_bits == 0) {
      Log::Warning("bin %d, grad_diff = %ld, grad_bits = %d", bin, grad_diff, grad_bits);
        out_buffer[grad_pos] |= static_cast<uint8_t>(grad_diff << 4);
        Log::Warning("8 bin %d, grad diff compressed = %d", static_cast<uint8_t>(grad_diff << 4));
        ++cur_half_bytes;
      } else if (grad_bits == 1) {
      Log::Warning("9 bin %d, grad_diff = %ld, grad_bits = %d", bin, grad_diff, grad_bits);
        out_buffer[grad_pos] |= static_cast<uint8_t>(grad_diff << 4);
        out_buffer[grad_pos + 1] |= (static_cast<uint8_t>(grad_diff >> 4) & 0x0f);
        cur_half_bytes += 2;
      } else if (grad_bits == 2) {
      Log::Warning("10 bin %d, grad_diff = %ld, grad_bits = %d", bin, grad_diff, grad_bits);
        out_buffer[grad_pos] |= static_cast<uint8_t>(grad_diff << 4);
        out_buffer[grad_pos + 1] = static_cast<uint8_t>(grad_diff >> 4);
        out_buffer[grad_pos + 2] |= (static_cast<uint8_t>(grad_diff >> 12) & 0x0f);
        cur_half_bytes += 4;
      } else if (grad_bits == 3) {
      Log::Warning("11 bin %d, grad_diff = %ld, grad_bits = %d", bin, grad_diff, grad_bits);
        out_buffer[grad_pos] |= static_cast<uint8_t>(grad_diff << 4);
        out_buffer[grad_pos + 1] = static_cast<uint8_t>(grad_diff >> 4);
        out_buffer[grad_pos + 2] = static_cast<uint8_t>(grad_diff >> 12);
        out_buffer[grad_pos + 3] = static_cast<uint8_t>(grad_diff >> 20);
        out_buffer[grad_pos + 4] |= (static_cast<uint8_t>(grad_diff >> 28) & 0x0f);
        cur_half_bytes += 8;
      }
    }
    ++bin;
  }
  Log::Warning("start_bin = %d, end_bin = %d, bin = %d", start_bin, end_bin, bin);
  for (; bin < end_bin; ++bin) {
    const data_size_t bin_offset = (bin << 1);
    const uint64_t hess_bits_pos = bin_offset / 4;
    const uint8_t hess_bits_offset = (bin_offset % 4) << 1;
    const uint8_t hess_bits = (bits_buffer[hess_bits_pos] >> hess_bits_offset) & 0x03;
    const int64_t hess = static_cast<int64_t>(static_cast<U_HIST_T>(in_buffer[bin_offset]));
    const int64_t hess_diff = hess - prev_hess;
    const uint64_t grad_bits_pos = (bin_offset + 1) / 4;
    const uint8_t grad_bits_offset = ((bin_offset + 1) % 4) << 1;
    const uint8_t grad_bits = (bits_buffer[grad_bits_pos] >> grad_bits_offset) & 0x03;
    const int64_t grad = static_cast<int64_t>(in_buffer[bin_offset + 1]);
    const int64_t grad_diff = grad - prev_grad;
    prev_grad = grad;
    prev_hess = hess;
    const uint64_t hess_pos = cur_half_bytes / 2;
    const uint8_t hess_offset = cur_half_bytes % 2;
    Log::Warning("bin %d, bits_buffer = %d, grad_bits_pos = %ld, grad_bits_offset = %d, hess_bits_pos = %ld, hess_bits_offset = %d", bin, bits_buffer[grad_bits_pos], grad_bits_pos, grad_bits_offset, hess_bits_pos, hess_bits_offset);
    if (hess_offset == 1) {
      if (hess_bits == 0) {
      Log::Warning("12 bin %d, hess_diff = %ld, hess_bits = %d", bin, hess_diff, hess_bits);
        out_buffer[hess_pos] |= static_cast<uint8_t>(hess_diff << 4);
      Log::Warning("0 compressed hess = %d", (static_cast<uint8_t>(hess_diff << 4)));
        ++cur_half_bytes;
      } else if (hess_bits == 1) {
      Log::Warning("13 bin %d, hess_diff = %ld, hess_bits = %d", bin, hess_diff, hess_bits);
        out_buffer[hess_pos] |= static_cast<uint8_t>(hess_diff << 4);
        out_buffer[hess_pos + 1] |= (static_cast<uint8_t>(hess_diff >> 4) & 0x0f);
        cur_half_bytes += 2;
      } else if (hess_bits == 2) {
      Log::Warning("14 bin %d, hess_diff = %ld, hess_bits = %d", bin, hess_diff, hess_bits);
        out_buffer[hess_pos] |= static_cast<uint8_t>(hess_diff << 4);
        out_buffer[hess_pos + 1] = static_cast<uint8_t>(hess_diff >> 4);
        out_buffer[hess_pos + 2] |= (static_cast<uint8_t>(hess_diff >> 12) & 0x0f);
        cur_half_bytes += 4;
      } else {
      Log::Warning("15 bin %d, hess_diff = %ld, hess_bits = %d", bin, hess_diff, hess_bits);
        out_buffer[hess_pos] |= static_cast<uint8_t>(hess_diff << 4);
        out_buffer[hess_pos + 1] = static_cast<uint8_t>(hess_diff >> 4);
        out_buffer[hess_pos + 2] = static_cast<uint8_t>(hess_diff >> 12);
        out_buffer[hess_pos + 3] = static_cast<uint8_t>(hess_diff >> 20);
        out_buffer[hess_pos + 4] |= (static_cast<uint8_t>(hess_diff >> 28) & 0x0f);
        cur_half_bytes += 8;
      }
    } else {
      if (hess_bits == 0) {
      Log::Warning("16 bin %d, hess_diff = %ld, hess_bits = %d", bin, hess_diff, hess_bits);
      Log::Warning("1 compressed hess = %d", (static_cast<uint8_t>(hess_diff) & 0x0f));
        out_buffer[hess_pos] |= (static_cast<uint8_t>(hess_diff) & 0x0f);
        ++cur_half_bytes;
      } else if (hess_bits == 1) {
      Log::Warning("17 bin %d, hess_diff = %ld, hess_bits = %d", bin, hess_diff, hess_bits);
        out_buffer[hess_pos] = static_cast<uint8_t>(hess_diff);
        cur_half_bytes += 2;
      } else if (hess_bits == 2) {
      Log::Warning("18 bin %d, hess_diff = %ld, hess_bits = %d", bin, hess_diff, hess_bits);
        out_buffer[hess_pos] = static_cast<uint8_t>(hess_diff);
        out_buffer[hess_pos + 1] = static_cast<uint8_t>(hess_diff >> 8);
        cur_half_bytes += 4;
      } else if (hess_bits == 3) {
      Log::Warning("19 bin %d, hess_diff = %ld, hess_bits = %d", bin, hess_diff, hess_bits);
        out_buffer[hess_pos] = static_cast<uint8_t>(hess_diff);
        out_buffer[hess_pos + 1] = static_cast<uint8_t>(hess_diff >> 8);
        out_buffer[hess_pos + 2] = static_cast<uint8_t>(hess_diff >> 16);
        out_buffer[hess_pos + 3] = static_cast<uint8_t>(hess_diff >> 24);
        cur_half_bytes += 8;
      }
    }
    const uint64_t grad_pos = cur_half_bytes / 2;
    const uint8_t grad_offset = cur_half_bytes % 2;
    if (grad_offset == 0) {
      if (grad_bits == 0) {
      Log::Warning("20 bin %d, grad_diff = %ld, grad_bits = %d", bin, grad_diff, grad_bits);
        out_buffer[grad_pos] |= (static_cast<uint8_t>(grad_diff) & 0x0f);
        Log::Warning("bin %d, grad diff compressed = %d", static_cast<uint8_t>(grad_diff) & 0x0f);
        ++cur_half_bytes;
      } else if (grad_bits == 1) {
      Log::Warning("21 bin %d, grad_diff = %ld, grad_bits = %d", bin, grad_diff, grad_bits);
        out_buffer[grad_pos] = static_cast<uint8_t>(grad_diff);
        cur_half_bytes += 2;
      } else if (grad_bits == 2) {
      Log::Warning("22 bin %d, grad_diff = %ld, grad_bits = %d", bin, grad_diff, grad_bits);
        out_buffer[grad_pos] = static_cast<uint8_t>(grad_diff);
        out_buffer[grad_pos + 1] = static_cast<uint8_t>(grad_diff >> 8);
        cur_half_bytes += 4;
      } else if (grad_bits == 3) {
      Log::Warning("23 bin %d, grad_diff = %ld, grad_bits = %d", bin, grad_diff, grad_bits);
        out_buffer[grad_pos] = static_cast<uint8_t>(grad_diff);
        out_buffer[grad_pos + 1] = static_cast<uint8_t>(grad_diff >> 8);
        out_buffer[grad_pos + 2] = static_cast<uint8_t>(grad_diff >> 16);
        out_buffer[grad_pos + 3] = static_cast<uint8_t>(grad_diff >> 24);
        cur_half_bytes += 8;
      }
    } else {
      if (grad_bits == 0) {
      Log::Warning("24 bin %d, grad_diff = %ld, grad_bits = %d", bin, grad_diff, grad_bits);
        out_buffer[grad_pos] |= static_cast<uint8_t>(grad_diff << 4);
        Log::Warning("bin %d, grad diff compressed = %d", static_cast<uint8_t>(grad_diff << 4));
        ++cur_half_bytes;
      } else if (grad_bits == 1) {
      Log::Warning("25 bin %d, grad_diff = %ld, grad_bits = %d", bin, grad_diff, grad_bits);
        out_buffer[grad_pos] |= static_cast<uint8_t>(grad_diff << 4);
        out_buffer[grad_pos + 1] |= (static_cast<uint8_t>(grad_diff >> 4) & 0x0f);
        cur_half_bytes += 2;
      } else if (grad_bits == 2) {
      Log::Warning("26 bin %d, grad_diff = %ld, grad_bits = %d", bin, grad_diff, grad_bits);
        out_buffer[grad_pos] |= static_cast<uint8_t>(grad_diff << 4);
        out_buffer[grad_pos + 1] = static_cast<uint8_t>(grad_diff >> 4);
        out_buffer[grad_pos + 2] |= (static_cast<uint8_t>(grad_diff >> 12) & 0x0f);
        cur_half_bytes += 4;
      } else if (grad_bits == 3) {
      Log::Warning("27 bin %d, grad_diff = %ld, grad_bits = %d", bin, grad_diff, grad_bits);
        out_buffer[grad_pos] |= static_cast<uint8_t>(grad_diff << 4);
        out_buffer[grad_pos + 1] = static_cast<uint8_t>(grad_diff >> 4);
        out_buffer[grad_pos + 2] = static_cast<uint8_t>(grad_diff >> 12);
        out_buffer[grad_pos + 3] = static_cast<uint8_t>(grad_diff >> 20);
        out_buffer[grad_pos + 4] |= (static_cast<uint8_t>(grad_diff >> 28) & 0x0f);
        cur_half_bytes += 8;
      }
    }
  }
  Log::Warning("finish writing thread_index = %d", omp_get_thread_num());
}

template <typename S_HIST_T, typename U_HIST_T>
void HistogramCompressor::Decompress(const uint8_t* in_buffer, const uint8_t* in_bits_buffer, const uint64_t* thread_half_byte_offset, data_size_t num_bin, S_HIST_T* out_buffer) {
  const data_size_t block_size = (num_bin + num_threads_ - 1) / num_threads_;
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const data_size_t start = thread_index * block_size;
    const data_size_t end = std::min(start + block_size, num_bin);
    const uint64_t half_byte_start = thread_half_byte_offset[thread_index];
    uint64_t cur_half_byte = half_byte_start;
    int64_t prev_grad = 0;
    int64_t prev_hess = 0;
    int64_t grad = 0;
    int64_t hess = 0;
    for (data_size_t bin = start; bin < end; ++bin) {
      const data_size_t bin_offset = (bin << 1);
      const uint64_t hess_bits_pos = bin_offset / 4;
      const uint8_t hess_bits_offset = (bin_offset % 4) << 1;
      const uint8_t hess_bits = (in_bits_buffer[hess_bits_pos] >> hess_bits_offset) & 0x03;
      const uint8_t hess_offset = cur_half_byte % 2;
      const uint64_t hess_pos = cur_half_byte / 2;
      if (hess_offset == 1) {
        if (hess_bits == 0) {
          hess = static_cast<int64_t>(static_cast<int8_t>(in_buffer[hess_pos] & 0xf0)) >> 4;
          ++cur_half_byte;
        } else if (hess_bits == 1) {
          hess = static_cast<int64_t>(static_cast<uint8_t>(in_buffer[hess_pos] >> 4) & 0x0f) |
            static_cast<int64_t>(static_cast<int8_t>(in_buffer[hess_pos + 1] << 4));
          cur_half_byte += 2;
        } else if (hess_bits == 2) {
          hess = static_cast<int64_t>(static_cast<uint8_t>(in_buffer[hess_pos] >> 4) & 0x0f) |
            (static_cast<int64_t>(static_cast<uint8_t>(in_buffer[hess_pos + 1])) << 4) |
            (static_cast<int64_t>(static_cast<int8_t>(in_buffer[hess_pos + 2] << 4)) << 8);
          cur_half_byte += 4;
        } else if (hess_bits == 3) {
          hess = static_cast<int64_t>(static_cast<uint8_t>(in_buffer[hess_pos] >> 4) & 0x0f) |
            (static_cast<int64_t>(static_cast<uint8_t>(in_buffer[hess_pos + 1])) << 4) |
            (static_cast<int64_t>(static_cast<uint8_t>(in_buffer[hess_pos + 2])) << 12) |
            (static_cast<int64_t>(static_cast<uint8_t>(in_buffer[hess_pos + 3])) << 20) |
            (static_cast<int64_t>(static_cast<int8_t>(in_buffer[hess_pos + 4] << 4)) << 24);
          cur_half_byte += 8;
        }
      } else {
        if (hess_bits == 0) {
          hess = static_cast<int64_t>(static_cast<int8_t>(in_buffer[hess_pos] << 4)) >> 4;
          cur_half_byte += 1;
        } else if (hess_bits == 1) {
          hess = static_cast<int64_t>(static_cast<int8_t>(in_buffer[hess_pos]));
          cur_half_byte += 2;
        } else if (hess_bits == 2) {
          hess = static_cast<int64_t>(static_cast<uint8_t>(in_buffer[hess_pos])) |
            (static_cast<int64_t>(static_cast<int8_t>(in_buffer[hess_pos + 1])) << 8);
          cur_half_byte += 4;
        } else if (hess_bits == 3) {
          hess = static_cast<int64_t>(static_cast<uint8_t>(in_buffer[hess_pos])) |
            (static_cast<int64_t>(static_cast<uint8_t>(in_buffer[hess_pos + 1])) << 8) |
            (static_cast<int64_t>(static_cast<uint8_t>(in_buffer[hess_pos + 2])) << 16) |
            (static_cast<int64_t>(static_cast<int8_t>(in_buffer[hess_pos + 3])) << 24);
          cur_half_byte += 8;
        }
      }
      Log::Warning("decompressing bin %d, hess diff %d", bin, hess);
      hess += prev_hess;
      prev_hess = hess;
      out_buffer[bin_offset] = static_cast<U_HIST_T>(hess);
      const uint64_t grad_bits_pos = (bin_offset + 1) / 4;
      const uint8_t grad_bits_offset = ((bin_offset + 1) % 4) << 1;
      const uint8_t grad_bits = (in_bits_buffer[grad_bits_pos] >> grad_bits_offset) & 0x03;
      const uint8_t grad_offset = cur_half_byte % 2;
      const uint64_t grad_pos = cur_half_byte / 2;
      if (grad_offset == 1) {
        if (grad_bits == 0) {
          grad = static_cast<int64_t>(static_cast<int8_t>(in_buffer[grad_pos] & 0xf0)) >> 4;
      Log::Warning("0 bits 0 decompressing bin %d, grad diff %d, pos value = %d", bin, grad, static_cast<int8_t>(in_buffer[grad_pos]));
          ++cur_half_byte;
        } else if (grad_bits == 1) {
          grad = static_cast<int64_t>(static_cast<uint8_t>(in_buffer[grad_pos] >> 4) & 0x0f) |
            static_cast<int64_t>(static_cast<int8_t>(in_buffer[grad_pos + 1] << 4));
      Log::Warning("1 bits 0 decompressing bin %d, grad diff %d, pos value = %d", bin, grad, static_cast<int8_t>(in_buffer[grad_pos]));
          cur_half_byte += 2;
        } else if (grad_bits == 2) {
          grad = static_cast<int64_t>(static_cast<uint8_t>(in_buffer[grad_pos] >> 4) & 0x0f) |
            (static_cast<int64_t>(static_cast<uint8_t>(in_buffer[grad_pos + 1])) << 4) |
            (static_cast<int64_t>(static_cast<int8_t>(in_buffer[grad_pos + 2] << 4)) << 8);
      Log::Warning("2 bits 0 decompressing bin %d, grad diff %d, pos value = %d", bin, grad, static_cast<int8_t>(in_buffer[grad_pos]));
          cur_half_byte += 4;
        } else if (grad_bits == 3) {
          grad = static_cast<int64_t>(static_cast<uint8_t>(in_buffer[grad_pos] >> 4) & 0x0f) |
            (static_cast<int64_t>(static_cast<uint8_t>(in_buffer[grad_pos + 1])) << 4) |
            (static_cast<int64_t>(static_cast<uint8_t>(in_buffer[grad_pos + 2])) << 12) |
            (static_cast<int64_t>(static_cast<uint8_t>(in_buffer[grad_pos + 3])) << 20) |
            (static_cast<int64_t>(static_cast<int8_t>(in_buffer[grad_pos + 4] << 4)) << 24);
      Log::Warning("3 bits 0 decompressing bin %d, grad diff %d, pos value = %d", bin, grad, static_cast<int8_t>(in_buffer[grad_pos]));
          cur_half_byte += 8;
        }
      } else {
        if (grad_bits == 0) {
          grad = static_cast<int64_t>(static_cast<int8_t>(in_buffer[grad_pos] << 4)) >> 4;
      Log::Warning("4 bits 0 decompressing bin %d, grad diff %d, pos value = %d", bin, grad, static_cast<int8_t>(in_buffer[grad_pos]));
          cur_half_byte += 1;
        } else if (grad_bits == 1) {
          grad = static_cast<int64_t>(static_cast<int8_t>(in_buffer[grad_pos]));
      Log::Warning("5 bits 0 decompressing bin %d, grad diff %d, pos value = %d", bin, grad, static_cast<int8_t>(in_buffer[grad_pos]));
          cur_half_byte += 2;
        } else if (grad_bits == 2) {
          grad = static_cast<int64_t>(static_cast<uint8_t>(in_buffer[grad_pos])) |
            (static_cast<int64_t>(static_cast<int8_t>(in_buffer[grad_pos + 1])) << 8);
      Log::Warning("6 bits 0 decompressing bin %d, grad diff %d, pos value = %d", bin, grad, static_cast<int8_t>(in_buffer[grad_pos]));
          cur_half_byte += 4;
        } else if (grad_bits == 3) {
          grad = static_cast<int64_t>(static_cast<uint8_t>(in_buffer[grad_pos])) |
            (static_cast<int64_t>(static_cast<uint8_t>(in_buffer[grad_pos + 1])) << 8) |
            (static_cast<int64_t>(static_cast<uint8_t>(in_buffer[grad_pos + 2])) << 16) |
            (static_cast<int64_t>(static_cast<int8_t>(in_buffer[grad_pos + 3])) << 24);
      Log::Warning("7 bits 0 decompressing bin %d, grad diff %d, pos value = %d", bin, grad, static_cast<int8_t>(in_buffer[grad_pos]));
          cur_half_byte += 8;
        }
      }
      Log::Warning("decompressing bin %d, grad diff %d", bin, grad);
      grad += prev_grad;
      prev_grad = grad;
      out_buffer[bin_offset + 1] = static_cast<S_HIST_T>(grad);
    }
  }
  std::vector<S_HIST_T> thread_grad_offset(num_threads_ + 1, 0);
  std::vector<U_HIST_T> thread_hess_offset(num_threads_ + 1, 0);
  for (int thread_index = 1; thread_index < num_threads_ + 1; ++thread_index) {
    const data_size_t start = thread_index * block_size;
    const data_size_t end = std::min(start + block_size, num_bin);
    thread_grad_offset[thread_index] = out_buffer[((start - 1) << 1) + 1] + thread_grad_offset[thread_index - 1];
    thread_hess_offset[thread_index] = static_cast<U_HIST_T>(out_buffer[(start - 1) << 1]) + thread_hess_offset[thread_index - 1];
  }
  #pragma omp parallel for schedule(static, 1) num_threads(num_threads_)
  for (int thread_index = 0; thread_index < num_threads_; ++thread_index) {
    const data_size_t start = thread_index * block_size;
    const data_size_t end = std::min(start + block_size, num_bin);
    const S_HIST_T grad_offset = thread_grad_offset[thread_index];
    const U_HIST_T hess_offset = thread_hess_offset[thread_index];
    for (data_size_t bin = start; bin < end; ++bin) {
      const data_size_t bin_offset = (bin << 1);
      out_buffer[bin_offset + 1] += grad_offset;
      out_buffer[bin_offset] = static_cast<U_HIST_T>(static_cast<U_HIST_T>(out_buffer[bin_offset]) + static_cast<U_HIST_T>(hess_offset));
    }
  }
}

void HistogramCompressor::Test() {
  const size_t test_len = 10000;
  std::vector<int32_t> int32_test_array(test_len * 2);
  std::mt19937 rand_eng(0);
  const int32_t num_range = 10000;
  std::vector<double> start_prob(num_range, 1.0f / num_range);
  std::discrete_distribution<int32_t> dist_start(start_prob.begin(), start_prob.end());
  const int32_t diff_range = 400;
  std::vector<double> diff_prob(diff_range, 1.0f / diff_range);
  std::discrete_distribution<int32_t> dist_diff(diff_prob.begin(), diff_prob.end());
  int32_t grad = -5000;//dist_start(rand_eng);
  int32_t hess = std::abs(dist_diff(rand_eng));
  for (size_t i = 0; i < test_len; ++i) {
    int32_t grad_diff = dist_diff(rand_eng) - 200;
    int32_t hess_diff = dist_diff(rand_eng) - 200;
    int32_test_array[(i << 1) + 1] = grad;
    int32_test_array[(i << 1)] = hess;
    grad = grad + grad_diff;
    hess = std::abs(hess + hess_diff);
  }
  std::vector<uint8_t> bits((test_len * 2 + 3) / 4, 0);
  std::vector<uint8_t> out_buffer(test_len * 2 * 8, 0);
  const int num_threads = num_threads_;
  std::vector<uint64_t> thread_total_half_bytes_offset(num_threads + 1, 0);
  Compress<int32_t, uint32_t>(int32_test_array.data(), out_buffer.data(), bits.data(), test_len, thread_total_half_bytes_offset.data());
  Log::Warning("finish compress, total half bytes = %ld", thread_total_half_bytes_offset.back());
  std::vector<int32_t> result(test_len * 2, 0);
  Decompress<int32_t, uint32_t>(out_buffer.data(), bits.data(), thread_total_half_bytes_offset.data(), static_cast<data_size_t>(test_len), result.data());

  for (size_t i = 0; i < test_len; ++i) {
    Log::Warning("i = %d, grad = %d, hess = %d, grad hat = %d, hess hat = %d",
      i, int32_test_array[(i << 1) + 1], int32_test_array[(i << 1)],
        result[(i << 1) + 1], result[(i << 1)]);
    CHECK_EQ(int32_test_array[(i << 1) + 1], result[(i << 1) + 1]);
    CHECK_EQ(static_cast<uint32_t>(int32_test_array[(i << 1)]), static_cast<uint32_t>(result[(i << 1)]));
  }
  Log::Warning("finish decompress, total half bytes = %ld", thread_total_half_bytes_offset.back());
}

}  // namespace LightGBM
