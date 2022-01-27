/*!
 * Copyright (c) 2022 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREELEARNER_HISTOGRAM_COMPRESSOR_HPP_
#define LIGHTGBM_TREELEARNER_HISTOGRAM_COMPRESSOR_HPP_

#include <LightGBM/meta.h>
#include <LightGBM/utils/threading.h>

namespace LightGBM {

class HistogramCompressor {
 public:
  HistogramCompressor(const int num_threads);

  template <typename S_HIST_T, typename U_HIST_T>
  void Compress(const S_HIST_T* in_buffer, uint8_t* out_buffer, uint8_t* out_bits_buffer, data_size_t num_bin, uint32_t* thread_total_half_bytes_offset);

  template <typename S_HIST_T, typename U_HIST_T>
  void Decompress(const uint8_t* in_buffer, const uint8_t* in_bits_buffer, const uint32_t* thread_half_byte_offset, data_size_t num_bin, S_HIST_T* out_buffer);

  void Test();

 private:
  template <typename S_HIST_T, typename U_HIST_T>
  uint32_t ComputeThreadHalfBytes(const S_HIST_T* in_buffer, uint8_t* out_bits_buffer,
    uint8_t* thread_first_bits_buffer,
    data_size_t start_bin, data_size_t end_bin);

  template <typename S_HIST_T, typename U_HIST_T>
  void WriteThreadCompressedData(
    const S_HIST_T* in_buffer, const uint8_t* bits_buffer,
    uint8_t* out_buffer,
    uint8_t* thread_first_buffer,
    data_size_t start_bin,
    data_size_t end_bin,
    uint32_t thread_start_half_bytes);

  std::vector<uint8_t> thread_first_bits_;
  std::vector<uint8_t> thread_first_;
  int num_threads_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_TREELEARNER_HISTOGRAM_COMPRESSOR_HPP_
