#include "tilepack.h"

#if APEXX_ENABLE_TENSORRT

#include <cuda_fp16.h>

namespace apexx::trt::plugins {

namespace {

__global__ void tilepack_fp16_kernel(
    const __half* feature_map,
    const int32_t* indices,
    __half* packed_out,
    int32_t batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t kmax,
    int32_t tile_size) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total =
      static_cast<int64_t>(batch) * kmax * channels * tile_size * tile_size;
  if (tid >= total) {
    return;
  }

  int64_t cursor = tid;
  const int32_t dx = static_cast<int32_t>(cursor % tile_size);
  cursor /= tile_size;
  const int32_t dy = static_cast<int32_t>(cursor % tile_size);
  cursor /= tile_size;
  const int32_t c = static_cast<int32_t>(cursor % channels);
  cursor /= channels;
  const int32_t k = static_cast<int32_t>(cursor % kmax);
  cursor /= kmax;
  const int32_t b = static_cast<int32_t>(cursor);

  const int32_t grid_w = width / tile_size;
  const int32_t tile_index = indices[b * kmax + k];
  const int32_t base_y = (tile_index / grid_w) * tile_size;
  const int32_t base_x = (tile_index % grid_w) * tile_size;
  const int32_t y = base_y + dy;
  const int32_t x = base_x + dx;

  __half value = __float2half(0.0F);
  if (tile_index >= 0 && y >= 0 && x >= 0 && y < height && x < width) {
    const int64_t input_offset =
        ((static_cast<int64_t>(b) * channels + c) * height + y) * width + x;
    value = feature_map[input_offset];
  }
  packed_out[tid] = value;
}

}  // namespace

cudaError_t launch_tilepack_fp16(
    const void* feature_map,
    const int32_t* indices,
    void* packed_out,
    int32_t batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t kmax,
    int32_t tile_size,
    cudaStream_t stream) noexcept {
  if (feature_map == nullptr || indices == nullptr || packed_out == nullptr || batch <= 0 ||
      channels <= 0 || height <= 0 || width <= 0 || kmax < 0 || tile_size <= 0 ||
      (height % tile_size) != 0 || (width % tile_size) != 0) {
    return cudaErrorInvalidValue;
  }

  const int64_t total =
      static_cast<int64_t>(batch) * kmax * channels * tile_size * tile_size;
  if (total == 0) {
    return cudaSuccess;
  }

  constexpr int32_t kThreads = 256;
  const int32_t blocks = static_cast<int32_t>((total + kThreads - 1) / kThreads);
  tilepack_fp16_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const __half*>(feature_map),
      indices,
      static_cast<__half*>(packed_out),
      batch,
      channels,
      height,
      width,
      kmax,
      tile_size);
  return cudaGetLastError();
}

}  // namespace apexx::trt::plugins

#endif  // APEXX_ENABLE_TENSORRT
