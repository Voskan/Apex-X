#include "tileunpackfusion.h"

#if APEXX_ENABLE_TENSORRT

#include <cstdint>
#include <limits>

#include "cuda_fp16.h"

namespace apexx::trt::plugins {

namespace {

__global__ void fill_int32_kernel(int32_t* out, int64_t n, int32_t value) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  out[tid] = value;
}

__device__ __forceinline__ int32_t compute_order_bits(int32_t kmax) {
  int32_t bits = 0;
  int32_t span = 1;
  while (span < kmax && bits < 30) {
    span <<= 1;
    ++bits;
  }
  return bits;
}

__device__ __forceinline__ bool decode_tile_origin(
    int32_t tile_index,
    int32_t grid_h,
    int32_t grid_w,
    int32_t tile_size,
    int32_t& base_y,
    int32_t& base_x) {
  if (tile_index < 0 || tile_index >= grid_h * grid_w) {
    return false;
  }
  base_y = (tile_index / grid_w) * tile_size;
  base_x = (tile_index % grid_w) * tile_size;
  return true;
}

__global__ void tileunpack_priority_kernel(
    const int32_t* indices,
    const int32_t* levels,
    int32_t* winner,
    int32_t batch,
    int32_t height,
    int32_t width,
    int32_t grid_h,
    int32_t grid_w,
    int32_t kmax,
    int32_t tile_size,
    int32_t tile_pixels) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(batch) * kmax * tile_pixels;
  if (tid >= total) {
    return;
  }

  int64_t cursor = tid;
  const int32_t pix = static_cast<int32_t>(cursor % tile_pixels);
  cursor /= tile_pixels;
  const int32_t k = static_cast<int32_t>(cursor % kmax);
  cursor /= kmax;
  const int32_t b = static_cast<int32_t>(cursor);

  const int32_t tile_index = indices[b * kmax + k];
  int32_t base_y = 0;
  int32_t base_x = 0;
  if (!decode_tile_origin(tile_index, grid_h, grid_w, tile_size, base_y, base_x)) {
    return;
  }

  const int32_t local_y = pix / tile_size;
  const int32_t local_x = pix - local_y * tile_size;
  const int32_t y = base_y + local_y;
  const int32_t x = base_x + local_x;
  if (y < 0 || x < 0 || y >= height || x >= width) {
    return;
  }

  const int32_t order_bits = compute_order_bits(kmax <= 0 ? 1 : kmax);
  const int32_t level = levels[b * kmax + k];
  const int32_t key = (level << order_bits) + k;
  const int64_t win_offset = (static_cast<int64_t>(b) * height + y) * width + x;
  atomicMax(winner + win_offset, key);
}

__global__ void tileunpack_scatter_kernel(
    const __half* base_map,
    const __half* packed_out,
    const int32_t* indices,
    const int32_t* levels,
    const __half* alpha_map,
    __half* merged_out,
    const int32_t* winner,
    int32_t batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t grid_h,
    int32_t grid_w,
    int32_t kmax,
    int32_t tile_size,
    int32_t tile_pixels,
    bool has_alpha) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(batch) * kmax * channels * tile_pixels;
  if (tid >= total) {
    return;
  }

  int64_t cursor = tid;
  const int32_t pix = static_cast<int32_t>(cursor % tile_pixels);
  cursor /= tile_pixels;
  const int32_t c = static_cast<int32_t>(cursor % channels);
  cursor /= channels;
  const int32_t k = static_cast<int32_t>(cursor % kmax);
  cursor /= kmax;
  const int32_t b = static_cast<int32_t>(cursor);

  const int32_t tile_index = indices[b * kmax + k];
  int32_t base_y = 0;
  int32_t base_x = 0;
  if (!decode_tile_origin(tile_index, grid_h, grid_w, tile_size, base_y, base_x)) {
    return;
  }

  const int32_t local_y = pix / tile_size;
  const int32_t local_x = pix - local_y * tile_size;
  const int32_t y = base_y + local_y;
  const int32_t x = base_x + local_x;
  if (y < 0 || x < 0 || y >= height || x >= width) {
    return;
  }

  const int32_t order_bits = compute_order_bits(kmax <= 0 ? 1 : kmax);
  const int32_t level = levels[b * kmax + k];
  const int32_t key = (level << order_bits) + k;
  const int64_t win_offset = (static_cast<int64_t>(b) * height + y) * width + x;
  if (winner[win_offset] != key) {
    return;
  }

  const int64_t in_offset =
      (((static_cast<int64_t>(b) * kmax + k) * channels + c) * tile_size + local_y) * tile_size +
      local_x;
  const int64_t out_offset = ((static_cast<int64_t>(b) * channels + c) * height + y) * width + x;
  const __half incoming = packed_out[in_offset];

  if (!has_alpha) {
    merged_out[out_offset] = incoming;
    return;
  }

  const int64_t alpha_offset = (static_cast<int64_t>(b) * height + y) * width + x;
  const __half alpha_h = alpha_map[alpha_offset];
  const float alpha = __half2float(alpha_h);
  const float base = __half2float(base_map[out_offset]);
  const float inc = __half2float(incoming);
  const float fused = base + alpha * (inc - base);
  merged_out[out_offset] = __float2half(fused);
}

}  // namespace

cudaError_t launch_tileunpackfusion_fp16(
    const void* base_map,
    const void* packed_out,
    const int32_t* indices,
    const int32_t* levels,
    const void* alpha_map,
    void* merged_out,
    void* winner_workspace,
    int32_t batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t kmax,
    int32_t tile_size,
    bool has_alpha,
    cudaStream_t stream) noexcept {
  if (base_map == nullptr || packed_out == nullptr || indices == nullptr || levels == nullptr ||
      merged_out == nullptr || winner_workspace == nullptr || batch <= 0 || channels <= 0 ||
      height <= 0 || width <= 0 || tile_size <= 0 || kmax < 0 || (height % tile_size) != 0 ||
      (width % tile_size) != 0) {
    return cudaErrorInvalidValue;
  }
  if (has_alpha && alpha_map == nullptr) {
    return cudaErrorInvalidValue;
  }

  const int64_t out_elements = static_cast<int64_t>(batch) * channels * height * width;
  const int64_t out_bytes = out_elements * static_cast<int64_t>(sizeof(__half));
  const auto copy_status = cudaMemcpyAsync(
      merged_out, base_map, static_cast<size_t>(out_bytes), cudaMemcpyDeviceToDevice, stream);
  if (copy_status != cudaSuccess) {
    return copy_status;
  }

  const int64_t win_elements = static_cast<int64_t>(batch) * height * width;
  constexpr int32_t kThreads = 256;
  const int32_t fill_blocks = static_cast<int32_t>((win_elements + kThreads - 1) / kThreads);
  fill_int32_kernel<<<fill_blocks, kThreads, 0, stream>>>(
      static_cast<int32_t*>(winner_workspace),
      win_elements,
      std::numeric_limits<int32_t>::min());
  auto status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  const int32_t tile_pixels = tile_size * tile_size;
  const int32_t grid_h = height / tile_size;
  const int32_t grid_w = width / tile_size;

  const int64_t prio_total = static_cast<int64_t>(batch) * kmax * tile_pixels;
  if (prio_total > 0) {
    const int32_t prio_blocks = static_cast<int32_t>((prio_total + kThreads - 1) / kThreads);
    tileunpack_priority_kernel<<<prio_blocks, kThreads, 0, stream>>>(
        indices,
        levels,
        static_cast<int32_t*>(winner_workspace),
        batch,
        height,
        width,
        grid_h,
        grid_w,
        kmax,
        tile_size,
        tile_pixels);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return status;
    }
  }

  const int64_t scatter_total = static_cast<int64_t>(batch) * kmax * channels * tile_pixels;
  if (scatter_total == 0) {
    return cudaSuccess;
  }
  const int32_t scatter_blocks = static_cast<int32_t>((scatter_total + kThreads - 1) / kThreads);
  tileunpack_scatter_kernel<<<scatter_blocks, kThreads, 0, stream>>>(
      static_cast<const __half*>(base_map),
      static_cast<const __half*>(packed_out),
      indices,
      levels,
      static_cast<const __half*>(alpha_map),
      static_cast<__half*>(merged_out),
      static_cast<const int32_t*>(winner_workspace),
      batch,
      channels,
      height,
      width,
      grid_h,
      grid_w,
      kmax,
      tile_size,
      tile_pixels,
      has_alpha);
  return cudaGetLastError();
}

}  // namespace apexx::trt::plugins

#endif  // APEXX_ENABLE_TENSORRT
