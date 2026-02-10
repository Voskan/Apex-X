#include "tilessm_scan.h"

#if APEXX_ENABLE_TENSORRT

#include <cmath>
#include <cstdint>

#include "cuda_fp16.h"

namespace apexx::trt::plugins {

namespace {

__global__ void tilessm_scan_kernel_fp16(
    const __half* tokens,      // [B,K,C]
    const __half* decay,       // [C]
    const __half* input_gain,  // [C]
    const __half* output_gain, // [C]
    const __half* state_bias,  // [C]
    const __half* init_state,  // [B,C] or nullptr
    __half* y_out,             // [B,K,C]
    __half* final_state_out,   // [B,C]
    int32_t batch,
    int32_t steps,
    int32_t channels,
    bool has_init_state,
    int32_t direction,
    float clamp_value) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(batch) * channels;
  if (tid >= total) {
    return;
  }
  const int32_t c = static_cast<int32_t>(tid % channels);
  const int32_t b = static_cast<int32_t>(tid / channels);

  float decay_v = __half2float(decay[c]);
  if (decay_v < 1.0e-6F) {
    decay_v = 1.0e-6F;
  }
  if (decay_v > 1.0F - 1.0e-6F) {
    decay_v = 1.0F - 1.0e-6F;
  }
  const float one_minus = 1.0F - decay_v;
  const float in_gain = __half2float(input_gain[c]);
  const float out_gain = __half2float(output_gain[c]);
  const float bias = __half2float(state_bias[c]);
  const float clamp_abs = fabsf(clamp_value);

  float state = 0.0F;
  if (has_init_state) {
    const int64_t init_off = static_cast<int64_t>(b) * channels + c;
    state = __half2float(init_state[init_off]);
  }

  if (direction == 1) {
    for (int32_t step = steps - 1; step >= 0; --step) {
      const int64_t off = (static_cast<int64_t>(b) * steps + step) * channels + c;
      float token = __half2float(tokens[off]);
      if (isnan(token)) {
        token = 0.0F;
      }
      if (token > clamp_abs) {
        token = clamp_abs;
      } else if (token < -clamp_abs) {
        token = -clamp_abs;
      }
      const float driven = in_gain * token + bias;
      state = decay_v * state + one_minus * driven;
      const float y = out_gain * state;
      y_out[off] = __float2half(y);
    }
  } else {
    for (int32_t step = 0; step < steps; ++step) {
      const int64_t off = (static_cast<int64_t>(b) * steps + step) * channels + c;
      float token = __half2float(tokens[off]);
      if (isnan(token)) {
        token = 0.0F;
      }
      if (token > clamp_abs) {
        token = clamp_abs;
      } else if (token < -clamp_abs) {
        token = -clamp_abs;
      }
      const float driven = in_gain * token + bias;
      state = decay_v * state + one_minus * driven;
      const float y = out_gain * state;
      y_out[off] = __float2half(y);
    }
  }

  const int64_t final_off = static_cast<int64_t>(b) * channels + c;
  final_state_out[final_off] = __float2half(state);
}

}  // namespace

cudaError_t launch_tilessm_scan_fp16(
    const void* tokens,
    const void* decay,
    const void* input_gain,
    const void* output_gain,
    const void* state_bias,
    const void* init_state,
    void* y_out,
    void* final_state_out,
    int32_t batch,
    int32_t steps,
    int32_t channels,
    bool has_init_state,
    int32_t direction,
    float clamp_value,
    cudaStream_t stream) noexcept {
  if (tokens == nullptr || decay == nullptr || input_gain == nullptr || output_gain == nullptr ||
      state_bias == nullptr || y_out == nullptr || final_state_out == nullptr || batch <= 0 ||
      steps < 0 || channels <= 0) {
    return cudaErrorInvalidValue;
  }
  if (has_init_state && init_state == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (direction != 0 && direction != 1) {
    return cudaErrorInvalidValue;
  }

  const int64_t total = static_cast<int64_t>(batch) * channels;
  if (total == 0) {
    return cudaSuccess;
  }
  constexpr int32_t kThreads = 256;
  const int32_t blocks = static_cast<int32_t>((total + kThreads - 1) / kThreads);

  tilessm_scan_kernel_fp16<<<blocks, kThreads, 0, stream>>>(
      static_cast<const __half*>(tokens),
      static_cast<const __half*>(decay),
      static_cast<const __half*>(input_gain),
      static_cast<const __half*>(output_gain),
      static_cast<const __half*>(state_bias),
      static_cast<const __half*>(init_state),
      static_cast<__half*>(y_out),
      static_cast<__half*>(final_state_out),
      batch,
      steps,
      channels,
      has_init_state,
      direction,
      clamp_value);
  return cudaGetLastError();
}

}  // namespace apexx::trt::plugins

#endif  // APEXX_ENABLE_TENSORRT
