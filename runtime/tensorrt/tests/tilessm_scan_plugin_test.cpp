#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#if APEXX_ENABLE_TENSORRT && APEXX_ENABLE_CUDA && APEXX_ENABLE_REAL_TILESSM_PLUGIN

#include "NvInfer.h"
#include "cuda_fp16.h"
#include "cuda_runtime_api.h"

#include "tilessm_scan.h"

namespace {

using apexx::trt::plugins::TileSSMScanPluginCreator;

bool check_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << what << " failed: " << cudaGetErrorString(status) << "\n";
  return false;
}

void reference_scan(
    const std::vector<__half>& tokens,
    const std::vector<__half>& decay,
    const std::vector<__half>& input_gain,
    const std::vector<__half>& output_gain,
    const std::vector<__half>& state_bias,
    const std::vector<__half>& init_state,
    int32_t batch,
    int32_t steps,
    int32_t channels,
    int32_t direction,
    float clamp_value,
    std::vector<__half>& out_y,
    std::vector<__half>& out_final) {
  const float clamp_abs = std::fabs(clamp_value);
  out_y.assign(tokens.size(), __float2half(0.0F));
  out_final.assign(static_cast<std::size_t>(batch * channels), __float2half(0.0F));

  for (int32_t b = 0; b < batch; ++b) {
    for (int32_t c = 0; c < channels; ++c) {
      float d = __half2float(decay[static_cast<std::size_t>(c)]);
      d = std::min(std::max(d, 1.0e-6F), 1.0F - 1.0e-6F);
      const float one_minus = 1.0F - d;
      const float in_g = __half2float(input_gain[static_cast<std::size_t>(c)]);
      const float out_g = __half2float(output_gain[static_cast<std::size_t>(c)]);
      const float bias = __half2float(state_bias[static_cast<std::size_t>(c)]);
      float state = __half2float(init_state[static_cast<std::size_t>(b * channels + c)]);

      if (direction == 1) {
        for (int32_t step = steps - 1; step >= 0; --step) {
          const std::size_t off =
              static_cast<std::size_t>((static_cast<int64_t>(b) * steps + step) * channels + c);
          float token = __half2float(tokens[off]);
          if (!std::isfinite(token)) {
            token = 0.0F;
          }
          token = std::min(std::max(token, -clamp_abs), clamp_abs);
          const float driven = in_g * token + bias;
          state = d * state + one_minus * driven;
          out_y[off] = __float2half(out_g * state);
        }
      } else {
        for (int32_t step = 0; step < steps; ++step) {
          const std::size_t off =
              static_cast<std::size_t>((static_cast<int64_t>(b) * steps + step) * channels + c);
          float token = __half2float(tokens[off]);
          if (!std::isfinite(token)) {
            token = 0.0F;
          }
          token = std::min(std::max(token, -clamp_abs), clamp_abs);
          const float driven = in_g * token + bias;
          state = d * state + one_minus * driven;
          out_y[off] = __float2half(out_g * state);
        }
      }
      out_final[static_cast<std::size_t>(b * channels + c)] = __float2half(state);
    }
  }
}

bool run_case(int32_t direction) {
  constexpr int32_t batch = 2;
  constexpr int32_t steps = 32;
  constexpr int32_t channels = 16;
  constexpr float clamp_value = 1.0e4F;
  constexpr float tol = 5.0e-2F;

  std::vector<__half> h_tokens(static_cast<std::size_t>(batch * steps * channels));
  std::vector<__half> h_decay(static_cast<std::size_t>(channels));
  std::vector<__half> h_in_gain(static_cast<std::size_t>(channels));
  std::vector<__half> h_out_gain(static_cast<std::size_t>(channels));
  std::vector<__half> h_bias(static_cast<std::size_t>(channels));
  std::vector<__half> h_init(static_cast<std::size_t>(batch * channels));

  for (std::size_t i = 0; i < h_tokens.size(); ++i) {
    const float v = std::sin(static_cast<float>(i) * 0.01F) * 2.0F;
    h_tokens[i] = __float2half(v);
  }
  for (int32_t c = 0; c < channels; ++c) {
    h_decay[static_cast<std::size_t>(c)] = __float2half(0.8F + 0.001F * static_cast<float>(c));
    h_in_gain[static_cast<std::size_t>(c)] = __float2half(0.5F + 0.01F * static_cast<float>(c));
    h_out_gain[static_cast<std::size_t>(c)] = __float2half(0.7F + 0.01F * static_cast<float>(c));
    h_bias[static_cast<std::size_t>(c)] = __float2half(0.01F * static_cast<float>(c - 2));
  }
  for (std::size_t i = 0; i < h_init.size(); ++i) {
    h_init[i] = __float2half(0.1F * std::cos(static_cast<float>(i) * 0.1F));
  }

  std::vector<__half> h_ref_y;
  std::vector<__half> h_ref_final;
  reference_scan(
      h_tokens,
      h_decay,
      h_in_gain,
      h_out_gain,
      h_bias,
      h_init,
      batch,
      steps,
      channels,
      direction,
      clamp_value,
      h_ref_y,
      h_ref_final);

  __half* d_tokens = nullptr;
  __half* d_decay = nullptr;
  __half* d_in_gain = nullptr;
  __half* d_out_gain = nullptr;
  __half* d_bias = nullptr;
  __half* d_init = nullptr;
  __half* d_y = nullptr;
  __half* d_final = nullptr;

  const std::size_t tokens_bytes = h_tokens.size() * sizeof(__half);
  const std::size_t vec_bytes = h_decay.size() * sizeof(__half);
  const std::size_t init_bytes = h_init.size() * sizeof(__half);
  const std::size_t y_bytes = h_ref_y.size() * sizeof(__half);
  const std::size_t final_bytes = h_ref_final.size() * sizeof(__half);

  if (!check_cuda(cudaMalloc(&d_tokens, tokens_bytes), "cudaMalloc(d_tokens)") ||
      !check_cuda(cudaMalloc(&d_decay, vec_bytes), "cudaMalloc(d_decay)") ||
      !check_cuda(cudaMalloc(&d_in_gain, vec_bytes), "cudaMalloc(d_in_gain)") ||
      !check_cuda(cudaMalloc(&d_out_gain, vec_bytes), "cudaMalloc(d_out_gain)") ||
      !check_cuda(cudaMalloc(&d_bias, vec_bytes), "cudaMalloc(d_bias)") ||
      !check_cuda(cudaMalloc(&d_init, init_bytes), "cudaMalloc(d_init)") ||
      !check_cuda(cudaMalloc(&d_y, y_bytes), "cudaMalloc(d_y)") ||
      !check_cuda(cudaMalloc(&d_final, final_bytes), "cudaMalloc(d_final)")) {
    return false;
  }

  if (!check_cuda(cudaMemcpy(d_tokens, h_tokens.data(), tokens_bytes, cudaMemcpyHostToDevice), "copy tokens") ||
      !check_cuda(cudaMemcpy(d_decay, h_decay.data(), vec_bytes, cudaMemcpyHostToDevice), "copy decay") ||
      !check_cuda(cudaMemcpy(d_in_gain, h_in_gain.data(), vec_bytes, cudaMemcpyHostToDevice), "copy input_gain") ||
      !check_cuda(cudaMemcpy(d_out_gain, h_out_gain.data(), vec_bytes, cudaMemcpyHostToDevice), "copy output_gain") ||
      !check_cuda(cudaMemcpy(d_bias, h_bias.data(), vec_bytes, cudaMemcpyHostToDevice), "copy state_bias") ||
      !check_cuda(cudaMemcpy(d_init, h_init.data(), init_bytes, cudaMemcpyHostToDevice), "copy init_state")) {
    return false;
  }

  TileSSMScanPluginCreator creator;
  nvinfer1::PluginField fields[2] = {
      nvinfer1::PluginField{"direction", &direction, nvinfer1::PluginFieldType::kINT32, 1},
      nvinfer1::PluginField{"clamp_value", &clamp_value, nvinfer1::PluginFieldType::kFLOAT32, 1},
  };
  nvinfer1::PluginFieldCollection fc{};
  fc.nbFields = 2;
  fc.fields = fields;
  nvinfer1::IPluginV2* plugin_raw = creator.createPlugin("tilessm_test", &fc);
  if (plugin_raw == nullptr) {
    std::cerr << "createPlugin failed\n";
    return false;
  }
  auto* plugin = static_cast<nvinfer1::IPluginV2DynamicExt*>(plugin_raw);
  if (plugin->initialize() != 0) {
    std::cerr << "plugin initialize failed\n";
    plugin->destroy();
    return false;
  }

  nvinfer1::PluginTensorDesc in_desc[6]{};
  in_desc[0].type = nvinfer1::DataType::kHALF;
  in_desc[0].format = nvinfer1::TensorFormat::kLINEAR;
  in_desc[0].dims.nbDims = 3;
  in_desc[0].dims.d[0] = batch;
  in_desc[0].dims.d[1] = steps;
  in_desc[0].dims.d[2] = channels;
  for (int i = 1; i <= 4; ++i) {
    in_desc[i].type = nvinfer1::DataType::kHALF;
    in_desc[i].format = nvinfer1::TensorFormat::kLINEAR;
    in_desc[i].dims.nbDims = 1;
    in_desc[i].dims.d[0] = channels;
  }
  in_desc[5].type = nvinfer1::DataType::kHALF;
  in_desc[5].format = nvinfer1::TensorFormat::kLINEAR;
  in_desc[5].dims.nbDims = 2;
  in_desc[5].dims.d[0] = batch;
  in_desc[5].dims.d[1] = channels;

  nvinfer1::PluginTensorDesc out_desc[2]{};
  out_desc[0].type = nvinfer1::DataType::kHALF;
  out_desc[0].format = nvinfer1::TensorFormat::kLINEAR;
  out_desc[0].dims.nbDims = 3;
  out_desc[0].dims.d[0] = batch;
  out_desc[0].dims.d[1] = steps;
  out_desc[0].dims.d[2] = channels;
  out_desc[1].type = nvinfer1::DataType::kHALF;
  out_desc[1].format = nvinfer1::TensorFormat::kLINEAR;
  out_desc[1].dims.nbDims = 2;
  out_desc[1].dims.d[0] = batch;
  out_desc[1].dims.d[1] = channels;

  const void* inputs[6] = {d_tokens, d_decay, d_in_gain, d_out_gain, d_bias, d_init};
  void* outputs[2] = {d_y, d_final};

  const int status = plugin->enqueue(in_desc, out_desc, inputs, outputs, nullptr, nullptr);
  if (status != 0 || !check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    std::cerr << "plugin enqueue failed with status " << status << "\n";
    plugin->terminate();
    plugin->destroy();
    return false;
  }

  std::vector<__half> h_out_y(h_ref_y.size());
  std::vector<__half> h_out_final(h_ref_final.size());
  if (!check_cuda(cudaMemcpy(h_out_y.data(), d_y, y_bytes, cudaMemcpyDeviceToHost), "copy y") ||
      !check_cuda(cudaMemcpy(h_out_final.data(), d_final, final_bytes, cudaMemcpyDeviceToHost), "copy final_state")) {
    plugin->terminate();
    plugin->destroy();
    return false;
  }

  for (std::size_t i = 0; i < h_ref_y.size(); ++i) {
    const float got = __half2float(h_out_y[i]);
    const float exp = __half2float(h_ref_y[i]);
    if (std::fabs(got - exp) > tol) {
      std::cerr << "y mismatch at " << i << ": got=" << got << " expected=" << exp << "\n";
      plugin->terminate();
      plugin->destroy();
      return false;
    }
  }
  for (std::size_t i = 0; i < h_ref_final.size(); ++i) {
    const float got = __half2float(h_out_final[i]);
    const float exp = __half2float(h_ref_final[i]);
    if (std::fabs(got - exp) > tol) {
      std::cerr << "final_state mismatch at " << i << ": got=" << got << " expected=" << exp
                << "\n";
      plugin->terminate();
      plugin->destroy();
      return false;
    }
  }

  // Lightweight benchmark inside the harness.
  cudaEvent_t ev_start{};
  cudaEvent_t ev_end{};
  check_cuda(cudaEventCreate(&ev_start), "cudaEventCreate(start)");
  check_cuda(cudaEventCreate(&ev_end), "cudaEventCreate(end)");
  constexpr int warmup = 20;
  constexpr int iters = 200;
  for (int i = 0; i < warmup; ++i) {
    plugin->enqueue(in_desc, out_desc, inputs, outputs, nullptr, nullptr);
  }
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

  check_cuda(cudaEventRecord(ev_start, nullptr), "cudaEventRecord(start)");
  for (int i = 0; i < iters; ++i) {
    plugin->enqueue(in_desc, out_desc, inputs, outputs, nullptr, nullptr);
  }
  check_cuda(cudaEventRecord(ev_end, nullptr), "cudaEventRecord(end)");
  check_cuda(cudaEventSynchronize(ev_end), "cudaEventSynchronize");
  float elapsed_ms = 0.0F;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_end), "cudaEventElapsedTime");
  const float avg_ms = elapsed_ms / static_cast<float>(iters);
  const double tokens_processed = static_cast<double>(batch) * steps * channels;
  const double tok_per_sec = tokens_processed / (static_cast<double>(avg_ms) * 1.0e-3);
  std::cout << "TileSSMScan direction=" << direction << " avg_ms=" << avg_ms
            << " tok/s=" << tok_per_sec << "\n";
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_end);

  plugin->terminate();
  plugin->destroy();
  cudaFree(d_tokens);
  cudaFree(d_decay);
  cudaFree(d_in_gain);
  cudaFree(d_out_gain);
  cudaFree(d_bias);
  cudaFree(d_init);
  cudaFree(d_y);
  cudaFree(d_final);
  return true;
}

}  // namespace

int main() {
  if (!run_case(0)) {
    return 2;
  }
  if (!run_case(1)) {
    return 3;
  }
  std::cout << "TileSSMScan plugin test passed.\n";
  return 0;
}

#else

int main() {
  std::cout << "TileSSMScan plugin test skipped (TensorRT/CUDA/real plugin unavailable).\n";
  return 0;
}

#endif
