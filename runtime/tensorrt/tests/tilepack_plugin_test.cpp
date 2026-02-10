#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#if APEXX_ENABLE_TENSORRT && APEXX_ENABLE_CUDA && APEXX_ENABLE_REAL_TILEPACK_PLUGIN

#include "NvInfer.h"
#include "cuda_fp16.h"
#include "cuda_runtime_api.h"

#include "tilepack.h"

namespace {

using apexx::trt::plugins::TilePackPluginCreator;

void fill_input(std::vector<__half>& input) {
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = __float2half(static_cast<float>(i % 97) * 0.125F);
  }
}

std::vector<__half> tilepack_reference_host(
    const std::vector<__half>& input,
    const std::vector<int32_t>& indices,
    int32_t batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t kmax,
    int32_t tile_size) {
  const int64_t total = static_cast<int64_t>(batch) * kmax * channels * tile_size * tile_size;
  std::vector<__half> out(static_cast<std::size_t>(total), __float2half(0.0F));
  const int32_t grid_w = width / tile_size;

  for (int32_t b = 0; b < batch; ++b) {
    for (int32_t k = 0; k < kmax; ++k) {
      const int32_t tile_index = indices[b * kmax + k];
      const int32_t base_y = (tile_index / grid_w) * tile_size;
      const int32_t base_x = (tile_index % grid_w) * tile_size;
      for (int32_t c = 0; c < channels; ++c) {
        for (int32_t dy = 0; dy < tile_size; ++dy) {
          for (int32_t dx = 0; dx < tile_size; ++dx) {
            const int64_t out_off = ((((static_cast<int64_t>(b) * kmax + k) * channels + c) *
                                      tile_size +
                                      dy) *
                                         tile_size +
                                     dx);
            const int32_t y = base_y + dy;
            const int32_t x = base_x + dx;
            if (tile_index >= 0 && y >= 0 && x >= 0 && y < height && x < width) {
              const int64_t in_off =
                  ((static_cast<int64_t>(b) * channels + c) * height + y) * width + x;
              out[static_cast<std::size_t>(out_off)] = input[static_cast<std::size_t>(in_off)];
            }
          }
        }
      }
    }
  }
  return out;
}

bool check_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << what << " failed: " << cudaGetErrorString(status) << "\n";
  return false;
}

}  // namespace

int main() {
  constexpr int32_t batch = 1;
  constexpr int32_t channels = 4;
  constexpr int32_t height = 8;
  constexpr int32_t width = 8;
  constexpr int32_t tile_size = 2;
  constexpr int32_t kmax = 4;

  std::vector<__half> h_input(static_cast<std::size_t>(batch * channels * height * width));
  std::vector<int32_t> h_idx = {0, 3, 10, 15};
  fill_input(h_input);

  const auto h_expected = tilepack_reference_host(
      h_input,
      h_idx,
      batch,
      channels,
      height,
      width,
      kmax,
      tile_size);

  const int64_t out_numel =
      static_cast<int64_t>(batch) * kmax * channels * tile_size * tile_size;
  std::vector<__half> h_out(static_cast<std::size_t>(out_numel), __float2half(0.0F));

  __half* d_input = nullptr;
  __half* d_output = nullptr;
  int32_t* d_idx = nullptr;
  if (!check_cuda(
          cudaMalloc(&d_input, h_input.size() * sizeof(__half)),
          "cudaMalloc(d_input)") ||
      !check_cuda(cudaMalloc(&d_idx, h_idx.size() * sizeof(int32_t)), "cudaMalloc(d_idx)") ||
      !check_cuda(
          cudaMalloc(&d_output, static_cast<std::size_t>(out_numel) * sizeof(__half)),
          "cudaMalloc(d_output)")) {
    return 2;
  }
  if (!check_cuda(
          cudaMemcpy(
              d_input,
              h_input.data(),
              h_input.size() * sizeof(__half),
              cudaMemcpyHostToDevice),
          "cudaMemcpy input") ||
      !check_cuda(
          cudaMemcpy(d_idx, h_idx.data(), h_idx.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
          "cudaMemcpy idx")) {
    return 2;
  }

  TilePackPluginCreator creator;
  nvinfer1::PluginField field{
      "tile_size",
      &tile_size,
      nvinfer1::PluginFieldType::kINT32,
      1,
  };
  nvinfer1::PluginFieldCollection fc{};
  fc.nbFields = 1;
  fc.fields = &field;

  nvinfer1::IPluginV2* plugin_pre = creator.createPlugin("tilepack_test", &fc);
  if (plugin_pre == nullptr) {
    std::cerr << "createPlugin failed\n";
    return 3;
  }

  std::vector<char> serial(plugin_pre->getSerializationSize());
  plugin_pre->serialize(serial.data());
  nvinfer1::IPluginV2* plugin_raw =
      creator.deserializePlugin("tilepack_test_deser", serial.data(), serial.size());
  plugin_pre->destroy();
  if (plugin_raw == nullptr) {
    std::cerr << "deserializePlugin failed\n";
    return 3;
  }

  auto* plugin = static_cast<nvinfer1::IPluginV2DynamicExt*>(plugin_raw);
  if (plugin->initialize() != 0) {
    std::cerr << "plugin initialize failed\n";
    plugin->destroy();
    return 4;
  }

  nvinfer1::PluginTensorDesc in_desc[2]{};
  nvinfer1::PluginTensorDesc out_desc[1]{};
  in_desc[0].type = nvinfer1::DataType::kHALF;
  in_desc[0].format = nvinfer1::TensorFormat::kLINEAR;
  in_desc[0].dims.nbDims = 4;
  in_desc[0].dims.d[0] = batch;
  in_desc[0].dims.d[1] = channels;
  in_desc[0].dims.d[2] = height;
  in_desc[0].dims.d[3] = width;

  in_desc[1].type = nvinfer1::DataType::kINT32;
  in_desc[1].format = nvinfer1::TensorFormat::kLINEAR;
  in_desc[1].dims.nbDims = 2;
  in_desc[1].dims.d[0] = batch;
  in_desc[1].dims.d[1] = kmax;

  out_desc[0].type = nvinfer1::DataType::kHALF;
  out_desc[0].format = nvinfer1::TensorFormat::kLINEAR;
  out_desc[0].dims.nbDims = 5;
  out_desc[0].dims.d[0] = batch;
  out_desc[0].dims.d[1] = kmax;
  out_desc[0].dims.d[2] = channels;
  out_desc[0].dims.d[3] = tile_size;
  out_desc[0].dims.d[4] = tile_size;

  const void* inputs[2] = {d_input, d_idx};
  void* outputs[1] = {d_output};
  const int status = plugin->enqueue(in_desc, out_desc, inputs, outputs, nullptr, nullptr);
  if (status != 0 || !check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    std::cerr << "plugin enqueue failed with status " << status << "\n";
    plugin->terminate();
    plugin->destroy();
    return 5;
  }

  if (!check_cuda(
          cudaMemcpy(
              h_out.data(),
              d_output,
              static_cast<std::size_t>(out_numel) * sizeof(__half),
              cudaMemcpyDeviceToHost),
          "cudaMemcpy output")) {
    plugin->terminate();
    plugin->destroy();
    return 6;
  }

  constexpr float kTolerance = 5e-3F;
  for (std::size_t i = 0; i < h_out.size(); ++i) {
    const float got = __half2float(h_out[i]);
    const float exp = __half2float(h_expected[i]);
    if (std::fabs(got - exp) > kTolerance) {
      std::cerr << "Mismatch at " << i << ": got=" << got << " expected=" << exp << "\n";
      plugin->terminate();
      plugin->destroy();
      return 7;
    }
  }

  plugin->terminate();
  plugin->destroy();
  cudaFree(d_input);
  cudaFree(d_idx);
  cudaFree(d_output);
  std::cout << "TilePack plugin test passed.\n";
  return 0;
}

#else

int main() {
  std::cout << "TilePack plugin test skipped (TensorRT/CUDA/real TilePack plugin unavailable).\n";
  return 0;
}

#endif
