#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#if APEXX_ENABLE_TENSORRT && APEXX_ENABLE_CUDA && APEXX_ENABLE_REAL_TILEUNPACKFUSION_PLUGIN

#include "NvInfer.h"
#include "cuda_fp16.h"
#include "cuda_runtime_api.h"

#include "tileunpackfusion.h"

namespace {

using apexx::trt::plugins::TileUnpackFusionPluginCreator;

bool check_cuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << what << " failed: " << cudaGetErrorString(status) << "\n";
  return false;
}

std::vector<__half> reference_unpack_priority_only(
    const std::vector<__half>& base,
    const std::vector<__half>& packed,
    const std::vector<int32_t>& indices,
    const std::vector<int32_t>& levels,
    int32_t batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t kmax,
    int32_t tile_size) {
  std::vector<__half> out = base;
  const int32_t grid_w = width / tile_size;
  const int32_t order_bits = [] (int32_t k) {
    int32_t bits = 0;
    int32_t span = 1;
    while (span < k && bits < 30) {
      span <<= 1;
      ++bits;
    }
    return bits;
  }(kmax);

  std::vector<int32_t> winner(static_cast<std::size_t>(batch * height * width), std::numeric_limits<int32_t>::min());
  for (int32_t b = 0; b < batch; ++b) {
    for (int32_t k = 0; k < kmax; ++k) {
      const int32_t tile_idx = indices[b * kmax + k];
      const int32_t y0 = (tile_idx / grid_w) * tile_size;
      const int32_t x0 = (tile_idx % grid_w) * tile_size;
      const int32_t key = (levels[b * kmax + k] << order_bits) + k;
      for (int32_t dy = 0; dy < tile_size; ++dy) {
        for (int32_t dx = 0; dx < tile_size; ++dx) {
          const int32_t y = y0 + dy;
          const int32_t x = x0 + dx;
          if (y < 0 || x < 0 || y >= height || x >= width) {
            continue;
          }
          const std::size_t woff = static_cast<std::size_t>((b * height + y) * width + x);
          winner[woff] = std::max(winner[woff], key);
        }
      }
    }
  }

  for (int32_t b = 0; b < batch; ++b) {
    for (int32_t k = 0; k < kmax; ++k) {
      const int32_t tile_idx = indices[b * kmax + k];
      const int32_t y0 = (tile_idx / grid_w) * tile_size;
      const int32_t x0 = (tile_idx % grid_w) * tile_size;
      const int32_t key = (levels[b * kmax + k] << order_bits) + k;
      for (int32_t c = 0; c < channels; ++c) {
        for (int32_t dy = 0; dy < tile_size; ++dy) {
          for (int32_t dx = 0; dx < tile_size; ++dx) {
            const int32_t y = y0 + dy;
            const int32_t x = x0 + dx;
            if (y < 0 || x < 0 || y >= height || x >= width) {
              continue;
            }
            const std::size_t woff = static_cast<std::size_t>((b * height + y) * width + x);
            if (winner[woff] != key) {
              continue;
            }
            const std::size_t in_off = static_cast<std::size_t>((((b * kmax + k) * channels + c) * tile_size + dy) * tile_size + dx);
            const std::size_t out_off = static_cast<std::size_t>(((b * channels + c) * height + y) * width + x);
            out[out_off] = packed[in_off];
          }
        }
      }
    }
  }

  return out;
}

}  // namespace

int main() {
  constexpr int32_t batch = 1;
  constexpr int32_t channels = 1;
  constexpr int32_t height = 8;
  constexpr int32_t width = 8;
  constexpr int32_t tile_size = 4;
  constexpr int32_t kmax = 3;

  std::vector<__half> h_base(static_cast<std::size_t>(batch * channels * height * width), __float2half(0.0F));
  std::vector<__half> h_packed(static_cast<std::size_t>(batch * kmax * channels * tile_size * tile_size), __float2half(0.0F));
  // All three tiles overlap exactly (same tile index), level should decide winner.
  std::vector<int32_t> h_indices = {0, 0, 0};
  std::vector<int32_t> h_levels = {0, 1, 2};
  for (int32_t k = 0; k < kmax; ++k) {
    for (int32_t dy = 0; dy < tile_size; ++dy) {
      for (int32_t dx = 0; dx < tile_size; ++dx) {
        const std::size_t off = static_cast<std::size_t>((((k)*channels + 0) * tile_size + dy) * tile_size + dx);
        h_packed[off] = __float2half(static_cast<float>(k + 1));  // 1,2,3
      }
    }
  }

  const auto h_expected = reference_unpack_priority_only(
      h_base, h_packed, h_indices, h_levels, batch, channels, height, width, kmax, tile_size);

  __half* d_base = nullptr;
  __half* d_packed = nullptr;
  int32_t* d_indices = nullptr;
  int32_t* d_levels = nullptr;
  __half* d_out = nullptr;

  const std::size_t base_bytes = h_base.size() * sizeof(__half);
  const std::size_t packed_bytes = h_packed.size() * sizeof(__half);
  const std::size_t idx_bytes = h_indices.size() * sizeof(int32_t);
  const std::size_t out_bytes = base_bytes;

  if (!check_cuda(cudaMalloc(&d_base, base_bytes), "cudaMalloc(d_base)") ||
      !check_cuda(cudaMalloc(&d_packed, packed_bytes), "cudaMalloc(d_packed)") ||
      !check_cuda(cudaMalloc(&d_indices, idx_bytes), "cudaMalloc(d_indices)") ||
      !check_cuda(cudaMalloc(&d_levels, idx_bytes), "cudaMalloc(d_levels)") ||
      !check_cuda(cudaMalloc(&d_out, out_bytes), "cudaMalloc(d_out)")) {
    return 2;
  }

  if (!check_cuda(cudaMemcpy(d_base, h_base.data(), base_bytes, cudaMemcpyHostToDevice), "copy base") ||
      !check_cuda(cudaMemcpy(d_packed, h_packed.data(), packed_bytes, cudaMemcpyHostToDevice), "copy packed") ||
      !check_cuda(cudaMemcpy(d_indices, h_indices.data(), idx_bytes, cudaMemcpyHostToDevice), "copy indices") ||
      !check_cuda(cudaMemcpy(d_levels, h_levels.data(), idx_bytes, cudaMemcpyHostToDevice), "copy levels")) {
    return 2;
  }

  TileUnpackFusionPluginCreator creator;
  nvinfer1::IPluginV2* plugin_raw = creator.createPlugin("tileunpackfusion_test", nullptr);
  if (plugin_raw == nullptr) {
    std::cerr << "createPlugin failed\n";
    return 3;
  }
  auto* plugin = static_cast<nvinfer1::IPluginV2DynamicExt*>(plugin_raw);
  if (plugin->initialize() != 0) {
    std::cerr << "plugin initialize failed\n";
    plugin->destroy();
    return 4;
  }

  nvinfer1::PluginTensorDesc in_desc[4]{};
  nvinfer1::PluginTensorDesc out_desc[1]{};

  in_desc[0].type = nvinfer1::DataType::kHALF;
  in_desc[0].format = nvinfer1::TensorFormat::kLINEAR;
  in_desc[0].dims.nbDims = 4;
  in_desc[0].dims.d[0] = batch;
  in_desc[0].dims.d[1] = channels;
  in_desc[0].dims.d[2] = height;
  in_desc[0].dims.d[3] = width;

  in_desc[1].type = nvinfer1::DataType::kHALF;
  in_desc[1].format = nvinfer1::TensorFormat::kLINEAR;
  in_desc[1].dims.nbDims = 5;
  in_desc[1].dims.d[0] = batch;
  in_desc[1].dims.d[1] = kmax;
  in_desc[1].dims.d[2] = channels;
  in_desc[1].dims.d[3] = tile_size;
  in_desc[1].dims.d[4] = tile_size;

  in_desc[2].type = nvinfer1::DataType::kINT32;
  in_desc[2].format = nvinfer1::TensorFormat::kLINEAR;
  in_desc[2].dims.nbDims = 2;
  in_desc[2].dims.d[0] = batch;
  in_desc[2].dims.d[1] = kmax;

  in_desc[3].type = nvinfer1::DataType::kINT32;
  in_desc[3].format = nvinfer1::TensorFormat::kLINEAR;
  in_desc[3].dims.nbDims = 2;
  in_desc[3].dims.d[0] = batch;
  in_desc[3].dims.d[1] = kmax;

  out_desc[0].type = nvinfer1::DataType::kHALF;
  out_desc[0].format = nvinfer1::TensorFormat::kLINEAR;
  out_desc[0].dims.nbDims = 4;
  out_desc[0].dims.d[0] = batch;
  out_desc[0].dims.d[1] = channels;
  out_desc[0].dims.d[2] = height;
  out_desc[0].dims.d[3] = width;

  const std::size_t workspace_bytes = plugin->getWorkspaceSize(in_desc, 4, out_desc, 1);
  void* workspace = nullptr;
  if (workspace_bytes > 0 && !check_cuda(cudaMalloc(&workspace, workspace_bytes), "cudaMalloc(workspace)")) {
    plugin->terminate();
    plugin->destroy();
    return 5;
  }

  const void* inputs[4] = {d_base, d_packed, d_indices, d_levels};
  void* outputs[1] = {d_out};
  const int status = plugin->enqueue(in_desc, out_desc, inputs, outputs, workspace, nullptr);
  if (status != 0 || !check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    std::cerr << "plugin enqueue failed with status " << status << "\n";
    plugin->terminate();
    plugin->destroy();
    return 6;
  }

  std::vector<__half> h_out(h_base.size(), __float2half(0.0F));
  if (!check_cuda(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost), "copy out")) {
    plugin->terminate();
    plugin->destroy();
    return 7;
  }

  constexpr float kTolerance = 5e-3F;
  for (std::size_t i = 0; i < h_out.size(); ++i) {
    const float got = __half2float(h_out[i]);
    const float exp = __half2float(h_expected[i]);
    if (std::fabs(got - exp) > kTolerance) {
      std::cerr << "Mismatch at " << i << ": got=" << got << " expected=" << exp << "\n";
      plugin->terminate();
      plugin->destroy();
      return 8;
    }
  }

  plugin->terminate();
  plugin->destroy();
  cudaFree(workspace);
  cudaFree(d_base);
  cudaFree(d_packed);
  cudaFree(d_indices);
  cudaFree(d_levels);
  cudaFree(d_out);
  std::cout << "TileUnpackFusion priority test passed.\n";
  return 0;
}

#else

int main() {
  std::cout << "TileUnpackFusion plugin test skipped (TensorRT/CUDA/real plugin unavailable).\n";
  return 0;
}

#endif
