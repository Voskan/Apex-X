#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if APEXX_ENABLE_TENSORRT
#include "NvInfer.h"
#include "cuda_runtime_api.h"

namespace apexx::trt::plugins {

static constexpr const char* kTilePackPluginName = "TilePack";
static constexpr const char* kTilePackPluginVersion = "1";
static constexpr const char* kTilePackPluginNamespace = "apexx";

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
    cudaStream_t stream) noexcept;

class TilePackPlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  explicit TilePackPlugin(int32_t tile_size) noexcept;
  TilePackPlugin(const void* serial_data, size_t serial_length) noexcept;
  ~TilePackPlugin() override = default;

  // IPluginV2 methods
  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override;
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  void setPluginNamespace(const char* plugin_namespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

  // IPluginV2Ext methods
  nvinfer1::DataType getOutputDataType(
      int index,
      const nvinfer1::DataType* input_types,
      int nb_inputs) const noexcept override;

  // IPluginV2DynamicExt methods
  nvinfer1::DimsExprs getOutputDimensions(
      int output_index,
      const nvinfer1::DimsExprs* inputs,
      int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) noexcept override;
  bool supportsFormatCombination(
      int pos,
      const nvinfer1::PluginTensorDesc* in_out,
      int nb_inputs,
      int nb_outputs) noexcept override;
  void configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* in,
      int nb_inputs,
      const nvinfer1::DynamicPluginTensorDesc* out,
      int nb_outputs) noexcept override;
  size_t getWorkspaceSize(
      const nvinfer1::PluginTensorDesc* inputs,
      int nb_inputs,
      const nvinfer1::PluginTensorDesc* outputs,
      int nb_outputs) const noexcept override;
  int enqueue(
      const nvinfer1::PluginTensorDesc* input_desc,
      const nvinfer1::PluginTensorDesc* output_desc,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override;
  void attachToContext(
      cudnnContext* cudnn_context,
      cublasContext* cublas_context,
      nvinfer1::IGpuAllocator* gpu_allocator) noexcept override;
  void detachFromContext() noexcept override;

 private:
  int32_t tile_size_{8};
  std::string namespace_{kTilePackPluginNamespace};
};

class TilePackPluginCreator final : public nvinfer1::IPluginCreator {
 public:
  TilePackPluginCreator() noexcept;
  ~TilePackPluginCreator() override = default;

  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
  nvinfer1::IPluginV2* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc) noexcept override;
  nvinfer1::IPluginV2* deserializePlugin(
      const char* name,
      const void* serial_data,
      size_t serial_length) noexcept override;
  void setPluginNamespace(const char* plugin_namespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

 private:
  std::vector<nvinfer1::PluginField> fields_{};
  nvinfer1::PluginFieldCollection field_collection_{};
  std::string namespace_{kTilePackPluginNamespace};
};

}  // namespace apexx::trt::plugins

#endif  // APEXX_ENABLE_TENSORRT
