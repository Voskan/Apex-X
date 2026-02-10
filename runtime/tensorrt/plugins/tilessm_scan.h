#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if APEXX_ENABLE_TENSORRT
#include "NvInfer.h"
#include "cuda_runtime_api.h"

namespace apexx::trt::plugins {

static constexpr const char* kTileSSMScanPluginName = "TileSSMScan";
static constexpr const char* kTileSSMScanPluginVersion = "1";
static constexpr const char* kTileSSMScanPluginNamespace = "apexx";

// direction: 0=forward, 1=backward
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
    cudaStream_t stream) noexcept;

class TileSSMScanPlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  TileSSMScanPlugin(int32_t direction, float clamp_value) noexcept;
  TileSSMScanPlugin(const void* serial_data, size_t serial_length) noexcept;
  ~TileSSMScanPlugin() override = default;

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
  int32_t num_inputs_{6};  // with init_state
  int32_t direction_{0};
  float clamp_value_{1.0e4F};
  std::string namespace_{kTileSSMScanPluginNamespace};
};

class TileSSMScanPluginCreator final : public nvinfer1::IPluginCreator {
 public:
  TileSSMScanPluginCreator() noexcept;
  ~TileSSMScanPluginCreator() override = default;

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
  std::string namespace_{kTileSSMScanPluginNamespace};
};

}  // namespace apexx::trt::plugins

#endif  // APEXX_ENABLE_TENSORRT
