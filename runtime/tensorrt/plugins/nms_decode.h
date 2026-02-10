#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if APEXX_ENABLE_TENSORRT
#include "NvInfer.h"
#include "cuda_runtime_api.h"

namespace apexx::trt::plugins {

static constexpr const char* kNMSDecodePluginName = "DecodeNMS";
static constexpr const char* kNMSDecodePluginVersion = "1";
static constexpr const char* kNMSDecodePluginNamespace = "apexx";

// Inputs:
// 0: cls_logits [B,N,C] FP16
// 1: box_reg    [B,N,4] FP16 (l/t/r/b logits)
// 2: quality    [B,N] FP16
// 3: centers    [N,2] FP16 (cx,cy)
// 4: strides    [N] FP16
// Outputs:
// 0: boxes      [B,max_det,4] FP16 (xyxy)
// 1: scores     [B,max_det] FP16
// 2: class_ids  [B,max_det] INT32
// 3: valid      [B] INT32
cudaError_t launch_nms_decode_fp16(
    const void* cls_logits,
    const void* box_reg,
    const void* quality,
    const void* centers,
    const void* strides,
    void* boxes_out,
    void* scores_out,
    int32_t* class_ids_out,
    int32_t* valid_counts_out,
    void* workspace,
    int32_t batch,
    int32_t anchors,
    int32_t classes,
    int32_t max_detections,
    int32_t pre_nms_topk,
    float score_threshold,
    float iou_threshold,
    cudaStream_t stream) noexcept;

class NMSDecodePlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  NMSDecodePlugin(
      int32_t max_detections,
      int32_t pre_nms_topk,
      float score_threshold,
      float iou_threshold) noexcept;
  NMSDecodePlugin(const void* serial_data, size_t serial_length) noexcept;
  ~NMSDecodePlugin() override = default;

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

  nvinfer1::DataType getOutputDataType(
      int index,
      const nvinfer1::DataType* input_types,
      int nb_inputs) const noexcept override;
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
  int32_t max_detections_{100};
  int32_t pre_nms_topk_{1000};
  float score_threshold_{0.05F};
  float iou_threshold_{0.6F};
  std::string namespace_{kNMSDecodePluginNamespace};
};

class NMSDecodePluginCreator final : public nvinfer1::IPluginCreator {
 public:
  NMSDecodePluginCreator() noexcept;
  ~NMSDecodePluginCreator() override = default;

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
  std::string namespace_{kNMSDecodePluginNamespace};
};

}  // namespace apexx::trt::plugins

#endif  // APEXX_ENABLE_TENSORRT
