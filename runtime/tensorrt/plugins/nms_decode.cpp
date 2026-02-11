#include "nms_decode.h"

#if APEXX_ENABLE_TENSORRT

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "NvInferRuntime.h"

namespace apexx::trt::plugins {

namespace {

int32_t read_int32(const void*& cursor, const void* end) noexcept {
  if (static_cast<const char*>(cursor) + sizeof(int32_t) > static_cast<const char*>(end)) {
    return 0;
  }
  int32_t value = 0;
  std::memcpy(&value, cursor, sizeof(int32_t));
  cursor = static_cast<const char*>(cursor) + sizeof(int32_t);
  return value;
}

float read_float32(const void*& cursor, const void* end) noexcept {
  if (static_cast<const char*>(cursor) + sizeof(float) > static_cast<const char*>(end)) {
    return 0.0F;
  }
  float value = 0.0F;
  std::memcpy(&value, cursor, sizeof(float));
  cursor = static_cast<const char*>(cursor) + sizeof(float);
  return value;
}

void write_int32(void*& cursor, int32_t value) noexcept {
  std::memcpy(cursor, &value, sizeof(int32_t));
  cursor = static_cast<char*>(cursor) + sizeof(int32_t);
}

void write_float32(void*& cursor, float value) noexcept {
  std::memcpy(cursor, &value, sizeof(float));
  cursor = static_cast<char*>(cursor) + sizeof(float);
}

size_t align_up(size_t value, size_t align) noexcept {
  if (align == 0) {
    return value;
  }
  const size_t rem = value % align;
  return rem == 0 ? value : value + (align - rem);
}

size_t per_batch_workspace_bytes(int64_t candidate_cap) noexcept {
  if (candidate_cap <= 0) {
    return 0;
  }
  size_t offset = 0;
  const size_t cap = static_cast<size_t>(candidate_cap);

  offset = align_up(offset, alignof(float));
  offset += cap * sizeof(float);      // scores
  offset = align_up(offset, alignof(float));
  offset += cap * 4 * sizeof(float);  // boxes
  offset = align_up(offset, alignof(int32_t));
  offset += cap * sizeof(int32_t);  // class_ids
  offset += cap * sizeof(int32_t);  // pair_ids
  offset += cap * sizeof(int32_t);  // suppressed
  offset += cap * sizeof(int32_t);  // keep_ids
  return offset;
}

}  // namespace

NMSDecodePlugin::NMSDecodePlugin(
    int32_t max_detections,
    int32_t pre_nms_topk,
    float score_threshold,
    float iou_threshold) noexcept
    : max_detections_(std::max(1, max_detections)),
      pre_nms_topk_(std::max(1, pre_nms_topk)),
      score_threshold_(std::clamp(score_threshold, 0.0F, 1.0F)),
      iou_threshold_(std::clamp(iou_threshold, 0.0F, 1.0F)) {}

NMSDecodePlugin::NMSDecodePlugin(const void* serial_data, size_t serial_length) noexcept {
  const void* cursor = serial_data;
  const void* end = static_cast<const char*>(serial_data) + serial_length;
  max_detections_ = std::max(1, read_int32(cursor, end));
  pre_nms_topk_ = std::max(1, read_int32(cursor, end));
  score_threshold_ = std::clamp(read_float32(cursor, end), 0.0F, 1.0F);
  iou_threshold_ = std::clamp(read_float32(cursor, end), 0.0F, 1.0F);
}

const char* NMSDecodePlugin::getPluginType() const noexcept { return kNMSDecodePluginName; }

const char* NMSDecodePlugin::getPluginVersion() const noexcept {
  return kNMSDecodePluginVersion;
}

int NMSDecodePlugin::getNbOutputs() const noexcept { return 4; }

int NMSDecodePlugin::initialize() noexcept { return 0; }

void NMSDecodePlugin::terminate() noexcept {}

size_t NMSDecodePlugin::getSerializationSize() const noexcept {
  return sizeof(int32_t) + sizeof(int32_t) + sizeof(float) + sizeof(float);
}

void NMSDecodePlugin::serialize(void* buffer) const noexcept {
  if (buffer == nullptr) {
    return;
  }
  void* cursor = buffer;
  write_int32(cursor, max_detections_);
  write_int32(cursor, pre_nms_topk_);
  write_float32(cursor, score_threshold_);
  write_float32(cursor, iou_threshold_);
}

void NMSDecodePlugin::destroy() noexcept { delete this; }

nvinfer1::IPluginV2DynamicExt* NMSDecodePlugin::clone() const noexcept {
  auto* plugin = new NMSDecodePlugin(
      max_detections_,
      pre_nms_topk_,
      score_threshold_,
      iou_threshold_);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void NMSDecodePlugin::setPluginNamespace(const char* plugin_namespace) noexcept {
  namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
}

const char* NMSDecodePlugin::getPluginNamespace() const noexcept { return namespace_.c_str(); }

nvinfer1::DataType NMSDecodePlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const noexcept {
  if (input_types == nullptr || nb_inputs != 5) {
    return nvinfer1::DataType::kFLOAT;
  }
  if (index == 0 || index == 1) {
    return input_types[0];
  }
  if (index == 2 || index == 3) {
    return nvinfer1::DataType::kINT32;
  }
  return nvinfer1::DataType::kFLOAT;
}

nvinfer1::DimsExprs NMSDecodePlugin::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) noexcept {
  nvinfer1::DimsExprs out{};
  if (inputs == nullptr || nb_inputs != 5) {
    return out;
  }
  if (output_index == 0) {
    out.nbDims = 3;
    out.d[0] = inputs[0].d[0];
    out.d[1] = expr_builder.constant(max_detections_);
    out.d[2] = expr_builder.constant(4);
    return out;
  }
  if (output_index == 1 || output_index == 2) {
    out.nbDims = 2;
    out.d[0] = inputs[0].d[0];
    out.d[1] = expr_builder.constant(max_detections_);
    return out;
  }
  if (output_index == 3) {
    out.nbDims = 1;
    out.d[0] = inputs[0].d[0];
    return out;
  }
  return out;
}

bool NMSDecodePlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int nb_inputs,
    int nb_outputs) noexcept {
  if (in_out == nullptr || nb_inputs != 5 || nb_outputs != 4) {
    return false;
  }
  const auto& desc = in_out[pos];
  if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }
  if (pos < 5) {
    return desc.type == nvinfer1::DataType::kHALF;
  }
  if (pos == 5 || pos == 6) {
    return desc.type == in_out[0].type;
  }
  if (pos == 7 || pos == 8) {
    return desc.type == nvinfer1::DataType::kINT32;
  }
  return false;
}

void NMSDecodePlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nb_outputs) noexcept {
  (void)in;
  (void)out;
  (void)nb_inputs;
  (void)nb_outputs;
}

size_t NMSDecodePlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int nb_inputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nb_outputs) const noexcept {
  (void)outputs;
  if (inputs == nullptr || nb_inputs != 5 || nb_outputs != 4) {
    return 0;
  }
  if (inputs[0].dims.nbDims != 3) {
    return 0;
  }
  const int64_t batch = inputs[0].dims.d[0];
  const int64_t anchors = inputs[0].dims.d[1];
  const int64_t classes = inputs[0].dims.d[2];
  if (batch <= 0 || anchors <= 0 || classes <= 0) {
    return 0;
  }
  const int64_t candidate_cap = std::min<int64_t>(
      static_cast<int64_t>(pre_nms_topk_),
      anchors * classes);
  if (candidate_cap <= 0) {
    return 0;
  }
  const size_t per_batch = per_batch_workspace_bytes(candidate_cap);
  if (per_batch == 0) {
    return 0;
  }
  if (batch > static_cast<int64_t>(std::numeric_limits<size_t>::max() / per_batch)) {
    return 0;
  }
  return static_cast<size_t>(batch) * per_batch;
}

int NMSDecodePlugin::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  if (input_desc == nullptr || output_desc == nullptr || inputs == nullptr || outputs == nullptr) {
    return -1;
  }

  if (input_desc[0].dims.nbDims != 3 || input_desc[1].dims.nbDims != 3 ||
      input_desc[2].dims.nbDims != 2 || input_desc[3].dims.nbDims != 2 ||
      input_desc[4].dims.nbDims != 1) {
    return -1;
  }
  if (output_desc[0].dims.nbDims != 3 || output_desc[1].dims.nbDims != 2 ||
      output_desc[2].dims.nbDims != 2 || output_desc[3].dims.nbDims != 1) {
    return -1;
  }

  for (int i = 0; i < 5; ++i) {
    if (input_desc[i].type != nvinfer1::DataType::kHALF) {
      return -1;
    }
  }
  if (output_desc[0].type != nvinfer1::DataType::kHALF ||
      output_desc[1].type != nvinfer1::DataType::kHALF ||
      output_desc[2].type != nvinfer1::DataType::kINT32 ||
      output_desc[3].type != nvinfer1::DataType::kINT32) {
    return -1;
  }

  const int32_t batch = input_desc[0].dims.d[0];
  const int32_t anchors = input_desc[0].dims.d[1];
  const int32_t classes = input_desc[0].dims.d[2];
  if (batch <= 0 || anchors <= 0 || classes <= 0) {
    return -1;
  }
  if (input_desc[1].dims.d[0] != batch || input_desc[1].dims.d[1] != anchors ||
      input_desc[1].dims.d[2] != 4) {
    return -1;
  }
  if (input_desc[2].dims.d[0] != batch || input_desc[2].dims.d[1] != anchors) {
    return -1;
  }
  if (input_desc[3].dims.d[0] != anchors || input_desc[3].dims.d[1] != 2) {
    return -1;
  }
  if (input_desc[4].dims.d[0] != anchors) {
    return -1;
  }

  if (output_desc[0].dims.d[0] != batch || output_desc[0].dims.d[1] != max_detections_ ||
      output_desc[0].dims.d[2] != 4) {
    return -1;
  }
  if (output_desc[1].dims.d[0] != batch || output_desc[1].dims.d[1] != max_detections_) {
    return -1;
  }
  if (output_desc[2].dims.d[0] != batch || output_desc[2].dims.d[1] != max_detections_) {
    return -1;
  }
  if (output_desc[3].dims.d[0] != batch) {
    return -1;
  }

  const size_t workspace_bytes = getWorkspaceSize(input_desc, 5, output_desc, 4);
  if (workspace_bytes > 0 && workspace == nullptr) {
    return -1;
  }

  const cudaError_t status = launch_nms_decode_fp16(
      inputs[0],
      inputs[1],
      inputs[2],
      inputs[3],
      inputs[4],
      outputs[0],
      outputs[1],
      static_cast<int32_t*>(outputs[2]),
      static_cast<int32_t*>(outputs[3]),
      workspace,
      batch,
      anchors,
      classes,
      max_detections_,
      pre_nms_topk_,
      score_threshold_,
      iou_threshold_,
      stream);
  return status == cudaSuccess ? 0 : -1;
}

void NMSDecodePlugin::attachToContext(
    cudnnContext* cudnn_context,
    cublasContext* cublas_context,
    nvinfer1::IGpuAllocator* gpu_allocator) noexcept {
  (void)cudnn_context;
  (void)cublas_context;
  (void)gpu_allocator;
}

void NMSDecodePlugin::detachFromContext() noexcept {}

NMSDecodePluginCreator::NMSDecodePluginCreator() noexcept {
  fields_.emplace_back(
      nvinfer1::PluginField{"max_detections", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
  fields_.emplace_back(
      nvinfer1::PluginField{"pre_nms_topk", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
  fields_.emplace_back(
      nvinfer1::PluginField{"score_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
  fields_.emplace_back(
      nvinfer1::PluginField{"iou_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
  field_collection_.nbFields = static_cast<int32_t>(fields_.size());
  field_collection_.fields = fields_.data();
}

const char* NMSDecodePluginCreator::getPluginName() const noexcept {
  return kNMSDecodePluginName;
}

const char* NMSDecodePluginCreator::getPluginVersion() const noexcept {
  return kNMSDecodePluginVersion;
}

const nvinfer1::PluginFieldCollection* NMSDecodePluginCreator::getFieldNames() noexcept {
  return &field_collection_;
}

nvinfer1::IPluginV2* NMSDecodePluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) noexcept {
  (void)name;
  int32_t max_detections = 100;
  int32_t pre_nms_topk = 1000;
  float score_threshold = 0.05F;
  float iou_threshold = 0.6F;
  if (fc != nullptr) {
    for (int32_t i = 0; i < fc->nbFields; ++i) {
      const nvinfer1::PluginField& field = fc->fields[i];
      if (field.name == nullptr || field.data == nullptr || field.length < 1) {
        continue;
      }
      const std::string field_name(field.name);
      if (field_name == "max_detections" && field.type == nvinfer1::PluginFieldType::kINT32) {
        max_detections = *static_cast<const int32_t*>(field.data);
      } else if (field_name == "pre_nms_topk" &&
                 field.type == nvinfer1::PluginFieldType::kINT32) {
        pre_nms_topk = *static_cast<const int32_t*>(field.data);
      } else if (field_name == "score_threshold" &&
                 field.type == nvinfer1::PluginFieldType::kFLOAT32) {
        score_threshold = *static_cast<const float*>(field.data);
      } else if (field_name == "iou_threshold" &&
                 field.type == nvinfer1::PluginFieldType::kFLOAT32) {
        iou_threshold = *static_cast<const float*>(field.data);
      }
    }
  }

  auto* plugin = new NMSDecodePlugin(
      max_detections,
      pre_nms_topk,
      score_threshold,
      iou_threshold);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

nvinfer1::IPluginV2* NMSDecodePluginCreator::deserializePlugin(
    const char* name,
    const void* serial_data,
    size_t serial_length) noexcept {
  (void)name;
  auto* plugin = new NMSDecodePlugin(serial_data, serial_length);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void NMSDecodePluginCreator::setPluginNamespace(const char* plugin_namespace) noexcept {
  namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
}

const char* NMSDecodePluginCreator::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

}  // namespace apexx::trt::plugins

using ApexXNMSDecodePluginCreator = apexx::trt::plugins::NMSDecodePluginCreator;
REGISTER_TENSORRT_PLUGIN(ApexXNMSDecodePluginCreator);

#endif  // APEXX_ENABLE_TENSORRT
