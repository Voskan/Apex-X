#include "tilessm_scan.h"

#if APEXX_ENABLE_TENSORRT

#include <algorithm>
#include <cstdint>
#include <cstring>
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

}  // namespace

TileSSMScanPlugin::TileSSMScanPlugin(int32_t direction, float clamp_value) noexcept
    : direction_(std::clamp(direction, 0, 1)), clamp_value_(std::max(clamp_value, 1.0F)) {}

TileSSMScanPlugin::TileSSMScanPlugin(const void* serial_data, size_t serial_length) noexcept {
  const void* cursor = serial_data;
  const void* end = static_cast<const char*>(serial_data) + serial_length;
  num_inputs_ = read_int32(cursor, end);
  if (num_inputs_ != 5 && num_inputs_ != 6) {
    num_inputs_ = 6;
  }
  direction_ = std::clamp(read_int32(cursor, end), 0, 1);
  clamp_value_ = std::max(read_float32(cursor, end), 1.0F);
}

const char* TileSSMScanPlugin::getPluginType() const noexcept { return kTileSSMScanPluginName; }

const char* TileSSMScanPlugin::getPluginVersion() const noexcept {
  return kTileSSMScanPluginVersion;
}

int TileSSMScanPlugin::getNbOutputs() const noexcept { return 2; }

int TileSSMScanPlugin::initialize() noexcept { return 0; }

void TileSSMScanPlugin::terminate() noexcept {}

size_t TileSSMScanPlugin::getSerializationSize() const noexcept {
  return sizeof(int32_t) + sizeof(int32_t) + sizeof(float);
}

void TileSSMScanPlugin::serialize(void* buffer) const noexcept {
  if (buffer == nullptr) {
    return;
  }
  void* cursor = buffer;
  write_int32(cursor, num_inputs_);
  write_int32(cursor, direction_);
  write_float32(cursor, clamp_value_);
}

void TileSSMScanPlugin::destroy() noexcept { delete this; }

nvinfer1::IPluginV2DynamicExt* TileSSMScanPlugin::clone() const noexcept {
  auto* plugin = new TileSSMScanPlugin(direction_, clamp_value_);
  plugin->num_inputs_ = num_inputs_;
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void TileSSMScanPlugin::setPluginNamespace(const char* plugin_namespace) noexcept {
  namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
}

const char* TileSSMScanPlugin::getPluginNamespace() const noexcept { return namespace_.c_str(); }

nvinfer1::DataType TileSSMScanPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const noexcept {
  if (index < 0 || index >= 2 || input_types == nullptr || nb_inputs < 5) {
    return nvinfer1::DataType::kFLOAT;
  }
  return input_types[0];
}

nvinfer1::DimsExprs TileSSMScanPlugin::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) noexcept {
  (void)expr_builder;
  nvinfer1::DimsExprs out{};
  if (inputs == nullptr || (nb_inputs != 5 && nb_inputs != 6)) {
    return out;
  }
  if (output_index == 0) {
    // y: [B,K,C]
    out.nbDims = 3;
    out.d[0] = inputs[0].d[0];
    out.d[1] = inputs[0].d[1];
    out.d[2] = inputs[0].d[2];
    return out;
  }
  if (output_index == 1) {
    // final_state: [B,C]
    out.nbDims = 2;
    out.d[0] = inputs[0].d[0];
    out.d[1] = inputs[0].d[2];
    return out;
  }
  return out;
}

bool TileSSMScanPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int nb_inputs,
    int nb_outputs) noexcept {
  if (in_out == nullptr || nb_outputs != 2 || (nb_inputs != 5 && nb_inputs != 6)) {
    return false;
  }
  const auto& desc = in_out[pos];
  if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }
  if (pos < nb_inputs) {
    return desc.type == nvinfer1::DataType::kHALF;
  }
  if (pos == nb_inputs || pos == nb_inputs + 1) {
    return desc.type == in_out[0].type;
  }
  return false;
}

void TileSSMScanPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nb_outputs) noexcept {
  (void)in;
  (void)out;
  if (nb_outputs != 2) {
    return;
  }
  if (nb_inputs == 5 || nb_inputs == 6) {
    num_inputs_ = nb_inputs;
  }
}

size_t TileSSMScanPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int nb_inputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nb_outputs) const noexcept {
  (void)inputs;
  (void)nb_inputs;
  (void)outputs;
  (void)nb_outputs;
  return 0;
}

int TileSSMScanPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  (void)workspace;
  if (input_desc == nullptr || output_desc == nullptr || inputs == nullptr || outputs == nullptr) {
    return -1;
  }
  if (num_inputs_ != 5 && num_inputs_ != 6) {
    return -1;
  }

  if (input_desc[0].dims.nbDims != 3 || input_desc[1].dims.nbDims != 1 ||
      input_desc[2].dims.nbDims != 1 || input_desc[3].dims.nbDims != 1 ||
      input_desc[4].dims.nbDims != 1 || output_desc[0].dims.nbDims != 3 ||
      output_desc[1].dims.nbDims != 2) {
    return -1;
  }
  if (num_inputs_ == 6 && input_desc[5].dims.nbDims != 2) {
    return -1;
  }

  for (int32_t i = 0; i < num_inputs_; ++i) {
    if (input_desc[i].type != nvinfer1::DataType::kHALF) {
      return -1;
    }
  }
  if (output_desc[0].type != nvinfer1::DataType::kHALF ||
      output_desc[1].type != nvinfer1::DataType::kHALF) {
    return -1;
  }

  const int32_t batch = input_desc[0].dims.d[0];
  const int32_t steps = input_desc[0].dims.d[1];
  const int32_t channels = input_desc[0].dims.d[2];
  if (batch <= 0 || steps < 0 || channels <= 0) {
    return -1;
  }
  if (input_desc[1].dims.d[0] != channels || input_desc[2].dims.d[0] != channels ||
      input_desc[3].dims.d[0] != channels || input_desc[4].dims.d[0] != channels) {
    return -1;
  }
  if (num_inputs_ == 6 &&
      (input_desc[5].dims.d[0] != batch || input_desc[5].dims.d[1] != channels)) {
    return -1;
  }
  if (output_desc[0].dims.d[0] != batch || output_desc[0].dims.d[1] != steps ||
      output_desc[0].dims.d[2] != channels || output_desc[1].dims.d[0] != batch ||
      output_desc[1].dims.d[1] != channels) {
    return -1;
  }

  const void* init_state = (num_inputs_ == 6) ? inputs[5] : nullptr;
  const cudaError_t status = launch_tilessm_scan_fp16(
      inputs[0],
      inputs[1],
      inputs[2],
      inputs[3],
      inputs[4],
      init_state,
      outputs[0],
      outputs[1],
      batch,
      steps,
      channels,
      num_inputs_ == 6,
      direction_,
      clamp_value_,
      stream);
  return status == cudaSuccess ? 0 : -1;
}

void TileSSMScanPlugin::attachToContext(
    cudnnContext* cudnn_context,
    cublasContext* cublas_context,
    nvinfer1::IGpuAllocator* gpu_allocator) noexcept {
  (void)cudnn_context;
  (void)cublas_context;
  (void)gpu_allocator;
}

void TileSSMScanPlugin::detachFromContext() noexcept {}

TileSSMScanPluginCreator::TileSSMScanPluginCreator() noexcept {
  fields_.emplace_back(
      nvinfer1::PluginField{"direction", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
  fields_.emplace_back(
      nvinfer1::PluginField{"clamp_value", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
  field_collection_.nbFields = static_cast<int32_t>(fields_.size());
  field_collection_.fields = fields_.data();
}

const char* TileSSMScanPluginCreator::getPluginName() const noexcept { return kTileSSMScanPluginName; }

const char* TileSSMScanPluginCreator::getPluginVersion() const noexcept {
  return kTileSSMScanPluginVersion;
}

const nvinfer1::PluginFieldCollection* TileSSMScanPluginCreator::getFieldNames() noexcept {
  return &field_collection_;
}

nvinfer1::IPluginV2* TileSSMScanPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) noexcept {
  (void)name;
  int32_t direction = 0;
  float clamp_value = 1.0e4F;
  if (fc != nullptr) {
    for (int32_t i = 0; i < fc->nbFields; ++i) {
      const nvinfer1::PluginField& field = fc->fields[i];
      if (field.name == nullptr || field.data == nullptr || field.length < 1) {
        continue;
      }
      const std::string fname(field.name);
      if (fname == "direction" && field.type == nvinfer1::PluginFieldType::kINT32) {
        direction = *static_cast<const int32_t*>(field.data);
      } else if (fname == "clamp_value" &&
                 field.type == nvinfer1::PluginFieldType::kFLOAT32) {
        clamp_value = *static_cast<const float*>(field.data);
      }
    }
  }
  auto* plugin = new TileSSMScanPlugin(direction, clamp_value);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

nvinfer1::IPluginV2* TileSSMScanPluginCreator::deserializePlugin(
    const char* name,
    const void* serial_data,
    size_t serial_length) noexcept {
  (void)name;
  auto* plugin = new TileSSMScanPlugin(serial_data, serial_length);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void TileSSMScanPluginCreator::setPluginNamespace(const char* plugin_namespace) noexcept {
  namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
}

const char* TileSSMScanPluginCreator::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

}  // namespace apexx::trt::plugins

using ApexXTileSSMScanPluginCreator = apexx::trt::plugins::TileSSMScanPluginCreator;
REGISTER_TENSORRT_PLUGIN(ApexXTileSSMScanPluginCreator);

#endif  // APEXX_ENABLE_TENSORRT
