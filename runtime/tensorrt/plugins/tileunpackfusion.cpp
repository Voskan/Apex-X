#include "tileunpackfusion.h"

#if APEXX_ENABLE_TENSORRT

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

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

void write_int32(void*& cursor, int32_t value) noexcept {
  std::memcpy(cursor, &value, sizeof(int32_t));
  cursor = static_cast<char*>(cursor) + sizeof(int32_t);
}

int32_t compute_order_bits(int32_t kmax) noexcept {
  int32_t bits = 0;
  int32_t span = 1;
  while (span < kmax && bits < 30) {
    span <<= 1;
    ++bits;
  }
  return bits;
}

}  // namespace

TileUnpackFusionPlugin::TileUnpackFusionPlugin() noexcept = default;

TileUnpackFusionPlugin::TileUnpackFusionPlugin(const void* serial_data, size_t serial_length) noexcept {
  const void* cursor = serial_data;
  const void* end = static_cast<const char*>(serial_data) + serial_length;
  const int32_t parsed_inputs = read_int32(cursor, end);
  num_inputs_ = (parsed_inputs == 5) ? 5 : 4;
}

const char* TileUnpackFusionPlugin::getPluginType() const noexcept {
  return kTileUnpackFusionPluginName;
}

const char* TileUnpackFusionPlugin::getPluginVersion() const noexcept {
  return kTileUnpackFusionPluginVersion;
}

int TileUnpackFusionPlugin::getNbOutputs() const noexcept { return 1; }

int TileUnpackFusionPlugin::initialize() noexcept { return 0; }

void TileUnpackFusionPlugin::terminate() noexcept {}

size_t TileUnpackFusionPlugin::getSerializationSize() const noexcept { return sizeof(int32_t); }

void TileUnpackFusionPlugin::serialize(void* buffer) const noexcept {
  if (buffer == nullptr) {
    return;
  }
  void* cursor = buffer;
  write_int32(cursor, num_inputs_);
}

void TileUnpackFusionPlugin::destroy() noexcept { delete this; }

nvinfer1::IPluginV2DynamicExt* TileUnpackFusionPlugin::clone() const noexcept {
  auto* plugin = new TileUnpackFusionPlugin();
  plugin->num_inputs_ = num_inputs_;
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void TileUnpackFusionPlugin::setPluginNamespace(const char* plugin_namespace) noexcept {
  namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
}

const char* TileUnpackFusionPlugin::getPluginNamespace() const noexcept { return namespace_.c_str(); }

nvinfer1::DataType TileUnpackFusionPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const noexcept {
  if (index != 0 || input_types == nullptr || nb_inputs < 4) {
    return nvinfer1::DataType::kFLOAT;
  }
  return input_types[0];
}

nvinfer1::DimsExprs TileUnpackFusionPlugin::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) noexcept {
  (void)expr_builder;
  nvinfer1::DimsExprs out{};
  if (output_index != 0 || inputs == nullptr || (nb_inputs != 4 && nb_inputs != 5)) {
    return out;
  }
  // Output shape matches base map.
  out.nbDims = 4;
  out.d[0] = inputs[0].d[0];
  out.d[1] = inputs[0].d[1];
  out.d[2] = inputs[0].d[2];
  out.d[3] = inputs[0].d[3];
  return out;
}

bool TileUnpackFusionPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int nb_inputs,
    int nb_outputs) noexcept {
  if (in_out == nullptr || (nb_inputs != 4 && nb_inputs != 5) || nb_outputs != 1) {
    return false;
  }
  const auto& desc = in_out[pos];
  if (desc.format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }

  if (pos == 0) {
    return desc.type == nvinfer1::DataType::kHALF;
  }
  if (pos == 1) {
    return desc.type == in_out[0].type;
  }
  if (pos == 2 || pos == 3) {
    return desc.type == nvinfer1::DataType::kINT32;
  }
  if (nb_inputs == 5 && pos == 4) {
    return desc.type == in_out[0].type;
  }
  if (pos == nb_inputs) {
    return desc.type == in_out[0].type;
  }
  return false;
}

void TileUnpackFusionPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nb_outputs) noexcept {
  (void)in;
  (void)out;
  if (nb_outputs != 1) {
    return;
  }
  if (nb_inputs == 4 || nb_inputs == 5) {
    num_inputs_ = nb_inputs;
  }
}

size_t TileUnpackFusionPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int nb_inputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nb_outputs) const noexcept {
  (void)outputs;
  if (inputs == nullptr || nb_outputs != 1 || (nb_inputs != 4 && nb_inputs != 5)) {
    return 0;
  }
  if (inputs[0].dims.nbDims != 4) {
    return 0;
  }
  const int64_t batch = inputs[0].dims.d[0];
  const int64_t height = inputs[0].dims.d[2];
  const int64_t width = inputs[0].dims.d[3];
  if (batch <= 0 || height <= 0 || width <= 0) {
    return 0;
  }
  const int64_t pixels = batch * height * width;
  if (pixels <= 0 || pixels > (std::numeric_limits<int64_t>::max() / static_cast<int64_t>(sizeof(int32_t)))) {
    return 0;
  }
  return static_cast<size_t>(pixels) * sizeof(int32_t);
}

int TileUnpackFusionPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  if (input_desc == nullptr || output_desc == nullptr || inputs == nullptr || outputs == nullptr) {
    return -1;
  }
  if (num_inputs_ != 4 && num_inputs_ != 5) {
    return -1;
  }
  if (input_desc[0].dims.nbDims != 4 || input_desc[1].dims.nbDims != 5 ||
      input_desc[2].dims.nbDims != 2 || input_desc[3].dims.nbDims != 2 ||
      output_desc[0].dims.nbDims != 4) {
    return -1;
  }
  if (num_inputs_ == 5 && input_desc[4].dims.nbDims != 4) {
    return -1;
  }

  if (input_desc[0].type != nvinfer1::DataType::kHALF ||
      input_desc[1].type != nvinfer1::DataType::kHALF ||
      input_desc[2].type != nvinfer1::DataType::kINT32 ||
      input_desc[3].type != nvinfer1::DataType::kINT32 ||
      output_desc[0].type != nvinfer1::DataType::kHALF) {
    return -1;
  }
  if (num_inputs_ == 5 && input_desc[4].type != nvinfer1::DataType::kHALF) {
    return -1;
  }

  const int32_t batch = input_desc[0].dims.d[0];
  const int32_t channels = input_desc[0].dims.d[1];
  const int32_t height = input_desc[0].dims.d[2];
  const int32_t width = input_desc[0].dims.d[3];
  const int32_t packed_batch = input_desc[1].dims.d[0];
  const int32_t kmax = input_desc[1].dims.d[1];
  const int32_t packed_channels = input_desc[1].dims.d[2];
  const int32_t tile_h = input_desc[1].dims.d[3];
  const int32_t tile_w = input_desc[1].dims.d[4];
  if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0 || packed_batch != batch ||
      kmax < 0 || packed_channels != channels || tile_h <= 0 || tile_w <= 0 || tile_h != tile_w) {
    return -1;
  }
  if (input_desc[2].dims.d[0] != batch || input_desc[2].dims.d[1] != kmax ||
      input_desc[3].dims.d[0] != batch || input_desc[3].dims.d[1] != kmax) {
    return -1;
  }
  if (height % tile_h != 0 || width % tile_h != 0) {
    return -1;
  }

  if (output_desc[0].dims.d[0] != batch || output_desc[0].dims.d[1] != channels ||
      output_desc[0].dims.d[2] != height || output_desc[0].dims.d[3] != width) {
    return -1;
  }

  const int32_t order_bits = compute_order_bits(std::max<int32_t>(kmax, 1));
  if (order_bits >= 30) {
    return -1;
  }

  const size_t workspace_size = getWorkspaceSize(input_desc, num_inputs_, output_desc, 1);
  if (workspace_size > 0 && workspace == nullptr) {
    return -1;
  }
  const void* alpha_map = (num_inputs_ == 5) ? inputs[4] : nullptr;

  const cudaError_t status = launch_tileunpackfusion_fp16(
      inputs[0],
      inputs[1],
      static_cast<const int32_t*>(inputs[2]),
      static_cast<const int32_t*>(inputs[3]),
      alpha_map,
      outputs[0],
      workspace,
      batch,
      channels,
      height,
      width,
      kmax,
      tile_h,
      num_inputs_ == 5,
      stream);
  return status == cudaSuccess ? 0 : -1;
}

void TileUnpackFusionPlugin::attachToContext(
    cudnnContext* cudnn_context,
    cublasContext* cublas_context,
    nvinfer1::IGpuAllocator* gpu_allocator) noexcept {
  (void)cudnn_context;
  (void)cublas_context;
  (void)gpu_allocator;
}

void TileUnpackFusionPlugin::detachFromContext() noexcept {}

TileUnpackFusionPluginCreator::TileUnpackFusionPluginCreator() noexcept {
  field_collection_.nbFields = static_cast<int32_t>(fields_.size());
  field_collection_.fields = fields_.data();
}

const char* TileUnpackFusionPluginCreator::getPluginName() const noexcept {
  return kTileUnpackFusionPluginName;
}

const char* TileUnpackFusionPluginCreator::getPluginVersion() const noexcept {
  return kTileUnpackFusionPluginVersion;
}

const nvinfer1::PluginFieldCollection* TileUnpackFusionPluginCreator::getFieldNames() noexcept {
  return &field_collection_;
}

nvinfer1::IPluginV2* TileUnpackFusionPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) noexcept {
  (void)name;
  (void)fc;
  auto* plugin = new TileUnpackFusionPlugin();
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

nvinfer1::IPluginV2* TileUnpackFusionPluginCreator::deserializePlugin(
    const char* name,
    const void* serial_data,
    size_t serial_length) noexcept {
  (void)name;
  auto* plugin = new TileUnpackFusionPlugin(serial_data, serial_length);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void TileUnpackFusionPluginCreator::setPluginNamespace(const char* plugin_namespace) noexcept {
  namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
}

const char* TileUnpackFusionPluginCreator::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

}  // namespace apexx::trt::plugins

using ApexXTileUnpackFusionPluginCreator = apexx::trt::plugins::TileUnpackFusionPluginCreator;
REGISTER_TENSORRT_PLUGIN(ApexXTileUnpackFusionPluginCreator);

#endif  // APEXX_ENABLE_TENSORRT
