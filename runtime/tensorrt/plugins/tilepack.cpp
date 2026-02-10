#include "tilepack.h"

#if APEXX_ENABLE_TENSORRT

#include <algorithm>
#include <cstdint>
#include <cstring>

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

}  // namespace

TilePackPlugin::TilePackPlugin(int32_t tile_size) noexcept : tile_size_(std::max<int32_t>(1, tile_size)) {}

TilePackPlugin::TilePackPlugin(const void* serial_data, size_t serial_length) noexcept {
  const void* cursor = serial_data;
  const void* end = static_cast<const char*>(serial_data) + serial_length;
  tile_size_ = std::max<int32_t>(1, read_int32(cursor, end));
}

const char* TilePackPlugin::getPluginType() const noexcept { return kTilePackPluginName; }

const char* TilePackPlugin::getPluginVersion() const noexcept { return kTilePackPluginVersion; }

int TilePackPlugin::getNbOutputs() const noexcept { return 1; }

int TilePackPlugin::initialize() noexcept { return 0; }

void TilePackPlugin::terminate() noexcept {}

size_t TilePackPlugin::getSerializationSize() const noexcept { return sizeof(int32_t); }

void TilePackPlugin::serialize(void* buffer) const noexcept {
  if (buffer == nullptr) {
    return;
  }
  void* cursor = buffer;
  write_int32(cursor, tile_size_);
}

void TilePackPlugin::destroy() noexcept { delete this; }

nvinfer1::IPluginV2DynamicExt* TilePackPlugin::clone() const noexcept {
  auto* plugin = new TilePackPlugin(tile_size_);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void TilePackPlugin::setPluginNamespace(const char* plugin_namespace) noexcept {
  namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
}

const char* TilePackPlugin::getPluginNamespace() const noexcept { return namespace_.c_str(); }

nvinfer1::DataType TilePackPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* input_types,
    int nb_inputs) const noexcept {
  if (index != 0 || input_types == nullptr || nb_inputs < 1) {
    return nvinfer1::DataType::kFLOAT;
  }
  return input_types[0];
}

nvinfer1::DimsExprs TilePackPlugin::getOutputDimensions(
    int output_index,
    const nvinfer1::DimsExprs* inputs,
    int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) noexcept {
  nvinfer1::DimsExprs out{};
  if (output_index != 0 || inputs == nullptr || nb_inputs != 2) {
    return out;
  }
  out.nbDims = 5;
  out.d[0] = inputs[0].d[0];  // B
  out.d[1] = inputs[1].d[1];  // K
  out.d[2] = inputs[0].d[1];  // C
  out.d[3] = expr_builder.constant(tile_size_);
  out.d[4] = expr_builder.constant(tile_size_);
  return out;
}

bool TilePackPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int nb_inputs,
    int nb_outputs) noexcept {
  if (in_out == nullptr || nb_inputs != 2 || nb_outputs != 1) {
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
    return desc.type == nvinfer1::DataType::kINT32;
  }
  if (pos == 2) {
    return desc.type == in_out[0].type;
  }
  return false;
}

void TilePackPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nb_outputs) noexcept {
  if (in == nullptr || out == nullptr || nb_inputs != 2 || nb_outputs != 1) {
    return;
  }
}

size_t TilePackPlugin::getWorkspaceSize(
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

int TilePackPlugin::enqueue(
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
  if (input_desc[0].dims.nbDims != 4 || input_desc[1].dims.nbDims != 2 ||
      output_desc[0].dims.nbDims != 5) {
    return -1;
  }
  if (input_desc[0].type != nvinfer1::DataType::kHALF ||
      input_desc[1].type != nvinfer1::DataType::kINT32 ||
      output_desc[0].type != nvinfer1::DataType::kHALF) {
    return -1;
  }

  const int32_t batch = input_desc[0].dims.d[0];
  const int32_t channels = input_desc[0].dims.d[1];
  const int32_t height = input_desc[0].dims.d[2];
  const int32_t width = input_desc[0].dims.d[3];
  const int32_t kmax = input_desc[1].dims.d[1];
  if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0 || kmax < 0) {
    return -1;
  }

  const cudaError_t status = launch_tilepack_fp16(
      inputs[0],
      static_cast<const int32_t*>(inputs[1]),
      outputs[0],
      batch,
      channels,
      height,
      width,
      kmax,
      tile_size_,
      stream);
  return status == cudaSuccess ? 0 : -1;
}

void TilePackPlugin::attachToContext(
    cudnnContext* cudnn_context,
    cublasContext* cublas_context,
    nvinfer1::IGpuAllocator* gpu_allocator) noexcept {
  (void)cudnn_context;
  (void)cublas_context;
  (void)gpu_allocator;
}

void TilePackPlugin::detachFromContext() noexcept {}

TilePackPluginCreator::TilePackPluginCreator() noexcept {
  fields_.emplace_back(
      nvinfer1::PluginField{"tile_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
  field_collection_.nbFields = static_cast<int32_t>(fields_.size());
  field_collection_.fields = fields_.data();
}

const char* TilePackPluginCreator::getPluginName() const noexcept { return kTilePackPluginName; }

const char* TilePackPluginCreator::getPluginVersion() const noexcept {
  return kTilePackPluginVersion;
}

const nvinfer1::PluginFieldCollection* TilePackPluginCreator::getFieldNames() noexcept {
  return &field_collection_;
}

nvinfer1::IPluginV2* TilePackPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) noexcept {
  (void)name;
  int32_t tile_size = 8;
  if (fc != nullptr) {
    for (int32_t i = 0; i < fc->nbFields; ++i) {
      const nvinfer1::PluginField& field = fc->fields[i];
      if (field.name == nullptr || field.data == nullptr) {
        continue;
      }
      if (std::string(field.name) == "tile_size" &&
          field.type == nvinfer1::PluginFieldType::kINT32 && field.length >= 1) {
        tile_size = *static_cast<const int32_t*>(field.data);
      }
    }
  }
  auto* plugin = new TilePackPlugin(tile_size);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

nvinfer1::IPluginV2* TilePackPluginCreator::deserializePlugin(
    const char* name,
    const void* serial_data,
    size_t serial_length) noexcept {
  (void)name;
  auto* plugin = new TilePackPlugin(serial_data, serial_length);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void TilePackPluginCreator::setPluginNamespace(const char* plugin_namespace) noexcept {
  namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
}

const char* TilePackPluginCreator::getPluginNamespace() const noexcept { return namespace_.c_str(); }

}  // namespace apexx::trt::plugins

REGISTER_TENSORRT_PLUGIN(apexx::trt::plugins::TilePackPluginCreator);

#endif  // APEXX_ENABLE_TENSORRT
