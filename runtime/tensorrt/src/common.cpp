#include "apexx_trt/common.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>

#include "apexx_trt/decode_nms_plugin.hpp"
#include "apexx_trt/tile_pack_plugin.hpp"
#include "apexx_trt/tile_ssm_scan_plugin.hpp"
#include "apexx_trt/tile_unpack_fusion_plugin.hpp"

namespace apexx::trt {

std::vector<PluginBuildInfo> list_plugin_build_info() {
  return {
      PluginBuildInfo{
          TilePackPlugin::kPluginName,
          PluginStub::tensorrt_enabled(),
          PluginStub::cuda_enabled(),
      },
      PluginBuildInfo{
          TileSSMScanPlugin::kPluginName,
          PluginStub::tensorrt_enabled(),
          PluginStub::cuda_enabled(),
      },
      PluginBuildInfo{
          TileUnpackFusionPlugin::kPluginName,
          PluginStub::tensorrt_enabled(),
          PluginStub::cuda_enabled(),
      },
      PluginBuildInfo{
          DecodeNMSPlugin::kPluginName,
          PluginStub::tensorrt_enabled(),
          PluginStub::cuda_enabled(),
      },
  };
}

std::string build_summary() {
  std::ostringstream out;
  out << "apexx_tensorrt_plugins build summary\n";
  out << "  TensorRT enabled: " << (PluginStub::tensorrt_enabled() ? "yes" : "no") << "\n";
  out << "  CUDA enabled: " << (PluginStub::cuda_enabled() ? "yes" : "no") << "\n";
  for (const PluginBuildInfo& info : list_plugin_build_info()) {
    out << "  - " << info.name << " (trt=" << (info.tensorrt_enabled ? "1" : "0")
        << ", cuda=" << (info.cuda_enabled ? "1" : "0") << ")\n";
  }
  return out.str();
}

namespace {

PluginStubPtr create_plugin_by_name(const std::string& plugin_name) {
  if (plugin_name == TilePackPlugin::kPluginName) {
    return create_tile_pack_plugin();
  }
  if (plugin_name == TileSSMScanPlugin::kPluginName) {
    return create_tile_ssm_scan_plugin();
  }
  if (plugin_name == TileUnpackFusionPlugin::kPluginName) {
    return create_tile_unpack_fusion_plugin();
  }
  if (plugin_name == DecodeNMSPlugin::kPluginName) {
    return create_decode_nms_plugin();
  }
  return nullptr;
}

}  // namespace

bool invoke_minimal_call_path(
    const std::string& plugin_name,
    const std::vector<float>& input,
    std::vector<float>& output,
    std::string* error) {
  PluginStubPtr plugin = create_plugin_by_name(plugin_name);
  if (!plugin) {
    if (error != nullptr) {
      *error = "unknown plugin name: " + plugin_name;
    }
    return false;
  }

  PluginEnqueueInputs inputs;
  PluginEnqueueOutputs outputs;
  inputs.tensors.push_back(DummyTensor{{static_cast<int64_t>(input.size())}, input});

  std::string local_error;
  const bool ok = plugin->enqueue(inputs, outputs, local_error);
  if (!ok || outputs.tensors.empty()) {
    if (error != nullptr) {
      *error = ok ? "enqueue returned no outputs" : local_error;
    }
    return false;
  }

  output = outputs.tensors.front().values;
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

}  // namespace apexx::trt

extern "C" {

int apexx_trt_abi_version() { return 1; }

const char* apexx_trt_build_summary_cstr() {
  static std::string summary = apexx::trt::build_summary();
  return summary.c_str();
}

int apexx_trt_invoke_minimal(
    const char* plugin_name,
    const float* input,
    int64_t input_size,
    float* output,
    int64_t output_size) {
  if (plugin_name == nullptr || input == nullptr || output == nullptr || input_size < 0 ||
      output_size < 0) {
    return -2;
  }

  std::vector<float> in(static_cast<std::size_t>(input_size));
  std::memcpy(
      in.data(),
      input,
      static_cast<std::size_t>(input_size) * sizeof(float));

  std::vector<float> out;
  std::string error;
  const bool ok = apexx::trt::invoke_minimal_call_path(plugin_name, in, out, &error);
  if (!ok) {
    return -3;
  }
  if (static_cast<int64_t>(out.size()) > output_size) {
    return -4;
  }

  std::memcpy(output, out.data(), out.size() * sizeof(float));
  return static_cast<int>(out.size());
}

}  // extern "C"
