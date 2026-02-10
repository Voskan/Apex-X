#include "apexx_trt/decode_nms_plugin.hpp"

#include <algorithm>

namespace apexx::trt {

const char* DecodeNMSPlugin::name() const noexcept { return kPluginName; }

const char* DecodeNMSPlugin::contract() const noexcept {
  return "Optional DET fused decode + batched NMS plugin (stub contract)";
}

bool DecodeNMSPlugin::enqueue(
    const PluginEnqueueInputs& inputs,
    PluginEnqueueOutputs& outputs,
    std::string& error) const noexcept {
  if (inputs.tensors.empty()) {
    error = "DecodeNMSPlugin expects at least one input tensor";
    return false;
  }
  DummyTensor out = inputs.tensors.front();
  const std::size_t keep = std::min<std::size_t>(out.values.size(), 32U);
  out.values.resize(keep);
  out.shape = {static_cast<int64_t>(keep)};
  outputs.tensors = {out};
  error.clear();
  return true;
}

PluginStubPtr create_decode_nms_plugin() { return std::make_unique<DecodeNMSPlugin>(); }

}  // namespace apexx::trt
