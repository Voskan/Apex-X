#include "apexx_trt/tile_ssm_scan_plugin.hpp"

#include <algorithm>

namespace apexx::trt {

const char* TileSSMScanPlugin::name() const noexcept { return kPluginName; }

const char* TileSSMScanPlugin::contract() const noexcept {
  return "Inputs: packed tile tokens + optional state -> Outputs: mixed tokens + next state";
}

bool TileSSMScanPlugin::enqueue(
    const PluginEnqueueInputs& inputs,
    PluginEnqueueOutputs& outputs,
    std::string& error) const noexcept {
  if (inputs.tensors.empty()) {
    error = "TileSSMScanPlugin expects at least one input tensor";
    return false;
  }
  const DummyTensor& in = inputs.tensors.front();
  DummyTensor out = in;
  if (out.values.empty()) {
    error.clear();
    outputs.tensors = {out};
    return true;
  }

  float state = 0.0F;
  constexpr float kDecay = 0.9F;
  constexpr float kOneMinusDecay = 1.0F - kDecay;
  for (float& value : out.values) {
    const float sanitized = std::clamp(value, -1.0e4F, 1.0e4F);
    state = kDecay * state + kOneMinusDecay * sanitized;
    value = state;
  }
  outputs.tensors = {out};
  error.clear();
  return true;
}

PluginStubPtr create_tile_ssm_scan_plugin() { return std::make_unique<TileSSMScanPlugin>(); }

}  // namespace apexx::trt
