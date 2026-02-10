#include "apexx_trt/tile_unpack_fusion_plugin.hpp"

#include <algorithm>

namespace apexx::trt {

const char* TileUnpackFusionPlugin::name() const noexcept { return kPluginName; }

const char* TileUnpackFusionPlugin::contract() const noexcept {
  return "Inputs: F_base, P_out, meta, optional gate -> Output: merged map with priority semantics";
}

bool TileUnpackFusionPlugin::enqueue(
    const PluginEnqueueInputs& inputs,
    PluginEnqueueOutputs& outputs,
    std::string& error) const noexcept {
  if (inputs.tensors.empty()) {
    error = "TileUnpackFusionPlugin expects at least one input tensor";
    return false;
  }
  DummyTensor out = inputs.tensors.front();
  for (float& value : out.values) {
    value = std::clamp(value, -1.0e4F, 1.0e4F);
  }
  outputs.tensors = {out};
  error.clear();
  return true;
}

PluginStubPtr create_tile_unpack_fusion_plugin() {
  return std::make_unique<TileUnpackFusionPlugin>();
}

}  // namespace apexx::trt
