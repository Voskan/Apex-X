#include "apexx_trt/tile_pack_plugin.hpp"

namespace apexx::trt {

const char* TilePackPlugin::name() const noexcept { return kPluginName; }

const char* TilePackPlugin::contract() const noexcept {
  return "Inputs: F[B,C,Hf,Wf], idx[B,K] -> Outputs: P[B,K,C,t,t], meta";
}

bool TilePackPlugin::enqueue(
    const PluginEnqueueInputs& inputs,
    PluginEnqueueOutputs& outputs,
    std::string& error) const noexcept {
  if (inputs.tensors.empty()) {
    error = "TilePackPlugin expects at least one input tensor";
    return false;
  }
  outputs.tensors.clear();
  outputs.tensors.push_back(inputs.tensors.front());
  error.clear();
  return true;
}

PluginStubPtr create_tile_pack_plugin() { return std::make_unique<TilePackPlugin>(); }

}  // namespace apexx::trt
