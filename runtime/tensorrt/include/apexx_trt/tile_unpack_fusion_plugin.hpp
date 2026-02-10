#pragma once

#include "apexx_trt/plugin_stub.hpp"

namespace apexx::trt {

class TileUnpackFusionPlugin final : public PluginStub {
 public:
  static constexpr const char* kPluginName = "TileUnpackFusion";

  const char* name() const noexcept override;
  const char* contract() const noexcept override;
  bool enqueue(
      const PluginEnqueueInputs& inputs,
      PluginEnqueueOutputs& outputs,
      std::string& error) const noexcept override;
};

PluginStubPtr create_tile_unpack_fusion_plugin();

}  // namespace apexx::trt
