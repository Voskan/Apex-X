#pragma once

#include "apexx_trt/plugin_stub.hpp"

namespace apexx::trt {

class DecodeNMSPlugin final : public PluginStub {
 public:
  static constexpr const char* kPluginName = "DecodeNMS";

  const char* name() const noexcept override;
  const char* contract() const noexcept override;
  bool enqueue(
      const PluginEnqueueInputs& inputs,
      PluginEnqueueOutputs& outputs,
      std::string& error) const noexcept override;
};

PluginStubPtr create_decode_nms_plugin();

}  // namespace apexx::trt
