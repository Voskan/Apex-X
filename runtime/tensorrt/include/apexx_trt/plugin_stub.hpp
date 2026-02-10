#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace apexx::trt {

struct DummyTensor {
  std::vector<int64_t> shape;
  std::vector<float> values;

  [[nodiscard]] int64_t numel() const noexcept {
    int64_t total = 1;
    for (const int64_t dim : shape) {
      if (dim <= 0) {
        return 0;
      }
      total *= dim;
    }
    return total;
  }
};

struct PluginEnqueueInputs {
  std::vector<DummyTensor> tensors;
};

struct PluginEnqueueOutputs {
  std::vector<DummyTensor> tensors;
};

class PluginStub {
 public:
  virtual ~PluginStub() = default;
  virtual const char* name() const noexcept = 0;
  virtual const char* contract() const noexcept = 0;
  virtual bool enqueue(
      const PluginEnqueueInputs& inputs,
      PluginEnqueueOutputs& outputs,
      std::string& error) const noexcept = 0;

  static constexpr bool tensorrt_enabled() noexcept {
#if APEXX_ENABLE_TENSORRT
    return true;
#else
    return false;
#endif
  }

  static constexpr bool cuda_enabled() noexcept {
#if APEXX_ENABLE_CUDA
    return true;
#else
    return false;
#endif
  }
};

using PluginStubPtr = std::unique_ptr<PluginStub>;

}  // namespace apexx::trt
