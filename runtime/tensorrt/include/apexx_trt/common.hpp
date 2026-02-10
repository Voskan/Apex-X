#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace apexx::trt {

#if defined(_WIN32)
#if defined(APEXX_TRT_BUILD_SHARED)
#define APEXX_TRT_API __declspec(dllexport)
#elif defined(APEXX_TRT_USE_SHARED)
#define APEXX_TRT_API __declspec(dllimport)
#else
#define APEXX_TRT_API
#endif
#else
#define APEXX_TRT_API
#endif

struct PluginBuildInfo {
  std::string name;
  bool tensorrt_enabled;
  bool cuda_enabled;
};

std::vector<PluginBuildInfo> list_plugin_build_info();
std::string build_summary();
bool invoke_minimal_call_path(
    const std::string& plugin_name,
    const std::vector<float>& input,
    std::vector<float>& output,
    std::string* error);

}  // namespace apexx::trt

extern "C" {
APEXX_TRT_API int apexx_trt_abi_version();
APEXX_TRT_API const char* apexx_trt_build_summary_cstr();
APEXX_TRT_API int apexx_trt_invoke_minimal(
    const char* plugin_name,
    const float* input,
    int64_t input_size,
    float* output,
    int64_t output_size);
}
