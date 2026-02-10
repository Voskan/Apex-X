#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

#if defined(_WIN32)
using LibHandle = HMODULE;
#else
using LibHandle = void*;
#endif

using AbiVersionFn = int (*)();
using BuildSummaryFn = const char* (*)();
using InvokeMinimalFn = int (*)(const char*, const float*, int64_t, float*, int64_t);

std::string default_library_name() {
#if defined(_WIN32)
  return "apexx_trt_plugins.dll";
#elif defined(__APPLE__)
  return "libapexx_trt_plugins.dylib";
#else
  return "libapexx_trt_plugins.so";
#endif
}

LibHandle open_library(const std::string& path) {
#if defined(_WIN32)
  return LoadLibraryA(path.c_str());
#else
  return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void close_library(LibHandle handle) {
  if (handle == nullptr) {
    return;
  }
#if defined(_WIN32)
  FreeLibrary(handle);
#else
  dlclose(handle);
#endif
}

void* find_symbol(LibHandle handle, const char* name) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(handle, name));
#else
  return dlsym(handle, name);
#endif
}

std::string resolve_library_path(int argc, char** argv) {
  if (argc > 1) {
    return std::string(argv[1]);
  }
  if (const char* env_path = std::getenv("APEXX_TRT_PLUGIN_LIB"); env_path != nullptr) {
    return std::string(env_path);
  }
  return default_library_name();
}

}  // namespace

int main(int argc, char** argv) {
  const std::string lib_path = resolve_library_path(argc, argv);
  std::cout << "Loading plugin library: " << lib_path << "\n";

  LibHandle handle = open_library(lib_path);
  if (handle == nullptr) {
    std::cerr << "Failed to load plugin library from " << lib_path << "\n";
    return 2;
  }

  const auto abi_version = reinterpret_cast<AbiVersionFn>(find_symbol(handle, "apexx_trt_abi_version"));
  const auto build_summary =
      reinterpret_cast<BuildSummaryFn>(find_symbol(handle, "apexx_trt_build_summary_cstr"));
  const auto invoke_minimal =
      reinterpret_cast<InvokeMinimalFn>(find_symbol(handle, "apexx_trt_invoke_minimal"));
  if (abi_version == nullptr || build_summary == nullptr || invoke_minimal == nullptr) {
    std::cerr << "Failed to resolve required symbols from plugin library\n";
    close_library(handle);
    return 3;
  }

  std::cout << "ABI version: " << abi_version() << "\n";
  std::cout << build_summary() << "\n";

  const std::vector<std::string> plugin_names = {
      "TilePack",
      "TileSSMScan",
      "TileUnpackFusion",
      "DecodeNMS",
  };

  std::vector<float> input(64);
  for (std::size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i);
  }
  std::vector<float> output(128, 0.0F);

  for (const std::string& plugin_name : plugin_names) {
    const int count = invoke_minimal(
        plugin_name.c_str(),
        input.data(),
        static_cast<int64_t>(input.size()),
        output.data(),
        static_cast<int64_t>(output.size()));
    if (count < 0) {
      std::cerr << "Plugin minimal call failed for " << plugin_name << ", status=" << count
                << "\n";
      close_library(handle);
      return 4;
    }
    std::cout << plugin_name << " ok, output_count=" << count;
    if (count > 0) {
      std::cout << ", first=" << output.front();
    }
    std::cout << "\n";
  }

  close_library(handle);
  std::cout << "TensorRT plugin harness passed.\n";
  return 0;
}
