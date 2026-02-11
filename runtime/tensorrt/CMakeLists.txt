cmake_minimum_required(VERSION 3.20)

project(apexx_tensorrt_plugins LANGUAGES CXX)
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(APEXX_ENABLE_TENSORRT "Enable TensorRT bindings when headers are present" ON)
option(APEXX_ENABLE_CUDA "Enable CUDA-aware plugin paths when CUDA is available" ON)
option(APEXX_BUILD_PLUGIN_TEST_HARNESS "Build plugin test harness executable" ON)
option(APEXX_ENABLE_REAL_TILEPACK_PLUGIN "Enable real TensorRT TilePack plugin build" ON)
option(
  APEXX_ENABLE_REAL_TILEUNPACKFUSION_PLUGIN
  "Enable real TensorRT TileUnpackFusion plugin build"
  ON
)
option(
  APEXX_ENABLE_REAL_TILESSM_PLUGIN
  "Enable real TensorRT TileSSMScan plugin build"
  ON
)
option(
  APEXX_ENABLE_REAL_NMS_DECODE_PLUGIN
  "Enable real TensorRT Decode+NMS plugin build"
  ON
)

set(APEXX_HAS_TENSORRT 0)
if(APEXX_ENABLE_TENSORRT)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
  if(TENSORRT_INCLUDE_DIR)
    set(APEXX_HAS_TENSORRT 1)
    message(STATUS "TensorRT headers found at: ${TENSORRT_INCLUDE_DIR}")
  else()
    message(WARNING "TensorRT headers not found. Shared plugin library will be skipped.")
  endif()
endif()

set(APEXX_HAS_CUDA 0)
if(APEXX_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(APEXX_HAS_CUDA 1)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
  else()
    message(WARNING "CUDA compiler not found. Shared plugin library will be skipped.")
  endif()
endif()

set(APEXX_PLUGIN_SOURCES
  src/common.cpp
  src/tile_pack_plugin.cpp
  src/tile_ssm_scan_plugin.cpp
  src/tile_unpack_fusion_plugin.cpp
  src/decode_nms_plugin.cpp
)

add_library(apexx_trt_plugin_core STATIC ${APEXX_PLUGIN_SOURCES})
set_target_properties(apexx_trt_plugin_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(apexx_trt_plugin_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(APEXX_HAS_TENSORRT)
  target_include_directories(apexx_trt_plugin_core PRIVATE ${TENSORRT_INCLUDE_DIR})
endif()

target_compile_definitions(apexx_trt_plugin_core
  PUBLIC
    APEXX_ENABLE_TENSORRT=${APEXX_HAS_TENSORRT}
    APEXX_ENABLE_CUDA=${APEXX_HAS_CUDA}
)

set(APEXX_CAN_BUILD_PLUGIN_SHARED 0)
if(APEXX_HAS_TENSORRT AND APEXX_HAS_CUDA)
  set(APEXX_CAN_BUILD_PLUGIN_SHARED 1)
endif()

set(APEXX_HAS_TENSORRT_RUNTIME_LIBS 0)
if(APEXX_CAN_BUILD_PLUGIN_SHARED)
  find_library(
    TENSORRT_NVINFER_LIB
    NAMES nvinfer
    HINTS ENV TENSORRT_ROOT
    PATH_SUFFIXES lib lib64
  )
  find_library(
    TENSORRT_NVINFER_PLUGIN_LIB
    NAMES nvinfer_plugin
    HINTS ENV TENSORRT_ROOT
    PATH_SUFFIXES lib lib64
  )
  find_library(
    CUDA_CUDART_LIB
    NAMES cudart
    HINTS ENV CUDA_HOME
    PATH_SUFFIXES lib64 lib x64
  )

  if(TENSORRT_NVINFER_LIB AND CUDA_CUDART_LIB)
    set(APEXX_HAS_TENSORRT_RUNTIME_LIBS 1)
  else()
    message(
      WARNING
      "TensorRT/CUDA runtime libraries not found (nvinfer/cudart). "
      "Shared plugin library will be skipped."
    )
  endif()
endif()

set(APEXX_BUILD_REAL_TILEPACK 0)
if(APEXX_CAN_BUILD_PLUGIN_SHARED AND APEXX_HAS_TENSORRT_RUNTIME_LIBS AND APEXX_ENABLE_REAL_TILEPACK_PLUGIN)
  set(APEXX_BUILD_REAL_TILEPACK 1)
endif()

set(APEXX_BUILD_REAL_TILEUNPACKFUSION 0)
if(APEXX_CAN_BUILD_PLUGIN_SHARED AND APEXX_HAS_TENSORRT_RUNTIME_LIBS AND APEXX_ENABLE_REAL_TILEUNPACKFUSION_PLUGIN)
  set(APEXX_BUILD_REAL_TILEUNPACKFUSION 1)
endif()

set(APEXX_BUILD_REAL_TILESSM 0)
if(APEXX_CAN_BUILD_PLUGIN_SHARED AND APEXX_HAS_TENSORRT_RUNTIME_LIBS AND APEXX_ENABLE_REAL_TILESSM_PLUGIN)
  set(APEXX_BUILD_REAL_TILESSM 1)
endif()

set(APEXX_BUILD_REAL_NMS_DECODE 0)
if(APEXX_CAN_BUILD_PLUGIN_SHARED AND APEXX_HAS_TENSORRT_RUNTIME_LIBS AND APEXX_ENABLE_REAL_NMS_DECODE_PLUGIN)
  set(APEXX_BUILD_REAL_NMS_DECODE 1)
endif()

if(APEXX_CAN_BUILD_PLUGIN_SHARED AND APEXX_HAS_TENSORRT_RUNTIME_LIBS)
  add_library(apexx_trt_plugins SHARED ${APEXX_PLUGIN_SOURCES})
  target_include_directories(apexx_trt_plugins PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_include_directories(apexx_trt_plugins PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(apexx_trt_plugins PRIVATE ${TENSORRT_NVINFER_LIB} ${CUDA_CUDART_LIB})
  if(TENSORRT_NVINFER_PLUGIN_LIB)
    target_link_libraries(apexx_trt_plugins PRIVATE ${TENSORRT_NVINFER_PLUGIN_LIB})
  endif()

  target_include_directories(apexx_trt_plugins PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/plugins)

  if(APEXX_BUILD_REAL_TILEPACK)
    target_sources(
      apexx_trt_plugins
      PRIVATE
        plugins/tilepack.cpp
        plugins/tilepack.cu
    )
  endif()

  if(APEXX_BUILD_REAL_TILEUNPACKFUSION)
    target_sources(
      apexx_trt_plugins
      PRIVATE
        plugins/tileunpackfusion.cpp
        plugins/tileunpackfusion.cu
    )
  endif()

  if(APEXX_BUILD_REAL_TILESSM)
    target_sources(
      apexx_trt_plugins
      PRIVATE
        plugins/tilessm_scan.cpp
        plugins/tilessm_scan.cu
    )
  endif()

  if(APEXX_BUILD_REAL_NMS_DECODE)
    target_sources(
      apexx_trt_plugins
      PRIVATE
        plugins/nms_decode.cpp
        plugins/nms_decode.cu
    )
  endif()

  target_compile_definitions(
    apexx_trt_plugins
    PUBLIC
      APEXX_ENABLE_TENSORRT=1
      APEXX_ENABLE_CUDA=1
    PRIVATE
      APEXX_TRT_BUILD_SHARED=1
      APEXX_ENABLE_REAL_TILEPACK_PLUGIN=${APEXX_BUILD_REAL_TILEPACK}
      APEXX_ENABLE_REAL_TILEUNPACKFUSION_PLUGIN=${APEXX_BUILD_REAL_TILEUNPACKFUSION}
      APEXX_ENABLE_REAL_TILESSM_PLUGIN=${APEXX_BUILD_REAL_TILESSM}
      APEXX_ENABLE_REAL_NMS_DECODE_PLUGIN=${APEXX_BUILD_REAL_NMS_DECODE}
  )
  set_target_properties(apexx_trt_plugins PROPERTIES OUTPUT_NAME "apexx_trt_plugins")
else()
  message(
    STATUS
      "Skipping shared TensorRT plugin library "
      "(requires TensorRT headers, CUDA compiler, and nvinfer/cudart libs)."
  )
endif()

add_executable(apexx_trt_plugin_info src/plugin_info_main.cpp)
target_link_libraries(apexx_trt_plugin_info PRIVATE apexx_trt_plugin_core)

if(APEXX_CAN_BUILD_PLUGIN_SHARED AND APEXX_HAS_TENSORRT_RUNTIME_LIBS AND APEXX_BUILD_PLUGIN_TEST_HARNESS)
  add_executable(apexx_trt_plugin_harness tests/plugin_harness_main.cpp)
  if(CMAKE_DL_LIBS)
    target_link_libraries(apexx_trt_plugin_harness PRIVATE ${CMAKE_DL_LIBS})
  endif()
endif()

function(apexx_register_trt_test target_name test_name)
  if(NOT BUILD_TESTING)
    return()
  endif()
  if(NOT TARGET ${target_name})
    return()
  endif()
  add_test(NAME ${test_name} COMMAND ${target_name})

  if(UNIX AND NOT APPLE)
    set(_apexx_ld_paths "${CMAKE_CURRENT_BINARY_DIR}")
    if(TENSORRT_NVINFER_LIB)
      get_filename_component(_nvinfer_dir "${TENSORRT_NVINFER_LIB}" DIRECTORY)
      list(APPEND _apexx_ld_paths "${_nvinfer_dir}")
      get_filename_component(_nvinfer_real "${TENSORRT_NVINFER_LIB}" REALPATH)
      get_filename_component(_nvinfer_real_dir "${_nvinfer_real}" DIRECTORY)
      list(APPEND _apexx_ld_paths "${_nvinfer_real_dir}")
    endif()
    if(TENSORRT_NVINFER_PLUGIN_LIB)
      get_filename_component(_nvinfer_plugin_dir "${TENSORRT_NVINFER_PLUGIN_LIB}" DIRECTORY)
      list(APPEND _apexx_ld_paths "${_nvinfer_plugin_dir}")
      get_filename_component(_nvinfer_plugin_real "${TENSORRT_NVINFER_PLUGIN_LIB}" REALPATH)
      get_filename_component(_nvinfer_plugin_real_dir "${_nvinfer_plugin_real}" DIRECTORY)
      list(APPEND _apexx_ld_paths "${_nvinfer_plugin_real_dir}")
    endif()
    if(CUDA_CUDART_LIB)
      get_filename_component(_cudart_dir "${CUDA_CUDART_LIB}" DIRECTORY)
      list(APPEND _apexx_ld_paths "${_cudart_dir}")
    endif()
    list(REMOVE_DUPLICATES _apexx_ld_paths)
    string(REPLACE ";" ":" _apexx_ld_path "${_apexx_ld_paths}")
    set_tests_properties(
      ${test_name}
      PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${_apexx_ld_path}:$ENV{LD_LIBRARY_PATH}"
    )
  endif()
endfunction()

if(APEXX_BUILD_REAL_TILEPACK)
  add_executable(apexx_trt_tilepack_test tests/tilepack_plugin_test.cpp)
  target_include_directories(
    apexx_trt_tilepack_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/plugins
      ${TENSORRT_INCLUDE_DIR}
  )
  target_link_libraries(
    apexx_trt_tilepack_test
    PRIVATE
      apexx_trt_plugins
      ${TENSORRT_NVINFER_LIB}
      ${CUDA_CUDART_LIB}
  )
  target_compile_definitions(apexx_trt_tilepack_test PRIVATE APEXX_ENABLE_REAL_TILEPACK_PLUGIN=1)
  if(TENSORRT_NVINFER_PLUGIN_LIB)
    target_link_libraries(apexx_trt_tilepack_test PRIVATE ${TENSORRT_NVINFER_PLUGIN_LIB})
  endif()
  apexx_register_trt_test(apexx_trt_tilepack_test trt_tilepack_plugin_native)
endif()

if(APEXX_BUILD_REAL_TILEUNPACKFUSION)
  add_executable(apexx_trt_tileunpackfusion_test tests/tileunpackfusion_plugin_test.cpp)
  target_include_directories(
    apexx_trt_tileunpackfusion_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/plugins
      ${TENSORRT_INCLUDE_DIR}
  )
  target_link_libraries(
    apexx_trt_tileunpackfusion_test
    PRIVATE
      apexx_trt_plugins
      ${TENSORRT_NVINFER_LIB}
      ${CUDA_CUDART_LIB}
  )
  target_compile_definitions(
    apexx_trt_tileunpackfusion_test
    PRIVATE
      APEXX_ENABLE_REAL_TILEUNPACKFUSION_PLUGIN=1
  )
  if(TENSORRT_NVINFER_PLUGIN_LIB)
    target_link_libraries(apexx_trt_tileunpackfusion_test PRIVATE ${TENSORRT_NVINFER_PLUGIN_LIB})
  endif()
  apexx_register_trt_test(apexx_trt_tileunpackfusion_test trt_tileunpackfusion_plugin_native)
endif()

if(APEXX_BUILD_REAL_TILESSM)
  add_executable(apexx_trt_tilessm_test tests/tilessm_scan_plugin_test.cpp)
  target_include_directories(
    apexx_trt_tilessm_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/plugins
      ${TENSORRT_INCLUDE_DIR}
  )
  target_link_libraries(
    apexx_trt_tilessm_test
    PRIVATE
      apexx_trt_plugins
      ${TENSORRT_NVINFER_LIB}
      ${CUDA_CUDART_LIB}
  )
  target_compile_definitions(
    apexx_trt_tilessm_test
    PRIVATE
      APEXX_ENABLE_REAL_TILESSM_PLUGIN=1
  )
  if(TENSORRT_NVINFER_PLUGIN_LIB)
    target_link_libraries(apexx_trt_tilessm_test PRIVATE ${TENSORRT_NVINFER_PLUGIN_LIB})
  endif()
  apexx_register_trt_test(apexx_trt_tilessm_test trt_tilessm_plugin_native)
endif()

if(APEXX_BUILD_REAL_NMS_DECODE)
  add_executable(apexx_trt_nms_decode_test tests/nms_decode_plugin_test.cpp)
  target_include_directories(
    apexx_trt_nms_decode_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/plugins
      ${TENSORRT_INCLUDE_DIR}
  )
  target_link_libraries(
    apexx_trt_nms_decode_test
    PRIVATE
      apexx_trt_plugins
      ${TENSORRT_NVINFER_LIB}
      ${CUDA_CUDART_LIB}
  )
  target_compile_definitions(
    apexx_trt_nms_decode_test
    PRIVATE
      APEXX_ENABLE_REAL_NMS_DECODE_PLUGIN=1
  )
  if(TENSORRT_NVINFER_PLUGIN_LIB)
    target_link_libraries(apexx_trt_nms_decode_test PRIVATE ${TENSORRT_NVINFER_PLUGIN_LIB})
  endif()
  apexx_register_trt_test(apexx_trt_nms_decode_test trt_nms_decode_plugin_native)
endif()
