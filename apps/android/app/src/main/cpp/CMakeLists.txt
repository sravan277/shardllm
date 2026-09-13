cmake_minimum_required(VERSION 3.22.1)

project(dllm_worker CXX)

# JNI stub for Phase 4. CPU-only, single-threaded sanity: no OpenMP/CUDA/Vulkan.
# Real llama.cpp static libs (llama/ggml/common, pinned tag — see
# ../../build-llama-ndk.ps1) are linked in Phase 5 by flipping
# DLLM_WITH_LLAMA=ON and pointing LLAMA_ANDROID_DIR at the build output.
option(DLLM_WITH_LLAMA "Link real llama.cpp static libs built by build-llama-ndk.ps1" OFF)
set(LLAMA_ANDROID_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/build-android-arm64"
    CACHE PATH "llama.cpp Android build output dir (libllama.a, libggml*.a, libcommon.a)")

add_library(dllm_worker SHARED worker.cpp)

find_library(log-lib log)
target_link_libraries(dllm_worker PRIVATE ${log-lib})

target_compile_features(dllm_worker PRIVATE cxx_std_17)
target_compile_options(dllm_worker PRIVATE -O2 -fvisibility=hidden)

if(DLLM_WITH_LLAMA)
    if(NOT EXISTS "${LLAMA_ANDROID_DIR}")
        message(FATAL_ERROR "DLLM_WITH_LLAMA=ON but LLAMA_ANDROID_DIR missing: ${LLAMA_ANDROID_DIR}. Run build-llama-ndk.ps1 first.")
    endif()
    # Phase 5 wiring point: add include dirs + link libllama/libggml/libcommon here.
    # target_include_directories(dllm_worker PRIVATE "${LLAMA_ANDROID_DIR}/../include" ...)
    # target_link_libraries(dllm_worker PRIVATE "${LLAMA_ANDROID_DIR}/libllama.a" ...)
endif()
