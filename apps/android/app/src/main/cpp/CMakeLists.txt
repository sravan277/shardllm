cmake_minimum_required(VERSION 3.22.1)

project(dllm_worker CXX)

# Phase 5: CPU-only, single-threaded worker linking real llama.cpp static libs
# (llama/ggml, pinned tag b7418 — see ../../build-llama-ndk.ps1).
# No OpenMP/CUDA/Vulkan.
# Flipped ON in Phase 5 (FORCE overrides a stale cached OFF from Phase 4 builds).
set(DLLM_WITH_LLAMA ON CACHE BOOL "Link real llama.cpp static libs built by build-llama-ndk.ps1" FORCE)
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
    # llama.h (include/) + ggml headers (ggml/include/).
    target_include_directories(dllm_worker PRIVATE
        "${LLAMA_ANDROID_DIR}/../include"
        "${LLAMA_ANDROID_DIR}/../ggml/include")
    # Static order: worker -> llama -> ggml -> ggml-base <-> ggml-cpu.
    # ggml-base and ggml-cpu reference each other (backend registry), so the
    # whole set is wrapped in a linker group for single-pass resolution.
    target_link_libraries(dllm_worker PRIVATE
        -Wl,--start-group
        "${LLAMA_ANDROID_DIR}/src/libllama.a"
        "${LLAMA_ANDROID_DIR}/ggml/src/libggml.a"
        "${LLAMA_ANDROID_DIR}/ggml/src/libggml-base.a"
        "${LLAMA_ANDROID_DIR}/ggml/src/libggml-cpu.a"
        -Wl,--end-group)
endif()
