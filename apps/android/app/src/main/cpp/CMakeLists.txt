cmake_minimum_required(VERSION 3.22.1)

project(dllm_worker CXX)

# Phase 5: CPU-only, single-threaded worker linking real llama.cpp static libs
# (llama/ggml, pinned tag b7418 — see ../../build-llama-ndk.ps1).
# No OpenMP/CUDA/Vulkan.
# Flipped ON in Phase 5 (FORCE overrides a stale cached OFF from Phase 4 builds).
set(DLLM_WITH_LLAMA ON CACHE BOOL "Link real llama.cpp static libs built by build-llama-ndk.ps1" FORCE)
set(LLAMA_ANDROID_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/build-android-arm64"
    CACHE PATH "llama.cpp Android build output dir (libllama.a, libggml*.a, libcommon.a)")

# Repo-root native/include, which owns the cross-platform C ABI (dllm_shim_*).
# This is the single source of truth for the shim surface: Android includes the
# same header the Rust coordinator does, so the two workers cannot drift.
# apps/android/app/src/main/cpp -> up 6 = repo root.
set(DLLM_SHIM_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../native/include"
    CACHE PATH "Directory containing dllm_shim.h")

add_library(dllm_worker SHARED worker.cpp dllm_shim_android.cpp)

find_library(log-lib log)
target_link_libraries(dllm_worker PRIVATE ${log-lib})

target_compile_features(dllm_worker PRIVATE cxx_std_17)
target_compile_options(dllm_worker PRIVATE -O2 -fvisibility=hidden)

if(DLLM_WITH_LLAMA)
    if(NOT EXISTS "${LLAMA_ANDROID_DIR}")
        message(FATAL_ERROR "DLLM_WITH_LLAMA=ON but LLAMA_ANDROID_DIR missing: ${LLAMA_ANDROID_DIR}. Run build-llama-ndk.ps1 first.")
    endif()
    # ggml-rpc.h only exists in the vendored tree; libggml-rpc.a only exists when
    # build-llama-ndk.ps1 configured with -DGGML_RPC=ON. Fail loudly and point at
    # the script rather than letting the link die with an undefined reference to
    # ggml_backend_rpc_start_server - the symptom of a GGML_RPC=OFF build.
    set(RPC_LIB "${LLAMA_ANDROID_DIR}/ggml/src/ggml-rpc/libggml-rpc.a")
    if(NOT EXISTS "${RPC_LIB}")
        message(FATAL_ERROR "DLLM_WITH_LLAMA=ON but libggml-rpc.a missing: ${RPC_LIB}\n"
            "The phone hosts ggml-rpc layers only with GGML_RPC=ON. Re-run build-llama-ndk.ps1.")
    endif()
    # llama.h (include/) + ggml headers (ggml/include/).
    target_include_directories(dllm_worker PRIVATE
        "${LLAMA_ANDROID_DIR}/../include"
        "${LLAMA_ANDROID_DIR}/../ggml/include")
    # GGML_USE_RPC mirrors how ggml/src/ggml-backend-reg.cpp was compiled in
    # build-llama-ndk.ps1 (it is set by ggml's own CMake when GGML_RPC=ON). Any
    # header that branches on it must see the same value the library was built
    # with, otherwise this target can disagree with libggml-base.a about whether
    # the RPC backend exists.
    target_compile_definitions(dllm_worker PRIVATE GGML_USE_RPC=1)
    # Static order: worker -> llama -> ggml -> ggml-base <-> ggml-rpc <-> ggml-cpu.
    # ggml-rpc and ggml-cpu both call into ggml-base and each other (the registry
    # walks backends and calls dev ifaces), so the whole set stays wrapped in a
    # linker group for single-pass resolution; adding an archive outside the group
    # is what produces "undefined reference to ..." on the NDK linker.
    target_link_libraries(dllm_worker PRIVATE
        -Wl,--start-group
        "${LLAMA_ANDROID_DIR}/src/libllama.a"
        "${LLAMA_ANDROID_DIR}/ggml/src/libggml.a"
        "${LLAMA_ANDROID_DIR}/ggml/src/libggml-base.a"
        "${LLAMA_ANDROID_DIR}/ggml/src/ggml-rpc/libggml-rpc.a"
        "${LLAMA_ANDROID_DIR}/ggml/src/libggml-cpu.a"
        -Wl,--end-group)
endif()

# dllm_shim.h is required by dllm_shim_android.cpp regardless of DLLM_WITH_LLAMA:
# it is the ABI contract, not a llama.cpp header. Missing it means the repo layout
# moved, which is worth a loud failure instead of a confusing include error.
if(NOT EXISTS "${DLLM_SHIM_INCLUDE_DIR}/dllm_shim.h")
    message(FATAL_ERROR "dllm_shim.h not found in ${DLLM_SHIM_INCLUDE_DIR}. "
        "Set -DDLLM_SHIM_INCLUDE_DIR=<repo>/native/include.")
endif()
target_include_directories(dllm_worker PRIVATE "${DLLM_SHIM_INCLUDE_DIR}")