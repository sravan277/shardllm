# dllm_shim — links llama.cpp + ggml-rpc and exposes the C ABI in include/dllm_shim.h
#
# This is the ONLY place that includes llama.cpp/ggml headers. Rust and JNI see
# just dllm_shim.h, which keeps the C++ toolchain requirement confined to this
# target and lets the shim be rebuilt without touching the crates.
#
# GGML_RPC is REQUIRED, not optional: without it `libggml-rpc.a` is not built,
# `ggml_backend_rpc_add_server` does not exist, and the entire point of this
# target (executing layers on another device) silently disappears. A configure
# without it is a hard error rather than a warning, because the failure mode is
# otherwise a shim that builds fine and offloads nothing.

cmake_minimum_required(VERSION 3.22)
project(dllm_shim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED LLAMA_SOURCE_DIR)
  set(LLAMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp")
endif()

if(NOT EXISTS "${LLAMA_SOURCE_DIR}/ggml/src/ggml-rpc/ggml-rpc.cpp")
  message(FATAL_ERROR
    "llama.cpp not found at ${LLAMA_SOURCE_DIR}.\n"
    "Run scripts/build-llama-win.ps1 first — it clones the pinned tag and "
    "configures with GGML_RPC=ON.")
endif()

# Build llama.cpp as a subproject with RPC enabled. BUILD_SHARED_LIBS=OFF so we
# get single .a files to link into one DLL instead of a runtime DLL zoo.
set(BUILD_SHARED_LIBS        OFF CACHE BOOL "" FORCE)
set(GGML_RPC                ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS       OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL              OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE             OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama" EXCLUDE_FROM_ALL)

# libggml-rpc is the piece that makes offload possible. Fail loudly if the
# option silently did not take effect.
if(NOT TARGET ggml-rpc)
  message(FATAL_ERROR
    "ggml-rpc target missing. GGML_RPC=ON was requested but the target was not "
    "created — check that the llama.cpp source tree actually contains "
    "ggml/src/ggml-rpc/.")
endif()

add_library(dllm_shim SHARED src/dllm_shim.cpp)

target_include_directories(dllm_shim
  PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${LLAMA_SOURCE_DIR}/include"
          "${LLAMA_SOURCE_DIR}/ggml/include"
          "${LLAMA_SOURCE_DIR}/ggml/src"
          # llama-model.h is an internal (non-installed) header. It is the only
          # way to read back the layer->device assignment llama.cpp actually
          # made (llama_model::dev_layer) instead of restating the requested
          # tensor_split. Stable across the b7418 pin; revisit on every bump.
          "${LLAMA_SOURCE_DIR}/src")

# Order matters on MinGW: llama -> ggml -> ggml-base -> ggml-rpc -> ggml-cpu.
# The start/end group lets the archives resolve circular references between them.
target_link_libraries(dllm_shim PRIVATE
  -Wl,--start-group
    llama ggml ggml-base ggml-rpc ggml-cpu
  -Wl,--end-group)

if(MSVC)
  target_compile_options(dllm_shim PRIVATE /W3)
else()
  target_compile_options(dllm_shim PRIVATE -O2 -fvisibility=hidden -Wall -Wextra)
  # -fvisibility=hidden does NOT hide anything in a PE/COFF DLL: ld exports every
  # defined symbol by default, and -Wl,--exclude-all-symbols is ignored by
  # mingw-w64 ld in this configuration. Without an explicit export table the
  # shim re-exports all of llama.cpp, ggml and (via auto-import) part of the CRT
  # -- ~3400 symbols. That is not cosmetic: the host links llama-cpp-2, which
  # statically links its OWN vendored copy of llama.cpp, and Windows resolves
  # imports by bare name, so a leaked `llama_decode` can bind to the wrong build.
  # dllm_shim.def (below) is what actually pins the export table.
endif()

# ------------------------------------------------------------- export table ---
# Generated from include/dllm_shim.h rather than hand-written, so the DLL can
# only ever export what the header declares. A hand-maintained .def drifts, and
# the drift is invisible until something binds to the wrong llama.cpp.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/dllm_shim.h" _dllm_shim_header)
string(REGEX MATCHALL "DLLM_SHIM_API[ \t\r\n]+[A-Za-z_][A-Za-z0-9_ \t\r\n]*[ \t\r\n*]+(dllm_shim_[A-Za-z0-9_]+)[ \t\r\n]*\\("
       _dllm_shim_matches "${_dllm_shim_header}")
list(LENGTH _dllm_shim_matches _dllm_shim_declared)
if(_dllm_shim_declared EQUAL 0)
  message(FATAL_ERROR
    "Parsed zero DLLM_SHIM_API declarations out of include/dllm_shim.h. The "
    "export-table regex has drifted from the header; fix it in native/CMakeLists.txt.")
endif()
list(TRANSFORM _dllm_shim_matches REPLACE
     ".*(dllm_shim_[A-Za-z0-9_]+)[ \t\r\n]*\\(" "\\1")
set(_dllm_shim_names "${_dllm_shim_matches}")
list(REMOVE_DUPLICATES _dllm_shim_names)
list(LENGTH _dllm_shim_names _dllm_shim_unique)
if(NOT _dllm_shim_unique EQUAL _dllm_shim_declared)
  message(FATAL_ERROR
    "Export-table mismatch: ${_dllm_shim_declared} DLLM_SHIM_API declarations in "
    "the header but ${_dllm_shim_unique} unique names. Refusing to link a DLL with "
    "a silently wrong export table.")
endif()
message(STATUS "dllm_shim export table: ${_dllm_shim_unique} symbols")

set(_dllm_shim_def "${CMAKE_CURRENT_BINARY_DIR}/dllm_shim.def")
set(_dllm_shim_def_body "EXPORTS\n")
foreach(_name IN LISTS _dllm_shim_names)
  string(APPEND _dllm_shim_def_body "  ${_name}\n")
endforeach()
file(GENERATE OUTPUT "${_dllm_shim_def}" CONTENT "${_dllm_shim_def_body}")
target_sources(dllm_shim PRIVATE "${_dllm_shim_def}")

set_target_properties(dllm_shim PROPERTIES
  # CMake defaults to a "lib" prefix for DLLs on this toolchain; drop it so the
  # artifact is dllm_shim.dll and matches what a Rust cdylib loader expects.
  PREFIX ""
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

# --------------------------------------------------------------------- tests
# End-to-end proof that a layer actually executes on an RPC worker: hosts a
# worker in a thread, registers it, then runs a session whose tensor_split spans
# local CPU + that worker. Off by default because it needs a real .gguf.
option(DLLM_SHIM_BUILD_TESTS "Build the native end-to-end tests" OFF)

if(DLLM_SHIM_BUILD_TESTS)
  add_executable(rpc_local tests/rpc_local.cpp)
  target_link_libraries(rpc_local PRIVATE dllm_shim)
  if(NOT MSVC)
    target_compile_options(rpc_local PRIVATE -O2 -Wall -Wextra)
    # The test opens a loopback socket to its own worker; MinGW has no
    # #pragma comment(lib) equivalent, so the import library is explicit.
    target_link_libraries(rpc_local PRIVATE ws2_32)
  endif()
  enable_testing()
  add_test(NAME rpc_local COMMAND rpc_local)
endif()