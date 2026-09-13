// Phase 4 JNI stub: CPU-only, single-threaded sanity bridge.
//
// Compiled by the NDK clang toolchain via CMake externalNativeBuild into
// libdllm_worker.so (arm64-v8a). No OpenMP/CUDA/Vulkan. Real llama.cpp
// decode (llama/ggml/common static libs from build-llama-ndk.ps1) plugs in
// behind these same three symbols in Phase 5 — Kotlin API is stable.

#include <jni.h>

#include <android/log.h>

#include <cstdio>
#include <mutex>
#include <string>

#define DLLM_TAG "dllm_worker"
#define DLLM_LOGI(...) __android_log_print(ANDROID_LOG_INFO, DLLM_TAG, __VA_ARGS__)
#define DLLM_LOGW(...) __android_log_print(ANDROID_LOG_WARN, DLLM_TAG, __VA_ARGS__)

namespace {

// Guarded by g_mu. Single-threaded use: callers hold the lock for the whole call.
std::mutex g_mu;
std::string g_model_path;
bool g_loaded = false;

std::string jstring_to_utf8(JNIEnv* env, jstring jstr) {
    if (jstr == nullptr) return {};
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    if (chars == nullptr) return {};
    std::string out(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return out;
}

bool file_readable(const std::string& path) {
    if (path.empty()) return false;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    std::fclose(f);
    return true;
}

}  // namespace

extern "C" {

JNIEXPORT jboolean JNICALL Java_com_dllm_mesh_worker_LlamaBridge_loadModel(
    JNIEnv* env, jobject /*thiz*/, jstring jpath) {
    std::lock_guard<std::mutex> lock(g_mu);
    const std::string path = jstring_to_utf8(env, jpath);
    if (!file_readable(path)) {
        DLLM_LOGW("loadModel: not readable: %s", path.c_str());
        g_loaded = false;
        g_model_path.clear();
        return JNI_FALSE;
    }
    // Stub: readability check only. Phase 5 parses the GGUF header + inits backend here.
    g_model_path = path;
    g_loaded = true;
    DLLM_LOGI("loadModel: staged (stub, no decode yet): %s", path.c_str());
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL Java_com_dllm_mesh_worker_LlamaBridge_inferChunk(
    JNIEnv* env, jobject /*thiz*/, jstring jprompt, jint max_tokens) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_loaded) {
        return env->NewStringUTF("[dllm-stub] no model loaded");
    }
    const std::string prompt = jstring_to_utf8(env, jprompt);
    const int cap = max_tokens > 0 ? max_tokens : 8;
    // Stub-streaming: deterministic echo so the Kotlin/WorkerService round-trip
    // is verifiable before real sampling lands in Phase 5.
    char buf[1024];
    std::snprintf(buf, sizeof(buf), "[dllm-stub] n=%d model=%s prompt=%.512s",
                  cap, g_model_path.c_str(), prompt.c_str());
    return env->NewStringUTF(buf);
}

JNIEXPORT void JNICALL
Java_com_dllm_mesh_worker_LlamaBridge_free(JNIEnv* /*env*/, jobject /*thiz*/) {
    std::lock_guard<std::mutex> lock(g_mu);
    // Phase 5: llama_free / backend teardown goes here.
    g_model_path.clear();
    g_loaded = false;
    DLLM_LOGI("free: released (stub)");
}

}  // extern "C"
