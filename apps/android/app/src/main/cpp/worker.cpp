// Phase 5 JNI worker: real llama.cpp CPU decode behind the stable LlamaBridge API.
//
// Compiled by the NDK clang toolchain via CMake externalNativeBuild into
// libdllm_worker.so (arm64-v8a). Links the static libs built by
// build-llama-ndk.ps1 (llama/ggml, tag b7418, android-28, CPU-only).
// Single-threaded use: one mutex guards model lifetime + inference.
//
// ADR-025 locks loadModel/inferChunk/free as the stable local-decode ABI. The
// symbols below are ADDITIVE (ADR-031): they expose the shared dllm_shim C ABI
// so this phone can HOST ggml-rpc layers for a coordinator on another device.
// The three original entry points are unchanged, and they deliberately do NOT
// take g_mu - hosting and local decode are independent lifecycles.

#include <jni.h>

#include <android/log.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "dllm_shim.h"
#include "ggml-rpc.h"
#include "llama.h"

#define DLLM_TAG "dllm_worker"
#define DLLM_LOGI(...) __android_log_print(ANDROID_LOG_INFO, DLLM_TAG, __VA_ARGS__)
#define DLLM_LOGW(...) __android_log_print(ANDROID_LOG_WARN, DLLM_TAG, __VA_ARGS__)

// Implemented in dllm_shim_android.cpp, declared here because these three are
// Android-local additions to dllm_shim.h (see that file for why they are not in
// the shared header yet, and delete these declarations when it grows the hooks).
extern "C" int dllm_shim_rpc_serve_start(const char *host, int32_t port, const char *cache_dir,
                                         int32_t n_threads, int32_t n_devices);
extern "C" int dllm_shim_rpc_serve_stop(void);
extern "C" int dllm_shim_rpc_serving(void);

namespace {

// One big lock: model lifetime + inference are strictly serialized.
std::mutex g_mu;
std::string g_model_path;
bool g_loaded = false;
llama_model* g_model = nullptr;
llama_context* g_ctx = nullptr;
const llama_vocab* g_vocab = nullptr;

constexpr int kCtxTokens = 2048;
constexpr unsigned kMaxThreads = 4;

// Sampling defaults (Phase 5): temp 0.7 / top-p 0.8 / top-k 20.
constexpr int kTopK = 20;
constexpr float kTopP = 0.8f;
constexpr float kTemp = 0.7f;

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

// Must hold g_mu. Tears down ctx/model/backend.
void release_state_locked() {
    if (g_ctx != nullptr) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_vocab = nullptr;
    llama_backend_free();
    g_model_path.clear();
    g_loaded = false;
}

// UTF-8 -> UTF-16 so jni NewString sees valid code units (no modified-UTF-8 trap).
std::u16string utf8_to_utf16(const std::string& s) {
    std::u16string out;
    const size_t n = s.size();
    size_t i = 0;
    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp = 0;
        size_t len = 0;
        if (c < 0x80) {
            cp = c;
            len = 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < n) {
            cp = c & 0x1Fu;
            len = 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < n) {
            cp = c & 0x0Fu;
            len = 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < n) {
            cp = c & 0x07u;
            len = 4;
        } else {
            ++i;  // stray byte, skip
            continue;
        }
        if (i + len > n) break;
        bool ok = true;
        for (size_t j = 1; j < len; ++j) {
            const unsigned char cj = static_cast<unsigned char>(s[i + j]);
            if ((cj & 0xC0) != 0x80) {
                ok = false;
                break;
            }
            cp = (cp << 6) | (cj & 0x3Fu);
        }
        if (!ok) {
            ++i;
            continue;
        }
        i += len;
        if (cp >= 0x10000) {
            cp -= 0x10000;
            out.push_back(static_cast<char16_t>(0xD800 | (cp >> 10)));
            out.push_back(static_cast<char16_t>(0xDC00 | (cp & 0x3FF)));
        } else {
            out.push_back(static_cast<char16_t>(cp));
        }
    }
    return out;
}

jstring utf16_to_jstring(JNIEnv* env, const std::u16string& s) {
    return env->NewString(reinterpret_cast<const jchar*>(s.data()),
                          static_cast<jsize>(s.size()));
}

}  // namespace

extern "C" {

JNIEXPORT jboolean JNICALL Java_com_dllm_mesh_worker_LlamaBridge_loadModel(
    JNIEnv* env, jobject /*thiz*/, jstring jpath) {
    std::lock_guard<std::mutex> lock(g_mu);
    const std::string path = jstring_to_utf8(env, jpath);
    if (!file_readable(path)) {
        DLLM_LOGW("loadModel: not readable: %s", path.c_str());
        release_state_locked();
        return JNI_FALSE;
    }
    // Replace any previous model.
    release_state_locked();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU-only
    g_model = llama_model_load_from_file(path.c_str(), mparams);
    if (g_model == nullptr) {
        DLLM_LOGW("loadModel: llama_model_load_from_file failed: %s", path.c_str());
        llama_backend_free();
        g_model_path.clear();
        return JNI_FALSE;
    }
    g_vocab = llama_model_get_vocab(g_model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = kCtxTokens;
    cparams.n_batch = kCtxTokens;
    const unsigned hc = std::thread::hardware_concurrency();
    cparams.n_threads = hc > 0 ? static_cast<int>(std::min(hc, kMaxThreads)) : 2;
    cparams.n_threads_batch = cparams.n_threads;
    cparams.no_perf = true;
    g_ctx = llama_init_from_model(g_model, cparams);
    if (g_ctx == nullptr) {
        DLLM_LOGW("loadModel: llama_init_from_model failed: %s", path.c_str());
        llama_model_free(g_model);
        g_model = nullptr;
        g_vocab = nullptr;
        llama_backend_free();
        g_model_path.clear();
        return JNI_FALSE;
    }

    g_model_path = path;
    g_loaded = true;
    DLLM_LOGI("loadModel: loaded %s (n_ctx=%u threads=%d)", path.c_str(),
              llama_n_ctx(g_ctx), cparams.n_threads);
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL Java_com_dllm_mesh_worker_LlamaBridge_inferChunk(
    JNIEnv* env, jobject /*thiz*/, jstring jprompt, jint max_tokens) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_loaded || g_ctx == nullptr || g_vocab == nullptr || g_model == nullptr) {
        return utf16_to_jstring(env, u"");
    }
    const std::string prompt = jstring_to_utf8(env, jprompt);
    int cap = max_tokens > 0 ? max_tokens : 8;

    // Two-pass tokenize (add BOS/EOS when the model's vocab is configured for it).
    int n_prompt = llama_tokenize(g_vocab, prompt.c_str(),
                                  static_cast<int32_t>(prompt.size()), nullptr, 0,
                                  true, true);
    if (n_prompt < 0) n_prompt = -n_prompt;
    if (n_prompt <= 0) {
        DLLM_LOGW("inferChunk: empty prompt tokenization");
        return utf16_to_jstring(env, u"");
    }
    std::vector<llama_token> tokens(static_cast<size_t>(n_prompt));
    n_prompt = llama_tokenize(g_vocab, prompt.c_str(),
                              static_cast<int32_t>(prompt.size()), tokens.data(),
                              static_cast<int32_t>(tokens.size()), true, true);
    if (n_prompt < 0) {
        DLLM_LOGW("inferChunk: tokenize failed (%d)", n_prompt);
        return utf16_to_jstring(env, u"");
    }
    tokens.resize(static_cast<size_t>(n_prompt));

    // No context shifting in Phase 5: refuse prompts that cannot fit.
    const int32_t room = static_cast<int32_t>(llama_n_ctx(g_ctx)) - n_prompt;
    if (room <= 0) {
        DLLM_LOGW("inferChunk: prompt too long for n_ctx (%d tokens)", n_prompt);
        return utf16_to_jstring(env, u"");
    }
    cap = std::min(cap, room);

    // Each chunk is independent: reset KV cache between calls.
    llama_memory_clear(llama_get_memory(g_ctx), 0);

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(kTopK));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(kTopP, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(kTemp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string out;
    llama_batch batch = llama_batch_get_one(tokens.data(), n_prompt);
    int decoded = 0;
    for (int i = 0; i < cap; ++i) {
        if (llama_decode(g_ctx, batch) != 0) {
            DLLM_LOGW("inferChunk: llama_decode failed at step %d", i);
            break;
        }
        llama_token tok = llama_sampler_sample(smpl, g_ctx, -1);
        if (llama_vocab_is_eog(g_vocab, tok)) break;
        char piece[64];
        const int n_piece =
            llama_token_to_piece(g_vocab, tok, piece, sizeof(piece), 0, false);
        if (n_piece < 0) break;
        out.append(piece, static_cast<size_t>(n_piece));
        batch = llama_batch_get_one(&tok, 1);
        ++decoded;
    }
    llama_sampler_free(smpl);
    DLLM_LOGI("inferChunk: n_prompt=%d decoded=%d", n_prompt, decoded);
    return utf16_to_jstring(env, utf8_to_utf16(out));
}

JNIEXPORT void JNICALL
Java_com_dllm_mesh_worker_LlamaBridge_free(JNIEnv* /*env*/, jobject /*thiz*/) {
    std::lock_guard<std::mutex> lock(g_mu);
    release_state_locked();
    DLLM_LOGI("free: released");
}

// ---------------------------------------------------------------------------
// ADR-031: ggml-rpc hosting surface. Additive to the ADR-025 three above.
//
// Every function here is non-blocking. dllm_shim_rpc_serve() blocks forever (the
// shared ABI is "block until the server stops"), so dllm_shim_rpc_serve_start()
// runs it on a private C++ thread and returns only once the port is confirmed
// accepting connections. That keeps the heartbeat coroutine and the main thread
// free, and means the JNI layer never owns a thread it cannot join.
//
// None of these take g_mu: the local-decode mutex must not be held while the RPC
// server spends minutes computing an assigned layer range, or a chat request
// would block behind offloaded layers.
// ---------------------------------------------------------------------------

JNIEXPORT jint JNICALL
Java_com_dllm_mesh_worker_LlamaBridge_shimInit(JNIEnv* /*env*/, jobject /*thiz*/) {
    const int rc = dllm_shim_init();
    DLLM_LOGI("shimInit: rc=%d abi=%d", rc, dllm_shim_abi_version());
    return rc;
}

JNIEXPORT void JNICALL
Java_com_dllm_mesh_worker_LlamaBridge_shimFree(JNIEnv* /*env*/, jobject /*thiz*/) {
    // Stop hosting first: dllm_shim_free() documents "call once, at shutdown,
    // after every session is closed", and the serve thread is still a session.
    dllm_shim_rpc_serve_stop();
    dllm_shim_free();
    DLLM_LOGI("shimFree: done");
}

JNIEXPORT jint JNICALL
Java_com_dllm_mesh_worker_LlamaBridge_shimAbiVersion(JNIEnv* /*env*/, jobject /*thiz*/) {
    return dllm_shim_abi_version();
}

JNIEXPORT jstring JNICALL
Java_com_dllm_mesh_worker_LlamaBridge_shimLastError(JNIEnv* env, jobject /*thiz*/) {
    const char* err = dllm_shim_last_error();
    if (err == nullptr) return utf16_to_jstring(env, u"");
    return utf16_to_jstring(env, utf8_to_utf16(std::string(err)));
}

// host must be a literal IPv4 address ("192.168.1.42") or "0.0.0.0"; llama.cpp
// resolves it with inet_addr() and rejects anything else. nDevices < 0 means
// "expose every accelerator this device has" (CPU on this build). Returns 0 only
// once the port is really listening; negative means the endpoint is NOT usable.
JNIEXPORT jint JNICALL
Java_com_dllm_mesh_worker_LlamaBridge_rpcServeStart(JNIEnv* env, jobject /*thiz*/,
                                                    jstring jhost, jint port,
                                                    jint nThreads, jint nDevices) {
    const std::string host = jstring_to_utf8(env, jhost);
    if (host.empty()) {
        DLLM_LOGW("rpcServeStart: empty host");
        return -1;
    }
    const int rc = dllm_shim_rpc_serve_start(host.c_str(), port, nullptr,
                                             nThreads, nDevices);
    if (rc != 0) {
        DLLM_LOGW("rpcServeStart: %s:%d failed rc=%d (%s)", host.c_str(), port, rc,
                  dllm_shim_last_error());
    } else {
        DLLM_LOGI("rpcServeStart: %s:%d hosting", host.c_str(), port);
    }
    return rc;
}

JNIEXPORT jint JNICALL
Java_com_dllm_mesh_worker_LlamaBridge_rpcServeStop(JNIEnv* /*env*/, jobject /*thiz*/) {
    const int rc = dllm_shim_rpc_serve_stop();
    DLLM_LOGI("rpcServeStop: rc=%d (endpoint withdrawn)", rc);
    return rc;
}

JNIEXPORT jboolean JNICALL
Java_com_dllm_mesh_worker_LlamaBridge_rpcServing(JNIEnv* /*env*/, jobject /*thiz*/) {
    return dllm_shim_rpc_serving() != 0 ? JNI_TRUE : JNI_FALSE;
}

// Read from ggml-rpc.h — the same header libggml-rpc.a was compiled from —
// rather than a hand-copied string. The pinned tag moves; a stale literal here
// would tell the coordinator this worker speaks a protocol it does not.
JNIEXPORT jstring JNICALL
Java_com_dllm_mesh_worker_LlamaBridge_rpcProtocolVersion(JNIEnv* env, jobject /*thiz*/) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%d.%d.%d", RPC_PROTO_MAJOR_VERSION,
             RPC_PROTO_MINOR_VERSION, RPC_PROTO_PATCH_VERSION);
    return utf16_to_jstring(env, utf8_to_utf16(std::string(buf)));
}

}  // extern "C"
