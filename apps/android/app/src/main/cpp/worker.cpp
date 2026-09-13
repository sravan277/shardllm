// Phase 5 JNI worker: real llama.cpp CPU decode behind the stable LlamaBridge API.
//
// Compiled by the NDK clang toolchain via CMake externalNativeBuild into
// libdllm_worker.so (arm64-v8a). Links the static libs built by
// build-llama-ndk.ps1 (llama/ggml, tag b7418, android-28, CPU-only).
// Single-threaded use: one mutex guards model lifetime + inference.

#include <jni.h>

#include <android/log.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "llama.h"

#define DLLM_TAG "dllm_worker"
#define DLLM_LOGI(...) __android_log_print(ANDROID_LOG_INFO, DLLM_TAG, __VA_ARGS__)
#define DLLM_LOGW(...) __android_log_print(ANDROID_LOG_WARN, DLLM_TAG, __VA_ARGS__)

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

}  // extern "C"
