/* rpc_local — end-to-end proof that the shim can put layers on an RPC worker.
 *
 * Everything happens on this one machine, which is the point: the RPC protocol is
 * exercised over a real loopback socket, so a passing run means activations
 * genuinely travelled host->worker->host, not that a mock was called.
 *
 * Sequence:
 *   1. resolve the model from %LOCALAPPDATA%\dllm\models (path overridable)
 *   2. start a worker in a background thread via dllm_shim_rpc_serve
 *   3. wait for the port to accept connections
 *   4. dllm_shim_add_rpc_server("127.0.0.1:<port>")
 *   5. open a session with tensor_split {1,1} (local CPU + the worker)
 *   6. generate, print tokens, print dllm_shim_session_report
 *
 * Exit codes: 0 tokens came back. 2 setup failed (no model / no worker).
 * 1 generate failed. 0 tokens despite a successful generate.
 */

#include "dllm_shim.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

namespace {

struct TokenSink {
    std::string         text;
    std::vector<int32_t> positions;
    int32_t             done_calls = 0;
    int32_t             stop_after = -1;   /* -1 = never stop early */
};

int on_token(void *user, int32_t pos, const char *utf8, int32_t len, int32_t done) {
    TokenSink *sink = static_cast<TokenSink *>(user);
    if (done) {
        sink->done_calls += 1;
        return 0;
    }
    if (utf8 && len > 0) {
        sink->text.append(utf8, (size_t) len);
    }
    sink->positions.push_back(pos);
    if (sink->stop_after > 0 && (int32_t) sink->positions.size() >= sink->stop_after) {
        return 1;   /* non-zero stops generation early */
    }
    return 0;
}

bool model_path(std::string &out) {
    if (const char * env = std::getenv("DLLM_TEST_MODEL")) {
        if (env[0] != '\0') {
            out = env;
            return true;
        }
    }
    const char * local = std::getenv("LOCALAPPDATA");
    if (!local) {
        local = std::getenv("HOME");
    }
    if (!local) {
        return false;
    }
    out = std::string(local) + "\\dllm\\models\\Qwen3-0.6B-Q4_K_M.gguf";
    return true;
}

bool file_exists(const std::string &p) {
    FILE *f = std::fopen(p.c_str(), "rb");
    if (!f) {
        return false;
    }
    std::fclose(f);
    return true;
}

bool port_is_open(const char *host, int port) {
    const int family = AF_INET;
    int fd = (int) socket(family, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }
    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((unsigned short) port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        closesocket(fd);
        return false;
    }
    const int rc = connect(fd, (sockaddr *) &addr, sizeof(addr));
    closesocket(fd);
    return rc == 0;
}

int find_free_port() {
    /* Bind to port 0, read the assigned port back, close. Racy in theory, fine
     * for a single-shot local test; a fixed port would collide with a worker
     * left over from a previous run. */
    const int fd = (int) socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    if (bind(fd, (sockaddr *) &addr, sizeof(addr)) != 0 || listen(fd, 1) != 0) {
        closesocket(fd);
        return -1;
    }
    int       len    = (int) sizeof(addr);
    sockaddr_in bound;
    std::memset(&bound, 0, sizeof(bound));
    if (getsockname(fd, (sockaddr *) &bound, &len) != 0) {
        closesocket(fd);
        return -1;
    }
    const int port = ntohs(bound.sin_port);
    closesocket(fd);
    return port;
}

} /* namespace */

int main() {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    printf("dllm_shim abi version: %d\n", dllm_shim_abi_version());

    /* ---- 1. model ---- */
    std::string model;
    if (!model_path(model)) {
        fprintf(stderr, "FAIL: cannot resolve a model path; set DLLM_TEST_MODEL\n");
        return 2;
    }
    if (!file_exists(model)) {
        fprintf(stderr, "FAIL: model not found at %s\n"
                        "      Download it from contracts/catalog.json "
                        "(unsloth/Qwen3-0.6B-GGUF / Qwen3-0.6B-Q4_K_M.gguf) or set "
                        "DLLM_TEST_MODEL.\n",
                model.c_str());
        return 2;
    }
    printf("model: %s\n", model.c_str());

    /* ---- 2. init ---- */
    if (dllm_shim_init() != 0) {
        fprintf(stderr, "FAIL: dllm_shim_init: %s\n", dllm_shim_last_error());
        return 2;
    }
    printf("shim init: ok\n");

    /* ---- 3. worker on a background thread ---- */
    const int port = find_free_port();
    if (port <= 0) {
        fprintf(stderr, "FAIL: could not find a free TCP port\n");
        return 2;
    }
    const int n_threads = 4;
    printf("worker: starting ggml-rpc server on 127.0.0.1:%d (%d threads, all devices)\n",
           port, n_threads);

    std::atomic<int> serve_rc{-1};
    std::thread      worker([&] {
        /* n_devices = -1 -> expose every accelerator (falls back to CPU). */
        serve_rc.store(dllm_shim_rpc_serve("127.0.0.1", port, nullptr, n_threads, -1));
    });

    bool up = false;
    for (int i = 0; i < 100 && !up; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        up = port_is_open("127.0.0.1", port);
    }
    if (!up) {
        fprintf(stderr, "FAIL: worker never started listening on 127.0.0.1:%d\n", port);
        worker.detach();
        return 2;
    }
    printf("worker: listening\n");

    /* ---- 4. register it ---- */
    const std::string endpoint = "127.0.0.1:" + std::to_string(port);
    const int         n_rpc    = dllm_shim_add_rpc_server(endpoint.c_str());
    if (n_rpc < 0) {
        fprintf(stderr, "FAIL: dllm_shim_add_rpc_server(%s): %s\n",
                endpoint.c_str(), dllm_shim_last_error());
        worker.detach();
        return 2;
    }
    printf("registered %s: %d RPC device(s), %d total\n",
           endpoint.c_str(), n_rpc, dllm_shim_rpc_device_count());

    for (int32_t i = 0; i < n_rpc; ++i) {
        size_t fre = 0, tot = 0;
        if (dllm_shim_rpc_device_memory(i, &fre, &tot) == 0) {
            printf("rpc device %d memory: %.1f MiB free / %.1f MiB total\n",
                   (int) i, fre / 1048576.0, tot / 1048576.0);
        } else {
            printf("rpc device %d memory: FAILED (%s)\n", (int) i, dllm_shim_last_error());
        }
    }

    /* ---- 5. session spanning local + worker ---- */
    const float split[2] = {1.0f, 1.0f};
    printf("opening session with tensor_split {1,1}, n_gpu_layers=-1, n_ctx=2048 ...\n");
    dllm_shim_session *sess = dllm_shim_session_open(model.c_str(), split, 2, -1, 2048);
    if (!sess) {
        fprintf(stderr, "FAIL: dllm_shim_session_open: %s\n", dllm_shim_last_error());
        worker.detach();
        return 2;
    }

    int32_t n_layer = 0, n_ctx = 0;
    dllm_shim_session_n_layer(sess, &n_layer);
    dllm_shim_session_n_ctx(sess, &n_ctx);
    printf("session: n_layer=%d n_ctx=%d\n", (int) n_layer, (int) n_ctx);

    /* ---- 6. generate ---- */
    dllm_shim_sampler smp;
    smp.top_k            = 20;
    smp.top_p            = 0.8f;
    smp.temp             = 0.7f;
    smp.min_p            = 0.0f;
    smp.presence_penalty = 1.5f;
    smp.seed             = -1;

    TokenSink sink;
    sink.stop_after = 24;

    const char *prompt = "The capital of France is";
    printf("prompt: %s\n", prompt);

    const auto t0 = std::chrono::steady_clock::now();
    const int rc = dllm_shim_session_generate(sess, prompt, 32, &smp, on_token, &sink);
    const double wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    if (rc != 0) {
        fprintf(stderr, "FAIL: dllm_shim_session_generate: %s\n", dllm_shim_last_error());
        dllm_shim_session_close(sess);
        worker.detach();
        return 1;
    }

    printf("tokens (%zu, %.1f ms wall): %s\n",
           sink.positions.size(), wall_ms, sink.text.c_str());
    printf("final callbacks (must be exactly 1): %d\n", (int) sink.done_calls);

    char report[2048];
    if (dllm_shim_session_report(sess, report, (int32_t) sizeof(report)) == 0) {
        printf("report: %s\n", report);
    } else {
        printf("report: FAILED (%s)\n", dllm_shim_last_error());
    }

    dllm_shim_session_close(sess);

    const bool ok = !sink.positions.empty() && sink.text.size() > 0 && sink.done_calls == 1;
    if (!ok) {
        fprintf(stderr, "FAIL: tokens=%zu text=%zu done_calls=%d\n",
                sink.positions.size(), sink.text.size(), (int) sink.done_calls);
    } else {
        printf("PASS: %zu tokens streamed across local CPU + RPC worker\n",
               sink.positions.size());
    }

    /* The worker blocks until stopped and the ABI has no way to stop it, so the
     * process exits rather than joining. */
    worker.detach();

#ifdef _WIN32
    WSACleanup();
#endif
    return ok ? 0 : 1;
}
