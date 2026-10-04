// Android implementation of the worker-hosting subset of the shared dllm_shim
// C ABI (native/include/dllm_shim.h).
//
// WHY THIS FILE EXISTS
// --------------------
// `native/src/dllm_shim.cpp` (the desktop/Rust implementation the shim header
// promises) is owned by another crew and does not exist yet, and the Android app
// cannot link a target outside its own CMake project. So this file implements
// exactly the shim symbols the phone needs - the same names, the same
// signatures, the same 0/negative return convention - against the SAME vendored
// llama.cpp the desktop shim will use. The JNI layer (worker.cpp) therefore
// cannot tell which implementation it is bound to, which is the whole point of
// the ABI: one coordinator drives either platform.
//
// When native/src/dllm_shim.cpp lands, DELETE this file and add it to
// app/src/main/cpp/CMakeLists.txt instead. The symbols are identical, so nothing
// else has to change. Do NOT keep both: duplicate definitions of the same
// extern "C" names.
//
// Requires GGML_RPC=ON in build-llama-ndk.ps1. Without it this file does not
// even compile (ggml-rpc.h is absent) and libggml-rpc.a is not linked, so the
// phone cannot host a single layer.

#include "dllm_shim.h"

#include <android/log.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"

#define SHIM_TAG "dllm_worker"
#define SHIM_LOGI(...) __android_log_print(ANDROID_LOG_INFO, SHIM_TAG, __VA_ARGS__)
#define SHIM_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, SHIM_TAG, __VA_ARGS__)

// Bumped on any ABI break; MUST stay equal to native/src/dllm_shim.cpp's value
// or the coordinator's startup handshake (dllm_shim_abi_version) will reject the
// worker instead of crashing it.
#define DLLM_SHIM_ABI_VERSION 1

// How long dllm_shim_rpc_serve_start waits for the listening socket to appear.
// Generous because the first start also pays for device/backend init; the wait
// is only reached when something is actually wrong (port busy, bad interface).
constexpr int kServeReadyTimeoutMs = 5000;
constexpr int kServeReadyPollMs = 50;

// Set when the listener has been observed accepting connections AND no stop has
// been requested. This - not "we asked for a listener" - is what the app
// advertises, so a coordinator is never pointed at a dead port.
std::atomic<bool> g_serving{false};
std::atomic<bool> g_stop_requested{false};
std::atomic<bool> g_init_done{false};
std::atomic<bool> g_start_in_flight{false};

// dllm_shim_last_error is documented as per-thread and owned by the callee.
thread_local std::string g_last_error;

namespace {

void set_error(const std::string& msg) {
    g_last_error = msg;
    SHIM_LOGE("%s", msg.c_str());
}

// Devices to expose, mirroring tools/rpc/rpc-server.cpp exactly: accelerators
// first, CPU as the fallback. The Android build is CPU-only (no Vulkan/CUDA in
// build-llama-ndk.ps1), so in practice this is always the single CPU device -
// but the selection is written out so a Vulkan phone build exposes the GPU
// without a second code path.
std::vector<ggml_backend_dev_t> select_devices(int32_t n_devices) {
    std::vector<ggml_backend_dev_t> devices;
    if (n_devices >= 0) {
        // Explicit count: take the first N in enumeration order. Only used by a
        // caller that knows what it wants; the app always passes -1.
        for (size_t i = 0; i < ggml_backend_dev_count() && static_cast<int32_t>(i) < n_devices; ++i) {
            devices.push_back(ggml_backend_dev_get(i));
        }
        return devices;
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            devices.push_back(dev);
        }
    }
    if (devices.empty()) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu != nullptr) {
            devices.push_back(cpu);
        }
    }
    return devices;
}

// True once something is accepting connections on `probe_host`:`port`.
//
// WHY a real connect() probe instead of "the thread started, assume it bound":
// ggml_backend_rpc_start_server() prints to stdout and returns nothing, so there
// is no callback to hang readiness off. Advertising an endpoint before the
// listen() has happened is exactly the kind of optimistic claim the rest of this
// app refuses to make (see WorkerCapabilities' placeholders) - the coordinator
// would race us and get a connection refused. connect() returning 0 proves the
// listening socket exists, which is the fact we actually want to report.
//
// The probe connection is closed immediately; the server logs
// "Accepted client connection" / "Client connection closed" for it and goes back
// to accept(). It never receives a command, so rpc_serve_client() bails on the
// first read and no work is scheduled.
bool probe_listening(const std::string& host, int port) {
    // A socket bound to 0.0.0.0 is reachable on loopback, so probe there; any
    // other bind address must be dialled on that exact address.
    const std::string target = (host == "0.0.0.0" || host.empty()) ? std::string("127.0.0.1") : host;

    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, target.c_str(), &addr.sin_addr) != 1) {
        ::close(fd);
        return false;
    }
    const bool up = ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    ::close(fd);
    return up;
}

}  // namespace

extern "C" {

int dllm_shim_init(void) {
    // Idempotent per the header. Touching ggml_backend_dev_count() forces
    // ggml_backend_registry's lazy constructor to run, which registers the CPU
    // backend and - because build-llama-ndk.ps1 compiles ggml-backend-reg.cpp
    // with -DGGML_USE_RPC - the RPC backend too. ggml_backend_load_all() is
    // deliberately NOT called: it only scans for dynamically loadable backends,
    // and on Android that is a wasted directory walk with no possible payoff.
    if (!g_init_done.exchange(true)) {
        ggml_backend_dev_count();
        SHIM_LOGI("dllm_shim_init: abi=%d backends registered", DLLM_SHIM_ABI_VERSION);
    }
    return 0;
}

void dllm_shim_free(void) {
    // Per the header this is a shutdown call: every session closed, serve
    // stopped. We have nothing global to release - the registry is a function
    // static and the serve thread owns its own backends.
    g_init_done = false;
    SHIM_LOGI("dllm_shim_free");
}

const char* dllm_shim_last_error(void) {
    return g_last_error.c_str();
}

int dllm_shim_abi_version(void) {
    return DLLM_SHIM_ABI_VERSION;
}

int dllm_shim_rpc_serve(const char* host, int32_t port, const char* cache_dir,
                        int32_t n_threads, int32_t n_devices) {
    if (g_init_done.load() == false) {
        dllm_shim_init();
    }
    if (host == nullptr || port <= 0 || port > 65535) {
        set_error("dllm_shim_rpc_serve: invalid endpoint");
        return -1;
    }
    if (g_serving.load()) {
        set_error("dllm_shim_rpc_serve: already serving");
        return -2;
    }

    std::vector<ggml_backend_dev_t> devices = select_devices(n_devices);
    if (devices.empty()) {
        set_error("dllm_shim_rpc_serve: no ggml backend devices to expose");
        return -3;
    }
    for (ggml_backend_dev_t dev : devices) {
        SHIM_LOGI("dllm_shim_rpc_serve: exposing device %s", ggml_backend_dev_name(dev));
    }

    size_t threads = n_threads > 0 ? static_cast<size_t>(n_threads) : 1;
    std::string endpoint = std::string(host) + ":" + std::to_string(port);

    g_stop_requested = false;
    g_start_in_flight = true;

    // Upstream llama.cpp: the listening socket is a function-local shared_ptr
    // inside an unconditional `while (true) { accept(); serve(); }` and
    // rpc_serve_client() is static. There is no exported stop, no socket handle
    // and no way to unblock accept() from outside - so this call does not return
    // until the process dies. Anything claiming otherwise on b7418 is wrong.
    SHIM_LOGI("dllm_shim_rpc_serve: serving %s (proto %d.%d.%d, %zu thread(s), %zu device(s))",
              endpoint.c_str(), RPC_PROTO_MAJOR_VERSION, RPC_PROTO_MINOR_VERSION,
              RPC_PROTO_PATCH_VERSION, threads, devices.size());
    ggml_backend_rpc_start_server(endpoint.c_str(), cache_dir, threads,
                                  devices.size(), devices.data());

    // Reached only if the upstream loop ever breaks (accept() error), or after
    // process teardown. Not a clean-shutdown path.
    g_serving = false;
    g_start_in_flight = false;
    set_error("dllm_shim_rpc_serve: server loop exited unexpectedly");
    return -4;
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Android-local extras. NOT declared in dllm_shim.h, which this crew does not
// own. They exist because dllm_shim_rpc_serve() blocks forever by design (the
// shared ABI is "block until the server stops") and the phone must be able to
// (a) start it without blocking a JNI caller and (b) tell the difference
// between "asked to serve" and "actually listening". If dllm_shim.h ever grows
// a stop/readiness hook, delete these two and call that instead.
// ---------------------------------------------------------------------------

extern "C" {

/** Non-blocking start: run dllm_shim_rpc_serve() on a private thread and return
 *  only once the port is really accepting connections.
 *
 *  Threading model and why (ADR-031): the thread lives in C++, not Kotlin.
 *  - The server executes the assigned layers synchronously on this thread
 *    (rpc_serve_client -> ggml_backend_graph_compute), so it is a long-lived,
 *    CPU-heavy worker. A JVM thread would work too, but then every lifecycle
 *    event (service start/stop, toggle off, process teardown) has to reason
 *    about a thread it cannot join or interrupt - see dllm_shim_rpc_serve_stop.
 *  - Keeping it native means the JNI surface stays strictly non-blocking: no
 *    JNI call can ever be the thing that blocks, so ANR risk on the main thread
 *    is structurally impossible rather than merely unlikely.
 *  - The heartbeat runs on Dispatchers.Default/IO in the JVM, so a busy RPC
 *    thread never delays the 30s tick; it is a native thread outside that pool.
 */
int dllm_shim_rpc_serve_start(const char* host, int32_t port, const char* cache_dir,
                              int32_t n_threads, int32_t n_devices) {
    if (g_serving.load() || g_start_in_flight.load()) {
        set_error("dllm_shim_rpc_serve_start: a server is already running");
        return -2;
    }
    const std::string host_copy = host != nullptr ? host : "0.0.0.0";
    const std::string cache_copy = cache_dir != nullptr ? cache_dir : "";
    const char* cache_arg = cache_dir != nullptr ? cache_copy.c_str() : nullptr;

    g_stop_requested = false;
    g_serving = false;
    g_start_in_flight = true;
    g_last_error.clear();

    // Detached deliberately: the loop it enters has no exit, so join() would be
    // a guaranteed deadlock. See dllm_shim_rpc_serve_stop for what "stop"
    // honestly means on this llama.cpp version.
    std::thread([host_copy, port, cache_arg, n_threads, n_devices]() {
        dllm_shim_rpc_serve(host_copy.c_str(), port, cache_arg, n_threads, n_devices);
    }).detach();

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kServeReadyTimeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
        if (probe_listening(host_copy, port)) {
            g_serving = true;
            g_start_in_flight = false;
            SHIM_LOGI("dllm_shim_rpc_serve_start: %s:%d is accepting connections", host_copy.c_str(), port);
            return 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kServeReadyPollMs));
    }

    g_start_in_flight = false;
    set_error("dllm_shim_rpc_serve_start: nothing listening on " + host_copy + ":" +
              std::to_string(port) + " after " + std::to_string(kServeReadyTimeoutMs) +
              "ms (port busy, or that interface has no address)");
    return -1;
}

/**
 * Request a stop. HONESTY NOTE - this does NOT close the listening socket.
 *
 * ggml_backend_rpc_start_server() (llama.cpp b7418) offers no shutdown hook: the
 * socket is a function local, accept() is unblockable from outside and
 * rpc_serve_client() is static. So a stop here means:
 *   1. g_serving goes false, so the app stops advertising the endpoint in its
 *      heartbeat and a coordinator is never sent to a worker we have retired;
 *   2. a further start on the same port is refused for the life of the process,
 *      because the old listener still holds it (SO_REUSEADDR does not permit two
 *      live listeners on one port);
 *   3. the port is genuinely released when the process exits.
 * An in-flight client keeps computing until it disconnects; that cannot be
 * interrupted either. Callers MUST treat "stopped" as "no longer advertised", not
 * as "port free", and say so in whatever they show the user.
 */
int dllm_shim_rpc_serve_stop(void) {
    if (g_serving.exchange(false) || g_start_in_flight.load()) {
        g_stop_requested = true;
        SHIM_LOGI("dllm_shim_rpc_serve_stop: endpoint withdrawn; the listening socket is released on process exit (llama.cpp exposes no stop hook)");
    }
    return 0;
}

/** True only when the port is confirmed listening and no stop was requested. */
int dllm_shim_rpc_serving(void) {
    return (g_serving.load() && !g_stop_requested.load()) ? 1 : 0;
}

}  // extern "C"