/* dllm_shim — implementation of the C ABI in include/dllm_shim.h.
 *
 * This is the only translation unit in the project that is allowed to know that
 * llama.cpp and ggml-rpc exist. Rust (crate `dllm-shim`) and Android (JNI in
 * worker.cpp) see only dllm_shim.h.
 *
 * Every llama.cpp / ggml signature used below was read out of the vendored tree
 * at third_party/llama.cpp (tag b7418) rather than from memory, because this
 * project bumps llama.cpp independently of llama-cpp-2 and the API moves.
 * Notable pins for that tag:
 *   - ggml/include/ggml-rpc.h exists and declares the RPC entry points directly
 *     (in older tags the header lived under ggml/src/ggml-rpc/).
 *   - `ggml_backend_rpc_add_server` returns a ggml_backend_reg_t (a *registry
 *     for that endpoint*), NOT a device count and NOT a backend. Upstream then
 *     calls ggml_backend_register() on it. See common/arg.cpp:add_rpc_devices.
 *   - `llama_model_params.devices` is a NULL-terminated ggml_backend_dev_t* and
 *     is consumed verbatim, in order (llama.cpp:834-837). That is the only way
 *     to control layer-placement order; the default selection puts RPC devices
 *     FIRST and excludes CPU entirely.
 *   - `tensor_split` is indexed by position in that devices list and must be
 *     llama_max_devices() (== 16) entries long.
 */

#include "dllm_shim.h"

#include "llama.h"
/* Internal header. llama_model::dev_layer() is public C++ and returns the
 * ggml_backend_dev_t llama.cpp ACTUALLY assigned to a layer
 * (llama-model.cpp:6978 -> pimpl->dev_layer[il].dev), including the CPU
 * fallback when a tensor did not fit. It is not in the stable C API, which is
 * why dllm_shim_session_report would otherwise have to guess the split from the
 * requested tensor_split -- i.e. restate intent instead of reporting reality. */
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

#define DLLM_SHIM_ABI_VERSION 1

/* The header forward-declares `typedef struct dllm_shim_session dllm_shim_session;`,
 * so the definition has to live at global scope — putting it in an anonymous
 * namespace would make every exported function below ambiguous. */
struct dllm_shim_session {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;

    std::string model_path;
    std::vector<float> requested_split;      /* caller-supplied ratios, verbatim */
    std::vector<ggml_backend_dev_t> devices; /* what we handed to llama.cpp */
    int32_t n_gpu_layers_req = 0;
    int32_t n_gpu_layers_eff = 0;

    std::atomic<bool> cancel{false};
    std::atomic<bool> busy{false};

    /* Perf snapshot from the most recent generate(), kept so the report can be
     * produced without the caller having to hold on to anything. */
    double   prefill_ms = 0.0;
    double   decode_ms  = 0.0;
    int32_t  n_prompt   = 0;
    int32_t  n_gen      = 0;
};

namespace {

/* ------------------------------------------------------------------ errors */

thread_local std::string g_last_error;

void set_err(const char * fmt, ...) {
    char    buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_last_error.assign(buf);
}

/* ------------------------------------------------------------------ RPC API */

/* Resolved dynamically through ggml_backend_reg_get_proc_address, exactly the
 * way upstream's common/arg.cpp and tools/rpc/rpc-server.cpp do it. Doing it
 * this way means a build without GGML_RPC degrades to "no RPC support" instead
 * of failing to link or crashing on a null function pointer. */
struct RpcApi {
    typedef ggml_backend_reg_t (*add_server_t)  (const char * endpoint);
    typedef void              (*start_server_t)(const char * endpoint, const char * cache_dir,
                                                 size_t n_threads, size_t n_devices,
                                                 ggml_backend_dev_t * devices);

    add_server_t  add_server   = nullptr;
    start_server_t start_server = nullptr;
    std::string    resolve_error;
};

const RpcApi & rpc_api() {
    static RpcApi api = [] {
        RpcApi a;
        /* `ggml_backend_reg_by_name("RPC")` only succeeds after the registry has
         * been touched at least once; llama_backend_init() below guarantees
         * that before any shim function resolves this. */
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
        if (!reg) {
            a.resolve_error =
                "the RPC backend is not registered. This build did not enable "
                "GGML_RPC, so no ggml-rpc symbols exist and layers cannot be "
                "offloaded. Re-run scripts/build-llama-win.ps1 and rebuild native/.";
            return a;
        }
        a.add_server = reinterpret_cast<RpcApi::add_server_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_add_server"));
        a.start_server = reinterpret_cast<RpcApi::start_server_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_start_server"));
        if (!a.add_server) {
            a.resolve_error = "RPC backend does not export ggml_backend_rpc_add_server";
        } else if (!a.start_server) {
            a.resolve_error = "RPC backend does not export ggml_backend_rpc_start_server";
        }
        return a;
    }();
    return api;
}

/* ------------------------------------------------------------------ globals */

std::once_flag g_init_once;
std::atomic<int> g_init_result{0};

/* Guards the endpoint list only. Session state is per-session and is never
 * touched concurrently (see the threading contract in dllm_shim.h). */
std::mutex g_endpoint_mutex;
std::vector<std::string> g_endpoints;

int do_init() {
    std::call_once(g_init_once, [] {
        /* Registers the statically linked CPU backend. With GGML_RPC=ON the
         * ggml backend registry constructor also registers ggml_backend_rpc_reg()
         * (ggml-backend-reg.cpp:222), which is what makes
         * ggml_backend_reg_by_name("RPC") resolve.
         *
         * ggml_backend_load_all() is deliberately NOT called: CPU and RPC are
         * linked statically, and scanning the executable directory for
         * backends would let an unrelated ggml-*.dll silently change the device
         * list that tensor_split is indexed against. */
        llama_backend_init();

        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) {
            set_err("no CPU ggml backend found after llama_backend_init()");
            g_init_result.store(-1);
            return;
        }
        g_init_result.store(0);
    });
    return g_init_result.load();
}

/* RPC devices, flattened across every registered endpoint, in registration
 * order. The base "RPC" registry that ggml_backend_rpc_reg() hands out has a
 * null context, so it reports 0 devices and must never be asked for one
 * (ggml_backend_reg_dev_get would GGML_ABORT). */
void collect_rpc_devices(std::vector<ggml_backend_dev_t> & out) {
    out.clear();
    const size_t n_reg = ggml_backend_reg_count();
    for (size_t i = 0; i < n_reg; ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (!reg) {
            continue;
        }
        const char * name = ggml_backend_reg_name(reg);
        /* Endpoint registries are named "RPC[host:port]"; the bare one is "RPC". */
        if (!name || std::strncmp(name, "RPC[", 4) != 0) {
            continue;
        }
        const size_t n_dev = ggml_backend_reg_dev_count(reg);
        for (size_t d = 0; d < n_dev; ++d) {
            out.push_back(ggml_backend_reg_dev_get(reg, d));
        }
    }
}

bool is_rpc_device(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) {
        return false;
    }
    const char * name = ggml_backend_reg_name(reg);
    if (!name) {
        return false;
    }
    /* Both "RPC" and "RPC[endpoint]" registries are RPC. */
    return std::strcmp(name, "RPC") == 0 || std::strncmp(name, "RPC[", 4) == 0;
}

/* Devices the model is allowed to place layers on, in ggml global enumeration
 * order, which is exactly what the header promises: local devices first, then
 * RPC devices in registration order (the registry constructor registers CPU
 * first; endpoint registries are appended by ggml_backend_register()).
 *
 * ACCEL devices are filtered out because llama-context.cpp:199-208 appends every
 * ACCEL backend on its own; listing one here too would initialise it twice. */
void collect_layer_devices(std::vector<ggml_backend_dev_t> & out, bool include_rpc) {
    out.clear();
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            continue;
        }
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        if (!include_rpc && is_rpc_device(dev)) {
            continue;
        }
        out.push_back(dev);
    }
}

/* ------------------------------------------------------------------ session */

/* ------------------------------------------------------------------- JSON */

/* snprintf into a fixed buffer, always NUL-terminated, never overflows. */
void copy_bounded(char * out, int32_t out_len, const std::string & s) {
    if (!out || out_len <= 0) {
        return;
    }
    const size_t n = (size_t) out_len - 1;
    const size_t c = s.size() < n ? s.size() : n;
    std::memcpy(out, s.data(), c);
    out[c] = '\0';
}

/* --------------------------------------------------------------- sampling */

llama_sampler * build_sampler_chain(const dllm_shim_sampler & s) {
    llama_sampler_chain_params cparams = llama_sampler_chain_default_params();
    cparams.no_perf = false;
    llama_sampler * chain = llama_sampler_chain_init(cparams);
    if (!chain) {
        return nullptr;
    }

    /* Order mirrors llama.cpp's own sampling pipeline (see the comment block on
     * llama_sampler_chain_* in llama.h): penalties shape the distribution
     * before it is truncated. */
    if (s.presence_penalty != 0.0f) {
        llama_sampler_chain_add(chain,
            llama_sampler_init_penalties(64, 1.0f, 0.0f, s.presence_penalty));
    }
    if (s.top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(s.top_k));
    }
    if (s.top_p > 0.0f && s.top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(s.top_p, 1));
    }
    if (s.min_p > 0.0f && s.min_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(s.min_p, 1));
    }
    if (s.temp < 0.0f) {
        /* Greedy. The catalog has no greedy mode, but a caller asking for
         * sub-zero temperature means "deterministic", and llama.cpp would
         * otherwise assert on it. */
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        if (s.temp > 0.0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_temp(s.temp));
        }
        const uint32_t seed = s.seed < 0 ? (uint32_t) LLAMA_DEFAULT_SEED : (uint32_t) s.seed;
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    }
    return chain;
}

} /* namespace */

/* =========================================================== lifecycle ==== */

extern "C" DLLM_SHIM_API int dllm_shim_init(void) {
    return do_init();
}

extern "C" DLLM_SHIM_API void dllm_shim_free(void) {
    /* ggml backend registries, RPC sockets and the llama.cpp global allocator
     * are process-lifetime singletons in this build. Sessions must already be
     * closed (the header requires it); tearing down the registry here would
     * leave llama_model/llama_context objects pointing at freed devices. */
    g_endpoints.clear();
}

extern "C" DLLM_SHIM_API const char * dllm_shim_last_error(void) {
    return g_last_error.c_str();
}

extern "C" DLLM_SHIM_API int dllm_shim_abi_version(void) {
    return DLLM_SHIM_ABI_VERSION;
}

/* ============================================================== workers ==== */

extern "C" DLLM_SHIM_API int dllm_shim_add_rpc_server(const char *host_port) {
    if (!host_port || host_port[0] == '\0') {
        set_err("dllm_shim_add_rpc_server: empty endpoint");
        return -1;
    }
    if (do_init() != 0) {
        return -1;
    }

    const RpcApi & api = rpc_api();
    if (!api.add_server) {
        set_err("dllm_shim_add_rpc_server: %s", api.resolve_error.c_str());
        return -1;
    }

    /* Upstream (common/arg.cpp:add_rpc_devices) calls ggml_backend_register on
     * the returned registry; without it the endpoint's devices never enter the
     * global list and no session can use them. */
    ggml_backend_reg_t reg = api.add_server(host_port);
    if (!reg) {
        set_err("dllm_shim_add_rpc_server: no RPC devices reported by %s "
                "(unreachable, wrong port, or protocol version mismatch)",
                host_port);
        return -1;
    }

    const size_t n_dev = ggml_backend_reg_dev_count(reg);
    if (n_dev == 0) {
        set_err("dllm_shim_add_rpc_server: %s registered zero devices", host_port);
        return -1;
    }

    ggml_backend_register(reg);

    {
        std::lock_guard<std::mutex> lock(g_endpoint_mutex);
        g_endpoints.push_back(host_port);
    }

    return (int) n_dev;
}

extern "C" DLLM_SHIM_API int dllm_shim_rpc_device_count(void) {
    if (do_init() != 0) {
        set_err("dllm_shim_rpc_device_count: llama backend not initialised");
        return -1;
    }
    std::vector<ggml_backend_dev_t> devs;
    collect_rpc_devices(devs);
    return (int) devs.size();
}

extern "C" DLLM_SHIM_API int dllm_shim_rpc_device_memory(int32_t      device_index,
                                                         size_t     *out_free_bytes,
                                                         size_t     *out_total_bytes) {
    if (out_free_bytes)  { *out_free_bytes  = 0; }
    if (out_total_bytes) { *out_total_bytes = 0; }

    if (device_index < 0) {
        set_err("dllm_shim_rpc_device_memory: device_index must be >= 0 (got %d)",
                device_index);
        return -1;
    }
    if (do_init() != 0) {
        set_err("dllm_shim_rpc_device_memory: llama backend not initialised");
        return -1;
    }

    std::vector<ggml_backend_dev_t> devs;
    collect_rpc_devices(devs);
    if ((size_t) device_index >= devs.size()) {
        set_err("dllm_shim_rpc_device_memory: index %d out of range (%zu RPC devices registered)",
                device_index, devs.size());
        return -1;
    }

    /* For an RPC device this is not a local query: ggml_backend_dev_memory
     * dispatches RPC_CMD_GET_DEVICE_MEMORY to the worker and returns whatever it
     * reports (ggml-rpc.cpp:1922 -> ggml_backend_rpc_get_device_memory). A dead
     * worker yields zeros, which is the honest answer rather than a guess. */
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(devs[(size_t) device_index], &free_bytes, &total_bytes);

    if (out_free_bytes)  { *out_free_bytes  = free_bytes; }
    if (out_total_bytes) { *out_total_bytes = total_bytes; }
    return 0;
}

extern "C" DLLM_SHIM_API int dllm_shim_rpc_serve(const char *host,
                                                 int32_t       port,
                                                 const char   *cache_dir,
                                                 int32_t       n_threads,
                                                 int32_t       n_devices) {
    if (!host || host[0] == '\0') {
        set_err("dllm_shim_rpc_serve: empty host");
        return -1;
    }
    if (port <= 0 || port > 65535) {
        set_err("dllm_shim_rpc_serve: port %d out of range", (int) port);
        return -1;
    }
    if (do_init() != 0) {
        set_err("dllm_shim_rpc_serve: llama backend not initialised");
        return -1;
    }

    const RpcApi & api = rpc_api();
    if (!api.start_server) {
        set_err("dllm_shim_rpc_serve: %s", api.resolve_error.c_str());
        return -1;
    }

    if (std::strcmp(host, "127.0.0.1") != 0 && std::strcmp(host, "localhost") != 0) {
        /* Same warning upstream tools/rpc/rpc-server.cpp prints. The RPC protocol
         * is plaintext TCP with no authentication whatsoever. */
        fprintf(stderr,
                "\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "WARNING: Host ('%s') is != '127.0.0.1'\n"
                "         Never expose the RPC server to an open network!\n"
                "         This is an experimental feature and is not secure!\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n",
                host);
    }

    /* Device selection, mirroring tools/rpc/rpc-server.cpp:get_devices:
     * non-CPU devices first, falling back to the CPU device. */
    std::vector<ggml_backend_dev_t> candidates;
    const size_t n_global = ggml_backend_dev_count();
    for (size_t i = 0; i < n_global; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            candidates.push_back(dev);
        }
    }
    if (candidates.empty()) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) {
            set_err("dllm_shim_rpc_serve: no local device to expose");
            return -1;
        }
        candidates.push_back(cpu);
    }

    /* n_devices < 0 means "every accelerator this device has". */
    if (n_devices >= 0 && (size_t) n_devices < candidates.size()) {
        candidates.resize((size_t) n_devices);
    }
    if (candidates.empty()) {
        set_err("dllm_shim_rpc_serve: n_devices=%d left no devices to expose", (int) n_devices);
        return -1;
    }

    const std::string endpoint = std::string(host) + ":" + std::to_string((int) port);
    const size_t threads = n_threads > 0 ? (size_t) n_threads : (size_t) GGML_DEFAULT_N_THREADS;

    /* Blocks until the server stops; there is no way to abort it from this ABI.
     * That matches upstream and is why the test drives it from its own thread. */
    api.start_server(endpoint.c_str(), cache_dir, threads, candidates.size(), candidates.data());
    return 0;
}

/* ============================================================= sessions ==== */

extern "C" DLLM_SHIM_API dllm_shim_session * dllm_shim_session_open(const char *model_path,
                                                                   const float *tensor_split,
                                                                   int32_t       n_split,
                                                                   int32_t       n_gpu_layers,
                                                                   int32_t       n_ctx) {
    if (!model_path || model_path[0] == '\0') {
        set_err("dllm_shim_session_open: empty model_path");
        return nullptr;
    }
    if (do_init() != 0) {
        return nullptr;
    }

    const bool have_split = tensor_split != nullptr && n_split > 0;

    std::unique_ptr<dllm_shim_session> s(new dllm_shim_session());
    s->model_path = model_path;
    s->n_gpu_layers_req = n_gpu_layers;

    /* Device list. With a split we let RPC devices in (that is the whole point);
     * without one the header says "everything local", so RPC devices are left
     * out entirely — passing them with tensor_split == NULL would make
     * llama.cpp split by free memory and silently offload anyway. */
    collect_layer_devices(s->devices, have_split);
    if (s->devices.empty()) {
        set_err("dllm_shim_session_open: no usable ggml devices");
        return nullptr;
    }

    /* llama.cpp indexes tensor_split by position in the devices list and copies
     * llama_max_devices() entries, so the buffer must be that long. */
    std::vector<float> split((size_t) llama_max_devices(), 0.0f);
    if (have_split) {
        if ((size_t) n_split > s->devices.size()) {
            set_err("dllm_shim_session_open: n_split=%d but only %zu devices are available "
                    "(register the workers with dllm_shim_add_rpc_server first)",
                    (int) n_split, s->devices.size());
            return nullptr;
        }
        for (int32_t i = 0; i < n_split; ++i) {
            if (tensor_split[i] < 0.0f) {
                set_err("dllm_shim_session_open: tensor_split[%d] = %f is negative",
                        (int) i, tensor_split[i]);
                return nullptr;
            }
            split[(size_t) i] = tensor_split[i];
        }
        s->requested_split.assign(tensor_split, tensor_split + n_split);
    }

    /* The devices array llama.cpp consumes must be NULL-terminated. */
    std::vector<ggml_backend_dev_t> dev_arg(s->devices.begin(), s->devices.end());
    dev_arg.push_back(nullptr);

    llama_model_params mparams = llama_model_default_params();
    mparams.devices      = dev_arg.data();
    mparams.split_mode   = LLAMA_SPLIT_MODE_LAYER;
    mparams.main_gpu     = 0;
    mparams.use_mmap     = true;
    mparams.tensor_split = have_split ? split.data() : nullptr;

    if (!have_split) {
        /* "Everything local" means everything local: zero offloaded layers. */
        mparams.n_gpu_layers = 0;
    } else if (n_gpu_layers < 0) {
        /* "Place every layer we can." llama.cpp clamps with
         *   i_gpu_start    = max(n_layer - n_gpu_layers, 0)
         *   act_gpu_layers = min(n_gpu_layers,  n_layer + 1)
         * so anything >= n_layer + 1 is exactly "all layers"; INT32_MAX needs no
         * knowledge of n_layer before the model is loaded. Same idiom as
         * llama_model_default_params()'s own 999. */
        mparams.n_gpu_layers = std::numeric_limits<int32_t>::max();
    } else {
        mparams.n_gpu_layers = n_gpu_layers;
    }

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        set_err("dllm_shim_session_open: llama_model_load_from_file('%s') failed", model_path);
        return nullptr;
    }
    s->model = model;
    s->n_gpu_layers_eff = mparams.n_gpu_layers;

    llama_context_params cparams = llama_context_default_params();
    if (n_ctx > 0) {
        cparams.n_ctx = (uint32_t) n_ctx;
    }
    if (cparams.n_batch  > cparams.n_ctx) { cparams.n_batch  = cparams.n_ctx; }
    if (cparams.n_ubatch > cparams.n_batch) { cparams.n_ubatch = cparams.n_batch; }
    cparams.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        set_err("dllm_shim_session_open: llama_init_from_model failed for '%s' "
                "(not enough device memory for n_ctx=%u?)",
                model_path, (unsigned) cparams.n_ctx);
        llama_model_free(model);
        return nullptr;
    }
    s->ctx = ctx;

    /* Report what was really done, not what was asked for: count the layers
     * llama.cpp did not keep on the CPU device. */
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t      n_lay = llama_model_n_layer(model);
    ggml_backend_dev_t  cpu   = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    int32_t             n_off = 0;
    for (int32_t il = 0; il < n_lay; ++il) {
        if (ggml_backend_dev_t d = model->dev_layer(il)) {
            if (!cpu || d != cpu) {
                ++n_off;
            }
        }
    }
    (void) vocab;
    s->n_gpu_layers_eff = n_off;

    /* Report the requested n_ctx rather than the clamp llama.cpp may have
     * applied; callers size their KV budget from this. */
    return s.release();
}

extern "C" DLLM_SHIM_API void dllm_shim_session_close(dllm_shim_session *session) {
    delete session;
}

extern "C" DLLM_SHIM_API int dllm_shim_session_generate(dllm_shim_session *session,
                                                        const char        *prompt_utf8,
                                                        int32_t            max_tokens,
                                                        const dllm_shim_sampler *sampler,
                                                        dllm_shim_token_cb  on_token,
                                                        void              *user) {
    if (!session || !session->ctx || !session->model) {
        set_err("dllm_shim_session_generate: null session");
        return -1;
    }
    if (!prompt_utf8) {
        set_err("dllm_shim_session_generate: null prompt");
        return -1;
    }
    if (!on_token) {
        set_err("dllm_shim_session_generate: null token callback");
        return -1;
    }

    bool expected = false;
    if (!session->busy.compare_exchange_strong(expected, true)) {
        set_err("dllm_shim_session_generate: a generation is already running on this session");
        return -1;
    }

    /* Every exit path past this point must emit exactly one done!=0 callback. */
    struct FinishGuard {
        dllm_shim_token_cb cb;
        void             * user;
        int32_t            pos;
        ~FinishGuard() {
            if (cb) {
                cb(user, pos, nullptr, 0, 1);
            }
        }
    };

    const llama_vocab * vocab = llama_model_get_vocab(session->model);
    int rc = 0;

    session->cancel.store(false);
    session->prefill_ms = 0.0;
    session->decode_ms  = 0.0;
    session->n_prompt   = 0;
    session->n_gen      = 0;
    llama_perf_context_reset(session->ctx);

    llama_memory_clear(llama_get_memory(session->ctx), true);

    /* ---- tokenise ---- */
    const int32_t   n_prompt_max = (int32_t) llama_n_ctx(session->ctx) - 1;
    std::vector<llama_token> prompt_tokens;
    {
        int32_t want = n_prompt_max > 0 ? n_prompt_max : 1024;
        prompt_tokens.resize((size_t) want);
        int32_t got = llama_tokenize(vocab, prompt_utf8, (int32_t) std::strlen(prompt_utf8),
                                     prompt_tokens.data(), want, /*add_special*/ true,
                                     /*parse_special*/ true);
        if (got < 0) {
            /* Negative means "needs more room"; grow once and retry. */
            prompt_tokens.resize((size_t) -got);
            got = llama_tokenize(vocab, prompt_utf8, (int32_t) std::strlen(prompt_utf8),
                                 prompt_tokens.data(), -got, true, true);
        }
        if (got < 0) {
            set_err("dllm_shim_session_generate: llama_tokenize failed (%d); prompt is too long "
                    "for n_ctx=%u", (int) got, (unsigned) llama_n_ctx(session->ctx));
            session->busy.store(false);
            return -1;
        }
        prompt_tokens.resize((size_t) got);
    }
    if (prompt_tokens.empty()) {
        set_err("dllm_shim_session_generate: prompt tokenised to zero tokens");
        session->busy.store(false);
        return -1;
    }
    session->n_prompt = (int32_t) prompt_tokens.size();

    /* ---- sampler ---- */
    dllm_shim_sampler def;
    std::memset(&def, 0, sizeof(def));
    def.top_k            = 40;
    def.top_p            = 0.95f;
    def.temp             = 0.8f;
    def.min_p            = 0.05f;
    def.presence_penalty = 0.0f;
    def.seed             = -1;
    llama_sampler * chain = build_sampler_chain(sampler ? *sampler : def);
    if (!chain) {
        set_err("dllm_shim_session_generate: llama_sampler_chain_init failed");
        session->busy.store(false);
        return -1;
    }

    const int32_t n_batch  = (int32_t) llama_n_batch(session->ctx);
    const int32_t n_ubatch = (int32_t) llama_n_ubatch(session->ctx);
    const int32_t chunk    = std::max(1, std::min(n_batch, n_ubatch));

    llama_batch batch = llama_batch_init((int32_t) chunk, 0, 1);
    int32_t pos = 0;
    {
        FinishGuard guard{ on_token, user, 0 };
        guard.pos = 0;

        /* ---- prefill: decode the prompt in chunks of n_ubatch ---- */
        for (size_t off = 0; off < prompt_tokens.size(); off += (size_t) chunk) {
            if (session->cancel.load()) {
                break;
            }
            const int32_t n = (int32_t) std::min((size_t) chunk, prompt_tokens.size() - off);
            batch.n_tokens = n;
            for (int32_t i = 0; i < n; ++i) {
                batch.token[i]    = prompt_tokens[(size_t) off + (size_t) i];
                batch.pos[i]      = pos + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                /* Only the very last prompt token needs logits. */
                batch.logits[i]   = (off + (size_t) n == prompt_tokens.size()) ? 1 : 0;
            }
            const int32_t drc = llama_decode(session->ctx, batch);
            if (drc != 0) {
                set_err("dllm_shim_session_generate: llama_decode failed on prompt token %d of %d "
                        "(rc=%d)", (int) (off + (size_t) n), (int) prompt_tokens.size(), (int) drc);
                rc = -1;
                break;
            }
            pos += n;
        }
        guard.pos = pos;

        /* ---- decode loop ---- */
        while (rc == 0) {
            if (session->cancel.load()) {
                break;
            }
            if (max_tokens >= 0 && session->n_gen >= max_tokens) {
                break;
            }
            if (pos >= (int32_t) llama_n_ctx(session->ctx)) {
                break;
            }

            const llama_token tok = llama_sampler_sample(chain, session->ctx, -1);

            if (llama_vocab_is_eog(vocab, tok)) {
                /* EOS/EOT is a stop condition, not text. */
                break;
            }

            /* Detokenise. llama_token_to_piece returns the byte count, which is
             * exactly the UTF-8 length the callback wants; a negative return
             * means the buffer was too small and reports the size needed. */
            char    piece[64];
            int32_t len = llama_token_to_piece(vocab, tok, piece, (int32_t) sizeof(piece),
                                               /*lstrip*/ 0, /*special*/ false);
            if (len < 0) {
                const int32_t need = -len;
                if (need > (int32_t) sizeof(piece)) {
                    std::vector<char> big((size_t) need);
                    len = llama_token_to_piece(vocab, tok, big.data(), need, 0, false);
                    if (len < 0) {
                        set_err("dllm_shim_session_generate: llama_token_to_piece failed for "
                                "token %d", (int) tok);
                        rc = -1;
                        break;
                    }
                    if (len > 0 && on_token(user, pos, big.data(), len, 0) != 0) {
                        break;
                    }
                } else {
                    set_err("dllm_shim_session_generate: llama_token_to_piece reported a "
                            "negative length (%d) for token %d", (int) len, (int) tok);
                    rc = -1;
                    break;
                }
            } else if (len > 0) {
                if (on_token(user, pos, piece, len, 0) != 0) {
                    break;
                }
            }

            session->n_gen += 1;

            batch.n_tokens = 1;
            batch.token[0]     = tok;
            batch.pos[0]       = pos;
            batch.n_seq_id[0]  = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]    = 1;

            const int32_t drc = llama_decode(session->ctx, batch);
            if (drc != 0) {
                set_err("dllm_shim_session_generate: llama_decode failed at pos %d (rc=%d)",
                        (int) pos, (int) drc);
                rc = -1;
                break;
            }
            pos += 1;
            guard.pos = pos;
        }

        const llama_perf_context_data perf = llama_perf_context(session->ctx);
        session->prefill_ms = perf.t_p_eval_ms;
        session->decode_ms  = perf.t_eval_ms;
    }

    llama_batch_free(batch);
    llama_sampler_free(chain);
    session->busy.store(false);
    return rc;
}

extern "C" DLLM_SHIM_API int dllm_shim_session_cancel(dllm_shim_session *session) {
    if (!session) {
        set_err("dllm_shim_session_cancel: null session");
        return -1;
    }
    if (!session->busy.load()) {
        set_err("dllm_shim_session_cancel: no generation in flight");
        return -1;
    }
    session->cancel.store(true);
    return 0;
}

extern "C" DLLM_SHIM_API int dllm_shim_session_n_layer(dllm_shim_session *session, int32_t *out) {
    if (!session || !session->model) {
        set_err("dllm_shim_session_n_layer: null session");
        return -1;
    }
    if (out) {
        *out = llama_model_n_layer(session->model);
    }
    return 0;
}

extern "C" DLLM_SHIM_API int dllm_shim_session_n_ctx(dllm_shim_session *session, int32_t *out) {
    if (!session || !session->ctx) {
        set_err("dllm_shim_session_n_ctx: null session");
        return -1;
    }
    if (out) {
        *out = (int32_t) llama_n_ctx(session->ctx);
    }
    return 0;
}

extern "C" DLLM_SHIM_API int dllm_shim_session_report(dllm_shim_session *session,
                                                      char              *out,
                                                      int32_t            out_len) {
    if (!out || out_len <= 0) {
        set_err("dllm_shim_session_report: bad output buffer");
        return -1;
    }
    out[0] = '\0';
    if (!session || !session->model || !session->ctx) {
        set_err("dllm_shim_session_report: null session");
        return -1;
    }

    const int32_t n_layer = llama_model_n_layer(session->model);
    const int32_t n_ctx   = (int32_t) llama_n_ctx(session->ctx);
    const int32_t n_dev   = (int32_t) ggml_backend_dev_count();

    /* layer_owner[i] = global ggml device index holding layer i's tensors.
     *
     * This is read back from llama_model::dev_layer(), i.e. from the assignment
     * llama.cpp actually made (including its CPU fallback when a tensor did not
     * fit on the requested device) — not recomputed from tensor_split. If the
     * internal header ever stops exposing dev_layer() this must degrade to
     * "layer_owner":null rather than emit a guess. */
    std::vector<int32_t> layer_owner;
    bool owner_known = true;
    layer_owner.reserve((size_t) std::max(0, n_layer));

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    int32_t n_offloaded = 0;

    for (int32_t il = 0; il < n_layer; ++il) {
        ggml_backend_dev_t dev = nullptr;
        try {
            dev = session->model->dev_layer(il);
        } catch (...) {
            dev = nullptr;
        }
        if (!dev) {
            owner_known = false;
            break;
        }
        int32_t idx = -1;
        for (int32_t i = 0; i < n_dev; ++i) {
            if (ggml_backend_dev_get((size_t) i) == dev) {
                idx = i;
                break;
            }
        }
        if (idx < 0) {
            /* Device is in use but not in the global list; do not invent an index. */
            owner_known = false;
            break;
        }
        layer_owner.push_back(idx);
        if (dev != cpu) {
            ++n_offloaded;
        }
    }

    const double   prefill_ms = session->prefill_ms;
    const double   decode_ms  = session->decode_ms;
    const double   pps        = decode_ms > 0.0
        ? ((double) session->n_gen * 1000.0 / decode_ms)
        : 0.0;

    std::string js;
    js.reserve(256 + (size_t) std::max(0, n_layer) * 3);
    js += "{\"n_layer\":" + std::to_string(n_layer);
    js += ",\"n_ctx\":"   + std::to_string(n_ctx);
    js += ",\"n_gpu_layers\":" + std::to_string(n_offloaded);
    js += ",\"tensor_split\":";
    if (session->requested_split.empty()) {
        js += "null";
    } else {
        js += "[";
        for (size_t i = 0; i < session->requested_split.size(); ++i) {
            if (i) { js += ","; }
            js += std::to_string(session->requested_split[i]);
        }
        js += "]";
    }
    js += ",\"devices\":" + std::to_string(n_dev);
    js += ",\"timing\":{\"prefill_ms\":" + std::to_string(prefill_ms);
    js += ",\"decode_ms\":"  + std::to_string(decode_ms);
    js += ",\"predicted_per_second\":" + std::to_string(pps);
    js += ",\"n_prompt_tokens\":" + std::to_string(session->n_prompt);
    js += ",\"n_generated\":"     + std::to_string(session->n_gen);
    js += "}";

    if (owner_known && (int32_t) layer_owner.size() == n_layer) {
        js += ",\"layer_owner\":[";
        for (size_t i = 0; i < layer_owner.size(); ++i) {
            if (i) { js += ","; }
            js += std::to_string(layer_owner[i]);
        }
        js += "]";
    } else {
        js += ",\"layer_owner\":null";
        js += ",\"layer_owner_note\":\"per-layer device ownership could not be read back "
              "from llama_model::dev_layer() in this build; refusing to guess from "
              "tensor_split\"";
    }

    js += "}";
    copy_bounded(out, out_len, js);
    return 0;
}
