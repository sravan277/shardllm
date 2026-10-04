/* dllm_shim — C ABI between the Rust coordinator and llama.cpp + ggml-rpc.
 *
 * Why this exists
 * ---------------
 * `llama-cpp-2` (the crate the Rust binary uses) vendors a llama.cpp tree with
 * NO `ggml/src/ggml-rpc/` at all, and its bindgen never includes `ggml-rpc.h`,
 * so zero `ggml_rpc*` symbols are reachable from Rust. ggml backend
 * registration is also C++-vtable-bound, so a backend cannot be fabricated from
 * Rust either. The only way to put transformer layers on another machine is to
 * own a C++ translation unit that links llama.cpp + ggml-rpc and exposes a
 * plain C surface.
 *
 * That is this file. It is deliberately boring C: no C++ types cross the
 * boundary, no exceptions, no ownership ambiguity. Both Rust (via `dllm-shim`)
 * and Android (via JNI in `worker.cpp`) call exactly these symbols.
 *
 * Threading contract: every function is safe to call from any thread EXCEPT
 * `dllm_shim_session_generate`, which must not be called concurrently with
 * another `*_generate` on the SAME session. Different sessions are independent.
 * A session is not internally synchronised on purpose — the coordinator owns
 * scheduling, and a hidden lock would hide pipeline deadlocks.
 *
 * Error convention: functions returning `int` return 0 on success and a
 * negative value on failure, unless documented otherwise. `dllm_shim_last_error`
 * always returns a human-readable string describing the most recent failure on
 * this thread; it is never NULL and never freed by the caller.
 */

#ifndef DLLM_SHIM_H
#define DLLM_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define DLLM_SHIM_API __declspec(dllexport)
#else
#  define DLLM_SHIM_API __attribute__((visibility("default")))
#endif

/* ---------------------------------------------------------------- lifecycle */

/* Initialise llama.cpp backends and register the RPC backend.
 *
 * `dllm_shim_add_rpc_server` is a no-op until this has returned 0, because the
 * RPC backend registers itself into the global device registry on init.
 * Idempotent: calling it twice is safe and returns 0. */
DLLM_SHIM_API int dllm_shim_init(void);

/* Release global backend state. Call once, at shutdown, after every session is
 * closed. */
DLLM_SHIM_API void dllm_shim_free(void);

/* Most recent error on this thread. Never NULL; caller must NOT free it.
 * Valid until the next shim call on the same thread. */
DLLM_SHIM_API const char *dllm_shim_last_error(void);

/* Library + protocol version, for a startup handshake so a mismatched shim is
 * detected instead of crashing. Bump on any ABI break. */
DLLM_SHIM_API int dllm_shim_abi_version(void);

/* ------------------------------------------------------------------- workers */

/* Register a remote ggml-rpc worker reachable at "host:port".
 *
 * The RPC protocol is plaintext TCP with no authentication — upstream llama.cpp
 * warns "Never expose the RPC server to an open network!". This is only
 * acceptable on a trusted LAN, and only for a worker that the coordinator has
 * already paired (see `dllm-serve`'s registry-TOFU allow-list). Returns the
 * number of RPC devices the endpoint exposed, or negative on failure.
 *
 * Each call adds to the global device list, so calling it twice with the same
 * endpoint registers duplicate devices — the caller owns de-duplication. */
DLLM_SHIM_API int dllm_shim_add_rpc_server(const char *host_port);

/* Number of RPC devices currently reachable, summed over all registered
 * endpoints. */
DLLM_SHIM_API int dllm_shim_rpc_device_count(void);

/* Free/total bytes reported by an RPC device, as measured by
 * `RPC_CMD_GET_DEVICE_MEMORY` — i.e. real numbers from the worker, never
 * assumed locally. Either pointer may be NULL. Returns 0 on success. */
DLLM_SHIM_API int dllm_shim_rpc_device_memory(int32_t device_index,
                                              size_t *out_free_bytes,
                                              size_t *out_total_bytes);

/* Host a worker: start the ggml-rpc server so ANOTHER coordinator can offload
 * layers to this device. Blocks until the server stops. Returns 0 on clean
 * shutdown.
 *
 * `n_devices` < 0 means "expose every accelerator this device has". `cache_dir`
 * may be NULL. */
DLLM_SHIM_API int dllm_shim_rpc_serve(const char *host,
                                      int32_t port,
                                      const char *cache_dir,
                                      int32_t n_threads,
                                      int32_t n_devices);

/* ------------------------------------------------------------------ sessions */

typedef struct dllm_shim_session dllm_shim_session;

/* Sampler configuration, mirroring the catalog's per-model defaults so the
 * coordinator cannot accidentally change sampling between devices. */
typedef struct dllm_shim_sampler {
    int32_t top_k;
    float   top_p;
    float   temp;
    float   min_p;
    float   presence_penalty;
    int32_t seed;          /* < 0 = non-deterministic */
} dllm_shim_sampler;

/* Token callback. Return non-zero to stop generation early.
 *
 * `text_utf8` is NOT NUL-terminated — use `text_len`. `done != 0` marks the
 * final callback of a generation, with `text_len == 0`. `user` is the pointer
 * handed to `dllm_shim_session_generate`. Called on the generating thread, so
 * the implementation must not block for long: the Rust side re-enters the
 * coordinator's commit loop from here. */
typedef int (*dllm_shim_token_cb)(void *user, int32_t pos,
                                  const char *text_utf8, int32_t text_len,
                                  int32_t done);

/* Open a session over `model_path`.
 *
 * `tensor_split` assigns layer shares across ggml devices in enumeration order:
 * local devices first, then RPC devices in the order they were registered. A
 * NULL `tensor_split` (or `n_split <= 0`) means "everything local". Values are
 * relative weights, not fractions — {1,1} is an even two-way split, {2,1} gives
 * the first device twice the layers. llama.cpp normalises them.
 *
 * `n_gpu_layers` < 0 means "place every layer we can", which is what a
 * distributed session wants.
 *
 * Returns NULL on failure; see `dllm_shim_last_error`. */
DLLM_SHIM_API dllm_shim_session *dllm_shim_session_open(const char *model_path,
                                                        const float *tensor_split,
                                                        int32_t n_split,
                                                        int32_t n_gpu_layers,
                                                        int32_t n_ctx);

/* Close a session and free it. Safe on NULL. */
DLLM_SHIM_API void dllm_shim_session_close(dllm_shim_session *session);

/* Generate a completion for `prompt_utf8` (NUL-terminated), invoking
 * `on_token` per token. `max_tokens` < 0 means "until context end or EOS".
 * Returns 0 on success.
 *
 * This is the call that actually spans devices: llama.cpp's scheduler walks the
 * layer graph and hands each layer's ops to whichever backend owns its tensors,
 * so layers assigned to an RPC device execute on that device and the resulting
 * activations travel over the wire. */
DLLM_SHIM_API int dllm_shim_session_generate(dllm_shim_session *session,
                                             const char *prompt_utf8,
                                             int32_t max_tokens,
                                             const dllm_shim_sampler *sampler,
                                             dllm_shim_token_cb on_token,
                                             void *user);

/* Ask an in-flight `generate` on this session to stop at the next token.
 * Returns 0 if a stop was requested, negative if no generation is running.
 * Safe to call from another thread. */
DLLM_SHIM_API int dllm_shim_session_cancel(dllm_shim_session *session);

/* Session metadata for the observability surface. Any out pointer may be NULL. */
DLLM_SHIM_API int dllm_shim_session_n_layer(dllm_shim_session *session, int32_t *out);
DLLM_SHIM_API int dllm_shim_session_n_ctx(dllm_shim_session *session, int32_t *out);

/* Fill `out` (NUL-terminated, truncated to `out_len`) with a JSON object
 * describing what this session is actually doing:
 *
 *   {"n_layer":28,"n_ctx":4096,"n_gpu_layers":28,
 *    "tensor_split":[1,1],"devices":3,
 *    "timing":{"prefill_ms":..,"decode_ms":..,"predicted_per_second":..},
 *    "layer_owner":[0,0,0,1,1,1,...]}
 *
 * `layer_owner[i]` is the ggml device index that holds layer i's tensors —
 * derived from the tensor-split ratios we requested AND the device memory
 * llama.cpp reported, so it reflects the assignment actually made, not a
 * restatement of intent. Indices are into llama.cpp's global device list; the
 * coordinator maps them to device_ids for display.
 *
 * Returns 0 on success, negative on failure. */
DLLM_SHIM_API int dllm_shim_session_report(dllm_shim_session *session,
                                           char *out, int32_t out_len);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DLLM_SHIM_H */