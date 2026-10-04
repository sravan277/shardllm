package com.dllm.mesh.worker

/**
 * JNI bridge to libdllm_worker.so (app/src/main/cpp/worker.cpp).
 *
 * Two responsibilities, deliberately separated:
 *
 * 1. **Local decode** ([loadModel], [inferChunk], [free]) — the ADR-025 ABI. CPU,
 * single-threaded sanity path locked by that ADR; do not change or remove these
 * three without a new ADR.
 * 2. **ggml-rpc hosting** ([shimInit], [rpcServeStart], [rpcServing],
 *    [rpcServeStop], [shimFree]) — ADR-031. Lets a coordinator on another device
 *    offload transformer layers to this phone. Thin wrappers over the shared
 *    `dllm_shim_*` C ABI in `native/include/dllm_shim.h`, so the Windows and
 *    Android workers present the same surface to one coordinator.
 *
 * Every call here is guarded by [nativeAvailable]: the .so is absent on a JVM
 * unit-test classpath, and an unguarded `external` call throws
 * [UnsatisfiedLinkError]. Callers use the same runCatching-style guard the
 * worker service already applies to [inferChunk].
 */
object LlamaBridge {
    /** False when running on a JVM/unit-test classpath without the .so. */
    val nativeAvailable: Boolean

    init {
        var ok = false
        runCatching {
            System.loadLibrary("dllm_worker")
            ok = true
        }
        nativeAvailable = ok
    }

    external fun loadModel(path: String): Boolean
    external fun inferChunk(prompt: String, maxTokens: Int): String
    external fun free()

    // ---- ADR-031: ggml-rpc hosting -----------------------------------------

    /**
     * Initialise llama.cpp/ggml backends and register the RPC backend.
     * Idempotent; returns 0 on success.
     */
    external fun shimInit(): Int

    /**
     * Release global backend state. Stops hosting first. Call at shutdown.
     */
    external fun shimFree()

    /**
     * `dllm_shim_abi_version()` — the shim ABI this .so was built against, for a
     * coordinator handshake. MUST equal the value the Windows shim reports.
     */
    external fun shimAbiVersion(): Int

    /** Most recent native failure on this thread, never null. */
    external fun shimLastError(): String

    /**
     * Start hosting transformer layers for a remote coordinator.
     *
     * THREADING: non-blocking. The blocking `dllm_shim_rpc_serve()` runs on a
     * private C++ thread; this returns only once the port is confirmed accepting
     * connections (or after a ~5s timeout). Never call it on the main thread
     * expecting it to be instant, and never call it from a lock the heartbeat
     * needs.
     *
     * @param host literal IPv4 (`192.168.1.42`) or `0.0.0.0` — llama.cpp parses
     *   it with `inet_addr`, so a hostname will NOT work.
     * @param nDevices `< 0` exposes every accelerator this device has (CPU only
     *   on this build).
     * @return 0 when the endpoint is live; negative when it is not usable, in
     *   which case [shimLastError] says why (port busy, bad interface).
     */
    external fun rpcServeStart(host: String, port: Int, nThreads: Int, nDevices: Int): Int

    /**
     * Withdraw the endpoint: stops advertising it and refuses further starts.
     *
     * HONEST LIMITATION: llama.cpp b7418 exposes no way to close its RPC
     * listening socket, so the port stays bound until the app process exits. This
     * means "stopped" == "no longer advertised", NOT "port free". Callers must not
     * promise a rebind without an app restart.
     */
    external fun rpcServeStop(): Int

    /** True only when the port is confirmed listening and no stop was requested. */
    external fun rpcServing(): Boolean

    /**
     * ggml-rpc protocol version this `.so` speaks, e.g. `"3.6.0"`.
     *
     * Read from the `ggml-rpc.h` the vendored libs were compiled with, not a
     * hand-kept literal: a coordinator must be told what this worker actually
     * implements, and the pinned llama.cpp tag moves.
     */
    external fun rpcProtocolVersion(): String
}