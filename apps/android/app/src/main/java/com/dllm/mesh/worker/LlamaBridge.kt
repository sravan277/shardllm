package com.dllm.mesh.worker

/**
 * JNI bridge to libdllm_worker.so (app/src/main/cpp/worker.cpp).
 *
 * Phase 4: CPU, single-threaded sanity. [loadModel] only checks the GGUF path
 * is readable; [inferChunk] returns a deterministic stub echo; [free] clears
 * native state. Real llama.cpp decode keeps this exact API in Phase 5.
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
}
