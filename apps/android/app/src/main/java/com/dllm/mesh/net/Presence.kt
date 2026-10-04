package com.dllm.mesh.net

import android.app.ActivityManager
import android.content.Context
import com.dllm.mesh.data.IdentityStore
import com.dllm.mesh.data.ModelTopology
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import java.util.concurrent.TimeUnit

/**
 * Shared LAN presence infra: the one place a device announces itself.
 *
 * Every heartbeat sender — Devices presence, [com.dllm.mesh.worker.WorkerService],
 * Pairing's connect/reconnect — funnels through [postHeartbeat], which reads the
 * compute-worker toggle and builds the body with
 * [IdentityStore.buildHeartbeatBody]. Role is `worker` when the toggle is on,
 * else `client`. node_id is never rotated here (only ensured).
 */
object Presence {

    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    /**
     * Announce this phone to a coordinator registry (upserts the paired row).
     * Throws on transport/HTTP error; callers translate with
     * [friendlyCause].
     *
     * @param rpcEndpoint `host:port` of the ggml-rpc server this phone is hosting
     *   for offloaded transformer layers (ADR-031), or null when it is hosting
     *   none. Null means "cannot host", which the coordinator must be able to see:
     *   the field is then absent from `capabilities`, never an empty string.
     * @param rpcProtocol ggml-rpc protocol version the native server speaks, from
     *   `LlamaBridge.rpcProtocolVersion()`. Omitted when unknown.
     */
    suspend fun postHeartbeat(
        appContext: Context,
        base: String,
        rpcEndpoint: String? = null,
        rpcProtocol: String? = null,
    ) {
        val store = IdentityStore(appContext)
        val id = store.ensureNodeId()
        val workerOn = store.workerEnabled.first()
        val (cpu, mem) = DeviceLoad.sample(appContext)
        val caps = if (workerOn) WorkerCapabilities.build(rpcEndpoint, rpcProtocol) else null
        val layers = if (workerOn) WorkerCapabilities.offeredLayers() else null
        val body = IdentityStore.buildHeartbeatBody(
            deviceId = id,
            role = if (workerOn) "worker" else "client",
            cpuPct = cpu,
            memPct = mem,
            capabilities = caps,
            workerActive = if (workerOn) true else null,
            layers = layers,
        ).toRequestBody("application/json; charset=utf-8".toMediaType())
        val req = Request.Builder()
            .url("${base.trimEnd('/')}/v1/devices/heartbeat")
            .post(body)
            .build()
        withContext(Dispatchers.IO) {
            http.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) throw IOException("HTTP ${resp.code}")
            }
        }
    }

    fun friendlyCause(e: Throwable): String {
        val msg = e.message.orEmpty()
        if (msg.startsWith("HTTP ")) return "server returned $msg"
        if (e is UnknownHostException || msg.contains("Unable to resolve host", ignoreCase = true)) {
            return "cannot resolve the server host"
        }
        if (e is SocketTimeoutException || msg.contains("timeout", ignoreCase = true)) {
            return "connection timed out"
        }
        if (e is ConnectException || msg.contains("refused", ignoreCase = true) ||
            msg.contains("failed to connect", ignoreCase = true)
        ) {
            return "connection refused — the server may be offline"
        }
        if (e is IOException && msg.isNotBlank()) return "network error ($msg)"
        if (msg.isNotBlank()) return msg
        return "unexpected error (${e::class.simpleName ?: "unknown"})"
    }
}

/**
 * Phone load sampler (RMX3785 fix): MEM via ActivityManager (reliable),
 * CPU via a single-shot /proc/stat aggregate (lifetime average, 0-100).
 * Returns `(cpuPct, memPct)`; either is null when it cannot be read —
 * callers must forward the nulls honestly, never invent numbers.
 */
object DeviceLoad {

    fun sample(context: Context): Pair<Double?, Double?> =
        readCpuPct() to readMemPct(context)

    private fun readMemPct(context: Context): Double? = runCatching {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val info = ActivityManager.MemoryInfo()
        am.getMemoryInfo(info)
        if (info.totalMem <= 0L) return@runCatching null
        val used = info.totalMem - info.availMem
        (used.toDouble() * 100.0 / info.totalMem.toDouble()).coerceIn(0.0, 100.0)
    }.getOrNull()

    private fun readCpuPct(): Double? = runCatching {
        // First line: cpu  user nice system idle iowait irq softirq steal ...
        val line = java.io.File("/proc/stat").bufferedReader().useLines { seq ->
            seq.firstOrNull { it.startsWith("cpu ") }
        } ?: return@runCatching null
        val parts = line.trim().split(Regex("\\s+")).drop(1).mapNotNull { it.toLongOrNull() }
        if (parts.size < 4) return@runCatching null
        val idle = parts[3] + parts.getOrElse(4) { 0L }
        val total = parts.sum()
        if (total <= 0L) return@runCatching null
        ((total - idle).toDouble() * 100.0 / total.toDouble()).coerceIn(0.0, 100.0)
    }.getOrNull()
}

/**
 * Worker capability advertisement for `/v1/plan` layer assignment.
 *
 * The three fields are NOT all the same kind of truth, and the split is
 * deliberate so the planner can tell them apart:
 *
 * - `layers` — a property of the served model, not of this phone. Read from
 *   [ModelTopology], so it can never disagree with the ranges the UI renders.
 * - `ms_per_layer_decode` — **PLACEHOLDER.** The honest number would be a
 *   measured per-layer decode cost from llama.cpp on this exact device; the JNI
 *   bridge returns decoded *text* only ([com.dllm.mesh.worker.LlamaBridge.inferChunk]),
 *   so no timing is available to report. The planner weights layers by this
 *   figure, so the constant biases assignment until real calibration lands
 *   (`POST /v1/devices/{id}/calibrate`). It is a constant on purpose: a
 *   fabricated measurement would be worse than a labelled estimate.
 * - `kv_pages` — **PLACEHOLDER.** True KV capacity depends on the loaded model's
 *   per-layer head count and the memory actually left after Android's own
 *   overhead; neither is readable from here without loading the model. Kept as a
 *   conservative fixed value.
 *
 * `rpc_endpoint` is different again, and is a *fact* rather than an estimate
 * (ADR-031): it is present only when a native connect() to that ggml-rpc port
 * succeeded, and its value is exactly the `host:port` string the coordinator
 * passes to `dllm_shim_add_rpc_server`. `rpc_port` is the same fact split for
 * registries that store host and port in separate columns, and `rpc_protocol` is
 * read from the vendored `ggml-rpc.h` at build time. All three are omitted — not
 * null, not blank — when this phone is not hosting, so "cannot host layers" is a
 * fact the planner can act on rather than a guess.
 *
 * `rpc_protocol` exists so a coordinator never confuses this endpoint with the
 * mesh activation-frame protocol (ADR-029): this is llama.cpp's own plaintext
 * ggml-rpc, and it must not be handed to the mesh client.
 */
object WorkerCapabilities {

    fun build(rpcEndpoint: String? = null, rpcProtocol: String? = null): JSONObject {
        val caps = JSONObject()
            .put("ms_per_layer_decode", MS_PER_LAYER_DECODE_PLACEHOLDER)
            .put("kv_pages", KV_PAGES_PLACEHOLDER)
            .put("layers", ModelTopology.TOTAL_LAYERS)
        if (!rpcEndpoint.isNullOrBlank()) {
            caps.put("rpc_endpoint", rpcEndpoint)
            caps.put("rpc_port", rpcEndpoint.substringAfterLast(':', "").toIntOrNull() ?: 0)
            if (!rpcProtocol.isNullOrBlank()) caps.put("rpc_protocol", rpcProtocol)
        }
        return caps
    }

    /** Layer range this worker offers today: the full single-device span. */
    fun offeredLayers(): List<Int> = ModelTopology.allLayers()

    /** PLACEHOLDER — see the class KDoc. Not a measurement of this phone. */
    private const val MS_PER_LAYER_DECODE_PLACEHOLDER = 35.0

    /** PLACEHOLDER — see the class KDoc. Not a measurement of this phone. */
    private const val KV_PAGES_PLACEHOLDER = 256
}
