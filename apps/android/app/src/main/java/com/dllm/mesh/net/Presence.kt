package com.dllm.mesh.net

import android.app.ActivityManager
import android.content.Context
import com.dllm.mesh.data.IdentityStore
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
 * Shared LAN presence infra (moved out of the removed Devices tab).
 *
 * Every heartbeat sender — Networks presence toggle, [com.dllm.mesh.worker.WorkerService],
 * Pairing "send request" — funnels through [postHeartbeat], which reads the
 * worker toggle + active group and builds the body with
 * [IdentityStore.buildHeartbeatBody]. Role is `worker` when the compute-worker
 * toggle is on, else `client`. node_id is never rotated here (only ensured).
 */
object Presence {

    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    /**
     * Announce this phone to a coordinator registry (auto-creates the paired
     * row). Throws on transport/HTTP error; callers translate with
     * [friendlyCause].
     */
    suspend fun postHeartbeat(appContext: Context, base: String) {
        val store = IdentityStore(appContext)
        val id = store.ensureNodeId()
        val workerOn = store.workerEnabled.first()
        val group = store.groupId.first().ifBlank { null }
        val (cpu, mem) = DeviceLoad.sample(appContext)
        val caps = if (workerOn) WorkerCapabilities.build() else null
        val layers = if (workerOn) WorkerCapabilities.offeredLayers() else null
        val body = IdentityStore.buildHeartbeatBody(
            deviceId = id,
            role = if (workerOn) "worker" else "client",
            cpuPct = cpu,
            memPct = mem,
            capabilities = caps,
            workerActive = if (workerOn) true else null,
            groupId = group,
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
 * Values are conservative phone-CPU estimates until the real per-layer
 * bench lands: `ms_per_layer_decode` ≈ measured single-thread decode cost,
 * `kv_pages` = KV cache pages this phone can hold, `layers` = total layers
 * of the served model (28 = Qwen3-0.6B shape; mirrored by [offeredLayers]).
 */
object WorkerCapabilities {

    fun build(): JSONObject = JSONObject()
        .put("ms_per_layer_decode", 35.0)
        .put("kv_pages", 256)
        .put("layers", 28)

    /** Layer range this worker offers today: the full single-device 0–27. */
    fun offeredLayers(): List<Int> = (0..27).toList()
}
