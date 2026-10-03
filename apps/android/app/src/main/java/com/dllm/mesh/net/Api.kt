package com.dllm.mesh.net

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Thrown when the coordinator answers 404 for an endpoint the app expects
 * (sessions rename/delete, usage, plan). Callers translate this into a
 * "coordinator needs upgrade" toast and an honest empty state â€” never a crash.
 */
class CoordinatorNeedsUpgradeException(message: String) : IOException(message)

/**
 * Thrown when the coordinator answers 403 for a usage query. Carries the
 * totals-only [summary] parsed out of the 403 body so the UI can still show
 * honest totals instead of nothing.
 */
class UsageForbiddenException(
    message: String,
    val summary: UsageSummary,
) : IOException(message)

/** Returns true when [e] signals a coordinator-too-old 404. */
fun isNeedsUpgrade(e: Throwable): Boolean {
    if (e is CoordinatorNeedsUpgradeException) return true
    val msg = e.message.orEmpty()
    return msg.contains("HTTP 404") || msg.contains(" 404")
}

/**
 * One server session from `GET /v1/sessions`.
 * Tolerates old + new shapes: id/session_id, optional title, optional
 * created_at/last_token_at. Missing title falls back to "New chat" upstream
 * (the server default for untitled chats).
 */
data class ChatSession(
    val id: String,
    val title: String,
    val model: String,
    val tokensOut: Int,
    val lastTokenAt: String?,
    val createdAt: String?,
)

data class PlanStage(
    val deviceId: String,
    val layerStart: Int,
    val layerEnd: Int,
    val stage: Int? = null,
)

data class UsageDevice(
    val deviceId: String,
    val deviceName: String?,
    val role: String?,
    val active: Boolean,
    val status: String?,
    val tokensOut: Int,
    val sessions: Int,
    val cpuPct: Double?,
    val memPct: Double?,
    val loadSource: String,
    val layerStart: Int?,
    val layerEnd: Int?,
    /** Heartbeat `worker_active` flag; null = never reported (older row). */
    val workerActive: Boolean?,
    /** Heartbeat `layers` array (e.g. phone offering 0-27); null = not reported. */
    val layers: List<Int>?,
)

data class UsageSummary(
    val tokensOutTotal: Int,
    val sessionsTotal: Int,
    val perDevice: List<UsageDevice>,
    val planStages: List<PlanStage>,
)

/**
 * One row of `GET /v1/devices` â€” the mesh's single registry of paired devices.
 *
 * WHY this exists (replacing the removed per-network device lists): the mesh is
 * one network now, so a device has exactly one home. Fields are split by
 * *authority*:
 * - registry authority (deviceId â€¦ lastSeen) comes from `/v1/devices`
 * - load/worker/layer fields come from `/v1/usage` + `/v1/plan`, because the
 *   registry deliberately reports no load numbers; every load field is nullable
 *   and the UI must render "not reporting" rather than a synthesised zero.
 *
 * [deviceName] stays null when the device never sent one â€” the server does not
 * invent a fallback, and neither does this client; the UI shows a short id.
 */
data class MeshDevice(
    val deviceId: String,
    val deviceName: String?,
    val role: String?,
    /** `paired` | `revoked`; null when the row predates the field. */
    val status: String?,
    /** True when last_seen is inside the coordinator's 90s activity window. */
    val active: Boolean,
    /** Raw ISO-8601 UTC timestamp as stored by the coordinator. */
    val lastSeen: String?,
    val pairedAt: String?,
    val cpuPct: Double? = null,
    val memPct: Double? = null,
    /** `live` | `reported` | `none` â€” provenance of cpu/mem. */
    val loadSource: String = "none",
    /** Heartbeat `worker_active`; null = the device has never reported it. */
    val workerActive: Boolean? = null,
    /** Layers actually assigned by `/v1/plan`; null = no assignment reported. */
    val assignedLayers: List<Int>? = null,
    /** Layers the device *offered* in its heartbeat; null = never offered. */
    val offeredLayers: List<Int>? = null,
    /** True for this phone's own row (matches the saved node_id). */
    val isSelf: Boolean = false,
) {
    /** A revoked row must never be shown as a healthy mesh member. */
    val revoked: Boolean get() = status == "revoked"

    /** Registry-liveness AND not revoked â€” what "online in the mesh" means. */
    val online: Boolean get() = active && !revoked
}

/**
 * Coordinator REST surface used by the chat-list + usage + devices screens.
 * SSE streaming stays in [SseClient]; this object owns the plain JSON calls.
 *
 * Contract (code against exactly this):
 * - `GET /v1/sessions` entries: {id/session_id, model, tokens_out,
 *   last_token_at, title}
 * - `POST /v1/sessions/{id}/rename` {title} -> {ok:true,id,title}
 *   (400 empty/>80 chars, 404 unknown)
 * - `DELETE /v1/sessions/{id}` -> {ok:true,id} (404 unknown)
 * - `GET /v1/usage` -> {tokens_out_total, sessions_total, per_device[],
 *   plan:{stages}, bandwidth:null}
 * - `GET /v1/plan` -> {plan_id, stages[]}
 * - `GET /v1/devices` -> {devices:[{device_id,device_name,role,permissions,
 *   status,active,last_seen,paired_at,paired_by}]} (bare array also accepted)
 * - `POST /v1/devices/{id}/approve` | `/revoke` -> {ok:true}
 *   (404 unknown device)
 * - `DELETE /v1/devices/{id}` -> {ok:true,id} (404 unknown; 400 when the id is
 *   the coordinator's own row â€” the server refuses to orphan the mesh)
 *
 * Every call throws [CoordinatorNeedsUpgradeException] on HTTP 404 so the UI
 * can toast "coordinator needs upgrade" and render an honest empty state.
 */
object DllmApi {
    private val http = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(15, TimeUnit.SECONDS)
        .build()

    private fun getOrThrow(baseUrl: String, path: String): String {
        val url = "${baseUrl.trimEnd('/')}$path"
        val req = Request.Builder().url(url).get().build()
        http.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) {
                if (resp.code == 404) {
                    throw CoordinatorNeedsUpgradeException(
                        "coordinator needs upgrade (GET $path -> HTTP 404)"
                    )
                }
                throw IOException("GET $url -> HTTP ${resp.code}")
            }
            return resp.body?.string().orEmpty()
        }
    }

    suspend fun listSessions(baseUrl: String): List<ChatSession> =
        withContext(Dispatchers.IO) {
            val text = getOrThrow(baseUrl, "/v1/sessions")
            parseSessions(text)
        }

    suspend fun renameSession(baseUrl: String, id: String, title: String) =
        withContext(Dispatchers.IO) {
            val url = "${baseUrl.trimEnd('/')}/v1/sessions/$id/rename"
            val payload = JSONObject().put("title", title).toString()
            val req = Request.Builder()
                .url(url)
                .post(payload.toRequestBody("application/json".toMediaType()))
                .build()
            http.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) {
                    if (resp.code == 404) {
                        throw CoordinatorNeedsUpgradeException(
                            "coordinator needs upgrade (POST rename -> HTTP 404)"
                        )
                    }
                    val body = resp.body?.string().orEmpty()
                    throw IOException("POST $url -> HTTP ${resp.code} $body")
                }
            }
        }

    suspend fun deleteSession(baseUrl: String, id: String) =
        withContext(Dispatchers.IO) {
            val url = "${baseUrl.trimEnd('/')}/v1/sessions/$id"
            val req = Request.Builder().url(url).delete().build()
            http.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) {
                    if (resp.code == 404) {
                        throw CoordinatorNeedsUpgradeException(
                            "coordinator needs upgrade (DELETE session -> HTTP 404)"
                        )
                    }
                    throw IOException("DELETE $url -> HTTP ${resp.code}")
                }
            }
        }

    /**
     * `GET /v1/usage`, optionally narrowed to this phone's own row.
     *
     * WHY no group parameter: the mesh is a single network now, so the group
     * query parameter (and its 403 "not a member" path) no longer has a meaning
     * this client can express. [deviceId] remains because "my share" is still a
     * real question the Usage tab asks.
     */
    suspend fun getUsage(
        baseUrl: String,
        deviceId: String? = null,
    ): UsageSummary =
        withContext(Dispatchers.IO) {
            val path = buildString {
                append("/v1/usage")
                if (!deviceId.isNullOrBlank()) {
                    append("?device_id=").append(urlEncode(deviceId.trim()))
                }
            }
            val url = "${baseUrl.trimEnd('/')}$path"
            val req = Request.Builder().url(url).get().build()
            http.newCall(req).execute().use { resp ->
                val body = resp.body?.string().orEmpty()
                if (resp.isSuccessful) return@withContext parseUsage(body)
                if (resp.code == 404) {
                    throw CoordinatorNeedsUpgradeException(
                        "coordinator needs upgrade (GET $path -> HTTP 404)"
                    )
                }
                if (resp.code == 403) {
                    // A denial still carries honest totals in the body
                    // (backend contract) â€” surface them instead of nothing.
                    val totals = runCatching { parseUsage(body) }.getOrNull()
                        ?: UsageSummary(0, 0, emptyList(), emptyList())
                    throw UsageForbiddenException(
                        "not permitted (GET $path -> HTTP 403)",
                        totals,
                    )
                }
                throw IOException("GET $url -> HTTP ${resp.code} $body")
            }
        }

    private fun urlEncode(raw: String): String =
        java.net.URLEncoder.encode(raw, "UTF-8")

    suspend fun getPlan(baseUrl: String): List<PlanStage> =
        withContext(Dispatchers.IO) {
            val text = getOrThrow(baseUrl, "/v1/plan")
            parsePlanStages(text)
        }

    /**
     * Best-effort stop: `POST /v1/sessions/{id}/stop`. The local
     * `streamJob.cancel()` in the ViewModel is the real cancel; this tells
     * the coordinator to stop decoding too. Unknown/stale servers 404 â€”
     * treated as stopped (local cancel already happened).
     */
    suspend fun stopSession(baseUrl: String, id: String) =
        withContext(Dispatchers.IO) {
            val url = "${baseUrl.trimEnd('/')}/v1/sessions/$id/stop"
            val req = Request.Builder()
                .url(url)
                .post("{}".toRequestBody("application/json".toMediaType()))
                .build()
            http.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful && resp.code != 404) {
                    throw IOException("POST $url -> HTTP ${resp.code}")
                }
            }
        }

    // ---- devices (the mesh is a single registry) -----------------------------

    /**
     * `GET /v1/devices` — every paired device the coordinator knows about.
     *
     * This is the only device list the app has: the mesh is one network, so a
     * device has one row here and there is no per-group enumeration to walk.
     * Load numbers are NOT in this response by design (see [MeshDevice]); the
     * caller merges them from `GET /v1/usage`.
     */
    suspend fun listDevices(baseUrl: String): List<MeshDevice> =
        withContext(Dispatchers.IO) {
            val text = getOrThrow(baseUrl, "/v1/devices")
            parseDevices(text)
        }

    /**
     * `POST /v1/devices/{id}/approve` — re-admit a revoked row.
     * 404 for an unknown id is surfaced as a [CoordinatorNeedsUpgradeException]
     * only when the endpoint itself is missing; a real "unknown device" 404 is
     * reported by the generic branch below with its body so the UI can say so.
     */
    suspend fun approveDevice(baseUrl: String, deviceId: String) =
        deviceStatusAction(baseUrl, deviceId, "approve", "approved")

    /** `POST /v1/devices/{id}/revoke` — mark a row revoked (it stops contributing). */
    suspend fun revokeDevice(baseUrl: String, deviceId: String) =
        deviceStatusAction(baseUrl, deviceId, "revoke", "revoked")

    private suspend fun deviceStatusAction(
        baseUrl: String,
        deviceId: String,
        action: String,
        pastTense: String,
    ) = withContext(Dispatchers.IO) {
        val url = "${baseUrl.trimEnd('/')}/v1/devices/$deviceId/$action"
        val req = Request.Builder()
            .url(url)
            .post("{}".toRequestBody("application/json".toMediaType()))
            .build()
        http.newCall(req).execute().use { resp ->
            if (resp.isSuccessful) return@withContext
            val body = resp.body?.string().orEmpty()
            throw IOException("device $pastTense failed: HTTP ${resp.code} $body")
        }
    }

    /**
     * `DELETE /v1/devices/{id}` — hard-delete the row.
     *
     * WHY the 400 is special: the coordinator refuses to delete its own row
     * ("never orphan the mesh"), and that refusal is the *correct* answer, not
     * a bug. The UI must show it verbatim rather than pretending it worked, so
     * the body is carried through in the message.
     */
    suspend fun deleteDevice(baseUrl: String, deviceId: String) =
        withContext(Dispatchers.IO) {
            val url = "${baseUrl.trimEnd('/')}/v1/devices/$deviceId"
            val req = Request.Builder().url(url).delete().build()
            http.newCall(req).execute().use { resp ->
                if (resp.isSuccessful) return@withContext
                val body = resp.body?.string().orEmpty()
                throw IOException("delete refused: HTTP ${resp.code} $body")
            }
        }

    /**
     * Parses `GET /v1/devices`: `{devices:[...]}`, a bare array, or a single
     * row. Rows without an id are skipped rather than rendered as blank cards.
     * No field is invented — a missing name stays null so the UI can fall back
     * to a short id instead of showing a fabricated "Unknown device".
     */
    fun parseDevices(text: String): List<MeshDevice> {
        val arr: JSONArray = runCatching {
            val trimmed = text.trim()
            if (trimmed.startsWith("[")) JSONArray(trimmed)
            else JSONObject(trimmed).optJSONArray("devices")
                ?: if (trimmed.startsWith("{") && trimmed.contains("\"device_id\"")) {
                    JSONArray().put(JSONObject(trimmed))
                } else {
                    JSONArray()
                }
        }.getOrNull() ?: JSONArray()
        val out = ArrayList<MeshDevice>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            val id = o.optString("device_id")
                .ifBlank { o.optString("id").ifBlank { o.optString("node_id") } }
            if (id.isBlank()) continue
            out.add(
                MeshDevice(
                    deviceId = id,
                    deviceName = o.optString("device_name").trim().ifBlank { null },
                    role = o.optString("role").trim().ifBlank { null },
                    status = o.optString("status").trim().ifBlank { null },
                    active = o.optBoolean("active", false),
                    lastSeen = o.optString("last_seen").trim().ifBlank { null },
                    pairedAt = o.optString("paired_at").trim().ifBlank { null },
                )
            )
        }
        return out
    }


    /** Accepts `{sessions:[...]}` or a bare array; skips entries without id. */
    fun parseSessions(text: String): List<ChatSession> {
        val arr: JSONArray = runCatching {
            val trimmed = text.trim()
            if (trimmed.startsWith("[")) JSONArray(trimmed)
            else JSONObject(trimmed).optJSONArray("sessions") ?: JSONArray()
        }.getOrNull() ?: JSONArray()
        val out = ArrayList<ChatSession>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            val id = o.optString("id").ifBlank { o.optString("session_id") }
            if (id.isBlank()) continue
            val title = o.optString("title").trim().ifBlank { "New chat" }
            val model = o.optString("model").trim().ifBlank { "unknown" }
            val tokensOut = o.optInt("tokens_out", 0).coerceAtLeast(0)
            val lastTokenAt = o.optString("last_token_at").trim().ifBlank { null }
            val createdAt = o.optString("created_at").trim().ifBlank { null }
            out.add(ChatSession(id, title, model, tokensOut, lastTokenAt, createdAt))
        }
        return out
    }

    fun parsePlanStages(text: String): List<PlanStage> {
        val obj = runCatching { JSONObject(text) }.getOrNull() ?: return emptyList()
        // Accepts /v1/plan ({stages}) and /v1/usage (.plan.stages).
        val stages: JSONArray? =
            obj.optJSONArray("stages")
                ?: obj.optJSONObject("plan")?.optJSONArray("stages")
        if (stages == null) return emptyList()
        val out = ArrayList<PlanStage>(stages.length())
        for (i in 0 until stages.length()) {
            val o = stages.optJSONObject(i) ?: continue
            val deviceId = o.optString("device_id").ifBlank { o.optString("deviceId") }
            if (deviceId.isBlank()) continue
            if (o.isNull("layer_start") || o.isNull("layer_end")) continue
            val lo = o.optInt("layer_start", -1)
            val hi = o.optInt("layer_end", -1)
            if (lo < 0 || hi < 0) continue
            val stage = if (o.isNull("stage")) null else o.optInt("stage")
            out.add(PlanStage(deviceId, lo, hi, stage))
        }
        return out
    }

    fun parseUsage(text: String): UsageSummary {
        val obj = JSONObject(text)
        val total = obj.optInt("tokens_out_total", 0).coerceAtLeast(0)
        val sessTotal = obj.optInt("sessions_total", 0).coerceAtLeast(0)
        val arr = obj.optJSONArray("per_device") ?: JSONArray()
        val devices = ArrayList<UsageDevice>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            val deviceId = o.optString("device_id").ifBlank { o.optString("id") }
            if (deviceId.isBlank()) continue
            val deviceName = o.optString("device_name").trim().ifBlank { null }
            val role = o.optString("role").trim().ifBlank { null }
            val active = o.optBoolean("active", false)
            val status = o.optString("status").trim().ifBlank { null }
            val tokensOut = o.optInt("tokens_out", 0).coerceAtLeast(0)
            val sess = o.optInt("sessions", 0).coerceAtLeast(0)
            val cpu = if (o.isNull("cpu_pct")) null else o.optDouble("cpu_pct").takeIf { it.isFinite() }
            val mem = if (o.isNull("mem_pct")) null else o.optDouble("mem_pct").takeIf { it.isFinite() }
            val loadSource = o.optString("load_source").trim().ifBlank { "none" }
            val lo = if (o.isNull("layer_start")) null else o.optInt("layer_start").takeIf { it >= 0 }
            val hi = if (o.isNull("layer_end")) null else o.optInt("layer_end").takeIf { it >= 0 }
            val workerActive = if (o.isNull("worker_active")) null
            else o.optBoolean("worker_active").let { it }
                .takeIf { o.opt("worker_active") is Boolean }
            val layers = parseLayersValue(o.opt("layers"))
            devices.add(
                UsageDevice(
                    deviceId = deviceId,
                    deviceName = deviceName,
                    role = role,
                    active = active,
                    status = status,
                    tokensOut = tokensOut,
                    sessions = sess,
                    cpuPct = cpu,
                    memPct = mem,
                    loadSource = loadSource,
                    layerStart = lo,
                    layerEnd = hi,
                    workerActive = workerActive,
                    layers = layers
                )
            )
        }
        val planStages = parsePlanStages(text)
        return UsageSummary(total, sessTotal, devices, planStages)
    }

    /**
     * Parses the heartbeat `layers` value from a usage entry: a JSON array
     * of layer ints (the phone's offer, e.g. 0-27), or an object with
     * `layer_start`/`layer_end`. Null when absent â€” never synthesized.
     */
    private fun parseLayersValue(v: Any?): List<Int>? {
        if (v == null || v == JSONObject.NULL) return null
        if (v is JSONArray) {
            val out = ArrayList<Int>(v.length())
            for (i in 0 until v.length()) {
                val n = v.optInt(i, -1)
                if (n >= 0) out.add(n)
            }
            return out.ifEmpty { null }
        }
        if (v is JSONObject) {
            val lo = v.optInt("layer_start", -1)
            val hi = v.optInt("layer_end", -1)
            if (lo < 0 || hi < lo) return null
            return (lo..hi).toList()
        }
        return null
    }
}
