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
 * "coordinator needs upgrade" toast and an honest empty state — never a crash.
 */
class CoordinatorNeedsUpgradeException(message: String) : IOException(message)

/**
 * Thrown when the coordinator answers 403 for a group-scoped usage query
 * (not a member / non-admin). Carries the totals-only [summary] parsed out
 * of the 403 body so the UI can still show honest totals instead of nothing.
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
 * One private group from `GET /v1/networks`.
 * Tolerates old + new shapes: id/group_id, member_count/members size.
 */
data class MeshNetwork(
    val id: String,
    val name: String,
    val openJoin: Boolean,
    val hasPassword: Boolean,
    val memberCount: Int?,
    val createdAt: String?,
)

/** One member row from `GET /v1/networks/{id}/devices`. */
data class NetworkDevice(
    val deviceId: String,
    val deviceName: String?,
    val role: String?,
    val active: Boolean,
    val status: String?,
    val lastSeen: String?,
) {
    val connected: Boolean get() = active && status != "revoked"
}

/**
 * Coordinator REST surface used by the chat-list + usage overhaul.
 * SSE streaming stays in [SseClient]; this object owns the plain JSON calls.
 *
 * Contract (backend crew implements in parallel — code against exactly this):
 * - `GET /v1/sessions` entries: {id/session_id, model, tokens_out,
 *   last_token_at, title}
 * - `POST /v1/sessions/{id}/rename` {title} -> {ok:true,id,title}
 *   (400 empty/>80 chars, 404 unknown)
 * - `DELETE /v1/sessions/{id}` -> {ok:true,id} (404 unknown)
 * - `GET /v1/usage` -> {tokens_out_total, sessions_total, per_device[],
 *   plan:{stages}, bandwidth:null}
 * - `GET /v1/plan` -> {plan_id, stages[]}
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

    suspend fun getUsage(
        baseUrl: String,
        groupId: String? = null,
        deviceId: String? = null,
    ): UsageSummary =
        withContext(Dispatchers.IO) {
            val path = buildString {
                append("/v1/usage")
                val q = ArrayList<String>(2)
                if (!groupId.isNullOrBlank()) {
                    q.add("group=${urlEncode(groupId.trim())}")
                }
                if (!deviceId.isNullOrBlank()) {
                    q.add("device_id=${urlEncode(deviceId.trim())}")
                }
                if (q.isNotEmpty()) {
                    append("?").append(q.joinToString("&"))
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
                    // Group-scoped denial still carries honest totals in the
                    // body (backend contract) — surface them instead of nothing.
                    val totals = runCatching { parseUsage(body) }.getOrNull()
                        ?: UsageSummary(0, 0, emptyList(), emptyList())
                    throw UsageForbiddenException(
                        "not a group member (GET $path -> HTTP 403)",
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
     * the coordinator to stop decoding too. Unknown/stale servers 404 —
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

    // ---- networks (private groups, all LAN-visible) -------------------------
    //
    // Contract (backend crew implements in parallel — code against exactly this):
    // - `GET /v1/networks` -> {networks:[{id,name,open_join,has_password,
    //   member_count,created_at}]} (bare array also accepted)
    // - `POST /v1/networks` {name,password?,open_join} -> {id,...} (400 bad name)
    // - `GET /v1/networks/{id}/devices` -> {devices:[{device_id,device_name,
    //   role,active,status,last_seen}]}; also accepts {paired:[...],
    //   active_now:[...]} (activeNow derived from `active` otherwise)
    // - `POST /v1/networks/{id}/join` {device_id,password?,qr_secret?}
    //   -> {ok:true} (403 wrong password, 404 unknown group)
    // - `POST /v1/networks/{id}/leave` {device_id} -> {ok:true}

    suspend fun listNetworks(baseUrl: String): List<MeshNetwork> =
        withContext(Dispatchers.IO) {
            val text = getOrThrow(baseUrl, "/v1/networks")
            parseNetworks(text)
        }

    suspend fun createNetwork(
        baseUrl: String,
        name: String,
        password: String?,
        openJoin: Boolean,
    ): MeshNetwork = withContext(Dispatchers.IO) {
        val url = "${baseUrl.trimEnd('/')}/v1/networks"
        val payload = JSONObject()
            .put("name", name)
            .put("open_join", openJoin)
        if (!password.isNullOrEmpty()) payload.put("password", password)
        val req = Request.Builder()
            .url(url)
            .post(payload.toString().toRequestBody("application/json".toMediaType()))
            .build()
        val text = http.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) {
                val body = resp.body?.string().orEmpty()
                throw IOException("POST $url -> HTTP ${resp.code} $body")
            }
            resp.body?.string().orEmpty()
        }
        parseNetworks(text).firstOrNull()
            ?: parseSingleNetwork(text)
            ?: throw IOException("Empty network from $url")
    }

    suspend fun getNetworkDevices(baseUrl: String, id: String): List<NetworkDevice> =
        withContext(Dispatchers.IO) {
            val text = getOrThrow(baseUrl, "/v1/networks/$id/devices")
            parseNetworkDevices(text)
        }

    suspend fun joinNetwork(
        baseUrl: String,
        id: String,
        deviceId: String,
        password: String?,
        qrSecret: String?,
        deviceName: String? = null,
    ) = withContext(Dispatchers.IO) {
        val url = "${baseUrl.trimEnd('/')}/v1/networks/$id/join"
        val payload = JSONObject().put("device_id", deviceId)
        if (!deviceName.isNullOrBlank()) payload.put("device_name", deviceName)
        if (!password.isNullOrEmpty()) payload.put("password", password)
        if (!qrSecret.isNullOrEmpty()) payload.put("qr_secret", qrSecret)
        val req = Request.Builder()
            .url(url)
            .post(payload.toString().toRequestBody("application/json".toMediaType()))
            .build()
        http.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) {
                val body = resp.body?.string().orEmpty()
                throw IOException("POST $url -> HTTP ${resp.code} $body")
            }
        }
    }

    suspend fun leaveNetwork(baseUrl: String, id: String, deviceId: String) =
        withContext(Dispatchers.IO) {
            val url = "${baseUrl.trimEnd('/')}/v1/networks/$id/leave"
            val payload = JSONObject().put("device_id", deviceId).toString()
            val req = Request.Builder()
                .url(url)
                .post(payload.toRequestBody("application/json".toMediaType()))
                .build()
            http.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) {
                    val body = resp.body?.string().orEmpty()
                    throw IOException("POST $url -> HTTP ${resp.code} $body")
                }
            }
        }

    fun parseNetworks(text: String): List<MeshNetwork> {
        val arr: JSONArray = runCatching {
            val trimmed = text.trim()
            if (trimmed.startsWith("[")) JSONArray(trimmed)
            else JSONObject(trimmed).optJSONArray("networks") ?: JSONArray()
        }.getOrNull() ?: JSONArray()
        val out = ArrayList<MeshNetwork>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            parseSingleNetwork(o)?.let { out.add(it) }
        }
        return out
    }

    private fun parseSingleNetwork(text: String): MeshNetwork? = runCatching {
        val o = JSONObject(text.trim())
        val inner = o.optJSONObject("network") ?: o
        parseSingleNetwork(inner)
    }.getOrNull()

    private fun parseSingleNetwork(o: JSONObject): MeshNetwork? {
        val id = o.optString("id").ifBlank { o.optString("group_id") }
        if (id.isBlank()) return null
        val name = o.optString("name").trim().ifBlank { id }
        val openJoin = o.optBoolean("open_join", true)
        val hasPassword = when {
            !o.isNull("has_password") -> o.optBoolean("has_password", false)
            !o.isNull("password_hash") -> o.optString("password_hash").isNotBlank()
            else -> false
        }
        val memberCount = when {
            !o.isNull("member_count") -> o.optInt("member_count").takeIf { it >= 0 }
            !o.isNull("members") -> o.optJSONArray("members")?.length()
            else -> null
        }
        val createdAt = o.optString("created_at").trim().ifBlank { null }
        return MeshNetwork(id, name, openJoin, hasPassword, memberCount, createdAt)
    }

    /** Accepts `{devices:[...]}` or `{paired:[...],active_now:[...]}` shapes. */
    fun parseNetworkDevices(text: String): List<NetworkDevice> {
        val obj = runCatching { JSONObject(text.trim()) }.getOrNull()
            ?: return emptyList()
        val arr: JSONArray = obj.optJSONArray("devices")
            ?: run {
                val paired = obj.optJSONArray("paired") ?: JSONArray()
                val activeNow = obj.optJSONArray("active_now") ?: JSONArray()
                val merged = JSONArray()
                for (i in 0 until paired.length()) {
                    merged.put(paired.optJSONObject(i) ?: continue)
                }
                for (i in 0 until activeNow.length()) {
                    val o = activeNow.optJSONObject(i) ?: continue
                    // active_now entries may be id strings — normalize to objects.
                    merged.put(o)
                }
                merged
            }
        val out = ArrayList<NetworkDevice>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            val id = o.optString("device_id")
                .ifBlank { o.optString("id").ifBlank { o.optString("node_id") } }
            if (id.isBlank()) continue
            out.add(
                NetworkDevice(
                    deviceId = id,
                    deviceName = o.optString("device_name").trim().ifBlank { null },
                    role = o.optString("role").trim().ifBlank { null },
                    active = o.optBoolean("active", false),
                    status = o.optString("status").trim().ifBlank { null },
                    lastSeen = o.optString("last_seen").trim().ifBlank { null },
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
     * `layer_start`/`layer_end`. Null when absent — never synthesized.
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
