package com.dllm.mesh.net

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.sse.EventSource
import okhttp3.sse.EventSourceListener
import okhttp3.sse.EventSources
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Streamed chat events from the coordinator SSE endpoint.
 * [eventId] carries the SSE `id` field so callers can persist it as
 * lastEventId for reconnect/resume.
 */
sealed interface ChatEvent {
    data class Token(val text: String, val eventId: String? = null) : ChatEvent
    data class Committed(val position: Int, val eventId: String? = null) : ChatEvent
    data class Status(val message: String, val eventId: String? = null) : ChatEvent
}

/**
 * Server identity from GET /api/node ->
 * {"node_id","fingerprint","quic_port","version"}.
 * Optional enhancement: older servers may 404; callers must handle that.
 */
data class NodeInfo(
    val nodeId: String,
    val fingerprint: String,
    val quicPort: Int?,
    val version: String,
)

/**
 * OkHttp SSE client (Phase 0 transport; QUIC/TCP-binary lands behind this
 * same shape in Phase 3 per research).
 *
 * Endpoints (contracts/ missing at scaffold time — assumptions documented in
 * README; MUST be reconciled with contracts/openapi.yaml when it lands):
 * - POST {base}/v1/sessions -> {"session_id": "..."}
 * - POST {base}/v1/sessions/{id}/messages {"text": "..."}
 * - GET  {base}/v1/sessions/{id}/events (Accept: text/event-stream,
 *   optional Last-Event-ID header; event types: token / committed / status)
 */
class SseClient(
    streamClient: OkHttpClient? = null,
    private val callClient: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(15, TimeUnit.SECONDS)
        .build(),
) {
    // readTimeout(0) = infinite: the default 10s timeout would kill idle SSE streams.
    private val streamClient: OkHttpClient =
        streamClient ?: callClient.newBuilder()
            .readTimeout(0, TimeUnit.SECONDS)
            .build()

    /**
     * Chat events for one session, with bounded exponential-backoff reconnect.
     *
     * WHY reconnect here: a phone on WiFi loses the coordinator constantly —
     * screen off, AP roam, laptop sleeps. The previous cold Flow died on the
     * first disconnect and the user got a permanently silent stream. Now a
     * dropped connection retries on its own, so a transient outage heals
     * without the user noticing or re-sending anything.
     *
     * Backoff is bounded (see [BACKOFF_INITIAL_MS] / [BACKOFF_MAX_MS]): a
     * coordinator that is genuinely gone must not turn into a hot reconnect loop
     * that drains the battery. Each retry re-sends `Last-Event-ID` for the newest
     * event seen so the server replays from there instead of dropping tokens.
     *
     * Cancellability: this is a plain `flow { }` built on [openEventStream], so
     * cancelling the collecting scope cancels both the delay and the live
     * EventSource — no orphan socket survives the ViewModel.
     */
    fun events(baseUrl: String, sessionId: String, lastEventId: String?): Flow<ChatEvent> =
        flow {
            var resumeFrom = lastEventId
            var attempt = 0
            var failure: IOException? = null
            var closedByServer = false
            while (true) {
                failure = null
                closedByServer = false
                // NOTE: `emit` deliberately sits outside the try below — a
                // cancellation or downstream failure must not be mistaken for a
                // network drop and retried. `catch` only sees upstream errors.
                try {
                    openEventStream(baseUrl, sessionId, resumeFrom)
                        .onEach { evt ->
                            // Real token/commit progress resets the backoff; a
                            // stream that is delivering must not inherit a stale
                            // long delay. A bare status line does not count.
                            if (evt !is ChatEvent.Status) attempt = 0
                            resumeFrom = evt.eventIdOrNull() ?: resumeFrom
                        }
                        .catch { e ->
                            if (e is CancellationException) throw e
                            failure = e as? IOException ?: IOException(e.message, e)
                        }
                        .collect { emit(it) }
                    closedByServer = failure == null
                } catch (e: CancellationException) {
                    throw e
                }
                val cause = failure
                if (cause == null) {
                    emit(
                        ChatEvent.Status(
                            if (closedByServer) {
                                "Stream closed by server — reconnecting…"
                            } else {
                                "Stream ended — reconnecting…"
                            }
                        )
                    )
                    delay(BACKOFF_INITIAL_MS)
                    attempt = 0
                } else {
                    val waitMs = backoffMs(attempt)
                    emit(
                        ChatEvent.Status(
                            "Stream lost (${cause.message ?: cause::class.simpleName}) — " +
                                "retrying in ${waitMs / 1000}s"
                        )
                    )
                    delay(waitMs)
                    attempt += 1
                }
            }
        }

    /**
     * One SSE connection attempt. Closes normally on a server-side close and
     * throws the transport failure otherwise, so [events] owns all retry policy
     * in exactly one place.
     */
    private fun openEventStream(
        baseUrl: String,
        sessionId: String,
        lastEventId: String?,
    ): Flow<ChatEvent> = callbackFlow {
        val url = "${baseUrl.trimEnd('/')}/v1/sessions/$sessionId/events"
        val reqBuilder = Request.Builder()
            .url(url)
            .header("Accept", "text/event-stream")
        if (!lastEventId.isNullOrEmpty()) {
            reqBuilder.header("Last-Event-ID", lastEventId)
        }
        val factory = EventSources.createFactory(streamClient)
        val listener = object : EventSourceListener() {
            override fun onOpen(eventSource: EventSource, response: okhttp3.Response) {
                trySend(ChatEvent.Status("Stream open (${response.code})"))
            }

            override fun onEvent(
                eventSource: EventSource,
                id: String?,
                type: String?,
                data: String,
            ) {
                // Contract shapes (contracts/event-log.md): token {"pos","text"},
                // commit {"pos"}, status = raw JSON. Legacy pre-fix replay wrapped
                // everything as {"kind","payload"} — unwrapped here for rollout.
                val evt: ChatEvent = when (type?.lowercase()) {
                    "token", "message", "delta", null, "" -> ChatEvent.Token(extractText(data), id)
                    "committed", "commit" ->
                        ChatEvent.Committed(extractPos(data), id)
                    else -> ChatEvent.Status("$type: $data", id)
                }
                trySend(evt)
            }

            override fun onClosed(eventSource: EventSource) {
                close()
            }

            override fun onFailure(
                eventSource: EventSource,
                t: Throwable?,
                response: okhttp3.Response?,
            ) {
                close(t ?: IOException("SSE failure (HTTP ${response?.code})"))
            }
        }
        val source = factory.newEventSource(reqBuilder.build(), listener)
        awaitClose { source.cancel() }
    }

    companion object {
        /**
         * First retry delay after a dropped stream. Short, because the most
         * common cause is a brief WiFi blip the user is already waiting on.
         */
        const val BACKOFF_INITIAL_MS = 500L

        /**
         * Ceiling for the exponential backoff. Reached after ~6 retries; a
         * coordinator that is off stays off, and this keeps the retry cost to
         * roughly one probe per 30s instead of a busy loop.
         */
        const val BACKOFF_MAX_MS = 30_000L

        /** Exponential backoff for [attempt] (0-based), clamped to [BACKOFF_MAX_MS]. */
        internal fun backoffMs(attempt: Int): Long {
            val shift = attempt.coerceIn(0, 16)
            val raw = BACKOFF_INITIAL_MS shl shift
            return if (raw <= 0L || raw > BACKOFF_MAX_MS) BACKOFF_MAX_MS else raw
        }

        /** The SSE `id` this event carried, when it had one (used for resume). */
        private fun ChatEvent.eventIdOrNull(): String? = when (this) {
            is ChatEvent.Token -> eventId
            is ChatEvent.Committed -> eventId
            is ChatEvent.Status -> eventId
        }

        /**
         * Parses a `status`-shaped payload for an embedded user turn.
         * The server broadcasts `user_message` rows as `event: status` with
         * `data = {"role":"user","text":"..."}`; returns the text, else null.
         * Used to rebuild cross-device history (phone <-> web) from the
         * events replay — `session_created` and other statuses return null.
         */
        fun extractUserText(data: String): String? {
            val raw = data.trim()
            if (!raw.startsWith("{")) return null
            return try {
                val obj = JSONObject(raw)
                val role = obj.optString("role", "")
                if (!role.equals("user", ignoreCase = true)) return null
                val text = obj.optString("text")
                    .ifBlank { obj.optString("content") }
                    .ifBlank { obj.optString("prompt") }
                text.ifBlank { null }
            } catch (_: Exception) { null }
        }

        /** Extracts `text` from contract `{"pos","text"}`; unwraps legacy `{"kind","payload"}`. */
        fun extractText(data: String): String {
            val raw = data.trim()
            if (!raw.startsWith("{")) return raw
            return try {
                val obj = JSONObject(raw)
                if (obj.has("text")) return obj.optString("text", "")
                // Legacy: {"kind":"token","payload":"{\"pos\":0,\"text\":\"hi\"}"}
                if (obj.has("payload")) {
                    val inner = obj.optString("payload", "")
                    if (inner.trim().startsWith("{")) {
                        JSONObject(inner).optString("text", inner)
                    } else inner
                } else raw
            } catch (_: Exception) { raw }
        }

        /** Extracts `pos` from contract `{"pos"}`; falls back to raw int + legacy wrapper. */
        fun extractPos(data: String): Int {
            data.trim().toIntOrNull()?.let { return it }
            return try {
                val obj = JSONObject(data)
                if (obj.has("pos")) obj.optInt("pos", 0)
                else if (obj.has("payload")) extractPos(obj.optString("payload", "0"))
                else 0
            } catch (_: Exception) { 0 }
        }
    }

    /** POSTs a chat message; returns the raw response body (session echo). */
    suspend fun postMessage(baseUrl: String, sessionId: String, text: String): String =
        withContext(Dispatchers.IO) {
            val url = "${baseUrl.trimEnd('/')}/v1/sessions/$sessionId/messages"
            val payload = JSONObject().put("text", text).toString()
            val req = Request.Builder()
                .url(url)
                .post(payload.toRequestBody("application/json".toMediaType()))
                .build()
            callClient.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) throw IOException("POST $url -> HTTP ${resp.code}")
                resp.body?.string() ?: ""
            }
        }

    /** Creates a session; returns the new session id. */
    suspend fun createSession(baseUrl: String): String =
        withContext(Dispatchers.IO) {
            val url = "${baseUrl.trimEnd('/')}/v1/sessions"
            val req = Request.Builder()
                .url(url)
                .post("{}".toRequestBody("application/json".toMediaType()))
                .build()
            callClient.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) throw IOException("POST $url -> HTTP ${resp.code}")
                val text = resp.body?.string().orEmpty()
                try {
                    val obj = JSONObject(text)
                    obj.optString("session_id")
                        .ifBlank { obj.optString("id") }
                        .ifBlank { text.trim().trim('"') }
                } catch (_: Exception) {
                    text.trim()
                }.ifBlank { throw IOException("Empty session id from $url") }
            }
        }

    /** GET {base}/api/health — returns HTTP status; throws on transport error. */
    suspend fun getHealth(baseUrl: String): Int =
        withContext(Dispatchers.IO) {
            val url = "${baseUrl.trimEnd('/')}/api/health"
            val req = Request.Builder().url(url).get().build()
            callClient.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) throw IOException("GET $url -> HTTP ${resp.code}")
                resp.code
            }
        }

    /**
     * GET {base}/api/node -> [NodeInfo]. Throws on transport error or HTTP
     * error (callers treat 404 as "server too old, node info unavailable").
     */
    suspend fun getNodeInfo(baseUrl: String): NodeInfo =
        withContext(Dispatchers.IO) {
            val url = "${baseUrl.trimEnd('/')}/api/node"
            val req = Request.Builder().url(url).get().build()
            val text = callClient.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) throw IOException("GET $url -> HTTP ${resp.code}")
                resp.body?.string().orEmpty()
            }
            val obj = JSONObject(text)
            NodeInfo(
                nodeId = obj.optString("node_id"),
                fingerprint = obj.optString("fingerprint"),
                quicPort = obj.optInt("quic_port", -1).takeIf { it >= 0 },
                version = obj.optString("version"),
            )
        }
}
