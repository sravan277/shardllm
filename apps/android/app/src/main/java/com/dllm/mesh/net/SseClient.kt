package com.dllm.mesh.net

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
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

    fun events(baseUrl: String, sessionId: String, lastEventId: String?): Flow<ChatEvent> =
        callbackFlow {
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
                    val evt: ChatEvent = when (type?.lowercase()) {
                        "token", "message", "delta", null, "" -> ChatEvent.Token(data, id)
                        "committed", "commit" ->
                            ChatEvent.Committed(data.trim().toIntOrNull() ?: 0, id)
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
}
