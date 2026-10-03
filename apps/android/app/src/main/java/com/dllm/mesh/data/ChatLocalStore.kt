package com.dllm.mesh.data

import android.content.Context
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.first
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

private val Context.chatDataStore by preferencesDataStore(name = "dllm_chats")

/** One locally stored chat message (mirrors ui ChatMessage without the dependency). */
data class LocalChatMsg(val role: String, val text: String)

/** One locally stored chat row (mirrors net ChatSession without the dependency). */
data class LocalChatSession(
    val id: String,
    val title: String,
    val model: String = "unknown",
    val tokensOut: Int = 0,
    val lastTokenAt: String? = null,
    val createdAt: String? = null,
    val updatedAt: Long = 0L,
)

/**
 * Local-only chat storage (DataStore file `dllm_chats`, keys prefixed per
 * device node_id). Replaces the old `GET /v1/sessions` server sync: the
 * burger drawer lists these rows, rename/delete apply here first and hit
 * the server best-effort. Message snapshots make chats readable offline;
 * the live SSE replay from 0 rebuilds turns when online.
 */
class ChatLocalStore(private val context: Context) {

    private fun indexKey(nodeId: String) = stringPreferencesKey("local_chats_index_$nodeId")

    private fun msgsKey(nodeId: String, sessionId: String) =
        stringPreferencesKey("local_chat_msgs_${nodeId}_$sessionId")

    suspend fun listSessions(nodeId: String): List<LocalChatSession> {
        if (nodeId.isBlank()) return emptyList()
        val raw = context.chatDataStore.data.first()[indexKey(nodeId)].orEmpty()
        if (raw.isBlank()) return emptyList()
        val arr = runCatching { JSONArray(raw) }.getOrNull() ?: return emptyList()
        val out = ArrayList<LocalChatSession>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            val id = o.optString("id")
            if (id.isBlank()) continue
            out.add(
                LocalChatSession(
                    id = id,
                    title = o.optString("title").trim().ifBlank { "New chat" },
                    model = o.optString("model").trim().ifBlank { "unknown" },
                    tokensOut = o.optInt("tokens_out", 0).coerceAtLeast(0),
                    lastTokenAt = o.optString("last_token_at").trim().ifBlank { null },
                    createdAt = o.optString("created_at").trim().ifBlank { null },
                    updatedAt = o.optLong("updated_at", 0L),
                )
            )
        }
        // Newest-first (matches the old website order).
        return out.sortedByDescending { it.updatedAt }
    }

    suspend fun upsertSession(nodeId: String, session: LocalChatSession) {
        if (nodeId.isBlank() || session.id.isBlank()) return
        val current = listSessions(nodeId).toMutableList()
        val idx = current.indexOfFirst { it.id == session.id }
        val row = session.copy(
            updatedAt = System.currentTimeMillis(),
            createdAt = session.createdAt ?: current.firstOrNull { it.id == session.id }?.createdAt,
        )
        if (idx >= 0) current[idx] = row else current.add(0, row)
        saveIndex(nodeId, current)
    }

    suspend fun renameSession(nodeId: String, id: String, title: String) {
        if (nodeId.isBlank() || id.isBlank()) return
        val current = listSessions(nodeId)
        val row = current.firstOrNull { it.id == id } ?: return
        upsertSession(nodeId, row.copy(title = title))
    }

    suspend fun deleteSession(nodeId: String, id: String) {
        if (nodeId.isBlank() || id.isBlank()) return
        saveIndex(nodeId, listSessions(nodeId).filterNot { it.id == id })
        context.chatDataStore.updateData { prefs ->
            val mutable = prefs.toMutablePreferences()
            mutable.remove(msgsKey(nodeId, id))
            mutable
        }
    }

    /** Migrate a local-only row to its server id after the first send. */
    suspend fun migrateSession(nodeId: String, fromId: String, toId: String) {
        if (nodeId.isBlank() || fromId == toId) return
        val current = listSessions(nodeId)
        val row = current.firstOrNull { it.id == fromId } ?: return
        val msgs = loadMessages(nodeId, fromId)
        saveIndex(nodeId, current.filterNot { it.id == fromId } + row.copy(id = toId))
        saveMessages(nodeId, toId, msgs)
        context.chatDataStore.updateData { prefs ->
            val mutable = prefs.toMutablePreferences()
            mutable.remove(msgsKey(nodeId, fromId))
            mutable
        }
    }

    suspend fun loadMessages(nodeId: String, sessionId: String): List<LocalChatMsg> {
        if (nodeId.isBlank() || sessionId.isBlank()) return emptyList()
        val raw = context.chatDataStore.data.first()[msgsKey(nodeId, sessionId)].orEmpty()
        if (raw.isBlank()) return emptyList()
        val arr = runCatching { JSONArray(raw) }.getOrNull() ?: return emptyList()
        val out = ArrayList<LocalChatMsg>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            val text = o.optString("text")
            if (text.isEmpty()) continue
            out.add(LocalChatMsg(role = o.optString("role").ifBlank { "assistant" }, text = text))
        }
        return out
    }

    suspend fun saveMessages(nodeId: String, sessionId: String, msgs: List<LocalChatMsg>) {
        if (nodeId.isBlank() || sessionId.isBlank()) return
        // Cap the snapshot so one long chat cannot blow the DataStore file.
        val tail = msgs.takeLast(200)
        val arr = JSONArray()
        for (m in tail) {
            arr.put(JSONObject().put("role", m.role).put("text", m.text))
        }
        context.chatDataStore.updateData { prefs ->
            val mutable = prefs.toMutablePreferences()
            mutable[msgsKey(nodeId, sessionId)] = arr.toString()
            mutable
        }
    }

    private suspend fun saveIndex(nodeId: String, sessions: List<LocalChatSession>) {
        val arr = JSONArray()
        for (s in sessions.take(200)) {
            arr.put(
                JSONObject()
                    .put("id", s.id)
                    .put("title", s.title)
                    .put("model", s.model)
                    .put("tokens_out", s.tokensOut)
                    .put("last_token_at", s.lastTokenAt ?: JSONObject.NULL)
                    .put("created_at", s.createdAt ?: JSONObject.NULL)
                    .put("updated_at", s.updatedAt),
            )
        }
        context.chatDataStore.updateData { prefs ->
            val mutable = prefs.toMutablePreferences()
            mutable[indexKey(nodeId)] = arr.toString()
            mutable
        }
    }

    companion object {
        /** Fresh local-only id for offline-created chats (migrated on send). */
        fun newLocalId(): String = "local-${UUID.randomUUID()}"
    }
}
