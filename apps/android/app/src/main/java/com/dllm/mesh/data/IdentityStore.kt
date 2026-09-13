package com.dllm.mesh.data

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import java.util.UUID

private val Context.meshDataStore by preferencesDataStore(name = "dllm_mesh")

/**
 * Single DataStore owner for Phase 0 identity + session state.
 *
 * Keys: node_id, coordinator_url, session_id, last_event_id, worker_enabled,
 * server_fingerprint, server_quic_port, server_node_id.
 *
 * Private-key note: the signing key itself lives in AndroidKeystore under
 * [KEYSTORE_ALIAS] (TODO Phase 3 pairing crypto). Only the alias name is
 * referenced here — never persist key bytes in DataStore.
 */
class IdentityStore(private val context: Context) {

    companion object {
        private val NODE_ID = stringPreferencesKey("node_id")
        private val COORDINATOR_URL = stringPreferencesKey("coordinator_url")
        private val SESSION_ID = stringPreferencesKey("session_id")
        private val LAST_EVENT_ID = stringPreferencesKey("last_event_id")
        private val WORKER_ENABLED = booleanPreferencesKey("worker_enabled")
        private val SERVER_FINGERPRINT = stringPreferencesKey("server_fingerprint")
        private val SERVER_QUIC_PORT = intPreferencesKey("server_quic_port")
        private val SERVER_NODE_ID = stringPreferencesKey("server_node_id")

        const val KEYSTORE_ALIAS = "dllm_mesh_signing"
        const val DEFAULT_COORDINATOR_URL = "http://192.168.1.10:8080"
    }

    val nodeId: Flow<String> =
        context.meshDataStore.data.map { it[NODE_ID] ?: "" }

    val coordinatorUrl: Flow<String> =
        context.meshDataStore.data.map { it[COORDINATOR_URL] ?: DEFAULT_COORDINATOR_URL }

    val sessionId: Flow<String> =
        context.meshDataStore.data.map { it[SESSION_ID] ?: "" }

    val lastEventId: Flow<String> =
        context.meshDataStore.data.map { it[LAST_EVENT_ID] ?: "" }

    val workerEnabled: Flow<Boolean> =
        context.meshDataStore.data.map { it[WORKER_ENABLED] ?: false }

    /** TOFU-pinned server fingerprint from pairing (empty = not yet paired). */
    val serverFingerprint: Flow<String> =
        context.meshDataStore.data.map { it[SERVER_FINGERPRINT] ?: "" }

    /** QUIC port advertised by the paired server (0 = unknown). */
    val serverQuicPort: Flow<Int> =
        context.meshDataStore.data.map { it[SERVER_QUIC_PORT] ?: 0 }

    /** node_id last reported by GET /api/node (empty = unknown). */
    val serverNodeId: Flow<String> =
        context.meshDataStore.data.map { it[SERVER_NODE_ID] ?: "" }

    suspend fun ensureNodeId(): String {
        val current = nodeId.first()
        if (current.isNotBlank()) return current
        val fresh = UUID.randomUUID().toString()
        context.meshDataStore.edit { it[NODE_ID] = fresh }
        return fresh
    }

    suspend fun setCoordinatorUrl(url: String) {
        context.meshDataStore.edit { it[COORDINATOR_URL] = url.trim() }
    }

    suspend fun setSessionId(id: String) {
        context.meshDataStore.edit { it[SESSION_ID] = id.trim() }
    }

    suspend fun setLastEventId(id: String) {
        context.meshDataStore.edit { it[LAST_EVENT_ID] = id }
    }

    suspend fun setWorkerEnabled(enabled: Boolean) {
        context.meshDataStore.edit { it[WORKER_ENABLED] = enabled }
    }

    suspend fun setServerFingerprint(fingerprint: String) {
        context.meshDataStore.edit { it[SERVER_FINGERPRINT] = fingerprint.trim() }
    }

    suspend fun setServerQuicPort(port: Int) {
        context.meshDataStore.edit { it[SERVER_QUIC_PORT] = port }
    }

    suspend fun setServerNodeId(nodeId: String) {
        context.meshDataStore.edit { it[SERVER_NODE_ID] = nodeId.trim() }
    }

    /** Persist a full pairing result: active server URL + TOFU pin data. */
    suspend fun savePairing(baseUrl: String, fingerprint: String, quicPort: Int?) {
        context.meshDataStore.edit {
            it[COORDINATOR_URL] = baseUrl.trim()
            it[SERVER_FINGERPRINT] = fingerprint.trim()
            it[SERVER_QUIC_PORT] = quicPort ?: 0
        }
    }
}
