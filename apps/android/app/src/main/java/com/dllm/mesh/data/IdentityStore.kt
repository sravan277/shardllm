package com.dllm.mesh.data

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
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
 * Keys: node_id, coordinator_url, session_id, last_event_id, worker_enabled.
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
}
