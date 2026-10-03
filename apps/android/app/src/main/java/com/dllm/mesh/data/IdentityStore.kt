package com.dllm.mesh.data

import android.content.Context
import android.os.Build
import android.provider.Settings
import android.util.Log
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import org.json.JSONArray
import org.json.JSONObject
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
 *
 * Identity rule (duplicate-device fix): scan / manual / nearby pairing flows
 * must NEVER write NODE_ID — they only ensure it exists (via [ensureNodeId])
 * or adopt a user-supplied one (via [setNodeId]). The only writers of NODE_ID
 * are [ensureNodeId] (first run) and [setNodeId] ("Use existing ID").
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
        private const val TAG = "IdentityStore"

        /** Display name sent as `device_name` — non-unique on purpose (many RMX3785s). */
        fun stableDeviceName(): String = Build.MODEL

        /**
         * Single heartbeat-body builder shared by every sender
         * (Devices presence, WorkerService, Pairing.connect).
         * Keeps device_id / device_name / role / permissions from diverging again.
         *
         * RMX3785 load fix: always emits `load:{cpu_pct,mem_pct}` (0-100,
         * JSON null when unknown — never synthesized). When this phone works as
         * a compute worker, callers also pass `capabilities` + `worker_active=true`
         * + `layers` so the backend `/v1/plan` can assign layers to it. Unknown
         * fields are ignored by older coordinators (serde defaults), so this stays
         * compatible.
         *
         * There is deliberately no `group_id`: the mesh is a single network, so a
         * device's membership is expressed by the coordinator's registry row, not
         * by a group id the phone carries.
         */
        fun buildHeartbeatBody(
            deviceId: String,
            deviceName: String = stableDeviceName(),
            role: String = "client",
            cpuPct: Double? = null,
            memPct: Double? = null,
            capabilities: JSONObject? = null,
            workerActive: Boolean? = null,
            layers: List<Int>? = null,
        ): String {
            val body = JSONObject()
                .put("device_id", deviceId)
                .put("role", role)
                .put("permissions", JSONArray(listOf("chat")))
                .put("device_name", deviceName)
                .put(
                    "load",
                    JSONObject()
                        .put("cpu_pct", cpuPct ?: JSONObject.NULL)
                        .put("mem_pct", memPct ?: JSONObject.NULL),
                )
            if (capabilities != null) body.put("capabilities", capabilities)
            if (workerActive != null) body.put("worker_active", workerActive)
            if (layers != null) body.put("layers", JSONArray(layers))
            return body.toString()
        }
    }

    val nodeId: Flow<String> =
        context.meshDataStore.data.map { it[NODE_ID] ?: "" }

    val coordinatorUrl: Flow<String> =
        context.meshDataStore.data.map { it[COORDINATOR_URL] ?: DEFAULT_COORDINATOR_URL }

    /**
     * True only when a coordinator URL was actually persisted by a pairing flow.
     *
     * WHY this is separate from [coordinatorUrl]: that flow substitutes
     * [DEFAULT_COORDINATOR_URL] on a fresh install, so reading it cannot tell
     * "paired with 192.168.1.10" from "never paired, showing the placeholder".
     * Auto-connect on launch needs that distinction, otherwise every first launch
     * would fire a heartbeat at a placeholder IP and report a scary failure.
     */
    val hasCoordinator: Flow<Boolean> =
        context.meshDataStore.data.map { !it[COORDINATOR_URL].isNullOrBlank() }

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
        // DataStore empty (fresh install or clear-data): prefer a deterministic
        // ID derived from ANDROID_ID so a reinstall reuses the same device_id
        // instead of minting a duplicate row. ANDROID_ID needs no permission,
        // works offline, and is stable per app-signing-key + device (reset only
        // on factory reset). Falls back to random UUID when unavailable.
        val stable = stableIdFromAndroidId()
        val fresh = stable ?: UUID.randomUUID().toString()
        context.meshDataStore.edit { it[NODE_ID] = fresh }
        Log.i(TAG, "node_id minted (stable=${stable != null}, id=$fresh)")
        if (stable == null) {
            Log.w(TAG, "ANDROID_ID unavailable — minted random node_id; a reinstall may duplicate the coordinator row.")
        }
        return fresh
    }

    /**
     * "Use existing ID": adopt a previously issued device_id (e.g. the stale
     * `RMX3785-realme` row) so the next heartbeat reuses that row instead of
     * the current UUID. Accepts legacy human-readable IDs as-is. Returns false
     * when the input is blank (nothing written).
     */
    suspend fun setNodeId(raw: String): Boolean {
        val id = raw.trim()
        if (id.isBlank()) return false
        context.meshDataStore.edit { it[NODE_ID] = id }
        Log.i(TAG, "node_id adopted existing id=$id")
        return true
    }

    /** Deterministic UUID from ANDROID_ID, or null when it cannot be read. */
    private fun stableIdFromAndroidId(): String? = runCatching {
        val androidId = Settings.Secure
            .getString(context.contentResolver, Settings.Secure.ANDROID_ID)
            ?.trim().orEmpty()
        // 9774d56d682e549c is the well-known broken ANDROID_ID on old emulators.
        if (androidId.isBlank() || androidId == "9774d56d682e549c") return@runCatching null
        UUID.nameUUIDFromBytes(("dllm-mesh:$androidId").toByteArray(Charsets.UTF_8)).toString()
    }.getOrNull()

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
