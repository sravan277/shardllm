package com.dllm.mesh.ui

import android.app.Application
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.dllm.mesh.data.IdentityStore
import com.dllm.mesh.net.NsdDiscovery
import com.dllm.mesh.net.Peer
import com.dllm.mesh.net.baseUrlMatches
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

data class DeviceEntry(val id: String, val name: String, val approved: Boolean)

class DevicesViewModel(application: Application) : AndroidViewModel(application) {

    private val store = IdentityStore(application)
    private val discovery = NsdDiscovery(application)
    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    val coordinatorUrl: StateFlow<String> = store.coordinatorUrl
        .stateIn(viewModelScope, SharingStarted.Eagerly, IdentityStore.DEFAULT_COORDINATOR_URL)
    val peers: StateFlow<List<Peer>> = discovery.peers

    private val _devices = MutableStateFlow<List<DeviceEntry>>(emptyList())
    val devices: StateFlow<List<DeviceEntry>> = _devices.asStateFlow()

    private val _status = MutableStateFlow("Idle.")
    val status: StateFlow<String> = _status.asStateFlow()

    init {
        viewModelScope.launch { store.ensureNodeId() }
    }

    fun startDiscovery() {
        discovery.startDiscovery()
        _status.value = "Discovering _dllm._tcp. …"
    }

    fun stopDiscovery() {
        discovery.stopDiscovery()
    }

    override fun onCleared() {
        discovery.stopDiscovery()
        discovery.unregisterService()
    }

    /** Attach = save the peer as the active server. */
    fun attachPeer(peer: Peer) {
        viewModelScope.launch {
            val base = "http://${peer.host}:${peer.port}"
            store.setCoordinatorUrl(base)
            _status.value = "Attached to $base (${peer.serviceName})."
        }
    }

    fun refreshDevices() {
        viewModelScope.launch {
            _status.value = "Loading devices…"
            runCatching {
                val base = store.coordinatorUrl.first().trimEnd('/')
                val req = Request.Builder().url("$base/v1/devices").get().build()
                val text = withContext(Dispatchers.IO) {
                    http.newCall(req).execute().use { resp ->
                        if (!resp.isSuccessful) error("HTTP ${resp.code}")
                        resp.body?.string().orEmpty()
                    }
                }
                parseDevices(text)
            }.onSuccess {
                _devices.value = it
                _status.value = "Devices refreshed (${it.size})."
            }.onFailure { e ->
                _status.value = "Refresh failed: ${e.message}"
            }
        }
    }

    fun approveDevice(id: String) {
        viewModelScope.launch {
            runCatching {
                val base = store.coordinatorUrl.first().trimEnd('/')
                val req = Request.Builder()
                    .url("$base/v1/devices/$id/approve")
                    .post(ByteArray(0).toRequestBody(null))
                    .build()
                withContext(Dispatchers.IO) {
                    http.newCall(req).execute().use {
                        if (!it.isSuccessful) error("HTTP ${it.code}")
                    }
                }
            }.onSuccess {
                _status.value = "Approved $id."
                refreshDevices()
            }.onFailure { e -> _status.value = "Approve failed: ${e.message}" }
        }
    }

    fun revokeDevice(id: String) {
        viewModelScope.launch {
            runCatching {
                val base = store.coordinatorUrl.first().trimEnd('/')
                val req = Request.Builder().url("$base/v1/devices/$id").delete().build()
                withContext(Dispatchers.IO) {
                    http.newCall(req).execute().use {
                        if (!it.isSuccessful) error("HTTP ${it.code}")
                    }
                }
            }.onSuccess {
                _status.value = "Revoked $id."
                refreshDevices()
            }.onFailure { e -> _status.value = "Revoke failed: ${e.message}" }
        }
    }

    private fun parseDevices(text: String): List<DeviceEntry> {
        val arr: JSONArray = runCatching {
            val obj = JSONObject(text)
            obj.optJSONArray("devices") ?: JSONArray(text)
        }.getOrNull() ?: JSONArray()
        return List(arr.length()) { i ->
            val o = arr.getJSONObject(i)
            DeviceEntry(
                id = o.optString("id").ifBlank { o.optString("node_id", "device-$i") },
                name = o.optString("name").ifBlank { o.optString("id", "device-$i") },
                approved = o.optBoolean("approved", true),
            )
        }
    }
}

@Composable
fun DevicesScreen(
    viewModel: DevicesViewModel = viewModel(),
    modifier: Modifier = Modifier,
) {
    val coordinatorUrl by viewModel.coordinatorUrl.collectAsState()
    val peers by viewModel.peers.collectAsState()
    val devices by viewModel.devices.collectAsState()
    val status by viewModel.status.collectAsState()

    Column(modifier = modifier.fillMaxSize().padding(16.dp)) {
        Text("Active server", color = Color(0xFFE8EDF2))
        Spacer(Modifier.height(4.dp))
        Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF1B2B28))) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(12.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    coordinatorUrl,
                    color = Color(0xFF2DD4BF),
                    modifier = Modifier.weight(1f),
                )
                Text("Active", color = Color(0xFF93A1B0))
            }
        }
        Spacer(Modifier.height(4.dp))
        Text("Change in Settings or Pairing.", color = Color(0xFF93A1B0))
        Spacer(Modifier.height(8.dp))

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = viewModel::startDiscovery) { Text("Discover") }
            OutlinedButton(onClick = viewModel::stopDiscovery) { Text("Stop") }
            OutlinedButton(onClick = viewModel::refreshDevices) { Text("Refresh") }
        }
        Spacer(Modifier.height(8.dp))

        Text("Nearby (${peers.size})", color = Color(0xFFE8EDF2))
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.height(160.dp)) {
            items(peers, key = { it.serviceName }) { peer ->
                Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF171D24))) {
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(12.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(peer.serviceName, color = Color(0xFFE8EDF2))
                            Text(
                                "${peer.host}:${peer.port}",
                                color = Color(0xFF93A1B0),
                            )
                        }
                        if (baseUrlMatches(coordinatorUrl, peer.host, peer.port)) {
                            Text("Active", color = Color(0xFF2DD4BF))
                        } else {
                            TextButton(onClick = { viewModel.attachPeer(peer) }) { Text("Attach") }
                        }
                    }
                }
            }
        }

        Spacer(Modifier.height(8.dp))
        Text("Devices (${devices.size})", color = Color(0xFFE8EDF2))
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.height(160.dp)) {
            items(devices, key = { it.id }) { device ->
                Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF171D24))) {
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(12.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text(device.name, color = Color(0xFFE8EDF2))
                            Text(device.id, color = Color(0xFF93A1B0))
                        }
                        if (device.approved) {
                            TextButton(onClick = { viewModel.revokeDevice(device.id) }) { Text("Revoke") }
                        } else {
                            TextButton(onClick = { viewModel.approveDevice(device.id) }) { Text("Approve") }
                        }
                    }
                }
            }
        }

        Spacer(Modifier.height(8.dp))
        Text(status, color = Color(0xFFF5B544))
    }
}
