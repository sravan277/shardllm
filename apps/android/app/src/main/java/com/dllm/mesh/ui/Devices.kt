package com.dllm.mesh.ui

import android.app.Application
import android.content.pm.PackageManager
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
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.dllm.mesh.data.IdentityStore
import com.dllm.mesh.net.NsdDiscovery
import com.dllm.mesh.net.Peer
import com.dllm.mesh.net.QrScanScreen
import com.dllm.mesh.net.QrShowScreen
import com.dllm.mesh.net.buildJoinPayload
import com.dllm.mesh.net.parseJoinPayload
import com.dllm.mesh.worker.WorkerService
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

    private val app = application
    private val store = IdentityStore(application)
    private val discovery = NsdDiscovery(application)
    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    val coordinatorUrl: StateFlow<String> = store.coordinatorUrl
        .stateIn(viewModelScope, SharingStarted.Eagerly, IdentityStore.DEFAULT_COORDINATOR_URL)
    val workerEnabled: StateFlow<Boolean> = store.workerEnabled
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val peers: StateFlow<List<Peer>> = discovery.peers

    private val _devices = MutableStateFlow<List<DeviceEntry>>(emptyList())
    val devices: StateFlow<List<DeviceEntry>> = _devices.asStateFlow()

    private val _status = MutableStateFlow("Idle.")
    val status: StateFlow<String> = _status.asStateFlow()

    // Placeholders until Phase 4 worker reports real values.
    private val _assignedShard = MutableStateFlow("— (Phase 4)")
    val assignedShard: StateFlow<String> = _assignedShard.asStateFlow()
    private val _ramLine = MutableStateFlow("RAM/KV: — (Phase 4)")
    val ramLine: StateFlow<String> = _ramLine.asStateFlow()

    init {
        viewModelScope.launch { store.ensureNodeId() }
    }

    fun setCoordinatorUrl(url: String) {
        if (url.isBlank()) return
        viewModelScope.launch { store.setCoordinatorUrl(url) }
    }

    fun setWorkerEnabled(enabled: Boolean) {
        viewModelScope.launch {
            store.setWorkerEnabled(enabled)
            if (enabled) WorkerService.start(app) else WorkerService.stop(app)
            _status.value = if (enabled) "Worker on — idle stub (no compute until Phase 4)."
            else "Worker off."
        }
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

    fun applyScannedJoin(raw: String) {
        val parsed = parseJoinPayload(raw)
        if (parsed == null) {
            _status.value = "QR unreadable — expected join JSON with url."
            return
        }
        viewModelScope.launch {
            store.setCoordinatorUrl(parsed.url)
            _status.value = "Coordinator set to ${parsed.url} — approve on the coordinator."
        }
    }

    suspend fun joinPayload(): String {
        val base = store.coordinatorUrl.first()
        val node = store.ensureNodeId()
        return buildJoinPayload(base, node)
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
    val context = LocalContext.current
    val coordinatorUrl by viewModel.coordinatorUrl.collectAsState()
    val workerEnabled by viewModel.workerEnabled.collectAsState()
    val peers by viewModel.peers.collectAsState()
    val devices by viewModel.devices.collectAsState()
    val status by viewModel.status.collectAsState()
    val shard by viewModel.assignedShard.collectAsState()
    val ramLine by viewModel.ramLine.collectAsState()

    var urlField by remember(coordinatorUrl) { mutableStateOf(coordinatorUrl) }
    var showQr by remember { mutableStateOf(false) }
    var showScan by remember { mutableStateOf(false) }
    var joinText by remember { mutableStateOf("") }
    val cameraGranted = remember(showScan) {
        ContextCompat.checkSelfPermission(context, android.Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
    }

    Column(modifier = modifier.fillMaxSize().padding(16.dp)) {
        Text("Coordinator", color = Color(0xFFE8EDF2))
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = urlField,
                onValueChange = { urlField = it },
                label = { Text("Coordinator address") },
                placeholder = { Text("http://192.168.1.10:8080") },
                modifier = Modifier.weight(1f),
                singleLine = true,
            )
            Button(onClick = { viewModel.setCoordinatorUrl(urlField) }) { Text("Save") }
        }
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = viewModel::startDiscovery) { Text("Discover") }
            OutlinedButton(onClick = viewModel::stopDiscovery) { Text("Stop") }
            OutlinedButton(onClick = viewModel::refreshDevices) { Text("Refresh") }
        }
        Spacer(Modifier.height(8.dp))

        Text("Nearby (${peers.size})", color = Color(0xFFE8EDF2))
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.height(140.dp)) {
            items(peers, key = { it.serviceName }) { peer ->
                Text(
                    "${peer.serviceName} — ${peer.host}:${peer.port}",
                    color = Color(0xFF93A1B0),
                )
            }
        }

        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { showScan = true }) { Text("Pair via QR") }
            OutlinedButton(onClick = { showQr = true }) { Text("Show join code") }
        }

        Spacer(Modifier.height(8.dp))
        Text("Devices (${devices.size})", color = Color(0xFFE8EDF2))
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.height(140.dp)) {
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

        Spacer(Modifier.height(12.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Worker role", color = Color(0xFFE8EDF2))
            Switch(checked = workerEnabled, onCheckedChange = viewModel::setWorkerEnabled)
        }
        Text("Assigned shard: $shard", color = Color(0xFF93A1B0))
        Text(ramLine, color = Color(0xFF93A1B0))
        Spacer(Modifier.height(8.dp))
        Text(status, color = Color(0xFFF5B544))
    }

    if (showQr) {
        Dialog(onDismissRequest = { showQr = false; joinText = "" }) {
            Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF101418))) {
                if (joinText.isEmpty()) {
                    Text("Loading…", Modifier.padding(16.dp), color = Color(0xFFE8EDF2))
                } else {
                    QrShowScreen(payload = joinText)
                }
                TextButton(
                    onClick = { showQr = false; joinText = "" },
                    modifier = Modifier.padding(8.dp),
                ) {
                    Text("Close")
                }
            }
        }
    }
    if (showScan) {
        Dialog(onDismissRequest = { showScan = false }) {
            Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF101418))) {
                if (cameraGranted) {
                    QrScanScreen(
                        onScanned = {
                            viewModel.applyScannedJoin(it)
                            showScan = false
                        },
                        onCancel = { showScan = false },
                    )
                } else {
                    Column(Modifier.padding(16.dp)) {
                        Text(
                            "Camera permission not granted — grant it in Settings, or paste the join text on the scan screen after granting.",
                            color = Color(0xFFE8EDF2),
                        )
                        TextButton(onClick = { showScan = false }) { Text("Close") }
                    }
                }
            }
        }
    }

    // Load the join payload when the dialog opens.
    if (showQr && joinText.isEmpty()) {
        androidx.compose.runtime.LaunchedEffect(showQr) {
            joinText = viewModel.joinPayload()
        }
    }
}
