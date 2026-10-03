package com.dllm.mesh.ui

import android.app.Application
import android.content.pm.PackageManager
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
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
import com.dllm.mesh.net.DllmApi
import com.dllm.mesh.net.MeshNetwork
import com.dllm.mesh.net.NetJoin
import com.dllm.mesh.net.NetworkDevice
import com.dllm.mesh.net.NsdDiscovery
import com.dllm.mesh.net.Peer
import com.dllm.mesh.net.Presence
import com.dllm.mesh.net.QrScanScreen
import com.dllm.mesh.net.cleanHost
import com.dllm.mesh.net.hostWarning
import com.dllm.mesh.net.parseJoinPayload
import com.dllm.mesh.net.parseNetUri
import com.dllm.mesh.net.parsePairUri
import com.dllm.mesh.worker.WorkerService
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlin.coroutines.cancellation.CancellationException

private val Teal = Color(0xFF2DD4BF)
private val Muted = Color(0xFF93A1B0)
private val Ink = Color(0xFFE8EDF2)
private val Amber = Color(0xFFF5B544)

class NetworksViewModel(application: Application) : AndroidViewModel(application) {

    private val app = application
    private val store = IdentityStore(application)
    private val discovery = NsdDiscovery(application)

    val coordinatorUrl: StateFlow<String> = store.coordinatorUrl
        .stateIn(viewModelScope, SharingStarted.Eagerly, IdentityStore.DEFAULT_COORDINATOR_URL)
    val workerEnabled: StateFlow<Boolean> = store.workerEnabled
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)
    val peers: StateFlow<List<Peer>> = discovery.peers

    private val _networks = MutableStateFlow<List<MeshNetwork>>(emptyList())
    val networks: StateFlow<List<MeshNetwork>> = _networks.asStateFlow()

    private val _selectedId = MutableStateFlow("")
    val selectedId: StateFlow<String> = _selectedId.asStateFlow()

    private val _detail = MutableStateFlow<List<NetworkDevice>>(emptyList())
    val detail: StateFlow<List<NetworkDevice>> = _detail.asStateFlow()

    private val _detailLoading = MutableStateFlow(false)
    val detailLoading: StateFlow<Boolean> = _detailLoading.asStateFlow()

    private val _pendingNet = MutableStateFlow<NetJoin?>(null)
    val pendingNet: StateFlow<NetJoin?> = _pendingNet.asStateFlow()

    private val _status = MutableStateFlow("")
    val status: StateFlow<String> = _status.asStateFlow()

    private var loadJob: Job? = null

    init {
        viewModelScope.launch { store.ensureNodeId() }
        viewModelScope.launch {
            store.coordinatorUrl.collect { refreshNetworks() }
        }
    }

    override fun onCleared() {
        discovery.stopDiscovery()
        discovery.unregisterService()
    }

    fun startDiscovery() = discovery.startDiscovery()

    fun stopDiscovery() = discovery.stopDiscovery()

    fun setWorkerEnabled(enabled: Boolean) {
        viewModelScope.launch {
            store.setWorkerEnabled(enabled)
            if (enabled) WorkerService.start(app) else WorkerService.stop(app)
            _status.value = if (enabled) "Worker on — heartbeat role=worker with load + capabilities."
            else "Worker off — heartbeat role=client."
        }
    }

    fun refreshNetworks() {
        loadJob?.cancel()
        loadJob = viewModelScope.launch {
            runCatching {
                DllmApi.listNetworks(store.coordinatorUrl.first())
            }.onSuccess { list ->
                _networks.value = list
                if (list.isEmpty()) {
                    _status.value = "No groups on this coordinator yet — create one below."
                } else if (_status.value.isBlank()) {
                    _status.value = ""
                }
                // Keep the detail fresh when its group still exists.
                val sel = _selectedId.value
                if (sel.isNotBlank()) {
                    if (list.none { it.id == sel }) {
                        _selectedId.value = ""
                        _detail.value = emptyList()
                    } else {
                        refreshDetail(sel)
                    }
                }
            }.onFailure { e ->
                if (e is CancellationException) throw e
                _networks.value = emptyList()
                _status.value = "Groups unavailable: ${Presence.friendlyCause(e)} — " +
                    "check the coordinator, then Refresh."
            }
        }
    }

    fun createNetwork(name: String, password: String, openJoin: Boolean) {
        val clean = name.trim()
        if (clean.isEmpty() || clean.length > 40) {
            _status.value = "Group name must be 1–40 chars — not created."
            return
        }
        viewModelScope.launch {
            runCatching {
                val base = store.coordinatorUrl.first()
                DllmApi.createNetwork(base, clean, password.ifBlank { null }, openJoin)
            }.onSuccess { created ->
                _status.value = "Group \"${created.name}\" created."
                refreshNetworks()
                selectNetwork(created.id)
            }.onFailure { e ->
                if (e is CancellationException) throw e
                _status.value = "Create failed: ${Presence.friendlyCause(e)}."
            }
        }
    }

    fun selectNetwork(id: String) {
        _selectedId.value = id
        _detail.value = emptyList()
        if (id.isBlank()) return
        viewModelScope.launch { store.setGroupId(id) }
        refreshDetail(id)
    }

    fun clearSelection() {
        _selectedId.value = ""
        _detail.value = emptyList()
    }

    private fun refreshDetail(id: String) {
        viewModelScope.launch {
            _detailLoading.value = true
            runCatching {
                DllmApi.getNetworkDevices(store.coordinatorUrl.first(), id)
            }.onSuccess {
                _detail.value = it
            }.onFailure { e ->
                if (e is CancellationException) throw e
                _status.value = "Group detail failed: ${Presence.friendlyCause(e)}."
            }
            _detailLoading.value = false
        }
    }

    fun joinWithPassword(id: String, password: String) {
        viewModelScope.launch {
            runCatching {
                val base = store.coordinatorUrl.first()
                val deviceId = store.ensureNodeId()
                DllmApi.joinNetwork(
                    base, id, deviceId,
                    password.ifBlank { null }, null,
                    IdentityStore.stableDeviceName(),
                )
                store.setGroupId(id)
            }.onSuccess {
                _status.value = "Joined group — heartbeats now carry its id."
                refreshDetail(id)
                refreshNetworks()
            }.onFailure { e ->
                if (e is CancellationException) throw e
                _status.value = "Join failed: ${Presence.friendlyCause(e)}. " +
                    "Wrong password gives HTTP 401."
            }
        }
    }

    fun leaveSelected() {
        val id = _selectedId.value
        if (id.isBlank()) return
        viewModelScope.launch {
            runCatching {
                val base = store.coordinatorUrl.first()
                val deviceId = store.ensureNodeId()
                DllmApi.leaveNetwork(base, id, deviceId)
                if (store.groupId.first() == id) store.setGroupId("")
            }.onSuccess {
                _status.value = "Left the group."
                _selectedId.value = ""
                _detail.value = emptyList()
                refreshNetworks()
            }.onFailure { e ->
                if (e is CancellationException) throw e
                _status.value = "Leave failed: ${Presence.friendlyCause(e)}."
            }
        }
    }

    /**
     * Handles a QR scan: `dllm://net?...` group codes first, then
     * `dllm://pair?...` server codes, then legacy join-JSON. Manual paste
     * goes through the same path (CameraX screen has a paste fallback).
     */
    // Identity: scan NEVER rotates node_id — it only ensures one exists.
    fun applyScanned(raw: String) {
        val net = parseNetUri(raw)
        if (net != null) {
            val host = cleanHost(net.host)
            if (host.isEmpty()) {
                _status.value = "QR has no host — not saved. Re-show the code on the server."
                return
            }
            viewModelScope.launch {
                store.ensureNodeId()
                store.setCoordinatorUrl(net.baseUrl())
                if (net.fingerprint.isNotBlank()) {
                    store.setServerFingerprint(net.fingerprint)
                }
                if (net.qrSecret.isNotBlank() || net.password.isNotBlank()) {
                    // Secret or embedded password: join immediately.
                    runCatching {
                        DllmApi.joinNetwork(
                            net.baseUrl(),
                            net.networkId,
                            store.ensureNodeId(),
                            net.password.ifBlank { null },
                            net.qrSecret.ifBlank { null },
                            IdentityStore.stableDeviceName(),
                        )
                        store.setGroupId(net.networkId)
                    }.onSuccess {
                        _pendingNet.value = null
                        _status.value = "Joined group via QR."
                        refreshNetworks()
                        selectNetwork(net.networkId)
                    }.onFailure { e ->
                        if (e is CancellationException) throw e
                        _pendingNet.value = net
                        _status.value = "QR points at ${net.baseUrl()} but join failed " +
                            "(${Presence.friendlyCause(e)}). Type the password below to retry."
                    }
                } else {
                    // No secret in the code: park it and ask for the password.
                    _pendingNet.value = net
                    _status.value = "QR targets group ${net.networkId} at ${net.baseUrl()} — " +
                        "enter its password below, then Join."
                }
            }
            return
        }
        val pair = parsePairUri(raw)
        if (pair != null) {
            val host = cleanHost(pair.host)
            if (host.isEmpty()) {
                _status.value = "QR has no host — not saved. Re-show the code on the server."
                return
            }
            viewModelScope.launch {
                store.ensureNodeId()
                store.savePairing("http://$host:${pair.port}", pair.fingerprint, pair.quicPort)
                _status.value = "Paired to http://$host:${pair.port} — pick a group below." +
                    (hostWarning(host)?.let { " Warning: $it" } ?: "")
                refreshNetworks()
            }
            return
        }
        val legacy = parseJoinPayload(raw)
        if (legacy != null) {
            viewModelScope.launch {
                store.ensureNodeId()
                store.setCoordinatorUrl(legacy.url.trim().trimEnd('/'))
                _status.value = "Coordinator set to ${legacy.url} (legacy join code)."
                refreshNetworks()
            }
            return
        }
        _status.value = "QR unreadable — expected a dllm://net?... or dllm://pair?... code."
    }

    fun joinPending(password: String) {
        val net = _pendingNet.value ?: return
        viewModelScope.launch {
            runCatching {
                DllmApi.joinNetwork(
                    net.baseUrl(),
                    net.networkId,
                    store.ensureNodeId(),
                    password.ifBlank { net.password.ifBlank { null } },
                    net.qrSecret.ifBlank { null },
                    IdentityStore.stableDeviceName(),
                )
                store.setGroupId(net.networkId)
            }.onSuccess {
                _pendingNet.value = null
                _status.value = "Joined group via QR."
                refreshNetworks()
                selectNetwork(net.networkId)
            }.onFailure { e ->
                if (e is CancellationException) throw e
                _status.value = "Join failed: ${Presence.friendlyCause(e)}. " +
                    "Wrong password gives HTTP 401."
            }
        }
    }

    fun dismissPending() {
        _pendingNet.value = null
    }
}

@Composable
fun NetworksScreen(
    viewModel: NetworksViewModel = viewModel(),
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val networks by viewModel.networks.collectAsState()
    val selectedId by viewModel.selectedId.collectAsState()
    val detail by viewModel.detail.collectAsState()
    val detailLoading by viewModel.detailLoading.collectAsState()
    val pendingNet by viewModel.pendingNet.collectAsState()
    val status by viewModel.status.collectAsState()
    val workerEnabled by viewModel.workerEnabled.collectAsState()
    val workerStats by WorkerService.stats.collectAsState()

    var showCreate by remember { mutableStateOf(false) }
    var createName by remember { mutableStateOf("") }
    var createPassword by remember { mutableStateOf("") }
    var createOpen by remember { mutableStateOf(true) }
    var joinPassword by remember { mutableStateOf("") }
    var pendingPassword by remember { mutableStateOf("") }
    var showScan by remember { mutableStateOf(false) }
    val cameraGranted = remember(showScan) {
        ContextCompat.checkSelfPermission(context, android.Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
    }

    val activeNow = detail.filter { it.connected }
    val pairedIdle = detail.filterNot { it.connected }

    Column(modifier = modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState())) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Private groups (${networks.size})", color = Ink)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = viewModel::refreshNetworks) { Text("Refresh") }
                Button(onClick = { showCreate = !showCreate }) { Text("New") }
            }
        }
        Spacer(Modifier.height(4.dp))
        Text("All groups are LAN-visible on the active coordinator.", color = Muted)
        Spacer(Modifier.height(8.dp))

        if (showCreate) {
            Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF171D24))) {
                Column(Modifier.fillMaxWidth().padding(12.dp)) {
                    Text("Create group", color = Ink)
                    Spacer(Modifier.height(8.dp))
                    OutlinedTextField(
                        value = createName,
                        onValueChange = { createName = it },
                        label = { Text("Name (1–40 chars)") },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true,
                    )
                    Spacer(Modifier.height(8.dp))
                    OutlinedTextField(
                        value = createPassword,
                        onValueChange = { createPassword = it },
                        label = { Text("Password (optional)") },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true,
                    )
                    Spacer(Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text("Open join", color = Ink)
                        Switch(checked = createOpen, onCheckedChange = { createOpen = it })
                    }
                    Spacer(Modifier.height(8.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = {
                                viewModel.createNetwork(createName, createPassword, createOpen)
                                createName = ""
                                createPassword = ""
                                showCreate = false
                            },
                            modifier = Modifier.weight(1f),
                        ) { Text("Create") }
                        TextButton(onClick = { showCreate = false }) { Text("Cancel") }
                    }
                }
            }
            Spacer(Modifier.height(8.dp))
        }

        if (networks.isEmpty()) {
            Text("No groups yet — create one above.", color = Muted)
        }
        networks.forEach { net ->
            val selected = net.id == selectedId
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = if (selected) Color(0xFF1B2B28) else Color(0xFF171D24),
                ),
                modifier = Modifier.fillMaxWidth().clickable {
                    if (selected) viewModel.clearSelection() else viewModel.selectNetwork(net.id)
                },
            ) {
                Column(Modifier.fillMaxWidth().padding(12.dp)) {
                    Text(net.name, color = Ink)
                    Text(
                        (if (net.openJoin) "open join" else "invite only") +
                            (if (net.hasPassword) " · locked" else " · no password") +
                            (net.memberCount?.let { " · $it members" } ?: ""),
                        color = Muted,
                    )
                }
            }
            Spacer(Modifier.height(8.dp))
        }

        // Detail: paired vs active-now, join with password, leave.
        if (selectedId.isNotBlank()) {
            val net = networks.firstOrNull { it.id == selectedId }
            Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF1B2B28))) {
                Column(Modifier.fillMaxWidth().padding(12.dp)) {
                    Text(net?.name ?: selectedId, color = Ink)
                    Spacer(Modifier.height(4.dp))
                    if (detailLoading) {
                        Text("Loading members…", color = Muted)
                    } else {
                        Text("Active now (${activeNow.size})", color = Teal)
                        if (activeNow.isEmpty()) Text("None active right now.", color = Muted)
                        activeNow.forEach { d ->
                            Text(
                                "● ${d.deviceName?.takeIf { it.isNotBlank() } ?: d.deviceId}",
                                color = Ink,
                            )
                        }
                        Spacer(Modifier.height(8.dp))
                        Text("Paired (${pairedIdle.size})", color = Ink)
                        if (pairedIdle.isEmpty()) Text("No idle members.", color = Muted)
                        pairedIdle.forEach { d ->
                            Text(
                                "○ ${d.deviceName?.takeIf { it.isNotBlank() } ?: d.deviceId}",
                                color = Muted,
                            )
                        }
                    }
                    Spacer(Modifier.height(8.dp))
                    OutlinedTextField(
                        value = joinPassword,
                        onValueChange = { joinPassword = it },
                        label = { Text("Group password (if locked)") },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true,
                    )
                    Spacer(Modifier.height(8.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = {
                                viewModel.joinWithPassword(selectedId, joinPassword)
                                joinPassword = ""
                            },
                            modifier = Modifier.weight(1f),
                        ) { Text("Join") }
                        OutlinedButton(
                            onClick = viewModel::leaveSelected,
                            modifier = Modifier.weight(1f),
                        ) { Text("Leave") }
                    }
                }
            }
            Spacer(Modifier.height(8.dp))
        }

        // QR join (dllm://net) + manual paste fallback.
        pendingNet?.let { net ->
            Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF171D24))) {
                Column(Modifier.fillMaxWidth().padding(12.dp)) {
                    Text("QR group: ${net.networkId}", color = Ink)
                    Text("at ${net.baseUrl()}", color = Muted)
                    Spacer(Modifier.height(8.dp))
                    OutlinedTextField(
                        value = pendingPassword,
                        onValueChange = { pendingPassword = it },
                        label = { Text("Group password") },
                        modifier = Modifier.fillMaxWidth(),
                        singleLine = true,
                    )
                    Spacer(Modifier.height(8.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = {
                                viewModel.joinPending(pendingPassword)
                                pendingPassword = ""
                            },
                            modifier = Modifier.weight(1f),
                        ) { Text("Join") }
                        TextButton(onClick = viewModel::dismissPending) { Text("Dismiss") }
                    }
                }
            }
            Spacer(Modifier.height(8.dp))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { showScan = true }, modifier = Modifier.weight(1f)) {
                Text("Scan group QR")
            }
        }
        Spacer(Modifier.height(4.dp))
        Text("Codes look like dllm://net?id=…&host=… (CameraX, paste fallback inside).", color = Muted)
        Spacer(Modifier.height(12.dp))

        // Worker toggle with honest active state.
        Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF171D24))) {
            Column(Modifier.fillMaxWidth().padding(12.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text("Compute worker", color = Ink)
                    Switch(checked = workerEnabled, onCheckedChange = viewModel::setWorkerEnabled)
                }
                val dot = if (workerStats.running && workerStats.ready) "●" else "○"
                val dotColor = if (workerStats.running && workerStats.ready) Teal else Muted
                Text(
                    "$dot ${workerStats.status}",
                    color = dotColor,
                )
                if (!workerEnabled) Text("Worker off — heartbeat role=client.", color = Muted)
                else Text("Worker on — heartbeat role=worker with load + capabilities.", color = Muted)
            }
        }
        Spacer(Modifier.height(12.dp))

        // NsdDiscovery + heartbeat infra (moved from the Devices tab).
        PresenceCard()
        Spacer(Modifier.height(8.dp))

        if (status.isNotEmpty()) Text(status, color = Amber)
    }

    if (showScan) {
        Dialog(onDismissRequest = { showScan = false }) {
            Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF101418))) {
                if (cameraGranted) {
                    QrScanScreen(
                        onScanned = {
                            viewModel.applyScanned(it)
                            showScan = false
                        },
                        onCancel = { showScan = false },
                    )
                } else {
                    Column(Modifier.padding(16.dp)) {
                        Text(
                            "Camera permission not granted — grant it in Settings, or paste the join text on the scan screen after granting.",
                            color = Ink,
                        )
                        TextButton(onClick = { showScan = false }) { Text("Close") }
                    }
                }
            }
        }
    }
}
