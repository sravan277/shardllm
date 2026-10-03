package com.dllm.mesh.ui

import android.app.Application
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Build
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.selection.SelectionContainer
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.dllm.mesh.data.IdentityStore
import com.dllm.mesh.net.NsdDiscovery
import com.dllm.mesh.net.Peer
import com.dllm.mesh.net.Presence
import com.dllm.mesh.net.baseUrlMatches
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Shared presence infra. The old Devices tab (grouped device lists +
 * approve/revoke UI) is gone — its NsdDiscovery + heartbeat pieces live on
 * here and are surfaced under Networks (nearby + presence) and Settings /
 * Networks (worker toggle). No approve/revoke exists on this client.
 */
class DevicesViewModel(application: Application) : AndroidViewModel(application) {

    private val app = application
    private val store = IdentityStore(application)
    private val discovery = NsdDiscovery(application)

    val coordinatorUrl: StateFlow<String> = store.coordinatorUrl
        .stateIn(viewModelScope, SharingStarted.Eagerly, IdentityStore.DEFAULT_COORDINATOR_URL)
    val peers: StateFlow<List<Peer>> = discovery.peers

    private val _status = MutableStateFlow("Idle.")
    val status: StateFlow<String> = _status.asStateFlow()

    val myNodeId: StateFlow<String> = store.nodeId
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")

    /** True while this phone's heartbeat loop runs (phone shows active everywhere). */
    private val _connected = MutableStateFlow(false)
    val connected: StateFlow<Boolean> = _connected.asStateFlow()
    private var presenceJob: Job? = null

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
        presenceJob?.cancel()
        presenceJob = null
        discovery.stopDiscovery()
        discovery.unregisterService()
    }

    /** Attach = save the peer as the active server. */
    // Identity: attaching a server NEVER rotates node_id.
    fun attachPeer(peer: Peer) {
        viewModelScope.launch {
            val base = "http://${peer.host}:${peer.port}"
            store.ensureNodeId()
            store.setCoordinatorUrl(base)
            _status.value = "Attached to $base (${peer.serviceName})."
        }
    }

    /**
     * Connect this phone to the mesh: heartbeat now, then every 30s (well
     * inside the server's 90s active window). Role + load + capabilities
     * come from [Presence] (worker when the toggle is on, else client).
     */
    fun connectPresence() {
        if (_connected.value) return
        viewModelScope.launch {
            val first = runCatching {
                Presence.postHeartbeat(app, store.coordinatorUrl.first().trimEnd('/'))
            }
            if (first.isFailure) {
                _status.value = "Connect failed: ${Presence.friendlyCause(first.exceptionOrNull()!!)}. " +
                    "Fix: check the server URL in the Pairing tab, then try again."
                return@launch
            }
            _connected.value = true
            _status.value = "Connected — this phone is now active on the coordinator."
            presenceJob?.cancel()
            presenceJob = viewModelScope.launch {
                while (isActive) {
                    delay(30_000)
                    val tick = runCatching {
                        Presence.postHeartbeat(app, store.coordinatorUrl.first().trimEnd('/'))
                    }
                    if (tick.isFailure) {
                        _connected.value = false
                        _status.value = "Heartbeat lost: ${Presence.friendlyCause(tick.exceptionOrNull()!!)}. " +
                            "Tap Connect to retry."
                        break
                    }
                }
                presenceJob = null
            }
        }
    }

    /** Stop heartbeating; the phone goes idle on the coordinator after ~90s. */
    fun disconnectPresence() {
        presenceJob?.cancel()
        presenceJob = null
        _connected.value = false
        _status.value = "Disconnected — this phone will go idle on the coordinator."
    }

    /**
     * Connect via a LAN peer: adopt it as the server, then announce this
     * phone there. Only offered when the peer is discovered nearby, so
     * Connect always targets a reachable coordinator.
     */
    fun connectViaPeer(peer: Peer) {
        viewModelScope.launch {
            val base = "http://${peer.host}:${peer.port}"
            store.ensureNodeId()
            store.setCoordinatorUrl(base)
            _status.value = "Attached to $base (${peer.serviceName})."
            connectPresence()
        }
    }
}

/** node_id advertised in a peer's mDNS TXT record (server sends lowercase keys). */
fun peerNodeId(peer: Peer): String =
    peer.txtMap.entries.firstOrNull { it.key.equals("node_id", ignoreCase = true) }?.value.orEmpty()

/**
 * Reusable presence card (Networks screen + anywhere else): active server,
 * nearby attach, this-phone connect. Approve/revoke intentionally absent.
 */
@Composable
fun PresenceCard(
    viewModel: DevicesViewModel = viewModel(),
    modifier: Modifier = Modifier,
) {
    val coordinatorUrl by viewModel.coordinatorUrl.collectAsState()
    val peers by viewModel.peers.collectAsState()
    val status by viewModel.status.collectAsState()
    val myNodeId by viewModel.myNodeId.collectAsState()
    val connected by viewModel.connected.collectAsState()
    val ctx = LocalContext.current

    Column(modifier = modifier) {
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
        Spacer(Modifier.height(8.dp))

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = viewModel::startDiscovery) { Text("Discover") }
            OutlinedButton(onClick = viewModel::stopDiscovery) { Text("Stop") }
        }
        Spacer(Modifier.height(8.dp))

        Text("Nearby (${peers.size})", color = Color(0xFFE8EDF2))
        Spacer(Modifier.height(4.dp))
        peers.forEach { peer ->
            Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF171D24))) {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(12.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text(peer.serviceName, color = Color(0xFFE8EDF2))
                        Text("${peer.host}:${peer.port}", color = Color(0xFF93A1B0))
                    }
                    if (baseUrlMatches(coordinatorUrl, peer.host, peer.port)) {
                        Text("Active", color = Color(0xFF2DD4BF))
                    } else {
                        TextButton(onClick = { viewModel.attachPeer(peer) }) { Text("Attach") }
                    }
                }
            }
            Spacer(Modifier.height(8.dp))
        }

        Text("This phone", color = Color(0xFFE8EDF2))
        Spacer(Modifier.height(4.dp))
        Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF1B2B28))) {
            Column(Modifier.fillMaxWidth().padding(12.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text(Build.MODEL, color = Color(0xFFE8EDF2))
                        Text(
                            if (connected) "Connected · active on coordinator" else "Offline · idle on coordinator",
                            color = if (connected) Color(0xFF2DD4BF) else Color(0xFF93A1B0),
                        )
                    }
                    if (connected) {
                        TextButton(onClick = viewModel::disconnectPresence) { Text("Disconnect") }
                    } else {
                        Button(onClick = viewModel::connectPresence) { Text("Connect") }
                    }
                }
                Spacer(Modifier.height(8.dp))
                SelectionContainer {
                    Text(
                        "device_id: ${myNodeId.ifBlank { "(loading…)" }}",
                        color = Color(0xFF93A1B0),
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    TextButton(
                        onClick = {
                            val cm = ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                            cm.setPrimaryClip(ClipData.newPlainText("device_id", myNodeId))
                        },
                        enabled = myNodeId.isNotBlank(),
                    ) { Text("Copy ID") }
                }
            }
        }
        Spacer(Modifier.height(8.dp))
        Text(status, color = Color(0xFFF5B544))
    }
}
