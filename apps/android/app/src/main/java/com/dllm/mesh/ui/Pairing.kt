package com.dllm.mesh.ui

import android.app.Application
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.pm.PackageManager
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
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
import com.dllm.mesh.net.Presence
import com.dllm.mesh.net.QrScanScreen
import com.dllm.mesh.net.SseClient
import com.dllm.mesh.net.cleanHost
import com.dllm.mesh.net.currentHostPort
import com.dllm.mesh.net.hostWarning
import com.dllm.mesh.net.normalizeBaseUrl
import com.dllm.mesh.net.parseJoinPayload
import com.dllm.mesh.net.parsePairUri
import com.dllm.mesh.net.splitHostPort
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
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.net.UnknownHostException

private const val HEARTBEAT_INTERVAL_MS = 30_000L

data class PairTestResult(
    val healthOk: Boolean,
    val nodeId: String = "",
    val serverFingerprint: String = "",
    val quicPort: Int? = null,
    val version: String = "",
    /** null = no comparison possible (nothing pinned or nothing reported). */
    val fingerprintMatch: Boolean? = null,
    val nodeError: String = "",
)

/**
 * Pairing *and* connection orchestration.
 *
 * WHY this ViewModel owns connecting: pairing and being connected used to be
 * three separate user actions (scan → Test → Send request → Use as server →
 * Connect), each of which could be forgotten individually, and a user who did
 * the first two still saw a dead chat. Pairing is now a one-time event and
 * connecting is a consequence: the primitives ([SseClient.getHealth],
 * [Presence.postHeartbeat]) all existed, they just had nothing tying them
 * together. So every path that changes the coordinator — scan, manual entry,
 * a LAN peer, app launch — ends in the same [connect] routine, and there is
 * exactly one place where "connected" is decided.
 *
 * Only one of this ViewModel and [DevicesViewModel] is alive at a time (the
 * NavHost disposes the non-visible destination), so the two never run competing
 * heartbeat loops against the same coordinator.
 */
class PairingViewModel(application: Application) : AndroidViewModel(application) {

    private val store = IdentityStore(application)
    private val sse = SseClient()

    val serverUrl: StateFlow<String> = store.coordinatorUrl
        .stateIn(viewModelScope, SharingStarted.Eagerly, IdentityStore.DEFAULT_COORDINATOR_URL)
    val expectedFingerprint: StateFlow<String> = store.serverFingerprint
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")
    val serverNodeId: StateFlow<String> = store.serverNodeId
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")
    val myNodeId: StateFlow<String> = store.nodeId
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")

    private val _status = MutableStateFlow("")
    val status: StateFlow<String> = _status.asStateFlow()

    private val _testing = MutableStateFlow(false)
    val testing: StateFlow<Boolean> = _testing.asStateFlow()

    private val _testResult = MutableStateFlow<PairTestResult?>(null)
    val testResult: StateFlow<PairTestResult?> = _testResult.asStateFlow()

    /**
     * True once the coordinator answered /api/health AND this phone's heartbeat
     * was accepted — the two conditions that together mean "I can actually chat
     * from here". Reset to false the moment either stops being true.
     */
    private val _connected = MutableStateFlow(false)
    val connected: StateFlow<Boolean> = _connected.asStateFlow()

    private var heartbeatJob: Job? = null

    private val discovery = NsdDiscovery(application)
    val nearbyPeers: StateFlow<List<Peer>> = discovery.peers

    init {
        viewModelScope.launch { store.ensureNodeId() }
        viewModelScope.launch {
            val id = store.ensureNodeId()
            advertise(id)
        }
        // Zero taps to be connected after the first successful pairing: if a
        // coordinator was really persisted (not the placeholder default), verify
        // it and announce this phone on launch. A first run with nothing saved
        // must stay quiet rather than probing 192.168.1.10 and reporting a
        // failure the user never caused.
        viewModelScope.launch {
            if (!store.hasCoordinator.first()) {
                _status.value = "No coordinator saved yet — scan, enter one, or tap Nearby."
                return@launch
            }
            connect()
        }
    }

    /**
     * Handles a QR scan: `dllm://pair?...` URIs, plus a legacy join-JSON
     * fallback. There is no group code any more — the mesh is a single network,
     * so a `dllm://net?...` code is simply unreadable and the user is told what
     * to re-show rather than being half-joined.
     */
    // Identity: scan NEVER rotates node_id — it only ensures one exists.
    fun applyScanned(raw: String) {
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
                _testResult.value = null
                _status.value = "Paired to http://$host:${pair.port}" +
                    (if (pair.fingerprint.isNotBlank()) " — fingerprint pinned." else " — no fingerprint in code.") +
                    (hostWarning(host)?.let { " Warning: $it" } ?: "")
                // Pairing is the one-time event; connecting follows immediately
                // so the user's first action already produces a working chat.
                connect()
            }
            return
        }
        val legacy = parseJoinPayload(raw)
        if (legacy != null) {
            viewModelScope.launch {
                // Identity: legacy join NEVER adopts the payload's node_id — it is
                // the *server's* ID in old codes, not this phone's.
                store.ensureNodeId()
                store.setCoordinatorUrl(normalizeBaseUrl(legacy.url))
                if (legacy.fingerprint.isNotBlank()) {
                    store.setServerFingerprint(legacy.fingerprint)
                }
                _testResult.value = null
                _status.value = "Coordinator set to ${legacy.url} (legacy join code)."
                connect()
            }
            return
        }
        _status.value = "QR unreadable — expected a dllm://pair?... code."
    }

    fun saveManual(host: String, portText: String, fingerprint: String) {
        viewModelScope.launch { saveManualSync(host, portText, fingerprint) }
    }

    /**
     * Manual entry that also connects. Saving without connecting would leave the
     * user staring at a saved-but-dead server, which is the exact confusion this
     * screen is being fixed for.
     */
    fun saveAndTest(host: String, portText: String, fingerprint: String) {
        viewModelScope.launch {
            if (saveManualSync(host, portText, fingerprint)) connect()
        }
    }

    /**
     * Validates + persists a manual pairing. Accepts `host` or `host:port`
     * pastes and strips schemes. A changed host drops the stale TOFU pin and
     * QUIC port so a new server is never trusted on an old pin.
     * Identity: manual entry NEVER rotates node_id — it only ensures one exists.
     */
    private suspend fun saveManualSync(host: String, portText: String, fingerprint: String): Boolean {
        var h = cleanHost(host)
        var port = portText.trim().toIntOrNull()
        if (port == null && portText.isBlank()) {
            val (sh, sp) = splitHostPort(h)
            if (sp.isNotEmpty()) {
                h = sh
                port = sp.toIntOrNull()
            }
        }
        if (h.isEmpty()) {
            _status.value = "Host is empty — not saved."
            return false
        }
        if (port == null || port <= 0 || port > 65535) {
            _status.value = "Port must be 1–65535 — not saved."
            return false
        }
        val prev = currentHostPort(store.coordinatorUrl.first())
        val hostChanged = !prev.first.equals(h, ignoreCase = true) ||
            (prev.second.isNotEmpty() && prev.second != port.toString())
        val typedFp = fingerprint.trim()
        val fp = when {
            typedFp.isNotBlank() -> typedFp
            hostChanged -> ""
            else -> store.serverFingerprint.first()
        }
        val quic = store.serverQuicPort.first().takeIf { it in 1..65535 && !hostChanged }
        store.ensureNodeId()
        store.savePairing("http://$h:$port", fp, quic)
        _testResult.value = null
        _status.value = "Paired to http://$h:$port" +
            (if (fp.isNotBlank()) " — fingerprint pinned." else " — no fingerprint pinned.") +
            (if (hostChanged) " Previous pin cleared for the new host." else "") +
            (hostWarning(h)?.let { " Warning: $it" } ?: "")
        return true
    }

    /** Hits GET /api/health then GET /api/node; reports fp match vs the pin. */
    fun testServer() {
        viewModelScope.launch { testServerSync() }
    }

    /**
     * The single connect routine: verify the coordinator, then announce this
     * phone. Returns quietly on failure with the reason in [status] — every
     * caller (launch, scan, manual save, peer tap, Reconnect button) funnels
     * through here so "connected" always means the same thing.
     */
    fun connect() {
        viewModelScope.launch { connectSync() }
    }

    /**
     * One-tap recovery for a phone that was connected and then lost the
     * coordinator (laptop slept, WiFi roamed). Restarts the whole sequence —
     * test + heartbeat — rather than only re-pinging, because a phone can pass
     * a health check and still fail to register if the coordinator restarted
     * with an empty registry.
     */
    fun reconnect() {
        heartbeatJob?.cancel()
        heartbeatJob = null
        _connected.value = false
        _status.value = "Reconnecting…"
        connect()
    }

    /** Stop heartbeating; this phone goes idle on the coordinator after ~90s. */
    fun disconnect() {
        heartbeatJob?.cancel()
        heartbeatJob = null
        _connected.value = false
        _status.value = "Disconnected — this phone will go idle on the coordinator."
    }

    /** Start LAN browsing for `_dllm._tcp.` coordinators (shown in NEARBY). */
    fun startNearby() {
        discovery.startDiscovery()
        _status.value = "Scanning the LAN for nearby coordinators…"
    }

    fun stopNearby() {
        discovery.stopDiscovery()
    }

    override fun onCleared() {
        heartbeatJob?.cancel()
        heartbeatJob = null
        discovery.stopDiscovery()
        discovery.unregisterService()
    }

    /**
     * One tap on a discovered coordinator: adopt it as the server, verify it and
     * announce this phone, all in a single action.
     *
     * WHY this replaced the two-step "Send request" then "Use as server": both
     * buttons did part of the same job — the first heartbeat against a peer
     * registers this phone there, and the second only wrote the URL. Split, they
     * allowed the half-state where a phone had announced itself to a coordinator
     * it was not using. A discovered peer is reachable by construction, so there
     * is nothing for the user to decide between.
     */
    // Identity: adopting a server NEVER rotates node_id.
    fun connectViaPeer(peer: Peer) {
        viewModelScope.launch {
            val base = "http://${peer.host}:${peer.port}"
            store.ensureNodeId()
            store.setCoordinatorUrl(base)
            _testResult.value = null
            _status.value = "Connecting to $base (${peer.serviceName})…"
            connectSync()
        }
    }

    /**
     * Advertise this phone on the LAN so a laptop browsing `_dllm._tcp.` sees it
     * without the phone having to pair first. Best-effort inside
     * [NsdDiscovery.registerService]; failure is silent because a missing
     * advertisement is invisible to the user and not actionable.
     */
    private suspend fun advertise(deviceId: String) {
        val workerOn = store.workerEnabled.first()
        discovery.registerService(
            txt = mapOf(
                "node_id" to deviceId,
                "role" to if (workerOn) "worker" else "client",
                "worker" to workerOn.toString(),
            ),
        )
    }

    /**
     * "Use existing ID": paste a previous device_id (e.g. `RMX3785-realme`)
     * to reuse its coordinator row. Never generates — only persists.
     */
    fun adoptExistingId(raw: String) {
        viewModelScope.launch {
            if (store.setNodeId(raw)) {
                _status.value = "Device ID set to ${raw.trim()} — heartbeats will reuse that row."
            } else {
                _status.value = "Device ID is empty — not saved."
            }
        }
    }

    /**
     * Verify then announce: GET /api/health (+ optional /api/node for identity
     * and the TOFU comparison) followed by one heartbeat that creates/refreshes
     * this phone's registry row.
     *
     * WHY the heartbeat is not optional: health only proves *something* answers
     * at that URL. A fresh install is not in the coordinator's registry until it
     * announces itself, and an unregistered device cannot be assigned pipeline
     * layers. Running both means a successful [connected] state always implies
     * "chat works AND the mesh knows this phone exists".
     */
    private suspend fun connectSync(): Boolean {
        val base = normalizeBaseUrl(store.coordinatorUrl.first())
        val healthy = testServerSync()
        if (!healthy) {
            _connected.value = false
            return false
        }
        val beat = runCatching { Presence.postHeartbeat(getApplication(), base) }
        if (beat.isFailure) {
            _connected.value = false
            _status.value = "$base answered health but rejected this phone: " +
                "${Presence.friendlyCause(beat.exceptionOrNull()!!)}. " +
                "Fix: check the server is running with the same device_id, then Reconnect."
            return false
        }
        _connected.value = true
        _status.value = "Connected to $base — this phone is registered and active."
        startHeartbeatLoop(base)
        return true
    }

    /**
     * Keep the registry row warm: re-announce every 30s, comfortably inside the
     * coordinator's 90s activity window so a single dropped packet never makes
     * the phone look gone.
     *
     * A failed tick stops the loop and clears [connected] rather than retrying
     * silently — the UI must be able to say "this phone is not currently
     * connected", and a background retry would make that untrue. [reconnect] is
     * the user's one tap back.
     */
    private fun startHeartbeatLoop(base: String) {
        heartbeatJob?.cancel()
        heartbeatJob = viewModelScope.launch {
            while (isActive) {
                delay(HEARTBEAT_INTERVAL_MS)
                val tick = runCatching { Presence.postHeartbeat(getApplication(), base) }
                if (tick.isFailure) {
                    _connected.value = false
                    _status.value = "Heartbeat to $base lost: " +
                        "${Presence.friendlyCause(tick.exceptionOrNull()!!)}. " +
                        "Tap Reconnect to retry."
                    break
                }
            }
            heartbeatJob = null
        }
    }

    /**
     * Runs the verification half of [connectSync]. Returns true only when
     * /api/health answered, so callers can decide whether announcing makes
     * sense. /api/node stays best-effort: older coordinators 404 it and that is
     * not a reason to call the server unusable.
     */
    private suspend fun testServerSync(): Boolean {
        val base = normalizeBaseUrl(store.coordinatorUrl.first())
        if (base.isBlank()) {
            _status.value = "No server configured — pair first."
            return false
        }
        _testing.value = true
        _testResult.value = null
        _status.value = "Probing $base…"
        val health = runCatching { sse.getHealth(base) }
        val healthErr = health.exceptionOrNull()
        if (healthErr != null) {
            _status.value = "Health check failed: ${connectionFix(base, healthErr)}"
            _testing.value = false
            return false
        }
        val expected = store.serverFingerprint.first()
        val node = runCatching { sse.getNodeInfo(base) }
        if (node.isFailure) {
            // /api/node is an optional enhancement; older servers 404.
            _testResult.value = PairTestResult(healthOk = true, nodeError = node.exceptionOrNull()?.message.orEmpty())
            _status.value = "Health OK (HTTP ${health.getOrNull()}), but /api/node unavailable — server may be older."
        } else {
            val info = node.getOrThrow()
            if (info.nodeId.isNotBlank()) store.setServerNodeId(info.nodeId)
            val match = if (expected.isBlank() || info.fingerprint.isBlank()) null
            else expected == info.fingerprint
            _testResult.value = PairTestResult(
                healthOk = true,
                nodeId = info.nodeId,
                serverFingerprint = info.fingerprint,
                quicPort = info.quicPort,
                version = info.version,
                fingerprintMatch = match,
            )
            _status.value = when (match) {
                true -> "Health OK — fingerprint MATCHES the pin."
                false -> "TOFU WARNING: server fingerprint differs from the pinned one — possible impersonation."
                null -> "Health OK — no fingerprint comparison available."
            }
        }
        _testing.value = false
        return true
    }

    /**
     * Cause + fix for a failed health probe (never a raw exception).
     * The classic case is a multi-homed coordinator advertising a VPN/PPP
     * address (e.g. 172.16.0.2) while the phone sits on WiFi (192.168.x).
     */
    private fun connectionFix(base: String, e: Throwable): String {
        val msg = e.message.orEmpty()
        if (e is UnknownHostException || msg.contains("Unable to resolve host", ignoreCase = true)) {
            return "cannot resolve the server host ($base). " +
                "Fix: check the Host spelling, or enter the laptop WiFi IP from `dllm id`."
        }
        if (e is SocketTimeoutException || msg.contains("timeout", ignoreCase = true)) {
            return "timed out after 15s to $base. " +
                "Cause: wrong server IP (laptop advertised a VPN address instead of WiFi) or firewall/AP isolation. " +
                "Fix: on the laptop run `dllm id`, use a 192.168.x candidate, fix Host below, then Save & Test."
        }
        if (e is ConnectException || msg.contains("refused", ignoreCase = true) ||
            msg.contains("failed to connect", ignoreCase = true)
        ) {
            return "connection refused at $base. " +
                "Fix: is `dllm serve` running on the laptop? Is the port 8080?"
        }
        return msg.ifBlank { "unexpected error (${e::class.simpleName ?: "unknown"}). " +
            "Fix: check Host/Port below, then Save & Test." }
    }
}

private enum class PairSection { NONE, SCAN, MANUAL, NEARBY }

@Composable
fun PairingScreen(
    viewModel: PairingViewModel = viewModel(),
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val serverUrl by viewModel.serverUrl.collectAsState()
    val expectedFp by viewModel.expectedFingerprint.collectAsState()
    val serverNodeId by viewModel.serverNodeId.collectAsState()
    val myNodeId by viewModel.myNodeId.collectAsState()
    val status by viewModel.status.collectAsState()
    val testing by viewModel.testing.collectAsState()
    val testResult by viewModel.testResult.collectAsState()
    val nearbyPeers by viewModel.nearbyPeers.collectAsState()
    val connected by viewModel.connected.collectAsState()
    var reuseId by remember { mutableStateOf("") }

    var section by remember { mutableStateOf(PairSection.NONE) }
    var showScan by remember { mutableStateOf(false) }
    val cameraGranted = remember(showScan) {
        ContextCompat.checkSelfPermission(context, android.Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
    }

    // Prefill the manual form from the saved server; refreshes after each save.
    val (savedHost, savedPort) = remember(serverUrl) { currentHostPort(serverUrl) }
    var hostField by remember(savedHost) { mutableStateOf(savedHost) }
    var portField by remember(savedPort) { mutableStateOf(savedPort) }
    var fpField by remember { mutableStateOf("") }
    val hostWarn = remember(hostField) { hostWarning(hostField) }

    Column(
        modifier = modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()),
    ) {
        Text("Paired server", color = MeshColors.Text)
        Spacer(Modifier.height(4.dp))
        Text(serverUrl, color = MeshColors.Teal)
        Text(
            if (expectedFp.isNotBlank()) "Pinned fp: $expectedFp" else "No fingerprint pinned yet.",
            color = MeshColors.Muted,
        )
        if (serverNodeId.isNotBlank()) Text("node_id: $serverNodeId", color = MeshColors.Muted)
        Spacer(Modifier.height(12.dp))

        // Connection state lives at the top because it is the only thing the
        // user actually needs most of the time: after the first pairing this
        // screen opens already connected, and the whole screen collapses to a
        // single Reconnect tap when it is not.
        Card(
            colors = CardDefaults.cardColors(
                containerColor = if (connected) MeshColors.PanelActive else MeshColors.Panel,
            ),
        ) {
            Column(Modifier.fillMaxWidth().padding(12.dp)) {
                Text(
                    if (connected) "● Connected" else "○ Not connected",
                    color = if (connected) MeshColors.Teal else MeshColors.Muted,
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    if (connected) {
                        "This phone is verified and registered on $serverUrl."
                    } else {
                        "Tap Reconnect to verify the server and announce this phone."
                    },
                    color = MeshColors.Muted,
                )
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = viewModel::reconnect,
                        enabled = !testing,
                        modifier = Modifier.weight(1f),
                    ) { Text(if (testing) "Connecting…" else "Reconnect") }
                    if (connected) {
                        OutlinedButton(
                            onClick = viewModel::disconnect,
                            modifier = Modifier.weight(1f),
                        ) { Text("Disconnect") }
                    }
                }
            }
        }
        Spacer(Modifier.height(16.dp))

        // This phone's stable identity — full ID + copy + "use existing" so a
        // stale duplicate row can be reconciled without reinstalling.
        Card(colors = CardDefaults.cardColors(containerColor = MeshColors.PanelActive)) {
            Column(Modifier.fillMaxWidth().padding(12.dp)) {
                Text("This phone identity", color = MeshColors.Text)
                Spacer(Modifier.height(4.dp))
                SelectionContainer {
                    Text(
                        "device_id: ${myNodeId.ifBlank { "(loading…)" }}",
                        color = MeshColors.Muted,
                    )
                }
                Spacer(Modifier.height(4.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(
                        onClick = {
                            val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                            cm.setPrimaryClip(ClipData.newPlainText("device_id", myNodeId))
                        },
                        enabled = myNodeId.isNotBlank(),
                    ) { Text("Copy ID") }
                }
                Spacer(Modifier.height(8.dp))
                OutlinedTextField(
                    value = reuseId,
                    onValueChange = { reuseId = it },
                    label = { Text("Use existing ID") },
                    placeholder = { Text("paste old device_id, e.g. RMX3785-realme") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                )
                Spacer(Modifier.height(4.dp))
                OutlinedButton(
                    onClick = { viewModel.adoptExistingId(reuseId); reuseId = "" },
                    enabled = reuseId.isNotBlank(),
                ) { Text("Apply ID") }
            }
        }
        Spacer(Modifier.height(16.dp))

        // Step 1 — pick a method first.
        Text("Add a coordinator", color = MeshColors.Text)
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(
                onClick = { section = PairSection.SCAN },
                modifier = Modifier.weight(1f),
            ) { Text("Scan QR") }
            OutlinedButton(
                onClick = { section = PairSection.MANUAL },
                modifier = Modifier.weight(1f),
            ) { Text("Manual") }
            OutlinedButton(
                onClick = { viewModel.startNearby(); section = PairSection.NEARBY },
                modifier = Modifier.weight(1f),
            ) { Text("Nearby") }
        }
        Spacer(Modifier.height(12.dp))

        // Step 2 — the chosen method's own section.
        when (section) {
            PairSection.SCAN -> {
                Card(colors = CardDefaults.cardColors(containerColor = MeshColors.Panel)) {
                    Column(Modifier.fillMaxWidth().padding(12.dp)) {
                        Text("Scan the QR shown by the coordinator.", color = MeshColors.Text)
                        Spacer(Modifier.height(4.dp))
                        Text("Codes look like dllm://pair?host=… (CameraX).", color = MeshColors.Muted)
                        Spacer(Modifier.height(8.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(onClick = { showScan = true }) { Text("Open scanner") }
                            TextButton(onClick = { section = PairSection.NONE }) { Text("Back") }
                        }
                        Spacer(Modifier.height(4.dp))
                        Text("No camera? The scanner also accepts pasted pair text.", color = MeshColors.Muted)
                    }
                }
                Spacer(Modifier.height(12.dp))
            }
            PairSection.MANUAL -> {
                Card(colors = CardDefaults.cardColors(containerColor = MeshColors.Panel)) {
                    Column(Modifier.fillMaxWidth().padding(12.dp)) {
                        Text("Manual entry", color = MeshColors.Text)
                        Spacer(Modifier.height(8.dp))
                        OutlinedTextField(
                            value = hostField,
                            onValueChange = { hostField = it },
                            label = { Text("Host") },
                            placeholder = { Text("192.168.1.10") },
                            modifier = Modifier.fillMaxWidth(),
                            singleLine = true,
                        )
                        if (hostWarn != null) {
                            Spacer(Modifier.height(4.dp))
                            Text(hostWarn, color = MeshColors.Amber)
                        }
                        Spacer(Modifier.height(8.dp))
                        OutlinedTextField(
                            value = portField,
                            onValueChange = { portField = it },
                            label = { Text("Port") },
                            placeholder = { Text("8080") },
                            modifier = Modifier.fillMaxWidth(),
                            singleLine = true,
                        )
                        Spacer(Modifier.height(8.dp))
                        OutlinedTextField(
                            value = fpField,
                            onValueChange = { fpField = it },
                            label = { Text("Fingerprint (optional)") },
                            placeholder = { Text("blank keeps the current pin") },
                            modifier = Modifier.fillMaxWidth(),
                            singleLine = true,
                        )
                        Spacer(Modifier.height(8.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(
                                onClick = { viewModel.saveManual(hostField, portField, fpField) },
                                modifier = Modifier.weight(1f),
                            ) { Text("Save") }
                            Button(
                                onClick = { viewModel.saveAndTest(hostField, portField, fpField) },
                                enabled = !testing,
                                modifier = Modifier.weight(1f),
                            ) { Text("Save & Test") }
                        }
                        Row {
                            TextButton(onClick = { section = PairSection.NONE }) { Text("Back") }
                        }
                    }
                }
                Spacer(Modifier.height(12.dp))
            }
            PairSection.NEARBY -> {
                Card(colors = CardDefaults.cardColors(containerColor = MeshColors.Panel)) {
                    Column(Modifier.fillMaxWidth().padding(12.dp)) {
                        Text("Coordinators on this WiFi LAN (${nearbyPeers.size})", color = MeshColors.Text)
                        Spacer(Modifier.height(4.dp))
                        Text(
                            "One tap connects: the coordinator becomes this phone's server " +
                                "and this phone announces itself there.",
                            color = MeshColors.Muted,
                        )
                        Spacer(Modifier.height(8.dp))
                        if (nearbyPeers.isEmpty()) {
                            Text("Nothing found yet — scanning…", color = MeshColors.Muted)
                            Spacer(Modifier.height(8.dp))
                        }
                        // Other phones advertise themselves on the same service
                        // type; they do not serve chat, so they are listed but not
                        // offered as a connect target.
                        val coordinators = nearbyPeers
                            .filterNot { NsdDiscovery.isNodeAdvertisement(it.serviceName) }
                        val otherPhones = nearbyPeers
                            .filter { NsdDiscovery.isNodeAdvertisement(it.serviceName) }
                        coordinators.forEach { peer ->
                            Row(
                                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Column(Modifier.weight(1f)) {
                                    Text(peer.serviceName, color = MeshColors.Text)
                                    Text("${peer.host}:${peer.port}", color = MeshColors.Muted)
                                }
                                Button(onClick = { viewModel.connectViaPeer(peer) }) {
                                    Text("Connect")
                                }
                            }
                        }
                        if (otherPhones.isNotEmpty()) {
                            Text(
                                "Also on this LAN: ${otherPhones.size} phone(s) advertising " +
                                    "presence only — pair to their coordinator instead.",
                                color = MeshColors.Muted,
                            )
                        }
                        Row {
                            TextButton(
                                onClick = { viewModel.stopNearby(); section = PairSection.NONE },
                            ) { Text("Back") }
                        }
                    }
                }
                Spacer(Modifier.height(12.dp))
            }
            PairSection.NONE -> {}
        }

        // Server details from the last verification. There is deliberately no
        // standalone "Test" button any more: Reconnect at the top already runs
        // the check *and* announces this phone, so a Test-only button could only
        // ever produce a half-state — green health, no registry row.
        testResult?.let { result ->
            Card(colors = CardDefaults.cardColors(containerColor = MeshColors.Panel)) {
                Column(Modifier.fillMaxWidth().padding(12.dp)) {
                    Text(
                        "Health: ${if (result.healthOk) "OK" else "FAILED"}",
                        color = if (result.healthOk) MeshColors.Teal else MeshColors.Amber,
                    )
                    if (result.nodeError.isNotBlank()) {
                        Text("Node info unavailable: ${result.nodeError}", color = MeshColors.Muted)
                    } else {
                        if (result.nodeId.isNotBlank()) {
                            Text("node_id: ${result.nodeId}", color = MeshColors.Text)
                        }
                        if (result.version.isNotBlank()) {
                            Text("version: ${result.version}", color = MeshColors.Muted)
                        }
                        result.quicPort?.let { Text("quic_port: $it", color = MeshColors.Muted) }
                        if (result.serverFingerprint.isNotBlank()) {
                            Text("Server fp: ${result.serverFingerprint}", color = MeshColors.Muted)
                        }
                        when (result.fingerprintMatch) {
                            true -> Text("Fingerprint: MATCH", color = MeshColors.Teal)
                            false -> Text(
                                "Fingerprint: MISMATCH — TOFU warning, possible impersonation. " +
                                    "Re-pair only if you trust this server.",
                                color = MeshColors.Amber,
                            )
                            null -> Text("Fingerprint: no comparison available.", color = MeshColors.Muted)
                        }
                    }
                }
            }
            Spacer(Modifier.height(8.dp))
        }

        if (status.isNotEmpty()) Text(status, color = MeshColors.Amber)
    }

    if (showScan) {
        Dialog(onDismissRequest = { showScan = false }) {
            Card(colors = CardDefaults.cardColors(containerColor = MeshColors.Ink)) {
                if (cameraGranted) {
                    QrScanScreen(
                        onScanned = {
                            viewModel.applyScanned(it)
                            showScan = false
                            section = PairSection.NONE
                        },
                        onCancel = { showScan = false },
                    )
                } else {
                    Column(Modifier.padding(16.dp)) {
                        Text(
                            "Camera permission not granted — grant it in Settings, or paste the pair text on the scan screen after granting.",
                            color = MeshColors.Text,
                        )
                        TextButton(onClick = { showScan = false }) { Text("Close") }
                    }
                }
            }
        }
    }
}
