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
import androidx.compose.foundation.rememberScrollState
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
import com.dllm.mesh.net.QrScanScreen
import com.dllm.mesh.net.SseClient
import com.dllm.mesh.net.normalizeBaseUrl
import com.dllm.mesh.net.parseJoinPayload
import com.dllm.mesh.net.parsePairUri
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

private val Teal = Color(0xFF2DD4BF)
private val Muted = Color(0xFF93A1B0)
private val Ink = Color(0xFFE8EDF2)
private val Amber = Color(0xFFF5B544)

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

class PairingViewModel(application: Application) : AndroidViewModel(application) {

    private val store = IdentityStore(application)
    private val sse = SseClient()

    val serverUrl: StateFlow<String> = store.coordinatorUrl
        .stateIn(viewModelScope, SharingStarted.Eagerly, IdentityStore.DEFAULT_COORDINATOR_URL)
    val expectedFingerprint: StateFlow<String> = store.serverFingerprint
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")
    val serverNodeId: StateFlow<String> = store.serverNodeId
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")

    private val _status = MutableStateFlow("")
    val status: StateFlow<String> = _status.asStateFlow()

    private val _testing = MutableStateFlow(false)
    val testing: StateFlow<Boolean> = _testing.asStateFlow()

    private val _testResult = MutableStateFlow<PairTestResult?>(null)
    val testResult: StateFlow<PairTestResult?> = _testResult.asStateFlow()

    init {
        viewModelScope.launch { store.ensureNodeId() }
    }

    /** Handles a QR scan: `dllm://pair?...` URIs, plus legacy join-JSON fallback. */
    fun applyScanned(raw: String) {
        val pair = parsePairUri(raw)
        if (pair != null) {
            viewModelScope.launch {
                store.savePairing(pair.baseUrl(), pair.fingerprint, pair.quicPort)
                _testResult.value = null
                _status.value = "Paired to ${pair.baseUrl()}" +
                    (if (pair.fingerprint.isNotBlank()) " — fingerprint pinned." else " — no fingerprint in code.")
            }
            return
        }
        val legacy = parseJoinPayload(raw)
        if (legacy != null) {
            viewModelScope.launch {
                store.setCoordinatorUrl(normalizeBaseUrl(legacy.url))
                if (legacy.fingerprint.isNotBlank()) {
                    store.setServerFingerprint(legacy.fingerprint)
                }
                _testResult.value = null
                _status.value = "Coordinator set to ${legacy.url} (legacy join code)."
            }
            return
        }
        _status.value = "QR unreadable — expected a dllm://pair?... code."
    }

    fun saveManual(host: String, portText: String, fingerprint: String) {
        val h = host.trim()
        val port = portText.trim().toIntOrNull()
        if (h.isEmpty()) {
            _status.value = "Host is empty — not saved."
            return
        }
        if (port == null || port <= 0 || port > 65535) {
            _status.value = "Port must be 1–65535 — not saved."
            return
        }
        viewModelScope.launch {
            val quic = store.serverQuicPort.first().takeIf { it in 1..65535 }
            store.savePairing("http://$h:$port", fingerprint.trim(), quic)
            _testResult.value = null
            _status.value = "Paired to http://$h:$port" +
                (if (fingerprint.isNotBlank()) " — fingerprint pinned." else " — no fingerprint pinned.")
        }
    }

    /** Hits GET /api/health then GET /api/node; reports fp match vs the pin. */
    fun testServer() {
        viewModelScope.launch {
            val base = normalizeBaseUrl(store.coordinatorUrl.first())
            if (base.isBlank()) {
                _status.value = "No server configured — pair first."
                return@launch
            }
            _testing.value = true
            _testResult.value = null
            _status.value = "Probing $base…"
            val health = runCatching { sse.getHealth(base) }
            if (health.isFailure) {
                _status.value = "Health check failed: ${health.exceptionOrNull()?.message}"
                _testing.value = false
                return@launch
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
        }
    }
}

@Composable
fun PairingScreen(
    viewModel: PairingViewModel = viewModel(),
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val serverUrl by viewModel.serverUrl.collectAsState()
    val expectedFp by viewModel.expectedFingerprint.collectAsState()
    val serverNodeId by viewModel.serverNodeId.collectAsState()
    val status by viewModel.status.collectAsState()
    val testing by viewModel.testing.collectAsState()
    val testResult by viewModel.testResult.collectAsState()

    var showScan by remember { mutableStateOf(false) }
    var hostField by remember { mutableStateOf("") }
    var portField by remember { mutableStateOf("") }
    var fpField by remember { mutableStateOf("") }
    val cameraGranted = remember(showScan) {
        ContextCompat.checkSelfPermission(context, android.Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
    }

    Column(
        modifier = modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()),
    ) {
        Text("Paired server", color = Ink)
        Spacer(Modifier.height(4.dp))
        Text(serverUrl, color = Teal)
        Text(
            if (expectedFp.isNotBlank()) "Pinned fp: $expectedFp" else "No fingerprint pinned yet.",
            color = Muted,
        )
        if (serverNodeId.isNotBlank()) Text("node_id: $serverNodeId", color = Muted)
        Spacer(Modifier.height(12.dp))

        Button(onClick = { showScan = true }, modifier = Modifier.fillMaxWidth()) {
            Text("Scan QR code")
        }
        Spacer(Modifier.height(4.dp))
        Text("Scans dllm://pair?... codes (CameraX).", color = Muted)
        Spacer(Modifier.height(16.dp))

        Text("Manual entry", color = Ink)
        Spacer(Modifier.height(8.dp))
        OutlinedTextField(
            value = hostField,
            onValueChange = { hostField = it },
            label = { Text("Host") },
            placeholder = { Text("192.168.1.10") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
        )
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = portField,
                onValueChange = { portField = it },
                label = { Text("Port") },
                placeholder = { Text("8080") },
                modifier = Modifier.weight(1f),
                singleLine = true,
            )
        }
        Spacer(Modifier.height(8.dp))
        OutlinedTextField(
            value = fpField,
            onValueChange = { fpField = it },
            label = { Text("Fingerprint (optional)") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
        )
        Spacer(Modifier.height(8.dp))
        OutlinedButton(
            onClick = { viewModel.saveManual(hostField, portField, fpField) },
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Save manual pairing") }
        Spacer(Modifier.height(16.dp))

        Button(
            onClick = viewModel::testServer,
            enabled = !testing,
            modifier = Modifier.fillMaxWidth(),
        ) { Text(if (testing) "Testing…" else "Test") }
        Spacer(Modifier.height(8.dp))

        testResult?.let { result ->
            Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF171D24))) {
                Column(Modifier.fillMaxWidth().padding(12.dp)) {
                    Text(
                        "Health: ${if (result.healthOk) "OK" else "FAILED"}",
                        color = if (result.healthOk) Teal else Amber,
                    )
                    if (result.nodeError.isNotBlank()) {
                        Text("Node info unavailable: ${result.nodeError}", color = Muted)
                    } else {
                        if (result.nodeId.isNotBlank()) Text("node_id: ${result.nodeId}", color = Ink)
                        if (result.version.isNotBlank()) Text("version: ${result.version}", color = Muted)
                        result.quicPort?.let { Text("quic_port: $it", color = Muted) }
                        if (result.serverFingerprint.isNotBlank()) {
                            Text("Server fp: ${result.serverFingerprint}", color = Muted)
                        }
                        when (result.fingerprintMatch) {
                            true -> Text("Fingerprint: MATCH", color = Teal)
                            false -> Text(
                                "Fingerprint: MISMATCH — TOFU warning, possible impersonation. " +
                                    "Re-pair only if you trust this server.",
                                color = Amber,
                            )
                            null -> Text("Fingerprint: no comparison available.", color = Muted)
                        }
                    }
                }
            }
            Spacer(Modifier.height(8.dp))
        }

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
                            "Camera permission not granted — grant it in Settings, or paste the pair text on the scan screen after granting.",
                            color = Ink,
                        )
                        TextButton(onClick = { showScan = false }) { Text("Close") }
                    }
                }
            }
        }
    }
}
