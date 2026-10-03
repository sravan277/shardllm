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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.dllm.mesh.data.IdentityStore
import com.dllm.mesh.net.normalizeBaseUrl
import com.dllm.mesh.worker.WorkerService
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

private val Teal = Color(0xFF2DD4BF)
private val Muted = Color(0xFF93A1B0)
private val Ink = Color(0xFFE8EDF2)
private val Amber = Color(0xFFF5B544)

class SettingsViewModel(application: Application) : AndroidViewModel(application) {

    private val app = application
    private val store = IdentityStore(application)

    val serverUrl: StateFlow<String> = store.coordinatorUrl
        .stateIn(viewModelScope, SharingStarted.Eagerly, IdentityStore.DEFAULT_COORDINATOR_URL)
    val expectedFingerprint: StateFlow<String> = store.serverFingerprint
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")
    val serverNodeId: StateFlow<String> = store.serverNodeId
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")
    val serverQuicPort: StateFlow<Int> = store.serverQuicPort
        .stateIn(viewModelScope, SharingStarted.Eagerly, 0)
    val workerEnabled: StateFlow<Boolean> = store.workerEnabled
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    private val _status = MutableStateFlow("")
    val status: StateFlow<String> = _status.asStateFlow()

    init {
        viewModelScope.launch { store.ensureNodeId() }
    }

    fun saveServerUrl(raw: String) {
        val url = normalizeBaseUrl(raw)
        if (url.isBlank()) {
            _status.value = "Server URL is empty — not saved."
            return
        }
        if (!url.startsWith("http://") && !url.startsWith("https://")) {
            _status.value = "URL must start with http:// or https:// — not saved."
            return
        }
        viewModelScope.launch {
            store.setCoordinatorUrl(url)
            _status.value = "Server saved: $url"
        }
    }

    fun setWorkerEnabled(enabled: Boolean) {
        viewModelScope.launch {
            store.setWorkerEnabled(enabled)
            if (enabled) WorkerService.start(app) else WorkerService.stop(app)
            _status.value = if (enabled) "Worker on — probes local model.gguf when present."
            else "Worker off."
        }
    }
}

@Composable
fun SettingsScreen(
    viewModel: SettingsViewModel = viewModel(),
    modifier: Modifier = Modifier,
) {
    val serverUrl by viewModel.serverUrl.collectAsState()
    val expectedFp by viewModel.expectedFingerprint.collectAsState()
    val serverNodeId by viewModel.serverNodeId.collectAsState()
    val serverQuicPort by viewModel.serverQuicPort.collectAsState()
    val workerEnabled by viewModel.workerEnabled.collectAsState()
    val status by viewModel.status.collectAsState()
    val workerStats by WorkerService.stats.collectAsState()

    val context = LocalContext.current
    val versionName = remember {
        runCatching {
            @Suppress("DEPRECATION")
            context.packageManager.getPackageInfo(context.packageName, 0).versionName ?: "unknown"
        }.getOrNull() ?: "unknown"
    }

    Column(modifier = modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState())) {
        Text("Active server", color = Ink)
        Spacer(Modifier.height(4.dp))
        Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF1B2B28))) {
            Column(modifier = Modifier.fillMaxWidth().padding(12.dp)) {
                Text(serverUrl, color = Teal)
                Spacer(Modifier.height(4.dp))
                Text(
                    if (expectedFp.isNotBlank()) "Pinned fp: ${shortFingerprint(expectedFp)}"
                    else "No fingerprint pinned yet.",
                    color = Muted,
                )
                Text(
                    if (serverNodeId.isNotBlank()) "node_id: $serverNodeId"
                    else "node_id: unknown",
                    color = Muted,
                )
                Text(
                    if (serverQuicPort in 1..65535) "quic_port: $serverQuicPort"
                    else "quic_port: unknown",
                    color = Muted,
                )
            }
        }
        Spacer(Modifier.height(4.dp))
        Text("Change server in the Pairing tab.", color = Muted)
        Spacer(Modifier.height(16.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Offer this phone as compute worker", color = Ink)
            Switch(checked = workerEnabled, onCheckedChange = viewModel::setWorkerEnabled)
        }
        // Honest active state: green dot only when the service runs AND the
        // local model probe passed (heartbeat role=worker is live then).
        val workerReady = workerStats.running && workerStats.ready
        Text(
            "${if (workerReady) "●" else "○"} ${workerStats.status}",
            color = if (workerReady) Teal else Muted,
        )
        Text(
            if (workerEnabled) "Worker on — heartbeat role=worker with load + capabilities."
            else "Worker off (default) — heartbeat role=client.",
            color = Muted,
        )
        Spacer(Modifier.height(8.dp))
        if (status.isNotEmpty()) Text(status, color = Amber)
        Spacer(Modifier.height(16.dp))

        Text("About", color = Ink)
        Spacer(Modifier.height(4.dp))
        Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF171D24))) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(12.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Version", color = Ink)
                Text(versionName, color = Muted)
            }
        }
    }
}

private fun shortFingerprint(fp: String): String =
    if (fp.length <= 16) fp else fp.take(16) + "…"
