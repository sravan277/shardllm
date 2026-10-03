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
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.dllm.mesh.data.IdentityStore
import com.dllm.mesh.net.normalizeBaseUrl
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class SettingsViewModel(application: Application) : AndroidViewModel(application) {

    private val store = IdentityStore(application)

    val serverUrl: StateFlow<String> = store.coordinatorUrl
        .stateIn(viewModelScope, SharingStarted.Eagerly, IdentityStore.DEFAULT_COORDINATOR_URL)
    val expectedFingerprint: StateFlow<String> = store.serverFingerprint
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")
    val serverNodeId: StateFlow<String> = store.serverNodeId
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")
    val serverQuicPort: StateFlow<Int> = store.serverQuicPort
        .stateIn(viewModelScope, SharingStarted.Eagerly, 0)

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
}

/**
 * Server identity and app metadata.
 *
 * The compute-worker toggle used to live here as well as on the old Networks
 * tab — two switches writing the same flag, with no way to see the effect of
 * either. It now lives only on the Devices tab, next to the roster that shows
 * what the worker's compute actually bought.
 */
@Composable
fun SettingsScreen(
    viewModel: SettingsViewModel = viewModel(),
    modifier: Modifier = Modifier,
) {
    val serverUrl by viewModel.serverUrl.collectAsState()
    val expectedFp by viewModel.expectedFingerprint.collectAsState()
    val serverNodeId by viewModel.serverNodeId.collectAsState()
    val serverQuicPort by viewModel.serverQuicPort.collectAsState()
    val status by viewModel.status.collectAsState()

    val context = LocalContext.current
    val versionName = remember {
        runCatching {
            @Suppress("DEPRECATION")
            context.packageManager.getPackageInfo(context.packageName, 0).versionName ?: "unknown"
        }.getOrNull() ?: "unknown"
    }

    Column(modifier = modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState())) {
        Text("Active server", color = MeshColors.Text)
        Spacer(Modifier.height(4.dp))
        Card(colors = CardDefaults.cardColors(containerColor = MeshColors.PanelActive)) {
            Column(modifier = Modifier.fillMaxWidth().padding(12.dp)) {
                Text(serverUrl, color = MeshColors.Teal)
                Spacer(Modifier.height(4.dp))
                Text(
                    if (expectedFp.isNotBlank()) "Pinned fp: ${shortFingerprint(expectedFp)}"
                    else "No fingerprint pinned yet.",
                    color = MeshColors.Muted,
                )
                Text(
                    if (serverNodeId.isNotBlank()) "node_id: $serverNodeId"
                    else "node_id: unknown",
                    color = MeshColors.Muted,
                )
                Text(
                    if (serverQuicPort in 1..65535) "quic_port: $serverQuicPort"
                    else "quic_port: unknown",
                    color = MeshColors.Muted,
                )
            }
        }
        Spacer(Modifier.height(4.dp))
        Text("Change server in the Pairing tab.", color = MeshColors.Muted)
        Spacer(Modifier.height(16.dp))

        Card(colors = CardDefaults.cardColors(containerColor = MeshColors.Panel)) {
            Column(Modifier.fillMaxWidth().padding(12.dp)) {
                Text("Compute worker", color = MeshColors.Text)
                Spacer(Modifier.height(4.dp))
                Text(
                    "This device's compute contribution is toggled on the Devices tab, " +
                        "where the mesh roster shows what each device is doing.",
                    color = MeshColors.Muted,
                )
            }
        }
        Spacer(Modifier.height(8.dp))
        if (status.isNotEmpty()) Text(status, color = MeshColors.Amber)
        Spacer(Modifier.height(16.dp))

        Text("About", color = MeshColors.Text)
        Spacer(Modifier.height(4.dp))
        Card(colors = CardDefaults.cardColors(containerColor = MeshColors.Panel)) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(12.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Version", color = MeshColors.Text)
                Text(versionName, color = MeshColors.Muted)
            }
        }
    }
}

private fun shortFingerprint(fp: String): String =
    if (fp.length <= 16) fp else fp.take(16) + "…"
