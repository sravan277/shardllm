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
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
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

class SettingsViewModel(application: Application) : AndroidViewModel(application) {

    private val app = application
    private val store = IdentityStore(application)

    val serverUrl: StateFlow<String> = store.coordinatorUrl
        .stateIn(viewModelScope, SharingStarted.Eagerly, IdentityStore.DEFAULT_COORDINATOR_URL)
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
            _status.value = if (enabled) "Worker on — idle stub (no compute until Phase 5)."
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
    val workerEnabled by viewModel.workerEnabled.collectAsState()
    val status by viewModel.status.collectAsState()

    var urlField by remember(serverUrl) { mutableStateOf(serverUrl) }

    Column(modifier = modifier.fillMaxSize().padding(16.dp)) {
        Text("Server", color = Color(0xFFE8EDF2))
        Spacer(Modifier.height(8.dp))
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = urlField,
                onValueChange = { urlField = it },
                label = { Text("Server base URL") },
                placeholder = { Text(IdentityStore.DEFAULT_COORDINATOR_URL) },
                modifier = Modifier.weight(1f),
                singleLine = true,
            )
            Button(onClick = { viewModel.saveServerUrl(urlField) }) { Text("Save") }
        }
        Spacer(Modifier.height(4.dp))
        Text("Active: $serverUrl", color = Color(0xFF93A1B0))
        Spacer(Modifier.height(16.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Offer this phone as compute worker (Phase 5)", color = Color(0xFFE8EDF2))
            Switch(checked = workerEnabled, onCheckedChange = viewModel::setWorkerEnabled)
        }
        Text(
            if (workerEnabled) "Worker on — idle stub, no compute until Phase 5."
            else "Worker off (default).",
            color = Color(0xFF93A1B0),
        )
        Spacer(Modifier.height(8.dp))
        if (status.isNotEmpty()) Text(status, color = Color(0xFFF5B544))
    }
}
