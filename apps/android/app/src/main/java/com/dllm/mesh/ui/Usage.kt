package com.dllm.mesh.ui

import android.app.Application
import android.widget.Toast
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
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.dllm.mesh.data.IdentityStore
import com.dllm.mesh.data.ModelTopology
import com.dllm.mesh.data.coversWholeModel
import com.dllm.mesh.data.layerRangeText
import com.dllm.mesh.net.DllmApi
import com.dllm.mesh.net.PlanStage
import com.dllm.mesh.net.UsageDevice
import com.dllm.mesh.net.UsageSummary
import com.dllm.mesh.net.isNeedsUpgrade
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlin.coroutines.cancellation.CancellationException

private const val UPGRADE_TOAST = "coordinator needs upgrade"

class UsageViewModel(application: Application) : AndroidViewModel(application) {

    private val store = IdentityStore(application)

    val coordinatorUrl: StateFlow<String> = store.coordinatorUrl
        .stateIn(viewModelScope, SharingStarted.Eagerly, IdentityStore.DEFAULT_COORDINATOR_URL)

    private val _summary = MutableStateFlow<UsageSummary?>(null)
    val summary: StateFlow<UsageSummary?> = _summary.asStateFlow()

    private val _planFallback = MutableStateFlow<List<PlanStage>>(emptyList())
    val planFallback: StateFlow<List<PlanStage>> = _planFallback.asStateFlow()

    private val _needsUpgrade = MutableStateFlow(false)
    val needsUpgrade: StateFlow<Boolean> = _needsUpgrade.asStateFlow()

    private val _loading = MutableStateFlow(false)
    val loading: StateFlow<Boolean> = _loading.asStateFlow()

    private val _status = MutableStateFlow("")
    val status: StateFlow<String> = _status.asStateFlow()

    private val _toast = MutableStateFlow<String?>(null)
    val toast: StateFlow<String?> = _toast.asStateFlow()

    init {
        viewModelScope.launch { store.ensureNodeId() }
        refresh()
    }

    fun clearToast() {
        _toast.value = null
    }

    fun refresh() {
        viewModelScope.launch {
            _loading.value = true
            runCatching {
                val base = store.coordinatorUrl.first()
                // Whole-mesh totals: the roster of per-device rows belongs to the
                // Devices tab now, and the mesh is a single network, so there is
                // no group to narrow this by. ensureNodeId() still runs so the
                // heartbeat later in the send path has a row to update.
                store.ensureNodeId()
                DllmApi.getUsage(base, null)
            }.onSuccess { usage ->
                _summary.value = usage
                _needsUpgrade.value = false
                _status.value = ""
                // usage.plan may be empty on transitional servers — fall back
                // to /v1/plan for the layer strip only (tokens stay from usage).
                if (usage.planStages.isEmpty()) {
                    runCatching {
                        val base = store.coordinatorUrl.first()
                        DllmApi.getPlan(base)
                    }.onSuccess { _planFallback.value = it }
                        .onFailure { _planFallback.value = emptyList() }
                } else {
                    _planFallback.value = emptyList()
                }
            }.onFailure { e ->
                if (e is CancellationException) throw e
                if (e is com.dllm.mesh.net.UsageForbiddenException) {
                    // A denial still carries honest totals: show them with an
                    // explanation instead of an empty error screen.
                    _summary.value = e.summary
                    _planFallback.value = emptyList()
                    _needsUpgrade.value = false
                    _status.value = "Totals only — this phone is not permitted to read " +
                        "per-device rows (HTTP 403). The coordinator lists the full breakdown " +
                        "on its own Devices page."
                    _loading.value = false
                    return@launch
                }
                if (isNeedsUpgrade(e)) {
                    _summary.value = null
                    _planFallback.value = emptyList()
                    _needsUpgrade.value = true
                    _status.value = "unavailable — coordinator needs upgrade"
                    _toast.value = UPGRADE_TOAST
                } else {
                    _status.value = "Usage refresh failed: ${e.message} — check coordinator."
                }
            }
            _loading.value = false
        }
    }
}

private fun shortDevice(id: String): String =
    if (id.length > 20) "${id.take(9)}…${id.takeLast(4)}" else id

@Composable
fun UsageScreen(
    viewModel: UsageViewModel = viewModel(),
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val summary by viewModel.summary.collectAsState()
    val planFallback by viewModel.planFallback.collectAsState()
    val needsUpgrade by viewModel.needsUpgrade.collectAsState()
    val loading by viewModel.loading.collectAsState()
    val status by viewModel.status.collectAsState()
    val toastMsg by viewModel.toast.collectAsState()

    LaunchedEffect(toastMsg) {
        if (toastMsg != null) {
            Toast.makeText(context, toastMsg, Toast.LENGTH_LONG).show()
            viewModel.clearToast()
        }
    }

    Column(modifier = modifier.fillMaxSize().padding(16.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Usage", color = MeshColors.Text)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = viewModel::refresh, enabled = !loading) {
                    Text(if (loading) "Loading…" else "Refresh")
                }
            }
        }
        Spacer(Modifier.height(4.dp))
        when {
            loading && summary == null && !needsUpgrade -> {
                Text("Loading usage…", color = MeshColors.Muted)
            }
            needsUpgrade -> {
                Card(colors = CardDefaults.cardColors(containerColor = MeshColors.Panel)) {
                    Column(Modifier.fillMaxWidth().padding(12.dp)) {
                        Text("Compute share unavailable", color = MeshColors.Text)
                        Spacer(Modifier.height(4.dp))
                        Text(
                            "unavailable — coordinator needs upgrade",
                            color = MeshColors.Amber,
                        )
                        Spacer(Modifier.height(4.dp))
                        Text(
                            "Upgrade the coordinator to serve GET /v1/usage. Nothing here is guessed.",
                            color = MeshColors.Muted,
                        )
                    }
                }
            }
            summary != null -> {
                val u = summary!!
                val stages = if (u.planStages.isNotEmpty()) u.planStages else planFallback
                val layersByDevice = LinkedHashMap<String, MutableList<Int>>()
                for (st in stages) {
                    val arr = layersByDevice.getOrPut(st.deviceId) { ArrayList() }
                    for (l in st.layerStart..st.layerEnd) arr.add(l)
                }
                val distinctLayers = layersByDevice.values.flatten().distinct().sorted()
                // Compared against the shared model topology: the "single-device
                // fast path" badge is a claim about the model, so it must be
                // checked against the same constants the planner was given.
                val singleDevice = layersByDevice.size == 1 &&
                    coversWholeModel(distinctLayers)

                Text(
                    "${u.tokensOutTotal} tokens out · ${u.sessionsTotal} sessions",
                    color = MeshColors.Muted,
                )
                Spacer(Modifier.height(8.dp))
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(u.perDevice, key = { it.deviceId }) { d: UsageDevice ->
                        Card(colors = CardDefaults.cardColors(containerColor = MeshColors.Panel)) {
                            Column(Modifier.fillMaxWidth().padding(12.dp)) {
                                Text(
                                    d.deviceName?.takeIf { it.isNotBlank() } ?: shortDevice(d.deviceId),
                                    color = MeshColors.Text,
                                )
                                Text(d.deviceId, color = MeshColors.Muted)
                                Spacer(Modifier.height(4.dp))
                                Text(
                                    "tokens_out: ${d.tokensOut} · sessions: ${d.sessions}",
                                    color = MeshColors.Teal,
                                )
                                Spacer(Modifier.height(4.dp))
                                // Honest load dot: green only when the server has
                                // real reported/live numbers, gray otherwise.
                                val hasLoad = (d.loadSource == "live" || d.loadSource == "reported") &&
                                    d.cpuPct != null && d.memPct != null
                                val loadLine = when {
                                    hasLoad ->
                                        "● CPU ${"%.0f".format(d.cpuPct)}% · MEM ${"%.0f".format(d.memPct)}% (${d.loadSource})"
                                    else -> "○ CPU/MEM: not reporting"
                                }
                                Text(
                                    loadLine,
                                    color = if (hasLoad) MeshColors.Teal else MeshColors.Muted,
                                )
                                if (!d.role.isNullOrBlank()) {
                                    Text("role: ${d.role}", color = MeshColors.Muted)
                                }
                                // Compute-worker pill from the heartbeat
                                // `worker_active` flag: green only when the
                                // device actually advertised worker_active=true.
                                val (workerDot, workerText, workerColor) = when (d.workerActive) {
                                    true -> Triple(
                                        "●",
                                        "worker active — takes pipeline layers",
                                        MeshColors.Teal,
                                    )
                                    false -> Triple(
                                        "○",
                                        "worker idle — chat only",
                                        MeshColors.Muted,
                                    )
                                    null -> Triple(
                                        "○",
                                        "worker state: not reporting",
                                        MeshColors.Muted,
                                    )
                                }
                                Text(
                                    "$workerDot $workerText",
                                    color = workerColor,
                                )
                                val layers = if (d.layerStart != null && d.layerEnd != null) {
                                    (d.layerStart..d.layerEnd).toList()
                                } else {
                                    layersByDevice[d.deviceId]?.toList()
                                        // Fall back to the heartbeat `layers`
                                        // offer (e.g. phone advertising 0-27).
                                        ?: d.layers?.toList()
                                        ?: emptyList()
                                }
                                Text(
                                    if (layers.isNotEmpty()) {
                                        "layers ${layerRangeText(layers)} (${layers.size} layers)"
                                    } else {
                                        "layers: —"
                                    },
                                    color = MeshColors.Muted,
                                )
                            }
                        }
                    }
                }
                Spacer(Modifier.height(8.dp))
                if (singleDevice) {
                    Text(
                        "All ${ModelTopology.TOTAL_LAYERS} layers on this coordinator — single-device fast path.",
                        color = MeshColors.Muted,
                    )
                }
                Text("Link bandwidth not yet measured.", color = MeshColors.Muted)
                if (status.isNotBlank()) {
                    Spacer(Modifier.height(4.dp))
                    Text(status, color = MeshColors.Amber)
                }
            }
            else -> {
                Text(
                    if (status.isNotBlank()) status
                    else "No usage data — check the coordinator, then Refresh.",
                    color = MeshColors.Amber,
                )
                Spacer(Modifier.height(8.dp))
                Button(onClick = viewModel::refresh, enabled = !loading) { Text("Retry") }
            }
        }
    }
}
