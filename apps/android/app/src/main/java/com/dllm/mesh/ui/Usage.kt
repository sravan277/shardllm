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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.dllm.mesh.data.IdentityStore
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

    val groupId: StateFlow<String> = store.groupId
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")

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
                val group = store.groupId.first().ifBlank { null }
                val deviceId = store.ensureNodeId()
                DllmApi.getUsage(base, group, deviceId)
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
                    // Group-scoped denial still carries honest totals: show
                    // them with an explanation instead of an empty error.
                    _summary.value = e.summary
                    _planFallback.value = emptyList()
                    _needsUpgrade.value = false
                    _status.value = "Totals only — this phone is not a group member " +
                        "(or not admin): per-device rows need a group join + admin."
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

private fun layerRangeText(layers: List<Int>): String {
    if (layers.isEmpty()) return "none"
    val sorted = layers.sorted()
    val ranges = ArrayList<String>()
    var s = sorted[0]
    var p = sorted[0]
    for (i in 1..sorted.size) {
        val c = if (i < sorted.size) sorted[i] else Int.MIN_VALUE
        if (c == p + 1) {
            p = c
            continue
        }
        ranges.add(if (s == p) "$s" else "$s–$p")
        if (i < sorted.size) {
            s = c
            p = c
        }
    }
    return ranges.joinToString(", ")
}

@Composable
fun UsageScreen(
    viewModel: UsageViewModel = viewModel(),
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val summary by viewModel.summary.collectAsState()
    val groupId by viewModel.groupId.collectAsState()
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
            Text("Usage", color = Color(0xFFE8EDF2))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = viewModel::refresh, enabled = !loading) {
                    Text(if (loading) "Loading…" else "Refresh")
                }
            }
        }
        Spacer(Modifier.height(4.dp))
        when {
            loading && summary == null && !needsUpgrade -> {
                Text("Loading usage…", color = Color(0xFF93A1B0))
            }
            needsUpgrade -> {
                Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF171D24))) {
                    Column(Modifier.fillMaxWidth().padding(12.dp)) {
                        Text("Compute share unavailable", color = Color(0xFFE8EDF2))
                        Spacer(Modifier.height(4.dp))
                        Text(
                            "unavailable — coordinator needs upgrade",
                            color = Color(0xFFF5B544),
                        )
                        Spacer(Modifier.height(4.dp))
                        Text(
                            "Upgrade the coordinator to serve GET /v1/usage. Nothing here is guessed.",
                            color = Color(0xFF93A1B0),
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
                val singleDevice = layersByDevice.size == 1 &&
                    distinctLayers.size == 28 &&
                    distinctLayers.firstOrNull() == 0 &&
                    distinctLayers.lastOrNull() == 27

                Text(
                    "${u.tokensOutTotal} tokens out · ${u.sessionsTotal} sessions" +
                        (groupId.ifBlank { null }?.let { " · group $it" } ?: ""),
                    color = Color(0xFF93A1B0),
                )
                Spacer(Modifier.height(8.dp))
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(u.perDevice, key = { it.deviceId }) { d: UsageDevice ->
                        Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF171D24))) {
                            Column(Modifier.fillMaxWidth().padding(12.dp)) {
                                Text(
                                    d.deviceName?.takeIf { it.isNotBlank() } ?: shortDevice(d.deviceId),
                                    color = Color(0xFFE8EDF2),
                                )
                                Text(d.deviceId, color = Color(0xFF93A1B0))
                                Spacer(Modifier.height(4.dp))
                                Text(
                                    "tokens_out: ${d.tokensOut} · sessions: ${d.sessions}",
                                    color = Color(0xFF2DD4BF),
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
                                    color = if (hasLoad) Color(0xFF2DD4BF) else Color(0xFF93A1B0),
                                )
                                if (!d.role.isNullOrBlank()) {
                                    Text("role: ${d.role}", color = Color(0xFF93A1B0))
                                }
                                // Compute-worker pill from the heartbeat
                                // `worker_active` flag: green only when the
                                // device actually advertised worker_active=true.
                                val (workerDot, workerText, workerColor) = when (d.workerActive) {
                                    true -> Triple(
                                        "●",
                                        "worker active — takes pipeline layers",
                                        Color(0xFF2DD4BF),
                                    )
                                    false -> Triple(
                                        "○",
                                        "worker idle — chat only",
                                        Color(0xFF93A1B0),
                                    )
                                    null -> Triple(
                                        "○",
                                        "worker state: not reporting",
                                        Color(0xFF93A1B0),
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
                                    color = Color(0xFF93A1B0),
                                )
                            }
                        }
                    }
                }
                Spacer(Modifier.height(8.dp))
                if (singleDevice) {
                    Text(
                        "All 28 layers on this coordinator — single-device fast path.",
                        color = Color(0xFF93A1B0),
                    )
                }
                Text("Link bandwidth not yet measured.", color = Color(0xFF93A1B0))
                if (status.isNotBlank()) {
                    Spacer(Modifier.height(4.dp))
                    Text(status, color = Color(0xFFF5B544))
                }
            }
            else -> {
                Text(
                    if (status.isNotBlank()) status
                    else "No usage data — check the coordinator, then Refresh.",
                    color = Color(0xFFF5B544),
                )
                Spacer(Modifier.height(8.dp))
                Button(onClick = viewModel::refresh, enabled = !loading) { Text("Retry") }
            }
        }
    }
}
