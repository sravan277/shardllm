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
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.dllm.mesh.data.IdentityStore
import com.dllm.mesh.data.layerRangeText
import com.dllm.mesh.net.DllmApi
import com.dllm.mesh.net.MeshDevice
import com.dllm.mesh.net.NsdDiscovery
import com.dllm.mesh.net.Peer
import com.dllm.mesh.net.Presence
import com.dllm.mesh.net.baseUrlMatches
import com.dllm.mesh.net.isNeedsUpgrade
import com.dllm.mesh.worker.WorkerService
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
import java.time.Duration
import java.time.Instant
import java.time.format.DateTimeParseException
import kotlin.coroutines.cancellation.CancellationException

/**
 * The Devices tab: one mesh, one coordinator, a list of the devices paired to
 * it, plus everything this phone needs to be a useful member.
 *
 * WHY this tab owns all three concerns (worker toggle, presence, roster): they
 * are one decision from one screen. "Should this phone contribute compute?" only
 * means something next to "what else is in the mesh and what is it doing?" —
 * splitting the toggle into Settings meant the operator had to hold two screens
 * in their head to answer one question. It also means the networks feature is
 * fully gone: there is no group to select, so a device list has exactly one
 * source of truth (`GET /v1/devices`).
 *
 * Roster data is merged from three endpoints because the coordinator splits
 * authority across them: `/v1/devices` is the registry (identity, status,
 * liveness), `/v1/usage` carries load and the worker flag, and `/v1/plan`
 * carries the layer assignment. Anything missing is rendered as "not reported",
 * never as a zero or a dash-with-no-explanation.
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

    /**
     * Whether this phone contributes compute. Lives here (not Settings) because
     * the answer only makes sense next to the rest of the mesh — see the class
     * KDoc.
     */
    val workerEnabled: StateFlow<Boolean> = store.workerEnabled
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    private val _devices = MutableStateFlow<List<MeshDevice>>(emptyList())
    val devices: StateFlow<List<MeshDevice>> = _devices.asStateFlow()

    private val _devicesLoading = MutableStateFlow(false)
    val devicesLoading: StateFlow<Boolean> = _devicesLoading.asStateFlow()

    /** Set when the coordinator is too old to serve `GET /v1/devices` at all. */
    private val _needsUpgrade = MutableStateFlow(false)
    val needsUpgrade: StateFlow<Boolean> = _needsUpgrade.asStateFlow()

    private var refreshJob: Job? = null

    init {
        viewModelScope.launch { store.ensureNodeId() }
        refreshDevices()
        viewModelScope.launch {
            val id = store.ensureNodeId()
            // Advertise so a laptop browsing _dllm._tcp. sees this phone at all.
            advertise(id)
        }
    }

    /**
     * Start/stop the compute-worker foreground service and persist the choice.
     *
     * WHY the persisted flag alone is not enough: the service must actually run
     * for the phone to take pipeline layers, and the heartbeat role switches on
     * this same flag. Persisting without starting the service would advertise
     * `role=worker` on a phone doing no work — exactly the kind of dishonest
     * "worker active" the rest of this screen is careful to avoid.
     */
    fun setWorkerEnabled(enabled: Boolean) {
        viewModelScope.launch {
            store.setWorkerEnabled(enabled)
            if (enabled) WorkerService.start(app) else WorkerService.stop(app)
            _status.value = if (enabled) {
                "Worker on — heartbeat role=worker with load + capabilities."
            } else {
                "Worker off — heartbeat role=client."
            }
            // The roster is a report of other devices' claims; re-read it so the
            // capability/plan view reflects the role change this phone just made.
            refreshDevices()
        }
    }

    fun startDiscovery() {
        discovery.startDiscovery()
        _status.value = "Discovering ${NsdDiscovery.SERVICE_TYPE} …"
    }

    fun stopDiscovery() {
        discovery.stopDiscovery()
    }

    override fun onCleared() {
        presenceJob?.cancel()
        presenceJob = null
        refreshJob?.cancel()
        discovery.stopDiscovery()
        discovery.unregisterService()
    }

    /** Attach = save the peer as the active coordinator. No heartbeat yet. */
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
     * Connect via a LAN peer: adopt it as the server, then announce this phone
     * there and start the heartbeat loop. One tap, because a discovered peer is
     * by definition reachable — making the user "attach" and then separately
     * "connect" was pure ceremony.
     */
    fun connectViaPeer(peer: Peer) {
        viewModelScope.launch {
            val base = "http://${peer.host}:${peer.port}"
            store.ensureNodeId()
            store.setCoordinatorUrl(base)
            _status.value = "Attached to $base (${peer.serviceName})."
            connectPresence()
            refreshDevices()
        }
    }

    /**
     * Re-read the mesh roster. Each source is best-effort: a missing `/v1/usage`
     * must not blank out the identity and status that `/v1/devices` gave us,
     * because those two together are enough to answer "is everyone still here?".
     */
    fun refreshDevices() {
        refreshJob?.cancel()
        refreshJob = viewModelScope.launch {
            _devicesLoading.value = true
            val base = store.coordinatorUrl.first().trimEnd('/')
            val self = store.nodeId.first()

            val registry = runCatching { DllmApi.listDevices(base) }
            registry.onFailure { e ->
                if (e is CancellationException) throw e
                _devices.value = emptyList()
                _needsUpgrade.value = isNeedsUpgrade(e)
                _status.value = "Device list unavailable: ${Presence.friendlyCause(e)}"
                _devicesLoading.value = false
                return@launch
            }
            _needsUpgrade.value = false

            // Load + worker flags: the registry intentionally reports none.
            val loadById = runCatching { DllmApi.getUsage(base, null).perDevice }
                .getOrElse { emptyList() }
                .associateBy { it.deviceId }

            // Assigned layers come from the planner, keyed by device.
            val assignedById = runCatching { DllmApi.getPlan(base) }
                .getOrElse { emptyList() }
                .groupBy { it.deviceId }

            _devices.value = registry.getOrThrow().map { row ->
                val load = loadById[row.deviceId]
                val assigned = assignedById[row.deviceId]
                    ?.flatMap { stage -> stage.layerStart..stage.layerEnd }
                row.copy(
                    cpuPct = load?.cpuPct,
                    memPct = load?.memPct,
                    loadSource = load?.loadSource ?: "none",
                    workerActive = load?.workerActive,
                    assignedLayers = assigned?.distinct()?.sorted(),
                    offeredLayers = load?.layers,
                    // The registry carries no "is this me" flag; the saved
                    // node_id is the only honest source for it.
                    isSelf = !self.isBlank() && row.deviceId == self,
                )
            }
            _devicesLoading.value = false
        }
    }

    /**
     * Revoke a device: the coordinator stops it contributing and the Devices
     * list shows it as revoked rather than quietly dropping the row. Errors are
     * surfaced verbatim — a 400 here is the server protecting the mesh, not a
     * client bug, and hiding it would leave the operator thinking it worked.
     */
    fun revokeDevice(device: MeshDevice) = deviceAction(device, "revoke") {
        DllmApi.revokeDevice(it, device.deviceId)
    }

    /** Re-admit a revoked device. */
    fun approveDevice(device: MeshDevice) = deviceAction(device, "approve") {
        DllmApi.approveDevice(it, device.deviceId)
    }

    /**
     * Hard-delete a device row (stale duplicates and drill junk).
     *
     * The coordinator refuses to delete its own row with HTTP 400 "never orphan
     * the mesh" — that refusal is passed through to the operator instead of
     * being retried or hidden.
     */
    fun forgetDevice(device: MeshDevice) = deviceAction(device, "forget") {
        DllmApi.deleteDevice(it, device.deviceId)
    }

    private fun deviceAction(
        device: MeshDevice,
        verb: String,
        call: suspend (String) -> Unit,
    ) {
        viewModelScope.launch {
            runCatching { call(store.coordinatorUrl.first().trimEnd('/')) }
                .onSuccess {
                    _status.value = "$verb ${shortDeviceId(device.deviceId)}: ok."
                    refreshDevices()
                }
                .onFailure { e ->
                    if (e is CancellationException) throw e
                    _status.value = "$verb ${shortDeviceId(device.deviceId)} failed — ${e.message ?: e::class.simpleName}"
                }
        }
    }

    /**
     * Advertise this phone on the LAN. Re-advertises whenever the worker role
     * changes so a peer browsing mDNS sees an accurate `role`.
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
}

/** node_id advertised in a peer's mDNS TXT record (server sends lowercase keys). */
fun peerNodeId(peer: Peer): String =
    peer.txtMap.entries.firstOrNull { it.key.equals("node_id", ignoreCase = true) }?.value.orEmpty()

/**
 * Compact device id for a row header: `9f2c1ab4…7d21`. Long enough to
 * recognise a phone at a glance, short enough that the full id still fits
 * underneath on its own line for copy/paste.
 */
fun shortDeviceId(id: String): String =
    if (id.length > 20) "${id.take(9)}…${id.takeLast(4)}" else id

/**
 * Human "last seen" text from the coordinator's ISO-8601 UTC `last_seen`.
 *
 * Returns null when the timestamp is absent or unparseable — the caller then
 * says "never" rather than inventing a duration. Relative time is what an
 * operator actually needs here ("12s ago" vs a wall-clock string they must
 * subtract from the current time in their head).
 */
fun relativeLastSeen(iso: String?, nowMs: Long = System.currentTimeMillis()): String? {
    val raw = iso?.trim().orEmpty()
    if (raw.isEmpty()) return null
    val instant = try {
        Instant.parse(raw)
    } catch (_: DateTimeParseException) {
        return null
    }
    val seconds = Duration.between(instant, Instant.ofEpochMilli(nowMs)).seconds
    return when {
        seconds < 0 -> "just now"
        seconds < 60 -> "${seconds}s ago"
        seconds < 3_600 -> "${seconds / 60}m ago"
        seconds < 86_400 -> "${seconds / 3_600}h ago"
        else -> "${seconds / 86_400}d ago"
    }
}

@Composable
fun DevicesScreen(
    viewModel: DevicesViewModel = viewModel(),
    modifier: Modifier = Modifier,
) {
    val devices by viewModel.devices.collectAsState()
    val devicesLoading by viewModel.devicesLoading.collectAsState()
    val needsUpgrade by viewModel.needsUpgrade.collectAsState()

    Column(
        modifier = modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState()),
    ) {
        WorkerToggleCard(viewModel)
        Spacer(Modifier.height(12.dp))

        PresenceCard(viewModel)
        Spacer(Modifier.height(16.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Mesh devices (${devices.size})", color = MeshColors.Text)
            OutlinedButton(onClick = viewModel::refreshDevices, enabled = !devicesLoading) {
                Text(if (devicesLoading) "Loading…" else "Refresh")
            }
        }
        Spacer(Modifier.height(4.dp))

        when {
            needsUpgrade -> {
                Text(
                    "This coordinator is too old to serve the device list — upgrade it. " +
                        "Nothing here is guessed.",
                    color = MeshColors.Amber,
                )
            }
            devicesLoading && devices.isEmpty() -> {
                Text("Loading devices…", color = MeshColors.Muted)
            }
            devices.isEmpty() -> {
                Text(
                    "No devices paired yet. This phone appears as soon as it connects.",
                    color = MeshColors.Muted,
                )
            }
            else -> {
                devices.forEach { device ->
                    MeshDeviceCard(
                        device = device,
                        onRevoke = { viewModel.revokeDevice(device) },
                        onApprove = { viewModel.approveDevice(device) },
                        onForget = { viewModel.forgetDevice(device) },
                    )
                    Spacer(Modifier.height(8.dp))
                }
            }
        }
    }
}

/**
 * "This device contributes compute" — the single home of the worker toggle.
 *
 * The live indicator requires [WorkerService.isLive], i.e. a heartbeat inside
 * the coordinator's own 90s window. A plain `running && ready` read would keep
 * showing a green dot after the OS killed the service, because those stats live
 * in a static field.
 */
@Composable
private fun WorkerToggleCard(viewModel: DevicesViewModel) {
    val workerEnabled by viewModel.workerEnabled.collectAsState()
    val myNodeId by viewModel.myNodeId.collectAsState()
    val workerStats by WorkerService.stats.collectAsState()
    val ctx = LocalContext.current

    val live = WorkerService.isLive(workerStats)
    val age = WorkerService.heartbeatAgeSeconds(workerStats)

    Card(colors = CardDefaults.cardColors(containerColor = MeshColors.Panel)) {
        Column(Modifier.fillMaxWidth().padding(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(Modifier.weight(1f)) {
                    Text("This device contributes compute", color = MeshColors.Text)
                    Text(
                        if (workerEnabled) "Offered as a worker to the planner."
                        else "Chat only — no pipeline layers.",
                        color = MeshColors.Muted,
                    )
                }
                Switch(checked = workerEnabled, onCheckedChange = viewModel::setWorkerEnabled)
            }
            Spacer(Modifier.height(4.dp))
            Text(
                "${if (live) "●" else "○"} ${workerStats.status}",
                color = if (live) MeshColors.Teal else MeshColors.Muted,
            )
            // Stale-by-timeout is reported explicitly: the operator needs to know
            // the difference between "switched off" and "supposed to be on but is
            // not answering".
            if (workerEnabled && !live && age != null) {
                Text(
                    "No worker heartbeat for ${age}s — the worker is not live on the " +
                        "coordinator even though the toggle is on.",
                    color = MeshColors.Amber,
                )
            }
            Spacer(Modifier.height(8.dp))
            SelectionContainer {
                Text(
                    "device_id: ${myNodeId.ifBlank { "(loading…)" }}",
                    color = MeshColors.Muted,
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
}

/**
 * One row of the mesh roster.
 *
 * Every field is rendered from a value the coordinator actually reported. A
 * missing value reads "not reporting" rather than a fabricated zero, because a
 * zero here would be indistinguishable from a genuinely idle device.
 */
@Composable
private fun MeshDeviceCard(
    device: MeshDevice,
    onRevoke: () -> Unit,
    onApprove: () -> Unit,
    onForget: () -> Unit,
) {
    val nowMs = nowMillisTicker()
    val lastSeen = relativeLastSeen(device.lastSeen, nowMs)
    // Load provenance decides the colour: teal only for a number the server
    // really has, grey for "not reporting".
    val hasLoad = (device.loadSource == "live" || device.loadSource == "reported") &&
        device.cpuPct != null && device.memPct != null

    Card(
        colors = CardDefaults.cardColors(
            containerColor = if (device.isSelf) MeshColors.PanelActive else MeshColors.Panel,
        ),
    ) {
        Column(Modifier.fillMaxWidth().padding(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    device.deviceName?.takeIf { it.isNotBlank() }
                        ?: shortDeviceId(device.deviceId),
                    color = MeshColors.Text,
                    modifier = Modifier.weight(1f),
                )
                Text(
                    when {
                        device.revoked -> "revoked"
                        device.active -> "active"
                        else -> "idle"
                    },
                    color = when {
                        device.revoked -> MeshColors.Amber
                        device.active -> MeshColors.Teal
                        else -> MeshColors.Muted
                    },
                )
            }
            Text(device.deviceId, color = MeshColors.Muted)
            Spacer(Modifier.height(4.dp))

            Text(
                buildString {
                    append("role: ").append(device.role ?: "unknown")
                    append(" · status: ").append(device.status ?: "unknown")
                    if (device.isSelf) append(" · this phone")
                },
                color = MeshColors.Muted,
            )
            Text(
                if (lastSeen != null) "last seen: $lastSeen" else "last seen: never reported",
                color = if (device.active) MeshColors.Teal else MeshColors.Muted,
            )
            Text(
                when (device.workerActive) {
                    true -> "worker active — takes pipeline layers"
                    false -> "worker idle — chat only"
                    null -> "worker state: not reporting"
                },
                color = if (device.workerActive == true) MeshColors.Teal else MeshColors.Muted,
            )
            Text(
                if (hasLoad) {
                    "CPU ${
                        "%.0f".format(device.cpuPct)
                    }% · MEM ${"%.0f".format(device.memPct)}% (${device.loadSource})"
                } else {
                    "CPU/MEM: not reporting"
                },
                color = if (hasLoad) MeshColors.Teal else MeshColors.Muted,
            )
            // Assigned beats offered: a plan assignment is a fact, an offer is
            // only a request, and conflating them would overstate the split.
            val layers = device.assignedLayers ?: device.offeredLayers
            Text(
                when {
                    !device.assignedLayers.isNullOrEmpty() ->
                        "layers ${layerRangeText(device.assignedLayers)} assigned"
                    !device.offeredLayers.isNullOrEmpty() ->
                        "layers ${layerRangeText(device.offeredLayers)} offered · not assigned"
                    else -> "layers: not reported"
                },
                color = MeshColors.Muted,
            )

            Spacer(Modifier.height(4.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (device.revoked) {
                    TextButton(onClick = onApprove) { Text("Approve") }
                } else {
                    TextButton(onClick = onRevoke) { Text("Revoke") }
                }
                TextButton(onClick = onForget) { Text("Forget") }
            }
        }
    }
}

/**
 * "Now" for the roster's relative timestamps, re-read once a second so
 * "12s ago" does not freeze while the screen is open.
 */
@Composable
private fun nowMillisTicker(): Long {
    val now = remember { mutableStateOf(System.currentTimeMillis()) }
    LaunchedEffect(Unit) {
        while (true) {
            delay(1_000)
            now.value = System.currentTimeMillis()
        }
    }
    return now.value
}

/**
 * Reusable presence card: active server, nearby coordinators, this-phone
 * connect. Extracted so the connection story lives in exactly one place.
 *
 * Approve/revoke are intentionally NOT here — the roster above owns those,
 * because they act on a mesh row rather than on this phone's link.
 */
@Composable
fun PresenceCard(
    viewModel: DevicesViewModel = viewModel(),
    modifier: Modifier = Modifier,
) {
    val coordinatorUrl by viewModel.coordinatorUrl.collectAsState()
    val peers by viewModel.peers.collectAsState()
    val status by viewModel.status.collectAsState()
    val connected by viewModel.connected.collectAsState()
    val myNodeId by viewModel.myNodeId.collectAsState()
    val ctx = LocalContext.current

    Column(modifier = modifier) {
        Text("Active server", color = MeshColors.Text)
        Spacer(Modifier.height(4.dp))
        Card(colors = CardDefaults.cardColors(containerColor = MeshColors.PanelActive)) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(12.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    coordinatorUrl,
                    color = MeshColors.Teal,
                    modifier = Modifier.weight(1f),
                )
                Text("Active", color = MeshColors.Muted)
            }
        }
        Spacer(Modifier.height(8.dp))

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = viewModel::startDiscovery) { Text("Discover") }
            OutlinedButton(onClick = viewModel::stopDiscovery) { Text("Stop") }
        }
        Spacer(Modifier.height(8.dp))

        val coordinators = peers.filterNot { NsdDiscovery.isNodeAdvertisement(it.serviceName) }
        val nodes = peers.filter { NsdDiscovery.isNodeAdvertisement(it.serviceName) }

        Text("Coordinators on this LAN (${coordinators.size})", color = MeshColors.Text)
        Spacer(Modifier.height(4.dp))
        if (coordinators.isEmpty()) {
            Text(
                "None found yet — tap Discover while on the same WiFi as the laptop.",
                color = MeshColors.Muted,
            )
            Spacer(Modifier.height(8.dp))
        }
        coordinators.forEach { peer ->
            Card(colors = CardDefaults.cardColors(containerColor = MeshColors.Panel)) {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(12.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text(peer.serviceName, color = MeshColors.Text)
                        Text("${peer.host}:${peer.port}", color = MeshColors.Muted)
                    }
                    if (baseUrlMatches(coordinatorUrl, peer.host, peer.port)) {
                        Text("Active", color = MeshColors.Teal)
                    } else {
                        // One tap adopts and connects: a discovered coordinator is
                        // reachable by construction, so splitting attach and
                        // connect into two taps only added a way to forget one.
                        Button(onClick = { viewModel.connectViaPeer(peer) }) { Text("Connect") }
                    }
                }
            }
            Spacer(Modifier.height(8.dp))
        }

        // Other phones advertising themselves. Shown, but not connectable: they
        // do not serve a coordinator port, and pretending otherwise would send
        // the user to a dead endpoint.
        if (nodes.isNotEmpty()) {
            Text("Other phones on this LAN (${nodes.size})", color = MeshColors.Text)
            Spacer(Modifier.height(4.dp))
            nodes.forEach { peer ->
                Card(colors = CardDefaults.cardColors(containerColor = MeshColors.Panel)) {
                    Column(Modifier.fillMaxWidth().padding(12.dp)) {
                        Text(peer.serviceName, color = MeshColors.Text)
                        Text("${peer.host}:${peer.port}", color = MeshColors.Muted)
                        val role = peer.txtMap.entries
                            .firstOrNull { it.key.equals("role", ignoreCase = true) }?.value
                        Text(
                            "presence only — role: ${role ?: "unknown"}. " +
                                "It does not serve chat; pair to its coordinator instead.",
                            color = MeshColors.Muted,
                        )
                        val advertisedId = peerNodeId(peer)
                        if (advertisedId.isNotBlank()) {
                            Text("device_id: $advertisedId", color = MeshColors.Muted)
                        }
                    }
                }
                Spacer(Modifier.height(8.dp))
            }
        }

        Text("This phone", color = MeshColors.Text)
        Spacer(Modifier.height(4.dp))
        Card(colors = CardDefaults.cardColors(containerColor = MeshColors.PanelActive)) {
            Column(Modifier.fillMaxWidth().padding(12.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text(Build.MODEL, color = MeshColors.Text)
                        Text(
                            if (connected) "Connected · active on coordinator" else "Offline · idle on coordinator",
                            color = if (connected) MeshColors.Teal else MeshColors.Muted,
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
                        color = MeshColors.Muted,
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
        Text(status, color = MeshColors.Amber)
    }
}
