package com.dllm.mesh.ui

import android.app.Application
import android.widget.Toast
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.dllm.mesh.data.ChatLocalStore
import com.dllm.mesh.data.IdentityStore
import com.dllm.mesh.data.LocalChatMsg
import com.dllm.mesh.data.LocalChatSession
import com.dllm.mesh.net.ChatEvent
import com.dllm.mesh.net.ChatSession
import com.dllm.mesh.net.DllmApi
import com.dllm.mesh.net.PlanStage
import com.dllm.mesh.net.SseClient
import com.dllm.mesh.net.isNeedsUpgrade
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

private const val UPGRADE_TOAST = "coordinator needs upgrade"
private const val FLUSH_IDLE_MS = 2000L

data class ChatMessage(val role: String, val text: String, val committed: Boolean = true)

private fun LocalChatSession.toChatSession(): ChatSession = ChatSession(
    id = id,
    title = title,
    model = model,
    tokensOut = tokensOut,
    lastTokenAt = lastTokenAt,
    createdAt = createdAt,
)

class ChatViewModel(application: Application) : AndroidViewModel(application) {

    private val store = IdentityStore(application)
    private val local = ChatLocalStore(application)
    private val sse = SseClient()

    val coordinatorUrl: StateFlow<String> = store.coordinatorUrl
        .stateIn(viewModelScope, SharingStarted.Eagerly, IdentityStore.DEFAULT_COORDINATOR_URL)
    val sessionId: StateFlow<String> = store.sessionId
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")
    val lastEventId: StateFlow<String> = store.lastEventId
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")

    // Local-only history: the burger drawer lists these rows. No server
    // GET /v1/sessions merge — phone and web converge via the per-session
    // SSE replay from 0 instead.
    private val _sessions = MutableStateFlow<List<ChatSession>>(emptyList())
    val sessions: StateFlow<List<ChatSession>> = _sessions.asStateFlow()

    private val _sessionsKnown = MutableStateFlow(false)
    val sessionsKnown: StateFlow<Boolean> = _sessionsKnown.asStateFlow()

    private val _needsUpgrade = MutableStateFlow(false)
    val needsUpgrade: StateFlow<Boolean> = _needsUpgrade.asStateFlow()

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _streaming = MutableStateFlow("")
    val streaming: StateFlow<String> = _streaming.asStateFlow()

    private val _streamActive = MutableStateFlow(false)
    val streamActive: StateFlow<Boolean> = _streamActive.asStateFlow()

    private val _status = MutableStateFlow("Loading chats…")
    val status: StateFlow<String> = _status.asStateFlow()

    private val _planStages = MutableStateFlow<List<PlanStage>>(emptyList())
    val planStages: StateFlow<List<PlanStage>> = _planStages.asStateFlow()

    private val _planKnown = MutableStateFlow(false)
    val planKnown: StateFlow<Boolean> = _planKnown.asStateFlow()

    private val _toast = MutableStateFlow<String?>(null)
    val toast: StateFlow<String?> = _toast.asStateFlow()

    private var currentId: String = ""
    private var streamJob: Job? = null
    private var flushJob: Job? = null

    init {
        viewModelScope.launch { store.ensureNodeId() }
        // Local index is the source of truth for the drawer. Reload when the
        // device id is known; auto-open the stored (or first) chat once.
        viewModelScope.launch {
            store.nodeId.collect { node ->
                if (node.isNotBlank()) {
                    refreshLocal(autoOpen = currentId.isBlank())
                }
            }
        }
        viewModelScope.launch {
            store.coordinatorUrl.collect { refreshPlan() }
        }
    }

    fun clearToast() {
        _toast.value = null
    }

    private fun toast(msg: String) {
        _toast.value = msg
    }

    private fun noteUpgradeOnce() {
        if (!_needsUpgrade.value) {
            _needsUpgrade.value = true
            toast(UPGRADE_TOAST)
        }
    }

    private suspend fun nodeId(): String = store.nodeId.first().ifBlank { store.ensureNodeId() }

    /** Reload the drawer from the local index. */
    fun refreshLocal(autoOpen: Boolean = false) {
        viewModelScope.launch {
            val node = nodeId()
            val rows = local.listSessions(node).map { it.toChatSession() }
            _sessions.value = rows
            _sessionsKnown.value = true
            if (rows.isEmpty()) {
                _status.value = "No chats yet — start a New chat."
            }
            if (autoOpen && currentId.isBlank()) {
                val stored = store.sessionId.first()
                val target = rows.firstOrNull { it.id == stored }?.id
                    ?: rows.firstOrNull()?.id
                if (target != null) {
                    selectSession(target)
                }
            }
        }
    }

    fun refreshPlan() {
        viewModelScope.launch {
            runCatching {
                val base = store.coordinatorUrl.first()
                // Prefer usage.plan (same source the Usage tab shows), else /v1/plan.
                // Group-scoped when this phone joined one, with device identity
                // so member/admin views work; 403 totals-only still carries plan.
                val group = store.groupId.first().ifBlank { null }
                val deviceId = nodeId()
                val usageStages = runCatching { DllmApi.getUsage(base, group, deviceId).planStages }
                    .recoverCatching { e ->
                        (e as? com.dllm.mesh.net.UsageForbiddenException)?.summary?.planStages
                            ?: throw e
                    }.getOrNull()
                if (!usageStages.isNullOrEmpty()) usageStages
                else DllmApi.getPlan(base)
            }.onSuccess { stages ->
                _planStages.value = stages
                _planKnown.value = true
            }.onFailure { e ->
                if (e is CancellationException) throw e
                _planKnown.value = false
                if (isNeedsUpgrade(e)) noteUpgradeOnce()
            }
        }
    }

    fun newChat() {
        viewModelScope.launch {
            persistSnapshot()
            flushStreamingNow()
            val base = store.coordinatorUrl.first()
            val node = nodeId()
            val sid = runCatching { sse.createSession(base) }.getOrNull()
            if (sid.isNullOrBlank()) {
                // Offline: local-only row, migrated to the server id on send.
                val lid = ChatLocalStore.newLocalId()
                currentId = lid
                store.setSessionId(lid)
                store.setLastEventId("")
                local.upsertSession(node, LocalChatSession(id = lid, title = "New chat"))
                _messages.value = emptyList()
                _streaming.value = ""
                _status.value = "Offline — this chat stays on this phone until you send."
                refreshLocal()
                return@launch
            }
            currentId = sid
            store.setSessionId(sid)
            store.setLastEventId("")
            local.upsertSession(node, LocalChatSession(id = sid, title = "New chat"))
            _messages.value = emptyList()
            _streaming.value = ""
            _status.value = "New chat $sid — streaming…"
            refreshLocal()
            restartStream()
        }
    }

    fun selectSession(id: String) {
        if (id.isBlank() || id == currentId) return
        viewModelScope.launch {
            persistSnapshot()
            flushStreamingNow()
            flushJob?.cancel()
            currentId = id
            store.setSessionId(id)
            store.setLastEventId("")
            // Instant paint from the local snapshot; the SSE replay from 0
            // rebuilds authoritative turns when online (cache cleared on the
            // first replayed event, kept as-is when offline).
            _messages.value = local.loadMessages(nodeId(), id)
                .map { ChatMessage(it.role, it.text) }
            _streaming.value = ""
            _status.value = "Loading history for $id…"
            restartStream()
        }
    }

    fun renameSession(id: String, title: String) {
        val clean = title.trim()
        if (clean.isEmpty()) {
            toast("Title cannot be empty.")
            return
        }
        if (clean.length > 80) {
            toast("Title must be 80 chars or fewer.")
            return
        }
        viewModelScope.launch {
            // Local first (drawer updates instantly), server best-effort.
            local.renameSession(nodeId(), id, clean)
            refreshLocal()
            if (!id.startsWith("local-")) {
                runCatching {
                    DllmApi.renameSession(store.coordinatorUrl.first(), id, clean)
                }.onFailure { e ->
                    if (e is CancellationException) throw e
                    if (isNeedsUpgrade(e)) noteUpgradeOnce()
                    else _status.value = "Server rename skipped: ${e.message} (kept locally)."
                }
            }
        }
    }

    fun deleteSession(id: String) {
        viewModelScope.launch {
            local.deleteSession(nodeId(), id)
            if (id == currentId) {
                flushJob?.cancel()
                streamJob?.cancel()
                _streamActive.value = false
                currentId = ""
                store.setSessionId("")
                store.setLastEventId("")
                _messages.value = emptyList()
                _streaming.value = ""
            }
            refreshLocal()
            if (!id.startsWith("local-")) {
                runCatching {
                    DllmApi.deleteSession(store.coordinatorUrl.first(), id)
                }.onFailure { e ->
                    if (e is CancellationException) throw e
                    if (isNeedsUpgrade(e)) noteUpgradeOnce()
                    else _status.value = "Server delete skipped: ${e.message} (removed locally)."
                }
            }
        }
    }

    fun send(text: String) {
        val body = text.trim()
        if (body.isEmpty()) return
        viewModelScope.launch {
            val base = store.coordinatorUrl.first()
            val node = nodeId()
            var sid = currentId.ifBlank { store.sessionId.first() }
            try {
                if (sid.isBlank() || sid.startsWith("local-")) {
                    val fresh = sse.createSession(base)
                    if (sid.startsWith("local-")) {
                        local.migrateSession(node, sid, fresh)
                    } else {
                        local.upsertSession(node, LocalChatSession(id = fresh, title = "New chat"))
                    }
                    sid = fresh
                    currentId = fresh
                    store.setSessionId(fresh)
                    store.setLastEventId("")
                    _messages.value = emptyList()
                    _streaming.value = ""
                    refreshLocal()
                    restartStream()
                }
                // Server-side title parity: first user turn names the chat.
                val known = local.listSessions(node).firstOrNull { it.id == sid }
                if (known != null && known.title == "New chat") {
                    local.renameSession(node, sid, body.take(40))
                    refreshLocal()
                }
                // A new user turn ends any pending assistant turn.
                flushStreamingNow()
                flushJob?.cancel()
                _messages.value = _messages.value + ChatMessage("user", body)
                persistSnapshot()
                sse.postMessage(base, sid, body)
                if (streamJob?.isActive != true) restartStream()
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                if (isNeedsUpgrade(e)) {
                    noteUpgradeOnce()
                    _status.value = "Send failed — coordinator needs upgrade."
                } else {
                    // Stay local: keep the turn on this phone so nothing is lost.
                    if (sid.startsWith("local-") || currentId.startsWith("local-")) {
                        _status.value = "Offline — kept on this phone, will sync on send."
                    } else {
                        _status.value = "Send failed: ${e.message} — check coordinator address."
                    }
                }
            }
        }
    }

    /** Stop button: cancel the SSE tail, tell the coordinator, keep the partial turn. */
    fun stopGeneration() {
        streamJob?.cancel()
        streamJob = null
        _streamActive.value = false
        viewModelScope.launch {
            val sid = currentId.ifBlank { store.sessionId.first() }
            if (sid.isNotBlank() && !sid.startsWith("local-")) {
                runCatching {
                    DllmApi.stopSession(store.coordinatorUrl.first(), sid)
                }
            }
            flushJob?.cancel()
            flushStreamingNow()
            _status.value = "Stopped."
        }
    }

    fun restartStream() {
        streamJob?.cancel()
        streamJob = viewModelScope.launch {
            val base = store.coordinatorUrl.first()
            val sid = currentId.ifBlank { store.sessionId.first() }
            if (sid.isBlank()) {
                _status.value = "No session — start a New chat."
                return@launch
            }
            if (sid.startsWith("local-")) {
                _status.value = "Local-only chat — send to sync it to the coordinator."
                return@launch
            }
            currentId = sid
            // Full replay from 0: the events log is both the history source
            // and the live tail for our own sessions.
            _status.value = "Streaming session $sid…"
            _streamActive.value = true
            try {
                var resetDone = false
                sse.events(base, sid, null).collect { event ->
                    // First replayed event wins over the local snapshot paint —
                    // avoids doubling cached turns with replayed ones.
                    if (!resetDone) {
                        resetDone = true
                        _messages.value = emptyList()
                        _streaming.value = ""
                    }
                    when (event) {
                        is ChatEvent.Token -> {
                            _streaming.value += event.text
                            event.eventId?.let { store.setLastEventId(it) }
                            scheduleFlush()
                        }
                        is ChatEvent.Committed -> {
                            // Durability mark only — NOT a turn boundary.
                            // Flushing here is the per-token-bubble bug: the
                            // server commits every token, so each word would
                            // become its own bubble. Turns end on the next
                            // user message or the idle timeout instead.
                            event.eventId?.let { store.setLastEventId(it) }
                            _status.value = "Committed #${event.position}"
                        }
                        is ChatEvent.Status -> {
                            event.eventId?.let { store.setLastEventId(it) }
                            val data = event.message
                            val userText = SseClient.extractUserText(data)
                            if (userText != null) {
                                handleRemoteUserMessage(userText)
                            } else if (data.startsWith("done:")
                                || data.startsWith("complete")
                                || data.startsWith("completed:")
                            ) {
                                flushStreamingNow()
                            } else if (!isSessionCreatedNoise(data)) {
                                if (data.isNotBlank()) _status.value = data.take(300)
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                _status.value = "Stream error: ${e.message} — tap the chat again to reload."
            } finally {
                _streamActive.value = false
            }
        }
    }

    private fun isSessionCreatedNoise(data: String): Boolean {
        val t = data.trim()
        // session_created broadcasts as status {"model":"..."} — history
        // boundary noise, not a user-visible status.
        return t.startsWith("{\"model\"") || t.startsWith("{\"model\":")
    }

    private fun handleRemoteUserMessage(text: String) {
        val clean = text.trim()
        if (clean.isEmpty()) return
        // The previous assistant turn (if any) ends where this user turn starts.
        flushJob?.cancel()
        flushStreamingNow()
        val last = _messages.value.lastOrNull()
        if (last?.role == "user" && last.text == clean) return // own echo; dedup
        _messages.value = _messages.value + ChatMessage("user", clean)
        persistSnapshot()
    }

    private fun scheduleFlush() {
        flushJob?.cancel()
        flushJob = viewModelScope.launch {
            delay(FLUSH_IDLE_MS)
            flushStreamingNow()
        }
    }

    private fun flushStreamingNow() {
        val partial = _streaming.value
        if (partial.isNotEmpty()) {
            _messages.value = _messages.value + ChatMessage("assistant", partial)
            _streaming.value = ""
            persistSnapshot()
        }
    }

    /** Best-effort snapshot of the open chat into the local store. */
    private fun persistSnapshot() {
        val sid = currentId
        if (sid.isBlank()) return
        viewModelScope.launch {
            val node = nodeId()
            val msgs = _messages.value.map { LocalChatMsg(it.role, it.text) }.toMutableList()
            if (_streaming.value.isNotEmpty()) {
                msgs.add(LocalChatMsg("assistant", _streaming.value))
            }
            // Ensure the index row exists even for raced paths.
            val known = local.listSessions(node).firstOrNull { it.id == sid }
            if (known == null) {
                local.upsertSession(node, LocalChatSession(id = sid, title = "New chat"))
                refreshLocal()
            }
            local.saveMessages(node, sid, msgs)
        }
    }
}

/** Compresses expanded layer lists into "0–27" style ranges. */
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

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    viewModel: ChatViewModel = viewModel(),
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val sessions by viewModel.sessions.collectAsState()
    val sessionsKnown by viewModel.sessionsKnown.collectAsState()
    val needsUpgrade by viewModel.needsUpgrade.collectAsState()
    val messages by viewModel.messages.collectAsState()
    val streaming by viewModel.streaming.collectAsState()
    val streamActive by viewModel.streamActive.collectAsState()
    val status by viewModel.status.collectAsState()
    val sessionId by viewModel.sessionId.collectAsState()
    val planStages by viewModel.planStages.collectAsState()
    val planKnown by viewModel.planKnown.collectAsState()
    val toastMsg by viewModel.toast.collectAsState()
    var input by remember { mutableStateOf("") }
    var menuFor by remember { mutableStateOf<String?>(null) }
    var renameFor by remember { mutableStateOf<ChatSession?>(null) }
    var renameText by remember { mutableStateOf("") }
    var renameError by remember { mutableStateOf("") }
    var deleteFor by remember { mutableStateOf<ChatSession?>(null) }
    var usageFor by remember { mutableStateOf<ChatSession?>(null) }
    var splitExpanded by remember { mutableStateOf(false) }
    val listState = rememberLazyListState()
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val drawerScope = rememberCoroutineScope()

    LaunchedEffect(toastMsg) {
        if (toastMsg != null) {
            Toast.makeText(context, toastMsg, Toast.LENGTH_LONG).show()
            viewModel.clearToast()
        }
    }

    LaunchedEffect(messages.size, streaming) {
        val total = messages.size + if (streaming.isNotEmpty()) 1 else 0
        if (total > 0) listState.animateScrollToItem(maxOf(0, total - 1))
    }

    // Group plan stages by device for the split row.
    val layersByDevice: Map<String, List<Int>> = remember(planStages) {
        val map = LinkedHashMap<String, MutableList<Int>>()
        for (st in planStages) {
            val arr = map.getOrPut(st.deviceId) { ArrayList() }
            for (l in st.layerStart..st.layerEnd) arr.add(l)
        }
        map.mapValues { it.value.distinct().sorted() }
    }
    val singleDeviceFastPath = layersByDevice.size == 1 &&
        layersByDevice.values.firstOrNull()?.size == 28 &&
        layersByDevice.values.firstOrNull()?.firstOrNull() == 0 &&
        layersByDevice.values.firstOrNull()?.lastOrNull() == 27

    val currentTitle = sessions.firstOrNull { it.id == sessionId }?.title ?: "Chat"

    // Local-only history lives behind the burger — no visible session list
    // in the main column anymore.
    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet {
                Column(Modifier.fillMaxWidth().padding(16.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            if (sessionsKnown) "Chats (${sessions.size})" else "Chats",
                            color = Color(0xFFE8EDF2),
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(onClick = { viewModel.refreshLocal() }) {
                                Text("Refresh")
                            }
                            Button(onClick = {
                                viewModel.newChat()
                                drawerScope.launch { drawerState.close() }
                            }) { Text("New chat") }
                        }
                    }
                    Spacer(Modifier.height(4.dp))
                    Text("Stored on this phone only.", color = Color(0xFF93A1B0))
                    if (needsUpgrade) {
                        Spacer(Modifier.height(4.dp))
                        Text(
                            "Server sync unavailable — coordinator needs upgrade.",
                            color = Color(0xFFF5B544),
                        )
                    }
                    Spacer(Modifier.height(8.dp))
                    LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        items(sessions, key = { it.id }) { s ->
                            val selected = s.id == sessionId
                            NavigationDrawerItem(
                                label = {
                                    Column {
                                        Text(s.title, color = Color(0xFFE8EDF2))
                                        Text(
                                            "${s.model} · ${s.tokensOut} out" +
                                                (s.lastTokenAt?.let { " · $it" } ?: ""),
                                            color = Color(0xFF93A1B0),
                                        )
                                    }
                                },
                                badge = {
                                    IconButton(onClick = { menuFor = s.id }) {
                                        Icon(
                                            Icons.Filled.MoreVert,
                                            contentDescription = "Chat options",
                                            tint = Color(0xFF93A1B0),
                                        )
                                    }
                                },
                                selected = selected,
                                onClick = {
                                    viewModel.selectSession(s.id)
                                    drawerScope.launch { drawerState.close() }
                                },
                            )
                            DropdownMenu(
                                expanded = menuFor == s.id,
                                onDismissRequest = { menuFor = null },
                            ) {
                                DropdownMenuItem(
                                    text = { Text("Rename") },
                                    onClick = {
                                        menuFor = null
                                        renameText = s.title
                                        renameError = ""
                                        renameFor = s
                                    },
                                )
                                DropdownMenuItem(
                                    text = { Text("Delete") },
                                    onClick = {
                                        menuFor = null
                                        deleteFor = s
                                    },
                                )
                                DropdownMenuItem(
                                    text = { Text("Usage") },
                                    onClick = {
                                        menuFor = null
                                        usageFor = s
                                    },
                                )
                            }
                        }
                    }
                    if (sessionsKnown && sessions.isEmpty()) {
                        Text("No chats yet — start a New chat.", color = Color(0xFF93A1B0))
                    }
                }
            }
        },
        modifier = modifier,
    ) {
        Scaffold(
            containerColor = Color(0xFF101418),
            topBar = {
                TopAppBar(
                    title = { Text(currentTitle) },
                    navigationIcon = {
                        IconButton(onClick = { drawerScope.launch { drawerState.open() } }) {
                            Icon(
                                Icons.Filled.Menu,
                                contentDescription = "Chat history",
                                tint = Color(0xFFE8EDF2),
                            )
                        }
                    },
                )
            },
        ) { padding ->
            Column(Modifier.fillMaxSize().padding(padding).padding(16.dp)) {
                // Layer toggle: truthful per-device ranges, never invented.
                TextButton(onClick = { splitExpanded = !splitExpanded }) {
                    Text("Model split ${if (splitExpanded) "▴" else "▾"}", color = Color(0xFF2DD4BF))
                }
                if (splitExpanded) {
                    Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF171D24))) {
                        Column(Modifier.fillMaxWidth().padding(12.dp)) {
                            if (!planKnown) {
                                Text(
                                    "Model split unavailable — coordinator needs upgrade.",
                                    color = Color(0xFFF5B544),
                                )
                            } else if (layersByDevice.isEmpty()) {
                                Text("No layer assignment reported.", color = Color(0xFF93A1B0))
                            } else {
                                for ((dev, layers) in layersByDevice) {
                                    Text(
                                        "$dev: layers ${layerRangeText(layers)} (${layers.size} layers)",
                                        color = Color(0xFFE8EDF2),
                                    )
                                }
                                if (singleDeviceFastPath) {
                                    Spacer(Modifier.height(4.dp))
                                    Text(
                                        "All 28 layers on this coordinator — single-device fast path.",
                                        color = Color(0xFF93A1B0),
                                    )
                                }
                            }
                        }
                    }
                    Spacer(Modifier.height(8.dp))
                }

                Text(status, color = Color(0xFF93A1B0))
                Spacer(Modifier.height(8.dp))

                LazyColumn(
                    state = listState,
                    modifier = Modifier.weight(1f).fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    items(messages) { msg ->
                        Card(
                            colors = CardDefaults.cardColors(
                                containerColor = if (msg.role == "user") Color(0xFF171D24) else Color(0xFF1B2B28),
                            ),
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Text(
                                text = "${msg.role}: ${msg.text}",
                                modifier = Modifier.padding(12.dp),
                                color = Color(0xFFE8EDF2),
                            )
                        }
                    }
                    if (streaming.isNotEmpty()) {
                        item {
                            Card(
                                colors = CardDefaults.cardColors(containerColor = Color(0xFF1B2B28)),
                                modifier = Modifier.fillMaxWidth(),
                            ) {
                                Text(
                                    text = "assistant: $streaming",
                                    modifier = Modifier.padding(12.dp),
                                    color = Color(0xFF2DD4BF),
                                )
                            }
                        }
                    }
                }

                Spacer(Modifier.height(8.dp))
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(
                        value = input,
                        onValueChange = { input = it },
                        label = { Text("Message") },
                        modifier = Modifier.weight(1f),
                    )
                    if (streamActive || streaming.isNotEmpty()) {
                        OutlinedButton(onClick = viewModel::stopGeneration) { Text("Stop") }
                    }
                    Button(onClick = {
                        viewModel.send(input)
                        input = ""
                    }) { Text("Send") }
                }
                Spacer(Modifier.width(0.dp))
            }
        }
    }

    // Rename dialog -> local rename + best-effort server rename.
    renameFor?.let { s ->
        AlertDialog(
            onDismissRequest = { renameFor = null },
            title = { Text("Rename chat") },
            text = {
                Column {
                    OutlinedTextField(
                        value = renameText,
                        onValueChange = {
                            renameText = it
                            renameError = ""
                        },
                        label = { Text("Title (1–80 chars)") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth(),
                    )
                    if (renameError.isNotEmpty()) {
                        Spacer(Modifier.height(4.dp))
                        Text(renameError, color = Color(0xFFF5B544))
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    val clean = renameText.trim()
                    if (clean.isEmpty()) {
                        renameError = "Title cannot be empty."
                        return@TextButton
                    }
                    if (clean.length > 80) {
                        renameError = "Title must be 80 chars or fewer."
                        return@TextButton
                    }
                    viewModel.renameSession(s.id, clean)
                    renameFor = null
                }) { Text("Rename") }
            },
            dismissButton = {
                TextButton(onClick = { renameFor = null }) { Text("Cancel") }
            },
        )
    }

    // Delete confirm -> local delete + best-effort server delete.
    deleteFor?.let { s ->
        AlertDialog(
            onDismissRequest = { deleteFor = null },
            title = { Text("Delete chat?") },
            text = { Text("Delete \"${s.title}\"? This cannot be undone.") },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.deleteSession(s.id)
                    deleteFor = null
                }) { Text("Delete") }
            },
            dismissButton = {
                TextButton(onClick = { deleteFor = null }) { Text("Cancel") }
            },
        )
    }

    // Per-chat Usage: real fields from the list entry only.
    usageFor?.let { s ->
        AlertDialog(
            onDismissRequest = { usageFor = null },
            title = { Text(s.title) },
            text = {
                Text(
                    "Model: ${s.model}\n" +
                        "Tokens out: ${s.tokensOut}\n" +
                        "Last activity: ${s.lastTokenAt ?: "—"}"
                )
            },
            confirmButton = {
                TextButton(onClick = { usageFor = null }) { Text("Close") }
            },
        )
    }
}
