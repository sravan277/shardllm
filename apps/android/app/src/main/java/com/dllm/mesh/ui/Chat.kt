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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.dllm.mesh.data.IdentityStore
import com.dllm.mesh.net.ChatEvent
import com.dllm.mesh.net.SseClient
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class ChatMessage(val role: String, val text: String, val committed: Boolean = true)

class ChatViewModel(application: Application) : AndroidViewModel(application) {

    private val store = IdentityStore(application)
    private val sse = SseClient()

    val coordinatorUrl: StateFlow<String> = store.coordinatorUrl
        .stateIn(viewModelScope, SharingStarted.Eagerly, IdentityStore.DEFAULT_COORDINATOR_URL)
    val sessionId: StateFlow<String> = store.sessionId
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")
    val lastEventId: StateFlow<String> = store.lastEventId
        .stateIn(viewModelScope, SharingStarted.Eagerly, "")

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _streaming = MutableStateFlow("")
    val streaming: StateFlow<String> = _streaming.asStateFlow()

    private val _status = MutableStateFlow("Idle — attach to a session, then send.")
    val status: StateFlow<String> = _status.asStateFlow()

    private val _attachField = MutableStateFlow("")
    val attachField: StateFlow<String> = _attachField.asStateFlow()

    private var streamJob: Job? = null

    init {
        // Reconnect the SSE stream whenever the persisted session changes.
        viewModelScope.launch {
            store.sessionId.collect { id ->
                if (id.isNotBlank()) restartStream() else {
                    streamJob?.cancel()
                    _status.value = "No session — enter a session id or send to create one."
                }
            }
        }
        viewModelScope.launch { store.ensureNodeId() }
    }

    fun onAttachFieldChange(value: String) {
        _attachField.value = value
    }

    fun setCoordinatorUrl(url: String) {
        if (url.isBlank()) return
        viewModelScope.launch { store.setCoordinatorUrl(url) }
    }

    fun attach() {
        val id = _attachField.value.trim()
        if (id.isEmpty()) return
        viewModelScope.launch {
            store.setSessionId(id)
            store.setLastEventId("")
            _messages.value = emptyList()
            _streaming.value = ""
            _status.value = "Attached to $id — streaming…"
        }
    }

    fun send(text: String) {
        val body = text.trim()
        if (body.isEmpty()) return
        viewModelScope.launch {
            val base = store.coordinatorUrl.first()
            var sid = store.sessionId.first()
            try {
                if (sid.isBlank()) {
                    sid = sse.createSession(base)
                    store.setSessionId(sid)
                    store.setLastEventId("")
                }
                _messages.value = _messages.value + ChatMessage("user", body)
                sse.postMessage(base, sid, body)
            } catch (e: Exception) {
                _status.value = "Send failed: ${e.message} — check coordinator address."
            }
        }
    }

    fun restartStream() {
        streamJob?.cancel()
        streamJob = viewModelScope.launch {
            val base = store.coordinatorUrl.first()
            val sid = store.sessionId.first()
            if (sid.isBlank()) {
                _status.value = "No session — enter a session id or send to create one."
                return@launch
            }
            val resumeFrom = store.lastEventId.first().ifBlank { null }
            _status.value = "Streaming session $sid…"
            runCatching {
                sse.events(base, sid, resumeFrom).collect { event ->
                    when (event) {
                        is ChatEvent.Token -> {
                            _streaming.value += event.text
                            event.eventId?.let { store.setLastEventId(it) }
                        }
                        is ChatEvent.Committed -> {
                            flushStreaming()
                            event.eventId?.let { store.setLastEventId(it) }
                            _status.value = "Committed #${event.position}"
                        }
                        is ChatEvent.Status -> {
                            event.eventId?.let { store.setLastEventId(it) }
                            if (event.message.isNotBlank()) _status.value = event.message
                        }
                    }
                }
            }.onFailure { e ->
                _status.value = "Stream error: ${e.message} — retrying keeps lastEventId."
            }
        }
    }

    private fun flushStreaming() {
        val partial = _streaming.value
        if (partial.isNotEmpty()) {
            _messages.value = _messages.value + ChatMessage("assistant", partial)
            _streaming.value = ""
        }
    }
}

@Composable
fun ChatScreen(
    viewModel: ChatViewModel = viewModel(),
    modifier: Modifier = Modifier,
) {
    val messages by viewModel.messages.collectAsState()
    val streaming by viewModel.streaming.collectAsState()
    val status by viewModel.status.collectAsState()
    val sessionId by viewModel.sessionId.collectAsState()
    val attachField by viewModel.attachField.collectAsState()
    var input by remember { mutableStateOf("") }
    val listState = rememberLazyListState()

    LaunchedEffect(messages.size, streaming) {
        val total = messages.size + if (streaming.isNotEmpty()) 1 else 0
        if (total > 0) listState.animateScrollToItem(maxOf(0, total - 1))
    }

    Column(modifier = modifier.fillMaxSize().padding(16.dp)) {
        // Attach-to-session row (session persistence lives in DataStore).
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = attachField,
                onValueChange = viewModel::onAttachFieldChange,
                label = { Text("Attach to session_id") },
                placeholder = { Text(if (sessionId.isBlank()) "e.g. sess_abc123" else sessionId) },
                modifier = Modifier.weight(1f),
                singleLine = true,
            )
            Spacer(Modifier.width(0.dp))
            OutlinedButton(onClick = viewModel::attach, enabled = attachField.isNotBlank()) {
                Text("Attach")
            }
        }
        Spacer(Modifier.height(8.dp))
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
            Button(onClick = {
                viewModel.send(input)
                input = ""
            }) { Text("Send") }
        }
    }
}
