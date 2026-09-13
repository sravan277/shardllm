package com.dllm.mesh.worker

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.wifi.WifiManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File

/** Snapshot of worker health. Observed by UI via [WorkerService.stats] (no binding). */
data class WorkerStats(
    val running: Boolean = false,
    val modelLoaded: Boolean = false,
    val modelPath: String? = null,
    val tokensDecoded: Long = 0L,
    val lastHeartbeatEpochMs: Long = 0L,
    val ready: Boolean = false,
    val status: String = "Stopped.",
)

/**
 * Phase 4 worker: foreground service (type connectedDevice) + JNI stub.
 *
 * Intent contract (owned here; sibling Settings toggle calls [start]/[stop]):
 * - `com.dllm.mesh.action.START_WORKER` → foreground + probe local GGUF + heartbeat.
 * - `com.dllm.mesh.action.STOP_WORKER`  → stopSelf.
 * A null/legacy action is treated as START for backwards compatibility.
 *
 * Phase 5 (NOT here): QUIC inference transport + real llama.cpp sampling.
 */
class WorkerService : Service() {

    companion object {
        const val ACTION_START_WORKER = "com.dllm.mesh.action.START_WORKER"
        const val ACTION_STOP_WORKER = "com.dllm.mesh.action.STOP_WORKER"
        const val EXTRA_MODEL_PATH = "com.dllm.mesh.extra.MODEL_PATH"

        const val CHANNEL_ID = "dllm_worker"
        const val NOTIF_ID = 1001
        private const val WAKE_TAG = "dllm:shard"
        private const val WIFI_TAG = "dllm:worker"
        private const val HEARTBEAT_MS = 30_000L

        private val _stats = MutableStateFlow(WorkerStats())
        /** Public read-only stats; updated with tokens decoded + last heartbeat. */
        val stats: StateFlow<WorkerStats> = _stats.asStateFlow()

        fun start(context: Context, modelPath: String? = null) {
            val intent = Intent(context, WorkerService::class.java)
                .setAction(ACTION_START_WORKER)
            if (modelPath != null) intent.putExtra(EXTRA_MODEL_PATH, modelPath)
            ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: Context) {
            // Deliver STOP via intent contract; onStartCommand() calls stopSelf().
            // stopService() fallback covers the case where the service is already down.
            runCatching {
                ContextCompat.startForegroundService(
                    context,
                    Intent(context, WorkerService::class.java).setAction(ACTION_STOP_WORKER),
                )
            }
            runCatching { context.stopService(Intent(context, WorkerService::class.java)) }
        }

        internal fun updateStats(block: (WorkerStats) -> WorkerStats) {
            _stats.value = block(_stats.value)
        }
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var heartbeatJob: Job? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var wifiLock: WifiManager.WifiLock? = null

    override fun onCreate() {
        super.onCreate()
        ensureChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP_WORKER -> {
                stopSelf()
                return START_NOT_STICKY
            }
            ACTION_START_WORKER, null -> handleStart(intent?.getStringExtra(EXTRA_MODEL_PATH))
            else -> handleStart(intent.getStringExtra(EXTRA_MODEL_PATH))
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        heartbeatJob?.cancel()
        heartbeatJob = null
        scope.cancel()
        runCatching { if (LlamaBridge.nativeAvailable) LlamaBridge.free() }
        releaseLocks()
        updateStats { it.copy(running = false, ready = false, status = "Stopped.") }
        super.onDestroy()
    }

    // ---- startup -----------------------------------------------------------

    private fun handleStart(explicitModelPath: String?) {
        startForegroundWith("DLLM Mesh worker — starting…")
        holdWifiLock()
        updateStats {
            it.copy(running = true, status = "Starting…", lastHeartbeatEpochMs = System.currentTimeMillis())
        }
        heartbeatJob?.cancel()
        heartbeatJob = scope.launch {
            probeLocalModel(explicitModelPath)
            while (true) {
                delay(HEARTBEAT_MS)
                updateStats { it.copy(lastHeartbeatEpochMs = System.currentTimeMillis()) }
            }
        }
    }

    /** Loads a local GGUF if present and reports readiness. No network here (Phase 5). */
    private suspend fun probeLocalModel(explicitModelPath: String?) {
        if (!LlamaBridge.nativeAvailable) {
            updateStats { it.copy(status = "Native lib missing — reinstall APK.") }
            startForegroundWith("DLLM Mesh worker — native lib missing")
            return
        }
        val candidates = buildList {
            if (!explicitModelPath.isNullOrBlank()) add(File(explicitModelPath))
            add(File(filesDir, "model.gguf"))
            add(File(File(filesDir, "models"), "model.gguf"))
            getExternalFilesDir("models")?.let { add(File(it, "model.gguf")) }
        }
        val gguf = candidates.firstOrNull { it.isFile && it.canRead() }
        if (gguf == null) {
            updateStats {
                it.copy(
                    modelLoaded = false, modelPath = null, ready = false,
                    status = "Idle — no local GGUF (place model.gguf in filesDir/models/).",
                    lastHeartbeatEpochMs = System.currentTimeMillis(),
                )
            }
            startForegroundWith("DLLM Mesh worker — idle, no local model")
            return
        }
        val ok = runCatching { LlamaBridge.loadModel(gguf.absolutePath) }.getOrDefault(false)
        if (!ok) {
            updateStats {
                it.copy(
                    modelLoaded = false, modelPath = null, ready = false,
                    status = "Model unreadable: ${gguf.absolutePath}",
                    lastHeartbeatEpochMs = System.currentTimeMillis(),
                )
            }
            startForegroundWith("DLLM Mesh worker — model unreadable")
            return
        }
        // Single-threaded JNI sanity round-trip (real llama.cpp decode, Phase 5).
        val sanity = runCatching { LlamaBridge.inferChunk("ping", 4) }.getOrNull()
        updateStats {
            it.copy(
                modelLoaded = true, modelPath = gguf.absolutePath, ready = true,
                status = "Ready — ${gguf.name} (sanity decode: $sanity)",
                lastHeartbeatEpochMs = System.currentTimeMillis(),
            )
        }
        startForegroundWith("DLLM Mesh worker — ready (${gguf.name})")
    }

    // ---- compute -----------------------------------------------------------

    /** Real JNI llama.cpp forward (Phase 5). WakeLock held ONLY around compute. */
    @Suppress("unused")
    private fun runShardCompute(prompt: String, maxTokens: Int): String {
        val power = getSystemService(Context.POWER_SERVICE) as PowerManager
        val lock = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKE_TAG).apply {
            setReferenceCounted(false)
            acquire(10 * 60 * 1000L) // fail-safe timeout; always released below
        }
        wakeLock = lock
        try {
            val out = if (LlamaBridge.nativeAvailable) {
                runCatching { LlamaBridge.inferChunk(prompt, maxTokens) }.getOrNull().orEmpty()
            } else {
                ""
            }
            // Decoded-token count is still estimated as maxTokens; inferChunk
            // returns text only. Swap for a real token counter with the
            // Phase 5 QUIC inference transport.
            updateStats {
                it.copy(
                    tokensDecoded = it.tokensDecoded + maxTokens,
                    lastHeartbeatEpochMs = System.currentTimeMillis(),
                )
            }
            return out
        } finally {
            if (lock.isHeld) runCatching { lock.release() }
            wakeLock = null
        }
    }

    // ---- foreground / locks ------------------------------------------------

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val manager = getSystemService(NotificationManager::class.java)
            if (manager.getNotificationChannel(CHANNEL_ID) == null) {
                manager.createNotificationChannel(
                    NotificationChannel(CHANNEL_ID, "Mesh worker", NotificationManager.IMPORTANCE_MIN),
                )
            }
        }
    }

    private fun startForegroundWith(text: String) {
        ensureChannel()
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("DLLM Mesh worker")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_sys_data_bluetooth)
            .setOngoing(true)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIF_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE,
            )
        } else {
            @Suppress("DEPRECATION")
            startForeground(NOTIF_ID, notification)
        }
    }

    private fun holdWifiLock() {
        if (wifiLock?.isHeld == true) return
        val wifi = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
        wifiLock = wifi.createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, WIFI_TAG).apply {
            setReferenceCounted(true)
            acquire()
        }
        // Phase 5: TLS dial to coordinator + shard fetch + heartbeat loop go here.
        // No QUIC inference transport in Phase 4 by design.
    }

    private fun releaseLocks() {
        wakeLock?.let { if (it.isHeld) runCatching { it.release() } }
        wakeLock = null
        wifiLock?.let { if (it.isHeld) runCatching { it.release() } }
        wifiLock = null
    }
}
