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
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.dllm.mesh.data.IdentityStore
import com.dllm.mesh.net.LanIp
import com.dllm.mesh.net.Presence
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.io.File

/**
 * Snapshot of worker health. Observed by UI via [WorkerService.stats] (no binding).
 *
 * [lastHeartbeatEpochMs] is the liveness signal, NOT [running]/[ready]: stats
 * live in a static field, so a value written just before the OS killed the
 * service would otherwise still read "Ready" afterwards. Freshness of the last
 * heartbeat is the only evidence that survives that scenario.
 */
data class WorkerStats(
    val running: Boolean = false,
    val modelLoaded: Boolean = false,
    val modelPath: String? = null,
    val tokensDecoded: Long = 0L,
    val lastHeartbeatEpochMs: Long = 0L,
    val ready: Boolean = false,
    val status: String = "Stopped.",
    /**
     * ggml-rpc endpoint this phone is hosting, as `host:port` — the exact string
     * `dllm_shim_add_rpc_server(host_port)` takes. Null whenever the RPC server
     * is not confirmed listening.
     *
     * NEVER synthesised: null means "not hosting", it does not mean "host unknown
     * but probably fine". A coordinator that reads a made-up endpoint fails at
     * connect time with a much worse error message than an absent field.
     */
    val rpcEndpoint: String? = null,
    /**
     * ggml-rpc port, or 0 when not hosting.
     *
     * PLACEHOLDER-FREE but also not a measurement: it is the port we bound, and
     * [rpcEndpoint] is non-null only once a real connect() to that port
     * succeeded. Kept as a separate field because the coordinator's device row
     * stores a port next to the host, not a joined string.
     */
    val rpcPort: Int = 0,
)

/**
 * Phase 4 worker: foreground service (type connectedDevice) + JNI stub.
 *
 * Intent contract (owned here; the sibling Devices tab toggle calls [start]/[stop]):
 * - `com.dllm.mesh.action.START_WORKER` → foreground + probe local GGUF + heartbeat.
 * - `com.dllm.mesh.action.STOP_WORKER`  → stopSelf().
 * A null/legacy action is treated as START for backwards compatibility.
 *
 * Phone offload readiness: while running, every heartbeat goes out as
 * role=worker with load + capabilities{ms_per_layer_decode,kv_pages,layers}
 * (+ worker_active + layers) via [Presence], so the backend `/v1/plan` can
 * assign layers to this phone.
 *
 * Liveness: every heartbeat tick refreshes [WorkerStats.lastHeartbeatEpochMs];
 * [isLive] combines that freshness with `running`/`ready` so the UI cannot show
 * a stale "Ready" for a service the OS has already killed.

 *
 * NOTE: the JNI [LlamaBridge.inferChunk] path is still local-only (sanity
 * round-trip + token counting). Remote shard execution over QUIC chaining
 * has NOT landed — no inference traffic leaves the phone yet.
 */
class WorkerService : Service() {

    companion object {
        const val ACTION_START_WORKER = "com.dllm.mesh.action.START_WORKER"
        const val ACTION_STOP_WORKER = "com.dllm.mesh.action.STOP_WORKER"
        const val EXTRA_MODEL_PATH = "com.dllm.mesh.extra.MODEL_PATH"

        const val CHANNEL_ID = "dllm_worker"
        const val NOTIF_ID = 1001
        private const val TAG = "WorkerService"
        private const val WAKE_TAG = "dllm:shard"
        private const val WIFI_TAG = "dllm:worker"
        private const val HEARTBEAT_MS = 30_000L

        /**
         * Port the ggml-rpc server binds (ADR-031). Upstream `rpc-server` defaults
         * to 50052; keeping it means a coordinator can be pointed here by hand
         * from a log without cross-referencing two halves of the codebase.
         */
        const val RPC_PORT = 50052

        /**
         * Threads the hosted CPU device may use per graph.
         *
         * NOT a measurement of this phone's decode throughput — a deliberate cap.
         * The RPC server computes assigned layers on a native thread via ggml-cpu's
         * pool; leaving every core to it would starve the JVM and risk the 30s
         * heartbeat. Same honesty rule as `WorkerCapabilities`' placeholders: the
         * value here is a ceiling chosen for liveness, and is labelled as such
         * rather than presented as a calibrated capability.
         */
        const val RPC_THREADS = 2

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

        /**
         * How long a worker heartbeat may go unconfirmed before the UI stops
         * calling the worker live. Matches the coordinator's own 90s device
         * activity window, so the phone and the registry agree on when a device
         * counts as gone.
         */
        const val LIVENESS_WINDOW_MS = 90_000L

        /**
         * True only when the worker is genuinely live right now: running, model
         * probe passed, AND the last heartbeat is inside [LIVENESS_WINDOW_MS].
         *
         * WHY heartbeat freshness is part of the test: [WorkerStats] is a static
         * field. If the service is killed the UI can still be holding a
         * `ready = true` snapshot, and "Ready" would be a lie until something
         * else rewrote it. A worker that has not confirmed itself within the
         * window is treated as dead — the honest, conservative direction.
         */
        fun isLive(stats: WorkerStats, nowMs: Long = System.currentTimeMillis()): Boolean =
            stats.running &&
                stats.ready &&
                stats.lastHeartbeatEpochMs > 0L &&
                (nowMs - stats.lastHeartbeatEpochMs) <= LIVENESS_WINDOW_MS

        /** Seconds since the last confirmed worker heartbeat, or null if never. */
        fun heartbeatAgeSeconds(stats: WorkerStats, nowMs: Long = System.currentTimeMillis()): Long? =
            if (stats.lastHeartbeatEpochMs <= 0L) null
            else ((nowMs - stats.lastHeartbeatEpochMs) / 1000L).coerceAtLeast(0L)
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var heartbeatJob: Job? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var wifiLock: WifiManager.WifiLock? = null

    /**
     * ggml-rpc protocol version the native server speaks, read from
     * `LlamaBridge.rpcProtocolVersion()` when hosting started. Empty = unknown,
     * and the heartbeat then omits `rpc_protocol` rather than guessing it.
     */
    @Volatile
    private var rpcProtocol: String = ""

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
        stopRpcServer("service stopped")
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
            // Host BEFORE the first heartbeat so the very first advertisement
            // already carries a live endpoint; the reverse order would make the
            // coordinator learn about this worker one tick before it could reach it.
            startRpcServer()
            sendWorkerHeartbeat()
            while (true) {
                delay(HEARTBEAT_MS)
                sendWorkerHeartbeat()
            }
        }
    }

    /**
     * Start the ggml-rpc server so a coordinator on another device can offload
     * transformer layers here (ADR-031).
     *
     * Runs on [scope]'s dispatcher, never the main thread: the JNI call blocks
     * until the port is confirmed listening (bounded at ~5s inside native code),
     * and an ANR is not an acceptable way to open a socket. It does NOT hold a
     * wake lock of its own — [holdWifiLock] already keeps the interface up for
     * the whole service, and a second lock would only add battery draw.
     *
     * Bind address: the best LAN candidate from [LanIp], NOT loopback (a
     * coordinator on another device cannot reach 127.0.0.1) and NOT the
     * single-IP route trick that ADR-026 documents as picking VPN/PPP addresses
     * on multi-homed hosts. If no candidate looks like a LAN — airplane mode, or
     * only cellular — we bind [LanIp.ANY_INTERFACE] and still advertise, because
     * a listener on every interface is strictly more reachable than no listener;
     * the address we advertise is then the best candidate we have, which the
     * coordinator may or may not be able to route to.
     */
    private suspend fun startRpcServer() {
        if (!LlamaBridge.nativeAvailable) {
            Log.w(TAG, "RPC hosting skipped: native lib missing")
            return
        }
        val initRc = runCatching { LlamaBridge.shimInit() }.getOrDefault(-1)
        if (initRc != 0) {
            Log.w(TAG, "RPC hosting skipped: shimInit rc=$initRc (${LlamaBridge.shimLastError()})")
            updateStats { it.copy(rpcEndpoint = null, rpcPort = 0) }
            return
        }
        val candidates = LanIp.candidates()
        val bindHost = candidates.firstOrNull() ?: LanIp.ANY_INTERFACE
        val port = RPC_PORT
        val rc = runCatching { LlamaBridge.rpcServeStart(bindHost, port, RPC_THREADS, -1) }
            .getOrDefault(-1)
        if (rc != 0 || !LlamaBridge.rpcServing()) {
            val why = runCatching { LlamaBridge.shimLastError() }.getOrDefault("unknown")
            Log.w(TAG, "RPC hosting failed on $bindHost:$port rc=$rc ($why)")
            updateStats { it.copy(rpcEndpoint = null, rpcPort = 0) }
            return
        }
        // Advertise the address a peer can actually dial. When we bound
        // 0.0.0.0 the bind address is not a dialable endpoint, so fall back to the
        // best candidate and, failing that, report nothing rather than a wildcard.
        val advertised = when {
            bindHost != LanIp.ANY_INTERFACE -> bindHost
            else -> candidates.firstOrNull()
        }
        if (advertised == null) {
            Log.w(TAG, "RPC bound to $bindHost:$port but no LAN address to advertise; withdrawing")
            LlamaBridge.rpcServeStop()
            updateStats { it.copy(rpcEndpoint = null, rpcPort = 0) }
            return
        }
        val endpoint = "$advertised:$port"
        val proto = runCatching { LlamaBridge.rpcProtocolVersion() }.getOrDefault("").trim()
        Log.i(TAG, "RPC hosting layers at $endpoint (bound $bindHost, candidates=$candidates, proto=$proto)")
        rpcProtocol = proto
        updateStats {
            it.copy(
                rpcEndpoint = endpoint,
                rpcPort = port,
                // Report the endpoint where the user can actually see it: stats
                // (Devices tab) and the ongoing notification.
                status = "${it.status} Hosting RPC $endpoint.",
            )
        }
        startForegroundWith("DLLM Mesh worker — hosting layers at $endpoint")
    }

    /**
     * Withdraw the RPC endpoint. Called on toggle-off (via [stop] -> onDestroy)
     * and on service teardown.
     *
     * Honest scope: llama.cpp b7418 has no RPC-server shutdown hook, so this stops
     * *advertising* and refuses a restart on the same port for the life of the
     * process; the listening socket is released when the app process exits. The
     * endpoint is therefore cleared from stats immediately so no heartbeat can
     * point a coordinator at a worker we have retired.
     */
    private fun stopRpcServer(reason: String) {
        if (!LlamaBridge.nativeAvailable) return
        runCatching { LlamaBridge.rpcServeStop() }
            .onFailure { Log.w(TAG, "rpcServeStop failed: ${it.message}") }
        runCatching { LlamaBridge.shimFree() }
            .onFailure { Log.w(TAG, "shimFree failed: ${it.message}") }
        rpcProtocol = ""
        updateStats { it.copy(rpcEndpoint = null, rpcPort = 0) }
        Log.i(TAG, "RPC endpoint withdrawn ($reason)")
    }

    /**
     * Worker lifeline: role=worker heartbeat with load + capabilities so
     * `/v1/plan` can assign layers. Failures only log — the loop retries on
     * the next tick and the service keeps running.
     *
     * The ggml-rpc endpoint rides along in `capabilities` (`rpc_endpoint` =
     * `host:port`, `rpc_port`) when this phone is hosting, and is absent when it
     * is not — so a coordinator can tell "this worker will accept layers" from
     * "this worker only chats".
     */
    private suspend fun sendWorkerHeartbeat() {
        val tick = runCatching {
            val base = IdentityStore(applicationContext).coordinatorUrl.first().trimEnd('/')
            Presence.postHeartbeat(
                applicationContext,
                base,
                _stats.value.rpcEndpoint,
                rpcProtocol.ifBlank { null },
            )
        }
        if (tick.isSuccess) {
            updateStats { it.copy(lastHeartbeatEpochMs = System.currentTimeMillis()) }
        } else {
            Log.w(TAG, "worker heartbeat failed: ${tick.exceptionOrNull()?.message}")
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
