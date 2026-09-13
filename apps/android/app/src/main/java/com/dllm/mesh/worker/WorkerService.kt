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

/**
 * Phase 0 worker stub: ForegroundService survives from day one (per research),
 * but NO real inference runs here yet — shard execution lands in Phase 4 via
 * JNI (llama.cpp .so). [runShardCompute] and [connect] are lock-lifecycle
 * stubs so Doze/OEM behaviour can be validated early.
 */
class WorkerService : Service() {

    companion object {
        const val CHANNEL_ID = "dllm_worker"
        const val NOTIF_ID = 1001
        private const val WAKE_TAG = "dllm:shard"
        private const val WIFI_TAG = "dllm:worker"

        fun start(context: Context) {
            val intent = Intent(context, WorkerService::class.java)
            ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, WorkerService::class.java))
        }
    }

    private var wakeLock: PowerManager.WakeLock? = null
    private var wifiLock: WifiManager.WifiLock? = null

    override fun onCreate() {
        super.onCreate()
        val manager = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
            manager.getNotificationChannel(CHANNEL_ID) == null
        ) {
            manager.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "Mesh worker", NotificationManager.IMPORTANCE_MIN),
            )
        }
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("DLLM Mesh worker")
            .setContentText("Idle — waiting for shard assignment (Phase 0 stub)")
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

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Phase 4: connect() to coordinator here, then await shard assignment.
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        releaseLocks()
        super.onDestroy()
    }

    /** TODO Phase 4: open transport to coordinator; hold HIGH_PERF WifiLock while linked. */
    private fun connect() {
        val wifi = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
        if (wifiLock?.isHeld != true) {
            wifiLock = wifi.createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, WIFI_TAG).apply {
                setReferenceCounted(true)
                acquire()
            }
        }
        // TODO(Phase 4): TLS dial + shard fetch + heartbeat loop.
    }

    /** TODO Phase 4: real JNI shard forward. WakeLock held ONLY around compute. */
    private fun runShardCompute() {
        val power = getSystemService(Context.POWER_SERVICE) as PowerManager
        val lock = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKE_TAG).apply {
            setReferenceCounted(false)
            acquire(10 * 60 * 1000L) // fail-safe timeout; always released below
        }
        wakeLock = lock
        try {
            // TODO(Phase 4): nativeForwardShard(...)
        } finally {
            if (lock.isHeld) runCatching { lock.release() }
            wakeLock = null
        }
    }

    private fun releaseLocks() {
        wakeLock?.let { if (it.isHeld) runCatching { it.release() } }
        wakeLock = null
        wifiLock?.let { if (it.isHeld) runCatching { it.release() } }
        wifiLock = null
    }
}
