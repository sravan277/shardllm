package com.dllm.mesh.net

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.net.wifi.WifiManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class Peer(
    val host: String,
    val port: Int,
    val txtMap: Map<String, String> = emptyMap(),
    val serviceName: String,
)

/**
 * mDNS helper for LAN discovery — browse for coordinators *and* advertise this
 * phone so a laptop can see it without the phone having to pair first.
 *
 * Service type `_dllm._tcp.` (NOT the `_llmshard._tcp` draft from research —
 * reconciled to the dllm namespace; confirm in pairing spec when it lands).
 * QR fallback remains mandatory: many APs block mDNS across bands.
 */
class NsdDiscovery(context: Context) {

    companion object {
        const val SERVICE_TYPE = "_dllm._tcp."

        /**
         * Service-name prefix for phones advertising themselves.
         *
         * WHY the prefix matters: a node advertisement is *presence only* — this
         * phone does not listen on a coordinator port — so a peer that browses
         * `_dllm._tcp.` must be able to tell "a phone saying hello" apart from
         * "a coordinator I could actually connect to". [isNodeAdvertisement] is
         * that test, and the UI refuses to offer Connect for a node row.
         */
        const val NODE_SERVICE_PREFIX = "dllm-node-"

        /**
         * Port carried in this phone's own advertisement.
         *
         * PLACEHOLDER: mDNS requires *some* port and the phone has no listener
         * yet (the JNI worker is local-only, nothing is served over the network).
         * [SELF_ADVERTISED_PORT] is a reserved, non-routable-for-this-app value
         * so nothing mistakes it for a working coordinator port.
         */
        const val SELF_ADVERTISED_PORT = 8770

        /** Deterministic mDNS name for a node advertisement of [deviceId]. */
        fun nodeServiceName(deviceId: String): String =
            NODE_SERVICE_PREFIX + deviceId.replace(Regex("[^A-Za-z0-9-]"), "-").take(24)

        /** True when [serviceName] is a phone node advertisement, not a coordinator. */
        fun isNodeAdvertisement(serviceName: String): Boolean =
            serviceName.startsWith(NODE_SERVICE_PREFIX, ignoreCase = true)
    }

    private val appContext = context.applicationContext
    private val nsdManager: NsdManager =
        appContext.getSystemService(Context.NSD_SERVICE) as NsdManager
    private val wifiManager: WifiManager =
        appContext.getSystemService(Context.WIFI_SERVICE) as WifiManager

    private val _peers = MutableStateFlow<List<Peer>>(emptyList())
    val peers: StateFlow<List<Peer>> = _peers.asStateFlow()

    private var discoveryListener: NsdManager.DiscoveryListener? = null
    private var registrationListener: NsdManager.RegistrationListener? = null
    private var multicastLock: WifiManager.MulticastLock? = null
    private var discovering = false

    fun startDiscovery() {
        if (discovering) return
        discovering = true
        _peers.value = emptyList()
        acquireMulticastLock()
        val listener = object : NsdManager.DiscoveryListener {
            override fun onDiscoveryStarted(regType: String) = Unit
            override fun onStartDiscoveryFailed(serviceType: String, errorCode: Int) {
                discovering = false
                releaseMulticastLock()
            }
            override fun onDiscoveryStopped(serviceType: String) = Unit
            override fun onStopDiscoveryFailed(serviceType: String, errorCode: Int) = Unit

            override fun onServiceFound(serviceInfo: NsdServiceInfo) {
                nsdManager.resolveService(serviceInfo, object : NsdManager.ResolveListener {
                    override fun onResolveFailed(si: NsdServiceInfo, errorCode: Int) = Unit
                    override fun onServiceResolved(si: NsdServiceInfo) {
                        val host = si.host?.hostAddress ?: return
                        val txt = si.attributes.mapValues { (_, v) ->
                            runCatching { String(v, Charsets.UTF_8) }.getOrDefault("")
                        }
                        val peer = Peer(
                            host = host,
                            port = si.port,
                            txtMap = txt,
                            serviceName = si.serviceName,
                        )
                        val current = _peers.value
                        if (current.none { it.serviceName == peer.serviceName }) {
                            _peers.value = current + peer
                        }
                    }
                })
            }

            override fun onServiceLost(serviceInfo: NsdServiceInfo) {
                _peers.value = _peers.value.filterNot { it.serviceName == serviceInfo.serviceName }
            }
        }
        discoveryListener = listener
        nsdManager.discoverServices(SERVICE_TYPE, NsdManager.PROTOCOL_DNS_SD, listener)
    }

    fun stopDiscovery() {
        discovering = false
        discoveryListener?.let { runCatching { nsdManager.stopServiceDiscovery(it) } }
        discoveryListener = null
        releaseMulticastLock()
    }

    /**
     * Advertise this phone on the LAN so a laptop (or another phone) can see it
     * in a `_dllm._tcp.` browse without pairing first. TXT carries the fields a
     * peer needs to describe the node honestly: its stable `node_id`, its
     * `role` and whether it is currently contributing compute.
     *
     * WHY this is best-effort: mDNS registration fails routinely (no WiFi, AP
     * filtering, 15-name quota) and none of those are fatal — the phone still
     * pairs by QR. Failures are swallowed on purpose; the caller has no useful
     * recovery and must not show a scary error for an invisible-to-the-user nicety.
     */
    fun registerService(port: Int = SELF_ADVERTISED_PORT, txt: Map<String, String> = emptyMap()) {
        unregisterService()
        val info = NsdServiceInfo().apply {
            serviceName = txt["node_id"]?.let { nodeServiceName(it) } ?: "dllm-node"
            serviceType = SERVICE_TYPE
            setPort(port)
            txt.forEach { (k, v) -> setAttribute(k, v) }
        }
        val listener = object : NsdManager.RegistrationListener {
            override fun onServiceRegistered(si: NsdServiceInfo) = Unit
            override fun onRegistrationFailed(si: NsdServiceInfo, errorCode: Int) = Unit
            override fun onServiceUnregistered(si: NsdServiceInfo) = Unit
            override fun onUnregistrationFailed(si: NsdServiceInfo, errorCode: Int) = Unit
        }
        registrationListener = listener
        runCatching { nsdManager.registerService(info, NsdManager.PROTOCOL_DNS_SD, listener) }
    }

    fun unregisterService() {
        registrationListener?.let { runCatching { nsdManager.unregisterService(it) } }
        registrationListener = null
    }

    private fun acquireMulticastLock() {
        if (multicastLock?.isHeld == true) return
        multicastLock = wifiManager.createMulticastLock("dllm:discovery").apply {
            setReferenceCounted(true)
            acquire()
        }
    }

    private fun releaseMulticastLock() {
        multicastLock?.let { if (it.isHeld) runCatching { it.release() } }
        multicastLock = null
    }
}
