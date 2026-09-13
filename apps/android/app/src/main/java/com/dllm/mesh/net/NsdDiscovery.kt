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
 * mDNS helper for coordinator discovery.
 *
 * Service type `_dllm._tcp.` (NOT the `_llmshard._tcp` draft from research —
 * reconciled to the dllm namespace; confirm in pairing spec when it lands).
 * QR fallback remains mandatory: many APs block mDNS across bands.
 */
class NsdDiscovery(context: Context) {

    companion object {
        const val SERVICE_TYPE = "_dllm._tcp."
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

    /** Advertise this device (worker role, Phase 4). Safe to call; no-op on failure. */
    fun registerService(port: Int, txt: Map<String, String> = emptyMap()) {
        unregisterService()
        val info = NsdServiceInfo().apply {
            serviceName = "dllm-node"
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
