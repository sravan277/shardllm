package com.dllm.mesh.net

import java.net.Inet4Address
import java.net.NetworkInterface
import java.util.Collections

/**
 * LAN IPv4 candidates for interfaces this phone can actually be reached on.
 *
 * WHY THIS EXISTS (and why it enumerates instead of picking one): the RPC server
 * has to be bound to an address a *coordinator on another device* can dial, and
 * a phone is aggressively multi-homed — wlan0, rmnet0/cellular, ap0 hotspot,
 * VPN (tun0), and often a PPP/VPN `/32`. ADR-026 documents the server-side trap
 * in exactly this shape: a single-IP trick (e.g. "open a UDP socket and read the
 * outbound route") picks the VPN/PPP address on a multi-homed host, because that
 * is the route the OS would actually use for outbound traffic — and then the
 * server binds an address no peer on the WiFi subnet can reach. So this helper
 * enumerates and ranks instead of guessing, and the caller binds the best
 * candidate, falling back to `0.0.0.0` when nothing looks like a LAN.
 *
 * Ranking is `192.168.* > 10.* > 172.16/12 > other-routable`, matching ADR-026's
 * server-side ordering so both ends of a link agree on what "the LAN address"
 * means. Loopback, link-local (169.254/16) and any non-IPv4 interface are
 * filtered out: none of them is reachable by a peer.
 *
 * No dependencies: `java.net.NetworkInterface` only.
 */
object LanIp {

    /** Bind address meaning "every interface". Used when no LAN candidate is found. */
    const val ANY_INTERFACE = "0.0.0.0"

    /**
     * All routable IPv4 addresses on up interfaces, best-ranked first.
     *
     * Returns an empty list when every interface is down or non-IPv4 (airplane
     * mode), which the caller must handle — not an invented 127.0.0.1.
     */
    fun candidates(): List<String> {
        val found = ArrayList<String>()
        // Plain try/catch, not runCatching: NetworkInterface enumeration throws
        // SocketException on a phone whose radios are being toggled, and this
        // runs on a coroutine that must not be interrupted by one bad interface.
        try {
            for (nic in NetworkInterface.getNetworkInterfaces()) {
                if (!nic.isUp || nic.isLoopback) continue
                for (addr in Collections.list(nic.inetAddresses)) {
                    val ip = (addr as? Inet4Address)?.hostAddress ?: continue
                    if (isRoutable(ip)) found.add(ip)
                }
            }
        } catch (e: Exception) {
            // Partial results are still better than none: a VPN interface that
            // refuses to enumerate must not hide a perfectly good wlan0 address.
        }
        return found.distinct().sortedByDescending { rank(it) }
    }

    /**
     * Best single LAN address, or null when the phone currently has none.
     *
     * Deliberately nullable: callers must decide between binding this address and
     * binding [ANY_INTERFACE], and inventing a loopback answer here would make the
     * RPC endpoint unreachable while looking healthy.
     */
    fun best(): String? = candidates().firstOrNull()

    /** Coarse trust ranking; higher is more likely to be the shared LAN. */
    private fun rank(ip: String): Int = when {
        ip.startsWith("192.168.") -> 300
        isPrivate10(ip) -> 200
        isPrivate172(ip) -> 100
        else -> 0
    }

    private fun isRoutable(ip: String): Boolean =
        !ip.startsWith("127.") && !ip.startsWith("169.254.") && (rank(ip) > 0)

    private fun isPrivate10(ip: String): Boolean = ip.startsWith("10.")

    /** 172.16.0.0/12 — i.e. 172.16 through 172.31, not all of 172/8. */
    private fun isPrivate172(ip: String): Boolean {
        val parts = ip.split('.')
        if (parts.size != 4 || parts[0] != "172") return false
        val second = parts[1].toIntOrNull() ?: return false
        return second in 16..31
    }
}