package com.dllm.mesh.net

import android.net.Uri

/**
 * Parsed `dllm://pair?host=&port=&quic=&fp=&v=` URI (server pairing contract).
 *
 * - host: server LAN IP/hostname (required)
 * - port: server HTTP port (required)
 * - quic: QUIC port (optional, 0/empty = unknown)
 * - fp: server fingerprint for TOFU pinning (optional, "" = not pinned)
 * - v: URI version (optional, defaults to 1)
 */
data class PairServer(
    val host: String,
    val port: Int,
    val quicPort: Int?,
    val fingerprint: String,
    val version: Int,
) {
    fun baseUrl(): String = "http://$host:$port"
}

/** Parses a `dllm://pair?...` URI; null when the text is not such a URI. */
fun parsePairUri(raw: String): PairServer? = runCatching {
    val uri = Uri.parse(raw.trim())
    if (!uri.scheme.equals("dllm", ignoreCase = true)) return@runCatching null
    // Accept both dllm://pair?... (host == "pair") and dllm:pair?... forms.
    val isPair = uri.host.equals("pair", ignoreCase = true) ||
        uri.pathSegments.firstOrNull().equals("pair", ignoreCase = true) ||
        uri.schemeSpecificPart.trimStart('/').startsWith("pair", ignoreCase = true)
    if (!isPair) return@runCatching null
    val host = uri.getQueryParameter("host")?.trim().orEmpty()
    val port = uri.getQueryParameter("port")?.trim()?.toIntOrNull()
    if (host.isEmpty() || port == null || port <= 0 || port > 65535) {
        return@runCatching null
    }
    PairServer(
        host = host,
        port = port,
        quicPort = uri.getQueryParameter("quic")?.trim()?.toIntOrNull()
            ?.takeIf { it in 1..65535 },
        fingerprint = uri.getQueryParameter("fp")?.trim().orEmpty(),
        version = uri.getQueryParameter("v")?.trim()?.toIntOrNull() ?: 1,
    )
}.getOrNull()

/** Normalizes a typed/pasted base URL: trims, drops trailing slashes. */
fun normalizeBaseUrl(raw: String): String = raw.trim().trimEnd('/')

/** Strips scheme, path, query and fragment from a pasted host. */
fun cleanHost(raw: String): String {
    var s = raw.trim()
    s = s.removePrefix("http://").removePrefix("https://").removePrefix("dllm://")
    val cut = s.indexOfFirst { it == '/' || it == '?' || it == '#' }
    if (cut >= 0) s = s.substring(0, cut)
    return s.trim()
}

/**
 * Splits pasted `host` or `host:port` text. Returns `(host, portText)` with
 * `portText` empty when no single numeric `:port` suffix is present.
 */
fun splitHostPort(text: String): Pair<String, String> {
    val s = cleanHost(text)
    val i = s.lastIndexOf(':')
    if (i > 0 && s.indexOf(':') == i && s.substring(i + 1).all { it.isDigit() }) {
        return s.substring(0, i) to s.substring(i + 1)
    }
    return s to ""
}

/** Extracts `(host, portText)` from a saved base URL for form prefill. */
fun currentHostPort(baseUrl: String): Pair<String, String> = runCatching {
    val uri = java.net.URI(normalizeBaseUrl(baseUrl))
    val host = uri.host.orEmpty()
    val port = uri.port.takeIf { it in 1..65535 }?.toString().orEmpty()
    host to port
}.getOrDefault("" to "")

/**
 * Fail-fast warning for a pairing host, or null when it looks fine.
 * Catches the classic multi-homed-coordinator mistake (VPN/PPP address),
 * loopback (points at the phone itself) and link-local addresses.
 */
fun hostWarning(host: String): String? {
    val h = cleanHost(host)
    if (h.isEmpty()) return null
    val low = h.lowercase()
    if (low == "localhost" || h.startsWith("127.")) {
        return "Loopback points at this phone itself — use the laptop WiFi IP from `dllm id`."
    }
    if (low.startsWith("169.254.")) {
        return "Link-local address — rarely routable. Prefer a 192.168.x address."
    }
    if (h.any { it.isWhitespace() || it == '/' }) {
        return "Host should be a bare IP or name, no spaces or slashes."
    }
    return null
}

/** True when [baseUrl] addresses [host]:[port] (attach-state matching). */
fun baseUrlMatches(baseUrl: String, host: String, port: Int): Boolean = runCatching {
    val uri = java.net.URI(normalizeBaseUrl(baseUrl))
    val schemeDefault = if (uri.scheme.equals("https", ignoreCase = true)) 443 else 80
    val uriPort = uri.port.takeIf { it != -1 } ?: schemeDefault
    uri.host.equals(host, ignoreCase = true) && uriPort == port
}.getOrDefault(false)

/**
 * Parsed `dllm://net?...` group-join URI (Networks QR contract).
 *
 * - host/port: coordinator LAN address (required — join is always LAN-visible)
 * - id: network/group id (required)
 * - secret: QR secret baked into the code (optional; skips the password prompt)
 * - pw: group password embedded in the code (optional; manual entry otherwise)
 * - fp/quic: same TOFU pin + QUIC port semantics as [PairServer]
 *
 * Accepts both `dllm://net?...` (host == "net") and `dllm:net?...` forms.
 */
data class NetJoin(
    val host: String,
    val port: Int,
    val networkId: String,
    val qrSecret: String,
    val password: String,
    val fingerprint: String,
    val quicPort: Int?,
) {
    fun baseUrl(): String = "http://$host:$port"
}

/** Parses a `dllm://net?...` URI; null when the text is not such a URI. */
fun parseNetUri(raw: String): NetJoin? = runCatching {
    val uri = Uri.parse(raw.trim())
    if (!uri.scheme.equals("dllm", ignoreCase = true)) return@runCatching null
    val isNet = uri.host.equals("net", ignoreCase = true) ||
        uri.pathSegments.firstOrNull().equals("net", ignoreCase = true) ||
        uri.schemeSpecificPart.trimStart('/').startsWith("net", ignoreCase = true)
    if (!isNet) return@runCatching null
    val host = uri.getQueryParameter("host")?.trim().orEmpty()
    val port = uri.getQueryParameter("port")?.trim()?.toIntOrNull()
    val id = uri.getQueryParameter("id")?.trim().orEmpty()
        .ifBlank { uri.getQueryParameter("group_id")?.trim().orEmpty() }
        .ifBlank { uri.getQueryParameter("network_id")?.trim().orEmpty() }
    if (host.isEmpty() || port == null || port <= 0 || port > 65535 || id.isEmpty()) {
        return@runCatching null
    }
    NetJoin(
        host = host,
        port = port,
        networkId = id,
        qrSecret = uri.getQueryParameter("secret")?.trim().orEmpty()
            .ifBlank { uri.getQueryParameter("qr_secret")?.trim().orEmpty() },
        password = uri.getQueryParameter("pw")?.trim().orEmpty()
            .ifBlank { uri.getQueryParameter("password")?.trim().orEmpty() },
        fingerprint = uri.getQueryParameter("fp")?.trim().orEmpty(),
        quicPort = uri.getQueryParameter("quic")?.trim()?.toIntOrNull()
            ?.takeIf { it in 1..65535 },
    )
}.getOrNull()
