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

/** True when [baseUrl] addresses [host]:[port] (attach-state matching). */
fun baseUrlMatches(baseUrl: String, host: String, port: Int): Boolean = runCatching {
    val uri = java.net.URI(normalizeBaseUrl(baseUrl))
    val uriPort = uri.port.takeIf { it != -1 } ?: 80
    uri.host.equals(host, ignoreCase = true) && uriPort == port
}.getOrDefault(false)
