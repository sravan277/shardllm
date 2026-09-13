# Pairing handshake

## 0. MVP bootstrap (implemented)

Stable coordinator identity for LAN TOFU pairing:

- Files under `%LOCALAPPDATA%\dllm\` (fallback: process working directory):
  `node-identity.crt` + `node-identity.key` (raw DER bytes, minted via
  `dllm-net Identity::generate`), plus `node-id.txt` (`dllm-<fp[..12]>`,
  re-derived from the cert fingerprint if missing). Generated once with
  `create_dir_all`; restart reuses the same bytes. Operator reads the
  fingerprint from the `stable node identity loaded` INFO log.
- `GET /api/node` -> `{"node_id","fingerprint","quic_port":8443,"version":"0.1.0"}`
  (`fingerprint` = lowercase hex `sha256(cert_der)`).
- `dllm id [--port 8080]` prints `node_id`, `fingerprint`, and one line:
  `dllm://pair?host=<lan-ip>&port=<http>&quic=8443&fp=<fingerprint>&v=0.1.0`
  (`<lan-ip>` = best-effort local IPv4, fallback `127.0.0.1`).
  The Android app scans/types this URI, then pins `fp` for QUIC mTLS TOFU
  (see `crates/dllm-net/src/transport.rs`).
- mDNS `_dllm._tcp.local.` TXT currently advertises
  `quic_port` / `node_id` / `ver` / `fp` (discovery only).

Full invite-secret + verify-code flow below (§2–§5) is the target design;
until it lands, the `fp` in this bootstrap URI is the pairing trust root.

Goal: admit ≤5 trusted devices to the LAN mesh with per-device revocable
mTLS credentials. No shared LAN password.

## 1. Discovery (mDNS, discovery-only)

- Service: `_dllm._tcp.local.`, port = LAN API port.
- TXT keys (all strings):

| Key | Example | Notes |
|---|---|---|
| `id` | `win-coord-01` | Stable device id |
| `ver` | `0.1.0` | Runtime version; clients reject major mismatch |
| `fp` | `SHA256:ab:cd:…` | SPKI fingerprint of current cert |
| `api` | `8443` | LAN API port (mirrors SRV) |
| `proto` | `dllm1` | Contract major; must match |
| `role` | `coordinator` | `coordinator` \| `worker` \| `client` |

- Pairing-time AP/client-isolation diagnostic: if mDNS sees nothing but the
  coordinator is reachable by IP, surface "router is isolating clients",
  not "no devices found".

## 2. Invite (QR or OTP — same payload)

QR encodes this JSON (OTP = 9-char base32 rendering of `secret`, rest typed/shown):

```json
{
  "coordinator_id": "win-coord-01",
  "ip": "192.168.1.10",
  "port": 8443,
  "proto": "dllm1",
  "secret": "JBSWY3DPEHPK3PXP",
  "expires_at": "2026-09-13T12:05:00Z"
}
```

- `secret`: 80-bit one-time invite, single-use, 5-min TTL.
- QR may be shown by any already-paired device (delegated invite attested by
  coordinator); OTP is coordinator-issued only.

## 3. Verify-code confirm (anti-MITM, mandatory)

1. Joiner opens TLS with `secret` (PAKE-ish bearer, NOT the long-term key) and
   sends its long-term public key + device info.
2. **Both sides display the same 6-digit code**: `code = Truncate6(SHA256(secret || joiner_pubkey || coordinator_pubkey))`.
3. User confirms match on both screens → coordinator signs the joiner cert.
   Mismatch/cancel → invite burned, event logged.

## 4. Pubkey exchange + allow-list record

- Long-term identity: Ed25519 device key; operational transport uses mTLS
  (QUIC on svc, TCP+TLS on Android MVP) with certs bound to that key.
- Coordinator appends to its allow-list (SQLite `devices` table + `devices.json`
  export) and replicates the fingerprint set to workers:

```json
{
  "device_id": "pixel-8-01",
  "pubkey": "ed25519:BASE64…",
  "cert_fp": "SHA256:…",
  "role": "worker",
  "permissions": ["infer", "chat"],
  "capabilities": { "layers": "9-18", "kv_pages": 256, "ms_per_layer_decode": 3.1 },
  "paired_at": "2026-09-13T12:04:00Z",
  "paired_by": "win-coord-01"
}
```

- Future connections: mTLS + allow-list check. Unknown fingerprint → drop
  before any inference/control bytes.

## 5. Revocation

- `POST /v1/devices/{id}/revoke` removes the fingerprint from the allow-list,
  pushes the new set to all workers, and kills that device's sessions/streams.
- Revoked certs are kept on a deny-list (same `cert_fp`) so a stale backup
  cannot re-add them. Re-pair requires a fresh invite + verify-code.
- Roles/permissions downgrade (`approve` with reduced scopes) uses the same
  push path as revocation.
