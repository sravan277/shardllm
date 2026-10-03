# How Everything Connects (LIVING DOC — update every phase)

> Last updated: contract-conformant frame codec + bound QUIC mesh + computed plan + Networks removal (2026-10-04). This is the global wiring map: which component talks to which, over what protocol/port, with what auth, and what works *today* vs planned. Update the edge table and flows on every phase. Rule: no new connection without a row here + a contract ref.

## Map

```text
                    mDNS _dllm._tcp.local. (discovery only)
   ┌──────────┐  ──────────────────────────────  ┌──────────────────┐
   │ Android  │   HTTP/SSE  :8080 (Phase 0 live)  │  dllm serve      │
   │ chat app │  ──────────────────────────────► │  (Windows, Rust) │
   │          │   QUIC dllm/1 :8443 (idle)         │  coordinator     │
   └──────────┘  ◄────────────────────────────── │  Axum + SQLite   │
                                                │  event log       │
   ┌──────────┐  HTTP/SSE  :8080 (Phase 0 live)  └──────────────────┘
   │ Web UI   │  ──────────────────────────────►         ▲
   │ :5173 dev│   (dist/ served by dllm, Phase 2)        │ QUIC (Ph3)
   │ or dist/ │                                          │ stage links
   └──────────┘                                     ┌────┴─────┐
                                                    │ workers  │
   Cloud (Phase 5): account/meta + opt-in backups   │ A→B→C    │
   ...account/device metadata, telemetry, backups   └──────────┘
   ...NEVER weights/activations/KV/plaintext chat
```

Stage links (worker `A→B→C`) are drawn for topology only: no activation frame is
dispatched between real devices yet, and `8443` carries a listener but no stage
traffic (see E4 + E9). Inference today is single-device.

## Edge table

| # | From → To | Protocol | Port | Auth | Phase 0 status | Contract |
|---|-----------|----------|------|------|----------------|----------|
| E1 | Web → dllm | HTTP JSON + SSE | 8080 | LAN allow-list (Phase 3) | LIVE (health/node/stats/pairing-uri/models/devices+device_name/heartbeat+load+caps/detail+live-load/`DELETE /v1/devices/{id}` hard-delete with 400 self-guard/sessions list+detail/SSE contract token-commit-status+`?last_event`/plan) | `contracts/openapi.yaml` |
| E2 | Android → dllm | HTTP JSON + SSE (OkHttp) | 8080 | LAN allow-list (Phase 3) | LIVE server-side (incl. `/api/stats`, `/api/node`, `/api/pairing-uri`, `POST /v1/devices/heartbeat` worker lifeline with name/load/caps, `GET /v1/sessions` feed, `GET /v1/devices/{id}` detail, `DELETE /v1/devices/{id}` hard-delete with 400 self-guard, `GET /v1/plan` single-device; SSE parser fixed to contract `{"pos","text"}`/`{"pos"}` with legacy `{"kind","payload"}` unwrap); app scaffolded with stable identity (node_id = deterministic UUID from ANDROID_ID via `IdentityStore.ensureNodeId`, reinstall-safe; pairing flows never rotate it; Use-existing-ID UI adopts a prior id; single shared `buildHeartbeatBody` for all senders) | `contracts/openapi.yaml` |
| E3 | dllm ↔ all | mDNS `_dllm._tcp.local.` TXT(quic_port,node_id,model,ver) | 5353 | none (discovery only) | advertise live; browse stub | `contracts/pairing.md` |
| E4 | Stage N → N+1 | QUIC + mTLS, ALPN `dllm/1`, ActivationFrame v1 | 8443/udp | mutual TLS, TOFU fingerprints | PARTIAL — codec + transport DONE: contract-conformant fixed-width LE frame codec (`dllm-net`, 15/15 unit tests + `frame_roundtrip` pinning exact offsets), mTLS TOFU `server`/`connect` (ADR-018), headless `pipe_pair` on 127.0.0.1:8443. NOT wired: **no stage-level activation execution between real devices — no pipeline run happens**; 2-device PLANNED | `contracts/activation-frame.md`, `acks.md` |
| E5 | dllm → Cloud | HTTPS background queue, zstd+age chunks | 443 | user token, opt-in only | PLANNED Phase 5 | MASTER_PLAN §13 |
| E6 | tray ↔ svc | loopback HTTP (same API) | 8080 | local-only bind | PLANNED (two-binary split) | research/02 §6 |
| E7 | dllm serve (maintenance task) → own SQLite event log | in-process (`spawn_blocking`), no network | — | n/a (self) | LIVE Phase 5: every 60 s prune events older than TTL (24 h default, `DLLM_EVENT_TTL_SECS`) + `PRAGMA wal_checkpoint(TRUNCATE)` | `contracts/event-log.md` |
| E8 | acceptance harness → dllm | HTTP JSON + SSE (curl.exe) | 8099 (ephemeral drill) | loopback only | `scripts/mvp-acceptance.ps1` — starts/stops its own serve | `scripts/mvp-acceptance.ps1` |
| E9 | dllm serve → paired peers (listener) | QUIC + mTLS listener, ALPN `dllm/1` | 8443/udp | mutual TLS; allow-list = registry TOFU (rows with `status == "paired"` AND a non-null non-empty `cert_fp`) | LIVE listener, idle links: `spawn_mesh_server` binds **synchronously** (`0.0.0.0:8443`, verified via `GET /v1/mesh`); each `Incoming` is awaited so the TLS handshake + allow-list check actually runs; monitor tick reaps closed links + refreshes `last_seen`; revoking a device tears its live link down. Bind failure logs + returns a finished stub task so HTTP serving continues (single-device chat must never require the mesh). No stage traffic yet | `contracts/activation-frame.md`, `contracts/pairing.md` |
| E10 | dllm serve → observers (ops/dashboards) | HTTP JSON | 8080 | LAN allow-list (Phase 3) | LIVE `GET /v1/mesh` → `{quic_port, endpoint, peers[], allowed_peer_count}` (peer: `device_id`, nullable `device_name`, `fingerprint`, nullable `rtt_us`, `connected`, `connected_since_ms`, `last_seen_ms`); `rtt_us` stays `null` until a real sample exists; `quic_port:0` + `endpoint:"unbound"` when the mesh did not bind. `/api/stats` also carries `mesh{peers_connected, allowed_peers}` | `contracts/openapi.yaml` |

## Flows

**F1 — Chat send (LIVE Phase 0, mock engine):** `POST /v1/sessions {}` → `{id}` → `POST /v1/sessions/{id}/messages {text}` → server appends events to SQLite log + broadcasts → client reads `GET /v1/sessions/{id}/events` (SSE contract `token {"pos","text"}` + `commit {"pos"}` + `status`, `?last_event=K`/`Last-Event-ID` resume) → token/commit events render.
**F2 — SSE resume (LIVE):** client sends `?last_event=K` (browsers) or `Last-Event-ID` (OkHttp); server replays missed rows from log as contract shapes, then live tail. Basis for view-switching without restart.
**F3 — Discovery + pairing (PARTIAL):** dllm advertises mDNS (live); Android NsdDiscovery scaffolded; QR/OTP + verify-code + pubkey exchange + allow-list = Phase 3 (server) / Phase 4 (app). See `contracts/pairing.md`.
**F4 — Pipeline token (PARTIAL — transport ready, executor NOT wired):** the wire pieces exist and are tested — coordinator binds 8443 (E9) and admits paired peers via mTLS TOFU, frames encode/decode byte-for-byte per `contracts/activation-frame.md` (fixed-width LE, 40-byte header, `DLLM1`, separate `ACK1` namespace for ACKs), ACK vocabulary + optimistic commit/tracker are implemented (`dllm-net`, `dllm-core::commit`), and `dllm_core::plan_layers` computes a real plan id. What does NOT happen yet: no coordinator task dispatches `(seq,pos)`, no stage appends tentative KV, no activation frame is forwarded to a worker, no tail sample returns `COMPUTED(pos,hash)`, no `COMMIT(pos-1)` piggyback or `TRUNCATE(pos)` abort crosses the wire. The `(seq,pos)` dispatch → forward activations → tail `COMPUTED(pos,hash)` → piggyback `COMMIT(pos-1)` / abort `TRUNCATE(pos)` sequence is the target, not current behaviour; proven end-to-end only in the headless `pipe_pair` loopback drill. See `contracts/acks.md`.
**F5 — Recovery (PLANNED):** stop at last COMMIT → replacement worker (has shard + capacity) → re-prefill from last checkpoint (K=64–128) → resume. See research/05.

## Ports / names registry (do not collide)

- `8080` LAN HTTP API + SSE + web dist (dev override `--port`).
- `8099` MVP acceptance drill only (`scripts/mvp-acceptance.ps1` starts/stops its own serve; never a service port).
- `8443/udp` QUIC worker traffic, ALPN `dllm/1` (TXT `quic_port`; dev override). **Really bound now** — `spawn_mesh_server` is called from `dllm serve` and binds `0.0.0.0:8443` synchronously (visible as `endpoint:"0.0.0.0:8443"` on `GET /v1/mesh`); it is a listener with no peers yet, not a working stage link (E4/E9).
- `5353/udp` mDNS (system).
- ALPN `dllm/1`. mDNS type `_dllm._tcp.local.`. Mutex `Global\dllm-coordinator-v1`.

## Local run topology (Phase 0)

`dllm serve --port 8080` (+ mDNS advertise + QUIC 8443 mesh listener, no stage traffic) → Web dev `npm run dev` in `apps/web` (proxy `/api,/v1 → :8080`, added Phase 1) or `apps/web/dist` served by dllm (built) → Android app pointed at `http://<lan-ip>:8080` (cleartext note: `usesCleartextTraffic=false` needs LAN exception or HTTPS in Phase 1 — see android README).

## Per-phase update checklist

- [ ] New/changed edge → row above + contract ref + status flip LIVE/PLANNED.
- [ ] Flow sequence still accurate;reno on protocol change.
- [ ] Ports/ALPN/mDNS names unchanged or registry updated.
- [ ] `docs/BUILD_STATUS.md` artifact rows match.
