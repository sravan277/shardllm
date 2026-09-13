# How Everything Connects (LIVING DOC — update every phase)

> Last updated: Phase 5 (2026-09-13). This is the global wiring map: which component talks to which, over what protocol/port, with what auth, and what works *today* vs planned. Update the edge table and flows on every phase. Rule: no new connection without a row here + a contract ref.

## Map

```text
                    mDNS _dllm._tcp.local. (discovery only)
   ┌──────────┐  ──────────────────────────────  ┌──────────────────┐
   │ Android  │   HTTP/SSE  :8080 (Phase 0 live)  │  dllm serve      │
   │ chat app │  ──────────────────────────────► │  (Windows, Rust) │
   │          │   QUIC dllm/1 :8443 (Phase 3)     │  coordinator     │
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

## Edge table

| # | From → To | Protocol | Port | Auth | Phase 0 status | Contract |
|---|-----------|----------|------|------|----------------|----------|
| E1 | Web → dllm | HTTP JSON + SSE | 8080 | LAN allow-list (Phase 3) | LIVE (health/models/node/stats/sessions/SSE) | `contracts/openapi.yaml` |
| E2 | Android → dllm | HTTP JSON + SSE (OkHttp) | 8080 | LAN allow-list (Phase 3) | LIVE server-side (incl. `/api/stats`, `/api/node`); app scaffolded | `contracts/openapi.yaml` |
| E3 | dllm ↔ all | mDNS `_dllm._tcp.local.` TXT(quic_port,node_id,model,ver) | 5353 | none (discovery only) | advertise live; browse stub | `contracts/pairing.md` |
| E4 | Stage N → N+1 | QUIC + mTLS, ALPN `dllm/1`, ActivationFrame v1 | 8443/udp | mutual TLS, TOFU fingerprints | DONE headless (`pipe_pair` loopback 127.0.0.1:8443); 2-device PLANNED | `contracts/activation-frame.md`, `acks.md` |
| E5 | dllm → Cloud | HTTPS background queue, zstd+age chunks | 443 | user token, opt-in only | PLANNED Phase 5 | MASTER_PLAN §13 |
| E6 | tray ↔ svc | loopback HTTP (same API) | 8080 | local-only bind | PLANNED (two-binary split) | research/02 §6 |
| E7 | dllm serve (maintenance task) → own SQLite event log | in-process (`spawn_blocking`), no network | — | n/a (self) | LIVE Phase 5: every 60 s prune events older than TTL (24 h default, `DLLM_EVENT_TTL_SECS`) + `PRAGMA wal_checkpoint(TRUNCATE)` | `contracts/event-log.md` |
| E8 | acceptance harness → dllm | HTTP JSON + SSE (curl.exe) | 8099 (ephemeral drill) | loopback only | `scripts/mvp-acceptance.ps1` — starts/stops its own serve | `scripts/mvp-acceptance.ps1` |

## Flows

**F1 — Chat send (LIVE Phase 0, mock engine):** `POST /v1/sessions {}` → `{id}` → `POST /v1/sessions/{id}/messages {text}` → server appends events to SQLite log + broadcasts → client reads `GET /v1/sessions/{id}/events` (SSE, `Last-Event-ID` resume) → token/commit events render.
**F2 — SSE resume (LIVE):** client sends `Last-Event-ID`; server replays missed rows from log, then live tail. Basis for view-switching without restart.
**F3 — Discovery + pairing (PARTIAL):** dllm advertises mDNS (live); Android NsdDiscovery scaffolded; QR/OTP + verify-code + pubkey exchange + allow-list = Phase 3 (server) / Phase 4 (app). See `contracts/pairing.md`.
**F4 — Pipeline token (PLANNED Phase 3):** coordinator dispatches `(seq,pos)` → stages append tentative KV, forward activations over E4 → tail samples once → `COMPUTED(pos,hash)` → coordinator piggybacks `COMMIT(pos-1)` on next dispatch; abort = `TRUNCATE(pos)`. See `contracts/acks.md`.
**F5 — Recovery (PLANNED):** stop at last COMMIT → replacement worker (has shard + capacity) → re-prefill from last checkpoint (K=64–128) → resume. See research/05.

## Ports / names registry (do not collide)

- `8080` LAN HTTP API + SSE + web dist (dev override `--port`).
- `8099` MVP acceptance drill only (`scripts/mvp-acceptance.ps1` starts/stops its own serve; never a service port).
- `8443/udp` QUIC worker traffic, ALPN `dllm/1` (TXT `quic_port`; dev override).
- `5353/udp` mDNS (system).
- ALPN `dllm/1`. mDNS type `_dllm._tcp.local.`. Mutex `Global\dllm-coordinator-v1`.

## Local run topology (Phase 0)

`dllm serve --port 8080` (+ mDNS advertise) → Web dev `npm run dev` in `apps/web` (proxy `/api,/v1 → :8080`, added Phase 1) or `apps/web/dist` served by dllm (built) → Android app pointed at `http://<lan-ip>:8080` (cleartext note: `usesCleartextTraffic=false` needs LAN exception or HTTPS in Phase 1 — see android README).

## Per-phase update checklist

- [ ] New/changed edge → row above + contract ref + status flip LIVE/PLANNED.
- [ ] Flow sequence still accurate;reno on protocol change.
- [ ] Ports/ALPN/mDNS names unchanged or registry updated.
- [ ] `docs/BUILD_STATUS.md` artifact rows match.
