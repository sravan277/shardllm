# contracts/ — Phase 0 frozen interfaces

Single source of truth for cross-target wire formats. If code and contracts
disagree, contracts win; change requires a contract rev + stub recompile.

| File | What | Consumed by |
|---|---|---|
| `manifest.schema.json` | JSON Schema (draft 2020-12) for `model-manifest.json`: arch, shards, hashes, tokenizer, sampling, runtime compat | Rust `crates/model-tools` (split/sign/verify), `apps/windows-svc` (fetch + verify), Android (shard-size display), Web (library UI) |
| `catalog.json` | Curated model list (no arbitrary URLs in MVP). Pins GGUF URL + SHA + default 9/10/9 plan + KV budgets | Rust svc (`dllm pull/run/list`), Android models screen, Web library |
| `activation-frame.md` | `ActivationFrame` v1 binary layout (`DLLM1`, LE, 8 MiB cap) between pipeline stages | Rust `crates/protocol` (encode/decode), Android JNI forwarder (same byte order) |
| `acks.md` | Inference ACK vocabulary + optimistic primary-backup flow (piggyback COMMIT, TRUNCATE abort), tail-sampler rule, seq/pos | Rust `crates/protocol` + `crates/coordinator` state machine; Android worker ACK sender |
| `pairing.md` | mDNS + QR/OTP handshake, verify-code, pubkey exchange, allow-list record, revocation | Rust svc (registry/mTLS), Android pairing UI, Web pairing management |
| `openapi.yaml` | LAN HTTP API (health, models, sessions, SSE events, devices, plan) | Rust svc (Axum impl), Web (fetch/SSE), Android (Cronet/OkHttp), `dllm` CLI |
| `event-log.md` | SQLite event-log DDL (WAL, hash chain), SSE JSON shapes, prune rule | Rust `crates/coordinator` (log), all clients (resume from `lastEventId`) |

## Rules

- `P<=3` stages on Wi-Fi. Default split 9/10/9 (layers 0–8 / 9–18 / 19–27).
- KV block 16 tokens, recompute-preempt (never swap over LAN).
- Cross-ISA contract is token-level: `max|Δlogit|<0.5` + top-1 >99.5% @ batch-1/temp-0.
- Transports may differ (QUIC on Windows svc, TCP+TLS on Android MVP) but
  `ActivationFrame` bytes, ACK strings, and OpenAPI shapes are identical.
- Wire dim between stages is always 1024 (`[batch, seq, 1024]`).
