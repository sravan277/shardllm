# contracts/ — Phase 0 frozen interfaces

Single source of truth for cross-target wire formats. If code and contracts
disagree, contracts win; change requires a contract rev + stub recompile.

| File | What | Consumed by |
|---|---|---|
| `manifest.schema.json` | JSON Schema (draft 2020-12) for `model-manifest.json`: arch, shards, hashes, tokenizer, sampling, runtime compat | **No code consumes it yet** — the split/sign/verify tooling it was written for does not exist (the old entry named `crates/model-tools` + `apps/windows-svc`, both fictional). Intended: the future shard splitter/signer, then Android (shard-size display) and Web (library UI) |
| `catalog.json` | Curated model list (no arbitrary URLs in MVP). Pins GGUF URL + SHA + default 9/10/9 plan + KV budgets | Rust `apps/dllm` (`include_str!` → `cmd_pull`/`cmd_run`/`cmd_list`) + `crates/dllm-serve` (`include_str!` → `GET /api/models`) + `crates/dllm-core` (sampler chain per catalog, `wire_dim` in `plan.rs`), Android models screen, Web library |
| `activation-frame.md` | `ActivationFrame` v1 binary layout — fixed-width little-endian, 40-byte header, magic `DLLM1` at offset 0, explicit `payload_length` u32 at 36, 8 MiB cap, no varints and **no postcard**. ACKs are a separate namespace: `ACK1` + u32-LE length + JSON | Rust `crates/dllm-net` (`encode_frame`/`decode_frame`/`encode_frame_header`/`decode_frame_header`, `FrameHeader`; `encode_ack`/`decode_ack`), Android JNI forwarder (same byte order, `ByteBuffer` only) |
| `acks.md` | Inference ACK vocabulary + optimistic primary-backup flow (piggyback COMMIT, TRUNCATE abort), tail-sampler rule, seq/pos | Rust `crates/dllm-net` (ACK codec + `ACK1` framing) + `crates/dllm-core/src/commit.rs` (commit tracker / `on_ack` routing); Android worker ACK sender |
| `pairing.md` | mDNS + QR/OTP handshake, verify-code, pubkey exchange, allow-list record, revocation | Rust `crates/dllm-store` (device registry + `cert_fp` TOFU row), `crates/dllm-net/src/transport.rs` (mTLS), `crates/dllm-serve/src/mesh.rs` (mesh allow-list projection), `apps/dllm` (mDNS advertise + `id`); Android pairing UI, Web pairing management |
| `openapi.yaml` | LAN HTTP API (health, models, sessions, SSE events, devices, plan, usage, mesh status) | Rust `crates/dllm-serve` (Axum impl, hosted by `apps/dllm serve`), Web (fetch/SSE), Android (Cronet/OkHttp), `dllm` CLI |
| `event-log.md` | SQLite event-log DDL (WAL, hash chain), SSE JSON shapes, prune rule | Rust `crates/dllm-store` (log) + `crates/dllm-serve` (SSE writer + replay), all clients (resume from `lastEventId`) |

## Rules

- `P<=3` stages on Wi-Fi. Default split 9/10/9 (layers 0–8 / 9–18 / 19–27).
- KV block 16 tokens, recompute-preempt (never swap over LAN).
- Cross-ISA contract is token-level: `max|Δlogit|<0.5` + top-1 >99.5% @ batch-1/temp-0.
- Transports may differ (QUIC + mTLS between Rust coordinator/workers, TCP+TLS on
  Android MVP) but `ActivationFrame` bytes, ACK strings, and OpenAPI shapes are
  identical.
- Wire dim between stages is always 1024 (`[batch, seq, 1024]`).
- Binary frames are **fixed-width little-endian, hand-rolled — not a
  serialization framework** (no varints, no postcard). A non-Rust peer (the
  Android JNI forwarder) must be able to encode byte-for-byte from the tables in
  `activation-frame.md` / `acks.md` using only fixed-width puts, so the wire can
  never depend on a Rust-only encoding rule (ADR-029).
- ACK frames do **not** reuse the `DLLM1` activation magic: they are `ACK1 ||
  u32-LE length || JSON`, so a stray ACK on a data stream can never be parsed as
  an activation frame.
