# 02 — Windows Coordinator Rust Stack Research (verified 2026-09-13 via crates.io API + docs.rs)

## 1. llama.cpp bindings — `llama-cpp-2 0.1.156` (current best)

Repo: `utilityai/llama-cpp-rs`. Alternatives rejected: `llama-rs` (unmaintained), `llama-cpp` v1 (stale), `candle`/`burn` (pure Rust, easy layer control, but 3–10× slower on CPU, weak GGUF/quant + sampler parity).

- Load: `LlamaBackend::init()` once → `LlamaModel::load_from_file(&backend, path, &LlamaModelParams::default().with_n_gpu_layers(n))` → `model.new_context(&backend, LlamaContextParams::default().with_n_ctx(n).with_n_batch(b).with_n_threads(t))`. GGUF metadata via `model.n_layer()/n_embd()/n_vocab()/meta_val_str()` or `llama_cpp_2::gguf` module.
- Tokenize/prefill/decode: `model.str_to_token(text, AddBos::Always)` → `LlamaBatch::new(n)` + `add(token, pos, seq_ids, logits_flag)` → `ctx.decode(&mut batch)`, loop `sampler.sample(&ctx, idx)` → `accept(token)` → `token_to_piece`, append, next decode. Sampler: `LlamaSampler::chain([top_k, top_p, min_p, temp, dist(seed)], false)` or `greedy()`.
- Features: `sampler`, `cuda`/`vulkan`/`dynamic-backends`. Windows CUDA needs CMake + clang (bindgen) + CUDA toolkit at build time; `GGML_BACKEND_DL` / `load_backends_from_path()` for dynamic GPU backends.

### Pitfalls (including the biggest architectural risk)

- **NO layer-range execution in the public C API, therefore not in `llama-cpp-2`.** Only `n_gpu_layers`, `main_gpu`, `tensor_split`, `split_mode` (CPU/GPU offload split) — not "run layers 0–12 only". True pipeline needs: (a) upstream `ggml-rpc` backend sidecars, (b) forking `llama-cpp-sys-2` to drive internal `llama_build_graph` (unstable), or (c) `candle` where you own the layer loop. **Design default: data-parallel replicas + coordinator routing; defer layer-pipeline to an RPC experiment spike in Phase 3.**
- `LlamaModel: Send+Sync`; `LlamaContext` single-owner, not concurrent. One context per worker behind `Mutex`; never `decode` on a Tokio thread — `spawn_blocking` only.
- Build weight: bindgen + C++ compile = minutes; pin `llama-cpp-2 = "=0.1.156"`.
- KV/context overflow is your bug: manage `n_ctx`, `seq_id`, `llama_memory_seq_rm`/state-save explicitly.

## 2. QUIC + mTLS — `quinn 0.11.11` + `rustls 0.23.44` + `tokio-rustls 0.26.5` + `rcgen 0.14.10`

`quinn 0.11.x` requires `rustls ^0.23`. Do NOT use `rustls 0.24.0-dev.x`.

- mTLS pattern: server `RootCertStore + WebPkiClientVerifier` → `ServerConfig` → `QuicServerConfig::try_from`; client `with_root_certificates + with_client_auth_cert` → `QuicClientConfig::try_from`. Matching `alpn_protocols` (e.g. `b"dllm/1"`); mismatched ALPN = silent handshake fail.
- Bidi streams: `conn.open_bi().await` / `accept_bi().await`; frame with `u32-LE length + postcard/serde_json` or `LengthDelimitedCodec`. One control stream + N data streams per peer.
- Priorities: `SendStream::set_priority` is a sender-side hint, not QoS. App-level priority (control > tokens > bulk) via separate streams + own scheduler.
- Certs: `rcgen` dev CA + per-node leaf (SAN = hostname + LAN IPs), persist in SQLite/exe dir; `x509-parser` for fingerprint logging.

Pitfalls: `QuicClientConfig::try_from` TLS1.3-only + ring quirks (read `inner` docs); **disable 0-RTT on LAN** (saves <1 ms, adds replay risk); SNI must match SAN (IP SANs when dialing by IP); test Windows UDP egress + `rebind()` behind VPNs.

## 3. Axum LAN REST/SSE — `axum 0.8.9` + `tower-http 0.7.1` + `rust-embed 8.12.0`

- SSE: `Sse::new(BroadcastStream…).keep_alive(KeepAlive 15s)`; source from `tokio::sync::broadcast` or `async-stream` bridging inference `mpsc`. `Event::default().json_data(v).event("token").id(seq)`.
- Static UI: single-exe via `rust-embed` (`#[folder="web/dist"]`) + SPA fallback; dev mode `ServeDir`. Explicit MIME; `Cache-Control: no-cache` for `/api/*`.
- State: `Arc<AppState>` via `with_state()`. Bind `0.0.0.0:PORT`; advertise same port in mDNS TXT.

Pitfalls: no `CompressionLayer`/`BufferLayer` in front of SSE; client disconnect must cancel inference (CancellationToken) or GPU work leaks; Axum 0.8 needs `tokio ^1.44`, `tower ^0.5`, `http 1.x`.

## 4. mDNS — `mdns-sd 0.21.3`

- One `ServiceDaemon::new()` (own thread, `Clone`, `flume` works sync+async). Advertise `ServiceInfo::new("_dllm._tcp.local.", instance, host.local., ip, api_port, txt)` with TXT `quic_port, node_id, model, ver`. Browse → `ServiceFound → ServiceResolved → ServiceRemoved`.
- Unique instance per node; handle `DnsNameChange`; publish both ports in TXT (SRV holds one port only). Debounce `ServiceResolved` (multi-NIC dups); 5s expiry sweeper.
- Windows pitfalls: firewall kills mDNS silently (inbound UDP 5353 + app ports); VPNs/Hyper-V/sleep-wake change interfaces (re-`register()`, `IP_CHECK_INTERVAL`); strict `.local.` + `._tcp.local.` formats.

## 5. SQLite event log — `rusqlite 0.40.2` (feature `bundled`)

- `PRAGMA journal_mode=WAL; synchronous=NORMAL; busy_timeout=5000; foreign_keys=ON`. Schema `events(rowid AUTOINCREMENT, id TEXT UNIQUE, session, kind, payload JSON, ts_utc, prev_hash)` + `INSERT OR IGNORE`.
- Single writer: `Arc<Mutex<Connection>>` + `spawn_blocking`, or dedicated writer thread with queue. Batch ~500/txn. Separate read connections (WAL allows concurrent readers).
- Pitfalls: always `spawn_blocking` (Connection is Send but blocking); one process one writer (tray vs service talk over loopback, not shared writable DB); bundled SQLite ≥ 3.51.3 (WAL-reset race fix); optional `BEFORE UPDATE/DELETE` triggers enforcing append-only.

## 6. Windows service + autostart + tray + single-instance

Crates: `windows-service 0.8.1`, `tray-icon 0.25.0` + `muda`, `tao 0.37.0` (event loop), `single-instance 0.3.3`, `winreg`.

- **Split processes:** `dllm-service` (Session 0, SCM, auto-start, Event Log) + `dllm-tray` (user session, tray icon). Service cannot show tray. IPC over loopback Axum + mTLS or named pipe.
- Tray icon must be created on the event-loop thread; PNG via `include_bytes!`.
- Single-instance: global named mutex `Global\dllm-coordinator-v1` or `single-instance` guard; autostart via `HKCU\…\Run` (desktop mode) or SCM (service mode).

Pitfall #1: Session 0 isolation — tray code in a service silently never appears. Test two binaries from day one.

## 7. Backup pipeline — `zstd 0.14.0` + `age 0.12.1` (preferred)

`tar/checkpoint → zstd → age (X25519)`. `age` already uses chacha20poly1305 internally; raw `chacha20poly1305 0.11` only for custom chunk-AEAD (then XChaCha20, random 24B nonce per file). Keys: `age-keygen` X25519; recipient in config, identity on restore host. Round-trip CI test mandatory. `age` has no signing — ship separate `ed25519` signature if cloud storage untrusted.

## 8. CLI + Tokio — `clap 4.6.6` + `tokio 1.53.1`

Subcommands: `serve / join / backup / keygen / service {install|uninstall|start}`. Runtime: `multi_thread, worker_threads = cores-2`; **all** llama `decode` in `spawn_blocking` or dedicated thread pool (one `LlamaContext` per thread via channel); inference threads = physical cores, leave 1–2 for Tokio/QUIC/Axum. `tracing` + `send_logs_to_tracing()` unifies C++ logs. Heap-allocate `LlamaBatch` (Windows stack overflow risk).

## Cargo workspace sketch

```toml
[workspace]
resolver = "2"
members = ["crates/dllm-core", "crates/dllm-net", "crates/dllm-serve",
           "crates/dllm-store", "crates/dllm-svc", "crates/dllm-cli"]
# MSRV stable ≥1.82. Build deps: clang/LLVM (bindgen), CMake, Ninja; CUDA optional.

[workspace.dependencies]
llama-cpp-2 = "=0.1.156"
llama-cpp-sys-2 = "=0.1.156"
quinn = "0.11"              # needs rustls 0.23
rustls = "=0.23.44"         # NOT 0.24-dev
tokio-rustls = "0.26"
rustls-pemfile = "2"
rcgen = "0.14"
x509-parser = "0.18"
sha2 = "0.11"
blake3 = "1.8"
axum = { version = "0.8", features = ["tokio", "json", "query"] }
tower-http = { version = "0.7", features = ["fs", "cors", "trace"] }
tokio-stream = "0.1"
async-stream = "0.3"
rust-embed = "8"
mime = "0.3"
mdns-sd = "0.21"
rusqlite = { version = "0.40", features = ["bundled"] }
windows-service = "0.8"
tray-icon = "0.25"
muda = "0.19"
tao = "0.37"
single-instance = "0.3"
winreg = "0.55"
windows-sys = "0.61"
zstd = "0.14"
age = { version = "0.12", features = ["armor", "async"] }
chacha20poly1305 = "0.11"
zeroize = "1"
clap = { version = "4.6", features = ["derive", "env"] }
tokio = { version = "1.53", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
postcard = "1"              # compact QUIC framing
flume = "0.11"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
anyhow = "1"
thiserror = "2"
```

`dllm-core` owns inference behind a `spawn_blocking` pool; `dllm-net` owns Quinn mTLS + framing + ALPN `dllm/1`; `dllm-store` owns WAL event log + checkpointed backup (`zstd → age`).
