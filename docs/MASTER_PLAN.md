# Distributed LAN LLM — MVP Master Plan

> Noted: 2026-09-13. Single source of truth for architecture, scope, UX, UI direction, environment setup, and phased build order.
> Stack locked: Windows `Rust + llama.cpp + Quinn` / Android `native Kotlin + Compose` / Web `Vite + React + TS`. Order locked: single-device fast path first.

## 1. Purpose

Build an Ollama-like, local-first platform that runs an LLM across up to five trusted Windows and Android devices on the same LAN. Devices collectively hold model weights and participate in inference according to measured compute, RAM, storage, and network characteristics.

Cloud services store account/setup metadata and user-opt-in encrypted session backups only. They never receive inference activations, KV cache, model weights, or perform model computation.

## 2. Locked decisions

| Area | Decision |
| --- | --- |
| Network scope | LAN only; no remote internet inference in MVP. |
| Compute method | Pipeline parallelism plus batching. |
| Initial model | Curated quantized Qwen3-0.6B, architecture supports larger models. |
| Model storage | Each device downloads only its assigned layer shards (shard-only; phones save storage/battery). Full 0.6B fallback practical on coordinator/spare. |
| Device count | Up to five paired devices in an active mesh. |
| Platforms | Windows native server/worker and Android native client/worker. |
| Windows role | Headless LAN server, pipeline coordinator, model manager, optional worker; tray + CLI, not a native chat app. |
| Android role | Primary chat client, pairing UI, device controls, optional inference worker (ForegroundService + wake lock from day one). |
| Browser role | Local web chat/dashboard served by the Windows server; cloud dashboard exposes metadata only. |
| Cloud role | User/device setup, pairing metadata, telemetry history, opt-in encrypted chat backups. No inference. |
| Backup | Explicitly user-enabled; opt-out respected. 10-day retention for backups + telemetry via automatic job. |
| Session recovery | Stage-local KV cache; recover by re-prefilling committed transcript (since last checkpoint) to a replacement plan. |

## 3. System topology

```text
                             Cloud
        account/device metadata · encrypted optional backups
                    telemetry history · 10-day retention
                                  ^
                                  | asynchronous, low priority
                                  |
 +------------------- Same LAN ------------------------------------------------+
 |                                                                          |
 |  Windows server / coordinator                                            |
 |  - device registry and scheduler                                         |
 |  - local API and local web UI                                            |
 |  - session event log                                                     |
 |  - optional model-layer worker                                           |
 |       |                         |                                        |
 |       | control links            | direct pipeline activation links      |
 |       v                         v                                        |
 |  Laptop worker              Android worker / chat client                |
 |                                                                          |
 +--------------------------------------------------------------------------+
```

Coordinator has a control connection to each paired device. Inference pipeline is NOT a full mesh: each stage sends activations only to its immediate next stage.

```text
Stage A (layers 0-15) -> Stage B (layers 16-23) -> Stage C (layers 24-27)
```

### Why pipeline parallelism

Model may not fit in any single device's RAM. Each device gets a contiguous layer range + proportional KV-cache share. Scheduler assigns ranges by measured stage time (calibration bench at pairing), not equal counts — faster devices get more layers.

Single-token decode stays sequential (causal-transformer constraint):

```text
input token -> Stage A -> activation -> Stage B -> activation -> Stage C -> next token
```

Throughput comes from batching: prefill chunk streaming, continuous multi-session interleave; speculative decoding deferred. Tensor parallelism deferred (too many per-layer barriers for heterogeneous Wi-Fi).

## 4. Architecture review fixes (adopted, not optional)

1. **Single-device fast path + latency baseline BEFORE pipeline.** Pipeline adds N-1 activation hops + commit RTT per token (50–150 ms on Wi-Fi) for a single chat, and 0.6B fits one device. Ship working single-device chat first; pipeline must beat/is justified against the baseline.
2. **KV checkpoints every 256–512 committed tokens** to coordinator/spare disk. Recovery replays since checkpoint, bounding stall (full-transcript replay at token 3,800 is unacceptable).
3. **LAN reality:** AP/client-isolation diagnostic at pairing (distinguish policy-block from not-found); Android worker uses ForegroundService + wake lock from day one (Doze/OEM killers).
4. **Scheduler cold start:** synthetic layer-timing calibration bench at pairing; no planning on missing history.
5. **Coordinator SPOF explicit:** coordinator death = full session loss, no failover in MVP (no consensus for 5-node mesh).
6. **GGUF shard tooling is a first-class item:** split shared tensors vs layers, hash-sign manifest, verify on fetch.
7. **Numeric consistency test:** same prompt single-device vs heterogeneous split; diff logits, not just text (ARM vs x86 quant drift).

## 5. Model packaging and placement

GGUF only for MVP. Signed manifest + quantized independently verifiable shards:

```text
model-manifest.json
shared-tensors.gguf
layers-00-to-15.gguf-shard
layers-16-to-23.gguf-shard
layers-24-to-27.gguf-shard
```

Manifest: architecture, tokenizer, quantization, shard hashes, compatible runtime version, layer boundaries. Devices prefetch only assigned shard(s). Scheduler never assigns uninstalled layers. Coordinator + ≥1 spare pre-cache enough shards for recovery without mid-chat downloads.

## 6. Inference flow

### 6.1 Session creation

1. Chat client submits request to Windows coordinator over LAN.
2. Coordinator validates paired-device permissions, creates `session_id`.
3. Selects workers with compatible shards + RAM/KV capacity + acceptable RTT/jitter.
4. Creates immutable `pipeline_plan_id` (worker order + layer ranges).
5. Reserves KV-cache pages at every selected stage.

### 6.2 Prompt prefill

1. Coordinator tokenizes prompt.
2. Stage A processes prompt-token chunks through owned layers, writes K/V, forwards hidden activations to Stage B; repeat per stage.
3. Final stage produces logits, samples first token locally; returns token ID + status.

Only activations, token IDs, control messages cross the LAN. Weights and KV cache do not.

### 6.3 Decode (per token)

1. Stage A reads own KV, computes layer range, appends tentative K/V, forwards activation; repeat downstream.
2. Final stage samples next token, sends token ID to coordinator.
3. All stages ack tentative cache update.
4. Coordinator records + broadcasts `COMMIT(token_position)` only after every stage succeeds.
5. Coordinator streams committed tokens to Android + web clients.

Final-stage local sampling avoids shipping full-vocabulary logits.

## 7. KV-cache design

Distributed by owned layer range (`Stage A: KV[0-15]`, etc.). Paged local allocator, 16/32-token pages, per-session budget. FP16 first for correctness; Q8 after validation. ~112 KiB/token for all 28 layers of Qwen3-0.6B (≈448 MiB at 4K tokens); each device holds its proportional share.

Failure recovery: stop at last committed token, discard tentative entries, pick replacement with needed shard + capacity, re-prefill from last checkpoint through replacement pipeline, resume streaming. No device with shard + capacity = session waits.

## 8. LAN networking and packets

QUIC + mTLS for all device-to-device traffic (encryption, retransmission, ordering, congestion control, stream multiplexing). No raw-UDP reliability. mDNS discovery-only; no multicast inference.

| Traffic | Mechanism | Priority |
| --- | --- | --- |
| Discovery | mDNS multicast | Low |
| Pairing, control, cancellation | Reliable QUIC control stream | Highest |
| Pipeline activations | Dedicated reliable ordered QUIC stream per stage edge/session | Highest |
| Token commit, stage status | Reliable QUIC control stream | Highest |
| Health/capability telemetry | Batched low-rate QUIC stream | Low |
| Model-shard prefetch | Resumable bulk stream | Lowest |
| Cloud session backup | Separate background internet queue | Lowest |

Application frames (bounded + backpressure; slow worker must not cause unbounded buffering):

```text
ActivationFrame {
  session_id,
  pipeline_plan_id,
  token_position,
  source_stage,
  target_stage,
  tensor_format,
  payload_length,
  activation_bytes
}
```

Inference-level ACKs (transport ACK ≠ inference ACK):

```text
RECEIVED(token_position)
COMPUTED(token_position)
KV_TENTATIVE(token_position)
COMMITTED(token_position)   # coordinator only
```

## 9. Pairing and device membership

Methods: QR invitation from any trusted paired device, short-lived OTP code, mDNS nearby discovery + explicit approval, approve/revoke in Android + web UI.

Flow: discover coordinator → one-time QR/code → both show short verify code → user confirms → exchange long-term pubkeys → coordinator registers identity/role/permissions/capabilities. Future connections use mTLS + allow-list. Per-device revocable credentials; no shared LAN password.

## 10. Plug-and-play (Ollama-like, curated catalog)

User picks a model name; system hides shards. Catalog is baked-in curated list only (no arbitrary GGUF URLs in MVP).

| Ollama | `dllm` equivalent (CLI `dllm.exe` + same button in Web/Android) |
| --- | --- |
| `ollama pull qwen3:0.6b` | `dllm pull qwen3-0.6b-q4` → fetch signed manifest → assign layers from calibration → resumable background prefetch → hash-verify → `ready` |
| `ollama run` | `dllm run qwen3-0.6b-q4` → auto-pull if missing → create `session_id` → attach + stream |
| `ollama list / ps` | `dllm list` (catalog + installed state) / `dllm ps` (active plan, per-stage latency, bottleneck) |
| `ollama rm` | `dllm rm` → GC unassigned shards, respect storage pressure |

Model state machine: `missing → fetching → verifying → ready → serving`, with progress over SSE to CLI/Web/Android. Windows tray + autostart-on-login; Android Wi-Fi-only default, pause on memory/LAN pressure, auto-resume. Offline-first after prefetch; cloud never hosts weights.

## 11. Apps and user interfaces

### 11.1 Windows native server

Autostart/headless coordinator; models/shards/worker runtime/scheduling/sessions; authenticated LAN API + serve Web `dist/`; tray (green/yellow/red) + QR/OTP display; compact ops view (NOT full chat). Full `dllm` CLI parity driving the same API.

### 11.2 Android native app

Primary chat (stream, attach to `session_id`, switch devices without restart, reconnect via last event ID); pairing (scan/show QR, OTP, mDNS list, verify-code, approve); worker toggle + assigned layers/RAM/KV/perf view; models screen (shard size/state/progress).

### 11.3 Local web UI

Chat + dashboard (active plan, stage latency, bottleneck, KV pages, tok/s, RTT/jitter/loss, queue, recovery + upload-queue states); model library (install/run/remove + progress + storage); pairing management; `session_id + lastEventId` reconnect.

### 11.4 Cloud-hosted dashboard

Metadata only: devices, status history, aggregate perf, backup/retention state. No plaintext chats, no inference relay.

### 11.5 UI design direction (beautiful, distinctive, not templated)

Subject = local mesh inference hardware. One memorable element per surface, everything else quiet.

- **Palette (LAN-signal theme):** deep ink `#101418`, panel `#171D24`, mesh-teal `#2DD4BF` (single accent for live pipeline flow), warm amber `#F5B544` for bottleneck/warning only, text `#E8EDF2` / muted `#93A1B0`.
- **Type:** `Space Grotesk` display/data + `Inter` body. Sentence case; no all-caps eyebrows, no `A · B · C` meta strings, no `→` buttons.
- **Layouts:** Web = left rail (sessions/devices) + center stream + right pipeline strip (`A → B → C` with live per-stage ms + one flowing activation pulse, honors reduced-motion). Android = chat-first with bottom-sheet device/worker panel. Windows = compact ops popover.
- **Copy voice:** plain verbs, action = result (`Install` → `Installed`); errors state cause + fix; empty states invite action. Motion only answers user action or the single pipeline pulse.

## 12. Seamless chat switching

Any client attaches to the same `session_id` on the coordinator; durable local append-only event log replays missed committed tokens from `lastEventId`. View switching never moves pipeline/KV. Coordinator migration and live KV migration deferred.

## 13. Cloud data, backup, retention

Cloud: account/device metadata, capability/usage history, session metadata, opt-in encrypted compressed backups. Never weights/KV/activations/plaintext prompts.

Pipeline: session events → local log → serialize → zstd → client-encrypt → chunked resumable throttled upload. Durable local queue, isolated from inference loop + LAN transport; pauses on inference activity/memory pressure/LAN degradation. Disabled = zero chat upload. 10-day expiry job for backups + telemetry.

## 14. Observability

Batched low-rate deltas: CPU/GPU throughput, free RAM/storage, shards/layers, KV pages/utilization, prefill/decode latency, tok/s, RTT/jitter/bandwidth/loss/retries, queue depth, health. Scheduler consumes pre-request; dashboard shows plan, stage latency, bottleneck, recovery events, upload queue.

## 15. Contracts (Phase 0 — unblocks all 3 apps)

- `ActivationFrame` layout + ACK vocabulary (§8).
- Manifest + catalog JSON-schema + model state machine (§5, §10).
- LAN API: `POST /v1/sessions`, `SSE /v1/sessions/{id}/events` (`lastEventId`), pairing handshake spec, event-log format.
- Exit: mocks/stubs compile on all targets.

## 16. Environment setup

Machine today: VS Code + Node 24 + Java 25. Missing: Python, Rust, CMake, Android SDK, Ollama.

- **Common:** Git, Python 3.12, CMake 3.28+ + Ninja, VS Build Tools 2022 (Desktop C++ workload), Ollama (baseline reference). VS Code: `rust-analyzer`, `Even Better TOML`.
- **Windows/Rust:** `rustup` stable MSVC toolchain; crates `tokio, axum, quinn + rustls, mdns-sd, rusqlite, llama-cpp-2`. Verify: `rustc --version`, `cargo --version`, `cmake --version`.
- **Web:** ready now — Vite 6 + React 18 + TS 5 (+ Tailwind optional). Output `dist/` embedded/served by Axum.
- **Android (defer to Phase 4):** Android Studio + SDK Platform 34 + Build-Tools + NDK r26 + ADB + 1 physical test device (same Wi-Fi). Libs: Compose + CameraX + NsdManager + DataStore; llama.cpp `.so` via JNI; Cronet/quiche-JNI QUIC client.

## 17. Repo layout

```text
MVP_1/
  docs/MASTER_PLAN.md         # this file
  contracts/                  # frames, ACKs, manifest/catalog schemas, OpenAPI, pairing spec
  crates/protocol             # Rust: frames, commit state machine
  crates/coordinator          # sessions, planner, SQLite event log, scheduler
  crates/model-tools          # GGUF splitter + signer/verifier + calibration bench
  apps/windows-svc            # Axum LAN API + static web serve + tray + dllm CLI
  apps/web                    # Vite+React+TS chat + dashboard (builds to windows-svc dist/)
  apps/android                # Kotlin+Compose chat + worker
  models/.gitignore           # weights never committed
```

## 18. Build order + exit criteria

- **Phase 0 — Contracts + layout.** Exit: stubs compile everywhere.
- **Phase 1 — Windows fast path.** Full 0.6B in-process via llama.cpp, REST+SSE, SQLite log, CLI `pull/run/list/ps`, bench harness. Exit: local chat works; baseline tok/s recorded.
- **Phase 2 — Local web UI.** Chat + dashboard skeleton + library + pairing UI. Exit: E2E chat + missed-token replay proven.
- **Phase 3 — Distribution.** Splitter/signer, Quinn mTLS streams + priorities, calibration-driven planner, COMMIT protocol, paged KV + backpressure, prefill + multi-session batching. Exit: 2-device pipeline functional.
- **Phase 4 — Android chat-first, then worker.** Pairing + attach + stream + toggles → JNI shard execution + FGS + stats. Exit: phone chats, then computes.
- **Phase 5 — Hardening.** Checkpoints, recovery/AP-diagnostic/numeric tests, backup queue + TTL, telemetry. Exit: MVP acceptance.

## 19. MVP boundaries

Included: trusted LAN mesh ≤5 devices; GGUF shards + pipeline inference; prefill + multi-session batching; stage-local paged KV + checkpointed re-prefill recovery; secure pairing, capability monitoring, UI switching; opt-in encrypted backups + 10-day retention; Ollama-like curated pull/run; beautiful per-target UI with full functionality above.

Deferred: internet inference/remote workers; tensor parallelism; continuous KV replication / zero-pause failover; live coordinator migration; cloud chat/inference; battery/cellular/thermal policy beyond FGS + Wi-Fi-only defaults.
