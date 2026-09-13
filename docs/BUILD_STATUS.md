# Build Status (LIVING — update on every artifact change)

> Last updated: Phase 5 hardening landed (2026-09-13) — MVP acceptance gate: harness `scripts/mvp-acceptance.ps1` DONE (first run pending at gate time); 2-device pipeline + on-device runtime verification stay PLANNED (human-gated).

| Artifact | Path | Status | Blocker / next |
|----------|------|--------|----------------|
| Contracts | `contracts/` (8 files) | DONE | reconcile deltas with code (see 02-agent notes in research index) |
| Windows svc scaffold | `apps/dllm` + `crates/*` | DONE (uncompiled) | `cargo build` pending zig-linker config |
| `dllm.exe` | `target\x86_64-pc-windows-gnu\debug\dllm.exe` (113 MB) | DONE | E2E smoke passed: health/session/message/SSE token stream/serve dist |
| Web UI scaffold | `apps/web` (vite+react+ts, deps installed) | DONE | DLLM UI (chat/models/devices, LAN-signal theme) |
| Web `dist/` | `apps/web/dist` (226 KB js) | DONE | built; served by `dllm serve` at `/` |
| Android scaffold | `apps/android` (18 files) | DONE (uncompiled) | SDK install via `setup-android.ps1` |
| `app-debug.apk` | `apps/android/app/build/outputs/apk/debug/app-debug.apk` (77 MB) | DONE | AGP 8.7.3 + JDK 17 + SDK 35; install on device to test |
| `dllm pull` | `apps/dllm/src/main.rs` (`cmd_pull`) | DONE | real resumable download: `model_dir` + `.part` resume + progress + size/sha verify + `.sha256` sidecar |
| `dllm list` | `apps/dllm/src/main.rs` (`cmd_list`) | DONE | shows installed state (`installed`/`partial`/`catalog` by size check) |
| `dllm run` | `apps/dllm/src/main.rs` (`cmd_run`) | DONE | pull-if-missing + serve |
| Model weights | `%LOCALAPPDATA%\dllm\models\` | DONE | unsloth Q4_K_M 397 MB installed (bench + serve use it; mock fallback keeps serve alive without it) |
| Engine swap (llama-cpp-2) | `apps/dllm/src/main.rs` + `crates/dllm-core/src/engine.rs` | DONE | real `LlamaEngine` (CPU-only `=0.1.156`, sampler chain per catalog); `MockEngine` fallback (ADR-013) |
| Bench | `crates/dllm-core/examples/bench_local.rs` + `docs/bench-baseline.json` | DONE (mean decode 16.6 tok/s, Qwen3-0.6B Q4_K_M, n_ctx 4096) | — |
| Research | `docs/research/` (6 files) | DONE | — |
| Master plan | `docs/MASTER_PLAN.md` | DONE | deltas logged in research/00-index |
| Quinn mTLS transport (TOFU) | `crates/dllm-net/src/transport.rs` | DONE | `server`/`connect`, fp pinning, Control20>Ack10>Activation0, cap-4 backpressure; tests green |
| Calibration planner | `crates/dllm-core/src/plan.rs` (`plan_layers`) | DONE | minimizes max stage time; tests green |
| Commit tracker | `crates/dllm-core/src/commit.rs` | DONE | piggyback/TRUNCATE, `on_ack` routing; tests green |
| Headless pipe_pair drill | `crates/dllm-core/examples/pipe_pair.rs` | DONE | `PIPE_PAIR PASS` (2 stages, 10 frames, committed_pos=7, truncate seen) on 127.0.0.1:8443 |
| Store TTL + checkpoint | `crates/dllm-store/src/lib.rs` (`prune_older_than`, `checkpoint` → `PRAGMA wal_checkpoint(TRUNCATE)`) | DONE | wired via `dllm_serve::spawn_maintenance` in `run_serve` (60 s cadence; TTL 24 h default, `DLLM_EVENT_TTL_SECS` override, ADR-023); tests: WAL recovery replay + backdated-prune append-only restore |
| `/api/stats` telemetry | `crates/dllm-serve/src/lib.rs` (`GET /api/stats`) | DONE | `uptime_s` / `engine` (llama\|mock) / `sessions` / `events` / `node_id` (ADR-024) |
| Real JNI decode (Android) | `apps/android/app/src/main/cpp/worker.cpp` (`LlamaBridge.loadModel/inferChunk/free`) | DONE (build-verified) | real llama.cpp `.so` in `app-debug.apk`; decode + sampler inside `inferChunk`; ABI locked to 3 symbols (ADR-025) |
| MVP acceptance harness | `scripts/mvp-acceptance.ps1` | DONE | artifacts + `dllm serve --port 8099` E2E + pipe_pair; PASS/FAIL checklist, nonzero exit on FAIL (first full run at gate time) |
| 2-physical-device pipeline | — | PLANNED | human-gated: needs 2nd device + pairing UX; logic proven headless only (`pipe_pair`) |
| On-device runtime verification | `apps/android` | PLANNED | human-gated: install APK on a device, pair, stream real decode end-to-end |

## Blockers log

1. No admin → no VS Build Tools → using rustup gnu toolchain + zig linker (user-space). ADR-001.
2. `Invoke-WebRequest` unusable in this shell (NonInteractive) → use `curl.exe`. Noted for scripts.
3. Android build needs JDK 17; machine has Java 25 → `setup-android.ps1` prefers JDK 17, warns otherwise.
4. Android cleartext HTTP blocked by default (`usesCleartextTraffic=false`) → LAN exception or HTTPS in Phase 1.
