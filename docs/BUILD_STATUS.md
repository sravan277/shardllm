# Build Status (LIVING — update on every artifact change)

> Last updated: Phase 1 pull/list/run landed; bench DONE (16.6 tok/s mean); engine swap IN PROGRESS (2026-09-13).

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
| Model weights | `%LOCALAPPDATA%\dllm\models\` | PLANNED Phase 1 | unsloth Q4_K_M 397 MB |
| Engine swap (llama-cpp-2) | `apps/dllm` | IN PROGRESS (sibling agent) | CPU-only `=0.1.156` |
| Bench | `crates/dllm-core/examples/bench_local.rs` + `docs/bench-baseline.json` | DONE (mean decode 16.6 tok/s, Qwen3-0.6B Q4_K_M, n_ctx 4096) | — |
| Research | `docs/research/` (6 files) | DONE | — |
| Master plan | `docs/MASTER_PLAN.md` | DONE | deltas logged in research/00-index |

## Blockers log

1. No admin → no VS Build Tools → using rustup gnu toolchain + zig linker (user-space). ADR-001.
2. `Invoke-WebRequest` unusable in this shell (NonInteractive) → use `curl.exe`. Noted for scripts.
3. Android build needs JDK 17; machine has Java 25 → `setup-android.ps1` prefers JDK 17, warns otherwise.
4. Android cleartext HTTP blocked by default (`usesCleartextTraffic=false`) → LAN exception or HTTPS in Phase 1.
