# `dllm` CLI (Phase 0)

Windows coordinator CLI + LAN server scaffold. No real inference yet.

## Prereqs

- `rustup` stable toolchain. Either:
  - **MSVC** (recommended): VS Build Tools 2022 with the *Desktop C++* workload, or
  - **gnu + MinGW**: `rustup toolchain install stable-x86_64-pc-windows-gnu` plus a MinGW-w64 gcc on `PATH`.
- Verify: `rustc --version`, `cargo --version`.

> No C++ / CMake / CUDA needed for Phase 0. `llama-cpp-2` is intentionally
> **not** a dependency yet — real model execution lands in Phase 1 behind a
> Cargo feature gate (see workspace `Cargo.toml`).

## Build

```powershell
cargo build -p dllm
```

## Run

```powershell
# LAN API on 0.0.0.0:8080 + mDNS `_dllm._tcp.local.`
cargo run -p dllm -- serve --port 8080

# Catalog names (reads contracts/catalog.json baked at compile time)
cargo run -p dllm -- list

# Shard fetch plan only (no download in Phase 0)
cargo run -p dllm -- pull qwen3-0.6b-q4

# Serve hint
cargo run -p dllm -- run qwen3-0.6b-q4

# Placeholder
cargo run -p dllm -- ps
```

API while serving:

- `GET /api/health`
- `GET /api/models`
- `POST /v1/sessions`
- `POST /v1/sessions/{id}/messages` with `{"text": "..."}`
- `GET /v1/sessions/{id}/events` (SSE, `Last-Event-ID` replay, keep-alive 15 s)

Static web UI is served from `web/dist` when built; otherwise `/` returns
`web UI not built yet`.

## Phase 0 scope

Mocks only: `MockEngine` streams a canned sentence, `pull` prints a plan,
`ps` prints a placeholder. SQLite event log at `./dllm-events.db` (WAL,
append-only triggers). Distribution (Quinn/mTLS pipeline), real downloads,
and recovery arrive in later phases.
