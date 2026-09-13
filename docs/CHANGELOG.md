# Changelog (newest first — append on every change)

## 2026-09-13 — Phase 3 headless pipe_pair drill (transport+planner+commit)

- Added `crates/dllm-core/examples/pipe_pair.rs` (zero manifest edits): 2 identities with swapped fingerprints, worker `server()` on 127.0.0.1:8443, coordinator `connect()` strict; `plan_layers(28, [16.6 tok/s local, remote])` → 2 stages over Control stream → worker Ack; 8 `ActivationFrame`s over Activation via `frame_channel`/`spawn_frame_recv_loop` with `KvTentative` per frame; piggyback `on_commit(7)`, `on_truncate(6)` + resend 6,7 as new tokens, re-commit to `committed_pos=Some(7)`; worker saw truncate. Prints `PIPE_PAIR PASS stages=2 layers=28 frames=10 resends=2 committed_pos=7`.
- Docs: ADRs 018–022 (mTLS TOFU, stream priorities, cap-4 backpressure, calibration planner, port 8443); `BUILD_STATUS.md` Phase 3 rows DONE, 2-physical-device pipeline stays PLANNED.

## 2026-09-13 — Phase 1 real `pull`/`list`/`run` + toolchain

- `dllm pull`: real resumable HF single-file download (`model_dir` `%LOCALAPPDATA%\dllm\models\`, `.part` + `Range` resume, progress log, size/sha verify, `.sha256` sidecar); `list` shows `installed`/`partial`/`catalog`; `run` = pull-if-missing + serve (`apps/dllm/src/main.rs`).
- Toolchain installs (user-scope, no admin): WinLibs MinGW `mingw64\bin` on PATH, cmake 4.4 + ninja via winget, `libclang` via `py -m pip install` (`LIBCLANG_PATH=...site-packages\clang\native`), llama-cpp-2 `=0.1.156` CPU-only.
- Engine swap + bench IN PROGRESS (sibling agents).

## 2026-09-13 — Phase 0 verified E2E (exe + site live)

- `dllm.exe` built (113 MB debug, WinLibs MinGW UCRT + gnu toolchain) and smoke-tested: `/api/health` → `{"ok":true,"proto":"dllm1"}`; `POST /v1/sessions` → id; message accepted; SSE streams `session_created`/`user_message`/tokens with event ids; `/` serves built web UI.
- Fixes along the way: borrowck in `plan.rs` greedy partition; axum 0.8 `nest_service("/",…)` panics → `fallback_service(ServeDir + ServeFile index.html)`; `WEB_DIST` corrected to `apps/web/dist` (CWD = workspace root).
- Test note: `curl.exe -d` JSON needs a body file (`--data-binary "@msg.json"`) — PowerShell quoting mangles inline JSON.
- Android: Temurin JDK 17 user-install in progress (winget Temurin id has no applicable installer; direct Adoptium zip instead).

## 2026-09-13 — Agent tooling (MCPs + skills)

- Connected project MCPs in `opencode.json`: `context7` ✓ and `gh_grep` ✓ (verified via `opencode mcp list`). Global `blender` ✓; global `playwright` ✗ pre-existing broken `["npx"]` command — flagged in `docs/MCP_SKILLS.md`, untouched.
- Installed 9 project-local skills under `.agents/skills/`: camerax; compose-state-and-effects, compose-performance, kotlin-concurrency-and-flow, compose-ui-testing-patterns; typescript-advanced-types; vercel-react-best-practices; webapp-testing; test-driven-development.
- Rejected/deferred with reasons in `docs/MCP_SKILLS.md`: GitHub MCP (needs PAT), SQLite MCP (archived ref), rust-mcp-server (36★, use cargo CLI), wireshark-mcp (heavy; tshark.exe on disk), androidbuild/droidagentkit (0–2★), Playwright MCP (E2E phase), phone-Material3 skill (doesn't exist; wear-only).
- Toolchain lesson: zig-as-`linker` rejects rustc's `-fno-use-linker-plugin`, and `windows-sys` needs real `dlltool` → switching to WinLibs MinGW UCRT (ADR-012). Added `docs/MCP_SKILLS.md`.

## 2026-09-13 — Phase 0 build-out

- Added `contracts/` (8 files): README, manifest.schema.json, catalog.json (qwen3-0.6b-q4/q8), activation-frame.md (v1, magic DLLM1, 8 MiB cap), acks.md (optimistic COMMIT/TRUNCATE), pairing.md, openapi.yaml, event-log.md.
- Added Rust workspace: `Cargo.toml` + `crates/dllm-core|net|store|serve` + `apps/dllm` (`dllm serve/list/pull/run/ps`, Axum SSE, mDNS advertise, WAL event log, MockEngine). No llama-cpp dep yet (Phase 1).
- Added Android project `apps/android` (Kotlin+Compose, FGS connectedDevice, NsdManager, CameraX+MLKit, OkHttp SSE, DataStore, setup-android.ps1).
- Added web scaffold `apps/web` (Vite+React+TS, deps installed) + DLLM UI rewrite (chat/models/devices, LAN-signal theme).
- Added docs discipline: `CONNECTIONS.md` (living wiring map), `BUILD_STATUS.md`, `CHANGELOG.md` (this file), `DECISIONS.md`, `SETUP.md`.
- Toolchain: rustup stable-gnu 1.98.1 + zig 0.16.0 (user-space, no admin).

## 2026-09-13 — Research + plan (earlier)

- 5 parallel crew research reports → `docs/research/` (00-index + 01..05).
- `docs/MASTER_PLAN.md` written. Adopted deltas: 9/10/9 shards, optimistic commit, fast-path-first, TCP-first Android, block-16 KV, connectedDevice FGS, token-level ISA contract.
