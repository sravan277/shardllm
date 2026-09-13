# MCPs + Skills (LIVING — update when tooling changes)

> Last updated: 2026-09-13. Policy: minimal, reputable, low-bloat. Remote MCPs need no download; skills are project-local under `.agents/skills/` so the whole crew shares them. Enable heavy tools per-task, not by default.

## Connected MCPs (`opencode.json`, verified via `opencode mcp list`)

| Server | Type | URL | Auth | Use for | Status |
|--------|------|-----|------|---------|--------|
| `context7` | remote | `https://mcp.context7.com/mcp` | none (anon limits) / optional `CONTEXT7_API_KEY` | version-correct docs: Axum 0.8, Quinn, rusqlite, CameraX/Compose/OkHttp, React 19/Vite 8, SSE | ✓ connected |
| `gh_grep` | remote | `https://mcp.grep.app` | none | real-world code examples (regex over public GitHub) | ✓ connected |

Rules: Context7 — always `resolve-library-id` first, narrow `topic`, small token limit (2 tools but hungry). Grep — literal/regex only, filter by language/repo/path (1 tool, public code only, variable quality).

## Installed skills (9, project-local `.agents/skills/`)

| Skill | Source | Trust | Use when |
|-------|--------|-------|----------|
| `vercel-react-best-practices` | `vercel-labs/agent-skills` | 708K installs | React 19 perf, SSE-safe fetching/memo |
| `typescript-advanced-types` | `wshobson/agents` | 74K installs | type-safe SSE/OpenAPI DTOs |
| `webapp-testing` | `anthropics/skills` | 155K installs | Playwright E2E vs `npm run dev` :5173 |
| `test-driven-development` | `obra/superpowers` | 224K installs | red-green-refactor for event-log/contract changes |
| `camerax` | `android/skills` (Google) | official | QR scan, CameraX lifecycle |
| `compose-state-and-effects` | `chrisbanes/skills` | ~1K★ author-starred | Compose state hoisting, Flow collection (SSE streams) |
| `compose-performance` | `chrisbanes/skills` | same | recomposition/skippability |
| `kotlin-concurrency-and-flow` | `chrisbanes/skills` | same | StateFlow/SharedFlow, cancellation |
| `compose-ui-testing-patterns` | `chrisbanes/skills` | same | Compose UI/screenshot tests |

Load via the `skill` tool when the task matches; `frontend-design` (built-in) still governs visual direction.

## Deliberately NOT installed (revisit on need)

| Candidate | Reason |
|-----------|--------|
| GitHub MCP (readonly) | needs `GITHUB_PAT`; Grep covers code search for now |
| SQLite MCP (`@modelcontextprotocol/server-sqlite`) | reference impl now archived; inspect DB via app/tests instead |
| `Vaiz/rust-mcp-server` | 36★, merely shells `cargo` — use `cargo` CLI directly (20+ tool defs saved) |
| `wireshark-mcp` | 51 tools (~14K tokens); `tshark.exe` present at `C:\Program Files\Wireshark\` (not on PATH — call by full path; offline `-r` reads need no admin, live capture needs Npcap+admin) |
| `androidbuild` / `droidagentkit` | 0–2★, 53/18 tools — use `setup-android.ps1` + Gradle directly |
| Playwright MCP | defer to web E2E phase (needs browser download) |
| Phone Material3 skill | does not exist in `android/skills` (only `wear-compose-m3`); phone Compose covered by chrisbanes |
| `Kotlin/kotlin-agent-skills`, `normaltusker/kotlin-mcp-server`, `amsavarthan/android-tools-mcp`, adb-MCPs | unneeded / key-burdened / Studio-coupled — on-demand only |

## Global config note (not ours — untouched)

`~/.config/opencode/opencode.jsonc`: `blender` ✓ connected; **`playwright` ✗ failed** — its command is bare `["npx"]` with no package, pre-existing breakage. Recommend fixing to `["npx", "-y", "@playwright/mcp@latest"]` or removing; saying the word and I'll patch it.
