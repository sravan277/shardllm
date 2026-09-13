# Event log — SQLite schema, SSE shapes, retention

Append-only, hash-chained per session. Sole resume path for
`Last-Event-ID` reconnects and view switching. Only `COMMITTED` tokens are
logged — tentative rows never touch this log.

## SQLite DDL

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA journal_size_limit = 67108864;  -- 64 MiB
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
  session_id  TEXT PRIMARY KEY,
  plan_id     INTEGER NOT NULL,
  model       TEXT NOT NULL,          -- catalog id, e.g. qwen3-0.6b-q4
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  closed_at   TEXT
);

CREATE TABLE IF NOT EXISTS events (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,  -- == SSE id / Last-Event-ID
  session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  pos         INTEGER NOT NULL,                   -- token_position
  type        TEXT NOT NULL CHECK (type IN ('token','commit','status')),
  data        TEXT NOT NULL DEFAULT '{}',         -- JSON payload (see below)
  prev_hash   TEXT NOT NULL,                      -- hex sha256 of previous row's (id||session_id||pos||type||data)
  hash        TEXT NOT NULL,                      -- hex sha256 of (id||session_id||pos||type||data||prev_hash)
  created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  UNIQUE (session_id, pos, type)
);
CREATE INDEX IF NOT EXISTS idx_events_session_id ON events(session_id, id);
```

- First row per session uses `prev_hash = 'GENESIS'`.
- Writers compute the chain inside one `BEGIN IMMEDIATE … COMMIT`; readers
  verify `hash` on replay after unclean shutdown.
- KV checkpoints (`checkpoint_K` = 64–128) are `status` rows, not a second store.

## SSE event shapes (`GET /v1/sessions/{id}/events`)

```text
event: token
id: 41
data: {"pos":12,"token":8134,"text":" hello","margin":2.31}

event: commit
id: 42
data: {"pos":12,"checkpoint":false}

event: status
id: 43
data: {"state":"serving","plan_id":7,"bottleneck":1}

: keep-alive (every 15s, no id)
```

- `id:` = `events.id`. Clients reconnect with `Last-Event-ID`; server replays
  `WHERE id > :last ORDER BY id`.
- `token` rows carry decoded text + sampler top-1 `margin` (cross-ISA watchdog).
- `commit` rows mark durability; `checkpoint:true` every `checkpoint_K` tokens
  (coordinator/spare snapshot point for re-prefill recovery).

## Retention / prune

- Live sessions: keep all rows (4K ctx ≈ 4K rows — trivial).
- On session close: keep rows 24 h for late reconnects, then
  `DELETE FROM events WHERE session_id = ?`; `DELETE FROM sessions` cascades.
- Hard cap job (hourly): delete closed-session rows older than 24 h and
  orphaned `fetching` sessions older than 10 days (matches cloud backup TTL).
- `VACUUM` never auto-runs; checkpoint via `PRAGMA wal_checkpoint(PASSIVE)` on close.

### MVP retention policy (Phase 0 — implemented)

The Phase 0 store (`crates/dllm-store`, `dllm-serve::spawn_maintenance`)
implements this simpler policy; where it differs from the target design
above, it is the operative contract until per-session close tracking lands:

- `Store::prune_older_than(cutoff_ts: i64) -> Result<usize, StoreError>`
  hard-deletes every event whose `ts` is older than `cutoff_ts` (Unix
  milliseconds). Age-based across ALL sessions — live and closed sessions
  are not distinguished yet (deviation from the closed-session design).
- `ts` is the fixed-width ISO-8601 UTC text of the column default; the
  cutoff is rendered into that exact format by SQLite (second precision),
  so the comparison is a plain string compare.
- The append-only `events_no_delete` trigger is dropped and re-created
  inside the prune's single `BEGIN IMMEDIATE` transaction. Deletes stay
  forbidden to every other caller, and a crash mid-prune rolls back
  atomically (WAL + transactional DDL).
- `dllm-serve::spawn_maintenance(store, ttl_secs)` runs the loop: every 60 s
  (`MAINTENANCE_INTERVAL_SECS`, first tick fires immediately) it prunes
  events older than `now - TTL` and immediately runs `Store::checkpoint()`
  (`PRAGMA wal_checkpoint(TRUNCATE)`) to shrink the `-wal` file to zero
  bytes — superseding the PASSIVE-on-close rule while the loop is active.
- TTL default: 24 h (`dllm_serve::DEFAULT_EVENT_TTL_SECS = 86400`); override
  with the `DLLM_EVENT_TTL_SECS` env var (integer seconds), which wins over
  the `ttl_secs` parameter.
- Wiring: `spawn_maintenance` is exported (not auto-spawned); the server
  startup path (`dllm serve`) should call it once inside the Tokio runtime.
- `GET /api/stats` surfaces log size for observability: `sessions`
  (`COUNT(DISTINCT session)`) and `events` (`COUNT(*)`).
