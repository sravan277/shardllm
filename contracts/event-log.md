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
