//! dllm-store: SQLite WAL append-only session event log.
//!
//! # Blocking contract
//!
//! Every method on [`Store`] performs blocking SQLite I/O. Callers on a
//! Tokio runtime MUST wrap calls in `tokio::task::spawn_blocking`
//! (single-writer model: one `Store` behind `Arc`, readers use separate
//! connections in later phases; Phase 0 shares one connection + `Mutex`).
//!
//! # Retention (MVP)
//!
//! [`Store::prune_older_than`] is the only authorized deletion path; see
//! `contracts/event-log.md` for the full policy (TTL prune + WAL truncate
//! checkpoint every 60 s).

use std::path::Path;
use std::sync::Mutex;

use rusqlite::TransactionBehavior;

/// One committed session event (row of the `events` table).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Event {
    /// SQLite `rowid` — doubles as the SSE `lastEventId`.
    pub id: i64,
    pub session: String,
    pub kind: String,
    /// JSON payload (token text, status, errors, ...).
    pub payload: String,
    /// UTC timestamp string (`strftime` at insert).
    pub ts: String,
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

/// Table + index + UPDATE guard, applied on every open.
const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    kind    TEXT NOT NULL,
    payload TEXT NOT NULL,
    ts      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_events_session_id
    ON events(session, id);
CREATE TRIGGER IF NOT EXISTS events_no_update
    BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'append-only: UPDATE forbidden');
END;
";

/// DELETE guard. Dropped and re-created inside [`Store::prune_older_than`]'s
/// transaction (the only authorized deletion path); a crash mid-prune rolls
/// the trigger back along with the deletes (WAL + transactional DDL).
const EVENTS_NO_DELETE_TRIGGER: &str = "
CREATE TRIGGER IF NOT EXISTS events_no_delete
    BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'append-only: DELETE forbidden');
END;
";

/// Append-only event log. `Mutex<Connection>` = single writer.
pub struct Store {
    conn: Mutex<rusqlite::Connection>,
}

impl Store {
    /// Open (or create) the DB and set WAL pragmas + append-only triggers.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let conn = rusqlite::Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA busy_timeout=5000;
             PRAGMA foreign_keys=ON;",
        )?;
        conn.execute_batch(&format!("{SCHEMA_SQL}{EVENTS_NO_DELETE_TRIGGER}"))?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Append one event; returns its `rowid` (SSE id).
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn append(
        &self,
        session: &str,
        kind: &str,
        payload_json: &str,
    ) -> Result<i64, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        conn.execute(
            "INSERT INTO events (session, kind, payload) VALUES (?1, ?2, ?3)",
            rusqlite::params![session, kind, payload_json],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Replay events for `session` with `id > last_id`, ascending.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn replay_since(
        &self,
        session: &str,
        last_id: i64,
    ) -> Result<Vec<Event>, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, session, kind, payload, ts
             FROM events WHERE session = ?1 AND id > ?2 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(rusqlite::params![session, last_id], |row| {
            Ok(Event {
                id: row.get(0)?,
                session: row.get(1)?,
                kind: row.get(2)?,
                payload: row.get(3)?,
                ts: row.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Hard-delete every event whose `ts` is older than `cutoff_ts`
    /// (Unix milliseconds). Returns the number of rows removed.
    ///
    /// `ts` is the fixed-width ISO-8601 UTC text produced by the column
    /// default, so lexicographic order == chronological order; the cutoff is
    /// rendered into that exact format by SQLite itself (second precision).
    /// Runs inside one `BEGIN IMMEDIATE` transaction that also drops and
    /// re-creates the append-only DELETE trigger: deletes are otherwise
    /// forbidden, and a crash mid-prune rolls back atomically.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn prune_older_than(&self, cutoff_ts: i64) -> Result<usize, StoreError> {
        let mut conn = self.conn.lock().expect("store mutex poisoned");
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch("DROP TRIGGER IF EXISTS events_no_delete;")?;
        let deleted = tx.execute(
            "DELETE FROM events
              WHERE ts < strftime('%Y-%m-%dT%H:%M:%fZ', ?1 / 1000, 'unixepoch')",
            rusqlite::params![cutoff_ts],
        )?;
        tx.execute_batch(EVENTS_NO_DELETE_TRIGGER)?;
        tx.commit()?;
        Ok(deleted)
    }

    /// Checkpoint the WAL, truncating the `-wal` file to zero bytes.
    /// Returns SQLite's `(busy, wal_frames, checkpointed_frames)` triple.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn checkpoint(&self) -> Result<(i64, i64, i64), StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .map_err(StoreError::from)
    }

    /// Total number of event rows (`SELECT COUNT(*)`).
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn count_events(&self) -> Result<usize, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM events", [], |row| row.get(0))?;
        Ok(n.max(0) as usize)
    }

    /// Number of distinct sessions present in the log.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn count_sessions(&self) -> Result<usize, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let n: i64 = conn.query_row(
            "SELECT COUNT(DISTINCT session) FROM events",
            [],
            |row| row.get(0),
        )?;
        Ok(n.max(0) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DB: AtomicU64 = AtomicU64::new(0);

    fn temp_db(tag: &str) -> PathBuf {
        let n = NEXT_DB.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "dllm-store-{}-{tag}-{n}.db",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }

    fn unix_ms() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }

    /// Backdated insert: the `ts` DEFAULT is overridable, so the test can
    /// seed old rows without waiting. Renders `ts_ms` into the exact same
    /// ISO-8601 text the column default produces.
    fn insert_backdated(store: &Store, session: &str, ts_ms: i64) {
        let conn = store.conn.lock().expect("store mutex poisoned");
        conn.execute(
            "INSERT INTO events (session, kind, payload, ts)
             VALUES (?1, 'token', '{}',
                     strftime('%Y-%m-%dT%H:%M:%fZ', ?2 / 1000, 'unixepoch'))",
            rusqlite::params![session, ts_ms],
        )
        .expect("backdated insert");
    }

    #[test]
    fn wal_recovery_replays_all_events_after_reopen() {
        let path = temp_db("recovery");
        {
            let store = Store::open(&path).expect("open");
            assert_eq!(store.append("sess-a", "session_created", "{}").unwrap(), 1);
            assert_eq!(store.append("sess-a", "token", r#"{"pos":0}"#).unwrap(), 2);
            assert_eq!(store.append("sess-b", "session_created", "{}").unwrap(), 3);
            store.checkpoint().expect("checkpoint");
        } // drop closes the WAL-backed connection

        let store = Store::open(&path).expect("reopen");
        let a = store.replay_since("sess-a", 0).expect("replay a");
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].kind, "session_created");
        assert_eq!(a[1].kind, "token");
        assert_eq!(a[1].payload, r#"{"pos":0}"#);
        let b = store.replay_since("sess-b", 0).expect("replay b");
        assert_eq!(b.len(), 1);
        assert_eq!(store.count_events().unwrap(), 3);
        assert_eq!(store.count_sessions().unwrap(), 2);
        // Resume from a mid-stream id returns only the tail.
        assert_eq!(store.replay_since("sess-a", 1).unwrap().len(), 1);

        cleanup(&path);
    }

    #[test]
    fn prune_older_than_removes_backdated_rows_and_restores_append_only() {
        let path = temp_db("prune");
        let store = Store::open(&path).expect("open");
        let now = unix_ms();

        insert_backdated(&store, "old", now - 2 * 3600_000);
        insert_backdated(&store, "old", now - 90 * 60_000);
        insert_backdated(&store, "keep", now - 30 * 60_000);
        store.append("keep", "token", "{}").expect("append");

        assert_eq!(store.count_events().unwrap(), 4);
        assert_eq!(store.count_sessions().unwrap(), 2);

        // One hour ago: removes the 90 min and 2 h rows only.
        assert_eq!(store.prune_older_than(now - 3600_000).unwrap(), 2);
        assert_eq!(store.count_events().unwrap(), 2);
        assert_eq!(store.count_sessions().unwrap(), 1);
        assert_eq!(store.replay_since("keep", 0).unwrap().len(), 2);

        // Future cutoff: prunes everything; a no-op prune reports 0.
        assert_eq!(store.prune_older_than(now + 3600_000).unwrap(), 2);
        assert_eq!(store.prune_older_than(now + 3600_000).unwrap(), 0);
        assert_eq!(store.count_events().unwrap(), 0);

        // Append-only contract fully restored after pruning (needs a row:
        // BEFORE DELETE row triggers never fire on an empty table).
        store.append("keep", "token", "{}").expect("append");
        {
            let conn = store.conn.lock().expect("store mutex poisoned");
            let err = conn.execute("DELETE FROM events", []).unwrap_err();
            assert!(err.to_string().contains("append-only"));
        }

        // WAL is truncated after a prune + checkpoint.
        let (busy, _, _) = store.checkpoint().expect("checkpoint");
        assert_eq!(busy, 0);

        cleanup(&path);
    }
}
