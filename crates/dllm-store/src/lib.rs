//! dllm-store: SQLite WAL append-only session event log.
//!
//! # Blocking contract
//!
//! Every method on [`Store`] performs blocking SQLite I/O. Callers on a
//! Tokio runtime MUST wrap calls in `tokio::task::spawn_blocking`
//! (single-writer model: one `Store` behind `Arc`, readers use separate
//! connections in later phases; Phase 0 shares one connection + `Mutex`).

use std::path::Path;
use std::sync::Mutex;

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
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS events (
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
             CREATE TRIGGER IF NOT EXISTS events_no_delete
                 BEFORE DELETE ON events
             BEGIN
                 SELECT RAISE(ABORT, 'append-only: DELETE forbidden');
             END;",
        )?;
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
}
