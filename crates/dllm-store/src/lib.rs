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
    #[error("invalid device status: {0} (want paired|revoked)")]
    InvalidStatus(String),
    #[error("invalid network name: must be 1-40 chars")]
    InvalidName,
    #[error("invalid member status: {0} (want paired|revoked)")]
    InvalidMemberStatus(String),
    #[error("invalid member role: {0} (want admin|member)")]
    InvalidRole(String),
}

/// One row of the `devices` registry (paired-device source of truth).
///
/// `permissions` is a JSON array string (e.g. `["infer","chat"]`);
/// `last_seen` / `paired_at` are UTC timestamp strings (`strftime` at write).
/// `active` is NOT stored — handlers derive it from `last_seen` (90 s window).
/// `device_name` is the friendly OS hostname (NULL = client falls back to id).
/// `cpu_pct` / `mem_pct` + `load_updated_at` hold the last reported load
/// (NULL = never reported; self uses live `sysinfo`, never this row).
/// `capabilities` is a JSON object string (NULL = not reported yet).
/// `worker_active` is 1/0/NULL (NULL = never reported, never synthesized).
/// `layers` is a JSON string for the last reported layer assignment
/// (NULL = never reported, never synthesized).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DeviceRow {
    pub device_id: String,
    pub role: String,
    pub permissions: String,
    pub cert_fp: String,
    pub status: String,
    pub paired_at: String,
    pub paired_by: String,
    pub last_seen: String,
    pub device_name: Option<String>,
    pub cpu_pct: Option<f64>,
    pub mem_pct: Option<f64>,
    pub load_updated_at: Option<String>,
    pub capabilities: Option<String>,
    pub worker_active: Option<i64>,
    pub layers: Option<String>,
}

/// One row of the `networks` table (private groups).
///
/// `password_hash` is a SHA256 hex string (NULL = open, no password).
/// `open_join` is 1/0 (1 = anyone with password/open can join).
/// `qr_secret` is a 32-hex-char secret baked into the QR string.
/// `admin_device_id` is the creator (also an `admin` member row).
/// `created_at` is UTC timestamp text.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NetworkRow {
    pub id: String,
    pub name: String,
    pub password_hash: Option<String>,
    pub open_join: i64,
    pub qr_secret: String,
    pub admin_device_id: String,
    pub created_at: String,
}

/// One row of `network_members` (group membership).
/// `active` is NOT stored — handlers derive it from `last_seen` (90 s window).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NetworkMemberRow {
    pub group_id: String,
    pub device_id: String,
    pub role: String,
    pub status: String,
    pub last_seen: String,
}

/// One session rolled up from the event log (real activity feed).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionSummary {
    pub id: String,
    /// Parsed from the `session_created` payload (`{"model": ...}`);
    /// `None` when the payload is missing/unparseable (honest unknown).
    pub model: Option<String>,
    /// `MIN(ts)` for the session.
    pub created_at: String,
    /// `COUNT(*)` of `kind = 'token'` rows.
    pub tokens_out: usize,
    /// `MAX(ts)` of `kind = 'token'` rows; `None` when no tokens yet.
    pub last_token_at: Option<String>,
    /// Latest explicit rename if present, else server-side default: first
    /// `user_message` text truncated to 40 chars, else `"New chat"`.
    /// Renames live in `session_titles` (prune-resistant) with the latest
    /// `session_renamed` event as fallback, so every client polling
    /// `GET /v1/sessions` sees the same title.
    pub title: String,
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
CREATE TABLE IF NOT EXISTS devices (
    device_id   TEXT PRIMARY KEY,
    role        TEXT NOT NULL DEFAULT 'worker',
    permissions TEXT NOT NULL DEFAULT '[]',
    cert_fp     TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'paired'
                CHECK (status IN ('paired','revoked')),
    paired_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    paired_by   TEXT NOT NULL DEFAULT '',
    last_seen   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    device_name TEXT,
    cpu_pct     REAL,
    mem_pct     REAL,
    load_updated_at TEXT,
    capabilities TEXT,
    worker_active INTEGER,
    layers TEXT
);
CREATE TABLE IF NOT EXISTS session_titles (
    session    TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS networks (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    password_hash   TEXT,
    open_join       INTEGER NOT NULL DEFAULT 1,
    qr_secret       TEXT NOT NULL,
    admin_device_id TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS network_members (
    group_id  TEXT NOT NULL REFERENCES networks(id) ON DELETE CASCADE,
    device_id TEXT NOT NULL,
    role      TEXT NOT NULL DEFAULT 'member'
              CHECK (role IN ('admin','member')),
    status    TEXT NOT NULL DEFAULT 'paired'
              CHECK (status IN ('paired','revoked')),
    last_seen TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (group_id, device_id)
);
CREATE INDEX IF NOT EXISTS idx_network_members_group
    ON network_members(group_id);
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

/// Migration for DBs created before `device_name`/`load`/`capabilities`
/// /`worker_active`/`layers` columns existed: `ALTER TABLE ... ADD COLUMN`
/// for each missing column. Fresh DBs already have them via `SCHEMA_SQL`
/// (no-op). Also creates `networks` + `network_members` tables for DBs
/// created before groups existed (idempotent).
fn ensure_device_columns(conn: &rusqlite::Connection) -> Result<(), StoreError> {
    let mut stmt = conn.prepare("PRAGMA table_info(devices)")?;
    let cols: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<Vec<_>, _>>()?;
    let has = |c: &str| cols.iter().any(|x| x == c);
    // (column, DDL type). Nullable on purpose: old rows stay NULL = unknown.
    for (col, ddl) in [
        ("device_name", "TEXT"),
        ("cpu_pct", "REAL"),
        ("mem_pct", "REAL"),
        ("load_updated_at", "TEXT"),
        ("capabilities", "TEXT"),
        ("worker_active", "INTEGER"),
        ("layers", "TEXT"),
    ] {
        if !has(col) {
            conn.execute_batch(&format!("ALTER TABLE devices ADD COLUMN {col} {ddl};"))?;
        }
    }
    // Networks tables for pre-group DBs (idempotent).
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS networks (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            password_hash   TEXT,
            open_join       INTEGER NOT NULL DEFAULT 1,
            qr_secret       TEXT NOT NULL,
            admin_device_id TEXT NOT NULL,
            created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );
        CREATE TABLE IF NOT EXISTS network_members (
            group_id  TEXT NOT NULL REFERENCES networks(id) ON DELETE CASCADE,
            device_id TEXT NOT NULL,
            role      TEXT NOT NULL DEFAULT 'member'
                      CHECK (role IN ('admin','member')),
            status    TEXT NOT NULL DEFAULT 'paired'
                      CHECK (status IN ('paired','revoked')),
            last_seen TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            PRIMARY KEY (group_id, device_id)
        );
        CREATE INDEX IF NOT EXISTS idx_network_members_group
            ON network_members(group_id);",
    )?;
    Ok(())
}

/// SHA256 hex of a network password (NULL = open, never hashed).
pub fn hash_password(password: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(password.as_bytes());
    format!("{:x}", h.finalize())
}

/// Constant-time-ish password check (hash compare; false on None hash).
pub fn verify_password(password: &str, hash: Option<&str>) -> bool {
    match hash {
        Some(want) => hash_password(password) == want,
        None => true,
    }
}

static NET_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Fresh `net-<hex>` id (time + pid + counter; PK retry on collision).
pub fn new_network_id() -> String {
    let n = NET_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("net-{:x}-{:x}-{}", (unix_ms() as u64) & 0xffffff, std::process::id(), n)
}

/// Fresh 32-hex-char QR secret (SHA256 of time + pid + counter).
pub fn new_qr_secret() -> String {
    use sha2::{Digest, Sha256};
    let n = NET_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut h = Sha256::new();
    h.update(format!("{}-{}-{}-qr", unix_ms(), std::process::id(), n).as_bytes());
    format!("{:x}", h.finalize())
}

/// Validate a network name (trimmed, 1-40 chars).
pub fn validate_network_name(name: &str) -> Result<String, StoreError> {
    let clean = name.trim().to_string();
    let len = clean.chars().count();
    if len == 0 || len > 40 {
        return Err(StoreError::InvalidName);
    }
    Ok(clean)
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
        ensure_device_columns(&conn)?;
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

    /// Insert a device row, or refresh `role`/`permissions`/`cert_fp` +
    /// `last_seen` on conflict. `status`/`paired_at`/`paired_by` are set
    /// only on insert (revocation survives re-announce; use
    /// [`Store::set_status`] to flip it explicitly).
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn upsert_device(
        &self,
        device_id: &str,
        role: &str,
        permissions_json: &str,
        cert_fp: &str,
        paired_by: &str,
    ) -> Result<(), StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        conn.execute(
            "INSERT INTO devices
                 (device_id, role, permissions, cert_fp, status, paired_by, last_seen)
             VALUES (?1, ?2, ?3, ?4, 'paired', ?5,
                     strftime('%Y-%m-%dT%H:%M:%fZ','now'))
             ON CONFLICT(device_id) DO UPDATE SET
                 role = excluded.role,
                 permissions = excluded.permissions,
                 cert_fp = excluded.cert_fp,
                 last_seen = strftime('%Y-%m-%dT%H:%M:%fZ','now')",
            rusqlite::params![device_id, role, permissions_json, cert_fp, paired_by],
        )?;
        Ok(())
    }

    /// Refresh `last_seen` to now (heartbeat / health ping).
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn touch_last_seen(&self, device_id: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        conn.execute(
            "UPDATE devices SET last_seen = strftime('%Y-%m-%dT%H:%M:%fZ','now')
              WHERE device_id = ?1",
            rusqlite::params![device_id],
        )?;
        Ok(())
    }

    /// Flip `status` (`paired`|`revoked`). Returns `true` when the row
    /// existed, `false` for unknown `device_id` (honest absent, no invent).
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn set_status(&self, device_id: &str, status: &str) -> Result<bool, StoreError> {
        if status != "paired" && status != "revoked" {
            return Err(StoreError::InvalidStatus(status.to_string()));
        }
        let conn = self.conn.lock().expect("store mutex poisoned");
        let n = conn.execute(
            "UPDATE devices SET status = ?1 WHERE device_id = ?2",
            rusqlite::params![status, device_id],
        )?;
        Ok(n > 0)
    }

    /// Hard-delete one device row by id. Returns `true` when a row was
    /// removed, `false` for unknown `device_id` (honest absent, no invent).
    /// Re-heartbeat after delete recreates the row as `paired` via
    /// [`Store::upsert_device`] (fresh `paired_at`, empty `paired_by` unless
    /// the caller passes one).
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn delete_device(&self, device_id: &str) -> Result<bool, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let n = conn.execute(
            "DELETE FROM devices WHERE device_id = ?1",
            rusqlite::params![device_id],
        )?;
        Ok(n > 0)
    }

    /// All device rows ordered by `device_id` (handlers reorder self-first).
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn list_devices(&self) -> Result<Vec<DeviceRow>, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT device_id, role, permissions, cert_fp,
                    status, paired_at, paired_by, last_seen,
                    device_name, cpu_pct, mem_pct, load_updated_at, capabilities,
                    worker_active, layers
              FROM devices ORDER BY device_id ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(DeviceRow {
                device_id: row.get(0)?,
                role: row.get(1)?,
                permissions: row.get(2)?,
                cert_fp: row.get(3)?,
                status: row.get(4)?,
                paired_at: row.get(5)?,
                paired_by: row.get(6)?,
                last_seen: row.get(7)?,
                device_name: row.get(8)?,
                cpu_pct: row.get(9)?,
                mem_pct: row.get(10)?,
                load_updated_at: row.get(11)?,
                capabilities: row.get(12)?,
                worker_active: row.get(13)?,
                layers: row.get(14)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// One device row by id (`None` = honest unknown, no invent).
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn get_device(&self, device_id: &str) -> Result<Option<DeviceRow>, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT device_id, role, permissions, cert_fp,
                    status, paired_at, paired_by, last_seen,
                    device_name, cpu_pct, mem_pct, load_updated_at, capabilities,
                    worker_active, layers
              FROM devices WHERE device_id = ?1",
        )?;
        let mut rows = stmt.query_map(rusqlite::params![device_id], |row| {
            Ok(DeviceRow {
                device_id: row.get(0)?,
                role: row.get(1)?,
                permissions: row.get(2)?,
                cert_fp: row.get(3)?,
                status: row.get(4)?,
                paired_at: row.get(5)?,
                paired_by: row.get(6)?,
                last_seen: row.get(7)?,
                device_name: row.get(8)?,
                cpu_pct: row.get(9)?,
                mem_pct: row.get(10)?,
                load_updated_at: row.get(11)?,
                capabilities: row.get(12)?,
                worker_active: row.get(13)?,
                layers: row.get(14)?,
            })
        })?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    /// Set the friendly name (`None`/empty clears to NULL = unknown).
    /// Returns `true` when the row existed.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn set_device_name(&self, device_id: &str, name: Option<&str>) -> Result<bool, StoreError> {
        let clean = name.map(str::trim).filter(|s| !s.is_empty());
        let conn = self.conn.lock().expect("store mutex poisoned");
        let n = conn.execute(
            "UPDATE devices SET device_name = ?1 WHERE device_id = ?2",
            rusqlite::params![clean, device_id],
        )?;
        Ok(n > 0)
    }

    /// Store a reported load sample + `load_updated_at` = now.
    /// `None` values clear that column (unknown). Returns `true` when the
    /// row existed. Self rows never read this (live `sysinfo` instead).
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn set_device_load(
        &self,
        device_id: &str,
        cpu_pct: Option<f64>,
        mem_pct: Option<f64>,
    ) -> Result<bool, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let n = conn.execute(
            "UPDATE devices SET cpu_pct = ?1, mem_pct = ?2,
                 load_updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
               WHERE device_id = ?3",
            rusqlite::params![cpu_pct, mem_pct, device_id],
        )?;
        Ok(n > 0)
    }

    /// Store reported capabilities JSON (`None` clears to NULL).
    /// Returns `true` when the row existed.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn set_device_capabilities(
        &self,
        device_id: &str,
        caps_json: Option<&str>,
    ) -> Result<bool, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let n = conn.execute(
            "UPDATE devices SET capabilities = ?1 WHERE device_id = ?2",
            rusqlite::params![caps_json, device_id],
        )?;
        Ok(n > 0)
    }

    /// Store reported `worker_active` (`None` clears to NULL = unknown).
    /// Returns `true` when the row existed. Only overwrite when the caller
    /// actually sent the field (never synthesize).
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn set_device_worker_active(
        &self,
        device_id: &str,
        worker_active: Option<bool>,
    ) -> Result<bool, StoreError> {
        let v: Option<i64> = worker_active.map(|b| if b { 1 } else { 0 });
        let conn = self.conn.lock().expect("store mutex poisoned");
        let n = conn.execute(
            "UPDATE devices SET worker_active = ?1 WHERE device_id = ?2",
            rusqlite::params![v, device_id],
        )?;
        Ok(n > 0)
    }

    /// Store reported `layers` JSON (`None` clears to NULL = unknown).
    /// Returns `true` when the row existed. Only overwrite when sent.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn set_device_layers(
        &self,
        device_id: &str,
        layers_json: Option<&str>,
    ) -> Result<bool, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let n = conn.execute(
            "UPDATE devices SET layers = ?1 WHERE device_id = ?2",
            rusqlite::params![layers_json, device_id],
        )?;
        Ok(n > 0)
    }

    // -----------------------------------------------------------------------
    // Networks / groups.
    // -----------------------------------------------------------------------

    /// All networks ordered by `created_at` then `id`.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn list_networks(&self) -> Result<Vec<NetworkRow>, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, name, password_hash, open_join, qr_secret, admin_device_id, created_at
               FROM networks ORDER BY created_at ASC, id ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(NetworkRow {
                id: row.get(0)?,
                name: row.get(1)?,
                password_hash: row.get(2)?,
                open_join: row.get(3)?,
                qr_secret: row.get(4)?,
                admin_device_id: row.get(5)?,
                created_at: row.get(6)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// One network by id (`None` = honest unknown).
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn get_network(&self, id: &str) -> Result<Option<NetworkRow>, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, name, password_hash, open_join, qr_secret, admin_device_id, created_at
               FROM networks WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map(rusqlite::params![id], |row| {
            Ok(NetworkRow {
                id: row.get(0)?,
                name: row.get(1)?,
                password_hash: row.get(2)?,
                open_join: row.get(3)?,
                qr_secret: row.get(4)?,
                admin_device_id: row.get(5)?,
                created_at: row.get(6)?,
            })
        })?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    /// Create a network with caller-supplied id/secret (low-level).
    /// `name` is validated (1-40 trimmed). `password_hash` is
    /// `Some(hex)` or `None` for open.
    ///
    /// Also inserts the admin membership row
    /// (`group_id`, `admin_device_id`, `admin`, `paired`).
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn create_network_with(
        &self,
        id: &str,
        name: &str,
        password_hash: Option<&str>,
        open_join: bool,
        qr_secret: &str,
        admin_device_id: &str,
    ) -> Result<NetworkRow, StoreError> {
        let clean = validate_network_name(name)?;
        let conn = self.conn.lock().expect("store mutex poisoned");
        conn.execute(
            "INSERT INTO networks (id, name, password_hash, open_join, qr_secret, admin_device_id)
              VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                id,
                clean,
                password_hash,
                if open_join { 1 } else { 0 },
                qr_secret,
                admin_device_id
            ],
        )?;
        conn.execute(
            "INSERT INTO network_members (group_id, device_id, role, status, last_seen)
              VALUES (?1, ?2, 'admin', 'paired', strftime('%Y-%m-%dT%H:%M:%fZ','now'))
              ON CONFLICT(group_id, device_id) DO UPDATE SET
                role = 'admin', status = 'paired',
                last_seen = strftime('%Y-%m-%dT%H:%M:%fZ','now')",
            rusqlite::params![id, admin_device_id],
        )?;
        let row: NetworkRow = conn.query_row(
            "SELECT id, name, password_hash, open_join, qr_secret, admin_device_id, created_at
               FROM networks WHERE id = ?1",
            rusqlite::params![id],
            |row| {
                Ok(NetworkRow {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    password_hash: row.get(2)?,
                    open_join: row.get(3)?,
                    qr_secret: row.get(4)?,
                    admin_device_id: row.get(5)?,
                    created_at: row.get(6)?,
                })
            },
        )?;
        Ok(row)
    }

    /// Create a network, generating id + secret. `password` is plaintext
    /// (`None`/empty = open). Returns the row.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn create_network(
        &self,
        name: &str,
        password: Option<&str>,
        open_join: bool,
        admin_device_id: &str,
    ) -> Result<NetworkRow, StoreError> {
        let clean_pw = password.map(str::trim).filter(|s| !s.is_empty());
        let hash = clean_pw.map(hash_password);
        // Retry on PK collision (time-based ids).
        for _ in 0..5 {
            let id = new_network_id();
            let secret = new_qr_secret();
            match self.create_network_with(
                &id,
                name,
                hash.as_deref(),
                open_join,
                &secret,
                admin_device_id,
            ) {
                Ok(row) => return Ok(row),
                Err(StoreError::Sqlite(rusqlite::Error::SqliteFailure(e, _)))
                    if e.code == rusqlite::ErrorCode::ConstraintViolation =>
                {
                    continue
                }
                Err(e) => return Err(e),
            }
        }
        // Final attempt surfaces the error.
        let id = new_network_id();
        let secret = new_qr_secret();
        self.create_network_with(&id, name, hash.as_deref(), open_join, &secret, admin_device_id)
    }

    /// Rotate the QR secret. Returns `true` when the network existed.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn rotate_qr_secret(&self, group_id: &str) -> Result<Option<String>, StoreError> {
        let secret = new_qr_secret();
        let conn = self.conn.lock().expect("store mutex poisoned");
        let n = conn.execute(
            "UPDATE networks SET qr_secret = ?1 WHERE id = ?2",
            rusqlite::params![secret, group_id],
        )?;
        if n > 0 {
            Ok(Some(secret))
        } else {
            Ok(None)
        }
    }

    /// Ensure the default group exists (seed from `node_id`).
    /// If `networks` is empty, creates `id = "default"`, `name = "Default"`,
    /// open join, fresh secret, admin = `node_id` (+ admin member row).
    /// Returns the default (or first) network.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn ensure_default_network(&self, node_id: &str) -> Result<NetworkRow, StoreError> {
        if let Some(row) = self.get_network("default")? {
            // Ensure the admin membership survives (e.g. legacy leave).
            let conn = self.conn.lock().expect("store mutex poisoned");
            let _ = conn.execute(
                "INSERT INTO network_members (group_id, device_id, role, status, last_seen)
                  VALUES ('default', ?1, 'admin', 'paired', strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                  ON CONFLICT(group_id, device_id) DO NOTHING",
                rusqlite::params![node_id],
            );
            return Ok(row);
        }
        let existing = self.list_networks()?;
        if let Some(first) = existing.into_iter().next() {
            return Ok(first);
        }
        let secret = new_qr_secret();
        self.create_network_with("default", "Default", None, true, &secret, node_id)
    }

    /// All members of a group ordered by `device_id`.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn list_members(&self, group_id: &str) -> Result<Vec<NetworkMemberRow>, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT group_id, device_id, role, status, last_seen
               FROM network_members WHERE group_id = ?1 ORDER BY device_id ASC",
        )?;
        let rows = stmt.query_map(rusqlite::params![group_id], |row| {
            Ok(NetworkMemberRow {
                group_id: row.get(0)?,
                device_id: row.get(1)?,
                role: row.get(2)?,
                status: row.get(3)?,
                last_seen: row.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// One membership (`None` = not a member / honest unknown).
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn get_member(
        &self,
        group_id: &str,
        device_id: &str,
    ) -> Result<Option<NetworkMemberRow>, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT group_id, device_id, role, status, last_seen
               FROM network_members WHERE group_id = ?1 AND device_id = ?2",
        )?;
        let mut rows = stmt.query_map(rusqlite::params![group_id, device_id], |row| {
            Ok(NetworkMemberRow {
                group_id: row.get(0)?,
                device_id: row.get(1)?,
                role: row.get(2)?,
                status: row.get(3)?,
                last_seen: row.get(4)?,
            })
        })?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    /// Count of `paired` members in a group (capacity gate: max 5).
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn count_paired_members(&self, group_id: &str) -> Result<usize, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let n: i64 = conn.query_row(
            "SELECT COUNT(*) FROM network_members WHERE group_id = ?1 AND status = 'paired'",
            rusqlite::params![group_id],
            |row| row.get(0),
        )?;
        Ok(n.max(0) as usize)
    }

    /// Join (upsert) a member as `paired` + `last_seen` = now.
    /// `role` must be `admin`|`member`. Preserves nothing else.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn upsert_member(
        &self,
        group_id: &str,
        device_id: &str,
        role: &str,
    ) -> Result<(), StoreError> {
        if role != "admin" && role != "member" {
            return Err(StoreError::InvalidRole(role.to_string()));
        }
        let conn = self.conn.lock().expect("store mutex poisoned");
        conn.execute(
            "INSERT INTO network_members (group_id, device_id, role, status, last_seen)
              VALUES (?1, ?2, ?3, 'paired', strftime('%Y-%m-%dT%H:%M:%fZ','now'))
              ON CONFLICT(group_id, device_id) DO UPDATE SET
                status = 'paired',
                last_seen = strftime('%Y-%m-%dT%H:%M:%fZ','now')",
            rusqlite::params![group_id, device_id, role],
        )?;
        Ok(())
    }

    /// Flip a member's `status` (`paired`|`revoked`). `false` = unknown member.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn set_member_status(
        &self,
        group_id: &str,
        device_id: &str,
        status: &str,
    ) -> Result<bool, StoreError> {
        if status != "paired" && status != "revoked" {
            return Err(StoreError::InvalidMemberStatus(status.to_string()));
        }
        let conn = self.conn.lock().expect("store mutex poisoned");
        let n = conn.execute(
            "UPDATE network_members SET status = ?1 WHERE group_id = ?2 AND device_id = ?3",
            rusqlite::params![status, group_id, device_id],
        )?;
        Ok(n > 0)
    }

    /// Refresh a member's `last_seen` to now. `false` = unknown member.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn touch_member_last_seen(
        &self,
        group_id: &str,
        device_id: &str,
    ) -> Result<bool, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let n = conn.execute(
            "UPDATE network_members SET last_seen = strftime('%Y-%m-%dT%H:%M:%fZ','now')
               WHERE group_id = ?1 AND device_id = ?2",
            rusqlite::params![group_id, device_id],
        )?;
        Ok(n > 0)
    }

    /// Leave: hard-delete the membership row. `true` when a row was removed.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn remove_member(&self, group_id: &str, device_id: &str) -> Result<bool, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let n = conn.execute(
            "DELETE FROM network_members WHERE group_id = ?1 AND device_id = ?2",
            rusqlite::params![group_id, device_id],
        )?;
        Ok(n > 0)
    }

    /// One rolled-up row per session from the event log, ordered by first
    /// appearance: `model` parsed from the `session_created` payload
    /// (`None` = honest unknown), `tokens_out` = `token`-kind count,
    /// `last_token_at` = newest token `ts` (`None` when no tokens yet),
    /// `title` = latest explicit rename if present, else first
    /// `user_message` text truncated to 40 chars, else `"New chat"`.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn list_sessions(&self) -> Result<Vec<SessionSummary>, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT session, MIN(ts),
                    (SELECT payload FROM events e2
                      WHERE e2.session = e.session AND kind = 'session_created'
                      ORDER BY id ASC LIMIT 1),
                    (SELECT COUNT(*) FROM events e3
                      WHERE e3.session = e.session AND kind = 'token'),
                    (SELECT MAX(ts) FROM events e4
                      WHERE e4.session = e.session AND kind = 'token')
              FROM events e GROUP BY session ORDER BY MIN(id) ASC",
        )?;
        let base: Vec<(String, Option<String>, String, usize, Option<String>)> = stmt
            .query_map([], |row| {
                let created_payload: Option<String> = row.get(2)?;
                let model = created_payload.as_deref().and_then(|p| {
                    serde_json::from_str::<serde_json::Value>(p)
                        .ok()
                        .and_then(|v| v.get("model")?.as_str().map(str::to_string))
                });
                Ok((
                    row.get::<_, String>(0)?,
                    model,
                    row.get::<_, String>(1)?,
                    {
                        let n: i64 = row.get(3)?;
                        n.max(0) as usize
                    },
                    row.get::<_, Option<String>>(4)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        drop(stmt);
        let mut out = Vec::with_capacity(base.len());
        for (id, model, created_at, tokens_out, last_token_at) in base {
            let title = Self::resolve_title_locked(&conn, &id)?;
            out.push(SessionSummary {
                id,
                model,
                created_at,
                tokens_out,
                last_token_at,
                title,
            });
        }
        Ok(out)
    }

    /// `true` when at least one event row exists for `session`.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn session_exists(&self, session: &str) -> Result<bool, StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        let n: i64 = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM events WHERE session = ?1)",
            rusqlite::params![session],
            |row| row.get(0),
        )?;
        Ok(n != 0)
    }

    /// Upsert the prune-resistant display title for `session`.
    /// Caller validates (trimmed, non-empty, <= 80 chars); stored verbatim.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn set_session_title(&self, session: &str, title: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().expect("store mutex poisoned");
        conn.execute(
            "INSERT INTO session_titles (session, title, updated_at)
              VALUES (?1, ?2, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
              ON CONFLICT(session) DO UPDATE SET
                title = excluded.title,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')",
            rusqlite::params![session, title],
        )?;
        Ok(())
    }

    /// Hard-delete every event for `session` plus its `session_titles` row.
    /// Returns `true` when anything was removed, `false` for unknown ids.
    /// Same trigger-drop/commit pattern as [`Store::prune_older_than`]:
    /// the append-only DELETE trigger is dropped and re-created inside one
    /// `BEGIN IMMEDIATE` transaction so a crash rolls back atomically.
    ///
    /// Blocking: call via `spawn_blocking`.
    pub fn delete_session(&self, session: &str) -> Result<bool, StoreError> {
        let mut conn = self.conn.lock().expect("store mutex poisoned");
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch("DROP TRIGGER IF EXISTS events_no_delete;")?;
        let events = tx.execute(
            "DELETE FROM events WHERE session = ?1",
            rusqlite::params![session],
        )?;
        let titles = tx.execute(
            "DELETE FROM session_titles WHERE session = ?1",
            rusqlite::params![session],
        )?;
        tx.execute_batch(EVENTS_NO_DELETE_TRIGGER)?;
        tx.commit()?;
        Ok(events > 0 || titles > 0)
    }

    /// Title resolution for one session (caller holds the lock):
    /// 1. `session_titles` row (explicit rename, survives TTL prune +
    ///    restarts — `prune_older_than` only touches `events`);
    /// 2. latest `session_renamed` event payload `{"title": ...}` (covers
    ///    rows written as events-only);
    /// 3. first `user_message` text (`text`/`content`/`prompt`/`message`,
    ///    trimmed, truncated to 40 chars);
    /// 4. `"New chat"`.
    fn resolve_title_locked(
        conn: &rusqlite::Connection,
        session: &str,
    ) -> Result<String, StoreError> {
        // 1. Side table.
        {
            let mut stmt =
                conn.prepare("SELECT title FROM session_titles WHERE session = ?1")?;
            let mut rows = stmt.query_map(rusqlite::params![session], |row| {
                row.get::<_, String>(0)
            })?;
            if let Some(r) = rows.next() {
                let t = r?;
                if !t.trim().is_empty() {
                    return Ok(t);
                }
            }
        }
        // 2. Latest rename event.
        {
            let mut stmt = conn.prepare(
                "SELECT payload FROM events
                  WHERE session = ?1 AND kind = 'session_renamed'
                  ORDER BY id DESC LIMIT 1",
            )?;
            let mut rows = stmt.query_map(rusqlite::params![session], |row| {
                row.get::<_, String>(0)
            })?;
            if let Some(r) = rows.next() {
                let payload = r?;
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&payload) {
                    if let Some(t) = v.get("title").and_then(|x| x.as_str()) {
                        let t = t.trim();
                        if !t.is_empty() {
                            return Ok(t.to_string());
                        }
                    }
                }
            }
        }
        // 3. First user message, truncated to 40 chars.
        {
            let mut stmt = conn.prepare(
                "SELECT payload FROM events
                  WHERE session = ?1 AND kind = 'user_message'
                  ORDER BY id ASC LIMIT 1",
            )?;
            let mut rows = stmt.query_map(rusqlite::params![session], |row| {
                row.get::<_, String>(0)
            })?;
            if let Some(r) = rows.next() {
                let payload = r?;
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&payload) {
                    for key in ["text", "content", "prompt", "message"] {
                        if let Some(t) = v.get(key).and_then(|x| x.as_str()) {
                            let t = t.trim();
                            if !t.is_empty() {
                                return Ok(truncate_chars(t, 40));
                            }
                            continue;
                        }
                    }
                }
            }
        }
        // 4. Fresh chat.
        Ok("New chat".to_string())
    }
}

/// Truncate to `max` chars (Unicode-safe, no ellipsis).
fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect()
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

    #[test]
    fn devices_upsert_touch_status_list_roundtrip() {
        let path = temp_db("devices");
        let store = Store::open(&path).expect("open");
        assert!(store.list_devices().unwrap().is_empty());

        // Self seed: coordinator row exists (count >= 1 is real).
        store
            .upsert_device("self-1", "coordinator", r#"["infer","chat"]"#, "fp-self", "self-1")
            .unwrap();
        // Worker lifeline insert; status defaults to paired.
        store
            .upsert_device("pixel-8", "worker", r#"["infer"]"#, "fp-pixel", "")
            .unwrap();
        let devs = store.list_devices().unwrap();
        assert_eq!(devs.len(), 2);
        let me = devs.iter().find(|d| d.device_id == "self-1").unwrap();
        assert_eq!(me.role, "coordinator");
        assert_eq!(me.status, "paired");
        assert_eq!(me.paired_by, "self-1");

        // Re-announce refreshes role/perms/fp but preserves paired_by + status.
        store.set_status("pixel-8", "revoked").unwrap();
        store
            .upsert_device("pixel-8", "worker", r#"["infer","chat"]"#, "fp-pixel-2", "someone")
            .unwrap();
        let w = store
            .list_devices()
            .unwrap()
            .into_iter()
            .find(|d| d.device_id == "pixel-8")
            .unwrap();
        assert_eq!(w.status, "revoked");
        assert_eq!(w.paired_by, "");
        assert_eq!(w.cert_fp, "fp-pixel-2");

        // Status flip + unknown id honesty + validation.
        assert!(store.set_status("pixel-8", "paired").unwrap());
        assert!(!store.set_status("ghost", "paired").unwrap());
        assert!(matches!(
            store.set_status("pixel-8", "banned"),
            Err(StoreError::InvalidStatus(_))
        ));
        store.touch_last_seen("pixel-8").unwrap();
        let w2 = store
            .list_devices()
            .unwrap()
            .into_iter()
            .find(|d| d.device_id == "pixel-8")
            .unwrap();
        assert_eq!(w2.status, "paired");
        assert!(!w2.last_seen.is_empty());

        cleanup(&path);
    }

    #[test]
    fn delete_device_roundtrip_unknown_and_reheartbeat_recreates() {
        let path = temp_db("devdelete");
        let store = Store::open(&path).expect("open");
        store
            .upsert_device("self-1", "coordinator", r#"["infer","chat"]"#, "fp-self", "self-1")
            .unwrap();
        store
            .upsert_device("drill-pixel-8", "worker", r#"["infer"]"#, "fp-drill", "")
            .unwrap();
        assert_eq!(store.list_devices().unwrap().len(), 2);

        // Delete roundtrip: true + row gone from list/get.
        assert!(store.delete_device("drill-pixel-8").unwrap());
        assert!(store.get_device("drill-pixel-8").unwrap().is_none());
        assert_eq!(store.list_devices().unwrap().len(), 1);

        // Delete unknown -> false (honest absent, no invent).
        assert!(!store.delete_device("ghost").unwrap());

        // Delete-then-re-heartbeat recreates as paired with fresh insert defaults.
        store
            .upsert_device("drill-pixel-8", "worker", r#"["infer"]"#, "fp-drill-2", "")
            .unwrap();
        let row = store.get_device("drill-pixel-8").unwrap().expect("recreated");
        assert_eq!(row.status, "paired");
        assert_eq!(row.cert_fp, "fp-drill-2");
        assert_eq!(store.list_devices().unwrap().len(), 2);

        cleanup(&path);
    }

    #[test]
    fn device_name_load_caps_roundtrip_and_migration() {
        let path = temp_db("devmeta");
        let store = Store::open(&path).expect("open");
        store
            .upsert_device("n1", "coordinator", "[]", "fp", "n1")
            .unwrap();
        // New columns default to NULL (unknown, never invented).
        let row = store.get_device("n1").unwrap().expect("row");
        assert_eq!(row.device_name, None);
        assert_eq!(row.cpu_pct, None);
        assert_eq!(row.capabilities, None);
        assert!(store.get_device("ghost").unwrap().is_none());

        assert!(store.set_device_name("n1", Some("HOST-1")).unwrap());
        assert!(store.set_device_load("n1", Some(10.0), Some(20.0)).unwrap());
        assert!(store
            .set_device_capabilities("n1", Some(r#"{"kv_pages":4}"#))
            .unwrap());
        let row = store.get_device("n1").unwrap().expect("row");
        assert_eq!(row.device_name.as_deref(), Some("HOST-1"));
        assert_eq!(row.cpu_pct, Some(10.0));
        assert_eq!(row.mem_pct, Some(20.0));
        assert!(row.load_updated_at.as_deref().is_some_and(|s| !s.is_empty()));
        assert_eq!(row.capabilities.as_deref(), Some(r#"{"kv_pages":4}"#));
        // Re-announce preserves friendly/load/caps (upsert touches only core cols).
        store.upsert_device("n1", "coordinator", "[]", "fp2", "x").unwrap();
        let row = store.get_device("n1").unwrap().expect("row");
        assert_eq!(row.device_name.as_deref(), Some("HOST-1"));
        assert_eq!(row.cpu_pct, Some(10.0));
        // Unknown ids report false, never create rows.
        assert!(!store.set_device_name("ghost", Some("X")).unwrap());
        assert!(!store.set_device_load("ghost", Some(1.0), Some(2.0)).unwrap());

        // Migration: legacy DB without the new columns gains them on open.
        let legacy = temp_db("legacy");
        {
            let conn = rusqlite::Connection::open(&legacy).unwrap();
            conn.execute_batch(
                "CREATE TABLE devices (
                    device_id TEXT PRIMARY KEY, role TEXT NOT NULL DEFAULT 'worker',
                    permissions TEXT NOT NULL DEFAULT '[]', cert_fp TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'paired', paired_at TEXT NOT NULL DEFAULT 'x',
                    paired_by TEXT NOT NULL DEFAULT '', last_seen TEXT NOT NULL DEFAULT 'y');",
            )
            .unwrap();
            conn.execute("INSERT INTO devices (device_id) VALUES ('old-1')", []).unwrap();
        }
        let migrated = Store::open(&legacy).expect("reopen migrates");
        let row = migrated.get_device("old-1").unwrap().expect("legacy row survives");
        assert_eq!(row.device_name, None);
        assert_eq!(migrated.list_devices().unwrap().len(), 1);

        cleanup(&path);
        cleanup(&legacy);
    }

    #[test]
    fn networks_create_members_default_seed_roundtrip() {
        let path = temp_db("networks");
        let store = Store::open(&path).expect("open");
        assert!(store.list_networks().unwrap().is_empty());

        // Default seed: id/name/open/admin + admin member row.
        let def = store.ensure_default_network("node-1").expect("seed");
        assert_eq!(def.id, "default");
        assert_eq!(def.name, "Default");
        assert_eq!(def.password_hash, None);
        assert_eq!(def.admin_device_id, "node-1");
        let m = store.get_member("default", "node-1").unwrap().expect("admin member");
        assert_eq!(m.role, "admin");
        assert_eq!(m.status, "paired");
        // Idempotent: second call returns the same row, no dupes.
        let def2 = store.ensure_default_network("node-1").expect("reseed");
        assert_eq!(def2.id, "default");
        assert_eq!(store.list_networks().unwrap().len(), 1);

        // Password hashing helpers (None hash = open group, always verifies).
        let h = hash_password("pw-1");
        assert!(verify_password("pw-1", Some(&h)));
        assert!(!verify_password("nope", Some(&h)));
        assert!(verify_password("anything", None));

        // Name validation: trimmed 1-40, else InvalidName.
        assert_eq!(validate_network_name("  Ab  ").unwrap(), "Ab");
        assert!(matches!(validate_network_name(""), Err(StoreError::InvalidName)));
        assert!(matches!(validate_network_name(&"x".repeat(41)), Err(StoreError::InvalidName)));
        assert!(store.create_network("", None, true, "node-1").is_err());
        assert!(store.create_network(&"y".repeat(41), None, true, "node-1").is_err());

        // Create with password + closed join; admin row is an admin member.
        let net = store
            .create_network("Alpha", Some("secret"), false, "node-1")
            .expect("create");
        assert_eq!(net.name, "Alpha");
        assert_eq!(net.open_join, 0);
        assert!(net.password_hash.is_some_and(|s| verify_password("secret", Some(&s))));
        assert!(!net.qr_secret.is_empty());
        let admin = store.get_member(&net.id, "node-1").unwrap().expect("admin");
        assert_eq!(admin.role, "admin");

        // Membership lifecycle: join as member, count, revoke, touch, leave.
        store.upsert_member(&net.id, "phone-1", "member").unwrap();
        assert!(matches!(
            store.upsert_member(&net.id, "phone-1", "super"),
            Err(StoreError::InvalidRole(_))
        ));
        assert_eq!(store.count_paired_members(&net.id).unwrap(), 2);
        assert!(store.set_member_status(&net.id, "phone-1", "revoked").unwrap());
        assert!(matches!(
            store.set_member_status(&net.id, "phone-1", "banned"),
            Err(StoreError::InvalidMemberStatus(_))
        ));
        assert_eq!(store.count_paired_members(&net.id).unwrap(), 1);
        assert!(store.touch_member_last_seen(&net.id, "phone-1").unwrap());
        assert!(!store.touch_member_last_seen(&net.id, "ghost").unwrap());
        assert!(store.remove_member(&net.id, "phone-1").unwrap());
        assert!(!store.remove_member(&net.id, "phone-1").unwrap());
        assert!(store.get_member(&net.id, "phone-1").unwrap().is_none());

        // QR rotation mints a fresh secret; unknown group reports None.
        let before = store.get_network(&net.id).unwrap().expect("net").qr_secret;
        let fresh = store.rotate_qr_secret(&net.id).unwrap().expect("rotated");
        assert_ne!(fresh, before);
        assert!(store.rotate_qr_secret("ghost").unwrap().is_none());

        // Unknown group reads are honest Nones, not errors.
        assert!(store.get_network("ghost").unwrap().is_none());
        assert!(store.list_members("ghost").unwrap().is_empty());

        cleanup(&path);
    }

    #[test]
    fn list_sessions_rolls_up_tokens_from_event_log() {        let path = temp_db("sessions");
        let store = Store::open(&path).expect("open");
        assert!(store.list_sessions().unwrap().is_empty());

        store
            .append("sess-a", "session_created", r#"{"model":"qwen3-0.6b-q4"}"#)
            .unwrap();
        store.append("sess-a", "token", r#"{"pos":0}"#).unwrap();
        store.append("sess-a", "token", r#"{"pos":1}"#).unwrap();
        store
            .append("sess-b", "session_created", r#"{"model":"other"}"#)
            .unwrap();

        let rows = store.list_sessions().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].id, "sess-a");
        assert_eq!(rows[0].model.as_deref(), Some("qwen3-0.6b-q4"));
        assert_eq!(rows[0].tokens_out, 2);
        assert!(rows[0].last_token_at.is_some());
        assert!(!rows[0].created_at.is_empty());
        assert_eq!(rows[1].id, "sess-b");
        assert_eq!(rows[1].tokens_out, 0);
        assert_eq!(rows[1].last_token_at, None);

        cleanup(&path);
    }
}
