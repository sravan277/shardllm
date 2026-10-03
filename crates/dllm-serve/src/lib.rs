//! dllm-serve: Axum LAN API (REST + SSE) + static web fallback.
//!
//! Routes:
//! - `GET /api/health`
//! - `GET /api/stats` (uptime, engine kind, session/event counts, node_id)
//! - `GET /api/models` (baked catalog JSON)
//! - `GET /api/node` (stable node_id + fingerprint + quic_port, pairing bootstrap)
//! - `POST /v1/sessions`
//! - `GET /v1/sessions` (entries carry `title`: latest rename, else first
//!   user message truncated to 40 chars, else "New chat")
//! - `DELETE /v1/sessions/{id}` (removes events + title; 404 unknown)
//! - `POST /v1/sessions/{id}/rename` body `{title}` (trimmed, 1–80 chars;
//!   404 unknown; persisted prune-resistant so web + Android stay in sync)
//! - `POST /v1/sessions/{id}/messages` (spawns generation; emits token+commit)
//! - `GET /v1/sessions/{id}/events` (SSE contract shapes token/commit/status,
//!   `?last_event=K` or `Last-Event-ID` resume, keep-alive 15 s)
//! - `GET /v1/devices` (registry incl `device_name`)
//! - `GET /v1/devices/{id}` (detail: device + live/reported/none load +
//!   self-only sessions + stage)
//! - `DELETE /v1/devices/{id}` (hard delete; 404 unknown; 400 self —
//!   never orphan the mesh)
//! - `POST /v1/devices/heartbeat` (optional device_name/load/capabilities)
//!
//! NOTE: never put `CompressionLayer` / `BufferLayer` in front of the SSE
//! route — it breaks streaming. `Store` calls below are tiny Phase 0 writes
//! done inline; production paths MUST wrap them in `spawn_blocking`
//! (see `dllm-store` docs). Phase 0 serves mock tokens only.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use dllm_core::{DeviceSpec, Engine, plan_layers};
use dllm_core::plan::TOTAL_LAYERS;
use dllm_store::Store;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tower_http::services::{ServeDir, ServeFile};

/// Baked model catalog. Requires `contracts/catalog.json` at compile time
/// (owned by the contracts mate; Phase 0 build needs that file present).
const CATALOG_JSON: &str = include_str!("../../../contracts/catalog.json");

/// Web dist dir, relative to the process working directory.
/// Run `dllm serve` from the workspace root (`MVP_1/`).
const WEB_DIST: &str = "apps/web/dist";

/// Broadcast message fanned out to SSE subscribers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseMsg {
    /// Store `rowid` — doubles as the SSE `id`.
    pub id: i64,
    pub session_id: String,
    pub event: String,
    pub data: String,
}

/// Engine flavor reported by `GET /api/stats`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    Llama,
    Mock,
}

impl EngineKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mock => "mock",
        }
    }

    /// `Engine` has no downcast hook, so discriminate by the concrete size
    /// behind the erased `Arc`: `LlamaEngine` is one `Arc<_>` (8 B on
    /// 64-bit), `MockEngine` a `Vec<String>` (24 B). Falls back to
    /// [`EngineKind::Mock`] for anything else.
    pub fn from_engine(engine: &Arc<dyn Engine>) -> Self {
        if std::mem::size_of_val(engine.as_ref()) == std::mem::size_of::<Arc<()>>() {
            Self::Llama
        } else {
            Self::Mock
        }
    }
}

/// One live generation run tracked for `POST /v1/sessions/{id}/stop`.
/// The decode loop polls `cancel` between tokens and emits
/// `status{"state":"cancelled"}` before exiting. `handle` lets `stop`
/// detect a still-running task (idempotent second stop).
pub struct GenRun {
    pub cancel: Arc<AtomicBool>,
    pub handle: tokio::task::JoinHandle<()>,
}

/// Shared app state.
pub struct AppState {
    pub engine: Arc<dyn Engine>,
    pub store: Arc<Store>,
    pub tx: broadcast::Sender<SseMsg>,
    pub node: NodeInfo,
    /// Engine flavor surfaced by `GET /api/stats` (see [`EngineKind::from_engine`]).
    pub engine_kind: EngineKind,
    /// Server start time; `GET /api/stats` reports `uptime_s`.
    pub started: Instant,
    /// HTTP port baked into `GET /api/pairing-uri` (the `--port` of `serve`).
    pub http_port: u16,
    /// Live generations per session for stop support.
    pub gen_runs: Mutex<HashMap<String, GenRun>>,
}

/// Stable node identity surfaced via `GET /api/node` (pairing bootstrap).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub fingerprint: String,
    pub quic_port: u16,
    pub version: String,
}

impl Default for NodeInfo {
    fn default() -> Self {
        Self {
            node_id: "dllm-dev-1".to_string(),
            fingerprint: "unknown".to_string(),
            quic_port: QUIC_PORT,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// QUIC transport port advertised for pairing (TOFU mTLS).
pub const QUIC_PORT: u16 = 8443;

/// Friendly self name: OS hostname, falling back to `node_id` when the
/// hostname is unavailable or empty. Single source of truth for the
/// self-seed (`dllm serve` delegates here so the `hostname` dep stays in
/// this crate).
pub fn self_device_name(fallback_node_id: &str) -> String {
    hostname::get()
        .ok()
        .and_then(|s| s.into_string().ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| fallback_node_id.to_string())
}

/// Fresh live load for THIS node only (`sysinfo`, never synthesized).
/// Returns `(cpu_pct 0-100, mem_pct 0-100)`. CPU needs two samples:
/// refresh, sleep 100 ms, refresh — otherwise the first reading is 0.
pub fn live_load() -> Option<(f32, f32)> {
    let mut sys = sysinfo::System::new_all();
    sys.refresh_cpu();
    std::thread::sleep(std::time::Duration::from_millis(100));
    sys.refresh_cpu();
    sys.refresh_memory();
    let cpus = sys.cpus();
    if cpus.is_empty() {
        return None;
    }
    let cpu: f32 = cpus.iter().map(|c| c.cpu_usage()).sum::<f32>() / cpus.len() as f32;
    let total = sys.total_memory() as f64;
    if total <= 0.0 {
        return None;
    }
    let mem = (sys.used_memory() as f64 * 100.0 / total) as f32;
    Some((cpu.clamp(0.0, 100.0), mem.clamp(0.0, 100.0)))
}

/// Map one stored event row to a contract-shaped SSE `(event, data)` pair
/// (see `contracts/event-log.md`):
/// - `token` rows -> `event: token`, `data: {"pos","text"}` (done stripped).
/// - `commit` rows -> `event: commit`, `data: {"pos"}`.
/// - everything else (`session_created`, `user_message`, `status`, ...)
///   -> `event: status`, `data` = raw payload.
/// Never wraps as `{kind,payload}` (the broken-chat shape).
pub fn sse_shape_for_stored(kind: &str, payload: &str) -> (String, String) {
    match kind {
        "token" => {
            match serde_json::from_str::<serde_json::Value>(payload) {
                Ok(v) => {
                    let pos = v.get("pos").cloned().unwrap_or(serde_json::Value::Null);
                    let text = v.get("text").cloned().unwrap_or(serde_json::Value::Null);
                    (
                        "token".to_string(),
                        serde_json::json!({"pos": pos, "text": text}).to_string(),
                    )
                }
                Err(_) => ("token".to_string(), payload.to_string()),
            }
        }
        "commit" => {
            match serde_json::from_str::<serde_json::Value>(payload) {
                Ok(v) => {
                    let pos = v.get("pos").cloned().unwrap_or(serde_json::Value::Null);
                    (
                        "commit".to_string(),
                        serde_json::json!({"pos": pos}).to_string(),
                    )
                }
                Err(_) => ("commit".to_string(), payload.to_string()),
            }
        }
        _ => ("status".to_string(), payload.to_string()),
    }
}

/// Map one live broadcast message to a contract-shaped SSE pair (same
/// rules as [`sse_shape_for_stored`]; `token` keeps `id:` = rowid).
pub fn sse_shape_for_live(event: &str, data: &str) -> (String, String) {
    sse_shape_for_stored(event, data)
}

/// Build shared state with a 256-slot broadcast channel.
pub fn new_state(engine: Arc<dyn Engine>, store: Arc<Store>) -> Arc<AppState> {
    new_state_with_node(engine, store, NodeInfo::default())
}

/// Build shared state with explicit node identity (preferred by `dllm serve`).
///
/// Note: event-log retention (`spawn_maintenance`) is opt-in — call it once
/// from the server's startup path while a Tokio runtime is active.
/// `http_port` defaults to 8080; use [`new_state_with_node_and_port`] when
/// the serve port is known so `GET /api/pairing-uri` is exact.
pub fn new_state_with_node(
    engine: Arc<dyn Engine>,
    store: Arc<Store>,
    node: NodeInfo,
) -> Arc<AppState> {
    new_state_with_node_and_port(engine, store, node, 8080)
}

/// Build shared state with explicit node identity + HTTP port.
pub fn new_state_with_node_and_port(
    engine: Arc<dyn Engine>,
    store: Arc<Store>,
    node: NodeInfo,
    http_port: u16,
) -> Arc<AppState> {
    let (tx, _rx) = broadcast::channel(256);
    let engine_kind = EngineKind::from_engine(&engine);
    // Best-effort default-group seed so `GET /v1/networks` is never empty
    // on a fresh node (auto-migrate already ran in `Store::open`).
    let _ = store.ensure_default_network(&node.node_id);
    Arc::new(AppState {
        engine,
        store,
        tx,
        node,
        engine_kind,
        started: Instant::now(),
        http_port,
        gen_runs: Mutex::new(HashMap::new()),
    })
}

/// Max paired members per network (join beyond this is 409 full).
pub const NETWORK_MAX_PAIRED: usize = 5;

/// Build the Axum router. Takes `Arc<AppState>` for `with_state`.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/stats", get(stats))
        .route("/api/models", get(models))
        .route("/api/node", get(node_info))
        .route("/api/pairing-uri", get(pairing_uri_handler))
        .route("/v1/devices", get(list_devices))
        .route("/v1/devices/heartbeat", post(heartbeat))
        .route("/v1/devices/{id}", get(device_detail).delete(delete_device))
        .route("/v1/devices/{id}/approve", post(approve_device))
        .route("/v1/devices/{id}/revoke", post(revoke_device))
        .route("/v1/networks", get(list_networks).post(create_network))
        .route("/v1/networks/{id}", get(get_network))
        .route("/v1/networks/{id}/devices", get(network_devices))
        .route("/v1/networks/{id}/join", post(join_network))
        .route("/v1/networks/{id}/leave", post(leave_network))
        .route("/v1/networks/{id}/rotate-qr", post(rotate_qr))
        .route("/v1/sessions", post(create_session).get(list_sessions))
        .route("/v1/sessions/{id}", delete(delete_session))
        .route("/v1/sessions/{id}/rename", post(rename_session))
        .route("/v1/sessions/{id}/stop", post(stop_session))
        .route("/v1/plan", get(get_plan))
        .route("/v1/usage", get(get_usage))
        .route("/v1/sessions/{id}/messages", post(post_message))
        .route("/v1/sessions/{id}/events", get(session_events))
        // Static web UI (built dist). Run the exe from the workspace root so
        // this relative path resolves. Unknown non-API paths fall back to
        // index.html (SPA). API routes above take precedence.
        .fallback_service(
            ServeDir::new(WEB_DIST)
                .not_found_service(ServeFile::new(format!("{WEB_DIST}/index.html"))),
        )
        .with_state(state)
}

/// Default event TTL: 24 h (overridable with `DLLM_EVENT_TTL_SECS`).
pub const DEFAULT_EVENT_TTL_SECS: u64 = 24 * 60 * 60;

/// Cadence of the retention loop (see [`spawn_maintenance`]).
pub const MAINTENANCE_INTERVAL_SECS: u64 = 60;

/// Spawn the event-log retention task: every
/// [`MAINTENANCE_INTERVAL_SECS`] (first tick fires immediately) prune events
/// older than `now - TTL`, then run [`Store::checkpoint`]
/// (`PRAGMA wal_checkpoint(TRUNCATE)`) right after each prune.
///
/// TTL resolution: `DLLM_EVENT_TTL_SECS` (integer seconds) wins over the
/// `ttl_secs` parameter — intended usage:
/// `spawn_maintenance(store, dllm_serve::DEFAULT_EVENT_TTL_SECS)`.
/// Must be called from within a Tokio runtime; the SQLite work runs on
/// `spawn_blocking`. Exported for the server's startup path (`dllm serve`);
/// not wired into [`new_state_with_node`] automatically.
pub fn spawn_maintenance(store: Arc<Store>, ttl_secs: u64) -> tokio::task::JoinHandle<()> {
    let ttl_secs = std::env::var("DLLM_EVENT_TTL_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(ttl_secs);
    tokio::spawn(async move {
        let mut ticker =
            tokio::time::interval(Duration::from_secs(MAINTENANCE_INTERVAL_SECS));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            let store = store.clone();
            let now_ms = unix_ms_now();
            let ttl_ms = i64::try_from(ttl_secs.saturating_mul(1000)).unwrap_or(i64::MAX);
            let cutoff = now_ms.saturating_sub(ttl_ms);
            let outcome = tokio::task::spawn_blocking(move || {
                let pruned = store.prune_older_than(cutoff)?;
                store.checkpoint()?;
                Ok::<usize, dllm_store::StoreError>(pruned)
            })
            .await;
            match outcome {
                Ok(Ok(0)) => {}
                Ok(Ok(pruned)) => {
                    tracing::info!(pruned, ttl_secs, "event log pruned; WAL truncated");
                }
                Ok(Err(e)) => tracing::warn!("event log maintenance failed: {e}"),
                Err(e) => tracing::warn!("event log maintenance task failed: {e}"),
            }
        }
    })
}

fn unix_ms_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

async fn health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    // Cheap honest "active": a health ping means this node was seen now.
    // Best-effort (missing self row just skips; Phase 0 inline write).
    let _ = state.store.touch_last_seen(&state.node.node_id);
    Json(serde_json::json!({ "ok": true, "version": "0.1.0", "proto": "dllm1" }))
}

async fn stats(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    // Phase 0: tiny inline reads (see crate docs on the blocking contract).
    let sessions = state.store.count_sessions().unwrap_or(0);
    let events = state.store.count_events().unwrap_or(0);
    Json(serde_json::json!({
        "uptime_s": state.started.elapsed().as_secs(),
        "engine": state.engine_kind.as_str(),
        "sessions": sessions,
        "events": events,
        "node_id": state.node.node_id,
    }))
}

async fn models() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        CATALOG_JSON,
    )
}

async fn node_info(State(state): State<Arc<AppState>>) -> Json<NodeInfo> {
    Json(state.node.clone())
}

/// Rank for a candidate LAN IPv4 (lower = better). Home WiFi `192.168/16`
/// first, then `10/8`, then `172.16/12`. Anything else (VPN/PPP/virtual,
/// CGNAT, public) sorts last so a multi-homed host never advertises a
/// VPN address like `172.16.0.2/32` while the phone sits on `192.168.x`.
fn lan_rank(v4: std::net::Ipv4Addr) -> u8 {
    let o = v4.octets();
    if o[0] == 192 && o[1] == 168 {
        0
    } else if o[0] == 10 {
        1
    } else if o[0] == 172 && (16..32).contains(&o[1]) {
        2
    } else {
        3
    }
}

/// Every usable local IPv4, best first, deduped.
///
/// Sources: hostname resolution (one entry per bound interface address —
/// the std-only way to enumerate interfaces) plus the outbound-route trick
/// as one more candidate. Loopback, unspecified, link-local and multicast
/// are excluded. Single source of truth — `dllm id` delegates here.
pub fn lan_candidates() -> Vec<String> {
    use std::net::ToSocketAddrs as _;
    let mut out: Vec<String> = Vec::new();
    if let Ok(name) = hostname::get() {
        if let Some(s) = name.to_str() {
            if let Ok(addrs) = format!("{s}:0").to_socket_addrs() {
                for a in addrs {
                    if let std::net::IpAddr::V4(v4) = a.ip() {
                        if !v4.is_unspecified()
                            && !v4.is_loopback()
                            && !v4.is_link_local()
                            && !v4.is_multicast()
                        {
                            out.push(v4.to_string());
                        }
                    }
                }
            }
        }
    }
    if let Ok(sock) = std::net::UdpSocket::bind("0.0.0.0:0") {
        if sock.connect("8.8.8.8:80").is_ok() {
            if let Ok(std::net::IpAddr::V4(v4)) = sock.local_addr().map(|a| a.ip()) {
                if !v4.is_unspecified() && !v4.is_loopback() {
                    out.push(v4.to_string());
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out.sort_by_key(|s| {
        s.parse::<std::net::Ipv4Addr>()
            .map(lan_rank)
            .unwrap_or(9)
    });
    out
}

/// Best-first pick from [`lan_candidates`].
/// Falls back to `127.0.0.1` when offline or on error.
pub fn lan_ipv4() -> String {
    lan_candidates()
        .into_iter()
        .next()
        .unwrap_or_else(|| "127.0.0.1".to_string())
}

/// Canonical pairing-URI string (single source of truth — `dllm id`
/// delegates here): `dllm://pair?host=<lan-ip>&port=<http>&quic=8443&fp=<fp>&v=<ver>`.
pub fn pairing_uri(host: &str, port: u16, fingerprint: &str) -> String {
    format!(
        "dllm://pair?host={host}&port={port}&quic={}&fp={fingerprint}&v={}",
        QUIC_PORT,
        env!("CARGO_PKG_VERSION"),
    )
}

async fn pairing_uri_handler(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let candidates = lan_candidates();
    let host = candidates
        .first()
        .cloned()
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let uri = pairing_uri(&host, state.http_port, &state.node.fingerprint);
    let mut v = serde_json::json!({ "uri": uri, "candidates": candidates });
    if candidates.len() > 1 {
        v["warning"] = serde_json::Value::String(format!(
            "multiple local IPs; showing best pick {host} — if the phone cannot reach it, retry with another entry from candidates"
        ));
    }
    Json(v)
}

/// `last_seen` counts as active inside this window (seconds).
pub const DEVICE_ACTIVE_WINDOW_SECS: i64 = 90;

/// Parse the SQLite UTC text (`%Y-%m-%dT%H:%M:%fZ`) to Unix millis.
/// `None` = unparseable (caller treats as inactive, never invents).
fn parse_sqlite_ts_ms(s: &str) -> Option<i64> {
    let s = s.strip_suffix('Z')?;
    let (date, time) = s.split_once('T')?;
    let mut d = date.split('-');
    let (y, mo, day): (i64, i64, i64) = (
        d.next()?.parse().ok()?,
        d.next()?.parse().ok()?,
        d.next()?.parse().ok()?,
    );
    if d.next().is_some() {
        return None;
    }
    let (hms, frac_ms) = match time.split_once('.') {
        Some((a, b)) => {
            let digits: String = b.chars().take(3).collect();
            if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
                return None;
            }
            let mut ms: i64 = digits.parse().ok()?;
            for _ in digits.len()..3 {
                ms *= 10;
            }
            (a, ms)
        }
        None => (time, 0),
    };
    let mut t = hms.split(':');
    let (h, mi, sec): (i64, i64, i64) = (
        t.next()?.parse().ok()?,
        t.next()?.parse().ok()?,
        t.next()?.parse().ok()?,
    );
    if t.next().is_some() {
        return None;
    }
    if !(1..=12).contains(&mo)
        || !(1..=31).contains(&day)
        || h > 23
        || mi > 59
        || sec > 60
    {
        return None;
    }
    // Days from civil (Howard Hinnant's algorithm).
    let y_adj = if mo <= 2 { y - 1 } else { y };
    let era = y_adj.div_euclid(400);
    let yoe = y_adj - era * 400;
    let mp = (mo + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    Some((days * 86400 + h * 3600 + mi * 60 + sec) * 1000 + frac_ms)
}

fn last_seen_active(last_seen: &str, now_ms: i64) -> bool {
    match parse_sqlite_ts_ms(last_seen) {
        // Small future-skew tolerance (5 s); otherwise strictly within window.
        Some(ts) => now_ms >= ts - 5_000 && now_ms - ts <= DEVICE_ACTIVE_WINDOW_SECS * 1000,
        None => false,
    }
}

#[derive(Debug, Serialize)]
struct DeviceView {
    device_id: String,
    /// Friendly OS hostname; NULL (None) = unknown, client falls back to id.
    /// Server never invents a fallback string here.
    device_name: Option<String>,
    role: String,
    permissions: Vec<String>,
    status: String,
    active: bool,
    last_seen: String,
    paired_at: String,
    paired_by: String,
}

/// Real registry read: every field from the `devices` table; `active` is
/// derived from `last_seen` (90 s window). Self row first, rest by id.
async fn list_devices(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let now_ms = unix_ms_now();
    let rows = state.store.list_devices().unwrap_or_default();
    let mut devs: Vec<DeviceView> = rows
        .into_iter()
        .map(|r| DeviceView {
            device_id: r.device_id,
            device_name: r.device_name.filter(|s| !s.trim().is_empty()),
            role: r.role,
            permissions: serde_json::from_str(&r.permissions).unwrap_or_default(),
            status: r.status,
            active: last_seen_active(&r.last_seen, now_ms),
            last_seen: r.last_seen,
            paired_at: r.paired_at,
            paired_by: r.paired_by,
        })
        .collect();
    devs.sort_by(|a, b| {
        let a_self = a.device_id == state.node.node_id;
        let b_self = b.device_id == state.node.node_id;
        b_self.cmp(&a_self).then_with(|| a.device_id.cmp(&b.device_id))
    });
    Json(serde_json::json!({ "devices": devs }))
}

#[derive(Debug, Deserialize)]
struct HeartbeatLoad {
    #[serde(default)]
    cpu_pct: Option<f64>,
    #[serde(default)]
    mem_pct: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct HeartbeatReq {
    device_id: String,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    permissions: Option<Vec<String>>,
    #[serde(default)]
    cert_fp: Option<String>,
    /// Friendly name (OS hostname); stored on the row, NULL when omitted.
    #[serde(default)]
    device_name: Option<String>,
    /// Optional reported load sample 0-100; stored + timestamped.
    #[serde(default)]
    load: Option<HeartbeatLoad>,
    /// Reported capabilities JSON; stored on the row.
    #[serde(default)]
    capabilities: Option<serde_json::Value>,
    /// Group id to refresh membership `last_seen` (explicit join required;
    /// heartbeat never auto-joins, it only touches an existing member row).
    #[serde(default)]
    group_id: Option<String>,
    /// Whether this device can take pipeline stages (phone offload prep).
    /// Persisted only when sent; never synthesized.
    #[serde(default)]
    worker_active: Option<bool>,
    /// Reported layer assignment JSON (e.g. `{"layer_start":0,"layer_end":8}`
    /// or `[0,8]`); persisted only when sent, never synthesized.
    #[serde(default)]
    layers: Option<serde_json::Value>,
}

/// Worker lifeline: upsert (default role `worker`) + `last_seen` = now +
/// status `paired`. Persists optional `device_name` / `load` /
/// `capabilities` / `worker_active` / `layers` on the row when present,
/// and touches the `network_members` row when `group_id` names an existing
/// membership. Returns `{ok:true}`.
async fn heartbeat(
    State(state): State<Arc<AppState>>,
    Json(body): Json<HeartbeatReq>,
) -> impl IntoResponse {
    if body.device_id.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": "device_id required" })),
        );
    }
    let role = body.role.unwrap_or_else(|| "worker".to_string());
    let perms = body.permissions.unwrap_or_default();
    let perms_json = serde_json::to_string(&perms).unwrap_or_else(|_| "[]".to_string());
    let cert_fp = body.cert_fp.unwrap_or_default();
    // Phase 0 inline writes; production: spawn_blocking.
    let _ = state.store.upsert_device(
        &body.device_id,
        &role,
        &perms_json,
        &cert_fp,
        "",
    );
    if let Some(name) = body.device_name.as_deref() {
        let clean = name.trim();
        if !clean.is_empty() {
            let _ = state.store.set_device_name(&body.device_id, Some(clean));
        }
    }
    if let Some(load) = body.load.as_ref() {
        // Honest store: only persist when at least one sample is present.
        // No synthesis, no clamping lies — values stored as reported.
        if load.cpu_pct.is_some() || load.mem_pct.is_some() {
            let _ = state.store.set_device_load(
                &body.device_id,
                load.cpu_pct,
                load.mem_pct,
            );
        }
    }
    if let Some(caps) = body.capabilities.as_ref() {
        let caps_str = caps.to_string();
        let _ = state
            .store
            .set_device_capabilities(&body.device_id, Some(&caps_str));
    }
    if let Some(wa) = body.worker_active {
        let _ = state.store.set_device_worker_active(&body.device_id, Some(wa));
    }
    if let Some(layers) = body.layers.as_ref() {
        let layers_str = layers.to_string();
        let _ = state
            .store
            .set_device_layers(&body.device_id, Some(&layers_str));
    }
    let _ = state.store.touch_last_seen(&body.device_id);
    let _ = state.store.set_status(&body.device_id, "paired");
    if let Some(gid) = body.group_id.as_deref() {
        let gid = gid.trim();
        if !gid.is_empty() {
            let _ = state.store.touch_member_last_seen(gid, &body.device_id);
        }
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({ "ok": true })),
    )
}

async fn approve_device(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.store.set_status(&id, "paired") {
        Ok(true) => (
            StatusCode::OK,
            Json(serde_json::json!({ "ok": true })),
        ),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "ok": false, "error": "unknown device" })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
        ),
    }
}

async fn revoke_device(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.store.set_status(&id, "revoked") {
        Ok(true) => (
            StatusCode::OK,
            Json(serde_json::json!({ "ok": true })),
        ),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "ok": false, "error": "unknown device" })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
        ),
    }
}

/// `DELETE /v1/devices/{id}`: hard-delete the registry row so drill junk /
/// stale dupes disappear from `GET /v1/devices`. Unknown id -> 404
/// `{ok:false}`. Refuses to delete the self coordinator row with 400
/// (never orphan the mesh). Broadcasts nothing. A later heartbeat for the
/// same id recreates the row as `paired` via `upsert_device`.
async fn delete_device(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if id == state.node.node_id {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": "cannot delete self coordinator" })),
        );
    }
    // Phase 0 inline write; production: spawn_blocking.
    match state.store.delete_device(&id) {
        Ok(true) => (
            StatusCode::OK,
            Json(serde_json::json!({ "ok": true, "id": id })),
        ),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "ok": false, "error": "unknown device" })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
        ),
    }
}

/// Canonical network QR string (single source of truth):
/// `dllm://net?g=<id>&h=<secret-hint>&p=<has-password 0/1>&fp=<fingerprint>&v=<version>#s=<secret>`
/// where `h` is the first 8 chars of the secret (hint, not the secret) and
/// `s` (fragment) carries the full secret. Scanned by the phone to join.
pub fn network_qr(network_id: &str, qr_secret: &str, has_password: bool, fingerprint: &str) -> String {
    let hint: String = qr_secret.chars().take(8).collect();
    let p = if has_password { "1" } else { "0" };
    format!(
        "dllm://net?g={network_id}&h={hint}&p={p}&fp={fingerprint}&v={}#s={qr_secret}",
        env!("CARGO_PKG_VERSION"),
    )
}

/// Build a [`DeviceSpec`] for planning from a registry row.
/// `decode_tps` resolution (never synthesized beyond documented fallbacks):
/// - `capabilities.decode_tps` when present and > 0;
/// - else `capabilities.ms_per_layer_decode` via `1000 / (ms * 28)`;
/// - else the bench baseline 16.6 tok/s (see `docs/bench-baseline.json`).
/// `bandwidth_mbps` from `capabilities.bandwidth_mbps` else 1000.
/// `kv_budget_mib` from `capabilities.kv_budget_mib` else 1024.
fn device_spec_from_row(row: &dllm_store::DeviceRow, fallback_id: &str) -> DeviceSpec {
    let caps: Option<serde_json::Value> = row
        .capabilities
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());
    let mut decode_tps = 16.6f64;
    let mut bandwidth_mbps = 1000.0f64;
    let mut kv_budget_mib = 1024u64;
    if let Some(c) = caps.as_ref() {
        if let Some(t) = c.get("decode_tps").and_then(|v| v.as_f64()) {
            if t.is_finite() && t > 0.0 {
                decode_tps = t;
            }
        } else if let Some(ms) = c
            .get("ms_per_layer_decode")
            .and_then(|v| v.as_f64())
        {
            if ms.is_finite() && ms > 0.0 {
                decode_tps = 1000.0 / (ms * TOTAL_LAYERS as f64);
                if !decode_tps.is_finite() || decode_tps <= 0.0 {
                    decode_tps = 16.6;
                }
            }
        }
        // Also accept ms_per_layer (alias) and per-layer decode inside `layers`?
        if let Some(bw) = c.get("bandwidth_mbps").and_then(|v| v.as_f64()) {
            if bw.is_finite() && bw > 0.0 {
                bandwidth_mbps = bw;
            }
        }
        if let Some(kv) = c.get("kv_budget_mib").and_then(|v| v.as_u64()) {
            kv_budget_mib = kv;
        }
    }
    let id = if row.device_id.is_empty() {
        fallback_id.to_string()
    } else {
        row.device_id.clone()
    };
    DeviceSpec::new(id, decode_tps, bandwidth_mbps, kv_budget_mib)
}

/// Candidate pipeline devices: `paired` + active (90 s window) rows, with
/// self always eligible (coordinator serves even when `worker_active` is
/// unset) and peers eligible only when `worker_active == 1`.
/// Returns device rows in plan order (self first, rest by id).
fn pipeline_candidates(
    rows: Vec<dllm_store::DeviceRow>,
    node_id: &str,
    now_ms: i64,
) -> Vec<dllm_store::DeviceRow> {
    let mut out: Vec<dllm_store::DeviceRow> = rows
        .into_iter()
        .filter(|r| {
            if r.status != "paired" {
                return false;
            }
            if !last_seen_active(&r.last_seen, now_ms) {
                return false;
            }
            if r.device_id == node_id {
                return true;
            }
            matches!(r.worker_active, Some(1))
        })
        .collect();
    out.sort_by(|a, b| {
        let a_self = a.device_id == node_id;
        let b_self = b.device_id == node_id;
        b_self.cmp(&a_self).then_with(|| a.device_id.cmp(&b.device_id))
    });
    out
}

/// Compute a [`dllm_core::PipelinePlan`] for the given rows (already
/// filtered to pipeline candidates). Calls [`plan_layers`] with
/// [`DeviceSpec`]s derived from registry capabilities; single-device
/// collapses to the full 0-27 range inside `plan_layers`.
fn compute_pipeline_plan(
    candidates: &[dllm_store::DeviceRow],
    node_id: &str,
) -> dllm_core::PipelinePlan {
    if candidates.is_empty() {
        // No eligible device (e.g. self row missing): honest single-stage
        // fallback on the coordinator id so callers never see zero stages.
        return plan_layers(
            TOTAL_LAYERS,
            &[DeviceSpec::new(node_id, 16.6, 1000.0, 1024)],
        );
    }
    let specs: Vec<DeviceSpec> = candidates
        .iter()
        .map(|r| device_spec_from_row(r, node_id))
        .collect();
    plan_layers(TOTAL_LAYERS, &specs)
}

/// Map a plan's stages to `device_id -> (layer_start, layer_end)`.
fn stage_map_for(
    plan: &dllm_core::PipelinePlan,
    device_ids_in_order: &[String],
) -> HashMap<String, (u32, u32)> {
    let mut m = HashMap::new();
    for (i, stage) in plan.stages.iter().enumerate() {
        if let Some(id) = device_ids_in_order.get(i) {
            m.insert(id.clone(), (stage.start, stage.end));
        }
    }
    m
}

/// `GET /v1/devices/{id}`: full registry row + honest load + sessions +
/// stage. Unknown id -> 404 `{ok:false}`. No fake load is ever synthesized:
/// - self (`id == node.node_id`) -> `load.source = "live"` (fresh `sysinfo`
///   cpu+mem; `{"source":"none"}` only if the sampler itself fails).
/// - peer with a reported `cpu_pct`+`mem_pct`+`load_updated_at` row ->
///   `load.source = "reported"` with those values.
/// - otherwise -> `load.source = "none"`.
/// Sessions are attributed to self ONLY (single-device serves all; peers
/// get `[]`). Stage mirrors `/v1/plan` for self, `null` for peers.
async fn device_detail(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    // Phase 0 inline reads; production: spawn_blocking.
    let row = match state.store.get_device(&id) {
        Ok(Some(r)) => r,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "ok": false, "error": "unknown device" })),
            );
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
            );
        }
    };
    let now_ms = unix_ms_now();
    let is_self = id == state.node.node_id;
    let permissions: Vec<String> =
        serde_json::from_str(&row.permissions).unwrap_or_default();
    let capabilities: Option<serde_json::Value> = row
        .capabilities
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());
    let layers: Option<serde_json::Value> = row
        .layers
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());
    let worker_active: Option<bool> = match row.worker_active {
        Some(1) => Some(true),
        Some(0) => Some(false),
        _ => None,
    };
    let device = serde_json::json!({
        "device_id": row.device_id,
        "device_name": row.device_name.filter(|s| !s.trim().is_empty()),
        "role": row.role,
        "permissions": permissions,
        "status": row.status,
        "active": last_seen_active(&row.last_seen, now_ms),
        "last_seen": row.last_seen,
        "paired_at": row.paired_at,
        "paired_by": row.paired_by,
        "cert_fp": row.cert_fp,
        "capabilities": capabilities,
        "worker_active": worker_active,
        "layers": layers,
    });
    let load = if is_self {
        match live_load() {
            Some((cpu, mem)) => serde_json::json!({
                "source": "live",
                "cpu_pct": cpu,
                "mem_pct": mem,
            }),
            None => serde_json::json!({ "source": "none" }),
        }
    } else if let (Some(cpu), Some(mem), Some(updated)) =
        (row.cpu_pct, row.mem_pct, row.load_updated_at.clone())
    {
        serde_json::json!({
            "source": "reported",
            "cpu_pct": cpu,
            "mem_pct": mem,
            "updated_at": updated,
        })
    } else {
        serde_json::json!({ "source": "none" })
    };
    let sessions: Vec<serde_json::Value> = if is_self {
        state.store.list_sessions().unwrap_or_default().into_iter().map(|s| {
            serde_json::json!({
                "id": s.id,
                "model": s.model.unwrap_or_else(|| "unknown".to_string()),
                "tokens_out": s.tokens_out,
                "last_token_at": s.last_token_at,
            })
        }).collect()
    } else {
        Vec::new()
    };
    // Stage comes from the live pipeline plan (self + worker_active peers
    // via `plan_layers`); `null` when this device holds no stage. Never
    // hardcoded: single-device still yields self 0-27 via `plan_layers`.
    let stage = {
        let all = state.store.list_devices().unwrap_or_default();
        let cands = pipeline_candidates(all, &state.node.node_id, now_ms);
        let plan = compute_pipeline_plan(&cands, &state.node.node_id);
        let ids: Vec<String> = cands.iter().map(|r| r.device_id.clone()).collect();
        let map = stage_map_for(&plan, &ids);
        match map.get(&id) {
            Some((s, e)) => serde_json::json!({ "layer_start": s, "layer_end": e }),
            None => serde_json::Value::Null,
        }
    };
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "device": device,
            "load": load,
            "sessions": sessions,
            "stage": stage,
        })),
    )
}

/// Real activity feed rolled up from the event log (see
/// [`dllm_store::Store::list_sessions`]). Every entry carries `title`:
/// latest rename if present, else the first user message truncated to 40
/// chars, else `"New chat"` — computed server-side so web and Android stay
/// in sync by polling this endpoint.
async fn list_sessions(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    // Phase 0 inline read; production: spawn_blocking.
    let rows = state.store.list_sessions().unwrap_or_default();
    let sessions: Vec<serde_json::Value> = rows
        .into_iter()
        .map(|s| {
            serde_json::json!({
                "id": s.id,
                "model": s.model.unwrap_or_else(|| "unknown".to_string()),
                "created_at": s.created_at,
                "tokens_out": s.tokens_out,
                "last_token_at": s.last_token_at,
                "title": s.title,
            })
        })
        .collect();
    Json(serde_json::json!({ "sessions": sessions }))
}

/// `DELETE /v1/sessions/{id}`: hard-delete the session's events plus its
/// rename row so it disappears from `GET /v1/sessions` and its SSE resume
/// position is gone (`GET .../events` becomes 404). Unknown id -> 404
/// `{ok:false}`. Broadcasts an ephemeral (unstored) `status` tombstone so
/// live SSE subscribers on that session can close.
async fn delete_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    // Phase 0 inline write; production: spawn_blocking.
    match state.store.delete_session(&id) {
        Ok(true) => {
            let _ = state.tx.send(SseMsg {
                id: 0,
                session_id: id.clone(),
                event: "status".to_string(),
                data: serde_json::json!({"deleted": true}).to_string(),
            });
            (
                StatusCode::OK,
                Json(serde_json::json!({ "ok": true, "id": id })),
            )
        }
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "ok": false, "error": "unknown session" })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
        ),
    }
}

#[derive(Debug, Default, Deserialize)]
struct RenameReq {
    #[serde(default)]
    title: Option<String>,
}

/// `POST /v1/sessions/{id}/rename` body `{title}`: trim, require 1–80
/// chars (400 `{ok:false}` otherwise), 404 for unknown ids. Persists to
/// the prune-resistant `session_titles` table AND appends a
/// `session_renamed` event (survives restarts; the table survives the 24 h
/// TTL prune which only touches `events`) then broadcasts so live SSE
/// subscribers see the rename as `status`.
async fn rename_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    body: Result<Json<RenameReq>, axum::extract::rejection::JsonRejection>,
) -> impl IntoResponse {
    let raw = match body {
        Ok(Json(b)) => b.title,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
            );
        }
    };
    let title = match raw {
        Some(t) => t.trim().to_string(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "ok": false, "error": "title required" })),
            );
        }
    };
    if title.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": "title must not be empty" })),
        );
    }
    if title.chars().count() > 80 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": "title must be at most 80 chars" })),
        );
    }
    // Phase 0 inline I/O; production: spawn_blocking.
    match state.store.session_exists(&id) {
        Ok(true) => {}
        Ok(false) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "ok": false, "error": "unknown session" })),
            );
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
            );
        }
    }
    if let Err(e) = state.store.set_session_title(&id, &title) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
        );
    }
    let payload = serde_json::json!({ "title": title }).to_string();
    let rowid = state.store.append(&id, "session_renamed", &payload).unwrap_or(0);
    let _ = state.tx.send(SseMsg {
        id: rowid,
        session_id: id.clone(),
        event: "session_renamed".to_string(),
        data: payload,
    });
    (
        StatusCode::OK,
        Json(serde_json::json!({ "ok": true, "id": id, "title": title })),
    )
}

/// `GET /v1/plan[?group=]`: pipeline plan via [`plan_layers`].
/// - No `group`: global candidates (self + active `worker_active` peers).
///   Single-device (self only) keeps the legacy `{plan_id:1, note}` shape
///   so old clients/tests stay green; multi-device returns the
///   `plan_layers` id + stages (balanced-split fallback lives inside
///   `plan_layers`).
/// - With `?group=<id>`: candidates scoped to that network's paired+active
///   members (self + `worker_active` peers in the group); 404 unknown group.
///   Always returns the `plan_layers` shape (never hardcoded self 0-27).
#[derive(Debug, Default, Deserialize)]
struct PlanQuery {
    #[serde(default)]
    group: Option<String>,
}

async fn get_plan(
    State(state): State<Arc<AppState>>,
    Query(q): Query<PlanQuery>,
) -> impl IntoResponse {
    let now_ms = unix_ms_now();
    if let Some(gid) = q.group.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        let net = match state.store.get_network(gid) {
            Ok(Some(n)) => n,
            Ok(None) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({ "ok": false, "error": "unknown network" })),
                );
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                );
            }
        };
        let _ = net;
        let members = state.store.list_members(gid).unwrap_or_default();
        let paired_ids: HashMap<String, _> = members
            .into_iter()
            .filter(|m| m.status == "paired")
            .map(|m| (m.device_id.clone(), m))
            .collect();
        let all = state.store.list_devices().unwrap_or_default();
        // Scope to group members; member `last_seen` gates activity when the
        // member row is fresher than the device row (heartbeat touches both).
        let mut scoped: Vec<dllm_store::DeviceRow> = Vec::new();
        for mut r in all.into_iter().filter(|r| paired_ids.contains_key(&r.device_id)) {
            if let Some(m) = paired_ids.get(&r.device_id) {
                // Prefer the fresher of member/device last_seen for activity.
                if last_seen_active(&m.last_seen, now_ms) && !last_seen_active(&r.last_seen, now_ms) {
                    r.last_seen = m.last_seen.clone();
                }
            }
            scoped.push(r);
        }
        // Ensure the admin/self is present even if its device row is missing?
        // No synthesis: only plan over real rows (missing = not planned).
        let cands = pipeline_candidates(scoped, &state.node.node_id, now_ms);
        // If the group has no worker_active peers yet, still plan self alone
        // when self is a member (single-stage 0-27 via plan_layers).
        let plan = compute_pipeline_plan(&cands, &state.node.node_id);
        let ids: Vec<String> = cands.iter().map(|r| r.device_id.clone()).collect();
        let stages: Vec<serde_json::Value> = plan
            .stages
            .iter()
            .enumerate()
            .map(|(i, s)| {
                serde_json::json!({
                    "stage": i,
                    "device_id": ids.get(i).cloned().unwrap_or_default(),
                    "layer_start": s.start,
                    "layer_end": s.end,
                })
            })
            .collect();
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "plan_id": plan.plan_id,
                "group": gid,
                "stages": stages,
            })),
        );
    }
    // Global (legacy) path.
    let all = state.store.list_devices().unwrap_or_default();
    let cands = pipeline_candidates(all, &state.node.node_id, now_ms);
    if cands.len() <= 1 {
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "plan_id": 1,
                "stages": [{
                    "stage": 0,
                    "device_id": state.node.node_id,
                    "layer_start": 0,
                    "layer_end": 27,
                }],
                "note": "single-device fast path",
            })),
        );
    }
    let plan = compute_pipeline_plan(&cands, &state.node.node_id);
    let ids: Vec<String> = cands.iter().map(|r| r.device_id.clone()).collect();
    let stages: Vec<serde_json::Value> = plan
        .stages
        .iter()
        .enumerate()
        .map(|(i, s)| {
            serde_json::json!({
                "stage": i,
                "device_id": ids.get(i).cloned().unwrap_or_default(),
                "layer_start": s.start,
                "layer_end": s.end,
            })
        })
        .collect();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "plan_id": plan.plan_id,
            "stages": stages,
            "note": "multi-device plan via plan_layers",
        })),
    )
}

/// `GET /v1/usage[?group=]`: usage roll-up for the web dashboard Usage tab.
/// All real data, nulls where unknown — never synthesized:
/// - `tokens_out_total` = sum of `token`-kind counts over all sessions;
///   `sessions_total` = session count (both from the event log).
/// - Without `?group`: legacy global shape (one entry per registry row,
///   self first). `per_device` entries carry `worker_active` + `layers`
///   from heartbeat caps/load plus `load_source` + plan layers.
/// - With `?group=<id>`: caller identity comes from `?device_id=` (aliases
///   `?caller=` / `?as=`) or the `x-device-id` header, falling back to the
///   coordinator id when absent. Unknown group -> 404; non-member or
///   revoked -> 403; non-admin member -> totals only (`per_device: []`);
///   admin -> full `per_device` + `plan` scoped to the group.
/// - `bandwidth` is NOT measured anywhere yet -> explicit null.
#[derive(Debug, Default, Deserialize)]
struct UsageQuery {
    #[serde(default)]
    group: Option<String>,
    #[serde(default)]
    device_id: Option<String>,
    #[serde(default)]
    caller: Option<String>,
    #[serde(default, alias = "as")]
    #[allow(dead_code)]
    as_caller: Option<String>,
}

fn usage_caller(q: &UsageQuery, headers: &HeaderMap, node_id: &str) -> String {
    if let Some(v) = q.device_id.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        return v.to_string();
    }
    if let Some(v) = q.caller.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        return v.to_string();
    }
    if let Some(v) = q.as_caller.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        return v.to_string();
    }
    if let Some(v) = headers
        .get("x-device-id")
        .and_then(|h| h.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return v.to_string();
    }
    node_id.to_string()
}

/// Build one `per_device` entry with honest load + heartbeat presence +
/// plan layers. `stage` is `Some((start,end))` when the device holds a
/// pipeline stage, else `None` (null layers, never synthesized).
fn usage_entry(
    r: dllm_store::DeviceRow,
    node_id: &str,
    now_ms: i64,
    tokens_out_total: usize,
    sessions_total: usize,
    stage: Option<(u32, u32)>,
) -> serde_json::Value {
    let is_self = r.device_id == node_id;
    let (load_source, cpu_pct, mem_pct) = if is_self {
        match live_load() {
            Some((cpu, mem)) => ("live", Some(cpu as f64), Some(mem as f64)),
            None => ("none", None, None),
        }
    } else if let (Some(cpu), Some(mem), Some(_)) =
        (r.cpu_pct, r.mem_pct, r.load_updated_at.as_deref())
    {
        ("reported", Some(cpu), Some(mem))
    } else {
        ("none", None, None)
    };
    let cpu_json = cpu_pct.map_or(serde_json::Value::Null, |v| serde_json::json!(v));
    let mem_json = mem_pct.map_or(serde_json::Value::Null, |v| serde_json::json!(v));
    let worker_active: serde_json::Value = match r.worker_active {
        Some(1) => serde_json::json!(true),
        Some(0) => serde_json::json!(false),
        _ => serde_json::Value::Null,
    };
    let layers: serde_json::Value = r
        .layers
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or(serde_json::Value::Null);
    // Honest attribution: sessions carry no device_id, so self gets all
    // tokens/sessions (single-device serves all) unless a group plan
    // distributes them — peers keep honest zeros for now.
    let (tokens_out, sess_count) = if is_self {
        (
            serde_json::json!(tokens_out_total),
            serde_json::json!(sessions_total),
        )
    } else {
        (serde_json::json!(0), serde_json::json!(0))
    };
    let (layer_start, layer_end) = match stage {
        Some((s, e)) => (serde_json::json!(s), serde_json::json!(e)),
        None => (serde_json::Value::Null, serde_json::Value::Null),
    };
    serde_json::json!({
        "device_id": r.device_id,
        "device_name": r.device_name.filter(|s| !s.trim().is_empty()),
        "role": r.role,
        "active": last_seen_active(&r.last_seen, now_ms),
        "status": r.status,
        "tokens_out": tokens_out,
        "sessions": sess_count,
        "cpu_pct": cpu_json,
        "mem_pct": mem_json,
        "load_source": load_source,
        "worker_active": worker_active,
        "layers": layers,
        "layer_start": layer_start,
        "layer_end": layer_end,
    })
}

async fn get_usage(
    State(state): State<Arc<AppState>>,
    Query(q): Query<UsageQuery>,
    headers: HeaderMap,
) -> impl IntoResponse {
    // Phase 0 inline reads; production: spawn_blocking.
    let now_ms = unix_ms_now();
    let sessions = state.store.list_sessions().unwrap_or_default();
    let tokens_out_total: usize = sessions.iter().map(|s| s.tokens_out).sum();
    let sessions_total = sessions.len();

    if let Some(gid) = q.group.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        let gid = gid.to_string();
        let net = match state.store.get_network(&gid) {
            Ok(Some(n)) => n,
            Ok(None) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({ "ok": false, "error": "unknown network" })),
                );
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                );
            }
        };
        let caller = usage_caller(&q, &headers, &state.node.node_id);
        let member = state.store.get_member(&gid, &caller).unwrap_or(None);
        let paired = matches!(member.as_ref(), Some(m) if m.status == "paired");
        if !paired {
            // Forbidden but honest: the web Usage tab parses totals out of a
            // 403 body, so include them instead of `{ok:false}` alone.
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({
                    "ok": false,
                    "error": "not a member",
                    "group": gid,
                    "tokens_out_total": tokens_out_total,
                    "sessions_total": sessions_total,
                    "bandwidth": null,
                })),
            );
        }
        let is_admin = member.as_ref().is_some_and(|m| m.role == "admin")
            || caller == net.admin_device_id;
        // Group-scoped plan (paired+active members via plan_layers).
        let members = state.store.list_members(&gid).unwrap_or_default();
        let paired_ids: HashMap<String, _> = members
            .into_iter()
            .filter(|m| m.status == "paired")
            .map(|m| (m.device_id.clone(), m))
            .collect();
        let all = state.store.list_devices().unwrap_or_default();
        let mut scoped: Vec<dllm_store::DeviceRow> = Vec::new();
        for mut r in all.into_iter().filter(|r| paired_ids.contains_key(&r.device_id)) {
            if let Some(m) = paired_ids.get(&r.device_id) {
                if last_seen_active(&m.last_seen, now_ms) && !last_seen_active(&r.last_seen, now_ms) {
                    r.last_seen = m.last_seen.clone();
                }
            }
            scoped.push(r);
        }
        scoped.sort_by(|a, b| {
            let a_self = a.device_id == state.node.node_id;
            let b_self = b.device_id == state.node.node_id;
            b_self.cmp(&a_self).then_with(|| a.device_id.cmp(&b.device_id))
        });
        let cands = pipeline_candidates(scoped.clone(), &state.node.node_id, now_ms);
        let plan = compute_pipeline_plan(&cands, &state.node.node_id);
        let ids: Vec<String> = cands.iter().map(|r| r.device_id.clone()).collect();
        let smap = stage_map_for(&plan, &ids);
        let plan_stages: Vec<serde_json::Value> = plan
            .stages
            .iter()
            .enumerate()
            .map(|(i, s)| {
                serde_json::json!({
                    "device_id": ids.get(i).cloned().unwrap_or_default(),
                    "layer_start": s.start,
                    "layer_end": s.end,
                })
            })
            .collect();
        if !is_admin {
            // Non-admin member: totals only, no per-device breakdown.
            return (
                StatusCode::OK,
                Json(serde_json::json!({
                    "group": gid,
                    "tokens_out_total": tokens_out_total,
                    "sessions_total": sessions_total,
                    "per_device": [],
                    "plan": { "stages": plan_stages },
                    "bandwidth": null,
                })),
            );
        }
        let per_device: Vec<serde_json::Value> = scoped
            .into_iter()
            .map(|r| {
                let st = smap.get(&r.device_id).copied();
                usage_entry(r, &state.node.node_id, now_ms, tokens_out_total, sessions_total, st)
            })
            .collect();
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "group": gid,
                "tokens_out_total": tokens_out_total,
                "sessions_total": sessions_total,
                "per_device": per_device,
                "plan": { "stages": plan_stages },
                "bandwidth": null,
            })),
        );
    }

    // Legacy global path (no group): preserves the old contract shape plus
    // the new `worker_active`/`layers` fields.
    let rows = state.store.list_devices().unwrap_or_default();
    let cands = pipeline_candidates(rows.clone(), &state.node.node_id, now_ms);
    let plan = compute_pipeline_plan(&cands, &state.node.node_id);
    let ids: Vec<String> = cands.iter().map(|r| r.device_id.clone()).collect();
    let smap = stage_map_for(&plan, &ids);
    let mut devs = rows;
    devs.sort_by(|a, b| {
        let a_self = a.device_id == state.node.node_id;
        let b_self = b.device_id == state.node.node_id;
        b_self.cmp(&a_self).then_with(|| a.device_id.cmp(&b.device_id))
    });
    // Legacy expectation: self always carries 0-27 even when the candidate
    // set is self-only; `compute_pipeline_plan` already yields that via
    // `plan_layers` (single stage 0-27). Peers without a stage get nulls.
    let per_device: Vec<serde_json::Value> = devs
        .into_iter()
        .map(|r| {
            let st = smap.get(&r.device_id).copied();
            usage_entry(r, &state.node.node_id, now_ms, tokens_out_total, sessions_total, st)
        })
        .collect();
    let plan_stages: Vec<serde_json::Value> = if cands.len() <= 1 {
        vec![serde_json::json!({
            "device_id": state.node.node_id,
            "layer_start": 0,
            "layer_end": 27,
        })]
    } else {
        plan.stages
            .iter()
            .enumerate()
            .map(|(i, s)| {
                serde_json::json!({
                    "device_id": ids.get(i).cloned().unwrap_or_default(),
                    "layer_start": s.start,
                    "layer_end": s.end,
                })
            })
            .collect()
    };
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "tokens_out_total": tokens_out_total,
            "sessions_total": sessions_total,
            "per_device": per_device,
            "plan": { "stages": plan_stages },
            "bandwidth": null,
        })),
    )
}

#[derive(Debug, Default, Deserialize)]
struct CreateSessionReq {
    #[serde(default)]
    model: Option<String>,
}

#[derive(Debug, Serialize)]
struct CreateSessionRes {
    id: String,
}

static SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

fn new_session_id() -> String {
    let n = SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("sess-{now}-{n}")
}

async fn create_session(
    State(state): State<Arc<AppState>>,
    body: Option<Json<CreateSessionReq>>,
) -> impl IntoResponse {
    let id = new_session_id();
    let model = body.and_then(|b| b.0.model).unwrap_or_else(|| "qwen3-0.6b-q4".to_string());
    let payload = serde_json::json!({ "model": model }).to_string();
    // Phase 0: tiny inline write; production MUST be spawn_blocking.
    let rowid = state.store.append(&id, "session_created", &payload).unwrap_or(0);
    let msg = SseMsg {
        id: rowid,
        session_id: id.clone(),
        event: "status".to_string(),
        data: payload,
    };
    let _ = state.tx.send(msg);
    (StatusCode::CREATED, Json(CreateSessionRes { id }))
}

#[derive(Debug, Deserialize)]
struct PostMessageReq {
    /// Accepts `text`, `content`, or `prompt` keys.
    #[serde(default, alias = "content", alias = "prompt")]
    text: String,
}

async fn post_message(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(body): Json<PostMessageReq>,
) -> impl IntoResponse {
    let prompt = body.text.clone();
    // Record the user turn (Phase 0 inline; production: spawn_blocking).
    let user_payload = serde_json::json!({ "role": "user", "text": prompt }).to_string();
    let user_row = state.store.append(&id, "user_message", &user_payload).unwrap_or(0);
    let _ = state.tx.send(SseMsg {
        id: user_row,
        session_id: id.clone(),
        event: "status".to_string(),
        data: user_payload,
    });

    // Spawn generation (real or mock, per the engine) -> store + broadcast.
    // Contract shapes (contracts/event-log.md): `event: token` carries
    // {"pos","text"} per chunk; `event: commit` carries {"pos"} — single
    // device commits each token durably on append, so one commit per token.
    // The loop polls `cancel` between tokens; on cancel it appends +
    // broadcasts `status{"state":"cancelled"}` and exits (idempotent stop).
    let mut rx = state.engine.generate_stream(prompt);
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_loop = cancel.clone();
    let state2 = state.clone();
    let session_id = id.clone();
    let session_for_cleanup = id.clone();
    let cancel_for_cleanup = cancel.clone();
    let state_for_cleanup = state.clone();
    let handle = tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            if cancel_loop.load(Ordering::Relaxed) {
                let payload = serde_json::json!({ "state": "cancelled" }).to_string();
                let store = state2.store.clone();
                let sess = session_id.clone();
                let pl = payload.clone();
                let rowid =
                    tokio::task::spawn_blocking(move || store.append(&sess, "status", &pl))
                        .await
                        .ok()
                        .and_then(|r| r.ok())
                        .unwrap_or(0);
                let _ = state2.tx.send(SseMsg {
                    id: rowid,
                    session_id: session_id.clone(),
                    event: "status".to_string(),
                    data: payload,
                });
                break;
            }
            let pos = ev.pos;
            let payload =
                serde_json::json!({ "pos": ev.pos, "text": ev.text, "done": ev.done })
                    .to_string();
            // Correct blocking discipline for the writer path.
            let store = state2.store.clone();
            let sess = session_id.clone();
            let pl = payload.clone();
            let rowid = tokio::task::spawn_blocking(move || store.append(&sess, "token", &pl))
                .await
                .ok()
                .and_then(|r| r.ok())
                .unwrap_or(0);
            let done = ev.done;
            let (tok_event, tok_data) = sse_shape_for_live("token", &payload);
            let _ = state2.tx.send(SseMsg {
                id: rowid,
                session_id: session_id.clone(),
                event: tok_event,
                data: tok_data,
            });
            // Durability mark: the token row above is committed (WAL).
            let commit_payload = serde_json::json!({ "pos": pos }).to_string();
            let store_c = state2.store.clone();
            let sess_c = session_id.clone();
            let commit_pl = commit_payload.clone();
            let commit_row =
                tokio::task::spawn_blocking(move || store_c.append(&sess_c, "commit", &commit_pl))
                    .await
                    .ok()
                    .and_then(|r| r.ok())
                    .unwrap_or(0);
            let _ = state2.tx.send(SseMsg {
                id: commit_row,
                session_id: session_id.clone(),
                event: "commit".to_string(),
                data: commit_payload,
            });
            if cancel_loop.load(Ordering::Relaxed) {
                let payload = serde_json::json!({ "state": "cancelled" }).to_string();
                let store = state2.store.clone();
                let sess = session_id.clone();
                let pl = payload.clone();
                let rowid =
                    tokio::task::spawn_blocking(move || store.append(&sess, "status", &pl))
                        .await
                        .ok()
                        .and_then(|r| r.ok())
                        .unwrap_or(0);
                let _ = state2.tx.send(SseMsg {
                    id: rowid,
                    session_id: session_id.clone(),
                    event: "status".to_string(),
                    data: payload,
                });
                break;
            }
            if done {
                break;
            }
        }
        // Best-effort cleanup: drop the run entry when this generation ends,
        // but only if it is still ours (a newer generation may have replaced
        // it — never delete a successor's flag).
        let finished_ours = {
            match state_for_cleanup.gen_runs.lock() {
                Ok(runs) => match runs.get(&session_for_cleanup) {
                    Some(cur) => Arc::ptr_eq(&cur.cancel, &cancel_for_cleanup),
                    None => false,
                },
                Err(_) => false,
            }
        };
        if finished_ours {
            if let Ok(mut runs) = state_for_cleanup.gen_runs.lock() {
                if let Some(cur) = runs.get(&session_for_cleanup) {
                    if Arc::ptr_eq(&cur.cancel, &cancel_for_cleanup) {
                        runs.remove(&session_for_cleanup);
                    }
                }
            }
        }
    });
    // Track the handle + cancel flag (replaces any prior run for this
    // session; the prior task keeps running but `stop` targets the latest).
    if let Ok(mut runs) = state.gen_runs.lock() {
        runs.insert(
            id.clone(),
            GenRun {
                cancel,
                handle,
            },
        );
    }

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "accepted": true, "session": id })),
    )
}

/// `POST /v1/sessions/{id}/stop`: cancel a live generation.
/// Sets the session's cancel flag; the decode loop observes it between
/// tokens and emits `status{"state":"cancelled"}`. Idempotent:
/// - running generation -> `{ok:true, cancelled:true}`;
/// - no live generation (finished / never started / already stopped) ->
///   `{ok:true, cancelled:false}`;
/// - unknown session (no events) -> 404 `{ok:false}`.
async fn stop_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let exists = state.store.session_exists(&id).unwrap_or(false);
    if !exists {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "ok": false, "error": "unknown session" })),
        );
    }
    let cancelled = match state.gen_runs.lock() {
        Ok(runs) => match runs.get(&id) {
            Some(run) => {
                if run.handle.is_finished() {
                    false
                } else if run.cancel.load(Ordering::Relaxed) {
                    false
                } else {
                    run.cancel.store(true, Ordering::Relaxed);
                    true
                }
            }
            None => false,
        },
        Err(_) => false,
    };
    (
        StatusCode::OK,
        Json(serde_json::json!({ "ok": true, "cancelled": cancelled })),
    )
}

// ---------------------------------------------------------------------------
// Networks / groups.
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct CreateNetworkReq {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    password: Option<String>,
    #[serde(default)]
    open_join: Option<serde_json::Value>,
    /// Optional creator override; defaults to the coordinator node_id.
    #[serde(default)]
    device_id: Option<String>,
    #[serde(default)]
    admin_device_id: Option<String>,
}

fn parse_open_join(v: Option<&serde_json::Value>) -> bool {
    match v {
        None => true,
        Some(serde_json::Value::Bool(b)) => *b,
        Some(serde_json::Value::Number(n)) => n.as_i64().unwrap_or(1) != 0,
        Some(_) => true,
    }
}

/// `GET /v1/networks`: every group (no secrets). Auto-seeds the default
/// group when the table is empty so the list is never stubbed-empty.
async fn list_networks(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let _ = state.store.ensure_default_network(&state.node.node_id);
    let nets = state.store.list_networks().unwrap_or_default();
    let out: Vec<serde_json::Value> = nets
        .into_iter()
        .map(|n| {
            let member_count = state.store.count_paired_members(&n.id).unwrap_or(0);
            serde_json::json!({
                "id": n.id,
                "name": n.name,
                "open_join": n.open_join != 0,
                "has_password": n.password_hash.is_some(),
                "admin_device_id": n.admin_device_id,
                "created_at": n.created_at,
                "member_count": member_count,
            })
        })
        .collect();
    (StatusCode::OK, Json(serde_json::json!({ "networks": out })))
}

/// `POST /v1/networks {name,password?,open_join?}` -> `{id,qr}`.
/// `name` trimmed 1-40 (400 otherwise, including a missing/garbled body).
/// `password` empty/absent = open. Admin defaults to the coordinator
/// node_id (override via `device_id`/`admin_device_id` for tests).
async fn create_network(
    State(state): State<Arc<AppState>>,
    body: Result<Json<CreateNetworkReq>, axum::extract::rejection::JsonRejection>,
) -> impl IntoResponse {
    let body = match body {
        Ok(Json(b)) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
            );
        }
    };
    let name = body.name.unwrap_or_default().trim().to_string();
    if name.is_empty() || name.chars().count() > 40 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": "name must be 1-40 chars" })),
        );
    }
    let admin = body
        .admin_device_id
        .as_deref()
        .or(body.device_id.as_deref())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(&state.node.node_id)
        .to_string();
    let open = body.open_join.as_ref().map(|v| parse_open_join(Some(v))).unwrap_or(true);
    let pw = body.password.as_deref().map(str::trim).filter(|s| !s.is_empty());
    match state.store.create_network(&name, pw, open, &admin) {
        Ok(row) => {
            // Ensure the admin also has a device row (friendly join path).
            let _ = state.store.upsert_device(&admin, "coordinator", r#"["infer","chat"]"#, "", &admin);
            let qr = network_qr(&row.id, &row.qr_secret, row.password_hash.is_some(), &state.node.fingerprint);
            (
                StatusCode::CREATED,
                Json(serde_json::json!({ "id": row.id, "qr": qr })),
            )
        }
        Err(dllm_store::StoreError::InvalidName) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": "name must be 1-40 chars" })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
        ),
    }
}

/// `GET /v1/networks/{id}`: group detail (no secret). 404 unknown.
async fn get_network(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.store.get_network(&id) {
        Ok(Some(n)) => {
            let member_count = state.store.count_paired_members(&n.id).unwrap_or(0);
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "id": n.id,
                    "name": n.name,
                    "open_join": n.open_join != 0,
                    "has_password": n.password_hash.is_some(),
                    "admin_device_id": n.admin_device_id,
                    "created_at": n.created_at,
                    "member_count": member_count,
                })),
            )
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "ok": false, "error": "unknown network" })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
        ),
    }
}

/// `GET /v1/networks/{id}/devices`: group members joined to the registry.
/// `active` reuses the 90 s `last_seen` rule (member row, falling back to
/// the device row). Each entry carries `worker_active`/`cpu_pct`/`mem_pct`
/// (null when never reported — never synthesized) plus the live group plan
/// layers (`layer_start`/`layer_end`, null when the device holds no stage).
async fn network_devices(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let net = match state.store.get_network(&id) {
        Ok(Some(n)) => n,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "ok": false, "error": "unknown network" })),
            );
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
            );
        }
    };
    let _ = net;
    let now_ms = unix_ms_now();
    let members = state.store.list_members(&id).unwrap_or_default();
    // Live group plan over paired+active worker candidates.
    let all = state.store.list_devices().unwrap_or_default();
    let by_id: HashMap<String, dllm_store::DeviceRow> =
        all.into_iter().map(|r| (r.device_id.clone(), r)).collect();
    let mut scoped_rows: Vec<dllm_store::DeviceRow> = Vec::new();
    for m in members.iter().filter(|m| m.status == "paired") {
        if let Some(r) = by_id.get(&m.device_id).cloned() {
            let mut r = r;
            if last_seen_active(&m.last_seen, now_ms) && !last_seen_active(&r.last_seen, now_ms) {
                r.last_seen = m.last_seen.clone();
            }
            scoped_rows.push(r);
        }
    }
    let cands = pipeline_candidates(scoped_rows, &state.node.node_id, now_ms);
    let plan = compute_pipeline_plan(&cands, &state.node.node_id);
    let ids: Vec<String> = cands.iter().map(|r| r.device_id.clone()).collect();
    let smap = stage_map_for(&plan, &ids);
    let mut devs: Vec<serde_json::Value> = Vec::new();
    for m in members {
        let row = by_id.get(&m.device_id);
        let (device_name, worker_active, cpu_pct, mem_pct, layers, device_last_seen) =
            match row {
                Some(r) => (
                    r.device_name.clone().filter(|s| !s.trim().is_empty()),
                    match r.worker_active {
                        Some(1) => serde_json::json!(true),
                        Some(0) => serde_json::json!(false),
                        _ => serde_json::Value::Null,
                    },
                    r.cpu_pct.map_or(serde_json::Value::Null, |v| serde_json::json!(v)),
                    r.mem_pct.map_or(serde_json::Value::Null, |v| serde_json::json!(v)),
                    r.layers
                        .as_deref()
                        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
                        .unwrap_or(serde_json::Value::Null),
                    r.last_seen.clone(),
                ),
                None => (
                    None,
                    serde_json::Value::Null,
                    serde_json::Value::Null,
                    serde_json::Value::Null,
                    serde_json::Value::Null,
                    m.last_seen.clone(),
                ),
            };
        // Active when EITHER the member or the device row was seen recently.
        let active =
            last_seen_active(&m.last_seen, now_ms) || last_seen_active(&device_last_seen, now_ms);
        let (ls, le) = match smap.get(&m.device_id) {
            Some((s, e)) => (serde_json::json!(s), serde_json::json!(e)),
            None => (serde_json::Value::Null, serde_json::Value::Null),
        };
        devs.push(serde_json::json!({
            "device_id": m.device_id,
            "device_name": device_name,
            "role": m.role,
            "status": m.status,
            "active": active,
            "last_seen": m.last_seen,
            "worker_active": worker_active,
            "cpu_pct": cpu_pct,
            "mem_pct": mem_pct,
            "layers": layers,
            "layer_start": ls,
            "layer_end": le,
        }));
    }
    devs.sort_by(|a, b| {
        let a_self = a["device_id"] == state.node.node_id;
        let b_self = b["device_id"] == state.node.node_id;
        b_self.cmp(&a_self).then_with(|| a["device_id"].as_str().cmp(&b["device_id"].as_str()))
    });
    (
        StatusCode::OK,
        Json(serde_json::json!({ "group_id": id, "devices": devs })),
    )
}

#[derive(Debug, Default, Deserialize)]
struct JoinReq {
    /// Joining device. Optional for web-compat: the browser `POST .../join
    /// {password?}` shape carries no identity (password check only, no
    /// membership row). Android always sends `device_id` (+ optional
    /// `qr_secret` from the QR scan, accepted and ignored — the password is
    /// the auth gate). Falls back to the `x-device-id` header when absent.
    #[serde(default)]
    device_id: Option<String>,
    #[serde(default)]
    device_name: Option<String>,
    #[serde(default)]
    password: Option<String>,
    /// Accepted for Android QR flows; not a second auth gate (ignored).
    #[serde(default)]
    #[allow(dead_code)]
    qr_secret: Option<String>,
}

/// `POST /v1/networks/{id}/join {password?}` (web) or
/// `{device_id,device_name?,password?,qr_secret?}` (Android):
/// - 404 unknown group;
/// - 401 `{error:"bad-password"}` when the group is password-protected and
///   the password is missing/wrong (never leak which);
/// - 403 `{error:"closed"}` when `open_join == 0` and the joiner is not the
///   admin (closed groups need an admin-side add; anonymous web checks
///   without identity are always refused on closed groups);
/// - 409 `{error:"full"}` when 5 devices are already paired and the joiner
///   is not already a paired member.
/// - Anonymous (no `device_id` and no `x-device-id` header): password-checked
///   only, returns `{ok:true, group_id}` without touching membership rows,
///   so the browser join button succeeds without polluting the registry.
async fn join_network(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    body: Result<Json<JoinReq>, axum::extract::rejection::JsonRejection>,
) -> impl IntoResponse {
    let body = match body {
        Ok(Json(b)) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
            );
        }
    };
    let device_id = body
        .device_id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| {
            headers
                .get("x-device-id")
                .and_then(|h| h.to_str().ok())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        });
    let net = match state.store.get_network(&id) {
        Ok(Some(n)) => n,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "ok": false, "error": "unknown network" })),
            );
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
            );
        }
    };
    if let Some(want) = net.password_hash.as_deref() {
        let got = body.password.as_deref().unwrap_or("");
        if dllm_store::hash_password(got) != want {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "ok": false, "error": "bad-password" })),
            );
        }
    }
    let Some(device_id) = device_id else {
        // Anonymous web password-check: no identity, no membership change.
        if net.open_join == 0 {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({ "ok": false, "error": "closed" })),
            );
        }
        return (
            StatusCode::OK,
            Json(serde_json::json!({ "ok": true, "group_id": id })),
        );
    };
    if net.open_join == 0 && device_id != net.admin_device_id {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({ "ok": false, "error": "closed" })),
        );
    }
    let existing = state.store.get_member(&id, &device_id).unwrap_or(None);
    let already_paired = matches!(existing.as_ref(), Some(m) if m.status == "paired");
    if !already_paired {
        let paired = state.store.count_paired_members(&id).unwrap_or(0);
        if paired >= NETWORK_MAX_PAIRED {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({ "ok": false, "error": "full" })),
            );
        }
    }
    // Ensure a device row exists (friendly path for fresh phones).
    let _ = state.store.upsert_device(&device_id, "worker", "[]", "", "");
    if let Some(name) = body.device_name.as_deref() {
        let clean = name.trim();
        if !clean.is_empty() {
            let _ = state.store.set_device_name(&device_id, Some(clean));
        }
    }
    let role = if device_id == net.admin_device_id { "admin" } else { "member" };
    if let Err(e) = state.store.upsert_member(&id, &device_id, role) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
        );
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({ "ok": true, "group_id": id, "device_id": device_id, "role": role })),
    )
}

#[derive(Debug, Deserialize)]
struct LeaveReq {
    device_id: String,
}

/// `POST /v1/networks/{id}/leave {device_id}`: hard-delete the membership
/// row (idempotent — leaving twice still returns `{ok:true}`). 404 unknown
/// group. The admin leaving does NOT delete the group (rows stay queryable);
/// the admin row is simply removed.
async fn leave_network(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    body: Option<Json<LeaveReq>>,
) -> impl IntoResponse {
    let device_id = body
        .map(|b| b.0.device_id.trim().to_string())
        .unwrap_or_default();
    if device_id.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": "device_id required" })),
        );
    }
    match state.store.get_network(&id) {
        Ok(Some(_)) => {}
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "ok": false, "error": "unknown network" })),
            );
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
            );
        }
    }
    let _ = state.store.remove_member(&id, &device_id);
    (
        StatusCode::OK,
        Json(serde_json::json!({ "ok": true, "group_id": id, "device_id": device_id })),
    )
}

#[derive(Debug, Deserialize)]
struct RotateReq {
    #[serde(default)]
    device_id: Option<String>,
}

/// `POST /v1/networks/{id}/rotate-qr` (admin only): mint a fresh `qr_secret`
/// and return the new `{ok:true, qr}`. Caller passes `{device_id}`; 403
/// unless the caller is the admin member (or `admin_device_id`).
async fn rotate_qr(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    body: Option<Json<RotateReq>>,
) -> impl IntoResponse {
    let net = match state.store.get_network(&id) {
        Ok(Some(n)) => n,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "ok": false, "error": "unknown network" })),
            );
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
            );
        }
    };
    let caller = body
        .and_then(|b| b.0.device_id)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| state.node.node_id.clone());
    let member = state.store.get_member(&id, &caller).unwrap_or(None);
    let is_admin = caller == net.admin_device_id
        || matches!(member.as_ref(), Some(m) if m.role == "admin" && m.status == "paired");
    if !is_admin {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({ "ok": false, "error": "admin only" })),
        );
    }
    match state.store.rotate_qr_secret(&id) {
        Ok(Some(secret)) => {
            let qr = network_qr(&id, &secret, net.password_hash.is_some(), &state.node.fingerprint);
            (
                StatusCode::OK,
                Json(serde_json::json!({ "ok": true, "qr": qr })),
            )
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "ok": false, "error": "unknown network" })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
        ),
    }
}

/// SSE: replay missed events from `?last_event=K` (preferred — browsers'
/// `EventSource` cannot send headers) or the `Last-Event-ID` header, then
/// stream live. Contract shapes per `contracts/event-log.md`:
/// `event: token` + `{"pos","text"}`, `event: commit` + `{"pos"}`,
/// `event: status` for the rest. `id:` is always `events.id`.
/// Unknown sessions (no events — never created or already deleted) ->
/// 404 `{ok:false}` so clients can drop the chat instead of hanging.
///
/// No compression on this route (see [`router`]).
#[derive(Debug, Default, Deserialize)]
struct EventsQuery {
    #[serde(default)]
    last_event: Option<i64>,
    #[serde(default, alias = "lastEventId", alias = "last_event_id")]
    last_event_id_alias: Option<i64>,
}

async fn session_events(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(q): Query<EventsQuery>,
    headers: HeaderMap,
) -> Response {
    // Phase 0 inline reads; production: spawn_blocking.
    match state.store.session_exists(&id) {
        Ok(true) => {}
        Ok(false) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "ok": false, "error": "unknown session" })),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
            )
                .into_response();
        }
    }
    let header_last: Option<i64> = headers
        .get("last-event-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse().ok());
    let last_id: i64 = q
        .last_event
        .or(q.last_event_id_alias)
        .or(header_last)
        .unwrap_or(0)
        .max(0);
    // Phase 0 inline read; production: spawn_blocking.
    let missed = state.store.replay_since(&id, last_id).unwrap_or_default();

    let session = id.clone();
    let rx = state.tx.subscribe();
    let stream = async_stream::stream! {
        for ev in missed {
            let (event, data) = sse_shape_for_stored(&ev.kind, &ev.payload);
            let sse = SseEvent::default()
                .event(event)
                .id(ev.id.to_string())
                .data(data);
            yield Ok::<_, axum::Error>(sse);
        }
        let mut rx = rx;
        loop {
            match rx.recv().await {
                Ok(msg) => {
                    if msg.session_id != session {
                        continue;
                    }
                    let (event, data) = sse_shape_for_live(&msg.event, &msg.data);
                    let sse = SseEvent::default()
                        .event(event)
                        .id(msg.id.to_string())
                        .data(data);
                    yield Ok(sse);
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    };

    Sse::new(stream)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dllm_core::MockEngine;
    use tower::ServiceExt;

    static NEXT_DB: AtomicU64 = AtomicU64::new(0);

    fn temp_db(tag: &str) -> std::path::PathBuf {
        let n = NEXT_DB.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "dllm-serve-{}-{tag}-{n}.db",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        path
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }

    #[tokio::test]
    async fn api_stats_reports_counts_engine_and_uptime() {
        let path = temp_db("stats");
        let store = Arc::new(Store::open(&path).expect("open store"));
        store.append("sess-a", "session_created", "{}").unwrap();
        store.append("sess-a", "token", "{}").unwrap();
        store.append("sess-b", "session_created", "{}").unwrap();

        let state = new_state(Arc::new(MockEngine::new()), store);
        let res = router(state)
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/stats")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        let body = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .expect("body");
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["engine"], "mock");
        assert_eq!(v["sessions"], 2);
        assert_eq!(v["events"], 3);
        assert_eq!(v["node_id"], "dllm-dev-1");
        assert!(v["uptime_s"].as_u64().is_some());

        cleanup(&path);
    }

    #[tokio::test]
    async fn maintenance_task_prunes_events_on_startup_tick() {
        // Determinism: the env override must not fight the ttl=0 used here.
        std::env::remove_var("DLLM_EVENT_TTL_SECS");

        let path = temp_db("maintenance");
        let store = Arc::new(Store::open(&path).expect("open store"));
        store.append("sess", "session_created", "{}").unwrap();
        store.append("sess", "token", "{}").unwrap();

        // Wait past one second so the first tick's cutoff (second precision)
        // strictly exceeds the events' timestamps.
        tokio::time::sleep(Duration::from_millis(1100)).await;

        let handle = spawn_maintenance(store.clone(), 0);
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while store.count_events().unwrap() != 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "maintenance loop did not prune in time"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        handle.abort();

        cleanup(&path);
    }

    fn seed_self(store: &Store, node_id: &str) {
        store
            .upsert_device(node_id, "coordinator", r#"["infer","chat"]"#, "fp-self", node_id)
            .unwrap();
        store.touch_last_seen(node_id).unwrap();
        store.set_status(node_id, "paired").unwrap();
    }

    async fn body_json(res: axum::response::Response) -> serde_json::Value {
        let body = axum::body::to_bytes(res.into_body(), usize::MAX)
            .await
            .expect("body");
        serde_json::from_slice(&body).unwrap()
    }

    fn post_json(uri: &str, value: serde_json::Value) -> axum::http::Request<axum::body::Body> {
        axum::http::Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(axum::body::Body::from(value.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn devices_heartbeat_list_approve_revoke_roundtrip() {
        let path = temp_db("devices");
        let store = Arc::new(Store::open(&path).expect("open store"));
        seed_self(&store, "dllm-dev-1");
        let state = new_state(Arc::new(MockEngine::new()), store);
        let app = || router(state.clone());

        // Worker lifeline.
        let res = app()
            .oneshot(post_json(
                "/v1/devices/heartbeat",
                serde_json::json!({"device_id": "pixel-8"}),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(body_json(res).await["ok"], true);

        // List: self first, worker active + paired.
        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/devices")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        let devs = v["devices"].as_array().unwrap();
        assert_eq!(devs.len(), 2);
        assert_eq!(devs[0]["device_id"], "dllm-dev-1");
        assert_eq!(devs[0]["role"], "coordinator");
        assert_eq!(devs[0]["status"], "paired");
        assert_eq!(devs[0]["active"], true);
        assert_eq!(devs[1]["device_id"], "pixel-8");
        assert_eq!(devs[1]["role"], "worker");
        assert_eq!(devs[1]["active"], true);

        // Revoke flips status; approve restores; unknown id is 404.
        for (uri, want) in [
            ("/v1/devices/pixel-8/revoke", "revoked"),
            ("/v1/devices/pixel-8/approve", "paired"),
        ] {
            let res = app()
                .oneshot(post_json(uri, serde_json::json!({})))
                .await
                .expect("oneshot");
            assert_eq!(res.status(), StatusCode::OK);
            let res = app()
                .oneshot(
                    axum::http::Request::builder()
                        .uri("/v1/devices")
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .expect("oneshot");
            let v = body_json(res).await;
            assert_eq!(v["devices"][1]["status"], want);
        }
        let res = app()
            .oneshot(post_json("/v1/devices/ghost/revoke", serde_json::json!({})))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::NOT_FOUND);

        cleanup(&path);
    }

    #[tokio::test]
    async fn sessions_list_plan_and_pairing_uri_are_real() {
        let path = temp_db("sesslist");
        let store = Arc::new(Store::open(&path).expect("open store"));
        seed_self(&store, "dllm-dev-1");
        store
            .append("sess-a", "session_created", r#"{"model":"qwen3-0.6b-q4"}"#)
            .unwrap();
        store.append("sess-a", "token", r#"{"pos":0}"#).unwrap();
        let state = new_state(Arc::new(MockEngine::new()), store);
        let app = || router(state.clone());

        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/sessions")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_eq!(v["sessions"][0]["id"], "sess-a");
        assert_eq!(v["sessions"][0]["model"], "qwen3-0.6b-q4");
        assert_eq!(v["sessions"][0]["tokens_out"], 1);
        assert!(v["sessions"][0]["last_token_at"].is_string());

        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/plan")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        let v = body_json(res).await;
        assert_eq!(v["plan_id"], 1);
        assert_eq!(v["stages"][0]["device_id"], "dllm-dev-1");
        assert_eq!(v["stages"][0]["layer_start"], 0);
        assert_eq!(v["stages"][0]["layer_end"], 27);
        assert_eq!(v["note"], "single-device fast path");

        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/pairing-uri")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        let v = body_json(res).await;
        let uri = v["uri"].as_str().unwrap();
        assert!(uri.starts_with("dllm://pair?host="));
        assert!(uri.contains("&quic=8443&"));

        // Health ping keeps the self row active (no invented numbers).
        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);

        cleanup(&path);
    }

    #[tokio::test]
    async fn device_detail_live_reported_none_and_404() {
        let path = temp_db("detail");
        let store = Arc::new(Store::open(&path).expect("open store"));
        seed_self(&store, "dllm-dev-1");
        store.set_device_name("dllm-dev-1", Some("TEST-HOST")).unwrap();
        store
            .append("sess-a", "session_created", r#"{"model":"qwen3-0.6b-q4"}"#)
            .unwrap();
        store.append("sess-a", "token", r#"{"pos":0,"text":"hi"}"#).unwrap();
        let state = new_state(Arc::new(MockEngine::new()), store);
        let app = || router(state.clone());

        // Heartbeat with friendly name + load + caps persists on the row.
        let res = app()
            .oneshot(post_json(
                "/v1/devices/heartbeat",
                serde_json::json!({
                    "device_id": "peer-1",
                    "device_name": "PEER-HOST",
                    "load": {"cpu_pct": 12.5, "mem_pct": 33.25},
                    "capabilities": {"kv_pages": 8}
                }),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);

        // List now includes device_name (null for rows without one is covered
        // by the self row having TEST-HOST and peer having PEER-HOST).
        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/devices")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        let v = body_json(res).await;
        let devs = v["devices"].as_array().unwrap();
        assert_eq!(devs[0]["device_name"], "TEST-HOST");
        assert_eq!(devs[1]["device_name"], "PEER-HOST");

        // Self detail: live load with real numbers, sessions attributed, stage set.
        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/devices/dllm-dev-1")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_eq!(v["device"]["device_id"], "dllm-dev-1");
        assert_eq!(v["device"]["device_name"], "TEST-HOST");
        assert!(v["device"]["cert_fp"].is_string());
        assert_eq!(v["load"]["source"], "live");
        assert!(v["load"]["cpu_pct"].as_f64().is_some());
        assert!(v["load"]["mem_pct"].as_f64().is_some());
        assert_eq!(v["sessions"].as_array().unwrap().len(), 1);
        assert_eq!(v["sessions"][0]["id"], "sess-a");
        assert_eq!(v["stage"]["layer_start"], 0);
        assert_eq!(v["stage"]["layer_end"], 27);

        // Peer detail: reported load + updated_at, no sessions, null stage.
        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/devices/peer-1")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_eq!(v["device"]["device_name"], "PEER-HOST");
        assert_eq!(v["load"]["source"], "reported");
        assert_eq!(v["load"]["cpu_pct"], 12.5);
        assert_eq!(v["load"]["mem_pct"], 33.25);
        assert!(v["load"]["updated_at"].is_string());
        assert_eq!(v["device"]["capabilities"]["kv_pages"], 8);
        assert!(v["sessions"].as_array().unwrap().is_empty());
        assert!(v["stage"].is_null());

        // Peer without load -> none.
        let res = app()
            .oneshot(post_json(
                "/v1/devices/heartbeat",
                serde_json::json!({"device_id": "peer-2"}),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/devices/peer-2")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        let v = body_json(res).await;
        assert_eq!(v["load"]["source"], "none");
        assert!(v["device"]["device_name"].is_null());

        // Unknown id -> 404 {ok:false}.
        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/devices/ghost")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        assert_eq!(body_json(res).await["ok"], false);

        cleanup(&path);
    }

    #[tokio::test]
    async fn usage_aggregates_tokens_per_device_plan_and_null_bandwidth() {
        let path = temp_db("usage");
        let store = Arc::new(Store::open(&path).expect("open store"));
        seed_self(&store, "dllm-dev-1");
        store.set_device_name("dllm-dev-1", Some("TEST-HOST")).unwrap();
        let state = new_state(Arc::new(MockEngine::new()), store);
        let app = || router(state.clone());

        // Peer heartbeat with friendly name + reported load.
        let res = app()
            .oneshot(post_json(
                "/v1/devices/heartbeat",
                serde_json::json!({
                    "device_id": "peer-1",
                    "device_name": "PEER-HOST",
                    "load": {"cpu_pct": 12.5, "mem_pct": 33.25},
                }),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);

        // Two sessions with tokens (3 token rows total) via the event log.
        state
            .store
            .append("sess-u1", "session_created", r#"{"model":"qwen3-0.6b-q4"}"#)
            .unwrap();
        state.store.append("sess-u1", "token", r#"{"pos":0,"text":"hi"}"#).unwrap();
        state.store.append("sess-u1", "token", r#"{"pos":1,"text":" yo"}"#).unwrap();
        state
            .store
            .append("sess-u2", "session_created", r#"{"model":"qwen3-0.6b-q4"}"#)
            .unwrap();
        state.store.append("sess-u2", "token", r#"{"pos":0,"text":"hey"}"#).unwrap();

        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/usage")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;

        // Totals rolled up from the event log.
        assert_eq!(v["tokens_out_total"], 3);
        assert_eq!(v["sessions_total"], 2);
        // Bandwidth is not measured anywhere yet: explicit null, never faked.
        assert!(v["bandwidth"].is_null());

        // Per-device: self first with all tokens/sessions + plan layers,
        // peer with honest zeros + null layers.
        let devs = v["per_device"].as_array().unwrap();
        assert_eq!(devs.len(), 2);
        assert_eq!(devs[0]["device_id"], "dllm-dev-1");
        assert_eq!(devs[0]["device_name"], "TEST-HOST");
        assert_eq!(devs[0]["role"], "coordinator");
        assert_eq!(devs[0]["active"], true);
        assert_eq!(devs[0]["status"], "paired");
        assert_eq!(devs[0]["tokens_out"], 3);
        assert_eq!(devs[0]["sessions"], 2);
        assert_eq!(devs[0]["load_source"], "live");
        assert!(devs[0]["cpu_pct"].as_f64().is_some());
        assert!(devs[0]["mem_pct"].as_f64().is_some());
        assert_eq!(devs[0]["layer_start"], 0);
        assert_eq!(devs[0]["layer_end"], 27);
        assert_eq!(devs[1]["device_id"], "peer-1");
        assert_eq!(devs[1]["device_name"], "PEER-HOST");
        assert_eq!(devs[1]["active"], true);
        assert_eq!(devs[1]["tokens_out"], 0);
        assert_eq!(devs[1]["sessions"], 0);
        assert_eq!(devs[1]["load_source"], "reported");
        assert_eq!(devs[1]["cpu_pct"], 12.5);
        assert_eq!(devs[1]["mem_pct"], 33.25);
        assert!(devs[1]["layer_start"].is_null());
        assert!(devs[1]["layer_end"].is_null());

        // Plan mirrors /v1/plan truthfully (single stage 0-27 on self).
        assert_eq!(v["plan"]["stages"].as_array().unwrap().len(), 1);
        assert_eq!(v["plan"]["stages"][0]["device_id"], "dllm-dev-1");
        assert_eq!(v["plan"]["stages"][0]["layer_start"], 0);
        assert_eq!(v["plan"]["stages"][0]["layer_end"], 27);

        cleanup(&path);
    }

    #[test]
    fn sse_shapes_follow_contract_never_wrapped() {
        let (e, d) = sse_shape_for_stored("token", r#"{"pos":3,"text":" hi","done":false}"#);
        assert_eq!(e, "token");
        let v: serde_json::Value = serde_json::from_str(&d).unwrap();
        assert_eq!(v["pos"], 3);
        assert_eq!(v["text"], " hi");
        assert!(v.get("done").is_none());
        assert!(v.get("kind").is_none());

        let (e, d) = sse_shape_for_stored("commit", r#"{"pos":3}"#);
        assert_eq!(e, "commit");
        assert_eq!(serde_json::from_str::<serde_json::Value>(&d).unwrap()["pos"], 3);

        let (e, d) = sse_shape_for_stored("session_created", r#"{"model":"m"}"#);
        assert_eq!(e, "status");
        assert_eq!(d, r#"{"model":"m"}"#);

        let (e, _) = sse_shape_for_live("user", r#"{"role":"user","text":"hi"}"#);
        assert_eq!(e, "status");
    }

    #[test]
    fn pairing_uri_format_matches_cli_contract() {        let uri = pairing_uri("192.168.1.10", 8080, "abc123");
        assert_eq!(
            uri,
            format!("dllm://pair?host=192.168.1.10&port=8080&quic=8443&fp=abc123&v={}", env!("CARGO_PKG_VERSION"))
        );
    }

    #[test]
    fn last_seen_window_parses_sqlite_ts() {
        // "now" in the exact SQLite default format is active.
        let now_ms = unix_ms_now();
        // Build a matching timestamp string without chrono (round to seconds).
        let secs = now_ms / 1000;
        let days = secs.div_euclid(86400);
        let rem = secs.rem_euclid(86400);
        // March-based civil inverse for test dates only (post-1970).
        let z = days + 719468;
        let era = z.div_euclid(146097);
        let doe = z - era * 146097;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let mut yy = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let dd = doy - (153 * mp + 2) / 5 + 1;
        let mm = if mp < 10 { mp + 3 } else { mp - 9 };
        yy += if mm <= 2 { 1 } else { 0 };
        let (y, m, d) = (yy, mm, dd);
        let ts = format!(
            "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.000Z",
            rem / 3600,
            (rem % 3600) / 60,
            rem % 60
        );
        assert!(last_seen_active(&ts, now_ms));
        assert!(!last_seen_active("not-a-timestamp", now_ms));
        assert!(!last_seen_active("2020-01-01T00:00:00.000Z", now_ms));
    }

    async fn get_json(
        app: &Router,
        uri: &str,
    ) -> (StatusCode, serde_json::Value) {
        let res = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri(uri)
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        let status = res.status();
        (status, body_json(res).await)
    }

    fn find_session<'a>(
        v: &'a serde_json::Value,
        id: &str,
    ) -> Option<&'a serde_json::Value> {
        v["sessions"]
            .as_array()?
            .iter()
            .find(|s| s["id"] == id)
    }

    #[tokio::test]
    async fn sessions_rename_roundtrip_default_fallback_and_validation() {
        let path = temp_db("rename");
        let store = Arc::new(Store::open(&path).expect("open store"));
        let state = new_state(Arc::new(MockEngine::new()), store);
        let app = || router(state.clone());

        // Create via the API.
        let res = app()
            .oneshot(post_json(
                "/v1/sessions",
                serde_json::json!({"model": "qwen3-0.6b-q4"}),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::CREATED);
        let id = body_json(res).await["id"].as_str().unwrap().to_string();

        // No user message yet -> "New chat".
        let (status, v) = get_json(&app(), "/v1/sessions").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(find_session(&v, &id).unwrap()["title"], "New chat");

        // First user message seeds the server-side default (40-char trunc).
        // Seed directly: avoids the MockEngine generation race.
        let long = "Hello world, this is my first chat message for testing titles!";
        state
            .store
            .append(&id, "user_message", &serde_json::json!({"role":"user","text":long}).to_string())
            .unwrap();
        let expected: String = long.chars().take(40).collect();
        let (_, v) = get_json(&app(), "/v1/sessions").await;
        assert_eq!(find_session(&v, &id).unwrap()["title"], expected);

        // Rename roundtrip (trimmed) -> list reflects it.
        let res = app()
            .oneshot(post_json(
                &format!("/v1/sessions/{id}/rename"),
                serde_json::json!({"title": "  My Trip  "}),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_eq!(v["ok"], true);
        assert_eq!(v["id"], id);
        assert_eq!(v["title"], "My Trip");
        let (_, v) = get_json(&app(), "/v1/sessions").await;
        assert_eq!(find_session(&v, &id).unwrap()["title"], "My Trip");

        // Latest rename wins.
        let res = app()
            .oneshot(post_json(
                &format!("/v1/sessions/{id}/rename"),
                serde_json::json!({"title": "Second"}),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        let (_, v) = get_json(&app(), "/v1/sessions").await;
        assert_eq!(find_session(&v, &id).unwrap()["title"], "Second");

        // Validation 400s: empty, whitespace-only, >80 chars, missing field.
        for bad in [
            serde_json::json!({"title": ""}),
            serde_json::json!({"title": "   "}),
            serde_json::json!({"title": "x".repeat(81)}),
            serde_json::json!({}),
        ] {
            let res = app()
                .oneshot(post_json(&format!("/v1/sessions/{id}/rename"), bad))
                .await
                .expect("oneshot");
            assert_eq!(res.status(), StatusCode::BAD_REQUEST);
            assert_eq!(body_json(res).await["ok"], false);
        }
        // 80 chars exactly is accepted.
        let res = app()
            .oneshot(post_json(
                &format!("/v1/sessions/{id}/rename"),
                serde_json::json!({"title": "y".repeat(80)}),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);

        // Unknown id -> 404.
        let res = app()
            .oneshot(post_json(
                "/v1/sessions/ghost/rename",
                serde_json::json!({"title": "Hi"}),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        assert_eq!(body_json(res).await["ok"], false);

        // Title survives a restart (reopen the same DB file).
        state
            .store
            .set_session_title(&id, "Restart-proof")
            .unwrap();
        state
            .store
            .append(&id, "session_renamed", r#"{"title":"Restart-proof"}"#)
            .unwrap();
        drop(state);
        let store2 = Arc::new(Store::open(&path).expect("reopen"));
        let state2 = new_state(Arc::new(MockEngine::new()), store2);
        let app2 = router(state2);
        let (_, v) = get_json(&app2, "/v1/sessions").await;
        assert_eq!(find_session(&v, &id).unwrap()["title"], "Restart-proof");

        cleanup(&path);
    }

    #[tokio::test]
    async fn sessions_delete_removes_from_list_and_events() {
        let path = temp_db("delete");
        let store = Arc::new(Store::open(&path).expect("open store"));
        let state = new_state(Arc::new(MockEngine::new()), store);
        let app = || router(state.clone());

        // Two sessions; first one renamed.
        let mk = || async {
            let res = app()
                .oneshot(post_json(
                    "/v1/sessions",
                    serde_json::json!({"model": "qwen3-0.6b-q4"}),
                ))
                .await
                .expect("oneshot");
            body_json(res).await["id"].as_str().unwrap().to_string()
        };
        let a = mk().await;
        let b = mk().await;
        let res = app()
            .oneshot(post_json(
                &format!("/v1/sessions/{a}/rename"),
                serde_json::json!({"title": "Doomed"}),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        // Resume position exists pre-delete.
        assert!(!state.store.replay_since(&a, 0).unwrap().is_empty());

        // Delete -> 200 {ok:true,id}.
        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .method("DELETE")
                    .uri(format!("/v1/sessions/{a}"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_eq!(v["ok"], true);
        assert_eq!(v["id"], a);

        // Gone from the list; the other session is untouched.
        let (_, v) = get_json(&app(), "/v1/sessions").await;
        assert!(find_session(&v, &a).is_none());
        let other = find_session(&v, &b).expect("sibling survives");
        assert_eq!(other["title"], "New chat");

        // SSE resume position is gone: store replay empty + HTTP events 404.
        assert!(state.store.replay_since(&a, 0).unwrap().is_empty());
        let (status, v) = get_json(&app(), &format!("/v1/sessions/{a}/events")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(v["ok"], false);

        // Unknown id -> 404 {ok:false}.
        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .method("DELETE")
                    .uri("/v1/sessions/ghost")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        assert_eq!(body_json(res).await["ok"], false);
        let (status, _) = get_json(&app(), "/v1/sessions/ghost/events").await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        cleanup(&path);
    }

    #[tokio::test]
    async fn networks_create_join_devices_roundtrip_web_and_android_shapes() {
        let path = temp_db("networks");
        let store = Arc::new(Store::open(&path).expect("open store"));
        seed_self(&store, "dllm-dev-1");
        let state = new_state(Arc::new(MockEngine::new()), store);
        let app = || router(state.clone());

        // List auto-seeds the default group (never stubbed-empty).
        let (status, v) = get_json(&app(), "/v1/networks").await;
        assert_eq!(status, StatusCode::OK);
        let nets = v["networks"].as_array().unwrap();
        assert!(nets.iter().any(|n| n["id"] == "default"));

        // Create: locked + open join. Bad names are 400, not 422.
        let res = app()
            .oneshot(post_json(
                "/v1/networks",
                serde_json::json!({"name": "Alpha", "password": "pw-1", "open_join": true}),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::CREATED);
        let v = body_json(res).await;
        let gid = v["id"].as_str().unwrap().to_string();
        assert!(v["qr"].as_str().unwrap().contains(&gid));
        for bad in [
            serde_json::json!({"name": ""}),
            serde_json::json!({"name": "   "}),
            serde_json::json!({"name": "x".repeat(41)}),
            serde_json::json!({}),
        ] {
            let res = app().oneshot(post_json("/v1/networks", bad)).await.expect("oneshot");
            assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        }

        // Join unknown group -> 404 (both shapes).
        for body in [
            serde_json::json!({"password": "pw-1"}),
            serde_json::json!({"device_id": "phone-1", "password": "pw-1"}),
        ] {
            let res = app()
                .oneshot(post_json("/v1/networks/ghost/join", body))
                .await
                .expect("oneshot");
            assert_eq!(res.status(), StatusCode::NOT_FOUND);
        }

        // Wrong password -> 401 (never leaks which half failed).
        let res = app()
            .oneshot(post_json(
                &format!("/v1/networks/{gid}/join"),
                serde_json::json!({"device_id": "phone-1", "password": "nope"}),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

        // Web shape (no device_id): password-checked only, 200, no member row.
        let res = app()
            .oneshot(post_json(
                &format!("/v1/networks/{gid}/join"),
                serde_json::json!({"password": "pw-1"}),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(body_json(res).await["ok"], true);
        // Empty body (web open-network join) also succeeds.
        let res = app()
            .oneshot(post_json(&format!("/v1/networks/{gid}/join"), serde_json::json!({})))
            .await
            .expect("oneshot");
        // Locked group + no password in the empty body -> still 401.
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);

        // Android shape: device_id + password (+ qr_secret accepted/ignored).
        let res = app()
            .oneshot(post_json(
                &format!("/v1/networks/{gid}/join"),
                serde_json::json!({
                    "device_id": "phone-1",
                    "device_name": "Pixel 8",
                    "password": "pw-1",
                    "qr_secret": "stale-secret-ignored",
                }),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_eq!(v["ok"], true);
        assert_eq!(v["role"], "member");

        // Header identity also works when the body carries no device_id.
        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(format!("/v1/networks/{gid}/join"))
                    .header("content-type", "application/json")
                    .header("x-device-id", "phone-2")
                    .body(axum::body::Body::from(
                        serde_json::json!({"password": "pw-1"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(body_json(res).await["device_id"], "phone-2");

        // Devices: admin first, both phones paired + active, names persisted.
        let (status, v) = get_json(&app(), &format!("/v1/networks/{gid}/devices")).await;
        assert_eq!(status, StatusCode::OK);
        let devs = v["devices"].as_array().unwrap();
        assert_eq!(devs[0]["device_id"], "dllm-dev-1");
        assert_eq!(devs[0]["role"], "admin");
        let p1 = devs.iter().find(|d| d["device_id"] == "phone-1").expect("phone-1");
        assert_eq!(p1["status"], "paired");
        assert_eq!(p1["active"], true);
        assert_eq!(p1["device_name"], "Pixel 8");
        assert!(devs.iter().any(|d| d["device_id"] == "phone-2"));
        let (status, _) = get_json(&app(), "/v1/networks/ghost/devices").await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        // Closed group: anonymous web check is refused, admin path still works.
        let res = app()
            .oneshot(post_json(
                "/v1/networks",
                serde_json::json!({"name": "Closed", "open_join": false}),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::CREATED);
        let closed = body_json(res).await["id"].as_str().unwrap().to_string();
        let res = app()
            .oneshot(post_json(&format!("/v1/networks/{closed}/join"), serde_json::json!({})))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
        let res = app()
            .oneshot(post_json(
                &format!("/v1/networks/{closed}/join"),
                serde_json::json!({"device_id": "outsider"}),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::FORBIDDEN);

        cleanup(&path);
    }

    #[tokio::test]
    async fn usage_group_admin_full_member_totals_only_stranger_forbidden() {
        let path = temp_db("usage-group");
        let store = Arc::new(Store::open(&path).expect("open store"));
        seed_self(&store, "dllm-dev-1");
        let state = new_state(Arc::new(MockEngine::new()), store);
        let app = || router(state.clone());

        let res = app()
            .oneshot(post_json("/v1/networks", serde_json::json!({"name": "G"})))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::CREATED);
        let gid = body_json(res).await["id"].as_str().unwrap().to_string();
        // phone-1 joins as a plain member (non-admin).
        let res = app()
            .oneshot(post_json(
                &format!("/v1/networks/{gid}/join"),
                serde_json::json!({"device_id": "phone-1"}),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        state.store.append("s1", "session_created", r#"{"model":"m"}"#).unwrap();
        state.store.append("s1", "token", r#"{"pos":0}"#).unwrap();

        // Admin (default coordinator caller): full breakdown.
        let (status, v) = get_json(&app(), &format!("/v1/usage?group={gid}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["tokens_out_total"], 1);
        assert_eq!(v["sessions_total"], 1);
        assert!(v["per_device"].as_array().unwrap().len() >= 2);
        assert!(v["plan"]["stages"].is_array());
        assert!(v["bandwidth"].is_null());

        // Non-admin member: totals only, empty per_device.
        let (status, v) =
            get_json(&app(), &format!("/v1/usage?group={gid}&device_id=phone-1")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["tokens_out_total"], 1);
        assert_eq!(v["per_device"].as_array().unwrap().len(), 0);

        // Stranger: 403 with honest totals in the body.
        let (status, v) =
            get_json(&app(), &format!("/v1/usage?group={gid}&device_id=ghost")).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert_eq!(v["ok"], false);
        assert_eq!(v["tokens_out_total"], 1);

        // Unknown group: 404.
        let (status, _) = get_json(&app(), "/v1/usage?group=ghost").await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        cleanup(&path);
    }

    #[tokio::test]
    async fn plan_group_scoped_and_session_stop_idempotent() {
        let path = temp_db("plan-stop");
        let store = Arc::new(Store::open(&path).expect("open store"));
        seed_self(&store, "dllm-dev-1");
        let state = new_state(Arc::new(MockEngine::new()), store);
        let app = || router(state.clone());

        let res = app()
            .oneshot(post_json("/v1/networks", serde_json::json!({"name": "G"})))
            .await
            .expect("oneshot");
        let gid = body_json(res).await["id"].as_str().unwrap().to_string();

        // Group plan: plan_layers shape (never hardcoded), unknown -> 404.
        let (status, v) = get_json(&app(), &format!("/v1/plan?group={gid}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["group"], gid);
        let stages = v["stages"].as_array().unwrap();
        assert!(!stages.is_empty());
        assert_eq!(stages[0]["layer_start"], 0);
        let (status, _) = get_json(&app(), "/v1/plan?group=ghost").await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        // Heartbeat worker flags persist and feed the plan/usage surface.
        let res = app()
            .oneshot(post_json(
                "/v1/devices/heartbeat",
                serde_json::json!({
                    "device_id": "phone-9",
                    "group_id": gid,
                    "worker_active": true,
                    "layers": {"layer_start": 0, "layer_end": 8},
                    "load": {"cpu_pct": 11.0, "mem_pct": 22.0},
                }),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        let row = state.store.get_device("phone-9").unwrap().expect("row");
        assert_eq!(row.worker_active, Some(1));
        assert!(row.layers.is_some_and(|s| s.contains("layer_start")));
        assert_eq!(row.cpu_pct, Some(11.0));

        // Stop: unknown session 404; idle session ok/cancelled:false (idempotent).
        let res = app()
            .oneshot(post_json("/v1/sessions/ghost/stop", serde_json::json!({})))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        let res = app()
            .oneshot(post_json("/v1/sessions", serde_json::json!({"model": "m"})))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::CREATED);
        let sid = body_json(res).await["id"].as_str().unwrap().to_string();
        for _ in 0..2 {
            let res = app()
                .oneshot(post_json(&format!("/v1/sessions/{sid}/stop"), serde_json::json!({})))
                .await
                .expect("oneshot");
            assert_eq!(res.status(), StatusCode::OK);
            let v = body_json(res).await;
            assert_eq!(v["ok"], true);
            assert_eq!(v["cancelled"], false);
        }
        // Empty-body stop (web shape) also works.
        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri(format!("/v1/sessions/{sid}/stop"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);

        cleanup(&path);
    }

    #[tokio::test]
    async fn devices_delete_roundtrip_unknown_self_and_reheartbeat() {
        let path = temp_db("devdelete");
        let store = Arc::new(Store::open(&path).expect("open store"));
        seed_self(&store, "dllm-dev-1");
        let state = new_state(Arc::new(MockEngine::new()), store);
        let app = || router(state.clone());

        // Seed a drill-junk worker via heartbeat.
        let res = app()
            .oneshot(post_json(
                "/v1/devices/heartbeat",
                serde_json::json!({"device_id": "drill-pixel-8"}),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);

        // Delete roundtrip -> 200 {ok:true,id} + gone from list + detail 404.
        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .method("DELETE")
                    .uri("/v1/devices/drill-pixel-8")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        let v = body_json(res).await;
        assert_eq!(v["ok"], true);
        assert_eq!(v["id"], "drill-pixel-8");
        let (_, v) = get_json(&app(), "/v1/devices").await;
        assert!(v["devices"].as_array().unwrap().iter().all(|d| d["device_id"] != "drill-pixel-8"));
        let (status, v) = get_json(&app(), "/v1/devices/drill-pixel-8").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(v["ok"], false);

        // Delete unknown -> 404 {ok:false}.
        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .method("DELETE")
                    .uri("/v1/devices/ghost")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        assert_eq!(body_json(res).await["ok"], false);

        // Delete self coordinator -> 400, row survives.
        let res = app()
            .oneshot(
                axum::http::Request::builder()
                    .method("DELETE")
                    .uri("/v1/devices/dllm-dev-1")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_json(res).await["ok"], false);
        let (status, _) = get_json(&app(), "/v1/devices/dllm-dev-1").await;
        assert_eq!(status, StatusCode::OK);

        // Delete-then-re-heartbeat recreates as paired + active.
        let res = app()
            .oneshot(post_json(
                "/v1/devices/heartbeat",
                serde_json::json!({"device_id": "drill-pixel-8"}),
            ))
            .await
            .expect("oneshot");
        assert_eq!(res.status(), StatusCode::OK);
        let (_, v) = get_json(&app(), "/v1/devices").await;
        let row = v["devices"].as_array().unwrap().iter().find(|d| d["device_id"] == "drill-pixel-8").expect("recreated");
        assert_eq!(row["status"], "paired");
        assert_eq!(row["active"], true);

        cleanup(&path);
    }
}
