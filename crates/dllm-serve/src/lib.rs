//! dllm-serve: Axum LAN API (REST + SSE) + static web fallback.
//!
//! Routes:
//! - `GET /api/health`
//! - `GET /api/stats` (uptime, engine kind, session/event counts, node_id)
//! - `GET /api/models` (baked catalog JSON)
//! - `GET /api/node` (stable node_id + fingerprint + quic_port, pairing bootstrap)
//! - `POST /v1/sessions`
//! - `POST /v1/sessions/{id}/messages` (spawns mock generation)
//! - `GET /v1/sessions/{id}/events` (SSE, keep-alive 15 s)
//!
//! NOTE: never put `CompressionLayer` / `BufferLayer` in front of the SSE
//! route — it breaks streaming. `Store` calls below are tiny Phase 0 writes
//! done inline; production paths MUST wrap them in `spawn_blocking`
//! (see `dllm-store` docs). Phase 0 serves mock tokens only.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use dllm_core::Engine;
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

/// Build shared state with a 256-slot broadcast channel.
pub fn new_state(engine: Arc<dyn Engine>, store: Arc<Store>) -> Arc<AppState> {
    new_state_with_node(engine, store, NodeInfo::default())
}

/// Build shared state with explicit node identity (preferred by `dllm serve`).
///
/// Note: event-log retention (`spawn_maintenance`) is opt-in — call it once
/// from the server's startup path while a Tokio runtime is active.
pub fn new_state_with_node(
    engine: Arc<dyn Engine>,
    store: Arc<Store>,
    node: NodeInfo,
) -> Arc<AppState> {
    let (tx, _rx) = broadcast::channel(256);
    let engine_kind = EngineKind::from_engine(&engine);
    Arc::new(AppState {
        engine,
        store,
        tx,
        node,
        engine_kind,
        started: Instant::now(),
    })
}

/// Build the Axum router. Takes `Arc<AppState>` for `with_state`.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/stats", get(stats))
        .route("/api/models", get(models))
        .route("/api/node", get(node_info))
        .route("/v1/sessions", post(create_session))
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

async fn health() -> Json<serde_json::Value> {
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
        event: "session".to_string(),
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
        event: "user".to_string(),
        data: user_payload,
    });

    // Spawn generation (real or mock, per the engine) -> store + broadcast.
    let mut rx = state.engine.generate_stream(prompt);
    let state2 = state.clone();
    let session_id = id.clone();
    tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
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
            let _ = state2.tx.send(SseMsg {
                id: rowid,
                session_id: session_id.clone(),
                event: "token".to_string(),
                data: payload,
            });
            if done {
                break;
            }
        }
    });

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({ "accepted": true, "session": id })),
    )
}

/// SSE: replay missed events from `Last-Event-ID`, then stream live.
///
/// No compression on this route (see [`router`]).
async fn session_events(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let last_id: i64 = headers
        .get("last-event-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    // Phase 0 inline read; production: spawn_blocking.
    let missed = state.store.replay_since(&id, last_id).unwrap_or_default();

    let session = id.clone();
    let rx = state.tx.subscribe();
    let stream = async_stream::stream! {
        for ev in missed {
            let data = serde_json::json!({ "kind": ev.kind, "payload": ev.payload }).to_string();
            let sse = SseEvent::default()
                .event("token")
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
                    let sse = SseEvent::default()
                        .event(msg.event.clone())
                        .id(msg.id.to_string())
                        .data(msg.data.clone());
                    yield Ok(sse);
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive"),
    )
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
}
