//! dllm-serve: Axum LAN API (REST + SSE) + static web fallback.
//!
//! Routes:
//! - `GET /api/health`
//! - `GET /api/models` (baked catalog JSON)
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
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use dllm_core::{Engine, MockEngine};
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

/// Shared app state.
pub struct AppState {
    pub engine: Arc<MockEngine>,
    pub store: Arc<Store>,
    pub tx: broadcast::Sender<SseMsg>,
}

/// Build shared state with a 256-slot broadcast channel.
pub fn new_state(engine: Arc<MockEngine>, store: Arc<Store>) -> Arc<AppState> {
    let (tx, _rx) = broadcast::channel(256);
    Arc::new(AppState { engine, store, tx })
}

/// Build the Axum router. Takes `Arc<AppState>` for `with_state`.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/models", get(models))
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

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true, "version": "0.1.0", "proto": "dllm1" }))
}

async fn models() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        CATALOG_JSON,
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

    // Spawn mock generation -> store + broadcast.
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
