/// `dllm` CLI: serve / list / pull / run / ps / rpc-worker (Phase 1: real pull).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};

/// Baked catalog (same file served by `GET /api/models`).
/// Requires `contracts/catalog.json` at compile time (contracts mate owns it).
const CATALOG_JSON: &str = include_str!("../../../contracts/catalog.json");

/// Context window for the served model. Matches `LlamaEngine`'s floor of 512 and
/// the catalog's 4K default.
const SERVE_CTX: u32 = 4096;

/// How often the supervisor re-derives the pipeline plan and, if the device
/// membership changed, reopens the distributed session.
///
/// A worker pairs *after* the coordinator has already loaded its model, so the
/// session cannot be opened once at startup and forgotten: something has to
/// notice that stage 1 now exists. 5 s is frequent enough that "I paired my
/// phone and nothing happened" is not a symptom, and rare enough that the
/// registry read (which is a blocking SQLite call) is not a busy loop.
const SUPERVISOR_INTERVAL_SECS: u64 = 5;

#[derive(Debug, Parser)]
#[command(name = "dllm", version, about = "Distributed LAN LLM coordinator CLI (Phase 0)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run the LAN API + SSE server and advertise over mDNS.
    Serve {
        /// HTTP port to bind (0.0.0.0).
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Print catalog models + installed state (Phase 0: catalog names only).
    List,
    /// Print the shard fetch plan for a model (real download lands Phase 1).
    Pull {
        /// Catalog model name, e.g. `qwen3-0.6b-q4`.
        model: String,
    },
    /// Auto-pull if missing, then serve + point at the web UI (Phase 0 note only).
    Run {
        /// Catalog model name.
        model: String,
    },
    /// Show active sessions/plans (Phase 0 placeholder).
    Ps,
    /// Print stable node_id + fingerprint + pairing URI (Android scan/type).
    Id {
        /// HTTP port baked into the pairing URI (must match `serve --port`).
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Host a ggml-rpc worker so another coordinator can offload layers here.
    ///
    /// Runs on the *worker* machine, not the coordinator: it exposes this
    /// device's ggml backends over the plaintext RPC protocol and blocks
    /// forever, because `dllm_shim_rpc_serve` has no stop path in the ABI
    /// (ADR-032). Stop it with Ctrl-C.
    RpcWorker {
        /// Interface to bind (the coordinator must be able to reach it).
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        /// TCP port for the ggml-rpc protocol.
        #[arg(long, default_value_t = dllm_shim::DEFAULT_RPC_PORT)]
        port: u16,
        /// Compute threads for the RPC backend.
        #[arg(long, default_value_t = 4)]
        threads: i32,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Serve { port } => run_serve(port).await,
        Commands::List => {
            cmd_list();
            Ok(())
        }
        Commands::Pull { model } => cmd_pull(&model).await,
        Commands::Run { model } => cmd_run(&model).await,
        Commands::Ps => {
            cmd_ps();
            Ok(())
        }
        Commands::Id { port } => {
            cmd_id(port)?;
            Ok(())
        }
        Commands::RpcWorker {
            host,
            port,
            threads,
        } => cmd_rpc_worker(&host, port, threads),
    }
}

/// `dllm rpc-worker`: expose this device's ggml backends to a coordinator.
///
/// Deliberately its own subcommand and its own process. The ABI has no way to
/// stop `dllm_shim_rpc_serve` (it blocks until the server dies), so calling it
/// from the coordinator would either wedge the HTTP server or force an
/// arbitrary `exit()` that skips the graceful shutdown. A worker being a separate
/// process is also what a real deployment looks like: the coordinator and the
/// worker are different machines.
///
/// The ggml-rpc protocol is plaintext TCP with **no authentication**, exactly as
/// upstream llama.cpp warns ("Never expose the RPC server to an open network!").
/// Bind to a LAN interface, never to a public one.
fn cmd_rpc_worker(host: &str, port: u16, threads: i32) -> anyhow::Result<()> {
    let lib = match dllm_shim::load_shim().lib() {
        Some(lib) => lib.clone(),
        None => anyhow::bail!(
            "cannot start a ggml-rpc worker without the shim: {}
\
             Build it with scripts/build-llama-win.ps1, or set DLLM_SHIM_DLL.",
            dllm_shim::load_shim().reason().unwrap_or("unknown reason")
        ),
    };
    tracing::info!(
        shim = %lib.path().display(),
        abi = lib.abi_version(),
        %host,
        port,
        threads,
        "serving ggml-rpc; WARNING: this protocol is plaintext TCP with no authentication, \
         so bind it to a trusted LAN only"
    );
    // Blocks forever. `n_devices = -1` exposes every accelerator and falls back
    // to the CPU device, which is what a CPU-only worker wants.
    dllm_core::ShimEngine::serve_rpc(lib.as_ref(), host, port, threads)
        .map_err(|e| anyhow::anyhow!("ggml-rpc worker stopped: {e}"))
}

async fn run_serve(port: u16) -> anyhow::Result<()> {
    // Stable node identity: load-or-generate persistent Quinn Identity.
    // It is NOT just advertised any more — the mesh server below binds it to
    // QUIC 8443 so paired peers can complete an mTLS handshake.
    let (identity, node_id, fingerprint) = load_or_create_identity()?;
    tracing::info!(%node_id, %fingerprint, "stable node identity loaded");
    let node = dllm_serve::NodeInfo {
        node_id: node_id.clone(),
        fingerprint: fingerprint.clone(),
        quic_port: dllm_serve::QUIC_PORT,
        version: env!("CARGO_PKG_VERSION").to_string(),
    };
    let store = Arc::new(dllm_store::Store::open("dllm-events.db")?);
    // Devices registry seed: the self row is real state (coordinator,
    // paired, just seen) so `GET /v1/devices` count >= 1 is never stubbed.
    // Friendly name = OS hostname (fallback: node_id, never invented).
    // Best-effort: a failed seed must not stop serving.
    {
        let _ = store.upsert_device(
            &node_id,
            "coordinator",
            r#"["infer","chat"]"#,
            &fingerprint,
            &node_id,
        );
        let friendly = dllm_serve::self_device_name(&node_id);
        let _ = store.set_device_name(&node_id, Some(&friendly));
        let _ = store.touch_last_seen(&node_id);
        let _ = store.set_status(&node_id, "paired");
    }
    // Bind the real mTLS QUIC mesh BEFORE axum::serve: the advertised
    // quic_port must be a live socket, and /v1/mesh must report real links.
    // A bind failure is logged and swallowed on purpose — single-device chat
    // never needs the mesh, so it must not stop the HTTP server booting.
    let mesh = dllm_serve::mesh::MeshState::new();
    let _mesh_handle = dllm_serve::mesh::spawn_mesh_server(
        identity,
        dllm_serve::QUIC_PORT,
        store.clone(),
        mesh.clone(),
    );
    let bound = mesh.bound_port();
    if bound == 0 {
        tracing::error!(
            port = dllm_serve::QUIC_PORT,
            "QUIC mesh not listening (bind failed); serving HTTP only"
        );
    } else {
        tracing::info!(
            port = bound,
            allowed_peers = mesh.allowed_peer_count(),
            "QUIC mesh listening on 0.0.0.0:{bound} (paired peers only)"
        );
    }

    // Engine selection, in order of capability, with every downgrade logged and
    // reported on `GET /api/stats`:
    //
    //   1. `ShimEngine`  — real distributed inference over the C++ shim. This is
    //      the only engine that can put layers on another device.
    //   2. `LlamaEngine` — real CPU inference via llama-cpp-2, single-device.
    //   3. `MockEngine`  — canned tokens.
    //
    // The shim is loaded at *runtime* (ADR-032), so on a machine that has never
    // run `scripts/build-llama-win.ps1` step 1 is a plain `Result::Err` and the
    // server still boots. That is a hard product requirement, not a nicety: the
    // HTTP API must never depend on a compiled native artifact.
    let (engine, engine_note, supervisor_source): (
        Arc<dyn dllm_core::Engine>,
        Option<String>,
        Option<(Arc<dllm_core::ShimEngine>, SupervisorSource)>,
    ) = match qwen_q4_path() {
        Some(path) => match try_shim_engine(&path, &node_id, &store, &mesh) {
            Ok((shim_engine, notes)) => {
                tracing::info!(
                    path = %path.display(),
                    stages = notes.len(),
                    "serving with ShimEngine (llama.cpp + ggml-rpc; layers can span devices)"
                );
                for note in &notes {
                    tracing::info!(%note, "pipeline stage note");
                }
                let source = SupervisorSource {
                    model_path: path.clone(),
                    model_name: "qwen3-0.6b-q4".to_string(),
                    node_id: node_id.clone(),
                };
                (
                    Arc::new(shim_engine.clone()),
                    None,
                    Some((Arc::new(shim_engine), source)),
                )
            }
            Err(e) => {
                // Honest, specific, and visible in the API — a coordinator
                // quietly running single-device while the user believes their
                // phone has a stage is the failure this log line exists to stop.
                let note = format!("{e:#}");
                tracing::warn!(
                    "distributed inference unavailable ({note}); falling back to LlamaEngine \
                     (all layers stay on this machine)"
                );
                match dllm_core::LlamaEngine::load(&path, SERVE_CTX) {
                    Ok(llama) => {
                        tracing::info!(path = %path.display(), "serving with LlamaEngine");
                        (Arc::new(llama), Some(note), None)
                    }
                    Err(e2) => {
                        tracing::warn!("{e2:#}; falling back to MockEngine");
                        (
                            Arc::new(dllm_core::MockEngine::new()),
                            Some(format!("{note}; LlamaEngine also failed: {e2:#}")),
                            None,
                        )
                    }
                }
            }
        },
        None => {
            tracing::warn!("qwen3-0.6b-q4 not in catalog; serving with MockEngine");
            (
                Arc::new(dllm_core::MockEngine::new()),
                Some("qwen3-0.6b-q4 weights are not installed".to_string()),
                None,
            )
        }
    };

    // Share the same MeshState with the API so GET /v1/mesh + /api/stats see it.
    let state = dllm_serve::new_state_with_mesh(engine, store.clone(), node, port, mesh.clone());
    // Make the fallback reason readable over HTTP, not just in the log. A
    // coordinator that is secretly single-device must be able to say so.
    state.set_engine_note(engine_note);

    // Keep the distributed session in step with the membership: a worker pairs
    // *after* startup, so the session opened above is single-device until the
    // supervisor notices and reopens it with the new stage.
    if let Some((shim, source)) = supervisor_source {
        spawn_session_supervisor(shim, store.clone(), mesh.clone(), source);
    }
    dllm_serve::spawn_maintenance(store, dllm_serve::DEFAULT_EVENT_TTL_SECS);
    let app = dllm_serve::router(state);

    // mDNS advertise `_dllm._tcp.local.` (discovery only; no inference here).
    let daemon = mdns_sd::ServiceDaemon::new()?;
    let instance = format!("dllm-node-{port}");
    let quic_port = dllm_serve::QUIC_PORT.to_string();
    let version = env!("CARGO_PKG_VERSION");
    let txt: &[(&str, &str)] = &[
        ("quic_port", &quic_port),
        ("node_id", &node_id),
        ("ver", version),
        ("fp", &fingerprint),
    ];
    let svc = mdns_sd::ServiceInfo::new(
        "_dllm._tcp.local.",
        &instance,
        "dllm.local.",
        "",
        port,
        txt,
    )?;
    daemon.register(svc)?;
    tracing::info!("dllm serving on 0.0.0.0:{port} (mDNS _dllm._tcp.local.)");

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

/// Everything the supervisor needs to re-derive a [`SessionRequest`] on a tick.
///
/// Deliberately small and owned (no `Arc<Store>`): the supervisor re-reads the
/// registry from scratch each tick so it cannot observe a stale cached plan, and
/// the registry is a local SQLite file.
#[derive(Debug, Clone)]
struct SupervisorSource {
    model_path: PathBuf,
    model_name: String,
    node_id: String,
}

/// Build a [`dllm_core::ShimEngine`] for the current membership.
///
/// Returns `(engine, stage_notes)`. `Err` is the "why are we single-device"
/// answer that becomes both the log line and `GET /api/stats` -> `engine_note`.
///
/// Every failure mode here is non-fatal to the process by construction:
/// - no DLL / wrong ABI -> [`dllm_shim::LoadError`];
/// - weights missing -> `session_request` still returns a single-stage request,
///   and `ShimSession::open` fails with an actionable message;
/// - a dead worker -> dropped as a stage, reported in `stage_notes`.
fn try_shim_engine(
    model_path: &PathBuf,
    node_id: &str,
    store: &dllm_store::Store,
    mesh: &dllm_serve::mesh::MeshState,
) -> Result<(dllm_core::ShimEngine, Vec<String>), anyhow::Error> {
    // `load_shim` memoises per process: a missing DLL is probed once, so the
    // warning appears once instead of once per request.
    let lib = match dllm_shim::load_shim().lib() {
        Some(lib) => lib.clone(),
        None => anyhow::bail!(
            "the native shim is unavailable: {}",
            dllm_shim::load_shim().reason().unwrap_or("unknown reason")
        ),
    };
    let (request, notes) = dllm_serve::session_request(
        model_path.clone(),
        "qwen3-0.6b-q4",
        SERVE_CTX,
        node_id,
        store,
        mesh,
    );
    let Some(request) = request else {
        anyhow::bail!("no pipeline session request could be built")
    };

    let devices = request.devices.len();
    let engine = dllm_core::ShimEngine::open(lib, request)?;
    if devices <= 1 {
        // Not a failure: it is the ADR-004 single-device fast path, and it is
        // reached *through* the same code that will later span devices.
        tracing::info!(
            devices,
            "shim session opened with a single stage; layers will be placed again if a paired \
             worker advertises an RPC endpoint"
        );
    }
    Ok((engine, notes))
}

/// Spawn the session supervisor.
///
/// The `Engine` trait deliberately has no downcast hook (it is a two-method
/// trait), so the supervisor is handed the concrete `Arc<ShimEngine>` alongside
/// the erased `Arc<dyn Engine>` the API uses. `ShimEngine` is one `Arc` inside,
/// so keeping both costs nothing and needs no `Any` cast.
fn spawn_session_supervisor(
    engine: Arc<dllm_core::ShimEngine>,
    store: Arc<dllm_store::Store>,
    mesh: Arc<dllm_serve::mesh::MeshState>,
    source: SupervisorSource,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker =
            tokio::time::interval(Duration::from_secs(SUPERVISOR_INTERVAL_SECS));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            // The registry read is blocking SQLite; keep it off the reactor.
            let (store, mesh, source) = (store.clone(), mesh.clone(), source.clone());
            let derived = tokio::task::spawn_blocking(move || {
                dllm_serve::session_request(
                    source.model_path,
                    &source.model_name,
                    SERVE_CTX,
                    &source.node_id,
                    &store,
                    &mesh,
                )
            })
            .await;
            let (request, notes) = match derived {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::warn!("pipeline supervisor task failed: {e}");
                    continue;
                }
            };
            for note in &notes {
                tracing::debug!(%note, "pipeline supervisor note");
            }
            let Some(request) = request else {
                continue;
            };
            // Idempotent: `ensure_session` reloads the 400 MB model only when the
            // derived `tensor_split` or the device order actually changed.
            let engine = engine.clone();
            let reopened = tokio::task::spawn_blocking(move || engine.ensure_session(&request))
                .await;
            match reopened {
                Ok(Ok(true)) => tracing::info!("pipeline membership changed; session reopened"),
                Ok(Ok(false)) => {}
                Ok(Err(e)) => tracing::warn!("could not reopen the distributed session: {e:#}"),
                Err(e) => tracing::warn!("session reopen task failed: {e}"),
            }
        }
    })
}

fn parse_catalog() -> serde_json::Value {
    serde_json::from_str(CATALOG_JSON).unwrap_or(serde_json::Value::Null)
}

fn find_model<'a>(v: &'a serde_json::Value, model: &str) -> Option<&'a serde_json::Value> {
    v.get("models").and_then(|m| m.as_array()).and_then(|models| {
        models.iter().find(|m| {
            m.get("name").and_then(|s| s.as_str()) == Some(model)
                || m.get("id").and_then(|s| s.as_str()) == Some(model)
        })
    })
}

/// Catalog path of the Q4 weights (`%LOCALAPPDATA%\dllm\models\...`), if known.
fn qwen_q4_path() -> Option<PathBuf> {
    let v = parse_catalog();
    let m = find_model(&v, "qwen3-0.6b-q4")?;
    let (path, _, _, _) = model_file(m)?;
    Some(path)
}

/// Model weights live outside the repo: `%LOCALAPPDATA%\dllm\models\` (else `./models/`).
fn model_dir() -> PathBuf {
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        PathBuf::from(local).join("dllm").join("models")
    } else {
        PathBuf::from("models")
    }
}

/// Stable node state lives next to the weights: `%LOCALAPPDATA%\dllm\`
/// (else the process working directory), mirroring [`model_dir`].
fn node_dir() -> PathBuf {
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        PathBuf::from(local).join("dllm")
    } else {
        PathBuf::from(".")
    }
}

/// Load-or-generate the persistent Quinn [`dllm_net::transport::Identity`].
///
/// DER bytes live at `node-identity.crt` / `node-identity.key` under
/// [`node_dir`]; `node-id.txt` next to them holds the stable `node_id`
/// (derived from the fingerprint on first run). Returns
/// `(identity, node_id, fingerprint)`.
fn load_or_create_identity(
) -> anyhow::Result<(dllm_net::transport::Identity, String, String)> {
    use dllm_net::transport::Identity;

    let dir = node_dir();
    std::fs::create_dir_all(&dir)?;
    let cert_path = dir.join("node-identity.crt");
    let key_path = dir.join("node-identity.key");
    let id_path = dir.join("node-id.txt");

    let identity = match (std::fs::read(&cert_path), std::fs::read(&key_path)) {
        (Ok(cert_der), Ok(key_der)) if !cert_der.is_empty() && !key_der.is_empty() => {
            Identity::from_der(cert_der, key_der)
        }
        _ => {
            let fresh = Identity::generate().map_err(|e| anyhow::anyhow!("{e:#}"))?;
            std::fs::write(&cert_path, &fresh.cert_der)?;
            std::fs::write(&key_path, &fresh.key_der)?;
            fresh
        }
    };
    let fingerprint = identity.fingerprint();

    let node_id = match std::fs::read_to_string(&id_path) {
        Ok(s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => {
            let derived = format!("dllm-{}", &fingerprint[..12.min(fingerprint.len())]);
            // Best-effort persist; a missing file just re-derives the same id.
            let _ = std::fs::write(&id_path, format!("{derived}\n"));
            derived
        }
    };
    Ok((identity, node_id, fingerprint))
}

/// Pairing-URI helper. Single source of truth lives in `dllm-serve`
/// (also serves `GET /api/pairing-uri`).
fn pairing_uri(host: &str, port: u16, fingerprint: &str) -> String {
    dllm_serve::pairing_uri(host, port, fingerprint)
}

fn cmd_id(port: u16) -> anyhow::Result<()> {
    let (_identity, node_id, fingerprint) = load_or_create_identity()?;
    let candidates = dllm_serve::lan_candidates();
    let host = candidates
        .first()
        .cloned()
        .unwrap_or_else(dllm_serve::lan_ipv4);
    let uri = pairing_uri(&host, port, &fingerprint);
    println!("node_id: {node_id}");
    println!("fingerprint: {fingerprint}");
    println!("candidates: {}", candidates.join(", "));
    println!("{uri}");
    Ok(())
}

fn model_file(m: &serde_json::Value) -> Option<(PathBuf, String, u64, String)> {
    let src = m.get("source")?;
    let file = src.get("file")?.as_str()?.to_string();
    let url = src.get("url")?.as_str()?.to_string();
    let bytes = src.get("bytes")?.as_u64()?;
    let sha = src
        .get("sha256")
        .and_then(|s| s.as_str())
        .unwrap_or("TODO-SHA256")
        .to_string();
    Some((model_dir().join(&file), url, bytes, sha))
}

fn installed_state(m: &serde_json::Value) -> &'static str {
    match model_file(m) {
        Some((path, _, expected, _)) => match std::fs::metadata(&path) {
            Ok(md) if md.len() == expected => "installed",
            Ok(_) => "partial",
            Err(_) => "catalog",
        },
        None => "catalog",
    }
}

fn cmd_list() {
    let v = parse_catalog();
    match v.get("models").and_then(|m| m.as_array()) {
        Some(models) => {
            for m in models {
                let name = m
                    .get("name")
                    .or_else(|| m.get("id"))
                    .and_then(|s| s.as_str())
                    .unwrap_or("?");
                println!("{name}\t{}", installed_state(m));
            }
        }
        None => {
            // Unknown schema: dump so the contracts change is visible.
            println!("{v:#}");
        }
    }
}

/// Real resumable download of the model's source GGUF + size/sha verify.
/// (Shard `files[]` stay byte-ranges of this same URL until the Phase 3 splitter.)
async fn cmd_pull(model: &str) -> anyhow::Result<()> {
    let v = parse_catalog();
    let m = find_model(&v, model);
    let m = match m {
        Some(m) => m,
        None => {
            println!("model '{model}' not in catalog; try `dllm list`.");
            return Ok(());
        }
    };
    let (dest, url, expected, sha) = model_file(m)
        .ok_or_else(|| anyhow::anyhow!("catalog entry '{model}' lacks source {{file,url,bytes}}"))?;
    std::fs::create_dir_all(model_dir())?;

    if let Ok(md) = std::fs::metadata(&dest) {
        if md.len() == expected {
            println!("{} already present ({} bytes, verified).", dest.display(), expected);
            return Ok(());
        }
        println!(
            "existing {} has {} bytes (want {expected}); re-downloading.",
            dest.display(),
            md.len()
        );
    }

    // Resume via `.part` sidecar; server (HuggingFace) supports Range.
    let part = dest.with_extension("part");
    let have = std::fs::metadata(&part).map(|md| md.len()).unwrap_or(0);
    if have >= expected && expected > 0 {
        std::fs::remove_file(&part)?;
    }
    let have = std::fs::metadata(&part).map(|md| md.len()).unwrap_or(0);

    let client = reqwest::Client::builder().user_agent("dllm/0.1.0").build()?;
    let mut req = client.get(&url);
    if have > 0 {
        req = req.header("Range", format!("bytes={have}-"));
    }
    let resp = req.send().await?.error_for_status()?;
    let resumed = resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    if have > 0 && !resumed {
        // Server ignored Range: restart from scratch.
        std::fs::remove_file(&part)?;
    }
    let total = resp.content_length().unwrap_or(expected.saturating_sub(have)) + if resumed { have } else { 0 };

    println!(
        "pull {model}: {} -> {} ({} total bytes{})",
        url,
        dest.display(),
        total,
        if resumed { ", resuming" } else { "" }
    );
    let mut out = tokio::fs::OpenOptions::new()
        .create(true)
        .append(resumed)
        .write(true)
        .truncate(!resumed)
        .open(&part)
        .await?;
    // Seed the hasher with existing prefix when resuming (re-read part file).
    let mut hasher = Sha256::new();
    if resumed {
        use tokio::io::AsyncReadExt;
        let mut f = tokio::fs::File::open(&part).await?;
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = f.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
    }

    use tokio::io::AsyncWriteExt;
    let mut done = if resumed { have } else { 0 };
    let t0 = Instant::now();
    let mut last_print = Instant::now();
    let mut resp = resp;
    while let Some(chunk) = resp.chunk().await? {
        out.write_all(&chunk).await?;
        hasher.update(&chunk);
        done += chunk.len() as u64;
        if last_print.elapsed().as_secs() >= 2 {
            let pct = if total > 0 { done as f64 * 100.0 / total as f64 } else { 0.0 };
            let mb_s = done as f64 / t0.elapsed().as_secs_f64() / 1e6;
            println!("  {done}/{total} bytes ({pct:.1}%) @ {mb_s:.1} MB/s");
            last_print = Instant::now();
        }
    }
    out.flush().await?;
    drop(out);

    let digest = format!("{:x}", hasher.finalize());
    if done != expected {
        anyhow::bail!("size mismatch: got {done} bytes, catalog wants {expected}");
    }
    if !sha.starts_with("TODO") && !sha.eq_ignore_ascii_case(&digest) {
        anyhow::bail!("sha256 mismatch: got {digest}, catalog wants {sha}");
    }
    tokio::fs::rename(&part, &dest).await?;
    // Audit trail: record the observed hash next to the weights.
    std::fs::write(dest.with_extension("sha256"), format!("{digest}\n"))?;
    println!("pulled {model}: {} bytes, sha256 {digest}", expected);
    if sha.starts_with("TODO") {
        println!("note: catalog sha256 is still TODO — paste this hash into contracts/catalog.json.");
    }
    Ok(())
}

async fn cmd_run(model: &str) -> anyhow::Result<()> {
    // Auto-pull if missing, then serve (engine swap is the next Phase 1 step).
    let v = parse_catalog();
    let need_pull = match find_model(&v, model) {
        Some(m) => !matches!(installed_state(m), "installed"),
        None => {
            println!("model '{model}' not in catalog; try `dllm list`.");
            return Ok(());
        }
    };
    if need_pull {
        cmd_pull(model).await?;
    }
    println!("run {model}: weights ready; serving (open http://127.0.0.1:8080/).");
    run_serve(8080).await
}

fn cmd_ps() {
    println!("no active sessions in Phase 0");
}
