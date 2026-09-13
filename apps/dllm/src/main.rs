//! `dllm` CLI: serve / list / pull / run / ps (Phase 1: real pull).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};

/// Baked catalog (same file served by `GET /api/models`).
/// Requires `contracts/catalog.json` at compile time (contracts mate owns it).
const CATALOG_JSON: &str = include_str!("../../../contracts/catalog.json");

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
    }
}

async fn run_serve(port: u16) -> anyhow::Result<()> {
    // Stable node identity: load-or-generate persistent Quinn Identity.
    let (_identity, node_id, fingerprint) = load_or_create_identity()?;
    tracing::info!(%node_id, %fingerprint, "stable node identity loaded");
    let node = dllm_serve::NodeInfo {
        node_id: node_id.clone(),
        fingerprint: fingerprint.clone(),
        quic_port: dllm_serve::QUIC_PORT,
        version: env!("CARGO_PKG_VERSION").to_string(),
    };
    let store = Arc::new(dllm_store::Store::open("dllm-events.db")?);
    // Prefer real inference when weights are present; the server must never
    // fail to start for lack of weights, so fall back to MockEngine.
    let engine: Arc<dyn dllm_core::Engine> = match qwen_q4_path() {
        Some(path) => match dllm_core::LlamaEngine::load(&path, 4096) {
            Ok(llama) => {
                tracing::info!(path = %path.display(), "serving with LlamaEngine");
                Arc::new(llama)
            }
            Err(e) => {
                tracing::warn!("{e:#}; falling back to MockEngine");
                Arc::new(dllm_core::MockEngine::new())
            }
        },
        None => {
            tracing::warn!("qwen3-0.6b-q4 not in catalog; serving with MockEngine");
            Arc::new(dllm_core::MockEngine::new())
        }
    };
    let state = dllm_serve::new_state_with_node(engine, store.clone(), node);
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

/// Best-effort local IPv4 for the pairing URI (outbound-route trick).
/// Falls back to `127.0.0.1` when offline or on error.
fn lan_ipv4() -> String {
    (|| -> anyhow::Result<String> {
        let sock = std::net::UdpSocket::bind("0.0.0.0:0")?;
        sock.connect("8.8.8.8:80")?;
        let ip = sock.local_addr()?.ip();
        match ip {
            std::net::IpAddr::V4(v4) if !v4.is_unspecified() => Ok(v4.to_string()),
            _ => anyhow::bail!("no ipv4 route"),
        }
    })()
    .unwrap_or_else(|_| "127.0.0.1".to_string())
}

fn pairing_uri(host: &str, port: u16, fingerprint: &str) -> String {
    format!(
        "dllm://pair?host={host}&port={port}&quic={}&fp={fingerprint}&v={}",
        dllm_serve::QUIC_PORT,
        env!("CARGO_PKG_VERSION"),
    )
}

fn cmd_id(port: u16) -> anyhow::Result<()> {
    let (_identity, node_id, fingerprint) = load_or_create_identity()?;
    let host = lan_ipv4();
    let uri = pairing_uri(&host, port, &fingerprint);
    println!("node_id: {node_id}");
    println!("fingerprint: {fingerprint}");
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
