//! `dllm` CLI: serve / list / pull / run / ps (Phase 0 mocks).

use std::sync::Arc;

use clap::{Parser, Subcommand};

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
        Commands::Pull { model } => {
            cmd_pull(&model);
            Ok(())
        }
        Commands::Run { model } => {
            cmd_run(&model);
            Ok(())
        }
        Commands::Ps => {
            cmd_ps();
            Ok(())
        }
    }
}

async fn run_serve(port: u16) -> anyhow::Result<()> {
    let store = Arc::new(dllm_store::Store::open("dllm-events.db")?);
    let engine = Arc::new(dllm_core::MockEngine::new());
    let state = dllm_serve::new_state(engine, store);
    let app = dllm_serve::router(state);

    // mDNS advertise `_dllm._tcp.local.` (discovery only; no inference here).
    let daemon = mdns_sd::ServiceDaemon::new()?;
    let instance = format!("dllm-node-{port}");
    let txt: &[(&str, &str)] = &[("quic_port", "0"), ("node_id", "dllm-dev-1"), ("ver", "0.1.0")];
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
                let state = m
                    .get("state")
                    .and_then(|s| s.as_str())
                    .unwrap_or("catalog");
                println!("{name}\t{state}");
            }
        }
        None => {
            // Unknown schema: dump so the contracts change is visible.
            println!("{v:#}");
        }
    }
}

fn cmd_pull(model: &str) {
    let v = parse_catalog();
    let found = v
        .get("models")
        .and_then(|m| m.as_array())
        .and_then(|models| {
            models.iter().find(|m| {
                m.get("name").and_then(|s| s.as_str()) == Some(model)
                    || m.get("id").and_then(|s| s.as_str()) == Some(model)
            })
        });
    match found {
        Some(m) => {
            println!("fetch plan for {model} (Phase 0: plan only, no download yet):");
            // Real catalog shape: `files[]` with {role, bytes, layer_start, layer_end, url}.
            if let Some(files) = m.get("files").or_else(|| m.get("shards")).and_then(|f| f.as_array()) {
                let total: u64 = files
                    .iter()
                    .filter_map(|f| f.get("bytes").and_then(|b| b.as_u64()))
                    .sum();
                for f in files {
                    let role = f.get("role").and_then(|s| s.as_str()).unwrap_or("?");
                    let bytes = f.get("bytes").and_then(|b| b.as_u64()).unwrap_or(0);
                    match (f.get("layer_start").and_then(|n| n.as_u64()), f.get("layer_end").and_then(|n| n.as_u64())) {
                        (Some(s), Some(e)) => println!("  {role} layers {s}-{e}: {bytes} bytes"),
                        _ => println!("  {role}: {bytes} bytes"),
                    }
                }
                println!("total bytes: {total}");
            }
            if let Some(ranges) = m.get("ranges").or_else(|| {
                m.get("default_plan").or_else(|| m.get("layers"))
            }) {
                println!("ranges: {ranges:#}");
            }
            // Always show the raw entry so shard hashes/boundaries are auditable.
            println!("{m:#}");
            println!("note: real resumable download + hash-verify lands in Phase 1.");
        }
        None => {
            println!("model '{model}' not in catalog; try `dllm list`.");
        }
    }
}

fn cmd_run(model: &str) {
    println!("run {model}: auto-pull if missing (Phase 1), then serve + open web UI.");
    println!("Phase 0: start `dllm serve --port 8080`, then open http://127.0.0.1:8080/");
}

fn cmd_ps() {
    println!("no active sessions in Phase 0");
}
