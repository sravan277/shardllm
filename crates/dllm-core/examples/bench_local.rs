//! Phase 1 bench harness: tok/s baseline for Qwen3-0.6B Q4_K_M (CPU).
//!
//! Zero-manifest example (`cargo run -p dllm-core --example bench_local`).
//! Reuses `model_dir()` logic from `apps/dllm/src/main.rs`:
//! `%LOCALAPPDATA%\dllm\models\` (else `./models/`).
//!
//! Measures per prompt: TTFT (prefill) + decode tok/s over the
//! `generate_stream` `TokenEvent` channel.

use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use dllm_core::{Engine, LlamaEngine};

const N_CTX: u32 = 4096;
const MODEL_FILE: &str = "Qwen3-0.6B-Q4_K_M.gguf";
const EXPECTED_BYTES: u64 = 396_705_472;

/// Same as `apps/dllm/src/main.rs::model_dir`.
fn model_dir() -> PathBuf {
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        PathBuf::from(local).join("dllm").join("models")
    } else {
        PathBuf::from("models")
    }
}

fn model_path() -> PathBuf {
    model_dir().join(MODEL_FILE)
}

/// Minimal UNIX-days -> YYYY-MM-DD (Howard Hinnant days_from_civil inverse),
/// so the example needs no extra date dependency.
fn today_ymd() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    // civil_from_days
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path = model_path();
    println!("model: {}", path.display());
    match std::fs::metadata(&path) {
        Ok(md) => {
            println!("bytes: {} (expected {EXPECTED_BYTES})", md.len());
            if md.len() != EXPECTED_BYTES {
                eprintln!(
                    "WARN: size mismatch (got {}, want {EXPECTED_BYTES}); continuing anyway",
                    md.len()
                );
            }
        }
        Err(e) => {
            anyhow::bail!("model file not found {}: {e}", path.display());
        }
    }

    let par = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    println!("threads: available_parallelism={par} OMP_NUM_THREADS={:?} GGML_N_THREADS={:?}",
        std::env::var("OMP_NUM_THREADS").ok(),
        std::env::var("GGML_N_THREADS").ok());

    println!("loading LlamaEngine (n_ctx={N_CTX}) ...");
    let t_load = Instant::now();
    let engine = LlamaEngine::load(&path, N_CTX)?;
    println!("loaded in {:.1}s", t_load.elapsed().as_secs_f64());

    let prompts: Vec<(&str, &str)> = vec![
        (
            "short_chat",
            "Hello! Who are you? Reply in one short sentence.",
        ),
        (
            "para_completion",
            "Complete the following paragraph in about 100 words, continuing the same calm explanatory tone:\n\nThe small coastal town woke early every morning to the sound of gulls and fishing boats heading out past the harbor lighthouse. Salt hung in the air, and the bakery on Harbor Street had already lit its ovens, sending the smell of fresh bread down toward the docks where nets were being mended and ropes coiled for the day ahead.",
        ),
        (
            "no_think",
            "Explain in one or two sentences why the sky is blue.\n/no_think",
        ),
    ];

    let mut rows: Vec<serde_json::Value> = Vec::new();

    println!();
    println!(
        "{:<16} {:>6} {:>10} {:>11} {:>9}",
        "prompt_id", "tokens", "ttft_ms", "decode_tps", "total_s"
    );
    println!("{}", "-".repeat(60));

    for (id, prompt) in &prompts {
        let t0 = Instant::now();
        let mut rx = engine.generate_stream(prompt.to_string());
        let mut tokens: u32 = 0;
        let mut first_at: Option<std::time::Duration> = None;
        let mut last_at: Option<std::time::Duration> = None;
        let mut text_out = String::new();
        loop {
            match rx.recv().await {
                Some(ev) => {
                    let now = t0.elapsed();
                    if first_at.is_none() {
                        first_at = Some(now);
                    }
                    last_at = Some(now);
                    tokens += 1;
                    text_out.push_str(&ev.text);
                    if ev.done {
                        break;
                    }
                }
                None => break, // channel closed: treat as end of stream
            }
        }
        let ttft_ms = first_at.map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0);
        let total_s = last_at.map(|d| d.as_secs_f64()).unwrap_or(0.0);
        let decode_s = match (first_at, last_at) {
            (Some(f), Some(l)) => (l.saturating_sub(f)).as_secs_f64(),
            _ => 0.0,
        };
        let decode_tps = if tokens > 1 && decode_s > 1e-9 {
            (tokens - 1) as f64 / decode_s
        } else {
            0.0
        };
        println!(
            "{:<16} {:>6} {:>10.0} {:>11.1} {:>9.2}",
            id, tokens, ttft_ms, decode_tps, total_s
        );
        let preview: String = text_out.chars().take(120).collect();
        println!("  -> {}", preview.replace('\n', " "));

        // Warn loudly on pathological throughput per task spec (<1 tok/s).
        if decode_tps < 1.0 && tokens > 1 {
            eprintln!(
                "WARN [{id}]: decode {decode_tps:.2} tok/s < 1 tok/s — check thread count / build flags"
            );
        }

        rows.push(serde_json::json!({
            "prompt_id": id,
            "tokens": tokens,
            "ttft_ms": (ttft_ms * 10.0).round() / 10.0,
            "decode_tps": (decode_tps * 10.0).round() / 10.0,
            "total_s": (total_s * 100.0).round() / 100.0,
        }));
    }

    let mean_decode_tps = if rows.is_empty() {
        0.0
    } else {
        let sum: f64 = rows
            .iter()
            .filter_map(|r| r.get("decode_tps").and_then(|v| v.as_f64()))
            .sum();
        (sum / rows.len() as f64 * 10.0).round() / 10.0
    };
    println!("{}", "-".repeat(60));
    println!("mean decode: {mean_decode_tps:.1} tok/s");

    let out = serde_json::json!({
        "date": today_ymd(),
        "model": "Qwen3-0.6B",
        "quant": "Q4_K_M",
        "n_ctx": N_CTX,
        "machine": "Windows 10, CPU MinGW",
        "per_prompt": rows,
        "mean_decode_tps": mean_decode_tps,
    });

    // Resolve docs/bench-baseline.json from the crate manifest dir so the
    // example works regardless of the cargo invocation directory.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let doc_path = manifest_dir.join("../../docs/bench-baseline.json");
    std::fs::write(&doc_path, serde_json::to_string_pretty(&out)?)?;
    println!("wrote {}", doc_path.display());
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
