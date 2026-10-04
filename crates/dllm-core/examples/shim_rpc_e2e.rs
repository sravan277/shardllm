//! End-to-end proof that **Rust** can put transformer layers on a second device
//! through `dllm-shim`.
//!
//! `native/tests/rpc_local.cpp` proves the same thing from C++. This is the
//! Rust-path equivalent, and it exists because "the shim works" is not the claim
//! — "the product can drive the shim" is. Nothing in `dllm serve` would have
//! proven that, since until now nothing called the shim at all.
//!
//! # What it does
//!
//! 1. Re-executes **itself** as a child process running `dllm_shim_rpc_serve`.
//!    A child process rather than a thread because the ABI has no stop path:
//!    `dllm_shim_rpc_serve` blocks until the server dies and nothing in the ABI
//!    can ask it to. (The C++ test has the same problem and calls
//!    `worker.detach()`; a child process is strictly tidier because the exit
//!    tears the socket down.) Changing the ABI is not an option here — see ADR-032.
//! 2. Waits for the worker to accept a TCP connection.
//! 3. Registers it with `dllm_shim_add_rpc_server("127.0.0.1:<port>")`.
//! 4. Computes a real 2-device [`plan_layers`] plan and derives `tensor_split`
//!    from it via [`tensor_split_for_plan`] — no hand-written `{1, 1}`.
//! 5. Generates and prints `dllm_shim_session_report`'s `layer_owner`.
//!
//! # The assertion
//!
//! `layer_owner` must name **more than one** ggml device. That is measured
//! reality read back out of `llama_model::dev_layer()`, not a restatement of the
//! requested split, so it cannot pass by accident.
//!
//! # Running it
//!
//! ```powershell
//! . .\scripts\build-env.ps1
//! $env:DLLM_TEST_MODEL = "$env:LOCALAPPDATA\dllm\models\Qwen3-0.6B-Q4_K_M.gguf"
//! cargo run -p dllm-core --example shim_rpc_e2e
//! ```
//!
//! `DLLM_SHIM_DLL` overrides DLL discovery; otherwise `native/build-win/` is
//! probed. Exit code 0 = pass, 2 = setup failed (no DLL / no model / worker did
//! not come up), 1 = the distribution assertion failed.

use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dllm_core::plan::{DeviceSpec, TOTAL_LAYERS, plan_layers};
use dllm_core::shim_engine::{SessionRequest, ShimEngine, tensor_split_for_plan};
use dllm_core::{Engine, PlannedDevice};
use dllm_shim::{RpcEndpoint, ShimLib};

/// Env var that turns this same binary into the RPC worker. An env var (rather
/// than a CLI flag) so the child invocation is a bare `current_exe()` with
/// `--worker <host> <port>`, which cannot be confused with the coordinator path.
const ROLE_ENV: &str = "DLLM_SHIM_E2E_ROLE";

/// Number of ggml-rpc server threads. Small on purpose: this drill contends for
/// the same cores as the coordinator.
const WORKER_THREADS: i32 = 4;

fn main() -> std::process::ExitCode {
    // The engine logs through `tracing`; without a subscriber every warn is
    // silently dropped, which is exactly how a failed generation would look like
    // "no tokens, no reason". `RUST_LOG=info` for the full picture.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let args: Vec<String> = std::env::args().collect();
    if std::env::var(ROLE_ENV).as_deref() == Ok("worker") {
        return run_worker(&args);
    }
    // The coordinator only awaits an mpsc receiver, so a current-thread runtime is
    // enough — and it keeps the decode thread (`shim-decode`) as the only other
    // thread in the picture, which makes the drill's output easier to read.
    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("FAIL: cannot build a tokio runtime: {e}");
            return std::process::ExitCode::from(2);
        }
    };
    match rt.block_on(run_coordinator()) {
        Ok(true) => std::process::ExitCode::SUCCESS,
        Ok(false) => std::process::ExitCode::from(1),
        Err(e) => {
            eprintln!("FAIL: {e:#}");
            std::process::ExitCode::from(2)
        }
    }
}

// ---------------------------------------------------------------------------
// Worker role: block in dllm_shim_rpc_serve until killed.
// ---------------------------------------------------------------------------

fn run_worker(args: &[String]) -> std::process::ExitCode {
    // `--worker <host> <port>`
    let host = args.get(1).cloned().unwrap_or_else(|| "127.0.0.1".into());
    let port: u16 = match args.get(2).and_then(|p| p.parse().ok()) {
        Some(p) => p,
        None => {
            eprintln!("worker: --worker <host> <port> required");
            return std::process::ExitCode::from(2);
        }
    };
    let lib = match dllm_shim::try_load() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("worker: cannot load the shim: {e}");
            return std::process::ExitCode::from(2);
        }
    };
    println!(
        "worker: shim abi {} loaded from {}; serving on {host}:{port}",
        lib.abi_version(),
        lib.path().display()
    );
    // n_devices = -1 -> every accelerator, falling back to the CPU device.
    match lib.rpc_serve(&host, port, None, WORKER_THREADS, -1) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("worker: rpc_serve failed: {e}");
            std::process::ExitCode::from(2)
        }
    }
}

// ---------------------------------------------------------------------------
// Coordinator role
// ---------------------------------------------------------------------------

async fn run_coordinator() -> anyhow::Result<bool> {
    // ---- 1. the shim ----
    let lib: Arc<ShimLib> = Arc::new(dllm_shim::try_load()?);
    println!(
        "shim: abi version {} loaded from {}",
        lib.abi_version(),
        lib.path().display()
    );

    // ---- 2. the model ----
    let model = model_path()?;
    println!("model: {}", model.display());

    // ---- 3. a worker, in a child process ----
    let port = free_port()?;
    let mut child = spawn_worker(port)?;
    println!("worker: child pid {} on 127.0.0.1:{port}", child.id());

    let up = wait_for_port(port, Duration::from_secs(20));
    if !up {
        // The child is killed by `Child`'s Drop? No — `Child` does not kill on
        // drop, so do it explicitly or the port stays bound and the next run
        // fails confusingly.
        let _ = child.kill();
        anyhow::bail!("worker never started listening on 127.0.0.1:{port}");
    }
    println!("worker: listening (real TCP accept, so activations really cross the socket)");

    // Everything after this point must kill the child on the way out.
    let outcome = coordinator_body(&lib, &model, port).await;
    let _ = child.kill();
    let _ = child.wait();
    outcome
}

async fn coordinator_body(lib: &Arc<ShimLib>, model: &PathBuf, port: u16) -> anyhow::Result<bool> {
    // ---- 4. plan two devices ----
    //
    // `DeviceSpec` values come from the registry in production; here they are
    // the calibration reference numbers so the planner produces the catalog's
    // 9/10/9-style balanced two-way split.
    let devices = vec![
        DeviceSpec::new("coordinator", 16.6, 1000.0, 1024),
        DeviceSpec::new("rpc-worker", 16.6, 1000.0, 1024),
    ];
    let plan = plan_layers(TOTAL_LAYERS, &devices);
    println!(
        "plan: plan_id {} -> {:?}",
        plan.plan_id,
        plan.stages
            .iter()
            .map(|s| format!("{}-{}", s.start, s.end))
            .collect::<Vec<_>>()
    );

    // ---- 5. plan -> tensor_split (the whole point) ----
    let ids: Vec<String> = devices.iter().map(|d| d.node_id.clone()).collect();
    let split = tensor_split_for_plan(&plan, &ids)?;
    println!(
        "tensor_split: {} (slots {:?})",
        dllm_shim::format_tensor_split(&split.weights),
        split.slot_of_stage
    );

    // ---- 6. a real Engine over the shim, with the worker as stage 1 ----
    let endpoint: RpcEndpoint = format!("127.0.0.1:{port}").parse()?;
    let request = SessionRequest {
        model_path: model.clone(),
        model_name: "qwen3-0.6b-q4".to_string(),
        n_ctx: 2048,
        plan: plan.clone(),
        // ORDER MATTERS: `devices[i]` belongs to `plan.stages[i]`, and it is the
        // i-th endpoint handed to `dllm_shim_add_rpc_server`. See the crate docs.
        devices: vec![
            PlannedDevice::local("coordinator"),
            PlannedDevice::remote("rpc-worker", endpoint.clone()),
        ],
    };

    let engine = ShimEngine::open(lib.clone(), request)?;
    println!(
        "registered RPC devices: {} ({} endpoint(s))",
        lib.rpc_device_count().unwrap_or(0),
        lib.registered_endpoints().len()
    );

    // ---- 7. generate ----
    let t0 = Instant::now();
    let mut rx = engine.generate_stream("The capital of France is".to_string());
    let mut text = String::new();
    let mut n_tokens = 0u32;
    let mut saw_done = false;
    // Drain until the channel *closes*, not until `done` arrives: the shim emits
    // its terminal callback from a `FinishGuard` that still has to unwind, and
    // the engine only refreshes its report after `dllm_shim_session_generate`
    // returns. Breaking on `done` would race the report read below and see
    // "no report" for a run that worked.
    while let Some(ev) = rx.recv().await {
        if ev.done {
            saw_done = true;
            continue;
        }
        n_tokens += 1;
        text.push_str(&ev.text);
        if n_tokens >= 24 {
            // Ask the shim to stop at the next token. Dropping `rx` would also
            // do it (the sink's send fails and returns non-zero), but the
            // explicit path is what a real `POST /v1/sessions/{id}/stop` uses.
            engine.cancel();
        }
    }
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("tokens: {n_tokens} in {wall_ms:.0} ms wall\ntext: {text}");
    assert!(saw_done, "the shim must deliver exactly one done event");

    // ---- 8. measured distribution ----
    let telemetry = engine
        .telemetry()
        .ok_or_else(|| anyhow::anyhow!("shim engine reported no telemetry"))?;
    let report = engine
        .last_report()
        .ok_or_else(|| anyhow::anyhow!("no shim report after generation"))?;

    println!("\nreport: {}", serde_json::to_string_pretty(&report)?);
    println!(
        "requested: {}",
        serde_json::to_string(&telemetry.requested_tensor_split)?
    );
    println!(
        "measured:  {}",
        serde_json::to_string_pretty(&telemetry.measured_stages)?
    );
    println!(
        "timing:    prefill {:.1} ms, decode {:.1} ms, {:.2} ms/token, {:.2} tok/s",
        telemetry.prefill_ms.unwrap_or(f64::NAN),
        telemetry.decode_ms.unwrap_or(f64::NAN),
        telemetry.decode_ms_per_token.unwrap_or(f64::NAN),
        telemetry.predicted_per_second.unwrap_or(f64::NAN),
    );
    for note in &telemetry.notes {
        println!("note: {note}");
    }

    // ---- 9. THE ASSERTION ----
    let owners = report.distinct_layer_owners();
    let per_device = report.layers_per_device();
    println!("\nlayer_owner distinct devices: {owners:?}");
    println!("layers per device: {per_device:?}");

    let ok = owners.len() > 1 && report.layers_per_device().iter().any(|(d, n)| *d != 0 && *n > 0);
    if ok {
        println!(
            "PASS: {n_tokens} tokens streamed with layers split across {} ggml devices {:?}",
            owners.len(),
            per_device
        );
    } else {
        println!(
            "FAIL: expected layers on more than one device, got {owners:?} \
             (single-device means the RPC registration did not take effect)"
        );
    }
    Ok(ok)
}

// ---------------------------------------------------------------------------
// Harness helpers
// ---------------------------------------------------------------------------

fn model_path() -> anyhow::Result<PathBuf> {
    if let Ok(p) = std::env::var("DLLM_TEST_MODEL") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
        anyhow::bail!("DLLM_TEST_MODEL does not exist: {}", p.display());
    }
    let local = std::env::var("LOCALAPPDATA")
        .map_err(|_| anyhow::anyhow!("set DLLM_TEST_MODEL (LOCALAPPDATA is unset)"))?;
    let p = PathBuf::from(local)
        .join("dllm")
        .join("models")
        .join("Qwen3-0.6B-Q4_K_M.gguf");
    if !p.is_file() {
        anyhow::bail!(
            "model not found at {}; run `dllm pull qwen3-0.6b-q4` or set DLLM_TEST_MODEL",
            p.display()
        );
    }
    Ok(p)
}

/// Bind port 0, read the assigned port, close. Racy in theory; a fixed port
/// would collide with a worker left over from a previous run.
fn free_port() -> anyhow::Result<u16> {
    let l = TcpListener::bind(("127.0.0.1", 0))?;
    Ok(l.local_addr()?.port())
}

fn spawn_worker(port: u16) -> anyhow::Result<Child> {
    let exe = std::env::current_exe()?;
    Command::new(exe)
        .arg("127.0.0.1")
        .arg(port.to_string())
        // Inherit stdio so the worker's own diagnostics are visible in the run
        // log; it is the same failure surface a real `dllm rpc-worker` has.
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .env(ROLE_ENV, "worker")
        .spawn()
        .map_err(|e| anyhow::anyhow!("could not spawn the worker child process: {e}"))
}

/// Poll until something accepts on `port`, or give up.
///
/// A successful TCP connect is the check the C++ drill uses too, and it is the
/// right one: it proves a real socket is listening, so the layers that follow
/// genuinely travel over the wire rather than into a loopback stub.
fn wait_for_port(port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if TcpStream::connect_timeout(
            &format!("127.0.0.1:{port}").parse().expect("valid socket addr"),
            Duration::from_millis(250),
        )
        .is_ok()
        {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}
