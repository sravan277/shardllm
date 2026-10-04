//! `ShimEngine`: an [`Engine`] that actually distributes transformer layers.
//!
//! # Why this exists
//!
//! Until now `dllm serve` ran either `LlamaEngine` (the `llama-cpp-2` crate,
//! `n_gpu_layers = 0`, physically unable to reach ggml-rpc) or `MockEngine`, and
//! the pipeline plan computed by [`plan_layers`] was rendered to JSON and thrown
//! away. `native/src/dllm_shim.cpp` proves a second device can hold layers; this
//! module is what makes the product call it.
//!
//! # What it does
//!
//! 1. Registers each planned worker's `host:port` with
//!    `dllm_shim_add_rpc_server`, **in plan stage order** (see
//!    [`crate::shim_engine::tensor_split_for_plan`] and the crate's ordering
//!    invariant).
//! 2. Derives `tensor_split` from the plan's per-stage layer counts.
//! 3. Opens one `dllm_shim_session` and streams tokens straight off the shim's
//!    per-token callback.
//! 4. Publishes the shim's report so `GET /v1/plan` and `GET /api/stats` show
//!    the device each layer **actually** landed on, next to the range it was
//!    **asked** to take.
//!
//! # What it honestly does not do
//!
//! - **No per-stage latency.** llama.cpp's perf counters in this build are
//!   whole-graph; there is no per-device timer behind the ABI, so per-stage
//!   `latency_ms` stays `null` and only the measured totals (prefill, decode,
//!   ms/token) are reported.
//! - **No chat template from the model.** `llama_chat_apply_template` is not in
//!   the ABI, so [`PromptStyle::ChatMl`] hardcodes the same ChatML fallback
//!   `LlamaEngine` uses.
//! - **No in-process RPC worker.** `dllm_shim_rpc_serve` blocks forever with no
//!   stop path, so `dllm rpc-worker` and the e2e example both run the worker in
//!   a child process.
//!
//! # Degradation
//!
//! Nothing here is allowed to stop the server booting. A missing DLL, an ABI
//! mismatch, an unreachable worker or a session that will not open all resolve
//! to a log line plus a value the caller can read back through
//! [`EngineTelemetry::notes`] — and, in `apps/dllm`, to a fallback to
//! `LlamaEngine` that is reported on `GET /api/stats`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use dllm_shim::{
    ModelReport, PlannedStage, PromptStyle, RpcEndpoint, Sampler, ShimLib, ShimSession, TokenSink,
    contiguous_runs,
};
use tokio::sync::mpsc;

use crate::engine::{
    Engine, EngineFlavor, EngineTelemetry, MeasuredStage, RequestedStage, TokenEvent,
};
use crate::plan::PipelinePlan;

/// Cheap to clone (one `Arc`), and the reason `Engine::flavor` exists: a
/// size-based engine sniff cannot tell this apart from `LlamaEngine`.
#[derive(Clone)]
pub struct ShimEngine {
    inner: Arc<ShimInner>,
}

/// Upper bound on new tokens per generation. Matches `dllm-shim`'s
/// [`dllm_shim::MAX_TOKENS`] and `LlamaEngine`'s `LLAMA_MAX_TOKENS`, so both
/// engines cap a runaway generation identically.
pub const SHIM_MAX_TOKENS: i32 = 512;

/// How many ggml device slots this process is assumed to contribute.
///
/// The shim's `collect_layer_devices` appends **every** non-ACCEL local ggml
/// backend, so a machine with a local CUDA/Vulkan device would have more than
/// one local slot and `tensor_split` would shift. The ABI exposes no way to
/// enumerate local devices before a session exists, so exactly one is assumed
/// (the CPU, which is what every supported build has) and the assumption is
/// checked afterwards against the report's `devices` count. A mismatch becomes
/// a note, not a silent mis-assignment.
pub const ASSUMED_LOCAL_DEVICES: usize = 1;

/// One device a plan stage can land on, in plan order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedDevice {
    /// Registry `device_id`; what the API calls the stage's owner.
    pub device_id: String,
    /// `host:port` of its ggml-rpc server, or `None` for this process.
    ///
    /// `None` means "local" — it does **not** mean the device is paired or
    /// healthy, only that the coordinator runs it. The caller (dllm-serve) has
    /// already filtered to paired + `worker_active` rows.
    pub endpoint: Option<RpcEndpoint>,
}

impl PlannedDevice {
    /// A stage held by this process.
    pub fn local(device_id: impl Into<String>) -> Self {
        Self {
            device_id: device_id.into(),
            endpoint: None,
        }
    }

    /// A stage held by a remote ggml-rpc worker.
    pub fn remote(device_id: impl Into<String>, endpoint: RpcEndpoint) -> Self {
        Self {
            device_id: device_id.into(),
            endpoint: Some(endpoint),
        }
    }
}

/// Everything needed to (re)open a distributed session.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionRequest {
    pub model_path: PathBuf,
    /// Catalog name reported by `model_list()` / telemetry.
    pub model_name: String,
    pub n_ctx: u32,
    /// The plan to turn into `tensor_split`. Must have one stage per entry in
    /// `devices`.
    pub plan: PipelinePlan,
    /// Devices, **in `plan.stages` order**. This is the ordering invariant: the
    /// k-th device here belongs to stage k, and it must be the k-th endpoint
    /// handed to `dllm_shim_add_rpc_server`.
    pub devices: Vec<PlannedDevice>,
}

impl SessionRequest {
    /// Single-device request: this process takes every layer. Still routed
    /// through the shim (with `tensor_split = [n]`), because that is the
    /// configuration the *degraded* multi-device path collapses into, and
    /// having one code path means the degraded case is the tested case.
    pub fn local_only(model_path: PathBuf, model_name: String, n_ctx: u32, device_id: String) -> Self {
        let n_layers = crate::plan::TOTAL_LAYERS;
        Self {
            model_path,
            model_name,
            n_ctx,
            plan: PipelinePlan {
                plan_id: 0,
                stages: vec![crate::plan::StageRange::new(0, n_layers - 1)],
            },
            devices: vec![PlannedDevice::local(device_id)],
        }
    }
}

/// The `tensor_split` a plan implies, plus the layout it produced.
///
/// Pure: no FFI, so it is unit-testable on a machine with no `dllm_shim.dll`.
/// The live variant adds "how many ggml devices each stage actually owns",
/// which only the shim can answer.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitPlan {
    /// `tensor_split` weights, one per ggml device slot, in slot order.
    pub weights: Vec<f32>,
    /// Requested stages, in plan order (the `requested` half of the report).
    pub requested: Vec<RequestedStage>,
    /// ggml slot index each requested stage starts at.
    pub slot_of_stage: Vec<usize>,
}

/// Derive `tensor_split` from a plan, assuming one ggml device per stage.
///
/// This is the whole bridge, and it is deliberately trivial so it is obviously
/// right: llama.cpp's `tensor_split` is a list of *relative* weights indexed by
/// ggml device slot, and the planner already decided how many layers each stage
/// should hold, so the weights are the layer counts. llama.cpp normalises them.
///
/// Worked example — the catalog default 9/10/9 over three devices:
///
/// ```text
/// plan.stages  = [0..8 (9 layers), 9..18 (10 layers), 19..27 (9 layers)]
/// device_ids   = ["self", "pixel-8", "laptop"]        (plan order, self first)
/// slot_of_stage= [0, 1, 2]                             (ggml device slots)
/// tensor_split = [9, 10, 9]                            (== layer counts)
/// ```
///
/// So a generation over that session asks llama.cpp for 9 layers on the local
/// ggml device (slot 0) and 13 layers across the two registered workers
/// (slots 1 and 2). Change the plan to 19/9 — which is what a worker twice as
/// fast earns — and the weights become `[19, 9]`; nothing else in the code
/// changes. The full behaviour is pinned by
/// `tests::changing_stage_layer_counts_changes_the_weights` and
/// `tests::a_plan_of_n_stages_yields_n_weights_that_sum_to_the_layer_count`.
///
/// (These examples are unit tests rather than doc-tests because `dllm-core`
/// links llama.cpp's static archives through a `-C link-args=` recipe that
/// rustdoc cannot accept on this toolchain — `-l advapi32` is a rustc flag, not
/// a rustdoc one. See `scripts/build-env.ps1`.)
pub fn tensor_split_for_plan(
    plan: &PipelinePlan,
    device_ids: &[String],
) -> Result<SplitPlan, SplitError> {
    if plan.stages.len() != device_ids.len() {
        return Err(SplitError::ArityMismatch {
            stages: plan.stages.len(),
            devices: device_ids.len(),
        });
    }
    let counts: Vec<u32> = plan.stages.iter().map(|s| s.len()).collect();
    let devices_per_stage = vec![ASSUMED_LOCAL_DEVICES; counts.len()];
    let layout = dllm_shim::tensor_split_from_stages(&counts, &devices_per_stage)
        .map_err(|e| SplitError::Layout(e.to_string()))?;
    let requested = plan
        .stages
        .iter()
        .enumerate()
        .map(|(i, range)| RequestedStage {
            stage: i,
            device_id: device_ids.get(i).cloned().unwrap_or_default(),
            layer_start: range.start,
            layer_end: range.end,
            layers: range.len(),
            endpoint: None,
        })
        .collect();
    Ok(SplitPlan {
        weights: layout.weights,
        requested,
        slot_of_stage: layout.slots.iter().map(|s| s.first_slot).collect(),
    })
}

/// Why a `tensor_split` could not be derived from a plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitError {
    /// Plan arity and device arity disagree — the ordering invariant is broken.
    ArityMismatch { stages: usize, devices: usize },
    /// The underlying layout rejected the plan (empty, zero layers, ...).
    Layout(String),
}

impl std::fmt::Display for SplitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArityMismatch { stages, devices } => write!(
                f,
                "plan has {stages} stage(s) but {devices} device(s); stage order and device \
                 order must match one-for-one because tensor_split is indexed by ggml device slot"
            ),
            Self::Layout(why) => write!(f, "cannot lay out stages: {why}"),
        }
    }
}

impl std::error::Error for SplitError {}

/// Why a distributed session could not be established.
#[derive(Debug, thiserror::Error)]
pub enum ShimEngineError {
    #[error("cannot derive tensor_split: {0}")]
    Split(#[from] SplitError),
    #[error("shim session failed: {0:#}")]
    Session(#[from] dllm_shim::SessionError),
}

impl ShimEngine {
    /// Load the shim and open the first session for `req`.
    ///
    /// `lib` must be the *same* library instance every other session uses: the
    /// ggml device registry it owns is process-global, so a second `ShimLib`
    /// would be a second, inconsistent view of the device list.
    pub fn open(lib: Arc<ShimLib>, req: SessionRequest) -> Result<Self, ShimEngineError> {
        let sampler = Sampler::default();
        let engine = Self {
            inner: Arc::new(ShimInner {
                lib,
                model_path: req.model_path.clone(),
                model_name: req.model_name.clone(),
                n_ctx: req.n_ctx,
                sampler,
                prompt_style: PromptStyle::default(),
                session: RwLock::new(None),
                gate: Mutex::new(()),
                layout: RwLock::new(None),
                report: RwLock::new(None),
                dropped: RwLock::new(Vec::new()),
            }),
        };
        engine.ensure_session(&req)?;
        Ok(engine)
    }

    /// Re-open the session if the requested layout differs from the live one.
    ///
    /// Returns `true` when a new session was opened. Idempotent by design: the
    /// supervisor calls this on a timer, and reloading a 400 MB model every tick
    /// would be absurd. The comparison is on the derived `tensor_split` plus the
    /// ordered device list, which is precisely the input that decides placement.
    ///
    /// Ordering note: the new session is opened *before* the old one is dropped.
    /// That transiently doubles weight memory (locally and on each worker), which
    /// is the price of never leaving the coordinator with no session because a
    /// reopen failed. Qwen3-0.6B Q4 is ~397 MB, so the spike is bounded and
    /// logged.
    pub fn ensure_session(&self, req: &SessionRequest) -> Result<bool, ShimEngineError> {
        let inner = &self.inner;
        if inner.model_path != req.model_path {
            return Err(ShimEngineError::Session(dllm_shim::SessionError::Open(
                format!(
                    "model path changed from {} to {}; build a new ShimEngine instead of \
                     re-pointing a live one",
                    inner.model_path.display(),
                    req.model_path.display()
                ),
            )));
        }
        let (plan, dropped) = inner.register_and_derive(req)?;

        // Nothing changed: keep the loaded session (and its resident weights).
        if let Some(current) = inner.layout().as_ref() {
            if current.weights == plan.weights && current.requested == plan.requested {
                return Ok(false);
            }
        }

        tracing::info!(
            model = %inner.model_path.display(),
            devices = plan.slot_of_stage.len(),
            tensor_split = %dllm_shim::format_tensor_split(&plan.weights),
            stages = plan.requested.len(),
            dropped_stages = dropped.len(),
            "opening distributed shim session"
        );

        // `n_gpu_layers = -1` -> "place every layer we can", which is the whole
        // point; an empty weights list would mean "everything local" and is
        // never produced here because a plan always has >= 1 stage.
        let session = ShimSession::open(
            inner.lib.clone(),
            &inner.model_path,
            &plan.weights,
            -1,
            inner.n_ctx,
        )?;

        // Take the gate for the swap: `dllm_shim_add_rpc_server` already ran
        // above (it is global and irreversible, so it must not depend on
        // whether a generation is in flight), but the model reload itself waits
        // for any running decode so weights are never unloaded mid-forward-pass.
        let gate = inner.gate.lock().unwrap_or_else(|e| e.into_inner());
        {
            *inner.session.write().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(session));
        }
        *inner.layout.write().unwrap_or_else(|e| e.into_inner()) = Some(plan);
        *inner.dropped.write().unwrap_or_else(|e| e.into_inner()) = dropped;

        // A fresh session's report is stale-by-construction: it describes a
        // model load that has not produced a token yet. Drop it so the API says
        // "not measured" instead of showing the previous session's numbers.
        *inner.report.write().unwrap_or_else(|e| e.into_inner()) = None;
        drop(gate);
        Ok(true)
    }

    /// Last observed `dllm_shim_session_report`.
    pub fn last_report(&self) -> Option<ModelReport> {
        self.inner.report().clone()
    }

    /// Ask an in-flight generation to stop at the next token.
    ///
    /// Wires `POST /v1/sessions/{id}/stop`'s cancel flag to the shim's
    /// `dllm_shim_session_cancel`. Returns `false` when nothing is generating,
    /// which the shim reports by returning -1 — honest, not an error.
    ///
    /// Deliberately does **not** take the generation gate: a stop request that
    /// queues behind the decode it is trying to interrupt would never arrive.
    /// It only needs the `Arc` (a short read lock) because the header documents
    /// `dllm_shim_session_cancel` as safe from any thread — it flips an atomic
    /// the decode loop polls between tokens.
    pub fn cancel(&self) -> bool {
        self.inner
            .session
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|s| s.cancel())
            .unwrap_or(false)
    }

    /// Host a ggml-rpc worker in **this** process.
    ///
    /// Blocks forever (the ABI has no stop path), so this is only for a process
    /// whose entire job is to be a worker — `dllm rpc-worker`. Never call it on
    /// the coordinator's HTTP thread.
    pub fn serve_rpc(
        lib: &ShimLib,
        host: &str,
        port: u16,
        n_threads: i32,
    ) -> Result<(), String> {
        // `n_devices = -1`: expose every accelerator, falling back to the CPU
        // device, which is what a CPU-only worker wants.
        lib.rpc_serve(host, port, None, n_threads, -1)
    }
}

struct ShimInner {
    lib: Arc<ShimLib>,
    model_path: PathBuf,
    model_name: String,
    n_ctx: u32,
    sampler: Sampler,
    prompt_style: PromptStyle,
    /// The live session, or `None` before the first successful open.
    ///
    /// An `RwLock` rather than a `Mutex` so that cheap observers — `cancel`,
    /// `telemetry`'s `n_layer`, the post-generation report read — can clone the
    /// `Arc` without queueing behind a decode that may run for seconds. The
    /// write side is only ever held for the pointer swap in
    /// `ensure_session`.
    session: RwLock<Option<Arc<ShimSession>>>,
    /// Exclusive gate over "a generation is running" / "the session is being
    /// replaced".
    ///
    /// The shim rejects two overlapping `generate` calls on one session, and
    /// swapping the session out from under a running decode would unload weights
    /// mid-forward-pass. Both are prevented here rather than with a per-request
    /// error, so a user never sees a generation fail because two chats overlapped.
    /// It is *not* a hidden pipeline lock: it is held only for the duration of
    /// one `dllm_shim_session_generate`.
    gate: Mutex<()>,
    layout: RwLock<Option<SplitPlan>>,
    report: RwLock<Option<ModelReport>>,
    /// Stages whose endpoint was unreachable on the last open.
    dropped: RwLock<Vec<dllm_shim::DroppedStage>>,
}

impl ShimInner {
    fn layout(&self) -> std::sync::RwLockReadGuard<'_, Option<SplitPlan>> {
        self.layout.read().unwrap_or_else(|e| e.into_inner())
    }

    fn report(&self) -> std::sync::RwLockReadGuard<'_, Option<ModelReport>> {
        self.report.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Register the request's endpoints (in plan order) and derive the weights.
    ///
    /// Returns the resolved [`SplitPlan`] plus the stages that could not be
    /// honoured. `Ok` here does **not** mean "distributed": a plan whose only
    /// worker is dead resolves to a one-entry plan, and the caller sees that in
    /// both `plan.requested.len()` and the returned dropped list.
    fn register_and_derive(
        &self,
        req: &SessionRequest,
    ) -> Result<(SplitPlan, Vec<dllm_shim::DroppedStage>), SplitError> {
        // THE ORDERING INVARIANT, in code: `stages` is built by walking
        // `req.plan.stages` and indexing `req.devices` at the same position.
        // Both were produced by `dllm-serve` from the same `pipeline_candidates`
        // ordering (self first, then by device_id), so stage k and device k are
        // the same machine. Re-ordering either list independently would make
        // `tensor_split[1]` land on the wrong worker, and nothing downstream
        // could detect it.
        let mut stages = Vec::with_capacity(req.plan.stages.len());
        for (i, device) in req.devices.iter().enumerate() {
            stages.push(PlannedStage {
                stage: i,
                device_id: device.device_id.clone(),
                endpoint: device.endpoint.clone(),
            });
        }
        if stages.len() != req.plan.stages.len() {
            return Err(SplitError::ArityMismatch {
                stages: req.plan.stages.len(),
                devices: stages.len(),
            });
        }

        // One irreversible global side effect, deduped and order-preserving.
        let outcome = self.lib.register_stages(&stages);

        // Weights come from the LIVE stage list, not the plan: a dropped stage
        // must not leave a weight behind, or llama.cpp would index a device that
        // is not in its device array. Its layers are redistributed across the
        // survivors by llama.cpp's own normalisation, and the reduction is
        // reported rather than hidden.
        let counts: Vec<u32> = outcome
            .live
            .iter()
            .map(|s| {
                req.plan
                    .stages
                    .get(s.stage)
                    .map(|r| r.len())
                    .unwrap_or(0)
            })
            .collect();
        let devices_per_stage = outcome.devices_per_stage();
        let layout = dllm_shim::tensor_split_from_stages(&counts, &devices_per_stage)
            .map_err(|e| SplitError::Layout(e.to_string()))?;

        let requested: Vec<RequestedStage> = outcome
            .live
            .iter()
            .map(|s| {
                let range = req
                    .plan
                    .stages
                    .get(s.stage)
                    .copied()
                    .unwrap_or(crate::plan::StageRange::new(0, 0));
                RequestedStage {
                    stage: s.stage,
                    device_id: s.device_id.clone(),
                    layer_start: range.start,
                    layer_end: range.end,
                    layers: range.len(),
                    endpoint: s.endpoint.as_ref().map(|e| e.as_endpoint_string()),
                }
            })
            .collect();

        Ok((
            SplitPlan {
                weights: layout.weights,
                requested,
                slot_of_stage: layout.slots.iter().map(|s| s.first_slot).collect(),
            },
            outcome.dropped,
        ))
    }
}

impl Engine for ShimEngine {
    fn model_list(&self) -> Vec<String> {
        vec![self.inner.model_name.clone()]
    }

    fn flavor(&self) -> EngineFlavor {
        EngineFlavor::Shim
    }

    fn generate_stream(&self, prompt: String) -> mpsc::Receiver<TokenEvent> {
        let (tx, rx) = mpsc::channel(32);
        // Dedicated (non-async) thread, same shape as `LlamaEngine`: the shim
        // calls us back *on the generating C++ thread*, and re-entering the
        // coordinator's commit loop from there must not block a reactor worker.
        let inner = self.inner.clone();
        if let Err(e) = std::thread::Builder::new()
            .name("shim-decode".to_string())
            .spawn(move || Self::generate_blocking(&inner, &prompt, &tx))
        {
            tracing::warn!("failed to spawn shim decode thread: {e}");
            // `tx` drops with the failed closure, closing `rx`: the caller sees a
            // stream that ended without tokens rather than a hang.
        }
        rx
    }

    fn telemetry(&self) -> Option<EngineTelemetry> {
        self.telemetry()
    }
}

impl ShimEngine {
    /// Measured telemetry, mapping ggml device slots back to `device_id`s.
    ///
    /// Returns `Some` **before the first generation**, with every timing field
    /// `None` and `measured_stages = None`: the model is loaded (which is real
    /// information: `n_layer`, `n_ctx`, `devices` are known) but no layer
    /// placement has been observed yet, so nothing about placement is claimed.
    pub fn telemetry(&self) -> Option<EngineTelemetry> {
        let inner = &self.inner;
        let layout = inner.layout();
        let report = inner.report();
        let dropped = inner
            .dropped
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        let mut notes: Vec<String> = Vec::new();
        for d in &dropped {
            notes.push(format!(
                "stage {} ({}) dropped: {} unreachable at {}",
                d.stage, d.device_id, d.endpoint, d.reason
            ));
        }

        // Map ggml slot -> device_id using the *requested* layout's slot order,
        // which is what registration established. A slot we cannot attribute is
        // reported under its numeric id rather than guessed at.
        let slot_owner: Vec<String> = layout
            .as_ref()
            .map(|l| {
                l.requested
                    .iter()
                    .map(|s| s.device_id.clone())
                    .collect()
            })
            .unwrap_or_default();

        let measured_stages = report.as_ref().and_then(|r| {
            let owner = r.layer_owner.as_ref()?;
            Some(
                contiguous_runs(owner)
                    .into_iter()
                    .map(|(dev, start, len)| MeasuredStage {
                        device_id: slot_owner
                            .get(dev as usize)
                            .cloned()
                            .unwrap_or_else(|| format!("ggml-device-{dev}")),
                        layer_start: start as u32,
                        layer_end: (start + len - 1) as u32,
                        layers: len as u32,
                        ggml_device: dev,
                    })
                    .collect(),
            )
        });

        if let Some(r) = report.as_ref() {
            if let Some(note) = r.layer_owner_note.as_deref() {
                notes.push(format!("layer_owner unavailable: {note}"));
            }
            // The local-device-count assumption is checked here, where the real
            // count finally exists.
            let expected = ASSUMED_LOCAL_DEVICES + inner.lib.rpc_device_count().unwrap_or(0);
            if r.devices as usize != expected {
                notes.push(format!(
                    "shim reports {} ggml devices but {} were expected ({} local + RPC); \
                     tensor_split slot order may not match the plan",
                    r.devices, expected, ASSUMED_LOCAL_DEVICES
                ));
            }
        }

        let timing = report.as_ref().and_then(|r| r.timing);
        Some(EngineTelemetry {
            engine: EngineFlavor::Shim.as_str().to_string(),
            model: inner.model_name.clone(),
            shim_abi_version: Some(inner.lib.abi_version()),
            devices: report.as_ref().map(|r| r.devices),
            n_layer: report.as_ref().map(|r| r.n_layer).or_else(|| {
                // No report yet (nothing generated since the session opened):
                // ask the session directly, which is still real information -
                // the model is loaded.
                inner
                    .session
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .as_ref()
                    .and_then(|s| s.n_layer())
            }),
            n_ctx: Some(
                report
                    .as_ref()
                    .map(|r| r.n_ctx)
                    .unwrap_or(inner.n_ctx as i32),
            ),
            n_gpu_layers: report.as_ref().map(|r| r.n_gpu_layers),
            requested_tensor_split: layout.as_ref().map(|l| l.weights.clone()),
            requested_stages: layout
                .as_ref()
                .map(|l| l.requested.clone())
                .unwrap_or_default(),
            measured_stages,
            prefill_ms: timing.map(|t| t.prefill_ms),
            decode_ms: timing.map(|t| t.decode_ms),
            decode_ms_per_token: timing.and_then(|t| t.decode_ms_per_token()),
            predicted_per_second: timing.map(|t| t.predicted_per_second),
            n_prompt_tokens: timing.map(|t| t.n_prompt_tokens),
            n_generated: timing.map(|t| t.n_generated),
            rpc_endpoints: inner
                .lib
                .registered_endpoints()
                .iter()
                .map(|e| e.as_endpoint_string())
                .collect(),
            notes,
        })
    }

    /// Blocking generate: token callback -> channel, then refresh the report.
    fn generate_blocking(inner: &ShimInner, prompt: &str, tx: &mpsc::Sender<TokenEvent>) {
        // The gate is held for the whole generation: serialises concurrent
        // `generate` calls (which the shim rejects outright) and keeps
        // `ensure_session` from swapping the session mid-stream. It is released
        // before the report refresh so a long decode does not block a worker
        // from joining.
        let gate = inner.gate.lock().unwrap_or_else(|e| e.into_inner());
        let session = match inner
            .session
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            Some(s) => s.clone(),
            None => {
                tracing::error!("shim session is not open; cannot generate");
                return;
            }
        };


        let full_prompt = inner.prompt_style.apply(prompt);
        let mut sink = ChannelSink {
            tx: tx.clone(),
            // The shim reports llama.cpp's *sequence* position, which counts the
            // prompt tokens; `TokenEvent.pos` is documented as "zero-based
            // position in this generation" and `MockEngine`/`LlamaEngine` both
            // start at 0. So the generation-relative index is counted here
            // rather than passed through.
            next_pos: 0,
            stop_after: None,
        };
        let result = session.generate(
            &full_prompt,
            SHIM_MAX_TOKENS,
            &inner.sampler,
            &mut sink,
        );
        drop(session); // release the Arc before touching the report lock
        // Release the gate before the report read so a supervisor tick (or a
        // worker joining) is not blocked behind a report that can take a while
        // on a slow link.
        drop(gate);


        match result {
            Ok(()) => {
                // Refresh what the shim measured. A failure here must not fail
                // the generation — the tokens already streamed — but it does
                // mean `measured_stages` stays stale, so say so.
                match session_report_of(inner) {
                    Ok(report) => {
                        *inner.report.write().unwrap_or_else(|e| e.into_inner()) = Some(report);
                    }
                    Err(e) => {
                        tracing::warn!("shim report unavailable after generation: {e}");
                    }
                }
            }
            Err(e) => {
                tracing::warn!("shim generation failed: {e:#}");
            }
        }
    }
}

/// Re-read the report, taking only a short read lock on the session handle.
fn session_report_of(inner: &ShimInner) -> Result<ModelReport, dllm_shim::ReportError> {
    let session = inner
        .session
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    match session.as_ref() {
        Some(s) => s.report(),
        None => Err(dllm_shim::ReportError::Shim(
            "session closed during generation".to_string(),
        )),
    }
}

/// Maps the shim's per-token callback onto an mpsc channel.
///
/// `on_token` runs on the C++ generating thread, so the send is a
/// `blocking_send` (the same choice `LlamaEngine` makes) and the whole body is
/// kept allocation-light and panic-free.
struct ChannelSink {
    tx: mpsc::Sender<TokenEvent>,
    next_pos: u32,
    /// `POST /v1/sessions/{id}/stop` sets this so the generation ends early.
    stop_after: Option<u32>,
}

impl TokenSink for ChannelSink {
    fn on_token(&mut self, _pos: i32, text: &str, done: bool) -> bool {
        if done {
            // Terminal callback. `text_len` is 0 here (and the pointer may even
            // be null), so this carries no text by contract.
            let _ = self.tx.blocking_send(TokenEvent {
                pos: self.next_pos,
                text: String::new(),
                done: true,
            });
            return false;
        }
        let ev = TokenEvent {
            pos: self.next_pos,
            text: text.to_string(),
            done: false,
        };
        self.next_pos += 1;
        if self.tx.blocking_send(ev).is_err() {
            // Receiver gone (client disconnected): stop generating. Returning
            // non-zero is the ABI's "stop at the next token".
            return true;
        }
        if let Some(limit) = self.stop_after {
            if self.next_pos >= limit {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{DeviceSpec, StageRange, TOTAL_LAYERS};

    fn ids(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("node-{i}")).collect()
    }

    #[test]
    fn a_plan_of_n_stages_yields_n_weights_that_sum_to_the_layer_count() {
        for n in 1..=5usize {
            let devices: Vec<DeviceSpec> = (0..n)
                .map(|i| DeviceSpec::new(format!("node-{i}"), 16.6, 1000.0, 1024))
                .collect();
            let plan = crate::plan::plan_layers(TOTAL_LAYERS, &devices);
            let split = tensor_split_for_plan(&plan, &ids(n)).expect("valid");
            assert_eq!(split.weights.len(), n, "one weight per stage");
            assert_eq!(
                split.weights.iter().sum::<f32>(),
                TOTAL_LAYERS as f32,
                "weights are layer counts"
            );
            assert_eq!(split.slot_of_stage, (0..n).collect::<Vec<_>>());
            for (i, s) in split.requested.iter().enumerate() {
                assert_eq!(s.stage, i);
                assert_eq!(s.device_id, format!("node-{i}"));
                assert_eq!(s.layers, plan.stages[i].len());
            }
        }
    }

    #[test]
    fn changing_stage_layer_counts_changes_the_weights() {
        let balanced = PipelinePlan {
            plan_id: 1,
            stages: vec![StageRange::new(0, 13), StageRange::new(14, 27)],
        };
        let skewed = PipelinePlan {
            plan_id: 2,
            stages: vec![StageRange::new(0, 22), StageRange::new(23, 27)],
        };
        let dev = ids(2);
        assert_eq!(
            tensor_split_for_plan(&balanced, &dev).unwrap().weights,
            vec![14.0, 14.0]
        );
        assert_eq!(
            tensor_split_for_plan(&skewed, &dev).unwrap().weights,
            vec![23.0, 5.0]
        );
        // And the real planner agrees with the hand-written 9/10/9 case, which
        // is what the module doc-comment walks through.
        let three = PipelinePlan::balanced_9_10_9();
        let split = tensor_split_for_plan(&three, &ids(3)).unwrap();
        assert_eq!(split.weights, vec![9.0, 10.0, 9.0]);
        assert_eq!(split.slot_of_stage, vec![0, 1, 2]);
        assert_eq!(split.weights.iter().sum::<f32>(), 28.0);
        assert_eq!(
            split
                .requested
                .iter()
                .map(|s| (s.device_id.as_str(), s.layers))
                .collect::<Vec<_>>(),
            vec![("node-0", 9), ("node-1", 10), ("node-2", 9)]
        );
    }

    #[test]
    fn arity_mismatch_is_a_named_error_not_a_silent_truncation() {
        let plan = PipelinePlan::balanced_9_10_9();
        assert_eq!(
            tensor_split_for_plan(&plan, &ids(2)),
            Err(SplitError::ArityMismatch {
                stages: 3,
                devices: 2
            })
        );
        assert!(tensor_split_for_plan(&plan, &ids(2))
            .unwrap_err()
            .to_string()
            .contains("ggml device slot"));
    }

    #[test]
    fn a_single_device_plan_is_one_weight_covering_every_layer() {
        let plan = crate::plan::plan_layers(TOTAL_LAYERS, &[DeviceSpec::new("self", 16.6, 1000.0, 1)]);
        let split = tensor_split_for_plan(&plan, &ids(1)).unwrap();
        assert_eq!(split.weights, vec![TOTAL_LAYERS as f32]);
        assert_eq!(split.requested[0].layer_start, 0);
        assert_eq!(split.requested[0].layer_end, TOTAL_LAYERS - 1);
    }

    #[test]
    fn local_only_request_carries_every_layer_on_one_stage() {
        let req = SessionRequest::local_only(
            PathBuf::from("model.gguf"),
            "qwen3-0.6b-q4".to_string(),
            4096,
            "dllm-self".to_string(),
        );
        assert_eq!(req.plan.stages.len(), 1);
        assert_eq!(req.plan.stages[0].len(), TOTAL_LAYERS);
        assert!(req.devices[0].endpoint.is_none(), "self is local, not remote");
        let split = tensor_split_for_plan(&req.plan, &req.devices.iter().map(|d| d.device_id.clone()).collect::<Vec<_>>()).unwrap();
        assert_eq!(split.weights, vec![TOTAL_LAYERS as f32]);
    }

    #[test]
    fn engine_flavor_and_wire_strings_are_stable() {
        assert_eq!(EngineFlavor::Llama.as_str(), "llama");
        assert_eq!(EngineFlavor::Shim.as_str(), "shim");
        assert_eq!(EngineFlavor::Mock.as_str(), "mock");
        assert_eq!(EngineFlavor::Shim.to_string(), "shim");
    }

    /// The integration claim itself, minus the DLL: `Arc<ShimEngine>` must
    /// coerce to `Arc<dyn Engine>` so `dllm-serve` needs no changes to its
    /// serving path.
    ///
    /// This is also the assertion that catches a size-based engine sniff
    /// silently reporting a distributed session as `"llama"`: `ShimEngine` is
    /// one `Arc`, the same size as `LlamaEngine`. The coercion below only
    /// compiles because `Engine` is object safe and `ShimEngine` implements
    /// every method.
    #[test]
    fn shim_engine_coerces_to_a_dyn_engine() {
        fn coerce(engine: Arc<ShimEngine>) -> Arc<dyn Engine> {
            engine
        }
        // A function item, not a call: the point is that the signature
        // type-checks, which needs no DLL and no model.
        let _coercer: fn(Arc<ShimEngine>) -> Arc<dyn Engine> = coerce;
        assert_eq!(EngineFlavor::Shim.as_str(), "shim");
    }

    /// The channel sink is pure logic over an mpsc channel, so the callback
    /// mapping (including the generation-relative position and the terminal
    /// `done` event) is testable with no native artifact.
    #[test]
    fn channel_sink_maps_the_shim_callback_onto_token_events() {
        let (tx, mut rx) = mpsc::channel(8);
        let mut sink = ChannelSink {
            tx,
            next_pos: 0,
            stop_after: None,
        };
        // The shim's positions count prompt tokens; the sink must not leak them.
        assert!(!sink.on_token(17, "Pa", false));
        assert!(!sink.on_token(18, "ris", false));
        assert!(!sink.on_token(19, "", true));

        let mut got = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            got.push(ev);
        }
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].pos, 0);
        assert_eq!(got[0].text, "Pa");
        assert!(!got[0].done);
        assert_eq!(got[2].pos, 2);
        assert_eq!(got[2].text, "");
        assert!(got[2].done, "the terminal callback must carry done=true");
    }

    #[test]
    fn channel_sink_stops_when_the_receiver_is_gone() {
        let (tx, rx) = mpsc::channel(8);
        drop(rx);
        let mut sink = ChannelSink {
            tx,
            next_pos: 0,
            stop_after: None,
        };
        // Returning true is the ABI's "stop generating"; a dropped receiver must
        // not make the decode loop spin to max_tokens.
        assert!(sink.on_token(0, "x", false));
    }

    #[test]
    fn channel_sink_honours_an_explicit_token_limit() {
        let (tx, mut rx) = mpsc::channel(8);
        let mut sink = ChannelSink {
            tx,
            next_pos: 0,
            stop_after: Some(2),
        };
        assert!(!sink.on_token(0, "a", false));
        assert!(sink.on_token(1, "b", false), "second token hits the limit");
        while rx.try_recv().is_ok() {}
    }
}
