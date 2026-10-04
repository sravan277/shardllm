//! `Engine` trait + `MockEngine` for Phase 0.
//!
//! Real inference (llama-cpp-2) lands in Phase 1 behind a Cargo feature gate.

use tokio::sync::mpsc;

/// One streamed generation event.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TokenEvent {
    /// Zero-based position in this generation.
    pub pos: u32,
    /// Text for this step (word for `MockEngine`).
    pub text: String,
    /// True on the final event of the stream.
    pub done: bool,
}

/// Inference engine abstraction.
///
/// `generate_stream` is synchronous and returns a channel receiver; the
/// engine spawns the generation task internally. Callers just drain `rx`.
pub trait Engine: Send + Sync + 'static {
    /// Names of locally available models (e.g. `qwen3-0.6b-q4`).
    fn model_list(&self) -> Vec<String>;

    /// Start a generation; tokens arrive on the returned channel.
    fn generate_stream(&self, prompt: String) -> mpsc::Receiver<TokenEvent>;

    /// Which engine is actually behind this `dyn Engine`.
    ///
    /// Reported by `GET /api/stats` as `engine`. There used to be no such hook:
    /// `dllm-serve` discriminated by `size_of_val(engine.as_ref())`, which
    /// works only while every engine has a *distinct* size. `ShimEngine` is one
    /// `Arc` — the same 8 bytes as `LlamaEngine` — so that heuristic would have
    /// silently labelled a distributed shim session as `"llama"` and hidden
    /// exactly the thing this repo is trying to prove. Implementations override
    /// it; the default keeps the old heuristic so nothing that predates this
    /// method changes behaviour.
    fn flavor(&self) -> EngineFlavor {
        // Legacy fallback: `LlamaEngine` is a single `Arc<LlamaInner>` (8 B on
        // 64-bit), `MockEngine` a `Vec<String>` (24 B).
        if std::mem::size_of_val(self) == std::mem::size_of::<Arc<()>>() {
            EngineFlavor::Llama
        } else {
            EngineFlavor::Mock
        }
    }

    /// Measured, prompt-free telemetry for `GET /api/stats` and `GET /v1/plan`.
    ///
    /// `None` means "this engine measures nothing about itself", which callers
    /// render as an explicit `null` — never as zeros (ADR-024/030 discipline:
    /// a null means not measured, not zero).
    fn telemetry(&self) -> Option<EngineTelemetry> {
        None
    }
}

/// Which engine implementation is live. Mirrors `dllm_serve::EngineKind` but
/// lives here so the `Engine` trait can return it without `dllm-serve` becoming
/// a dependency of `dllm-core`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EngineFlavor {
    /// `llama-cpp-2`, CPU-only, `n_gpu_layers = 0`.
    Llama,
    /// The C++ `dllm_shim` over llama.cpp + ggml-rpc; may span devices.
    Shim,
    /// Canned Phase 0 tokens.
    Mock,
}

impl EngineFlavor {
    /// Wire value used by `GET /api/stats`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Shim => "shim",
            Self::Mock => "mock",
        }
    }
}

impl std::fmt::Display for EngineFlavor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One stage the planner *asked* for. Intent, not measurement.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RequestedStage {
    /// Position in `PipelinePlan::stages`, which is also ggml slot order.
    pub stage: usize,
    pub device_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub layers: u32,
    /// `host:port` for a remote worker; `None` for this process.
    pub endpoint: Option<String>,
}

/// One contiguous run of layers a ggml device **actually** holds, read back out
/// of `llama_model::dev_layer()` via `dllm_shim_session_report`.
///
/// This is the counterpart to [`RequestedStage`], and it is deliberately kept
/// even when the two agree: a Distribution view that only showed the request
/// could not tell a working pipeline from a plan nobody applied.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MeasuredStage {
    pub device_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub layers: u32,
    /// ggml global device index the layers landed on.
    pub ggml_device: i32,
}

/// Everything an engine can honestly say about itself after a generation.
///
/// Every field is `Option` for the same reason the API's are: `null` = not
/// measured. `notes` carries degradations that are true but not numeric — a
/// dropped stage, an assumed local-device count — so they are reported instead
/// of disappearing.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct EngineTelemetry {
    /// Redundant with `EngineFlavor`, so a `measured` blob is self-describing
    /// when it is copied out of the API response on its own.
    pub engine: String,
    /// Catalog name of the loaded model.
    pub model: String,
    /// `dllm_shim_abi_version()`, shim engine only.
    pub shim_abi_version: Option<i32>,
    /// `ggml_backend_dev_count()` at report time (local + every RPC device).
    pub devices: Option<i32>,
    pub n_layer: Option<i32>,
    pub n_ctx: Option<i32>,
    /// Layers off the CPU device, as counted from `dev_layer()`.
    pub n_gpu_layers: Option<i32>,
    /// The `tensor_split` handed to `dllm_shim_session_open`, verbatim.
    pub requested_tensor_split: Option<Vec<f32>>,
    /// The plan this request was derived from, in stage order.
    pub requested_stages: Vec<RequestedStage>,
    /// Measured reality from `layer_owner`. `None` when the shim could not read
    /// it back — never reconstructed from `requested_tensor_split`.
    pub measured_stages: Option<Vec<MeasuredStage>>,
    pub prefill_ms: Option<f64>,
    pub decode_ms: Option<f64>,
    /// `decode_ms / n_generated`; `None` when nothing was generated.
    pub decode_ms_per_token: Option<f64>,
    pub predicted_per_second: Option<f64>,
    pub n_prompt_tokens: Option<i32>,
    pub n_generated: Option<i32>,
    /// `host:port` of every registered RPC worker, in registration order.
    pub rpc_endpoints: Vec<String>,
    /// True statements about degradation, in the order they were discovered.
    pub notes: Vec<String>,
}

/// Phase 0 mock: streams a canned sentence word-by-word with 30 ms delays.
#[derive(Debug, Clone, Default)]
pub struct MockEngine {
    models: Vec<String>,
}

impl MockEngine {
    pub fn new() -> Self {
        Self {
            models: vec!["qwen3-0.6b-q4".to_string()],
        }
    }

    pub fn with_models(models: Vec<String>) -> Self {
        Self { models }
    }
}

impl Engine for MockEngine {
    fn model_list(&self) -> Vec<String> {
        self.models.clone()
    }

    fn flavor(&self) -> EngineFlavor {
        EngineFlavor::Mock
    }

    fn generate_stream(&self, _prompt: String) -> mpsc::Receiver<TokenEvent> {
        let (tx, rx) = mpsc::channel(32);
        tokio::spawn(async move {
            let canned = "Hello from the distributed LAN mesh streaming one word at a time.";
            let words: Vec<&str> = canned.split_whitespace().collect();
            let n = words.len() as u32;
            for (i, word) in words.into_iter().enumerate() {
                tokio::time::sleep(std::time::Duration::from_millis(30)).await;
                let done = (i as u32) + 1 == n;
                let ev = TokenEvent {
                    pos: i as u32,
                    text: word.to_string(),
                    done,
                };
                if tx.send(ev).await.is_err() {
                    break;
                }
                if done {
                    break;
                }
            }
        });
        rx
    }
}

// ---------------------------------------------------------------------------
// `LlamaEngine`: real CPU inference via llama-cpp-2 (Phase 1).
// ---------------------------------------------------------------------------

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, OnceLock};

/// Upper bound on new tokens per `generate_stream` call.
const LLAMA_MAX_TOKENS: u32 = 512;
/// `penalty_last_n` window for the penalties sampler.
const LLAMA_PENALTY_LAST_N: i32 = 64;

/// Process-wide llama backend guard.
///
/// `LlamaBackend::init()` may only succeed once per process, so the first
/// `LlamaEngine::load` initializes it and every later load (or retry after a
/// failed load) reuses the same guard. It is intentionally never dropped.
static LLAMA_BACKEND: OnceLock<llama_cpp_2::llama_backend::LlamaBackend> = OnceLock::new();

fn llama_backend() -> anyhow::Result<&'static llama_cpp_2::llama_backend::LlamaBackend> {
    // `OnceLock::get_or_try_init` is unavailable on this toolchain, so open-code
    // it: the loser of an init race observes the winner's guard via `get()`.
    if let Some(b) = LLAMA_BACKEND.get() {
        return Ok(b);
    }
    match llama_cpp_2::llama_backend::LlamaBackend::init() {
        Ok(b) => Ok(LLAMA_BACKEND.get_or_init(|| b)),
        Err(first_err) => {
            // Another thread may have won the race; poll briefly before giving up.
            for _ in 0..100 {
                if let Some(b) = LLAMA_BACKEND.get() {
                    return Ok(b);
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Err(anyhow::anyhow!("llama backend init failed: {first_err}"))
        }
    }
}

/// Real inference engine backed by llama.cpp (CPU-only, `n_gpu_layers = 0`).
///
/// Cheap to clone: the model is shared behind an `Arc`. Each
/// `generate_stream` call builds its own context + sampler inside a dedicated
/// blocking thread, so concurrent generations do not share mutable state.
pub struct LlamaEngine {
    inner: Arc<LlamaInner>,
}

struct LlamaInner {
    model: llama_cpp_2::model::LlamaModel,
    n_ctx: u32,
}

impl LlamaEngine {
    /// Load GGUF weights from `model_path` with a `n_ctx`-token context.
    ///
    /// `n_ctx` values below 512 are clamped to 512 so there is always room
    /// for at least the generation budget.
    pub fn load(model_path: &Path, n_ctx: u32) -> anyhow::Result<Self> {
        // Pre-check existence: `LlamaModel::load_from_file` debug-asserts on
        // the path, which would panic (not `Err`) on missing weights and take
        // down the server instead of falling back to `MockEngine`.
        if !model_path.is_file() {
            anyhow::bail!("model file not found: {}", model_path.display());
        }
        let backend = llama_backend()?;
        let params = llama_cpp_2::model::params::LlamaModelParams::default().with_n_gpu_layers(0);
        let model = llama_cpp_2::model::LlamaModel::load_from_file(backend, model_path, &params)
            .map_err(|e| {
                anyhow::anyhow!("failed to load model {}: {e}", model_path.display())
            })?;
        tracing::info!(
            path = %model_path.display(),
            n_ctx,
            "loaded llama model"
        );
        Ok(Self {
            inner: Arc::new(LlamaInner {
                model,
                n_ctx: n_ctx.max(512),
            }),
        })
    }

    /// Blocking prefill + decode loop. Runs on a dedicated (non-async)
    /// thread; streams tokens via `blocking_send`, stopping on send error.
    fn generate_blocking(
        inner: &LlamaInner,
        prompt: &str,
        tx: &tokio::sync::mpsc::Sender<TokenEvent>,
    ) {
        if let Err(e) = Self::generate_inner(inner, prompt, tx) {
            // Setup/mid-stream failure: log and close the channel. The
            // caller (`dllm-serve`) ends the stream when `rx` closes.
            tracing::warn!("llama generation failed: {e:#}");
        }
    }

    fn generate_inner(
        inner: &LlamaInner,
        prompt: &str,
        tx: &tokio::sync::mpsc::Sender<TokenEvent>,
    ) -> anyhow::Result<()> {
        use llama_cpp_2::context::params::LlamaContextParams;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::AddBos;
        use llama_cpp_2::sampling::LlamaSampler;

        let backend = llama_backend()?;
        let model = &inner.model;
        let n_ctx = inner.n_ctx;

        // Chat-template prompt (Qwen3, non-thinking via `/no_think` suffix).
        let full_prompt = chat_prompt(model, prompt);

        // Fresh context per generation (no shared mutable state).
        let ctx_params =
            LlamaContextParams::default().with_n_ctx(Some(
                NonZeroU32::new(n_ctx).expect("n_ctx clamped to >= 512 in load()"),
            ));
        let mut ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| anyhow::anyhow!("failed to create llama context: {e}"))?;

        // Prefill: tokenize, then truncate the head if the prompt would leave
        // no room for the generation budget.
        let mut prompt_tokens = model
            .str_to_token(&full_prompt, AddBos::Always)
            .map_err(|e| anyhow::anyhow!("failed to tokenize prompt: {e}"))?;
        let max_prompt =
            (n_ctx.saturating_sub(LLAMA_MAX_TOKENS) as usize).max(64);
        if prompt_tokens.len() > max_prompt {
            tracing::warn!(
                prompt_tokens = prompt_tokens.len(),
                max_prompt,
                "prompt truncated to fit context"
            );
            prompt_tokens.truncate(max_prompt);
        }
        if prompt_tokens.is_empty() {
            anyhow::bail!("prompt tokenized to zero tokens");
        }

        let mut batch = LlamaBatch::new(prompt_tokens.len(), 1);
        batch
            .add_sequence(&prompt_tokens, 0, false)
            .map_err(|e| anyhow::anyhow!("failed to build prefill batch: {e}"))?;
        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("prefill decode failed: {e}"))?;

        // Sampler chain: non-thinking params per contracts/catalog.json
        // (`sampling.non_thinking`: temp 0.7 / top_p 0.8 / top_k 20 /
        // presence_penalty 1.5). Chain ends with `dist` (required).
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as u32 ^ d.subsec_nanos())
            .unwrap_or(0x2b992dd5);
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(model.n_vocab(), LLAMA_PENALTY_LAST_N, 1.0, 0.0, 1.5),
            LlamaSampler::temp(0.7),
            LlamaSampler::top_k(20),
            LlamaSampler::top_p(0.8, 1),
            LlamaSampler::dist(seed),
        ]);
        // Seed repetition tracking with the prompt tokens.
        sampler.accept_many(&prompt_tokens);

        // Autoregressive decode loop; `done` is sent only on EOS or when the
        // 512-token budget is exhausted.
        let mut n_past = prompt_tokens.len() as i32;
        let mut pos: u32 = 0;
        let mut generated: u32 = 0;
        // `LlamaSampler::sample` takes the position *within the most recently
        // decoded batch* holding the logits: the last prompt token after
        // prefill, then each single fed-back token (index 0).
        let mut sample_idx: i32 = prompt_tokens.len() as i32 - 1;
        loop {
            let token = sampler.sample(&ctx, sample_idx);
            sampler.accept(token);
            generated += 1;
            let last = model.is_eog_token(token) || generated >= LLAMA_MAX_TOKENS;
            let text = decode_piece(model, token);
            // Skip empty (e.g. control) pieces mid-stream, but always emit
            // the terminal event so the stream ends with `done = true`.
            if !text.is_empty() || last {
                let ev = TokenEvent {
                    pos,
                    text,
                    done: last,
                };
                pos += 1;
                if tx.blocking_send(ev).is_err() {
                    // Receiver gone: stop generating.
                    return Ok(());
                }
            }
            if last {
                return Ok(());
            }
            let mut next = LlamaBatch::new(1, 1);
            next.add(token, n_past, &[0], true)
                .map_err(|e| anyhow::anyhow!("failed to build decode batch: {e}"))?;
            n_past += 1;
            ctx.decode(&mut next)
                .map_err(|e| anyhow::anyhow!("decode failed at token {generated}: {e}"))?;
            sample_idx = 0;
        }
    }
}

impl Engine for LlamaEngine {
    fn model_list(&self) -> Vec<String> {
        vec!["qwen3-0.6b-q4".to_string()]
    }

    fn flavor(&self) -> EngineFlavor {
        EngineFlavor::Llama
    }

    /// `None`: `llama-cpp-2` exposes no per-device placement information at
    /// all (it cannot even reach ggml-rpc), so this engine genuinely has nothing
    /// measured to report and the API renders an explicit `null`.
    fn telemetry(&self) -> Option<EngineTelemetry> {
        None
    }

    fn generate_stream(&self, prompt: String) -> mpsc::Receiver<TokenEvent> {
        let (tx, rx) = mpsc::channel(32);
        let inner = self.inner.clone();
        if let Err(e) = std::thread::Builder::new()
            .name("llama-decode".to_string())
            .spawn(move || Self::generate_blocking(&inner, &prompt, &tx))
        {
            tracing::warn!("failed to spawn llama decode thread: {e}");
            // `tx` is dropped with the failed closure, closing `rx`.
        }
        rx
    }
}

/// Build the chat-template prompt for a user turn.
///
/// Uses the model's baked-in template with `/no_think` appended (Qwen3
/// convention to disable thinking mode, matching the non-thinking sampling
/// params). Falls back to a minimal ChatML prompt if the model has no
/// usable template.
fn chat_prompt(model: &llama_cpp_2::model::LlamaModel, user_text: &str) -> String {
    use llama_cpp_2::model::LlamaChatMessage;

    let content = format!("{user_text}\n/no_think");
    let fallback = format!("<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n");
    match LlamaChatMessage::new("user".to_string(), content) {
        Ok(msg) => model
            .chat_template(None)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .and_then(|tmpl| {
                model
                    .apply_chat_template(&tmpl, &[msg], true)
                    .map_err(|e| anyhow::anyhow!("{e}"))
            })
            .unwrap_or_else(|e| {
                tracing::warn!("chat template unavailable ({e:#}); using ChatML fallback");
                fallback
            }),
        Err(e) => {
            tracing::warn!("invalid chat message ({e}); using ChatML fallback");
            fallback
        }
    }
}

/// Decode one token to text, growing the buffer on demand. Never fails:
/// undecodable tokens map to U+FFFD via `from_utf8_lossy`, unknown tokens
/// to an empty string (skipped mid-stream by the caller).
fn decode_piece(model: &llama_cpp_2::model::LlamaModel, token: llama_cpp_2::token::LlamaToken) -> String {
    let mut size = 32usize;
    loop {
        match model.token_to_piece_bytes(token, size, false, None) {
            Ok(bytes) => return String::from_utf8_lossy(&bytes).into_owned(),
            Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(_)) if size < 1024 => {
                size *= 2;
            }
            Err(_) => return String::new(),
        }
    }
}
