//! Safe ownership of a `dllm_shim_session` plus the token-callback trampoline.
//!
//! # Ownership
//!
//! The ABI hands back a raw pointer with exactly one way to release it
//! (`dllm_shim_session_close`). [`ShimSession`] takes that pointer, closes it in
//! `Drop`, and exposes no way to detach it, so the pointer cannot outlive the
//! library that owns the `llama_model`/`llama_context` behind it.
//!
//! # Send / Sync
//!
//! The header says every function is callable from any thread *except*
//! `dllm_shim_session_generate`, which must not run concurrently with another
//! `generate` on the same session (the shim enforces it with a
//! compare-exchange and returns -1). It also says "a session is not internally
//! synchronised on purpose — a hidden lock would hide pipeline deadlocks". So
//! we do not hide one here either: `ShimSession` is `Send + Sync` (so it can
//! live inside an `Arc` behind `dyn Engine`), and the *caller* serialises
//! generations. `dllm-core`'s `ShimEngine` holds the session behind a `Mutex`
//! for exactly that reason.
//!
//! # Panics must not cross into C++
//!
//! The callback runs on a C++ thread inside `llama_decode`. Rust's default
//! `extern "C"` boundary is `nounwind`, so a panic that reached it would abort
//! the process — but only *after* unwinding through C++ frames that have no
//! idea how to clean up, which is worse than aborting. The trampoline
//! therefore wraps its body in [`std::panic::catch_unwind`]: a panic is caught,
//! logged, and converted into "stop generating" (return `1`), which the shim
//! honours at the next token boundary and unwinds through its own
//! `FinishGuard`. Aborting instead would be defensible but would take the HTTP
//! server down over a formatting bug in a log line; catching keeps the failure
//! local and recoverable.
//!
//! Note the asymmetry: `catch_unwind` needs `UnwindSafe`, which the raw `user`
//! pointer is not, so the body is entered through [`std::panic::AssertUnwindSafe`]
//! justified by the fact that the context is owned by the *generating* thread
//! for the whole duration of the call and is reclaimed after it returns.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;

use crate::abi::{self, SamplerC, SessionPtr, TokenCb};
use crate::loader::ShimLib;
use crate::report::{self, ModelReport, REPORT_BUFFER_BYTES};

/// Upper bound on generated tokens per call. Mirrors `dllm-core`'s
/// `LLAMA_MAX_TOKENS` so both engines cap a runaway generation identically.
pub const MAX_TOKENS: i32 = 512;

/// Sampling configuration, in Rust terms.
///
/// Mirrors the catalog's `sampling.non_thinking` block (temp 0.7 / top_p 0.8 /
/// top_k 20 / presence_penalty 1.5), which is what `LlamaEngine` uses, so
/// switching engines does not silently change output quality.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampler {
    pub top_k: i32,
    pub top_p: f32,
    pub temp: f32,
    pub min_p: f32,
    pub presence_penalty: f32,
    /// `< 0` means non-deterministic (the shim substitutes `LLAMA_DEFAULT_SEED`).
    pub seed: i32,
}

impl Default for Sampler {
    /// `contracts/catalog.json` -> `sampling.non_thinking`.
    fn default() -> Self {
        Self {
            top_k: 20,
            top_p: 0.8,
            temp: 0.7,
            min_p: 0.0,
            presence_penalty: 1.5,
            seed: -1,
        }
    }
}

impl Sampler {
    fn to_c(self) -> SamplerC {
        SamplerC {
            top_k: self.top_k,
            top_p: self.top_p,
            temp: self.temp,
            min_p: self.min_p,
            presence_penalty: self.presence_penalty,
            seed: self.seed,
        }
    }
}

/// How a prompt is wrapped before it reaches `llama_tokenize`.
///
/// The shim ABI exposes `llama_chat_apply_template` nowhere, so a shim-backed
/// session cannot ask the model for its own chat template the way
/// `LlamaEngine` does. [`ChatMl`] therefore hardcodes the Qwen3-style template
/// as a **fallback**, exactly like `LlamaEngine`'s `chat_prompt` fallback branch,
/// so both engines produce the same shape of prompt. A model with a different
/// template needs the ABI extended; until then this is a stated limitation, not
/// a silent guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PromptStyle {
    /// Pass the prompt through untouched (plus the shim's implicit BOS).
    Raw,
    /// Wrap in ChatML and append `/no_think`, matching `LlamaEngine`.
    #[default]
    ChatMl,
}

impl PromptStyle {
    /// Apply the wrapping.
    pub fn apply(self, prompt: &str) -> String {
        match self {
            Self::Raw => prompt.to_string(),
            Self::ChatMl => {
                let content = format!("{prompt}\n/no_think");
                format!("<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")
            }
        }
    }
}

/// Sink for per-token callbacks.
///
/// `on_token` returns `true` to stop generation, which the trampoline turns into
/// the non-zero return the shim expects. It is called on the C++ generating
/// thread and must not block for long — the coordinator's commit loop re-enters
/// from here.
pub trait TokenSink: Send {
    /// `pos` is llama.cpp's *sequence* position (it counts prompt tokens), not
    /// a generation-relative index. Callers that need a 0-based generation
    /// index must count themselves; [`ShimSession::generate`] does not rewrite
    /// it because the raw value is sometimes what you want.
    fn on_token(&mut self, pos: i32, text: &str, done: bool) -> bool;
}

/// Why a session could not be opened or driven.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("model path is not NUL-safe: {0}")]
    PathNotNulSafe(#[from] std::ffi::NulError),
    #[error("dllm_shim_session_open failed: {0}")]
    Open(String),
    #[error("tensor_split has {len} entries but only {devices} ggml device(s) are available")]
    SplitTooLong { len: usize, devices: usize },
    #[error("generation failed: {0}")]
    Generate(String),
    #[error("token callback panicked: {0}")]
    CallbackPanicked(String),
}

/// An open shim session.
///
/// Holds an `Arc<ShimLib>` as well as the raw pointer: the `Library` must not be
/// unloaded while a session is live, and tying the two together makes that
/// structural rather than a documented convention.
#[derive(Debug)]
pub struct ShimSession {
    lib: Arc<ShimLib>,
    ptr: NonNull<c_void>,
    model_path: PathBuf,
    n_ctx: i32,
}

// SAFETY: see the module docs. The session is a handle to a `llama_model` +
// `llama_context` pair that the shim owns; the shim's contract is that
// different threads may call into it, and that `generate` must not overlap with
// itself on one session — which the caller serialises. Moving the handle
// between threads therefore moves nothing that is thread-affine.
unsafe impl Send for ShimSession {}
// SAFETY: as above; `&ShimSession` only exposes metadata and `cancel`, both of
// which the header explicitly documents as safe from any thread.
unsafe impl Sync for ShimSession {}

impl ShimSession {
    /// Open a session over `model_path`.
    ///
    /// `tensor_split` is passed verbatim to the shim, indexed by ggml device
    /// slot — see [`crate::split`] for how to derive it from a plan and
    /// [`crate::registry`] for the registration order it assumes.
    ///
    /// `n_gpu_layers`:
    /// - `-1` — "place every layer we can". This is what a distributed session
    ///   wants; the shim maps it to `INT32_MAX`, which llama.cpp clamps to
    ///   "all layers" without needing to know `n_layer` up front.
    /// - `0` — everything stays on the CPU device. Together with an empty
    ///   `tensor_split` this is the honest single-device configuration.
    pub fn open(
        lib: Arc<ShimLib>,
        model_path: &Path,
        tensor_split: &[f32],
        n_gpu_layers: i32,
        n_ctx: u32,
    ) -> Result<Self, SessionError> {
        // Pre-check existence: the shim returns NULL with an error string for a
        // missing file, but the C++ path would otherwise spend time in
        // `llama_model_load_from_file` producing that error. Checking here
        // yields the actionable message the fallback path logs.
        if !model_path.is_file() {
            return Err(SessionError::Open(format!(
                "model file not found: {}",
                model_path.display()
            )));
        }
        let c_path = CString::new(model_path.as_os_str().as_encoded_bytes())?;

        // `n_split <= 0` is the header's "everything local" signal: pass a null
        // pointer, not an empty slice, because the shim distinguishes them.
        let (split_ptr, n_split) = if tensor_split.is_empty() {
            (std::ptr::null(), 0i32)
        } else {
            (tensor_split.as_ptr(), tensor_split.len() as i32)
        };

        // SAFETY: `c_path` outlives the call; `split_ptr` points at
        // `tensor_split`, which the shim copies into its own
        // `llama_max_devices()`-long buffer before returning and never retains.
        let ptr = unsafe {
            (lib.api().session_open)(
                c_path.as_ptr(),
                split_ptr,
                n_split,
                n_gpu_layers,
                n_ctx as i32,
            )
        };
        if ptr.is_null() {
            return Err(SessionError::Open(lib.last_error()));
        }
        // SAFETY: non-null, and the shim only hands out pointers it owns.
        let ptr = unsafe { NonNull::new_unchecked(ptr) };

        let n_ctx_eff = Self::read_i32(&lib, ptr, |api, p, out| {
            // SAFETY: `p` is a live session and `out` is a valid local.
            unsafe { (api.session_n_ctx)(p, out) }
        })
        .unwrap_or(n_ctx as i32);

        Ok(Self {
            lib,
            ptr,
            model_path: model_path.to_path_buf(),
            n_ctx: n_ctx_eff,
        })
    }

    fn read_i32(
        lib: &ShimLib,
        ptr: NonNull<c_void>,
        f: impl FnOnce(&abi::ShimApi, SessionPtr, *mut i32) -> std::ffi::c_int,
    ) -> Option<i32> {
        let mut out: i32 = 0;
        // `f` is a closure that performs the `unsafe` call itself (each caller
        // documents its own invariants), so this frame needs no unsafe block of
        // its own.
        let rc = f(lib.api(), ptr.as_ptr(), &mut out);
        (rc == 0).then_some(out)
    }

    /// Transformer layers in the loaded model.
    pub fn n_layer(&self) -> Option<i32> {
        Self::read_i32(&self.lib, self.ptr, |api, p, out| unsafe {
            (api.session_n_layer)(p, out)
        })
    }

    /// Context size llama.cpp gave this session.
    pub fn n_ctx(&self) -> i32 {
        self.n_ctx
    }

    /// The model this session was opened over.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// The ABI version of the library backing this session.
    pub fn abi_version(&self) -> std::ffi::c_int {
        self.lib.abi_version()
    }

    /// Ask an in-flight generation to stop at the next token.
    ///
    /// Documented by the header as safe from another thread; returns `false`
    /// when nothing is running, which is honest rather than an error.
    pub fn cancel(&self) -> bool {
        // SAFETY: the session is live for as long as `&self` exists, and the
        // header documents `session_cancel` as callable from any thread.
        unsafe { (self.lib.api().session_cancel)(self.ptr.as_ptr()) == 0 }
    }

    /// What this session is *actually* doing, per the shim.
    pub fn report(&self) -> Result<ModelReport, report::ReportError> {
        let mut buf = vec![0u8; REPORT_BUFFER_BYTES];
        // SAFETY: `buf` is at least one byte and stays alive across the call;
        // the shim truncates to `buf.len()` and always NUL-terminates within
        // it (its own `copy_bounded`).
        let rc = unsafe { (self.lib.api().session_report)(self.ptr.as_ptr(), buf.as_mut_ptr().cast(), buf.len() as i32) };
        if rc != 0 {
            return Err(report::ReportError::Shim(self.lib.last_error()));
        }
        report::parse_report(&buf)
    }

    /// Run one generation, streaming tokens into `sink`.
    ///
    /// `max_tokens < 0` means "until EOS or context end". The sink is dropped
    /// (its box reclaimed) before this returns, so it cannot be touched again
    /// by a stray callback — and the shim guarantees no callback fires after
    /// `dllm_shim_session_generate` returns, because its `FinishGuard` runs the
    /// terminal callback before the function exits.
    ///
    /// # Panics
    /// A panic inside `sink` is caught by the trampoline and turned into an
    /// early stop; it is then re-raised here as [`SessionError::CallbackPanicked`]
    /// so the caller sees it on its own thread rather than in the C++ stack.
    /// That re-raise is a *new* panic, on a Rust thread, so it is safe.
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: i32,
        sampler: &Sampler,
        sink: &mut dyn TokenSink,
    ) -> Result<(), SessionError> {
        let c_prompt = CString::new(prompt)?;
        let sampler_c = sampler.to_c();

        // Heap-allocate the context so its address is stable for the whole call
        // and it survives the FFI boundary as a plain `void*`.
        let mut ctx = CallbackCtx {
            sink,
            panicked: None,
        };
        let user = std::ptr::from_mut(&mut ctx).cast::<c_void>();

        // SAFETY: `user` points at `ctx`, which lives until the end of this
        // function; the trampoline only touches it through `&mut CallbackCtx`
        // and the shim never retains it past the call. `&mut ctx` being on this
        // stack while C++ holds the pointer is fine because the shim's contract
        // is that no callback fires after `generate` returns.
        let rc = unsafe {
            (self.lib.api().session_generate)(
                self.ptr.as_ptr(),
                c_prompt.as_ptr(),
                max_tokens,
                &sampler_c,
                token_trampoline,
                user,
            )
        };

        // The callback is done either way; surface a panic before the rc so the
        // cause is not buried under a generic generate error.
        if let Some(msg) = ctx.panicked.take() {
            return Err(SessionError::CallbackPanicked(msg));
        }
        if rc != 0 {
            return Err(SessionError::Generate(self.lib.last_error()));
        }
        Ok(())
    }
}

impl Drop for ShimSession {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `dllm_shim_session_open` and is closed exactly
        // once. The header requires `dllm_shim_free` (which we never call) to
        // run only after every session is closed, so this ordering is correct.
        unsafe { (self.lib.api().session_close)(self.ptr.as_ptr()) }
    }
}

/// What the C++ thread reads and writes while generating.
struct CallbackCtx<'a> {
    sink: &'a mut dyn TokenSink,
    /// Message from the first panic seen in a callback, if any.
    panicked: Option<String>,
}

/// The `dllm_shim_token_cb` we hand to C++.
///
/// # Why `catch_unwind`
///
/// See the module docs: unwinding from here would cross a C++ frame with no
/// unwind information, which is undefined behaviour, and Rust's `nounwind`
/// `extern "C"` shim would abort the process anyway. Catching keeps a bug in
/// the sink from killing the coordinator's HTTP server.
unsafe extern "C" fn token_trampoline(
    user: *mut c_void,
    pos: std::ffi::c_int,
    text_utf8: *const c_char,
    text_len: std::ffi::c_int,
    done: std::ffi::c_int,
) -> c_int {
    // A null `user` would be a bug on our side; report "stop" rather than
    // dereference. `done` still gets a chance to be counted by C++'s own guard.
    if user.is_null() {
        return 1;
    }
    let ctx = unsafe { &mut *(user.cast::<CallbackCtx<'_>>()) };

    // SAFETY: `ctx` is the live `CallbackCtx` the caller created for this call
    // and the shim has not returned yet, so it cannot have been reclaimed.
    // The result also catches a panic raised by `CStr::from_ptr` on a
    // non-UTF-8 piece, which llama.cpp can emit mid-token.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // `text_utf8` is NOT NUL-terminated (the header is explicit), so slice
        // exactly `text_len` bytes. `done != 0` arrives with `text_len == 0`
        // and, from `FinishGuard`, a null pointer — hence the length check
        // before touching the pointer at all.
        let piece = if text_len > 0 && !text_utf8.is_null() {
            // SAFETY: the shim guarantees `text_len` readable bytes at
            // `text_utf8` for the duration of this call.
            let bytes = unsafe { std::slice::from_raw_parts(text_utf8.cast::<u8>(), text_len as usize) };
            String::from_utf8_lossy(bytes).into_owned()
        } else {
            String::new()
        };
        ctx.sink.on_token(pos, &piece, done != 0)
    }));

    match outcome {
        Ok(true) => 1, // sink asked to stop
        Ok(false) => 0,
        Err(payload) => {
            let msg = panic_message(&payload);
            tracing::error!(panic = %msg, "token sink panicked; stopping generation");
            // Recorded so `ShimSession::generate` can re-raise on our own
            // thread once C++ is fully unwound. First panic wins: a second one
            // while unwinding is impossible, but a repeated failure across
            // tokens would otherwise overwrite the root cause.
            if ctx.panicked.is_none() {
                ctx.panicked = Some(msg);
            }
            1
        }
    }
}

/// Best-effort panic message extraction that never itself panics.
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// The trampoline's address as the C `dllm_shim_token_cb` type.
///
/// Exposed so a test (and the e2e example) can assert the fn pointer coerces at
/// the exact declared ABI without going through a live session.
pub fn token_callback_pointer() -> TokenCb {
    token_trampoline
}

/// Read the shim's per-thread error string as an owned `String`.
///
/// Convenience for callers that catch a `NULL` return and want the message
/// without holding a [`ShimLib`] handle. Only valid on the thread that made the
/// failing call.
pub fn last_error(lib: &ShimLib) -> String {
    lib.last_error()
}

/// A [`TokenSink`] that accumulates into a `String`, for examples and tests.
#[derive(Debug, Default)]
pub struct TextCollector {
    pub text: String,
    pub positions: Vec<i32>,
    pub done_calls: usize,
    /// Stop after this many tokens; `-1` never stops early.
    pub stop_after: i32,
}

impl TokenSink for TextCollector {
    fn on_token(&mut self, _pos: i32, text: &str, done: bool) -> bool {
        if done {
            self.done_calls += 1;
            return false;
        }
        if !text.is_empty() {
            self.text.push_str(text);
        }
        self.positions.push(_pos);
        self.stop_after > 0 && self.positions.len() as i32 >= self.stop_after
    }
}

/// Convenience: `&str` from a C string the shim returned.
///
/// The header promises `dllm_shim_last_error` is never NULL, but a `NULL` from a
/// mismatched build would be a crash, so this degrades to a placeholder instead.
pub fn cstr_to_string(p: *const c_char) -> String {
    if p.is_null() {
        return "<null>".to_string();
    }
    // SAFETY: caller guarantees `p` is a NUL-terminated C string valid for this
    // call; the shim returns a pointer into a thread_local `std::string` whose
    // buffer stays alive until the next shim call on the same thread.
    unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() }
}
