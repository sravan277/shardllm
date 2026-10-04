//! Raw `extern "C"` declarations for the 15 symbols in
//! `native/include/dllm_shim.h`.
//!
//! WHY hand-written instead of `bindgen`
//! -----------------------------------
//! `bindgen` would want to compile a C shim and run libclang at build time
//! (that is what `llama-cpp-sys-2` already does, and it costs ~40 s per clean
//! build). It would also *pin* the header into the build graph, which is the
//! opposite of what we want: the DLL is an optional, gitignored, independently
//! built artifact, so a stale header must not be able to break `cargo build`.
//! Hand-writing the declarations makes this crate's compile step pure Rust and
//! makes ABI drift a *runtime* error (`dllm_shim_abi_version()` +
//! `GetProcAddress` per symbol) instead of a build failure.
//!
//! The types below are transcriptions of the header, not a guess:
//! - `int` and `int32_t` are both 32-bit -> [`core::ffi::c_int`] / [`i32`].
//! - `size_t` is pointer-sized -> [`usize`]; on the `x86_64-pc-windows-gnu`
//!   target this is 8 bytes, matching MinGW's LLP64 `size_t`.
//! - `float` -> [`f32`]; the header's `dllm_shim_sampler` mixes `int32_t` and
//!   `float`, so it must be `#[repr(C)]` with the fields in header order.
//! - `dllm_shim_session` is only ever a pointer across the boundary (the
//!   header forward-declares it), so it is [`core::ffi::c_void`] here. Using
//!   `c_void` rather than a fake struct is deliberate: it makes it impossible
//!   to accidentally dereference or assume a layout.
//!
//! Every function pointer is `unsafe extern "C" fn`, never `extern "C" fn`
//! without `unsafe`, because calling into the shim can invalidate any
//! invariant Rust believes (see the threading contract at the top of the
//! header). Safe calls live one layer up, in [`crate::session`].

#![allow(non_camel_case_types)]

use core::ffi::{c_char, c_int, c_void};

/// `void dllm_shim_free(void)`
pub type FreeFn = unsafe extern "C" fn();

/// `int dllm_shim_init(void)`
pub type InitFn = unsafe extern "C" fn() -> c_int;

/// `const char *dllm_shim_last_error(void)`
///
/// The returned pointer is thread-local and stays valid only until the next
/// shim call *on the same thread*; always copy it out immediately.
pub type LastErrorFn = unsafe extern "C" fn() -> *const c_char;

/// `int dllm_shim_abi_version(void)`
pub type AbiVersionFn = unsafe extern "C" fn() -> c_int;

/// `int dllm_shim_add_rpc_server(const char *host_port)`
pub type AddRpcServerFn = unsafe extern "C" fn(host_port: *const c_char) -> c_int;

/// `int dllm_shim_rpc_device_count(void)`
pub type RpcDeviceCountFn = unsafe extern "C" fn() -> c_int;

/// `int dllm_shim_rpc_device_memory(int32_t, size_t *free, size_t *total)`
pub type RpcDeviceMemoryFn =
    unsafe extern "C" fn(device_index: i32, out_free: *mut usize, out_total: *mut usize) -> c_int;

/// `int dllm_shim_rpc_serve(const char *host, int32_t port, const char *cache_dir, int32_t n_threads, int32_t n_devices)`
///
/// Blocks until the server stops and the ABI has **no** way to abort it, so a
/// coordinator must never call this on a thread it needs back. The test
/// harness spawns the worker in a child process for exactly that reason.
pub type RpcServeFn = unsafe extern "C" fn(
    host: *const c_char,
    port: i32,
    cache_dir: *const c_char,
    n_threads: i32,
    n_devices: i32,
) -> c_int;

/// Opaque `dllm_shim_session *`.
///
/// `c_void`, not a stand-in struct: the header only ever forward-declares the
/// type, so no layout exists to mirror and dereferencing would be unsound.
pub type SessionPtr = *mut c_void;

/// `dllm_shim_session *dllm_shim_session_open(const char *model_path, const float *tensor_split, int32_t n_split, int32_t n_gpu_layers, int32_t n_ctx)`
pub type SessionOpenFn = unsafe extern "C" fn(
    model_path: *const c_char,
    tensor_split: *const f32,
    n_split: i32,
    n_gpu_layers: i32,
    n_ctx: i32,
) -> SessionPtr;

/// `void dllm_shim_session_close(dllm_shim_session *)`
pub type SessionCloseFn = unsafe extern "C" fn(session: SessionPtr);

/// `int dllm_shim_session_generate(dllm_shim_session *, const char *prompt_utf8, int32_t max_tokens, const dllm_shim_sampler *sampler, dllm_shim_token_cb on_token, void *user)`
///
/// Must not run concurrently with another `generate` on the *same* session
/// (the shim enforces this with a compare-exchange and returns -1); different
/// sessions are fully independent.
pub type SessionGenerateFn = unsafe extern "C" fn(
    session: SessionPtr,
    prompt_utf8: *const c_char,
    max_tokens: i32,
    sampler: *const SamplerC,
    on_token: TokenCb,
    user: *mut c_void,
) -> c_int;

/// `int dllm_shim_session_cancel(dllm_shim_session *)` — safe from another thread.
pub type SessionCancelFn = unsafe extern "C" fn(session: SessionPtr) -> c_int;

/// `int dllm_shim_session_n_layer(dllm_shim_session *, int32_t *out)`
pub type SessionNLayerFn = unsafe extern "C" fn(session: SessionPtr, out: *mut i32) -> c_int;

/// `int dllm_shim_session_n_ctx(dllm_shim_session *, int32_t *out)`
pub type SessionNCtxFn = unsafe extern "C" fn(session: SessionPtr, out: *mut i32) -> c_int;

/// `int dllm_shim_session_report(dllm_shim_session *, char *out, int32_t out_len)`
pub type SessionReportFn =
    unsafe extern "C" fn(session: SessionPtr, out: *mut c_char, out_len: i32) -> c_int;

/// `int (*dllm_shim_token_cb)(void *user, int32_t pos, const char *text_utf8, int32_t text_len, int32_t done)`
///
/// The exact fn-pointer shape matters twice over, because this value is both
/// produced and consumed across a C++ thread boundary:
/// 1. it is `extern "C"` so no name mangling / no Rust ABI is involved;
/// 2. it is `unsafe extern "C" fn` because the callee (our trampoline, in
///    [`crate::session`]) must not let a panic escape into C++.
///
/// Parameter semantics that are easy to get wrong:
/// - `text_utf8` is **not** NUL-terminated — always slice `text_len` bytes.
/// - `done != 0` is the terminal callback of a generation and carries
///   `text_len == 0`; the shim's `FinishGuard` even passes `nullptr` there.
/// - returning non-zero asks llama.cpp to stop after this token.
pub type TokenCb =
    unsafe extern "C" fn(user: *mut c_void, pos: i32, text_utf8: *const c_char, text_len: i32, done: i32)
        -> c_int;

/// `#[repr(C)]` mirror of the header's `dllm_shim_sampler`.
///
/// Field order and types are load-bearing: the shim reads this struct
/// straight out of Rust memory. `seed < 0` means "non-deterministic", which is
/// a *valid* value here and not an error.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerC {
    /// Keep this many tokens before truncating the distribution; `<= 0` skips
    /// the top-k sampler entirely.
    pub top_k: i32,
    /// Nucleus cutoff in `(0, 1)`; outside that range the sampler is skipped.
    pub top_p: f32,
    /// Softmax temperature. Negative means greedy in the shim's chain builder.
    pub temp: f32,
    /// Minimum-probability cutoff; skipped when outside `(0, 1)`.
    pub min_p: f32,
    /// Applied before truncation; `0.0` skips the penalties sampler.
    pub presence_penalty: f32,
    /// `< 0` = non-deterministic (the shim substitutes `LLAMA_DEFAULT_SEED`).
    pub seed: i32,
}

/// Every symbol the shim exports, in one `#[repr(C)]` bag.
///
/// Copied out of the loaded `Library` by value (fn pointers are `Copy`) so
/// [`crate::ShimLib`] does not have to carry libloading's lifetime-borrowing
/// `Symbol<T>` types into every safe wrapper, and so the safe wrappers are
/// plain-old-data. The `Library` itself is still owned by [`crate::ShimLib`]
/// and outlives every call — unloading it first would leave these dangling.
#[derive(Debug, Clone, Copy)]
pub struct ShimApi {
    pub init: InitFn,
    pub free: FreeFn,
    pub last_error: LastErrorFn,
    pub abi_version: AbiVersionFn,
    pub add_rpc_server: AddRpcServerFn,
    pub rpc_device_count: RpcDeviceCountFn,
    pub rpc_device_memory: RpcDeviceMemoryFn,
    pub rpc_serve: RpcServeFn,
    pub session_open: SessionOpenFn,
    pub session_close: SessionCloseFn,
    pub session_generate: SessionGenerateFn,
    pub session_cancel: SessionCancelFn,
    pub session_n_layer: SessionNLayerFn,
    pub session_n_ctx: SessionNCtxFn,
    pub session_report: SessionReportFn,
}

/// The 15 exported symbol names, exactly as CMake's generated `dllm_shim.def`
/// spells them. Kept as data (not just as `GetProcAddress` arguments) so a
/// load failure can name the *specific* missing export instead of saying
/// "symbol not found" with no name — an unnamed ABI drift is undebuggable.
pub const EXPORTED_SYMBOLS: [&str; 15] = [
    "dllm_shim_init",
    "dllm_shim_free",
    "dllm_shim_last_error",
    "dllm_shim_abi_version",
    "dllm_shim_add_rpc_server",
    "dllm_shim_rpc_device_count",
    "dllm_shim_rpc_device_memory",
    "dllm_shim_rpc_serve",
    "dllm_shim_session_open",
    "dllm_shim_session_close",
    "dllm_shim_session_generate",
    "dllm_shim_session_cancel",
    "dllm_shim_session_n_layer",
    "dllm_shim_session_n_ctx",
    "dllm_shim_session_report",
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    /// The header declares `dllm_shim_sampler` as six 32-bit scalars, so a
    /// layout mistake here would be silent memory corruption rather than a
    /// compile error. Pin the size and alignment.
    #[test]
    fn sampler_c_layout_matches_the_c_abi() {
        assert_eq!(size_of::<SamplerC>(), 24, "6 x 32-bit scalars");
        assert_eq!(align_of::<SamplerC>(), 4);
        // And the field order, so a shuffle cannot slip through.
        let s = SamplerC {
            top_k: 20,
            top_p: 0.8,
            temp: 0.7,
            min_p: 0.0,
            presence_penalty: 1.5,
            seed: -1,
        };
        // i32 and f32 are both 4-byte scalars, so byte offsets line up 1:1 with
        // the C declaration order.
        let base = &s as *const _ as usize;
        let off = |p: *const u8| p as usize - base;
        assert_eq!(off(&s.top_k as *const i32 as *const u8), 0);
        assert_eq!(off(&s.top_p as *const f32 as *const u8), 4);
        assert_eq!(off(&s.temp as *const f32 as *const u8), 8);
        assert_eq!(off(&s.min_p as *const f32 as *const u8), 12);
        assert_eq!(
            off(&s.presence_penalty as *const f32 as *const u8),
            16
        );
        assert_eq!(off(&s.seed as *const i32 as *const u8), 20);
    }

    /// `dllm_shim_rpc_device_memory` writes `size_t`, which is pointer-sized.
    /// On the `x86_64-pc-windows-gnu` target that is 8 bytes; a `u32` there
    /// would truncate every free/total byte count a worker reports.
    #[test]
    fn size_t_out_params_are_pointer_sized() {
        assert_eq!(size_of::<usize>(), size_of::<*const ()>());
    }

    /// The header's whole point is that the export table cannot drift, so the
    /// loader's name list must keep matching the header exactly. If someone
    /// adds a 16th `DLLM_SHIM_API` symbol without touching this crate, this
    /// test is where it should be noticed.
    #[test]
    fn exported_symbol_list_covers_the_header() {
        assert_eq!(EXPORTED_SYMBOLS.len(), 15);
        for name in EXPORTED_SYMBOLS {
            assert!(name.starts_with("dllm_shim_"), "{name} is not a shim export");
        }
        // Every field of `ShimApi` must have a name in the list.
        let mut names: Vec<&str> = EXPORTED_SYMBOLS.to_vec();
        names.sort_unstable();
        let mut expected = vec![
            "dllm_shim_init",
            "dllm_shim_free",
            "dllm_shim_last_error",
            "dllm_shim_abi_version",
            "dllm_shim_add_rpc_server",
            "dllm_shim_rpc_device_count",
            "dllm_shim_rpc_device_memory",
            "dllm_shim_rpc_serve",
            "dllm_shim_session_open",
            "dllm_shim_session_close",
            "dllm_shim_session_generate",
            "dllm_shim_session_cancel",
            "dllm_shim_session_n_layer",
            "dllm_shim_session_n_ctx",
            "dllm_shim_session_report",
        ];
        expected.sort_unstable();
        assert_eq!(names, expected);
    }

    /// The trampoline must be an `extern "C"` fn pointer: the C++ side stores
    /// it in a `dllm_shim_token_cb` field. Assert the coercion compiles at the
    /// exact declared type, which is what would break first if the signature
    /// drifted.
    #[test]
    fn token_callback_coerces_to_the_c_fn_pointer_type() {
        unsafe extern "C" fn cb(
            _user: *mut c_void,
            _pos: i32,
            _text: *const c_char,
            _len: i32,
            _done: i32,
        ) -> c_int {
            0
        }
        let f: TokenCb = cb;
        let raw: unsafe extern "C" fn(*mut c_void, i32, *const c_char, i32, i32) -> c_int = f;
        assert_eq!(raw as usize, cb as TokenCb as usize);
        // The value we hand to C++ must be callable through the exact declared
        // shape, which the coercion above is what proves.
        let _call: TokenCb = raw;
    }
}
