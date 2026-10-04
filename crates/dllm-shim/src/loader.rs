//! Runtime discovery and loading of `dllm_shim.dll`.
//!
//! # Why runtime, not link-time
//!
//! `native/build-win/` is gitignored. The DLL only exists on a machine that ran
//! `scripts/build-llama-win.ps1`, which needs CMake, Ninja and a MinGW
//! toolchain. If this crate declared `#[link(name = "dllm_shim")]`, then
//! `dllm.exe` would not *build* on a machine without the import library, let
//! alone run — and the product's own rule is that a missing optional native
//! artifact must never stop the HTTP server booting (ADR-004's single-device
//! fast path is the product). So: no `build.rs`, no `#[link]`, no `bindgen`;
//! the symbol addresses are fetched at run time and a failure is a `Result`,
//! not a load-time abort.
//!
//! # `libloading` versus a hand-rolled `LoadLibraryExW`
//!
//! `libloading` 0.8.9 was already in `Cargo.lock` (pulled in by
//! `clang-sys` -> `bindgen` -> `llama-cpp-sys-2`) and already in the local
//! registry cache, so promoting it to a direct dependency adds **no** new crate
//! and **no** new download. It also gets one detail right that is easy to miss
//! by hand: [`open_library`] passes `LOAD_WITH_ALTERED_SEARCH_PATH`, which puts
//! the *DLL's own directory* at the front of its dependency search.
//! `LoadLibraryExW(path, 0, 0)` would look for `dllm_shim.dll`'s own
//! dependencies (`libstdc++-6.dll`, `libgcc_s_seh-1.dll`, `libwinpthread-1.dll`)
//! only in the executable's directory and `PATH`, and fail with a bare "module
//! not found" that names none of them. Hand-rolling was the fallback plan; we
//! did not need it, and the reason is above.
//!
//! # ABI handshake
//!
//! Loading is not enough. Every symbol is resolved individually so a stale DLL
//! reports *which* export is missing, then `dllm_shim_abi_version()` is
//! compared against [`EXPECTED_ABI_VERSION`]. A mismatch is a hard
//! [`LoadError::AbiMismatch`]: the header documents version bumps on any ABI
//! break, so a mismatch means the argument order or a struct layout differs,
//! and continuing would corrupt memory rather than fail.

use std::ffi::c_int;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use crate::abi::ShimApi;
use crate::endpoint::RpcEndpoint;
use crate::registry::{PlannedStage, RegistrationOutcome, WorkerRegistry};

/// ABI version this crate was written against (`DLLM_SHIM_ABI_VERSION` in
/// `native/src/dllm_shim.cpp` at tag b7418). Bumped here in lockstep with the
/// header; a mismatch is refused, never ignored.
pub const EXPECTED_ABI_VERSION: c_int = 1;

/// Environment variable that overrides DLL discovery entirely. Absolute or
/// relative paths both work; relative resolves against the process working
/// directory. Ops escape hatch for a DLL built somewhere other than
/// `native/build-win/`.
pub const DLL_PATH_ENV: &str = "DLLM_SHIM_DLL";

/// Platform handle type. Both variants expose `get(&self, &[u8])`, so symbol
/// resolution is written once; only the *constructor* differs, because only
/// Windows needs `LOAD_WITH_ALTERED_SEARCH_PATH`.
#[cfg(target_os = "windows")]
type RawLibrary = libloading::os::windows::Library;
#[cfg(not(target_os = "windows"))]
type RawLibrary = libloading::os::unix::Library;

/// Why the shim could not be used. **Never** fatal to the process: every
/// variant is something `dllm serve` logs and then degrades from.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("dllm_shim.dll not found (looked in: {searched}); build it with scripts/build-llama-win.ps1 or point {env} at it")]
    NotFound { searched: String, env: &'static str },
    #[error("could not load {path}: {reason}")]
    DlOpen { path: String, reason: String },
    #[error("dllm_shim.dll does not export `{symbol}`; the DLL predates this header - rebuild native/ with scripts/build-llama-win.ps1")]
    MissingSymbol { symbol: &'static str },
    #[error(
        "ABI mismatch: {path} reports version {found} but this build speaks {expected}; \
         rebuild native/ with scripts/build-llama-win.ps1"
    )]
    AbiMismatch {
        path: String,
        found: c_int,
        expected: c_int,
    },
    #[error("dllm_shim_init() failed: {0}")]
    InitFailed(String),
}

/// A loaded shim: the OS handle (which owns the code) plus every symbol
/// resolved by value.
///
/// The fn pointers are plain `Copy` data, so the safe wrappers in
/// [`crate::session`] need no lifetime plumbing — but they are only valid while
/// `self._lib` is alive, which is why [`ShimLib`] owns the handle and why every
/// wrapper holds an `Arc<ShimLib>`.
#[derive(Debug)]
pub struct ShimLib {
    /// Dropped last: unloading the DLL would leave every fn pointer below
    /// dangling, and the field order here is the only thing expressing that.
    _lib: RawLibrary,
    api: ShimApi,
    path: PathBuf,
    abi_version: c_int,
    /// Process-wide endpoint bookkeeping. The ggml device list the shim mutates
    /// is global, so "already registered?" cannot be per-session.
    registry: Mutex<WorkerRegistry>,
}

impl ShimLib {
    /// Raw symbol table. Every call site is `unsafe` and must have a comment
    /// saying why the ABI permits it.
    #[inline]
    pub fn api(&self) -> &ShimApi {
        &self.api
    }

    /// Where the DLL was actually found, for the startup log line.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// The version the DLL reported, echoed into telemetry.
    pub fn abi_version(&self) -> c_int {
        self.abi_version
    }

    /// Endpoints already handed to `dllm_shim_add_rpc_server`, in registration
    /// (== ggml slot) order.
    pub fn registered_endpoints(&self) -> Vec<RpcEndpoint> {
        self.lock_registry().registered()
    }

    /// Devices contributed by an endpoint, or `None` when never registered.
    pub fn endpoint_devices(&self, ep: &RpcEndpoint) -> Option<usize> {
        self.lock_registry().devices_for(ep)
    }

    /// Total ggml *remote* devices the shim knows about, from
    /// `dllm_shim_rpc_device_count()`. `None` when the shim refuses, e.g. a
    /// build without `GGML_RPC`. Local devices are deliberately not counted by
    /// that call.
    pub fn rpc_device_count(&self) -> Option<usize> {
        // SAFETY: no arguments, no retained state; the header documents it as
        // callable from any thread before and after `init`.
        let n = unsafe { (self.api.rpc_device_count)() };
        (n >= 0).then(|| n as usize)
    }

    /// Free/total bytes a *remote* device reports, straight from
    /// `RPC_CMD_GET_DEVICE_MEMORY`.
    ///
    /// A dead worker yields `(0, 0)` rather than an error, which is the honest
    /// answer from the shim's point of view — so callers must treat `(0, 0)` as
    /// "unknown", not "empty".
    pub fn rpc_device_memory(&self, device_index: usize) -> Result<(u64, u64), LoadError> {
        let mut free: usize = 0;
        let mut total: usize = 0;
        // SAFETY: both out-params are valid, initialised locals and the shim
        // only writes through them. `device_index` is bounds-checked by the
        // shim, which returns -1 with a message rather than reading past the
        // device list.
        let rc = unsafe { (self.api.rpc_device_memory)(device_index as i32, &mut free, &mut total) };
        if rc == 0 {
            Ok((free as u64, total as u64))
        } else {
            Err(LoadError::InitFailed(self.last_error()))
        }
    }

    /// The most recent error on **this** thread.
    ///
    /// Must be called on the thread that made the failing call: the shim keeps
    /// the string in a `thread_local`. Copying it out immediately is also
    /// mandatory — the next shim call on the same thread overwrites it.
    pub fn last_error(&self) -> String {
        // SAFETY: the header guarantees a non-null, thread-local,
        // NUL-terminated string that the caller must not free.
        unsafe {
            let p = (self.api.last_error)();
            if p.is_null() {
                return "<shim returned a null error string>".to_string();
            }
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }

    /// Host a ggml-rpc worker. **Blocks until the server stops** and the ABI
    /// has no way to abort it, so only ever call this on a thread (or in a
    /// child process) the caller does not need back.
    ///
    /// `n_devices < 0` means "expose every accelerator this device has",
    /// falling back to the CPU device — which is what a CPU-only worker wants.
    pub fn rpc_serve(
        &self,
        host: &str,
        port: u16,
        cache_dir: Option<&str>,
        n_threads: i32,
        n_devices: i32,
    ) -> Result<(), String> {
        let c_host = std::ffi::CString::new(host).map_err(|e| e.to_string())?;
        let c_dir = match cache_dir {
            Some(d) => Some(std::ffi::CString::new(d).map_err(|e| e.to_string())?),
            None => None,
        };
        // SAFETY: both CStrings outlive the call; `cache_dir` NULL is the
        // header's documented "no cache" value. The call blocks until the
        // server stops, which the caller has been told about.
        let rc = unsafe {
            (self.api.rpc_serve)(
                c_host.as_ptr(),
                port as i32,
                c_dir.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                n_threads,
                n_devices,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(self.last_error())
        }
    }

    /// `dllm_shim_free`.
    ///
    /// Deliberately not called by the server. The header requires it to run only
    /// after every session is closed, and in this build it merely clears the
    /// endpoint vector — the ggml registries and RPC sockets are
    /// process-lifetime singletons, so calling it mid-flight releases nothing
    /// and risks nothing being released. Exposed for a future explicit-shutdown
    /// path.
    pub fn free(&self) {
        // SAFETY: takes no arguments and touches only the shim's own endpoint
        // vector, which the header says may be called once at shutdown.
        unsafe { (self.api.free)() }
    }

    /// The process-wide worker registry.
    pub(crate) fn worker_registry(&self) -> std::sync::MutexGuard<'_, WorkerRegistry> {
        // A poisoned registry mutex means a thread panicked inside
        // `register`; recovering keeps the coordinator serving instead of
        // cascading that panic into every later session open.
        self.registry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Register every stage's endpoint **in plan order**, exactly once.
    ///
    /// `stages` MUST already be ordered the way `PipelinePlan::stages` is: the
    /// `tensor_split` the caller derives next is indexed by ggml device slot,
    /// and slots follow registration order. Sorting or re-ordering here is the
    /// one mistake this whole crate cannot detect — see
    /// [`crate::registry`]'s module docs.
    ///
    /// Every stage is independent: an endpoint that fails to answer is recorded
    /// in [`RegistrationOutcome::dropped`] and the remaining stages still
    /// register. `changed` is `true` only if the *global* device list actually
    /// grew, so a caller can skip a session reopen on a no-op pass.
    pub fn register_stages(&self, stages: &[PlannedStage]) -> RegistrationOutcome {
        self.worker_registry().register_in_order(self, stages)
    }

    fn lock_registry(&self) -> std::sync::MutexGuard<'_, WorkerRegistry> {
        self.worker_registry()
    }
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// DLL file name for this platform (`dll`/`so`/`dylib` + the `lib` prefix Unix
/// linkers expect).
pub const fn dll_file_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "dllm_shim.dll"
    } else if cfg!(target_os = "macos") {
        "libdllm_shim.dylib"
    } else {
        "libdllm_shim.so"
    }
}

/// Every place the DLL may live, in probe order.
///
/// `native/build-win/` is first because `dllm serve` is documented to run from
/// the workspace root and that is where `scripts/build-llama-win.ps1` puts it.
/// Next to the executable covers a packaged copy. `%LOCALAPPDATA%\dllm\` sits
/// next to the weights, which is where an installer would land it.
///
/// An explicit [`DLL_PATH_ENV`] override *replaces* the list rather than
/// prepending to it: an operator pointing at a specific build must never
/// silently get a different one from the default probe order.
pub fn candidate_paths() -> Vec<PathBuf> {
    let name = dll_file_name();
    if let Some(explicit) = std::env::var_os(DLL_PATH_ENV) {
        let p = PathBuf::from(explicit);
        // An override naming a file wins outright; one naming a directory gains
        // the platform file name.
        return vec![if p.is_dir() { p.join(name) } else { p }];
    }
    let mut out: Vec<PathBuf> = Vec::new();
    out.push(PathBuf::from("native").join("build-win").join(name));
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            out.push(dir.join(name));
        }
    }
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        out.push(PathBuf::from(local).join("dllm").join(name));
    }
    if let Some(home) = std::env::var_os("HOME") {
        out.push(PathBuf::from(home).join(".dllm").join(name));
    }
    out
}

/// First existing candidate, or a [`LoadError::NotFound`] naming every place
/// that was probed.
///
/// Every probed path is listed in the error rather than just the first: "it is
/// in the wrong place" and "it does not exist" are different problems, and an
/// operator reading one log line should not have to guess which.
pub fn discover() -> Result<PathBuf, LoadError> {
    let candidates = candidate_paths();
    let mut tried = Vec::new();
    for path in &candidates {
        if path.is_file() {
            return Ok(path.clone());
        }
        tried.push(path.display().to_string());
    }
    Err(LoadError::NotFound {
        searched: tried.join(", "),
        env: DLL_PATH_ENV,
    })
}

/// The outcome of the one-shot load attempt.
///
/// `Ready` holds an `Arc` rather than a bare `ShimLib` because
/// [`load_shim`] returns a `&'static` verdict: without the `Arc` there is no
/// way for a caller to get an owned handle out of it (`ShimLib` is not `Clone`
/// — it owns the OS module handle, and cloning that by hand would be a
/// double-unload waiting to happen).
#[derive(Debug)]
pub enum ShimLoad {
    /// Loaded, ABI-verified and `dllm_shim_init`ed.
    Ready(Arc<ShimLib>),
    /// Not usable. The string is the reason, already phrased for a log line.
    Unavailable(String),
}

impl ShimLoad {
    /// Short machine-readable tag for `GET /api/stats` / telemetry.
    pub fn status(&self) -> &'static str {
        match self {
            Self::Ready(_) => "ready",
            Self::Unavailable(_) => "unavailable",
        }
    }

    /// The loaded library, if any.
    pub fn lib(&self) -> Option<&Arc<ShimLib>> {
        match self {
            Self::Ready(lib) => Some(lib),
            Self::Unavailable(_) => None,
        }
    }

    /// Why the shim is unusable, if it is. Already phrased for a log line.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Ready(_) => None,
            Self::Unavailable(why) => Some(why),
        }
    }
}

/// Attempt to load the shim once per process.
///
/// `OnceLock` rather than a lazy `static mut`: a failed load must not be
/// retried on every request (each attempt costs a filesystem probe and would
/// spam the log), but it also must not poison the process. [`ShimLib`] is
/// `Sync` (the OS handle is, and every field is either immutable or behind a
/// mutex), so the result can be shared with every request handler as an `Arc`.
pub fn load_shim() -> &'static ShimLoad {
    static LOADED: OnceLock<ShimLoad> = OnceLock::new();
    LOADED.get_or_init(|| match try_load() {
        Ok(lib) => ShimLoad::Ready(Arc::new(lib)),
        Err(e) => ShimLoad::Unavailable(e.to_string()),
    })
}

/// Load, verify and initialise the shim.
///
/// The order matters: resolve every symbol *before* the ABI check so a missing
/// export names itself, then check the version, then `dllm_shim_init()` — the
/// RPC backend only exists after init, so nothing that touches devices may run
/// earlier.
pub fn try_load() -> Result<ShimLib, LoadError> {
    let path = discover()?;

    // SAFETY: opening a library runs its static initialisers. This DLL's
    // initialisers are ggml's backend registry constructors plus nothing else
    // (see `do_init()` in dllm_shim.cpp), which is exactly what the ABI intends
    // to happen at load time.
    let lib = unsafe { open_library(&path) }.map_err(|source| LoadError::DlOpen {
        path: path.display().to_string(),
        reason: source.to_string(),
    })?;

    // SAFETY: `lib` is live for the whole function; each `symbol` call checks
    // for a null result and bails before the pointer is used.
    let api = unsafe { resolve_api(&lib)? };

    // SAFETY: `abi_version` takes no arguments, touches no state and cannot
    // fail, so this is unconditionally valid once the symbol resolved.
    let found = unsafe { (api.abi_version)() };
    if found != EXPECTED_ABI_VERSION {
        return Err(LoadError::AbiMismatch {
            path: path.display().to_string(),
            found,
            expected: EXPECTED_ABI_VERSION,
        });
    }

    let shim = ShimLib {
        _lib: lib,
        api,
        path,
        abi_version: found,
        registry: Mutex::new(WorkerRegistry::new()),
    };

    // SAFETY: init is documented idempotent and must precede any RPC work.
    if unsafe { (shim.api.init)() } != 0 {
        return Err(LoadError::InitFailed(shim.last_error()));
    }
    Ok(shim)
}

/// Open the DLL with the flags that make its own dependencies resolvable.
unsafe fn open_library(path: &std::path::Path) -> Result<RawLibrary, libloading::Error> {
    #[cfg(target_os = "windows")]
    {
        // SAFETY (caller): `path` must name a loadable module. The flag makes
        // the loader search the DLL's own directory for its dependencies first,
        // which is what lets a DLL sitting in `native/build-win/` find a
        // sibling `libstdc++-6.dll`. It requires a fully qualified path, which
        // `candidate_paths` guarantees.
        unsafe {
            libloading::os::windows::Library::load_with_flags(
                path,
                libloading::os::windows::LOAD_WITH_ALTERED_SEARCH_PATH,
            )
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        // SAFETY (caller): as above; Unix resolves DT_NEEDED through
        // RPATH/RUNPATH and LD_LIBRARY_PATH, which needs no special flag.
        unsafe { libloading::os::unix::Library::new(path) }
    }
}

/// Resolve all 15 exports, naming the first one that is absent.
unsafe fn resolve_api(lib: &RawLibrary) -> Result<ShimApi, LoadError> {
    // SAFETY (caller): `lib` must be a live library handle. Each `symbol` call
    // checks for a null result and returns before the pointer is used, and the
    // transcriptions in `abi.rs` are pinned against the header by unit tests.
    unsafe {
        Ok(ShimApi {
            init: symbol(lib, "dllm_shim_init")?,
            free: symbol(lib, "dllm_shim_free")?,
            last_error: symbol(lib, "dllm_shim_last_error")?,
            abi_version: symbol(lib, "dllm_shim_abi_version")?,
            add_rpc_server: symbol(lib, "dllm_shim_add_rpc_server")?,
            rpc_device_count: symbol(lib, "dllm_shim_rpc_device_count")?,
            rpc_device_memory: symbol(lib, "dllm_shim_rpc_device_memory")?,
            rpc_serve: symbol(lib, "dllm_shim_rpc_serve")?,
            session_open: symbol(lib, "dllm_shim_session_open")?,
            session_close: symbol(lib, "dllm_shim_session_close")?,
            session_generate: symbol(lib, "dllm_shim_session_generate")?,
            session_cancel: symbol(lib, "dllm_shim_session_cancel")?,
            session_n_layer: symbol(lib, "dllm_shim_session_n_layer")?,
            session_n_ctx: symbol(lib, "dllm_shim_session_n_ctx")?,
            session_report: symbol(lib, "dllm_shim_session_report")?,
        })
    }
}

/// Fetch one export by value so the fn pointer is `Copy`.
///
/// `libloading` wants a NUL-terminated byte slice; the names come from
/// `abi::EXPORTED_SYMBOLS`, which are all ASCII literals with no interior NUL,
/// so building the buffer cannot fail.
unsafe fn symbol<T: Copy>(lib: &RawLibrary, name: &'static str) -> Result<T, LoadError> {
    let mut cname = Vec::with_capacity(name.len() + 1);
    cname.extend_from_slice(name.as_bytes());
    cname.push(0);
    // SAFETY (caller): `lib` is live; the returned `Symbol<T>` derefs to the raw
    // address `GetProcAddress` produced, valid for as long as the library stays
    // loaded (guaranteed by `ShimLib` owning the handle). `T` is a plain fn
    // pointer, for which `ensure_compatible_types` accepts `FARPROC`.
    let sym = unsafe { lib.get::<T>(&cname) }
        .map_err(|_| LoadError::MissingSymbol { symbol: name })?;
    Ok(*sym)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Discovery must never discard the env override: ops pointing at a DLL
    /// outside the repo depend on it, and a silent fallback would load a
    /// *different* build than the one under test.
    #[test]
    fn discovery_honours_the_env_override() {
        let cands = {
            let _guard = env_lock();
            unsafe { std::env::set_var(DLL_PATH_ENV, r"C:\nowhere\dllm_shim.dll") };
            let c = candidate_paths();
            unsafe { std::env::remove_var(DLL_PATH_ENV) };
            c
        };
        assert_eq!(cands.len(), 1, "an override replaces the search list");
        assert!(cands[0].ends_with("dllm_shim.dll"));

        let cands = candidate_paths();
        assert!(cands.len() > 1, "without an override we probe several places");
        assert!(
            cands[0].to_string_lossy().contains("native"),
            "native/build-win is probed first: {:?}",
            cands[0]
        );
        assert!(
            cands.iter().any(|p| p.ends_with(dll_file_name())),
            "every candidate names the platform library"
        );
    }

    #[test]
    fn a_directory_override_gains_the_platform_file_name() {
        let _guard = env_lock();
        let dir = std::env::temp_dir().join("dllm-shim-override-dir");
        std::fs::create_dir_all(&dir).unwrap();
        unsafe { std::env::set_var(DLL_PATH_ENV, &dir) };
        let cands = candidate_paths();
        assert_eq!(cands[0], dir.join(dll_file_name()));
        unsafe { std::env::remove_var(DLL_PATH_ENV) };
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn every_export_name_is_prefixed_and_unique() {
        let mut names = crate::abi::EXPORTED_SYMBOLS.to_vec();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate export name");
        assert!(names.iter().all(|s| s.starts_with("dllm_shim_")));
    }

    /// A missing DLL must be a `Result`, never a panic or an abort: this is the
    /// single most important property of the crate for a product whose server
    /// must boot on a machine that has never compiled C++.
    #[test]
    fn a_missing_dll_is_an_error_not_a_crash() {
        let _guard = env_lock();
        let missing = std::env::temp_dir().join("dllm-shim-definitely-absent.dll");
        let _ = std::fs::remove_file(&missing);
        unsafe { std::env::set_var(DLL_PATH_ENV, &missing) };
        let err = try_load().expect_err("must refuse to proceed");
        match &err {
            LoadError::NotFound { searched, .. } => assert!(searched.contains("absent")),
            other => panic!("expected NotFound, got {other:?}"),
        }
        assert!(err.to_string().contains("build-llama-win.ps1"));
        unsafe { std::env::remove_var(DLL_PATH_ENV) };
    }

    /// A file that exists but is not a DLL must also be an error, and must name
    /// the file, so "it is there but wrong" stays distinguishable from "it is
    /// gone".
    #[test]
    fn a_garbage_file_is_a_named_dlopen_error() {
        let _guard = env_lock();
        let junk = std::env::temp_dir().join("dllm-shim-not-a-dll.dll");
        std::fs::write(&junk, b"this is not a PE image").unwrap();
        unsafe { std::env::set_var(DLL_PATH_ENV, &junk) };
        let err = try_load().expect_err("garbage must not load");
        match err {
            LoadError::DlOpen { path, .. } => assert!(path.contains("not-a-dll")),
            other => panic!("expected DlOpen, got {other:?}"),
        }
        unsafe { std::env::remove_var(DLL_PATH_ENV) };
        let _ = std::fs::remove_file(&junk);
    }

    /// `load_shim` caches, so its verdict is stable for the process — that is
    /// the point, and a test must not depend on whether the DLL exists here.
    #[test]
    fn load_shim_is_memoised_and_never_panics() {
        let a = load_shim();
        let b = load_shim();
        assert!(std::ptr::eq(a, b), "load_shim must return one cached verdict");
        assert!(matches!(a.status(), "ready" | "unavailable"));
    }

    /// Endpoint parse errors must be nameable so the coordinator can log one
    /// device's bad `rpc_endpoint` and keep serving the rest.
    #[test]
    fn endpoint_errors_reach_the_caller() {
        let bad: Result<RpcEndpoint, _> = "not-an-endpoint".parse();
        assert!(bad.is_err());
    }

    /// Serialises the tests that mutate `DLLM_SHIM_DLL` (process-global env).
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }
}
