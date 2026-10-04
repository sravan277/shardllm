//! `dllm-shim`: the Rust side of the `dllm_shim` C ABI.
//!
//! # What this crate is
//!
//! `llama-cpp-2` — the crate the coordinator's inference engine used to be
//! built on — vendors a llama.cpp tree with **no** `ggml/src/ggml-rpc/` at all,
//! and its bindgen never includes `ggml-rpc.h`, so zero `ggml_rpc*` symbols are
//! reachable from Rust. ggml backend registration is also C++-vtable-bound, so a
//! backend cannot be fabricated from Rust either. The only way to put
//! transformer layers on another machine is to own a C++ translation unit that
//! links llama.cpp + ggml-rpc and exposes a plain C surface — which is
//! `native/src/dllm_shim.cpp`.
//!
//! This crate is that C surface, seen from Rust. It has:
//!
//! - **no `build.rs`** and **no `#[link]`**. The DLL lives in
//!   `native/build-win/`, which is gitignored and may simply not exist, so a
//!   link-time dependency would stop `cargo build` — let alone `dllm serve` —
//!   from working on a machine that has never compiled C++. Every symbol is
//!   resolved at run time and a missing DLL is a `Result`, never an abort
//!   (see [`loader`]).
//! - **no `bindgen`**. The declarations in [`abi`] are hand-written
//!   transcriptions of `native/include/dllm_shim.h`, pinned by layout unit tests
//!   so a mismatch fails in `cargo test` rather than corrupting memory.
//!
//! # The ordering invariant, restated
//!
//! `tensor_split` is indexed by position in the `ggml_backend_dev_t*` array
//! llama.cpp consumes verbatim: **local devices first, then RPC devices in
//! registration order**. So the coordinator must register workers in the same
//! order the plan numbers its stages, and must register each endpoint at most
//! once. [`registry::WorkerRegistry::register_in_order`] enforces the first;
//! [`registry`] enforces the second. See [`split`] for the full statement.
//!
//! # Example (end-to-end shape; the runnable version is
//! `crates/dllm-core/examples/shim_rpc_e2e.rs`)
//!
//! ```no_run
//! use std::sync::Arc;
//! use dllm_shim::{ShimSession, Sampler, TextCollector};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // A missing DLL is an ordinary error, never a crash.
//! let lib = match dllm_shim::try_load() {
//!     Ok(lib) => lib,
//!     Err(why) => {
//!         eprintln!("running single-device: {why}");
//!         return Ok(());
//!     }
//! };
//! let lib: Arc<_> = Arc::new(lib);
//!
//! // [15, 13] = 15 layers on the local device, 13 on the first registered RPC
//! // worker (see `split::tensor_split_from_stages`).
//! let session = ShimSession::open(lib, std::path::Path::new("model.gguf"), &[15.0, 13.0], -1, 2048)?;
//! let mut sink = TextCollector::default();
//! session.generate("The capital of France is", 24, &Sampler::default(), &mut sink)?;
//! println!("{}", sink.text);
//!
//! // Measured, not requested:
//! let report = session.report()?;
//! println!("{:?}", report.layers_per_device());
//! # Ok(()) }
//! ```

pub mod abi;
pub mod endpoint;
pub mod loader;
pub mod registry;
pub mod report;
pub mod session;
pub mod split;

pub use endpoint::{DEFAULT_RPC_PORT, EndpointError, RpcEndpoint, dedup_endpoints};
pub use loader::{EXPECTED_ABI_VERSION, LoadError, ShimLoad, ShimLib, load_shim, try_load};
pub use registry::{DroppedStage, PlannedStage, RegistrationOutcome, ResolvedStage, WorkerRegistry};
pub use report::{ModelReport, ReportError, ShimTiming, format_tensor_split, parse_report, parse_report_str};
pub use session::{
    MAX_TOKENS, PromptStyle, Sampler, SessionError, ShimSession, TextCollector, TokenSink,
};
pub use split::{
    DeviceLayout, SplitError, StageSlot, contiguous_runs, layers_from_tensor_split,
    tensor_split_from_stages,
};
