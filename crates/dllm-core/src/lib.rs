//! dllm-core: inference `Engine` trait + pipeline planning.
//!
//! Stable Rust only. The one native dependency is `dllm-shim`, and even that is
//! loaded lazily at run time: `crates/dllm-core/src/shim_engine.rs` compiles and
//! its unit tests pass on a machine where `native/build-win/dllm_shim.dll` does
//! not exist.

pub mod commit;
pub mod engine;
pub mod plan;
pub mod shim_engine;

pub use commit::{CommitOutcome, CommitTracker};
pub use engine::{
    Engine, EngineFlavor, EngineTelemetry, LlamaEngine, MeasuredStage, MockEngine, RequestedStage,
    TokenEvent,
};
pub use plan::{
    ACTIVATION_BYTES_PER_TOKEN, Calibration, DeviceSpec, PipelinePlan, StageRange, WIRE_DIM,
    greedy_partition, plan_layers,
};
pub use shim_engine::{
    PlannedDevice, SessionRequest, ShimEngine, ShimEngineError, SplitError, SplitPlan,
    tensor_split_for_plan,
};
