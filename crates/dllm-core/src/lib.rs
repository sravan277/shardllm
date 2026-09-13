//! dllm-core: inference `Engine` trait + pipeline planning.
//!
//! Stable Rust only, no C dependencies.

pub mod commit;
pub mod engine;
pub mod plan;

pub use commit::{CommitOutcome, CommitTracker};
pub use engine::{Engine, LlamaEngine, MockEngine, TokenEvent};
pub use plan::{
    ACTIVATION_BYTES_PER_TOKEN, Calibration, DeviceSpec, PipelinePlan, StageRange, WIRE_DIM,
    greedy_partition, plan_layers,
};
