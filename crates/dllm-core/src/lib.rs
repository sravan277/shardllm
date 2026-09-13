//! dllm-core: inference `Engine` trait + pipeline planning.
//!
//! Stable Rust only, no C dependencies.

pub mod engine;
pub mod plan;

pub use engine::{Engine, LlamaEngine, MockEngine, TokenEvent};
pub use plan::{Calibration, PipelinePlan, StageRange, greedy_partition};
