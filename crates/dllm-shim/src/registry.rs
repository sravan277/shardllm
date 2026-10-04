//! The live bridge between [`WorkerRegistry`] bookkeeping and
//! `dllm_shim_add_rpc_server`.
//!
//! Split out from [`crate::endpoint`] so that module stays pure arithmetic (and
//! therefore testable with no native artifact), while this one owns the one
//! irreversible side effect in the whole crate.
//!
//! # Why registration order is not a detail
//!
//! `llama_model_params.devices` is a NULL-terminated `ggml_backend_dev_t*`
//! that llama.cpp consumes **verbatim, in order** (llama.cpp b7418,
//! `llama-model.cpp:834-837`). The shim fills it with local devices first and
//! then the RPC devices *in registration order*, and `tensor_split` is indexed
//! by position in that same list. Therefore:
//!
//! > If the coordinator registers workers in a different order than the plan's
//! > stages are numbered, then `tensor_split[1]` is applied to the wrong
//! > machine and every `layer_owner` index in the report is off by the number
//! > of intervening devices.
//!
//! Nothing inside the shim can detect that, so it is made structural here:
//! [`WorkerRegistry::register_in_order`] takes the stages **already in plan
//! order**, appends each new endpoint to the global device list in that same
//! order, and returns the resulting per-stage device-slot counts. Callers must
//! not sort or re-order the stage list.

use std::sync::Mutex;

use crate::endpoint::{RpcEndpoint, RpcRegistry};
use crate::loader::ShimLib;

/// One pipeline stage as the planner decided it.
///
/// `stage` is the index into `PipelinePlan::stages`; it is carried so a caller
/// can attribute a dropped stage back to its `device_id` when an endpoint turns
/// out to be unreachable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedStage {
    /// Index of this stage in the plan's `stages` vector.
    pub stage: usize,
    /// Registry `device_id` that owns the stage (for the API's benefit).
    pub device_id: String,
    /// Where to dial it, or `None` for "this process" (a local ggml device).
    pub endpoint: Option<RpcEndpoint>,
}

/// How a stage resolved against the devices that actually exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedStage {
    pub stage: usize,
    pub device_id: String,
    /// ggml devices this stage owns. `1` for local CPU and for a worker
    /// exposing one ggml backend; more when a worker exposes several.
    pub devices: usize,
    /// Present only when `endpoint` was `Some` — a local stage has nothing to
    /// dial.
    pub endpoint: Option<RpcEndpoint>,
}

/// A stage that could not be honoured, and why.
///
/// The coordinator's contract is that a dead worker degrades the stage count
/// and *says so*; it never fails session creation because one endpoint is down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DroppedStage {
    pub stage: usize,
    pub device_id: String,
    pub endpoint: RpcEndpoint,
    pub reason: String,
}

/// Result of one `register_in_order` pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistrationOutcome {
    /// Stages that are live, in plan order, with their device-slot counts.
    pub live: Vec<ResolvedStage>,
    /// Stages dropped because their endpoint was unreachable.
    pub dropped: Vec<DroppedStage>,
    /// `true` when at least one stage was newly added to the global device
    /// list. `false` means every endpoint was already registered, so the caller
    /// can skip rebuilding `tensor_split` and reopening the session.
    pub changed: bool,
}

impl RegistrationOutcome {
    /// ggml devices per live stage, aligned with `live` — the second argument
    /// to [`crate::split::tensor_split_from_stages`].
    pub fn devices_per_stage(&self) -> Vec<usize> {
        self.live.iter().map(|s| s.devices).collect()
    }
}

/// Process-wide record of which endpoints the shim already knows.
///
/// Deliberately **not** generic: it holds a `&'static ShimLib` reference so
/// registration can happen at most once per endpoint for the whole process,
/// which is the only correct behaviour given that the ggml device list is global
/// and the ABI offers no way to remove an endpoint.
#[derive(Debug)]
pub struct WorkerRegistry {
    /// Endpoints already handed to the shim, in registration order.
    seen: Mutex<RpcRegistry>,
}

impl WorkerRegistry {
    pub fn new() -> Self {
        Self {
            seen: Mutex::new(RpcRegistry::new()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, RpcRegistry> {
        self.seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Endpoints already registered, in ggml slot order.
    pub fn registered(&self) -> Vec<RpcEndpoint> {
        self.lock().registered()
    }

    /// Devices an endpoint contributed, or `None` if never registered.
    pub fn devices_for(&self, ep: &RpcEndpoint) -> Option<usize> {
        self.lock().devices_for(ep)
    }

    /// Register every stage's endpoint **in plan order**, exactly once.
    ///
    /// `stages` must already be ordered the way `PipelinePlan::stages` is; the
    /// returned `live` vector is in that same order, and
    /// [`RegistrationOutcome::devices_per_stage`] lines up 1:1 with the
    /// resulting `tensor_split` slots. Getting the input order wrong is the
    /// single failure mode this function exists to make impossible-by-review.
    ///
    /// Every stage is independent: an endpoint that fails to answer is recorded
    /// in [`RegistrationOutcome::dropped`] and the remaining stages still
    /// register. `changed` is `true` only if the *global* device list actually
    /// grew, so a caller can skip a session reopen on a no-op pass.
    pub fn register_in_order(
        &self,
        lib: &ShimLib,
        stages: &[PlannedStage],
    ) -> RegistrationOutcome {
        let mut live = Vec::with_capacity(stages.len());
        let mut dropped = Vec::new();
        let mut changed = false;

        for s in stages {
            let Some(endpoint) = s.endpoint.clone() else {
                // Local stage: this process's own ggml devices. Exactly one slot
                // is assumed, which the shim's `collect_layer_devices` upholds
                // on every build without a second local accelerator; the
                // session report's `devices` count is compared against this
                // assumption afterwards so a mismatch surfaces as data instead
                // of silence.
                live.push(ResolvedStage {
                    stage: s.stage,
                    device_id: s.device_id.clone(),
                    devices: 1,
                    endpoint: None,
                });
                continue;
            };

            if let Some(devices) = self.lock().devices_for(&endpoint) {
                // Already in the global device list. Re-adding would append a
                // duplicate block and shift every later stage, so this is a
                // no-op that still reports the real slot count.
                live.push(ResolvedStage {
                    stage: s.stage,
                    device_id: s.device_id.clone(),
                    devices,
                    endpoint: Some(endpoint),
                });
                continue;
            }

            match register_endpoint(lib, &endpoint) {
                Ok(devices) => {
                    self.lock().record(endpoint.clone(), devices);
                    changed = true;
                    tracing::info!(
                        endpoint = %endpoint,
                        devices,
                        stage = s.stage,
                        device_id = %s.device_id,
                        "registered ggml-rpc worker (device slots appended in plan stage order)"
                    );
                    live.push(ResolvedStage {
                        stage: s.stage,
                        device_id: s.device_id.clone(),
                        devices,
                        endpoint: Some(endpoint),
                    });
                }
                Err(reason) => {
                    // A dead endpoint must not make session creation fail: the
                    // stage is dropped, its layers get redistributed by
                    // llama.cpp across whoever is left, and the reduction is
                    // reported rather than hidden.
                    tracing::warn!(
                        endpoint = %endpoint,
                        stage = s.stage,
                        device_id = %s.device_id,
                        %reason,
                        "ggml-rpc worker unreachable; dropping its stage (plan degrades)"
                    );
                    dropped.push(DroppedStage {
                        stage: s.stage,
                        device_id: s.device_id.clone(),
                        endpoint,
                        reason,
                    });
                }
            }
        }

        RegistrationOutcome {
            live,
            dropped,
            changed,
        }
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// One `dllm_shim_add_rpc_server` call.
///
/// `Ok(0)` is not success: the header says the function returns "the number of
/// RPC devices the endpoint exposed, or negative on failure", and an endpoint
/// that exposed nothing cannot own a stage. Treated as an error so the stage is
/// dropped and reported rather than silently occupying zero slots.
fn register_endpoint(lib: &ShimLib, endpoint: &RpcEndpoint) -> Result<usize, String> {
    let c = std::ffi::CString::new(endpoint.as_endpoint_string())
        .map_err(|e| format!("endpoint is not NUL-safe: {e}"))?;
    // SAFETY: `c` outlives the call; the shim copies the string into its own
    // `g_endpoints` vector before returning. This is the one call in the crate
    // that mutates process-global ggml state, which is exactly why it lives
    // behind the `WorkerRegistry` dedup.
    let n = unsafe { (lib.api().add_rpc_server)(c.as_ptr()) };
    if n < 0 {
        return Err(lib.last_error());
    }
    if n == 0 {
        return Err("endpoint registered zero ggml devices".to_string());
    }
    Ok(n as usize)
}
